"""
RAIA Platform - Unified Authentication & Authorization System
Enterprise-grade JWT-based authentication with RBAC support
"""

import secrets
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import structlog
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import get_database
from app.models.schemas import User, UserRole, Organization
from app.services.user_service import UserService

logger = structlog.get_logger(__name__)
settings = get_settings()

# Password hashing
pwd_context = CryptContext(schemes=settings.PASSWORD_HASH_SCHEMES, deprecated="auto")

# JWT token security
security = HTTPBearer(auto_error=False)


class AuthenticationError(Exception):
    """Authentication-related errors"""
    pass


class AuthorizationError(Exception):
    """Authorization-related errors"""
    pass


def create_password_hash(password: str) -> str:
    """
    Create a secure hash of the password.
    """
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash.
    """
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(
    subject: Union[str, Any], 
    expires_delta: Optional[timedelta] = None,
    additional_claims: Optional[Dict[str, Any]] = None
) -> str:
    """
    Create a JWT access token.
    
    Args:
        subject: The subject (user ID) for the token
        expires_delta: Token expiration time
        additional_claims: Additional claims to include in token
    """
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
        )
    
    to_encode = {
        "exp": expire,
        "iat": datetime.utcnow(),
        "sub": str(subject),
        "type": "access"
    }
    
    if additional_claims:
        to_encode.update(additional_claims)
    
    encoded_jwt = jwt.encode(
        to_encode, 
        settings.JWT_SECRET_KEY, 
        algorithm=settings.JWT_ALGORITHM
    )
    
    return encoded_jwt


def create_refresh_token(subject: Union[str, Any]) -> str:
    """
    Create a JWT refresh token.
    """
    expire = datetime.utcnow() + timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS)
    
    to_encode = {
        "exp": expire,
        "iat": datetime.utcnow(),
        "sub": str(subject),
        "type": "refresh",
        "jti": secrets.token_urlsafe(32)  # JWT ID for token revocation
    }
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM
    )
    
    return encoded_jwt


def decode_token(token: str) -> Dict[str, Any]:
    """
    Decode and validate a JWT token.
    
    Args:
        token: JWT token string
        
    Returns:
        Dict containing token claims
        
    Raises:
        JWTError: If token is invalid or expired
    """
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM]
        )
        return payload
    except JWTError as e:
        logger.warning("JWT decode error", error=str(e))
        raise


async def get_current_user_from_token(
    token: str,
    db: AsyncSession
) -> User:
    """
    Get current user from JWT token.
    
    Args:
        token: JWT token string
        db: Database session
        
    Returns:
        User object
        
    Raises:
        HTTPException: If token is invalid or user not found
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = decode_token(token)
        user_id: str = payload.get("sub")
        
        if user_id is None:
            raise credentials_exception
            
        token_type: str = payload.get("type")
        if token_type != "access":
            raise credentials_exception
            
    except JWTError:
        raise credentials_exception
    
    user_service = UserService()
    user = await user_service.get_user_by_id(db, user_id)
    
    if user is None:
        raise credentials_exception
        
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is disabled"
        )
    
    return user


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncSession = Depends(get_database)
) -> User:
    """
    FastAPI dependency to get current authenticated user.
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return await get_current_user_from_token(credentials.credentials, db)


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """
    FastAPI dependency to get current active user.
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user


class PermissionChecker:
    """
    Permission checker for role-based access control (RBAC).
    """
    
    def __init__(self, required_permissions: List[str]):
        self.required_permissions = required_permissions
    
    async def __call__(
        self,
        current_user: User = Depends(get_current_active_user),
        db: AsyncSession = Depends(get_database)
    ) -> User:
        """
        Check if user has required permissions.
        """
        user_service = UserService()
        user_permissions = await user_service.get_user_permissions(db, current_user.id)
        
        missing_permissions = set(self.required_permissions) - set(user_permissions)
        
        if missing_permissions:
            logger.warning(
                "Permission denied",
                user_id=current_user.id,
                required=self.required_permissions,
                missing=list(missing_permissions)
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Missing: {', '.join(missing_permissions)}"
            )
        
        return current_user


class RoleChecker:
    """
    Role checker for role-based access control.
    """
    
    def __init__(self, required_roles: List[str]):
        self.required_roles = required_roles
    
    async def __call__(
        self,
        current_user: User = Depends(get_current_active_user),
        db: AsyncSession = Depends(get_database)
    ) -> User:
        """
        Check if user has required roles.
        """
        user_service = UserService()
        user_roles = await user_service.get_user_roles(db, current_user.id)
        user_role_names = [role.name for role in user_roles]
        
        if not any(role in user_role_names for role in self.required_roles):
            logger.warning(
                "Role access denied",
                user_id=current_user.id,
                required_roles=self.required_roles,
                user_roles=user_role_names
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient role permissions. Required: {', '.join(self.required_roles)}"
            )
        
        return current_user


class OrganizationChecker:
    """
    Organization access checker for multi-tenancy.
    """
    
    def __init__(self, allow_super_admin: bool = True):
        self.allow_super_admin = allow_super_admin
    
    async def __call__(
        self,
        organization_id: str,
        current_user: User = Depends(get_current_active_user),
        db: AsyncSession = Depends(get_database)
    ) -> User:
        """
        Check if user has access to the specified organization.
        """
        user_service = UserService()
        
        # Super admin can access all organizations
        if self.allow_super_admin and await user_service.is_super_admin(db, current_user.id):
            return current_user
        
        # Check if user belongs to the organization
        if current_user.organization_id != organization_id:
            logger.warning(
                "Organization access denied",
                user_id=current_user.id,
                user_org=current_user.organization_id,
                requested_org=organization_id
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this organization"
            )
        
        return current_user


# Common permission dependencies
require_admin = PermissionChecker(["admin"])
require_user_management = PermissionChecker(["user:read", "user:write"])
require_agent_evaluation = PermissionChecker(["agent_evaluation:read"])
require_model_explainability = PermissionChecker(["model_explainability:read"])
require_data_management = PermissionChecker(["data:read"])
require_monitoring = PermissionChecker(["monitoring:read"])

# Common role dependencies
require_admin_role = RoleChecker(["admin", "super_admin"])
require_analyst_role = RoleChecker(["analyst", "admin", "super_admin"])
require_viewer_role = RoleChecker(["viewer", "analyst", "admin", "super_admin"])


def generate_api_key() -> str:
    """
    Generate a secure API key for service-to-service authentication.
    """
    return f"raia_{secrets.token_urlsafe(32)}"


async def authenticate_api_key(
    api_key: str,
    db: AsyncSession
) -> Optional[User]:
    """
    Authenticate using API key for service accounts.
    """
    user_service = UserService()
    user = await user_service.get_user_by_api_key(db, api_key)
    
    if user and user.is_active:
        return user
    
    return None


class APIKeyChecker:
    """
    API key authentication checker for service accounts.
    """
    
    async def __call__(
        self,
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
        db: AsyncSession = Depends(get_database)
    ) -> User:
        """
        Authenticate using API key.
        """
        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing API key",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        user = await authenticate_api_key(credentials.credentials, db)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )
        
        return user


# API key dependency
require_api_key = APIKeyChecker()


async def create_user_tokens(user: User) -> Dict[str, str]:
    """
    Create access and refresh tokens for user.
    """
    additional_claims = {
        "org_id": user.organization_id,
        "roles": [role.name for role in user.roles] if user.roles else [],
        "permissions": []  # Will be populated by user service
    }
    
    access_token = create_access_token(
        subject=user.id,
        additional_claims=additional_claims
    )
    
    refresh_token = create_refresh_token(subject=user.id)
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60
    }


async def get_current_active_user_ws(token: str) -> User:
    """
    Get current active user for WebSocket connections.
    
    Args:
        token: JWT token string
        
    Returns:
        User object
        
    Raises:
        HTTPException: If token is invalid or user not found
    """
    from app.core.database import get_database
    from app.services.user_service import UserService
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials"
    )
    
    try:
        payload = decode_token(token)
        user_id: str = payload.get("sub")
        
        if user_id is None:
            raise credentials_exception
            
        token_type: str = payload.get("type")
        if token_type != "access":
            raise credentials_exception
            
    except JWTError:
        raise credentials_exception
    
    # Get database session
    db = await get_database().__anext__()
    try:
        user_service = UserService()
        user = await user_service.get_user_by_id(db, user_id)
        
        if user is None:
            raise credentials_exception
            
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account is disabled"
            )
        
        return user
    finally:
        await db.close()