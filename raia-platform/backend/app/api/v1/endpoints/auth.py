"""
RAIA Platform - Authentication Endpoints
JWT-based authentication with login, logout, and token refresh
"""

import structlog
from datetime import timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import (
    create_user_tokens,
    decode_token,
    get_current_active_user,
    verify_password
)
from app.core.config import get_settings
from app.core.database import get_database
from app.models.schemas import (
    LoginRequest,
    LoginResponse, 
    TokenRefreshRequest,
    PasswordChangeRequest,
    User,
    UserResponse
)
from app.services.user_service import UserService

logger = structlog.get_logger(__name__)
settings = get_settings()
security = HTTPBearer()

router = APIRouter()


@router.post("/login", response_model=LoginResponse)
async def login(
    login_data: LoginRequest,
    db: AsyncSession = Depends(get_database)
) -> LoginResponse:
    """
    Authenticate user and return JWT tokens.
    
    - **username**: Username or email address
    - **password**: User password
    
    Returns access token, refresh token, and user information.
    """
    user_service = UserService()
    
    # Get user by username or email
    user = await user_service.get_user_by_username(db, login_data.username)
    if not user:
        user = await user_service.get_user_by_email(db, login_data.username)
    
    if not user:
        logger.warning("Login attempt with invalid username", username=login_data.username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    # Verify password
    if not verify_password(login_data.password, user.hashed_password):
        logger.warning("Login attempt with invalid password", user_id=user.id)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    # Check if user is active
    if not user.is_active:
        logger.warning("Login attempt by inactive user", user_id=user.id)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Account is disabled"
        )
    
    # Update last login timestamp
    await user_service.update_last_login(db, user.id)
    
    # Create tokens
    tokens = await create_user_tokens(user)
    
    # Get full user data for response
    user_data = await user_service.get_user_with_details(db, user.id)
    
    logger.info("User logged in successfully", user_id=user.id, username=user.username)
    
    return LoginResponse(
        access_token=tokens["access_token"],
        refresh_token=tokens["refresh_token"],
        token_type=tokens["token_type"],
        expires_in=tokens["expires_in"],
        user=UserResponse.from_orm(user_data)
    )


@router.post("/refresh", response_model=dict)
async def refresh_token(
    refresh_data: TokenRefreshRequest,
    db: AsyncSession = Depends(get_database)
) -> dict:
    """
    Refresh access token using refresh token.
    
    - **refresh_token**: Valid refresh token
    
    Returns new access token and refresh token.
    """
    try:
        # Decode refresh token
        payload = decode_token(refresh_data.refresh_token)
        user_id = payload.get("sub")
        token_type = payload.get("type")
        
        if token_type != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type"
            )
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload"
            )
        
    except Exception as e:
        logger.warning("Invalid refresh token", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    
    # Get user
    user_service = UserService()
    user = await user_service.get_user_by_id(db, user_id)
    
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive"
        )
    
    # Create new tokens
    tokens = await create_user_tokens(user)
    
    logger.info("Token refreshed successfully", user_id=user.id)
    
    return {
        "access_token": tokens["access_token"],
        "refresh_token": tokens["refresh_token"],
        "token_type": tokens["token_type"],
        "expires_in": tokens["expires_in"]
    }


@router.post("/logout")
async def logout(
    current_user: User = Depends(get_current_active_user)
) -> dict:
    """
    Logout current user.
    
    In a stateless JWT implementation, logout is handled client-side
    by discarding the tokens. This endpoint is provided for logging
    and potential future token blacklisting features.
    """
    logger.info("User logged out", user_id=current_user.id, username=current_user.username)
    
    return {
        "message": "Successfully logged out",
        "detail": "Please discard your access and refresh tokens"
    }


@router.post("/change-password")
async def change_password(
    password_data: PasswordChangeRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_database)
) -> dict:
    """
    Change user password.
    
    - **current_password**: Current user password
    - **new_password**: New password (minimum 8 characters)
    
    Requires authentication.
    """
    user_service = UserService()
    
    # Verify current password
    if not verify_password(password_data.current_password, current_user.hashed_password):
        logger.warning("Password change attempt with invalid current password", user_id=current_user.id)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect"
        )
    
    # Validate new password
    if len(password_data.new_password) < settings.PASSWORD_MIN_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Password must be at least {settings.PASSWORD_MIN_LENGTH} characters long"
        )
    
    # Update password
    await user_service.update_password(db, current_user.id, password_data.new_password)
    
    logger.info("Password changed successfully", user_id=current_user.id)
    
    return {
        "message": "Password changed successfully",
        "detail": "Please use your new password for future logins"
    }


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_database)
) -> UserResponse:
    """
    Get current user information.
    
    Returns detailed information about the authenticated user,
    including roles and organization details.
    """
    user_service = UserService()
    user_with_details = await user_service.get_user_with_details(db, current_user.id)
    
    return UserResponse.from_orm(user_with_details)


@router.put("/me", response_model=UserResponse)
async def update_current_user(
    user_update: dict,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_database)
) -> UserResponse:
    """
    Update current user information.
    
    Users can update their own profile information such as:
    - full_name
    - preferences
    
    Email and username changes require admin approval.
    """
    user_service = UserService()
    
    # Filter allowed fields for self-update
    allowed_fields = {"full_name", "preferences"}
    filtered_update = {k: v for k, v in user_update.items() if k in allowed_fields}
    
    if not filtered_update:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid fields provided for update"
        )
    
    # Update user
    updated_user = await user_service.update_user(db, current_user.id, filtered_update)
    
    logger.info("User updated profile", user_id=current_user.id, fields=list(filtered_update.keys()))
    
    return UserResponse.from_orm(updated_user)


@router.post("/verify-token")
async def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_database)
) -> dict:
    """
    Verify if a token is valid.
    
    Returns token information and user details if valid.
    """
    try:
        # Decode token
        payload = decode_token(credentials.credentials)
        user_id = payload.get("sub")
        token_type = payload.get("type", "access")
        exp = payload.get("exp")
        
        # Get user
        user_service = UserService()
        user = await user_service.get_user_by_id(db, user_id)
        
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive"
            )
        
        return {
            "valid": True,
            "token_type": token_type,
            "user_id": user_id,
            "username": user.username,
            "expires_at": exp,
            "organization_id": user.organization_id
        }
        
    except Exception as e:
        logger.debug("Token verification failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )


@router.get("/permissions")
async def get_user_permissions(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_database)
) -> dict:
    """
    Get current user's permissions and roles.
    
    Returns detailed information about user permissions for frontend
    authorization and UI customization.
    """
    user_service = UserService()
    
    # Get user roles
    user_roles = await user_service.get_user_roles(db, current_user.id)
    
    # Get user permissions
    user_permissions = await user_service.get_user_permissions(db, current_user.id)
    
    return {
        "user_id": current_user.id,
        "username": current_user.username,
        "organization_id": current_user.organization_id,
        "roles": [
            {
                "id": role.id,
                "name": role.name,
                "description": role.description
            }
            for role in user_roles
        ],
        "permissions": user_permissions,
        "is_super_admin": await user_service.is_super_admin(db, current_user.id),
        "feature_flags": {
            "agent_evaluation": settings.ENABLE_AGENT_EVALUATION,
            "model_explainability": settings.ENABLE_MODEL_EXPLAINABILITY,
            "data_drift_monitoring": settings.ENABLE_DATA_DRIFT_MONITORING,
            "fairness_analysis": settings.ENABLE_FAIRNESS_ANALYSIS,
            "real_time_monitoring": settings.ENABLE_REAL_TIME_MONITORING,
            "custom_dashboards": settings.ENABLE_CUSTOM_DASHBOARDS
        }
    }