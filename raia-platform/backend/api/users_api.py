# User Management API
import os
import uuid
import json
import hashlib
import secrets
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from enum import Enum

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Form, UploadFile, File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import FileResponse
from pydantic import BaseModel, EmailStr, validator
from sqlalchemy.orm import Session
from sqlalchemy import Column, String, DateTime, Text, Boolean, JSON, UUID as PG_UUID, Integer, Float, ForeignKey, func, desc, and_
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

# Authentication and security
from passlib.context import CryptContext
from jose import JWTError, jwt
import pyotp
import qrcode
from io import BytesIO
import base64

# Email and notifications
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

Base = declarative_base()
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/users", tags=["users"])

# Security configuration
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/v1/users/token")
security = HTTPBearer()

# JWT configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Enums
class UserRole(str, Enum):
    ADMIN = "admin"
    MANAGER = "manager"
    DATA_SCIENTIST = "data_scientist"
    BUSINESS_USER = "business_user"
    VIEWER = "viewer"

class UserStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"
    SUSPENDED = "suspended"
    DELETED = "deleted"

class NotificationType(str, Enum):
    EMAIL = "email"
    IN_APP = "in_app"
    SMS = "sms"
    WEBHOOK = "webhook"

# Database Models
class User(Base):
    """User account information"""
    __tablename__ = "users"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Basic information
    username = Column(String(255), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    first_name = Column(String(255))
    last_name = Column(String(255))
    display_name = Column(String(255))
    
    # Authentication
    hashed_password = Column(String(255), nullable=False)
    is_email_verified = Column(Boolean, default=False)
    email_verification_token = Column(String(255))
    password_reset_token = Column(String(255))
    password_reset_expires = Column(DateTime)
    
    # 2FA
    two_factor_enabled = Column(Boolean, default=False)
    two_factor_secret = Column(String(255))
    backup_codes = Column(JSON)  # List of backup codes
    
    # Profile information
    avatar_url = Column(String(1000))
    bio = Column(Text)
    title = Column(String(255))
    department = Column(String(255))
    location = Column(String(255))
    timezone = Column(String(100), default="UTC")
    language = Column(String(10), default="en")
    
    # Account status
    role = Column(String(50), default=UserRole.BUSINESS_USER)
    status = Column(String(50), default=UserStatus.PENDING)
    is_superuser = Column(Boolean, default=False)
    
    # Organization and team
    organization_id = Column(String(255))
    team_ids = Column(JSON)  # List of team IDs
    manager_id = Column(String(255))
    
    # Preferences and settings
    preferences = Column(JSON)  # User preferences
    notification_settings = Column(JSON)  # Notification preferences
    ui_settings = Column(JSON)  # UI customization
    
    # Usage and activity
    last_login = Column(DateTime)
    last_active = Column(DateTime)
    login_count = Column(Integer, default=0)
    failed_login_attempts = Column(Integer, default=0)
    last_failed_login = Column(DateTime)
    is_locked = Column(Boolean, default=False)
    locked_until = Column(DateTime)
    
    # API access
    api_key = Column(String(255), unique=True)
    api_key_expires = Column(DateTime)
    rate_limit_tier = Column(String(50), default="standard")
    
    # Audit fields
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String(255))
    last_password_change = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")
    activity_logs = relationship("UserActivityLog", back_populates="user", cascade="all, delete-orphan")
    notifications = relationship("UserNotification", back_populates="user", cascade="all, delete-orphan")

class UserSession(Base):
    """User session tracking"""
    __tablename__ = "user_sessions"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    
    # Session information
    session_token = Column(String(255), unique=True, nullable=False)
    refresh_token = Column(String(255), unique=True)
    device_info = Column(JSON)  # Browser, OS, device type
    ip_address = Column(String(45))
    user_agent = Column(String(1000))
    location = Column(JSON)  # GeoIP location data
    
    # Session lifecycle
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    last_used = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    user = relationship("User", back_populates="sessions")

class UserActivityLog(Base):
    """User activity tracking"""
    __tablename__ = "user_activity_logs"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    
    # Activity details
    action = Column(String(255), nullable=False)
    resource_type = Column(String(100))  # model, dataset, experiment, etc.
    resource_id = Column(String(255))
    details = Column(JSON)  # Additional activity details
    
    # Context
    ip_address = Column(String(45))
    user_agent = Column(String(1000))
    session_id = Column(String(255))
    
    # Metadata
    timestamp = Column(DateTime, default=datetime.utcnow)
    success = Column(Boolean, default=True)
    error_message = Column(Text)
    
    # Relationships
    user = relationship("User", back_populates="activity_logs")

class UserNotification(Base):
    """User notifications"""
    __tablename__ = "user_notifications"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    
    # Notification content
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    notification_type = Column(String(50), default=NotificationType.IN_APP)
    category = Column(String(100))  # system, experiment, model, etc.
    
    # Notification data
    data = Column(JSON)  # Additional notification data
    action_url = Column(String(1000))  # URL for notification action
    
    # Status
    is_read = Column(Boolean, default=False)
    is_sent = Column(Boolean, default=False)
    sent_at = Column(DateTime)
    read_at = Column(DateTime)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)
    
    # Relationships
    user = relationship("User", back_populates="notifications")

class Organization(Base):
    """Organization/Company information"""
    __tablename__ = "organizations"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Basic information
    name = Column(String(255), nullable=False)
    display_name = Column(String(255))
    description = Column(Text)
    
    # Contact information
    email = Column(String(255))
    phone = Column(String(50))
    website = Column(String(255))
    
    # Address
    address_line1 = Column(String(255))
    address_line2 = Column(String(255))
    city = Column(String(100))
    state = Column(String(100))
    country = Column(String(100))
    postal_code = Column(String(20))
    
    # Organization settings
    settings = Column(JSON)  # Organization-wide settings
    branding = Column(JSON)  # Logo, colors, themes
    
    # Billing and subscription
    subscription_plan = Column(String(100))
    subscription_status = Column(String(50))
    billing_email = Column(String(255))
    
    # Status
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Team(Base):
    """Team/Group information"""
    __tablename__ = "teams"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id = Column(PG_UUID(as_uuid=True), ForeignKey('organizations.id'))
    
    # Basic information
    name = Column(String(255), nullable=False)
    description = Column(Text)
    
    # Team settings
    settings = Column(JSON)
    permissions = Column(JSON)  # Team-level permissions
    
    # Leadership
    lead_user_id = Column(String(255))
    
    # Status
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String(255))

# Pydantic Models
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    role: UserRole = UserRole.BUSINESS_USER
    organization_id: Optional[str] = None
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v

class UserUpdate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    display_name: Optional[str] = None
    bio: Optional[str] = None
    title: Optional[str] = None
    department: Optional[str] = None
    location: Optional[str] = None
    timezone: Optional[str] = None
    language: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None
    notification_settings: Optional[Dict[str, Any]] = None

class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    first_name: Optional[str]
    last_name: Optional[str]
    display_name: Optional[str]
    role: str
    status: str
    avatar_url: Optional[str]
    is_email_verified: bool
    two_factor_enabled: bool
    last_login: Optional[datetime]
    created_at: datetime
    
    class Config:
        orm_mode = True

class UserProfileResponse(BaseModel):
    id: str
    username: str
    email: str
    first_name: Optional[str]
    last_name: Optional[str]
    display_name: Optional[str]
    bio: Optional[str]
    title: Optional[str]
    department: Optional[str]
    location: Optional[str]
    avatar_url: Optional[str]
    role: str
    status: str
    timezone: str
    language: str
    preferences: Optional[Dict[str, Any]]
    notification_settings: Optional[Dict[str, Any]]
    last_login: Optional[datetime]
    created_at: datetime
    
    class Config:
        orm_mode = True

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse

class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str
    
    @validator('new_password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        return v

class PasswordResetRequest(BaseModel):
    email: EmailStr

class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str

class NotificationCreate(BaseModel):
    title: str
    message: str
    notification_type: NotificationType = NotificationType.IN_APP
    category: Optional[str] = None
    data: Optional[Dict[str, Any]] = {}
    action_url: Optional[str] = None

class NotificationResponse(BaseModel):
    id: str
    title: str
    message: str
    notification_type: str
    category: Optional[str]
    is_read: bool
    created_at: datetime
    action_url: Optional[str]
    
    class Config:
        orm_mode = True

# Dependency injection
def get_db():
    pass

# User Management Service
class UserManagementService:
    """Service for user management and authentication"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Hash password"""
        return pwd_context.hash(password)
    
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None):
        """Create JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    def create_refresh_token(self, data: dict, expires_delta: Optional[timedelta] = None):
        """Create JWT refresh token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[str]:
        """Verify JWT token and return user ID"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            user_id: str = payload.get("sub")
            if user_id is None:
                return None
            return user_id
        except JWTError:
            return None
    
    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return self.db.query(User).filter(User.id == user_id, User.status != UserStatus.DELETED).first()
    
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        return self.db.query(User).filter(User.username == username, User.status != UserStatus.DELETED).first()
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        return self.db.query(User).filter(User.email == email, User.status != UserStatus.DELETED).first()
    
    async def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username/email and password"""
        user = await self.get_user_by_username(username)
        if not user:
            user = await self.get_user_by_email(username)
        
        if not user or not self.verify_password(password, user.hashed_password):
            return None
        
        # Check if account is locked
        if user.is_locked and user.locked_until and user.locked_until > datetime.utcnow():
            return None
        
        # Reset failed login attempts on successful login
        user.failed_login_attempts = 0
        user.last_failed_login = None
        user.is_locked = False
        user.locked_until = None
        user.last_login = datetime.utcnow()
        user.last_active = datetime.utcnow()
        user.login_count += 1
        
        self.db.commit()
        return user
    
    async def create_user(self, user_data: UserCreate, created_by: Optional[str] = None) -> User:
        """Create a new user"""
        
        # Check if username or email already exists
        existing_user = await self.get_user_by_username(user_data.username)
        if existing_user:
            raise HTTPException(status_code=400, detail="Username already exists")
        
        existing_user = await self.get_user_by_email(user_data.email)
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already exists")
        
        # Create user
        hashed_password = self.get_password_hash(user_data.password)
        
        user = User(
            username=user_data.username,
            email=user_data.email,
            hashed_password=hashed_password,
            first_name=user_data.first_name,
            last_name=user_data.last_name,
            role=user_data.role,
            organization_id=user_data.organization_id,
            created_by=created_by,
            email_verification_token=secrets.token_urlsafe(32),
            api_key=self._generate_api_key(),
            api_key_expires=datetime.utcnow() + timedelta(days=365)
        )
        
        # Set display name
        if user_data.first_name and user_data.last_name:
            user.display_name = f"{user_data.first_name} {user_data.last_name}"
        else:
            user.display_name = user_data.username
        
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        
        # Send verification email
        await self._send_verification_email(user)
        
        logger.info(f"Created user {user.username} (ID: {user.id})")
        return user
    
    async def login(self, username: str, password: str, device_info: Optional[Dict] = None, ip_address: Optional[str] = None) -> Dict[str, Any]:
        """Login user and create session"""
        
        user = await self.authenticate_user(username, password)
        if not user:
            # Record failed login attempt
            failed_user = await self.get_user_by_username(username) or await self.get_user_by_email(username)
            if failed_user:
                await self._record_failed_login(failed_user, ip_address)
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        if user.status not in [UserStatus.ACTIVE, UserStatus.PENDING]:
            raise HTTPException(status_code=403, detail="Account is not active")
        
        # Create tokens
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        refresh_token_expires = timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        
        access_token = self.create_access_token(
            data={"sub": str(user.id), "username": user.username},
            expires_delta=access_token_expires
        )
        refresh_token = self.create_refresh_token(
            data={"sub": str(user.id), "type": "refresh"}
        )
        
        # Create session
        session = UserSession(
            user_id=user.id,
            session_token=access_token,
            refresh_token=refresh_token,
            device_info=device_info or {},
            ip_address=ip_address,
            expires_at=datetime.utcnow() + access_token_expires
        )
        
        self.db.add(session)
        
        # Log activity
        await self._log_user_activity(user.id, "login", details={"ip_address": ip_address})
        
        self.db.commit()
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            "user": user
        }
    
    async def refresh_token(self, refresh_token: str) -> Dict[str, Any]:
        """Refresh access token"""
        
        user_id = self.verify_token(refresh_token)
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid refresh token")
        
        user = await self.get_user_by_id(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Verify refresh token exists in session
        session = self.db.query(UserSession).filter(
            UserSession.user_id == user.id,
            UserSession.refresh_token == refresh_token,
            UserSession.is_active == True
        ).first()
        
        if not session:
            raise HTTPException(status_code=401, detail="Invalid refresh token")
        
        # Create new tokens
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        new_access_token = self.create_access_token(
            data={"sub": str(user.id), "username": user.username},
            expires_delta=access_token_expires
        )
        
        # Update session
        session.session_token = new_access_token
        session.expires_at = datetime.utcnow() + access_token_expires
        session.last_used = datetime.utcnow()
        
        user.last_active = datetime.utcnow()
        self.db.commit()
        
        return {
            "access_token": new_access_token,
            "token_type": "bearer",
            "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60
        }
    
    async def logout(self, user_id: str, session_token: str) -> Dict[str, Any]:
        """Logout user and invalidate session"""
        
        session = self.db.query(UserSession).filter(
            UserSession.user_id == user_id,
            UserSession.session_token == session_token
        ).first()
        
        if session:
            session.is_active = False
            self.db.commit()
        
        await self._log_user_activity(user_id, "logout")
        
        return {"message": "Logged out successfully"}
    
    async def setup_2fa(self, user_id: str) -> Dict[str, Any]:
        """Setup two-factor authentication"""
        
        user = await self.get_user_by_id(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        if user.two_factor_enabled:
            raise HTTPException(status_code=400, detail="2FA already enabled")
        
        # Generate secret
        secret = pyotp.random_base32()
        
        # Generate QR code
        totp = pyotp.TOTP(secret)
        provisioning_uri = totp.provisioning_uri(
            name=user.email,
            issuer_name="RAIA Platform"
        )
        
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(provisioning_uri)
        qr.make(fit=True)
        
        qr_img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to base64
        buffer = BytesIO()
        qr_img.save(buffer, format='PNG')
        qr_code_data = base64.b64encode(buffer.getvalue()).decode()
        
        # Generate backup codes
        backup_codes = [secrets.token_hex(4) for _ in range(10)]
        
        # Store secret (not enabled yet)
        user.two_factor_secret = secret
        user.backup_codes = backup_codes
        
        self.db.commit()
        
        return {
            "secret": secret,
            "qr_code": f"data:image/png;base64,{qr_code_data}",
            "backup_codes": backup_codes
        }
    
    async def enable_2fa(self, user_id: str, token: str) -> Dict[str, Any]:
        """Enable 2FA after verification"""
        
        user = await self.get_user_by_id(user_id)
        if not user or not user.two_factor_secret:
            raise HTTPException(status_code=400, detail="2FA setup not found")
        
        # Verify token
        totp = pyotp.TOTP(user.two_factor_secret)
        if not totp.verify(token):
            raise HTTPException(status_code=400, detail="Invalid 2FA token")
        
        user.two_factor_enabled = True
        self.db.commit()
        
        await self._log_user_activity(user_id, "enable_2fa")
        
        return {"message": "2FA enabled successfully"}
    
    async def change_password(self, user_id: str, current_password: str, new_password: str) -> Dict[str, Any]:
        """Change user password"""
        
        user = await self.get_user_by_id(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        if not self.verify_password(current_password, user.hashed_password):
            raise HTTPException(status_code=400, detail="Current password is incorrect")
        
        user.hashed_password = self.get_password_hash(new_password)
        user.last_password_change = datetime.utcnow()
        
        # Invalidate all sessions except current one
        self.db.query(UserSession).filter(
            UserSession.user_id == user.id,
            UserSession.is_active == True
        ).update({'is_active': False})
        
        self.db.commit()
        
        await self._log_user_activity(user_id, "change_password")
        
        return {"message": "Password changed successfully"}
    
    async def request_password_reset(self, email: str) -> Dict[str, Any]:
        """Request password reset"""
        
        user = await self.get_user_by_email(email)
        if not user:
            # Don't reveal if email exists
            return {"message": "If the email exists, a reset link has been sent"}
        
        # Generate reset token
        reset_token = secrets.token_urlsafe(32)
        user.password_reset_token = reset_token
        user.password_reset_expires = datetime.utcnow() + timedelta(hours=1)
        
        self.db.commit()
        
        # Send reset email
        await self._send_password_reset_email(user, reset_token)
        
        return {"message": "If the email exists, a reset link has been sent"}
    
    async def create_notification(self, user_id: str, notification_data: NotificationCreate) -> UserNotification:
        """Create user notification"""
        
        notification = UserNotification(
            user_id=user_id,
            title=notification_data.title,
            message=notification_data.message,
            notification_type=notification_data.notification_type,
            category=notification_data.category,
            data=notification_data.data,
            action_url=notification_data.action_url
        )
        
        self.db.add(notification)
        self.db.commit()
        self.db.refresh(notification)
        
        # Send notification based on type
        if notification_data.notification_type == NotificationType.EMAIL:
            await self._send_email_notification(user_id, notification)
        
        return notification
    
    async def get_user_notifications(self, user_id: str, unread_only: bool = False, limit: int = 50) -> List[UserNotification]:
        """Get user notifications"""
        
        query = self.db.query(UserNotification).filter(UserNotification.user_id == user_id)
        
        if unread_only:
            query = query.filter(UserNotification.is_read == False)
        
        return query.order_by(desc(UserNotification.created_at)).limit(limit).all()
    
    async def mark_notification_read(self, user_id: str, notification_id: str) -> Dict[str, Any]:
        """Mark notification as read"""
        
        notification = self.db.query(UserNotification).filter(
            UserNotification.id == notification_id,
            UserNotification.user_id == user_id
        ).first()
        
        if not notification:
            raise HTTPException(status_code=404, detail="Notification not found")
        
        notification.is_read = True
        notification.read_at = datetime.utcnow()
        self.db.commit()
        
        return {"message": "Notification marked as read"}
    
    # Private helper methods
    def _generate_api_key(self) -> str:
        """Generate API key"""
        return f"raia_{secrets.token_urlsafe(32)}"
    
    async def _record_failed_login(self, user: User, ip_address: Optional[str]):
        """Record failed login attempt"""
        
        user.failed_login_attempts += 1
        user.last_failed_login = datetime.utcnow()
        
        # Lock account after 5 failed attempts
        if user.failed_login_attempts >= 5:
            user.is_locked = True
            user.locked_until = datetime.utcnow() + timedelta(minutes=30)
        
        self.db.commit()
        
        await self._log_user_activity(str(user.id), "failed_login", success=False, details={"ip_address": ip_address})
    
    async def _log_user_activity(self, user_id: str, action: str, resource_type: Optional[str] = None, 
                                resource_id: Optional[str] = None, success: bool = True, 
                                details: Optional[Dict] = None, ip_address: Optional[str] = None):
        """Log user activity"""
        
        activity_log = UserActivityLog(
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            success=success,
            details=details or {},
            ip_address=ip_address
        )
        
        self.db.add(activity_log)
        self.db.commit()
    
    async def _send_verification_email(self, user: User):
        """Send email verification"""
        # Implementation would send actual email
        logger.info(f"Sending verification email to {user.email}")
    
    async def _send_password_reset_email(self, user: User, reset_token: str):
        """Send password reset email"""
        # Implementation would send actual email
        logger.info(f"Sending password reset email to {user.email}")
    
    async def _send_email_notification(self, user_id: str, notification: UserNotification):
        """Send email notification"""
        # Implementation would send actual email
        logger.info(f"Sending email notification to user {user_id}")

# Current user dependency
async def get_current_user(
    token: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """Get current authenticated user"""
    
    service = UserManagementService(db)
    user_id = service.verify_token(token.credentials)
    
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    user = await service.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Update last active
    user.last_active = datetime.utcnow()
    db.commit()
    
    return user

# API Endpoints
@router.post("/register", response_model=UserResponse)
async def register_user(
    user_data: UserCreate,
    db: Session = Depends(get_db)
):
    """Register a new user"""
    
    service = UserManagementService(db)
    user = await service.create_user(user_data)
    return user

@router.post("/token", response_model=TokenResponse)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """Login and get access token"""
    
    service = UserManagementService(db)
    result = await service.login(form_data.username, form_data.password)
    return result

@router.post("/refresh")
async def refresh_token(
    refresh_token: str,
    db: Session = Depends(get_db)
):
    """Refresh access token"""
    
    service = UserManagementService(db)
    result = await service.refresh_token(refresh_token)
    return result

@router.post("/logout")
async def logout(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Logout current user"""
    
    service = UserManagementService(db)
    # Would need session token from request
    result = await service.logout(str(current_user.id), "session_token")
    return result

@router.get("/me", response_model=UserProfileResponse)
async def get_current_user_profile(current_user: User = Depends(get_current_user)):
    """Get current user profile"""
    return current_user

@router.put("/me", response_model=UserProfileResponse)
async def update_current_user_profile(
    user_data: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update current user profile"""
    
    # Update fields
    for field, value in user_data.dict(exclude_unset=True).items():
        setattr(current_user, field, value)
    
    current_user.updated_at = datetime.utcnow()
    db.commit()
    
    return current_user

@router.post("/me/change-password")
async def change_password(
    password_data: PasswordChangeRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Change current user password"""
    
    service = UserManagementService(db)
    result = await service.change_password(
        str(current_user.id),
        password_data.current_password,
        password_data.new_password
    )
    return result

@router.post("/me/setup-2fa")
async def setup_2fa(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Setup two-factor authentication"""
    
    service = UserManagementService(db)
    result = await service.setup_2fa(str(current_user.id))
    return result

@router.post("/me/enable-2fa")
async def enable_2fa(
    token: str = Form(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Enable two-factor authentication"""
    
    service = UserManagementService(db)
    result = await service.enable_2fa(str(current_user.id), token)
    return result

@router.post("/me/avatar")
async def upload_avatar(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Upload user avatar"""
    
    # Save avatar file
    avatar_dir = f"/tmp/raia_avatars/{current_user.id}"
    os.makedirs(avatar_dir, exist_ok=True)
    
    file_path = f"{avatar_dir}/avatar_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Update user record
    current_user.avatar_url = file_path
    current_user.updated_at = datetime.utcnow()
    db.commit()
    
    return {"avatar_url": file_path}

@router.get("/me/notifications", response_model=List[NotificationResponse])
async def get_notifications(
    unread_only: bool = False,
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user notifications"""
    
    service = UserManagementService(db)
    notifications = await service.get_user_notifications(str(current_user.id), unread_only, limit)
    return notifications

@router.put("/me/notifications/{notification_id}/read")
async def mark_notification_read(
    notification_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Mark notification as read"""
    
    service = UserManagementService(db)
    result = await service.mark_notification_read(str(current_user.id), notification_id)
    return result

@router.post("/password-reset")
async def request_password_reset(
    request: PasswordResetRequest,
    db: Session = Depends(get_db)
):
    """Request password reset"""
    
    service = UserManagementService(db)
    result = await service.request_password_reset(request.email)
    return result

@router.get("/", response_model=List[UserResponse])
async def get_users(
    skip: int = 0,
    limit: int = 100,
    role: Optional[str] = None,
    status: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get users (admin/manager only)"""
    
    if current_user.role not in [UserRole.ADMIN, UserRole.MANAGER]:
        raise HTTPException(status_code=403, detail="Permission denied")
    
    query = db.query(User).filter(User.status != UserStatus.DELETED)
    
    if role:
        query = query.filter(User.role == role)
    if status:
        query = query.filter(User.status == status)
    
    users = query.offset(skip).limit(limit).all()
    return users

@router.get("/{user_id}", response_model=UserProfileResponse)
async def get_user(
    user_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user by ID"""
    
    # Users can view their own profile or admins/managers can view others
    if str(current_user.id) != user_id and current_user.role not in [UserRole.ADMIN, UserRole.MANAGER]:
        raise HTTPException(status_code=403, detail="Permission denied")
    
    service = UserManagementService(db)
    user = await service.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return user

@router.post("/{user_id}/notifications", response_model=NotificationResponse)
async def create_notification(
    user_id: str,
    notification_data: NotificationCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create notification for user (admin/manager only)"""
    
    if current_user.role not in [UserRole.ADMIN, UserRole.MANAGER]:
        raise HTTPException(status_code=403, detail="Permission denied")
    
    service = UserManagementService(db)
    notification = await service.create_notification(user_id, notification_data)
    return notification