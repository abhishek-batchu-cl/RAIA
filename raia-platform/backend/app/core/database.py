"""
RAIA Platform - Database Configuration and Management
Enterprise-grade PostgreSQL with async support, connection pooling, and migrations
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

import structlog
from sqlalchemy import event, pool
from sqlalchemy.ext.asyncio import (
    AsyncSession, 
    async_sessionmaker, 
    create_async_engine,
    AsyncEngine
)
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

from app.core.config import get_settings
from app.models.schemas import Base

logger = structlog.get_logger(__name__)
settings = get_settings()

# Global database engine and session maker
_engine: Optional[AsyncEngine] = None
_session_factory: Optional[async_sessionmaker[AsyncSession]] = None


def create_database_engine() -> AsyncEngine:
    """
    Create and configure the database engine with connection pooling.
    """
    global _engine
    
    if _engine is not None:
        return _engine
    
    # Database configuration from settings
    db_config = settings.get_database_config()
    
    # Create async engine with connection pooling
    _engine = create_async_engine(
        str(db_config["url"]),
        echo=db_config["echo"],
        future=True,
        pool_size=db_config["pool_size"],
        max_overflow=db_config["max_overflow"],
        pool_timeout=db_config["pool_timeout"],
        pool_recycle=3600,  # Recycle connections every hour
        pool_pre_ping=True,  # Validate connections before use
        poolclass=QueuePool,
        connect_args={
            "command_timeout": 60,
            "server_settings": {
                "jit": "off",  # Disable JIT for better performance with small queries
                "application_name": "raia_platform",
            },
        },
    )
    
    # Add event listeners for monitoring
    @event.listens_for(_engine.sync_engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        """Set database connection parameters"""
        logger.debug("New database connection established")
    
    @event.listens_for(_engine.sync_engine, "checkout")
    def receive_checkout(dbapi_connection, connection_record, connection_proxy):
        """Log connection checkout from pool"""
        logger.debug("Connection checked out from pool")
    
    @event.listens_for(_engine.sync_engine, "checkin")
    def receive_checkin(dbapi_connection, connection_record):
        """Log connection checkin to pool"""
        logger.debug("Connection checked in to pool")
    
    logger.info(
        "Database engine created",
        url=str(db_config["url"]).split("@")[-1],  # Log without credentials
        pool_size=db_config["pool_size"],
        max_overflow=db_config["max_overflow"]
    )
    
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """
    Get or create the database session factory.
    """
    global _session_factory
    
    if _session_factory is not None:
        return _session_factory
    
    engine = create_database_engine()
    
    _session_factory = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=True,
        autocommit=False,
    )
    
    logger.info("Database session factory created")
    return _session_factory


@asynccontextmanager
async def get_database() -> AsyncGenerator[AsyncSession, None]:
    """
    Get a database session with automatic cleanup.
    This is the main dependency for FastAPI endpoints.
    """
    session_factory = get_session_factory()
    
    async with session_factory() as session:
        try:
            logger.debug("Database session created")
            yield session
        except Exception as e:
            logger.error("Database session error", error=str(e))
            await session.rollback()
            raise
        finally:
            await session.close()
            logger.debug("Database session closed")


async def create_tables():
    """
    Create all database tables.
    This should be called during application startup.
    """
    engine = create_database_engine()
    
    try:
        logger.info("Creating database tables...")
        
        async with engine.begin() as conn:
            # Create all tables defined in models
            await conn.run_sync(Base.metadata.create_all)
            
        logger.info("Database tables created successfully")
        
        # Initialize default data
        await initialize_default_data()
        
    except Exception as e:
        logger.error("Failed to create database tables", error=str(e))
        raise


async def drop_tables():
    """
    Drop all database tables.
    WARNING: This will delete all data!
    """
    engine = create_database_engine()
    
    try:
        logger.warning("Dropping all database tables...")
        
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            
        logger.warning("All database tables dropped")
        
    except Exception as e:
        logger.error("Failed to drop database tables", error=str(e))
        raise


async def initialize_default_data():
    """
    Initialize default data for the platform.
    This includes default roles, permissions, and system organization.
    """
    from app.services.user_service import UserService
    from app.services.organization_service import OrganizationService
    
    try:
        logger.info("Initializing default platform data...")
        
        async with get_database() as db:
            user_service = UserService()
            org_service = OrganizationService()
            
            # Create default organization
            default_org = await org_service.get_organization_by_slug(db, "system")
            if not default_org:
                from app.models.schemas import OrganizationCreate
                default_org_data = OrganizationCreate(
                    name="System Organization",
                    slug="system",
                    description="Default system organization for platform administration"
                )
                default_org = await org_service.create_organization(db, default_org_data)
                logger.info("Default organization created", org_id=default_org.id)
            
            # Create default roles and permissions
            await _create_default_roles_and_permissions(db, user_service)
            
            # Create system admin user if not exists
            admin_user = await user_service.get_user_by_username(db, "admin")
            if not admin_user:
                from app.models.schemas import UserCreate
                from app.core.auth import create_password_hash
                
                # Get super_admin role
                super_admin_role = await user_service.get_role_by_name(db, "super_admin")
                
                admin_user_data = UserCreate(
                    email="admin@raia-platform.local",
                    username="admin",
                    full_name="System Administrator",
                    password="admin123!@#",  # This should be changed immediately
                    organization_id=default_org.id,
                    role_ids=[super_admin_role.id] if super_admin_role else []
                )
                
                admin_user = await user_service.create_user(db, admin_user_data)
                logger.warning(
                    "Default admin user created", 
                    username="admin",
                    password="admin123!@#",
                    message="CHANGE PASSWORD IMMEDIATELY!"
                )
            
        logger.info("Default platform data initialized successfully")
        
    except Exception as e:
        logger.error("Failed to initialize default data", error=str(e))
        # Don't raise here as this is not critical for startup


async def _create_default_roles_and_permissions(db: AsyncSession, user_service):
    """Create default roles and permissions for the platform."""
    
    # Default permissions
    default_permissions = [
        # User management
        {"name": "user:read", "description": "View users", "resource": "user", "action": "read"},
        {"name": "user:write", "description": "Create and update users", "resource": "user", "action": "write"},
        {"name": "user:delete", "description": "Delete users", "resource": "user", "action": "delete"},
        
        # Organization management
        {"name": "org:read", "description": "View organizations", "resource": "organization", "action": "read"},
        {"name": "org:write", "description": "Create and update organizations", "resource": "organization", "action": "write"},
        {"name": "org:delete", "description": "Delete organizations", "resource": "organization", "action": "delete"},
        
        # Model management
        {"name": "model:read", "description": "View models", "resource": "model", "action": "read"},
        {"name": "model:write", "description": "Create and update models", "resource": "model", "action": "write"},
        {"name": "model:delete", "description": "Delete models", "resource": "model", "action": "delete"},
        {"name": "model:deploy", "description": "Deploy models", "resource": "model", "action": "deploy"},
        
        # Agent evaluation
        {"name": "agent_evaluation:read", "description": "View agent evaluations", "resource": "agent_evaluation", "action": "read"},
        {"name": "agent_evaluation:write", "description": "Create and run agent evaluations", "resource": "agent_evaluation", "action": "write"},
        {"name": "agent_evaluation:delete", "description": "Delete agent evaluations", "resource": "agent_evaluation", "action": "delete"},
        
        # Model explainability
        {"name": "model_explainability:read", "description": "View model explanations", "resource": "model_explainability", "action": "read"},
        {"name": "model_explainability:write", "description": "Generate model explanations", "resource": "model_explainability", "action": "write"},
        
        # Data management
        {"name": "data:read", "description": "View data", "resource": "data", "action": "read"},
        {"name": "data:write", "description": "Upload and manage data", "resource": "data", "action": "write"},
        {"name": "data:delete", "description": "Delete data", "resource": "data", "action": "delete"},
        
        # Monitoring and analytics
        {"name": "monitoring:read", "description": "View monitoring data", "resource": "monitoring", "action": "read"},
        {"name": "analytics:read", "description": "View analytics", "resource": "analytics", "action": "read"},
        
        # System administration
        {"name": "admin", "description": "Full system administration", "resource": "system", "action": "admin"},
        {"name": "system:read", "description": "View system information", "resource": "system", "action": "read"},
        {"name": "system:write", "description": "Modify system settings", "resource": "system", "action": "write"},
    ]
    
    # Create permissions
    created_permissions = []
    for perm_data in default_permissions:
        permission = await user_service.get_permission_by_name(db, perm_data["name"])
        if not permission:
            from app.models.schemas import Permission
            permission = Permission(**perm_data)
            db.add(permission)
            await db.flush()
            created_permissions.append(permission)
        else:
            created_permissions.append(permission)
    
    await db.commit()
    logger.info(f"Created {len(created_permissions)} permissions")
    
    # Default roles with their permissions
    default_roles = [
        {
            "name": "super_admin",
            "description": "Super administrator with all permissions",
            "permissions": [p.name for p in created_permissions]
        },
        {
            "name": "admin", 
            "description": "Organization administrator",
            "permissions": [
                "user:read", "user:write", "user:delete",
                "model:read", "model:write", "model:delete", "model:deploy",
                "agent_evaluation:read", "agent_evaluation:write", "agent_evaluation:delete",
                "model_explainability:read", "model_explainability:write",
                "data:read", "data:write", "data:delete",
                "monitoring:read", "analytics:read"
            ]
        },
        {
            "name": "analyst",
            "description": "Data analyst and model evaluator",
            "permissions": [
                "model:read", "model:write",
                "agent_evaluation:read", "agent_evaluation:write",
                "model_explainability:read", "model_explainability:write",
                "data:read", "data:write",
                "monitoring:read", "analytics:read"
            ]
        },
        {
            "name": "viewer",
            "description": "Read-only access to platform features",
            "permissions": [
                "model:read",
                "agent_evaluation:read",
                "model_explainability:read",
                "data:read",
                "monitoring:read", "analytics:read"
            ]
        }
    ]
    
    # Create roles
    permission_map = {p.name: p for p in created_permissions}
    
    for role_data in default_roles:
        role = await user_service.get_role_by_name(db, role_data["name"])
        if not role:
            from app.models.schemas import Role
            role = Role(
                name=role_data["name"],
                description=role_data["description"],
                is_system_role=True
            )
            db.add(role)
            await db.flush()
            
            # Add permissions to role
            for perm_name in role_data["permissions"]:
                if perm_name in permission_map:
                    role.permissions.append(permission_map[perm_name])
    
    await db.commit()
    logger.info(f"Created {len(default_roles)} default roles")


async def get_database_health() -> dict:
    """
    Check database health and return status information.
    """
    try:
        async with get_database() as db:
            # Simple query to test connectivity
            result = await db.execute("SELECT 1 as health_check")
            row = result.fetchone()
            
            if row and row[0] == 1:
                return {
                    "status": "healthy",
                    "message": "Database connection successful",
                    "timestamp": logger.info("Database health check passed")
                }
            else:
                return {
                    "status": "unhealthy", 
                    "message": "Database query failed",
                    "timestamp": logger.error("Database health check failed")
                }
                
    except Exception as e:
        logger.error("Database health check error", error=str(e))
        return {
            "status": "unhealthy",
            "message": f"Database connection failed: {str(e)}",
            "error": str(e)
        }


async def cleanup_database():
    """
    Clean up database connections and resources.
    This should be called during application shutdown.
    """
    global _engine, _session_factory
    
    if _engine:
        logger.info("Closing database connections...")
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("Database connections closed")