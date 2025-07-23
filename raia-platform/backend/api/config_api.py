# Configuration Management API
import os
import uuid
import json
import logging
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from enum import Enum

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, validator, SecretStr
from sqlalchemy.orm import Session
from sqlalchemy import Column, String, DateTime, Text, Boolean, JSON, UUID as PG_UUID, Integer, Float, ForeignKey, func, desc, and_
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

# Configuration validation
from jsonschema import validate, ValidationError
import configparser
import toml

# Environment and secrets management
import hvac  # HashiCorp Vault
from cryptography.fernet import Fernet
import base64

Base = declarative_base()
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/config", tags=["configuration"])

# Enums
class ConfigScope(str, Enum):
    SYSTEM = "system"
    ORGANIZATION = "organization"
    USER = "user"
    APPLICATION = "application"
    ENVIRONMENT = "environment"

class ConfigFormat(str, Enum):
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"
    INI = "ini"
    ENV = "env"

class ConfigCategory(str, Enum):
    DATABASE = "database"
    API = "api"
    SECURITY = "security"
    STORAGE = "storage"
    ML_CONFIG = "ml_config"
    UI_SETTINGS = "ui_settings"
    NOTIFICATION = "notification"
    INTEGRATION = "integration"
    MONITORING = "monitoring"
    GENERAL = "general"

class ConfigStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"
    DRAFT = "draft"

# Database Models
class ConfigEntry(Base):
    """Configuration entries"""
    __tablename__ = "config_entries"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Configuration identification
    key = Column(String(255), nullable=False, index=True)
    display_name = Column(String(255))
    description = Column(Text)
    
    # Configuration data
    value = Column(JSON)  # The actual configuration value
    default_value = Column(JSON)  # Default fallback value
    data_type = Column(String(50))  # string, integer, float, boolean, object, array
    
    # Scope and context
    scope = Column(String(50), nullable=False, index=True)
    scope_id = Column(String(255), index=True)  # organization_id, user_id, etc.
    category = Column(String(100), index=True)
    
    # Format and validation
    format = Column(String(50), default=ConfigFormat.JSON)
    schema = Column(JSON)  # JSON schema for validation
    constraints = Column(JSON)  # Additional constraints
    
    # Security and access
    is_sensitive = Column(Boolean, default=False)  # Contains sensitive data
    is_encrypted = Column(Boolean, default=False)
    access_level = Column(String(50), default="internal")  # public, internal, restricted
    
    # Behavior
    is_readonly = Column(Boolean, default=False)
    is_required = Column(Boolean, default=False)
    can_override = Column(Boolean, default=True)
    
    # Metadata
    tags = Column(JSON)
    environment = Column(String(100))  # dev, staging, prod
    version = Column(String(50), default="1.0.0")
    
    # Status and lifecycle
    status = Column(String(50), default=ConfigStatus.ACTIVE)
    deprecated_at = Column(DateTime)
    expires_at = Column(DateTime)
    
    # Change tracking
    change_reason = Column(Text)
    changed_by = Column(String(255))
    approved_by = Column(String(255))
    approved_at = Column(DateTime)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    history = relationship("ConfigHistory", back_populates="config_entry", cascade="all, delete-orphan")

class ConfigHistory(Base):
    """Configuration change history"""
    __tablename__ = "config_history"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    config_entry_id = Column(PG_UUID(as_uuid=True), ForeignKey('config_entries.id'), nullable=False)
    
    # Change details
    old_value = Column(JSON)
    new_value = Column(JSON)
    change_type = Column(String(50))  # create, update, delete, restore
    change_reason = Column(Text)
    
    # Change metadata
    changed_by = Column(String(255), nullable=False)
    change_source = Column(String(100))  # api, ui, import, migration
    client_info = Column(JSON)  # IP, user agent, etc.
    
    # Approval workflow
    requires_approval = Column(Boolean, default=False)
    approved_by = Column(String(255))
    approved_at = Column(DateTime)
    approval_reason = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    config_entry = relationship("ConfigEntry", back_populates="history")

class ConfigTemplate(Base):
    """Configuration templates"""
    __tablename__ = "config_templates"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    
    # Template configuration
    template_data = Column(JSON, nullable=False)  # Template structure
    template_format = Column(String(50), default=ConfigFormat.JSON)
    category = Column(String(100))
    
    # Template metadata
    version = Column(String(50), default="1.0.0")
    tags = Column(JSON)
    
    # Usage tracking
    usage_count = Column(Integer, default=0)
    last_used = Column(DateTime)
    
    # Access control
    is_public = Column(Boolean, default=False)
    created_by = Column(String(255), nullable=False)
    organization_id = Column(String(255))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ConfigEnvironment(Base):
    """Environment-specific configuration sets"""
    __tablename__ = "config_environments"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    
    # Environment details
    environment_type = Column(String(50), nullable=False)  # development, staging, production
    base_environment = Column(String(255))  # Parent environment to inherit from
    
    # Configuration overrides
    config_overrides = Column(JSON)  # Environment-specific config values
    
    # Status
    is_active = Column(Boolean, default=True)
    
    # Access control
    created_by = Column(String(255), nullable=False)
    organization_id = Column(String(255))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Pydantic Models
class ConfigEntryCreate(BaseModel):
    key: str
    display_name: Optional[str] = None
    description: Optional[str] = None
    value: Any
    default_value: Optional[Any] = None
    data_type: str = "string"
    scope: ConfigScope
    scope_id: Optional[str] = None
    category: ConfigCategory = ConfigCategory.GENERAL
    format: ConfigFormat = ConfigFormat.JSON
    schema: Optional[Dict[str, Any]] = None
    is_sensitive: bool = False
    is_readonly: bool = False
    is_required: bool = False
    tags: Optional[List[str]] = []
    environment: Optional[str] = None
    change_reason: Optional[str] = None
    
    @validator('key')
    def validate_key(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Key cannot be empty')
        if not v.replace('.', '_').replace('-', '_').replace('_', '').isalnum():
            raise ValueError('Key must contain only alphanumeric characters, dots, dashes, and underscores')
        return v.strip()

class ConfigEntryUpdate(BaseModel):
    display_name: Optional[str] = None
    description: Optional[str] = None
    value: Optional[Any] = None
    default_value: Optional[Any] = None
    schema: Optional[Dict[str, Any]] = None
    is_sensitive: Optional[bool] = None
    is_readonly: Optional[bool] = None
    status: Optional[ConfigStatus] = None
    tags: Optional[List[str]] = None
    change_reason: Optional[str] = None

class ConfigEntryResponse(BaseModel):
    id: str
    key: str
    display_name: Optional[str]
    description: Optional[str]
    value: Any
    data_type: str
    scope: str
    scope_id: Optional[str]
    category: str
    format: str
    is_sensitive: bool
    is_readonly: bool
    is_required: bool
    status: str
    version: str
    environment: Optional[str]
    created_at: datetime
    updated_at: datetime
    changed_by: Optional[str]
    
    class Config:
        orm_mode = True

class ConfigTemplateCreate(BaseModel):
    name: str
    description: Optional[str] = None
    template_data: Dict[str, Any]
    template_format: ConfigFormat = ConfigFormat.JSON
    category: Optional[ConfigCategory] = None
    tags: Optional[List[str]] = []
    is_public: bool = False

class ConfigTemplateResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    template_data: Dict[str, Any]
    category: Optional[str]
    version: str
    usage_count: int
    is_public: bool
    created_by: str
    created_at: datetime
    
    class Config:
        orm_mode = True

class ConfigEnvironmentCreate(BaseModel):
    name: str
    description: Optional[str] = None
    environment_type: str
    base_environment: Optional[str] = None
    config_overrides: Optional[Dict[str, Any]] = {}

class ConfigBulkUpdate(BaseModel):
    updates: List[Dict[str, Any]]
    change_reason: Optional[str] = None
    requires_approval: bool = False

# Configuration Management Service
class ConfigurationService:
    """Service for configuration management"""
    
    def __init__(self, db: Session):
        self.db = db
        
        # Encryption key for sensitive values
        self.encryption_key = self._get_encryption_key()
        
        # Configuration cache
        self.config_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Validation schemas
        self.validation_schemas = {
            'database': {
                'type': 'object',
                'properties': {
                    'host': {'type': 'string'},
                    'port': {'type': 'integer', 'minimum': 1, 'maximum': 65535},
                    'database': {'type': 'string'},
                    'username': {'type': 'string'},
                    'password': {'type': 'string'},
                    'ssl': {'type': 'boolean'}
                },
                'required': ['host', 'database']
            },
            'api': {
                'type': 'object',
                'properties': {
                    'base_url': {'type': 'string', 'format': 'uri'},
                    'timeout': {'type': 'integer', 'minimum': 1},
                    'rate_limit': {'type': 'integer', 'minimum': 1},
                    'api_key': {'type': 'string'}
                }
            }
        }
    
    async def create_config(self, config_data: ConfigEntryCreate, user_id: str) -> ConfigEntry:
        """Create a new configuration entry"""
        
        # Check if key already exists in the same scope
        existing = self._get_config_by_key(config_data.key, config_data.scope, config_data.scope_id)
        if existing:
            raise HTTPException(
                status_code=409, 
                detail=f"Configuration key '{config_data.key}' already exists in this scope"
            )
        
        # Validate value against schema
        if config_data.schema:
            try:
                validate(config_data.value, config_data.schema)
            except ValidationError as e:
                raise HTTPException(status_code=400, detail=f"Validation error: {e.message}")
        
        # Encrypt sensitive values
        value_to_store = config_data.value
        is_encrypted = False
        
        if config_data.is_sensitive:
            value_to_store = self._encrypt_value(config_data.value)
            is_encrypted = True
        
        # Create configuration entry
        config_entry = ConfigEntry(
            key=config_data.key,
            display_name=config_data.display_name or config_data.key,
            description=config_data.description,
            value=value_to_store,
            default_value=config_data.default_value,
            data_type=config_data.data_type,
            scope=config_data.scope,
            scope_id=config_data.scope_id,
            category=config_data.category,
            format=config_data.format,
            schema=config_data.schema,
            is_sensitive=config_data.is_sensitive,
            is_encrypted=is_encrypted,
            is_readonly=config_data.is_readonly,
            is_required=config_data.is_required,
            tags=config_data.tags,
            environment=config_data.environment,
            changed_by=user_id
        )
        
        self.db.add(config_entry)
        self.db.commit()
        self.db.refresh(config_entry)
        
        # Record history
        await self._record_history(config_entry, None, config_entry.value, "create", user_id, config_data.change_reason)
        
        # Clear cache
        self._clear_cache()
        
        logger.info(f"Created configuration {config_entry.key} (ID: {config_entry.id})")
        return config_entry
    
    async def update_config(self, config_id: str, config_data: ConfigEntryUpdate, user_id: str) -> ConfigEntry:
        """Update a configuration entry"""
        
        config_entry = self.db.query(ConfigEntry).filter(ConfigEntry.id == config_id).first()
        if not config_entry:
            raise HTTPException(status_code=404, detail="Configuration not found")
        
        if config_entry.is_readonly:
            raise HTTPException(status_code=403, detail="Configuration is read-only")
        
        # Store old value for history
        old_value = config_entry.value
        
        # Update fields
        update_data = config_data.dict(exclude_unset=True)
        
        for field, value in update_data.items():
            if field == 'value':
                # Validate new value
                if config_entry.schema:
                    try:
                        validate(value, config_entry.schema)
                    except ValidationError as e:
                        raise HTTPException(status_code=400, detail=f"Validation error: {e.message}")
                
                # Encrypt if sensitive
                if config_entry.is_sensitive:
                    value = self._encrypt_value(value)
                
                setattr(config_entry, field, value)
            elif field != 'change_reason':
                setattr(config_entry, field, value)
        
        config_entry.updated_at = datetime.utcnow()
        config_entry.changed_by = user_id
        
        self.db.commit()
        
        # Record history
        await self._record_history(config_entry, old_value, config_entry.value, "update", user_id, config_data.change_reason)
        
        # Clear cache
        self._clear_cache()
        
        return config_entry
    
    async def get_config(self, key: str, scope: ConfigScope, scope_id: Optional[str] = None, environment: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get configuration value with hierarchy resolution"""
        
        # Check cache first
        cache_key = f"{key}:{scope}:{scope_id or 'none'}:{environment or 'none'}"
        cached = self.config_cache.get(cache_key)
        
        if cached and (datetime.utcnow() - cached['timestamp']).total_seconds() < self.cache_ttl:
            return cached['value']
        
        # Resolution hierarchy: specific -> general
        config_value = None
        
        # 1. Try exact match
        config_entry = self._get_config_by_key(key, scope, scope_id, environment)
        
        if not config_entry:
            # 2. Try without environment
            config_entry = self._get_config_by_key(key, scope, scope_id)
        
        if not config_entry and scope != ConfigScope.SYSTEM:
            # 3. Try system scope
            config_entry = self._get_config_by_key(key, ConfigScope.SYSTEM)
        
        if config_entry:
            value = config_entry.value
            
            # Decrypt if needed
            if config_entry.is_encrypted:
                value = self._decrypt_value(value)
            
            config_value = {
                'key': config_entry.key,
                'value': value,
                'data_type': config_entry.data_type,
                'source': {
                    'scope': config_entry.scope,
                    'scope_id': config_entry.scope_id,
                    'environment': config_entry.environment
                },
                'metadata': {
                    'is_sensitive': config_entry.is_sensitive,
                    'is_readonly': config_entry.is_readonly,
                    'category': config_entry.category,
                    'version': config_entry.version,
                    'updated_at': config_entry.updated_at.isoformat()
                }
            }
            
            # Cache the result
            self.config_cache[cache_key] = {
                'value': config_value,
                'timestamp': datetime.utcnow()
            }
        
        return config_value
    
    async def get_config_batch(self, keys: List[str], scope: ConfigScope, scope_id: Optional[str] = None, environment: Optional[str] = None) -> Dict[str, Any]:
        """Get multiple configuration values"""
        
        results = {}
        
        for key in keys:
            config_value = await self.get_config(key, scope, scope_id, environment)
            if config_value:
                results[key] = config_value
        
        return results
    
    async def list_configs(self, scope: Optional[ConfigScope] = None, category: Optional[ConfigCategory] = None, 
                          environment: Optional[str] = None, user_id: Optional[str] = None, 
                          skip: int = 0, limit: int = 100) -> List[ConfigEntry]:
        """List configuration entries with filtering"""
        
        query = self.db.query(ConfigEntry).filter(ConfigEntry.status == ConfigStatus.ACTIVE)
        
        if scope:
            query = query.filter(ConfigEntry.scope == scope)
        
        if category:
            query = query.filter(ConfigEntry.category == category)
        
        if environment:
            query = query.filter(
                (ConfigEntry.environment == environment) | 
                (ConfigEntry.environment.is_(None))
            )
        
        # Apply access control
        if user_id:
            # Users can see public configs, their own user configs, and organization configs
            org_id = self._get_user_org(user_id)
            query = query.filter(
                (ConfigEntry.access_level == 'public') |
                (ConfigEntry.scope == ConfigScope.USER and ConfigEntry.scope_id == user_id) |
                (ConfigEntry.scope == ConfigScope.ORGANIZATION and ConfigEntry.scope_id == org_id) |
                (ConfigEntry.scope == ConfigScope.SYSTEM)
            )
        
        return query.order_by(ConfigEntry.key).offset(skip).limit(limit).all()
    
    async def create_template(self, template_data: ConfigTemplateCreate, user_id: str) -> ConfigTemplate:
        """Create a configuration template"""
        
        template = ConfigTemplate(
            name=template_data.name,
            description=template_data.description,
            template_data=template_data.template_data,
            template_format=template_data.template_format,
            category=template_data.category,
            tags=template_data.tags,
            is_public=template_data.is_public,
            created_by=user_id,
            organization_id=self._get_user_org(user_id)
        )
        
        self.db.add(template)
        self.db.commit()
        self.db.refresh(template)
        
        return template
    
    async def apply_template(self, template_id: str, scope: ConfigScope, scope_id: Optional[str], 
                           environment: Optional[str], overrides: Optional[Dict[str, Any]], 
                           user_id: str) -> Dict[str, Any]:
        """Apply a configuration template"""
        
        template = self.db.query(ConfigTemplate).filter(ConfigTemplate.id == template_id).first()
        if not template:
            raise HTTPException(status_code=404, detail="Template not found")
        
        # Update usage count
        template.usage_count += 1
        template.last_used = datetime.utcnow()
        
        created_configs = []
        errors = []
        
        # Apply template configurations
        for key, config_def in template.template_data.items():
            try:
                # Apply overrides if provided
                final_value = overrides.get(key, config_def.get('value', config_def.get('default_value')))
                
                config_data = ConfigEntryCreate(
                    key=key,
                    display_name=config_def.get('display_name', key),
                    description=config_def.get('description'),
                    value=final_value,
                    default_value=config_def.get('default_value'),
                    data_type=config_def.get('data_type', 'string'),
                    scope=scope,
                    scope_id=scope_id,
                    category=ConfigCategory(config_def.get('category', 'general')),
                    format=ConfigFormat(config_def.get('format', 'json')),
                    schema=config_def.get('schema'),
                    is_sensitive=config_def.get('is_sensitive', False),
                    is_readonly=config_def.get('is_readonly', False),
                    is_required=config_def.get('is_required', False),
                    environment=environment,
                    change_reason=f"Applied from template: {template.name}"
                )
                
                # Check if config already exists
                existing = self._get_config_by_key(key, scope, scope_id, environment)
                if existing:
                    # Update existing
                    update_data = ConfigEntryUpdate(
                        value=final_value,
                        change_reason=f"Updated from template: {template.name}"
                    )
                    config_entry = await self.update_config(str(existing.id), update_data, user_id)
                else:
                    # Create new
                    config_entry = await self.create_config(config_data, user_id)
                
                created_configs.append({
                    'key': key,
                    'id': str(config_entry.id),
                    'action': 'updated' if existing else 'created'
                })
                
            except Exception as e:
                errors.append({
                    'key': key,
                    'error': str(e)
                })
        
        self.db.commit()
        
        return {
            'template_name': template.name,
            'applied_configs': created_configs,
            'errors': errors,
            'success_count': len(created_configs),
            'error_count': len(errors)
        }
    
    async def bulk_update(self, updates: ConfigBulkUpdate, user_id: str) -> Dict[str, Any]:
        """Bulk update multiple configurations"""
        
        results = []
        errors = []
        
        for update in updates.updates:
            try:
                config_id = update.get('id')
                if not config_id:
                    raise ValueError("Missing configuration ID")
                
                update_data = ConfigEntryUpdate(
                    **{k: v for k, v in update.items() if k != 'id'},
                    change_reason=updates.change_reason
                )
                
                config_entry = await self.update_config(config_id, update_data, user_id)
                
                results.append({
                    'id': config_id,
                    'key': config_entry.key,
                    'status': 'success'
                })
                
            except Exception as e:
                errors.append({
                    'id': update.get('id'),
                    'error': str(e)
                })
        
        return {
            'results': results,
            'errors': errors,
            'success_count': len(results),
            'error_count': len(errors)
        }
    
    async def export_config(self, scope: ConfigScope, scope_id: Optional[str] = None, 
                          environment: Optional[str] = None, format: ConfigFormat = ConfigFormat.JSON) -> str:
        """Export configurations to specified format"""
        
        configs = await self.list_configs(scope, environment=environment)
        
        # Filter by scope_id if provided
        if scope_id:
            configs = [c for c in configs if c.scope_id == scope_id]
        
        # Build export data
        export_data = {}
        
        for config in configs:
            value = config.value
            
            # Decrypt if needed (only for authorized exports)
            if config.is_encrypted:
                value = self._decrypt_value(value)
            
            # Don't export sensitive values in plain text
            if config.is_sensitive:
                value = "*** SENSITIVE ***"
            
            export_data[config.key] = {
                'value': value,
                'data_type': config.data_type,
                'category': config.category,
                'is_sensitive': config.is_sensitive,
                'is_required': config.is_required,
                'description': config.description
            }
        
        # Format output
        if format == ConfigFormat.JSON:
            return json.dumps(export_data, indent=2, default=str)
        elif format == ConfigFormat.YAML:
            return yaml.dump(export_data, default_flow_style=False)
        elif format == ConfigFormat.TOML:
            return toml.dumps(export_data)
        else:
            raise HTTPException(status_code=400, detail=f"Export format {format} not supported")
    
    async def import_config(self, config_data: str, format: ConfigFormat, scope: ConfigScope, 
                          scope_id: Optional[str], environment: Optional[str], user_id: str) -> Dict[str, Any]:
        """Import configurations from formatted data"""
        
        try:
            # Parse input data
            if format == ConfigFormat.JSON:
                data = json.loads(config_data)
            elif format == ConfigFormat.YAML:
                data = yaml.safe_load(config_data)
            elif format == ConfigFormat.TOML:
                data = toml.loads(config_data)
            else:
                raise HTTPException(status_code=400, detail=f"Import format {format} not supported")
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse {format} data: {str(e)}")
        
        created_configs = []
        updated_configs = []
        errors = []
        
        # Process each configuration
        for key, config_def in data.items():
            try:
                # Extract configuration details
                if isinstance(config_def, dict):
                    value = config_def.get('value')
                    data_type = config_def.get('data_type', 'string')
                    category = config_def.get('category', 'general')
                    description = config_def.get('description')
                    is_sensitive = config_def.get('is_sensitive', False)
                    is_required = config_def.get('is_required', False)
                else:
                    # Simple key-value pair
                    value = config_def
                    data_type = 'string'
                    category = 'general'
                    description = None
                    is_sensitive = False
                    is_required = False
                
                # Check if configuration exists
                existing = self._get_config_by_key(key, scope, scope_id, environment)
                
                if existing:
                    # Update existing
                    update_data = ConfigEntryUpdate(
                        value=value,
                        change_reason=f"Imported from {format} data"
                    )
                    await self.update_config(str(existing.id), update_data, user_id)
                    updated_configs.append(key)
                else:
                    # Create new
                    create_data = ConfigEntryCreate(
                        key=key,
                        value=value,
                        data_type=data_type,
                        scope=scope,
                        scope_id=scope_id,
                        category=ConfigCategory(category),
                        description=description,
                        is_sensitive=is_sensitive,
                        is_required=is_required,
                        environment=environment,
                        change_reason=f"Imported from {format} data"
                    )
                    await self.create_config(create_data, user_id)
                    created_configs.append(key)
                    
            except Exception as e:
                errors.append({
                    'key': key,
                    'error': str(e)
                })
        
        return {
            'created_configs': created_configs,
            'updated_configs': updated_configs,
            'errors': errors,
            'total_processed': len(created_configs) + len(updated_configs) + len(errors)
        }
    
    # Private helper methods
    def _get_config_by_key(self, key: str, scope: ConfigScope, scope_id: Optional[str] = None, 
                          environment: Optional[str] = None) -> Optional[ConfigEntry]:
        """Get configuration by key with scope resolution"""
        
        query = self.db.query(ConfigEntry).filter(
            ConfigEntry.key == key,
            ConfigEntry.scope == scope,
            ConfigEntry.status == ConfigStatus.ACTIVE
        )
        
        if scope_id:
            query = query.filter(ConfigEntry.scope_id == scope_id)
        else:
            query = query.filter(ConfigEntry.scope_id.is_(None))
        
        if environment:
            query = query.filter(ConfigEntry.environment == environment)
        else:
            query = query.filter(ConfigEntry.environment.is_(None))
        
        return query.first()
    
    async def _record_history(self, config_entry: ConfigEntry, old_value: Any, new_value: Any, 
                            change_type: str, user_id: str, reason: Optional[str]):
        """Record configuration change history"""
        
        history = ConfigHistory(
            config_entry_id=config_entry.id,
            old_value=old_value,
            new_value=new_value,
            change_type=change_type,
            change_reason=reason,
            changed_by=user_id,
            change_source='api'
        )
        
        self.db.add(history)
    
    def _get_encryption_key(self) -> bytes:
        """Get encryption key for sensitive values"""
        
        # In production, this should come from a secure key management system
        key_env = os.getenv('CONFIG_ENCRYPTION_KEY')
        
        if key_env:
            return base64.urlsafe_b64decode(key_env.encode())
        else:
            # Generate a key for development (not secure for production)
            return Fernet.generate_key()
    
    def _encrypt_value(self, value: Any) -> str:
        """Encrypt a sensitive value"""
        
        fernet = Fernet(self.encryption_key)
        
        # Convert value to JSON string
        json_value = json.dumps(value, default=str)
        
        # Encrypt
        encrypted = fernet.encrypt(json_value.encode())
        
        return base64.urlsafe_b64encode(encrypted).decode()
    
    def _decrypt_value(self, encrypted_value: str) -> Any:
        """Decrypt a sensitive value"""
        
        try:
            fernet = Fernet(self.encryption_key)
            
            # Decode and decrypt
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_value.encode())
            decrypted = fernet.decrypt(encrypted_bytes)
            
            # Parse JSON
            return json.loads(decrypted.decode())
            
        except Exception as e:
            logger.error(f"Failed to decrypt value: {str(e)}")
            return None
    
    def _clear_cache(self):
        """Clear configuration cache"""
        self.config_cache.clear()
    
    def _get_user_org(self, user_id: str) -> Optional[str]:
        """Get user's organization ID"""
        return "default_org"

# Dependency injection
def get_db():
    pass

def get_current_user():
    return "current_user_id"

# API Endpoints
@router.post("/", response_model=ConfigEntryResponse)
async def create_configuration(
    config_data: ConfigEntryCreate,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Create a new configuration entry"""
    
    service = ConfigurationService(db)
    config = await service.create_config(config_data, current_user)
    
    # Don't return sensitive values
    if config.is_sensitive:
        config.value = "*** SENSITIVE ***"
    
    return config

@router.get("/", response_model=List[ConfigEntryResponse])
async def list_configurations(
    scope: Optional[ConfigScope] = None,
    category: Optional[ConfigCategory] = None,
    environment: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """List configuration entries"""
    
    service = ConfigurationService(db)
    configs = await service.list_configs(scope, category, environment, current_user, skip, limit)
    
    # Mask sensitive values
    for config in configs:
        if config.is_sensitive:
            config.value = "*** SENSITIVE ***"
    
    return configs

@router.get("/{config_id}", response_model=ConfigEntryResponse)
async def get_configuration(
    config_id: str,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Get a specific configuration entry"""
    
    config = db.query(ConfigEntry).filter(ConfigEntry.id == config_id).first()
    if not config:
        raise HTTPException(status_code=404, detail="Configuration not found")
    
    # Mask sensitive values
    if config.is_sensitive:
        config.value = "*** SENSITIVE ***"
    
    return config

@router.put("/{config_id}", response_model=ConfigEntryResponse)
async def update_configuration(
    config_id: str,
    config_data: ConfigEntryUpdate,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Update a configuration entry"""
    
    service = ConfigurationService(db)
    config = await service.update_config(config_id, config_data, current_user)
    
    # Mask sensitive values
    if config.is_sensitive:
        config.value = "*** SENSITIVE ***"
    
    return config

@router.delete("/{config_id}")
async def delete_configuration(
    config_id: str,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Delete a configuration entry"""
    
    config = db.query(ConfigEntry).filter(ConfigEntry.id == config_id).first()
    if not config:
        raise HTTPException(status_code=404, detail="Configuration not found")
    
    if config.is_readonly:
        raise HTTPException(status_code=403, detail="Configuration is read-only")
    
    config.status = ConfigStatus.INACTIVE
    config.updated_at = datetime.utcnow()
    config.changed_by = current_user
    
    db.commit()
    
    return {"message": "Configuration deleted successfully"}

@router.get("/resolve/{key}")
async def resolve_configuration(
    key: str,
    scope: ConfigScope,
    scope_id: Optional[str] = None,
    environment: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Resolve configuration value with hierarchy"""
    
    service = ConfigurationService(db)
    config_value = await service.get_config(key, scope, scope_id, environment)
    
    if not config_value:
        raise HTTPException(status_code=404, detail="Configuration not found")
    
    # Mask sensitive values
    if config_value['metadata']['is_sensitive']:
        config_value['value'] = "*** SENSITIVE ***"
    
    return config_value

@router.post("/batch")
async def get_batch_configuration(
    keys: List[str],
    scope: ConfigScope,
    scope_id: Optional[str] = None,
    environment: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Get multiple configuration values"""
    
    service = ConfigurationService(db)
    configs = await service.get_config_batch(keys, scope, scope_id, environment)
    
    # Mask sensitive values
    for key, config_value in configs.items():
        if config_value['metadata']['is_sensitive']:
            config_value['value'] = "*** SENSITIVE ***"
    
    return configs

@router.post("/templates", response_model=ConfigTemplateResponse)
async def create_template(
    template_data: ConfigTemplateCreate,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Create a configuration template"""
    
    service = ConfigurationService(db)
    template = await service.create_template(template_data, current_user)
    return template

@router.get("/templates", response_model=List[ConfigTemplateResponse])
async def list_templates(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """List configuration templates"""
    
    service = ConfigurationService(db)
    
    query = db.query(ConfigTemplate).filter(
        (ConfigTemplate.created_by == current_user) |
        (ConfigTemplate.is_public == True) |
        (ConfigTemplate.organization_id == service._get_user_org(current_user))
    )
    
    templates = query.order_by(desc(ConfigTemplate.created_at)).offset(skip).limit(limit).all()
    return templates

@router.post("/templates/{template_id}/apply")
async def apply_template(
    template_id: str,
    scope: ConfigScope,
    scope_id: Optional[str] = None,
    environment: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Apply a configuration template"""
    
    service = ConfigurationService(db)
    result = await service.apply_template(template_id, scope, scope_id, environment, overrides, current_user)
    return result

@router.post("/bulk-update")
async def bulk_update_configurations(
    updates: ConfigBulkUpdate,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Bulk update multiple configurations"""
    
    service = ConfigurationService(db)
    result = await service.bulk_update(updates, current_user)
    return result

@router.get("/export/{scope}")
async def export_configurations(
    scope: ConfigScope,
    scope_id: Optional[str] = None,
    environment: Optional[str] = None,
    format: ConfigFormat = ConfigFormat.JSON,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Export configurations"""
    
    service = ConfigurationService(db)
    exported_data = await service.export_config(scope, scope_id, environment, format)
    
    return {
        'format': format,
        'scope': scope,
        'data': exported_data
    }

@router.post("/import/{scope}")
async def import_configurations(
    scope: ConfigScope,
    config_data: str,
    format: ConfigFormat,
    scope_id: Optional[str] = None,
    environment: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Import configurations"""
    
    service = ConfigurationService(db)
    result = await service.import_config(config_data, format, scope, scope_id, environment, current_user)
    return result

@router.get("/{config_id}/history")
async def get_configuration_history(
    config_id: str,
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Get configuration change history"""
    
    config = db.query(ConfigEntry).filter(ConfigEntry.id == config_id).first()
    if not config:
        raise HTTPException(status_code=404, detail="Configuration not found")
    
    history = db.query(ConfigHistory).filter(
        ConfigHistory.config_entry_id == config_id
    ).order_by(desc(ConfigHistory.created_at)).offset(skip).limit(limit).all()
    
    # Mask sensitive values in history
    for entry in history:
        if config.is_sensitive:
            if entry.old_value:
                entry.old_value = "*** SENSITIVE ***"
            if entry.new_value:
                entry.new_value = "*** SENSITIVE ***"
    
    return [
        {
            'id': str(h.id),
            'old_value': h.old_value,
            'new_value': h.new_value,
            'change_type': h.change_type,
            'change_reason': h.change_reason,
            'changed_by': h.changed_by,
            'created_at': h.created_at
        }
        for h in history
    ]