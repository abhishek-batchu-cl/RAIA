"""
RAIA Platform - Unified Data Models and Schemas
Comprehensive Pydantic models combining features from both platforms
"""

from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, EmailStr, validator
from sqlalchemy import Column, String, Boolean, DateTime, Text, Integer, Float, JSON, ForeignKey, Table, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import uuid

Base = declarative_base()

# Association tables for many-to-many relationships
user_roles_association = Table(
    'user_roles',
    Base.metadata,
    Column('user_id', PGUUID(as_uuid=True), ForeignKey('users.id')),
    Column('role_id', PGUUID(as_uuid=True), ForeignKey('roles.id'))
)

role_permissions_association = Table(
    'role_permissions',
    Base.metadata,
    Column('role_id', PGUUID(as_uuid=True), ForeignKey('roles.id')),
    Column('permission_id', PGUUID(as_uuid=True), ForeignKey('permissions.id'))
)


# Enums
class UserStatusEnum(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING = "pending"


class OrganizationStatusEnum(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    TRIAL = "trial"
    SUSPENDED = "suspended"


class ModelTypeEnum(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    NLP = "nlp"
    COMPUTER_VISION = "computer_vision"
    MULTIMODAL = "multimodal"


class ModelStatusEnum(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    TRAINING = "training"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"


class EvaluationStatusEnum(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AlertSeverityEnum(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ModelProviderEnum(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"
    AZURE = "azure"
    AWS = "aws"


# SQLAlchemy Models
class Organization(Base):
    __tablename__ = "organizations"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, index=True)
    slug = Column(String(100), nullable=False, unique=True, index=True)
    description = Column(Text)
    status = Column(SQLEnum(OrganizationStatusEnum), default=OrganizationStatusEnum.ACTIVE)
    settings = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    users = relationship("User", back_populates="organization")
    models = relationship("Model", back_populates="organization")
    evaluations = relationship("Evaluation", back_populates="organization")


class User(Base):
    __tablename__ = "users"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), nullable=False, unique=True, index=True)
    username = Column(String(100), nullable=False, unique=True, index=True)
    full_name = Column(String(255), nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    status = Column(SQLEnum(UserStatusEnum), default=UserStatusEnum.ACTIVE)
    api_key = Column(String(255), unique=True, index=True)
    last_login_at = Column(DateTime)
    preferences = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Foreign Keys
    organization_id = Column(PGUUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    
    # Relationships
    organization = relationship("Organization", back_populates="users")
    roles = relationship("Role", secondary=user_roles_association, back_populates="users")
    chat_sessions = relationship("ChatSession", back_populates="user")
    evaluations = relationship("Evaluation", back_populates="created_by_user")


class Role(Base):
    __tablename__ = "roles"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False, unique=True, index=True)
    description = Column(Text)
    is_system_role = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    users = relationship("User", secondary=user_roles_association, back_populates="roles")
    permissions = relationship("Permission", secondary=role_permissions_association, back_populates="roles")


class Permission(Base):
    __tablename__ = "permissions"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False, unique=True, index=True)
    description = Column(Text)
    resource = Column(String(100), nullable=False)
    action = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    roles = relationship("Role", secondary=role_permissions_association, back_populates="permissions")


class Model(Base):
    __tablename__ = "models"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text)
    model_type = Column(SQLEnum(ModelTypeEnum), nullable=False)
    provider = Column(SQLEnum(ModelProviderEnum), nullable=False)
    version = Column(String(50), nullable=False)
    status = Column(SQLEnum(ModelStatusEnum), default=ModelStatusEnum.ACTIVE)
    metadata = Column(JSON, default=dict)
    configuration = Column(JSON, default=dict)
    performance_metrics = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Foreign Keys
    organization_id = Column(PGUUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    
    # Relationships
    organization = relationship("Organization", back_populates="models")
    evaluations = relationship("Evaluation", back_populates="model")
    predictions = relationship("Prediction", back_populates="model")


class Evaluation(Base):
    __tablename__ = "evaluations"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    status = Column(SQLEnum(EvaluationStatusEnum), default=EvaluationStatusEnum.PENDING)
    evaluation_type = Column(String(100), nullable=False)  # agent_evaluation, explainability, fairness, etc.
    configuration = Column(JSON, nullable=False)
    results = Column(JSON, default=dict)
    metrics = Column(JSON, default=dict)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Foreign Keys
    organization_id = Column(PGUUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    model_id = Column(PGUUID(as_uuid=True), ForeignKey("models.id"), nullable=False)
    created_by = Column(PGUUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Relationships
    organization = relationship("Organization", back_populates="evaluations")
    model = relationship("Model", back_populates="evaluations")
    created_by_user = relationship("User", back_populates="evaluations")


class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    input_data = Column(JSON, nullable=False)
    output_data = Column(JSON, nullable=False)
    confidence_score = Column(Float)
    explanation_data = Column(JSON, default=dict)
    processing_time_ms = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Foreign Keys
    model_id = Column(PGUUID(as_uuid=True), ForeignKey("models.id"), nullable=False)
    
    # Relationships
    model = relationship("Model", back_populates="predictions")


class Document(Base):
    __tablename__ = "documents"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_type = Column(String(50), nullable=False)
    file_size = Column(Integer, nullable=False)
    file_path = Column(String(500), nullable=False)
    content_hash = Column(String(64), nullable=False, index=True)
    processed = Column(Boolean, default=False)
    processing_metadata = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime)
    
    # Foreign Keys
    organization_id = Column(PGUUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)


class ChatSession(Base):
    __tablename__ = "chat_sessions"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255))
    configuration = Column(JSON, default=dict)
    messages = Column(JSON, default=list)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Foreign Keys
    user_id = Column(PGUUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    organization_id = Column(PGUUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="chat_sessions")


class Alert(Base):
    __tablename__ = "alerts"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    severity = Column(SQLEnum(AlertSeverityEnum), nullable=False)
    alert_type = Column(String(100), nullable=False)
    source = Column(String(100), nullable=False)
    metadata = Column(JSON, default=dict)
    acknowledged = Column(Boolean, default=False)
    acknowledged_by = Column(PGUUID(as_uuid=True), ForeignKey("users.id"))
    acknowledged_at = Column(DateTime)
    resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Foreign Keys
    organization_id = Column(PGUUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)


# Pydantic Schemas
class BaseSchema(BaseModel):
    class Config:
        orm_mode = True
        use_enum_values = True
        allow_population_by_field_name = True


# Organization Schemas
class OrganizationBase(BaseSchema):
    name: str = Field(..., min_length=1, max_length=255)
    slug: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    settings: Dict[str, Any] = {}


class OrganizationCreate(OrganizationBase):
    pass


class OrganizationUpdate(BaseSchema):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    settings: Optional[Dict[str, Any]] = None
    status: Optional[OrganizationStatusEnum] = None


class OrganizationResponse(OrganizationBase):
    id: UUID
    status: OrganizationStatusEnum
    created_at: datetime
    updated_at: datetime


# User Schemas
class UserBase(BaseSchema):
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=100)
    full_name: str = Field(..., min_length=1, max_length=255)
    is_active: bool = True
    preferences: Dict[str, Any] = {}


class UserCreate(UserBase):
    password: str = Field(..., min_length=8, max_length=128)
    organization_id: UUID
    role_ids: List[UUID] = []


class UserUpdate(BaseSchema):
    email: Optional[EmailStr] = None
    username: Optional[str] = Field(None, min_length=3, max_length=100)
    full_name: Optional[str] = Field(None, min_length=1, max_length=255)
    is_active: Optional[bool] = None
    preferences: Optional[Dict[str, Any]] = None
    role_ids: Optional[List[UUID]] = None


class UserResponse(UserBase):
    id: UUID
    status: UserStatusEnum
    is_verified: bool
    last_login_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    organization: OrganizationResponse
    roles: List["RoleResponse"] = []


# Authentication Schemas
class LoginRequest(BaseSchema):
    username: str
    password: str


class LoginResponse(BaseSchema):
    access_token: str
    refresh_token: str
    token_type: str
    expires_in: int
    user: UserResponse


class TokenRefreshRequest(BaseSchema):
    refresh_token: str


class PasswordChangeRequest(BaseSchema):
    current_password: str
    new_password: str = Field(..., min_length=8, max_length=128)


# Role & Permission Schemas
class PermissionResponse(BaseSchema):
    id: UUID
    name: str
    description: Optional[str]
    resource: str
    action: str


class RoleBase(BaseSchema):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None


class RoleCreate(RoleBase):
    permission_ids: List[UUID] = []


class RoleResponse(RoleBase):
    id: UUID
    is_system_role: bool
    created_at: datetime
    permissions: List[PermissionResponse] = []


# Model Schemas
class ModelBase(BaseSchema):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    model_type: ModelTypeEnum
    provider: ModelProviderEnum
    version: str
    configuration: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}


class ModelCreate(ModelBase):
    pass


class ModelUpdate(BaseSchema):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    status: Optional[ModelStatusEnum] = None
    configuration: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class ModelResponse(ModelBase):
    id: UUID
    status: ModelStatusEnum
    performance_metrics: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    organization: OrganizationResponse


# Evaluation Schemas
class EvaluationBase(BaseSchema):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    evaluation_type: str
    configuration: Dict[str, Any]


class EvaluationCreate(EvaluationBase):
    model_id: UUID


class EvaluationResponse(EvaluationBase):
    id: UUID
    status: EvaluationStatusEnum
    results: Dict[str, Any]
    metrics: Dict[str, Any]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    created_at: datetime
    model: ModelResponse
    created_by_user: UserResponse


# Prediction Schemas
class PredictionRequest(BaseSchema):
    input_data: Dict[str, Any]
    model_id: UUID
    include_explanation: bool = False


class PredictionResponse(BaseSchema):
    id: UUID
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    confidence_score: Optional[float]
    explanation_data: Dict[str, Any]
    processing_time_ms: Optional[int]
    created_at: datetime


# Chat Schemas
class ChatMessage(BaseSchema):
    role: str = Field(..., regex="^(user|assistant|system)$")
    content: str = Field(..., min_length=1)
    metadata: Dict[str, Any] = {}
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ChatRequest(BaseSchema):
    message: str = Field(..., min_length=1)
    session_id: Optional[UUID] = None
    configuration: Dict[str, Any] = {}


class ChatResponse(BaseSchema):
    message: ChatMessage
    session_id: UUID
    sources: List[Dict[str, Any]] = []
    processing_time_ms: int


# Document Schemas
class DocumentUpload(BaseSchema):
    name: str = Field(..., min_length=1, max_length=255)
    file_type: str
    content: bytes


class DocumentResponse(BaseSchema):
    id: UUID
    name: str
    original_filename: str
    file_type: str
    file_size: int
    processed: bool
    processing_metadata: Dict[str, Any]
    created_at: datetime
    processed_at: Optional[datetime]


# Alert Schemas
class AlertBase(BaseSchema):
    title: str = Field(..., min_length=1, max_length=255)
    message: str = Field(..., min_length=1)
    severity: AlertSeverityEnum
    alert_type: str
    source: str
    metadata: Dict[str, Any] = {}


class AlertCreate(AlertBase):
    pass


class AlertResponse(AlertBase):
    id: UUID
    acknowledged: bool
    acknowledged_at: Optional[datetime]
    resolved: bool
    resolved_at: Optional[datetime]
    created_at: datetime


# Dashboard & Analytics Schemas
class MetricValue(BaseSchema):
    name: str
    value: Union[int, float, str]
    unit: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class DashboardMetrics(BaseSchema):
    total_models: int
    total_evaluations: int
    active_sessions: int
    alerts_count: int
    system_health: str
    recent_metrics: List[MetricValue] = []


class PerformanceMetrics(BaseSchema):
    accuracy: Optional[float] = Field(None, ge=0.0, le=1.0)
    precision: Optional[float] = Field(None, ge=0.0, le=1.0)
    recall: Optional[float] = Field(None, ge=0.0, le=1.0)
    f1_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    auc_roc: Optional[float] = Field(None, ge=0.0, le=1.0)
    mse: Optional[float] = Field(None, ge=0.0)
    rmse: Optional[float] = Field(None, ge=0.0)
    mae: Optional[float] = Field(None, ge=0.0)
    r2_score: Optional[float] = Field(None, ge=-1.0, le=1.0)


# Explanation Schemas
class ExplanationRequest(BaseSchema):
    model_id: UUID
    input_data: Dict[str, Any]
    explanation_type: str = Field(..., regex="^(shap|lime|grad_cam|integrated_gradients)$")
    options: Dict[str, Any] = {}


class ExplanationResponse(BaseSchema):
    explanation_type: str
    feature_importance: Dict[str, float]
    visualizations: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: datetime


# Fairness Analysis Schemas
class FairnessRequest(BaseSchema):
    model_id: UUID
    dataset: Dict[str, Any]
    protected_attributes: List[str]
    favorable_outcome: Union[str, int, float]


class FairnessResponse(BaseSchema):
    protected_attributes: List[str]
    metrics: Dict[str, float]
    bias_detected: bool
    recommendations: List[str]
    visualizations: Dict[str, Any]


# System Health Schema
class SystemHealthResponse(BaseSchema):
    status: str = Field(..., regex="^(healthy|degraded|unhealthy)$")
    services: Dict[str, str]
    metrics: Dict[str, Union[int, float, str]]
    last_checked: datetime


# Agent Evaluation Specific Enums and Models
class AgentConfigurationStatusEnum(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    DRAFT = "draft"
    ARCHIVED = "archived"


class EvaluationMetricTypeEnum(str, Enum):
    RELEVANCE = "relevance"
    GROUNDEDNESS = "groundedness"
    COHERENCE = "coherence"
    SIMILARITY = "similarity"
    FLUENCY = "fluency"
    ACCURACY = "accuracy"


class ChatSessionStatusEnum(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    ARCHIVED = "archived"


# Agent Evaluation SQLAlchemy Models
class AgentConfiguration(Base):
    __tablename__ = "agent_configurations"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    version = Column(String(50), nullable=False, default="1.0")
    status = Column(SQLEnum(AgentConfigurationStatusEnum), default=AgentConfigurationStatusEnum.DRAFT)
    
    # Model configuration
    model_name = Column(String(255), nullable=False)
    model_provider = Column(SQLEnum(ModelProviderEnum), nullable=False)
    temperature = Column(Float, default=0.7)
    max_tokens = Column(Integer, default=2000)
    top_p = Column(Float, default=1.0)
    frequency_penalty = Column(Float, default=0.0)
    presence_penalty = Column(Float, default=0.0)
    
    # RAG configuration
    retrieval_strategy = Column(String(50), default="similarity")
    retrieval_k = Column(Integer, default=5)
    chunk_size = Column(Integer, default=1000)
    chunk_overlap = Column(Integer, default=200)
    embedding_model = Column(String(255), default="all-MiniLM-L6-v2")
    
    # Prompts
    system_prompt = Column(Text, nullable=False)
    evaluation_prompt = Column(Text)
    
    # Additional settings
    configuration = Column(JSON, default=dict)
    metadata = Column(JSON, default=dict)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Foreign Keys
    organization_id = Column(PGUUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    created_by = Column(PGUUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Relationships
    evaluations = relationship("AgentEvaluation", back_populates="agent_configuration")
    chat_sessions = relationship("AgentChatSession", back_populates="agent_configuration")


class AgentEvaluation(Base):
    __tablename__ = "agent_evaluations"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    status = Column(SQLEnum(EvaluationStatusEnum), default=EvaluationStatusEnum.PENDING)
    
    # Dataset information
    total_questions = Column(Integer, default=0)
    completed_questions = Column(Integer, default=0)
    dataset_metadata = Column(JSON, default=dict)
    
    # Overall metrics
    overall_score = Column(Float)
    metrics_summary = Column(JSON, default=dict)
    
    # Execution details
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    execution_time_seconds = Column(Integer)
    total_tokens_used = Column(Integer, default=0)
    
    # Results
    results = Column(JSON, default=dict)
    error_message = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Foreign Keys
    organization_id = Column(PGUUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    agent_configuration_id = Column(PGUUID(as_uuid=True), ForeignKey("agent_configurations.id"), nullable=False)
    created_by = Column(PGUUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Relationships
    agent_configuration = relationship("AgentConfiguration", back_populates="evaluations")
    evaluation_results = relationship("EvaluationResult", back_populates="evaluation")


class EvaluationResult(Base):
    __tablename__ = "evaluation_results"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Question and answers
    question = Column(Text, nullable=False)
    expected_answer = Column(Text)
    actual_answer = Column(Text, nullable=False)
    context_used = Column(Text)
    
    # Individual metrics
    relevance_score = Column(Float)
    groundedness_score = Column(Float)
    coherence_score = Column(Float)
    similarity_score = Column(Float)
    fluency_score = Column(Float)
    accuracy_score = Column(Float)
    overall_score = Column(Float)
    
    # Execution details
    tokens_used = Column(Integer, default=0)
    response_time_ms = Column(Integer, default=0)
    sources_retrieved = Column(Integer, default=0)
    
    # Additional data
    metadata = Column(JSON, default=dict)
    sources = Column(JSON, default=list)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Foreign Keys
    evaluation_id = Column(PGUUID(as_uuid=True), ForeignKey("agent_evaluations.id"), nullable=False)
    
    # Relationships
    evaluation = relationship("AgentEvaluation", back_populates="evaluation_results")


class AgentChatSession(Base):
    __tablename__ = "agent_chat_sessions"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255))
    status = Column(SQLEnum(ChatSessionStatusEnum), default=ChatSessionStatusEnum.ACTIVE)
    
    # Session data
    messages = Column(JSON, default=list)
    session_metadata = Column(JSON, default=dict)
    
    # Usage statistics
    total_messages = Column(Integer, default=0)
    total_tokens_used = Column(Integer, default=0)
    avg_response_time_ms = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_activity_at = Column(DateTime, default=datetime.utcnow)
    
    # Foreign Keys
    organization_id = Column(PGUUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    user_id = Column(PGUUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    agent_configuration_id = Column(PGUUID(as_uuid=True), ForeignKey("agent_configurations.id"), nullable=False)
    
    # Relationships
    agent_configuration = relationship("AgentConfiguration", back_populates="chat_sessions")


class VectorDocument(Base):
    __tablename__ = "vector_documents"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(String(255), nullable=False, index=True)  # External document reference
    filename = Column(String(255), nullable=False)
    file_type = Column(String(50), nullable=False)
    
    # Content and processing
    content = Column(Text)
    chunk_count = Column(Integer, default=0)
    embedding_model = Column(String(255), nullable=False)
    
    # Processing metadata
    processing_status = Column(String(50), default="pending")
    processing_error = Column(Text)
    processed_at = Column(DateTime)
    
    # Document metadata
    metadata = Column(JSON, default=dict)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Foreign Keys
    organization_id = Column(PGUUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    uploaded_by = Column(PGUUID(as_uuid=True), ForeignKey("users.id"), nullable=False)


# Agent Evaluation Pydantic Schemas

# Agent Configuration Schemas
class AgentConfigurationBase(BaseSchema):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    version: str = Field(default="1.0", max_length=50)
    model_name: str = Field(..., min_length=1, max_length=255)
    model_provider: ModelProviderEnum
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2000, gt=0, le=100000)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    retrieval_strategy: str = Field(default="similarity", max_length=50)
    retrieval_k: int = Field(default=5, gt=0, le=50)
    chunk_size: int = Field(default=1000, gt=0, le=10000)
    chunk_overlap: int = Field(default=200, ge=0, le=2000)
    embedding_model: str = Field(default="all-MiniLM-L6-v2", max_length=255)
    system_prompt: str = Field(..., min_length=1)
    evaluation_prompt: Optional[str] = None
    configuration: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}


class AgentConfigurationCreate(AgentConfigurationBase):
    pass


class AgentConfigurationUpdate(BaseSchema):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    status: Optional[AgentConfigurationStatusEnum] = None
    model_name: Optional[str] = Field(None, min_length=1, max_length=255)
    model_provider: Optional[ModelProviderEnum] = None
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, gt=0, le=100000)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    frequency_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0)
    retrieval_strategy: Optional[str] = Field(None, max_length=50)
    retrieval_k: Optional[int] = Field(None, gt=0, le=50)
    chunk_size: Optional[int] = Field(None, gt=0, le=10000)
    chunk_overlap: Optional[int] = Field(None, ge=0, le=2000)
    embedding_model: Optional[str] = Field(None, max_length=255)
    system_prompt: Optional[str] = Field(None, min_length=1)
    evaluation_prompt: Optional[str] = None
    configuration: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class AgentConfigurationResponse(AgentConfigurationBase):
    id: UUID
    status: AgentConfigurationStatusEnum
    created_at: datetime
    updated_at: datetime


# Agent Evaluation Schemas
class EvaluationDataItem(BaseSchema):
    question: str = Field(..., min_length=1)
    expected_answer: Optional[str] = None
    context: Optional[str] = None
    metadata: Dict[str, Any] = {}
    chat_history: Optional[List[Dict[str, Any]]] = None


class AgentEvaluationCreate(BaseSchema):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    agent_configuration_id: UUID
    dataset: List[EvaluationDataItem] = Field(..., min_items=1)
    evaluation_config: Dict[str, Any] = {}


class EvaluationMetrics(BaseSchema):
    relevance: Optional[float] = Field(None, ge=0.0, le=5.0)
    groundedness: Optional[float] = Field(None, ge=0.0, le=5.0)
    coherence: Optional[float] = Field(None, ge=0.0, le=5.0)
    similarity: Optional[float] = Field(None, ge=0.0, le=1.0)
    fluency: Optional[float] = Field(None, ge=0.0, le=5.0)
    accuracy: Optional[float] = Field(None, ge=0.0, le=5.0)
    overall: Optional[float] = Field(None, ge=0.0, le=5.0)

    def average(self) -> float:
        scores = [v for v in [self.relevance, self.groundedness, self.coherence, 
                            self.similarity, self.fluency, self.accuracy] if v is not None]
        return sum(scores) / len(scores) if scores else 0.0


class EvaluationResultResponse(BaseSchema):
    id: UUID
    question: str
    expected_answer: Optional[str]
    actual_answer: str
    context_used: Optional[str]
    metrics: EvaluationMetrics
    tokens_used: int
    response_time_ms: int
    sources_retrieved: int
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    created_at: datetime


class AgentEvaluationResponse(BaseSchema):
    id: UUID
    name: str
    description: Optional[str]
    status: EvaluationStatusEnum
    total_questions: int
    completed_questions: int
    overall_score: Optional[float]
    metrics_summary: Dict[str, Any]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    execution_time_seconds: Optional[int]
    total_tokens_used: int
    results: Dict[str, Any]
    error_message: Optional[str]
    created_at: datetime
    updated_at: datetime
    agent_configuration: AgentConfigurationResponse
    evaluation_results: List[EvaluationResultResponse] = []


# Chat Schemas for Agent
class AgentChatMessage(BaseSchema):
    role: str = Field(..., regex="^(user|assistant|system)$")
    content: str = Field(..., min_length=1)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = {}
    sources: List[Dict[str, Any]] = []
    tokens_used: int = 0
    response_time_ms: int = 0


class AgentChatRequest(BaseSchema):
    message: str = Field(..., min_length=1)
    agent_configuration_id: UUID
    session_id: Optional[UUID] = None
    include_sources: bool = True
    chat_history: List[Dict[str, Any]] = []


class AgentChatResponse(BaseSchema):
    answer: str
    session_id: UUID
    agent_configuration_id: UUID
    context: Optional[str] = None
    sources: List[Dict[str, Any]] = []
    tokens_used: int
    response_time_ms: int
    model_name: str
    message: AgentChatMessage


class AgentChatSessionResponse(BaseSchema):
    id: UUID
    name: Optional[str]
    status: ChatSessionStatusEnum
    messages: List[AgentChatMessage]
    total_messages: int
    total_tokens_used: int
    avg_response_time_ms: int
    created_at: datetime
    updated_at: datetime
    last_activity_at: datetime
    agent_configuration: AgentConfigurationResponse


# Document Management Schemas
class VectorDocumentResponse(BaseSchema):
    id: UUID
    document_id: str
    filename: str
    file_type: str
    chunk_count: int
    embedding_model: str
    processing_status: str
    processing_error: Optional[str]
    processed_at: Optional[datetime]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


class DocumentUploadResponse(BaseSchema):
    document: VectorDocumentResponse
    chunks_created: int
    processing_time_ms: int
    success: bool
    message: str


class DocumentSearchRequest(BaseSchema):
    query: str = Field(..., min_length=1)
    n_results: int = Field(default=5, gt=0, le=50)
    filter_metadata: Optional[Dict[str, Any]] = None


class DocumentSearchResult(BaseSchema):
    document_chunk: str
    metadata: Dict[str, Any]
    similarity_score: float
    chunk_index: int
    source_document: str


class DocumentSearchResponse(BaseSchema):
    query: str
    results: List[DocumentSearchResult]
    total_results: int
    search_time_ms: int


# Model Comparison Schemas
class ModelComparisonRequest(BaseSchema):
    agent_configuration_ids: List[UUID] = Field(..., min_items=2)
    dataset: List[EvaluationDataItem] = Field(..., min_items=1)
    metrics_to_compare: List[EvaluationMetricTypeEnum] = []


class ModelComparisonResult(BaseSchema):
    agent_configuration_id: UUID
    agent_name: str
    model_name: str
    metrics: EvaluationMetrics
    performance_summary: Dict[str, Any]
    execution_stats: Dict[str, Any]


class ModelComparisonResponse(BaseSchema):
    comparison_id: UUID
    results: List[ModelComparisonResult]
    statistical_analysis: Dict[str, Any]
    recommendations: List[str]
    created_at: datetime


# Task Status Schemas
class TaskStatus(BaseSchema):
    task_id: UUID
    status: str = Field(..., regex="^(pending|running|completed|failed|cancelled)$")
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    total_items: int = 0
    completed_items: int = 0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    estimated_completion: Optional[datetime] = None


# Model Explainability Enums and Database Models
class ExplanationMethodEnum(str, Enum):
    SHAP_TREE = "shap_tree"
    SHAP_KERNEL = "shap_kernel"
    SHAP_DEEP = "shap_deep"
    SHAP_LINEAR = "shap_linear"
    LIME_TABULAR = "lime_tabular"
    LIME_TEXT = "lime_text"
    LIME_IMAGE = "lime_image"
    INTEGRATED_GRADIENTS = "integrated_gradients"
    GRADIENT_SHAP = "gradient_shap"
    DEEP_LIFT = "deep_lift"
    ANCHORS = "anchors"
    COUNTERFACTUALS = "counterfactuals"
    PERMUTATION_IMPORTANCE = "permutation_importance"


class ExplanationStatusEnum(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DriftStatusEnum(str, Enum):
    NO_DRIFT = "no_drift"
    DRIFT_DETECTED = "drift_detected"
    SIGNIFICANT_DRIFT = "significant_drift"


class FairnessLevelEnum(str, Enum):
    FAIR = "fair"
    CONCERNING = "concerning"
    UNFAIR = "unfair"


class ModelExplanation(Base):
    __tablename__ = "model_explanations"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    explanation_type = Column(String(50), nullable=False)  # "instance", "global", "counterfactual"
    method = Column(SQLEnum(ExplanationMethodEnum), nullable=False)
    status = Column(SQLEnum(ExplanationStatusEnum), default=ExplanationStatusEnum.PENDING)
    
    # Input data
    input_data = Column(JSON, nullable=False)
    feature_names = Column(JSON, default=list)
    parameters = Column(JSON, default=dict)
    
    # Results
    explanation_data = Column(JSON, default=dict)
    feature_importance = Column(JSON, default=dict)
    visualizations = Column(JSON, default=dict)
    
    # Metadata
    processing_time_ms = Column(Integer)
    error_message = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    
    # Foreign Keys
    organization_id = Column(PGUUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    model_id = Column(PGUUID(as_uuid=True), ForeignKey("models.id"), nullable=False)
    created_by = Column(PGUUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Relationships
    model = relationship("Model", foreign_keys=[model_id])


class FairnessAnalysis(Base):
    __tablename__ = "fairness_analyses"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    
    # Dataset information
    target_column = Column(String(255), nullable=False)
    sensitive_attributes = Column(JSON, nullable=False)  # List of sensitive attribute names
    dataset_metadata = Column(JSON, default=dict)
    
    # Analysis configuration
    model_type = Column(String(50), nullable=False)  # "classification", "regression"
    test_size = Column(Float, default=0.2)
    random_state = Column(Integer, default=42)
    
    # Results
    baseline_metrics = Column(JSON, default=dict)
    fairness_metrics = Column(JSON, default=dict)
    group_metrics = Column(JSON, default=dict)
    fairness_level = Column(SQLEnum(FairnessLevelEnum))
    
    # Bias mitigation results
    mitigation_applied = Column(Boolean, default=False)
    mitigation_method = Column(String(100))
    mitigation_constraint = Column(String(100))
    mitigation_results = Column(JSON, default=dict)
    
    # Report data
    html_report = Column(Text)
    recommendations = Column(JSON, default=list)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    
    # Foreign Keys
    organization_id = Column(PGUUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    model_id = Column(PGUUID(as_uuid=True), ForeignKey("models.id"), nullable=False)
    created_by = Column(PGUUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Relationships
    model = relationship("Model", foreign_keys=[model_id])


class DataDriftAnalysis(Base):
    __tablename__ = "data_drift_analyses"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    
    # Dataset information
    reference_period = Column(JSON, nullable=False)  # {"start": "date", "end": "date"}
    current_period = Column(JSON, nullable=False)
    columns_analyzed = Column(JSON, nullable=False)  # List of column names
    
    # Analysis results
    overall_drift_detected = Column(Boolean, default=False)
    drift_status = Column(SQLEnum(DriftStatusEnum), default=DriftStatusEnum.NO_DRIFT)
    total_columns = Column(Integer, default=0)
    drifted_columns = Column(Integer, default=0)
    drift_percentage = Column(Float, default=0.0)
    
    # Column-specific results
    column_drift_results = Column(JSON, default=dict)
    drift_scores = Column(JSON, default=dict)
    statistical_tests = Column(JSON, default=dict)
    
    # Model drift (if applicable)
    model_drift_detected = Column(Boolean, default=False)
    prediction_drift_score = Column(Float)
    performance_degradation = Column(JSON, default=dict)
    
    # Report data
    html_report = Column(Text)
    visualization_data = Column(JSON, default=dict)
    summary_statistics = Column(JSON, default=dict)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    
    # Foreign Keys
    organization_id = Column(PGUUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    model_id = Column(PGUUID(as_uuid=True), ForeignKey("models.id"), nullable=True)  # Optional for data-only drift
    created_by = Column(PGUUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Relationships
    model = relationship("Model", foreign_keys=[model_id])


class ModelMonitoringSession(Base):
    __tablename__ = "model_monitoring_sessions"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    is_active = Column(Boolean, default=True)
    
    # Monitoring configuration
    business_metrics = Column(JSON, default=list)  # List of business metrics to track
    performance_thresholds = Column(JSON, default=dict)
    alert_settings = Column(JSON, default=dict)
    monitoring_frequency = Column(String(50), default="hourly")  # hourly, daily, weekly
    
    # Current status
    last_check_at = Column(DateTime)
    health_status = Column(String(50), default="healthy")  # healthy, warning, critical
    active_alerts_count = Column(Integer, default=0)
    
    # Historical tracking
    prediction_count = Column(Integer, default=0)
    total_processing_time_ms = Column(Integer, default=0)
    avg_confidence_score = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Foreign Keys
    organization_id = Column(PGUUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    model_id = Column(PGUUID(as_uuid=True), ForeignKey("models.id"), nullable=False)
    created_by = Column(PGUUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Relationships
    model = relationship("Model", foreign_keys=[model_id])
    monitoring_metrics = relationship("ModelMonitoringMetric", back_populates="monitoring_session")


class ModelMonitoringMetric(Base):
    __tablename__ = "model_monitoring_metrics"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    metric_name = Column(String(100), nullable=False)
    metric_type = Column(String(50), nullable=False)  # "performance", "business", "drift", "explanation"
    metric_value = Column(Float, nullable=False)
    
    # Additional context
    batch_size = Column(Integer)
    metadata = Column(JSON, default=dict)
    
    # Threshold information
    threshold_value = Column(Float)
    threshold_breached = Column(Boolean, default=False)
    severity_level = Column(String(20))  # "low", "medium", "high", "critical"
    
    recorded_at = Column(DateTime, default=datetime.utcnow)
    
    # Foreign Keys
    monitoring_session_id = Column(PGUUID(as_uuid=True), ForeignKey("model_monitoring_sessions.id"), nullable=False)
    
    # Relationships
    monitoring_session = relationship("ModelMonitoringSession", back_populates="monitoring_metrics")


class ABTest(Base):
    __tablename__ = "ab_tests"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    
    # Test configuration
    champion_model_id = Column(PGUUID(as_uuid=True), ForeignKey("models.id"), nullable=False)
    challenger_model_id = Column(PGUUID(as_uuid=True), ForeignKey("models.id"), nullable=False)
    traffic_split = Column(Float, default=0.5)  # Percentage of traffic to challenger
    success_metrics = Column(JSON, default=list)
    
    # Test period
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    is_active = Column(Boolean, default=True)
    
    # Results
    champion_results = Column(JSON, default=dict)
    challenger_results = Column(JSON, default=dict)
    statistical_significance = Column(JSON, default=dict)
    winner_model_id = Column(PGUUID(as_uuid=True), ForeignKey("models.id"))
    
    # Traffic tracking
    champion_traffic_count = Column(Integer, default=0)
    challenger_traffic_count = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    
    # Foreign Keys
    organization_id = Column(PGUUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    created_by = Column(PGUUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Relationships
    champion_model = relationship("Model", foreign_keys=[champion_model_id])
    challenger_model = relationship("Model", foreign_keys=[challenger_model_id])
    winner_model = relationship("Model", foreign_keys=[winner_model_id])


# Monitoring and Analytics Schemas
class AgentPerformanceMetrics(BaseSchema):
    agent_configuration_id: UUID
    time_period: str
    total_evaluations: int
    avg_score: float
    avg_response_time_ms: int
    total_tokens_used: int
    success_rate: float
    error_rate: float
    metrics_breakdown: Dict[str, float]


class SystemUsageMetrics(BaseSchema):
    total_agent_configurations: int
    total_evaluations: int
    total_chat_sessions: int
    total_documents: int
    active_users: int
    total_tokens_consumed: int
    avg_evaluation_score: float
    system_uptime: str
    timestamp: datetime


# Model Explainability Pydantic Schemas

# Explanation Request/Response Schemas
class ModelExplanationRequest(BaseSchema):
    model_id: UUID
    explanation_type: str = Field(..., regex="^(instance|global|counterfactual)$")
    method: ExplanationMethodEnum
    input_data: Dict[str, Any]
    feature_names: List[str] = []
    parameters: Dict[str, Any] = {}


class ModelExplanationResponse(BaseSchema):
    id: UUID
    explanation_type: str
    method: ExplanationMethodEnum
    status: ExplanationStatusEnum
    explanation_data: Dict[str, Any]
    feature_importance: Dict[str, float]
    visualizations: Dict[str, Any]
    processing_time_ms: Optional[int]
    error_message: Optional[str]
    created_at: datetime
    completed_at: Optional[datetime]


class InstanceExplanationRequest(BaseSchema):
    instance_data: Dict[str, Any]
    method: ExplanationMethodEnum = ExplanationMethodEnum.SHAP_TREE
    parameters: Dict[str, Any] = {}


class GlobalExplanationRequest(BaseSchema):
    method: ExplanationMethodEnum = ExplanationMethodEnum.SHAP_TREE
    training_data: Optional[List[Dict[str, Any]]] = None
    parameters: Dict[str, Any] = {}


class CounterfactualRequest(BaseSchema):
    instance: Dict[str, Any]
    desired_outcome: Optional[Union[float, str]] = None
    max_features_to_change: int = Field(default=5, ge=1, le=10)
    feature_ranges: Optional[Dict[str, Tuple[float, float]]] = None


class WhatIfAnalysisRequest(BaseSchema):
    base_instance: Dict[str, Any]
    scenarios: List[Dict[str, Any]]  # List of feature modifications
    include_explanations: bool = False


class WhatIfAnalysisResponse(BaseSchema):
    base_prediction: float
    scenario_results: List[Dict[str, Any]]
    feature_impact_analysis: Dict[str, Any]


# Fairness Analysis Schemas
class FairnessAnalysisRequest(BaseSchema):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    model_id: UUID
    dataset: List[Dict[str, Any]]  # Training/test data
    target_column: str
    sensitive_attributes: List[str]
    model_type: str = Field(..., regex="^(classification|regression)$")
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)
    random_state: int = 42


class BiaseMitigationRequest(BaseSchema):
    fairness_analysis_id: UUID
    method: str = Field(..., regex="^(exponentiated_gradient|grid_search|threshold_optimizer)$")
    constraint: str = Field(..., regex="^(demographic_parity|equalized_odds)$")


class FairnessAnalysisResponse(BaseSchema):
    id: UUID
    name: str
    description: Optional[str]
    target_column: str
    sensitive_attributes: List[str]
    model_type: str
    baseline_metrics: Dict[str, Any]
    fairness_metrics: Dict[str, Any]
    group_metrics: Dict[str, Any]
    fairness_level: FairnessLevelEnum
    mitigation_applied: bool
    mitigation_results: Dict[str, Any]
    html_report: Optional[str]
    recommendations: List[str]
    created_at: datetime
    completed_at: Optional[datetime]


# Data Drift Analysis Schemas
class DataDriftAnalysisRequest(BaseSchema):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    model_id: Optional[UUID] = None  # Optional for data-only drift
    reference_data: List[Dict[str, Any]]
    current_data: List[Dict[str, Any]]
    columns_to_analyze: Optional[List[str]] = None  # If None, analyze all
    drift_threshold: float = Field(default=0.1, ge=0.01, le=1.0)


class DriftAnalysisResponse(BaseSchema):
    id: UUID
    name: str
    description: Optional[str]
    drift_status: DriftStatusEnum
    overall_drift_detected: bool
    total_columns: int
    drifted_columns: int
    drift_percentage: float
    column_drift_results: Dict[str, Any]
    drift_scores: Dict[str, float]
    model_drift_detected: bool
    prediction_drift_score: Optional[float]
    performance_degradation: Dict[str, Any]
    html_report: Optional[str]
    visualization_data: Dict[str, Any]
    summary_statistics: Dict[str, Any]
    created_at: datetime
    completed_at: Optional[datetime]


# Model Monitoring Schemas
class ModelMonitoringSessionRequest(BaseSchema):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    model_id: UUID
    business_metrics: List[str] = []
    performance_thresholds: Dict[str, float] = {}
    alert_settings: Dict[str, Any] = {}
    monitoring_frequency: str = Field(default="hourly", regex="^(hourly|daily|weekly|realtime)$")


class ModelMonitoringSessionResponse(BaseSchema):
    id: UUID
    name: str
    description: Optional[str]
    is_active: bool
    business_metrics: List[str]
    performance_thresholds: Dict[str, float]
    monitoring_frequency: str
    last_check_at: Optional[datetime]
    health_status: str
    active_alerts_count: int
    prediction_count: int
    avg_confidence_score: Optional[float]
    created_at: datetime
    updated_at: datetime


class PredictionBatchRequest(BaseSchema):
    monitoring_session_id: UUID
    predictions: List[Dict[str, Any]]
    true_labels: Optional[List[Union[float, str]]] = None
    business_outcomes: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = {}


class ModelHealthDashboardResponse(BaseSchema):
    monitoring_session: ModelMonitoringSessionResponse
    performance_trends: Dict[str, Any]
    business_impact: Dict[str, Any]
    data_quality_status: Dict[str, Any]
    recent_alerts: List[Dict[str, Any]]
    feature_importance_trends: Dict[str, Any]
    recommendations: List[str]
    charts: Dict[str, Any]


# A/B Testing Schemas
class ABTestRequest(BaseSchema):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    champion_model_id: UUID
    challenger_model_id: UUID
    traffic_split: float = Field(default=0.5, ge=0.1, le=0.9)
    success_metrics: List[str] = []
    duration_days: int = Field(default=14, ge=1, le=90)


class ABTestResponse(BaseSchema):
    id: UUID
    name: str
    description: Optional[str]
    champion_model_id: UUID
    challenger_model_id: UUID
    traffic_split: float
    success_metrics: List[str]
    start_date: datetime
    end_date: datetime
    is_active: bool
    champion_results: Dict[str, Any]
    challenger_results: Dict[str, Any]
    statistical_significance: Dict[str, Any]
    winner_model_id: Optional[UUID]
    champion_traffic_count: int
    challenger_traffic_count: int
    created_at: datetime
    completed_at: Optional[datetime]


# Feature Importance Analysis Schemas
class FeatureImportanceAnalysisResponse(BaseSchema):
    feature_importance: Dict[str, float]
    feature_rankings: List[Dict[str, Any]]
    importance_method: str
    confidence_intervals: Optional[Dict[str, List[float]]]
    feature_interactions: Optional[Dict[str, Any]]
    visualization_data: Dict[str, Any]
    statistical_summary: Dict[str, Any]


class DecisionTreeExtractionResponse(BaseSchema):
    tree_structure: Dict[str, Any]
    tree_rules: List[str]
    max_depth: int
    n_leaves: int
    n_nodes: int
    is_surrogate: bool
    surrogate_accuracy: Optional[float]
    visualization_data: Dict[str, Any]


# Advanced Analysis Schemas
class SensitivityAnalysisResponse(BaseSchema):
    feature_sensitivities: Dict[str, float]
    most_sensitive_feature: str
    least_sensitive_feature: str
    sensitivity_rankings: List[Dict[str, Any]]
    perturbation_analysis: Dict[str, Any]


class PartialDependenceResponse(BaseSchema):
    feature_name: str
    feature_values: List[float]
    partial_dependence_values: List[float]
    ice_curves: Optional[List[List[float]]]
    feature_interaction_effects: Optional[Dict[str, Any]]


# Monitoring Alert Schemas
class MonitoringAlert(BaseSchema):
    id: UUID
    title: str
    message: str
    severity: AlertSeverityEnum
    metric_name: str
    current_value: float
    threshold_value: float
    monitoring_session_id: UUID
    model_id: UUID
    acknowledged: bool = False
    resolved: bool = False
    created_at: datetime


# Batch Processing Schemas
class BatchExplanationRequest(BaseSchema):
    model_id: UUID
    instances: List[Dict[str, Any]]
    method: ExplanationMethodEnum
    parameters: Dict[str, Any] = {}
    batch_size: int = Field(default=100, ge=1, le=1000)


class BatchExplanationResponse(BaseSchema):
    batch_id: UUID
    status: str
    total_instances: int
    completed_instances: int
    explanations: List[ModelExplanationResponse]
    processing_time_ms: int
    created_at: datetime
    completed_at: Optional[datetime]


# Advanced Services Database Models

# RAG Evaluation Models
class RAGEvaluationStatusEnum(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class RAGSystem(Base):
    __tablename__ = "rag_systems"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    system_id = Column(String(255), nullable=False, unique=True, index=True)
    system_name = Column(String(255), nullable=False)
    system_type = Column(String(100), default="retrieval_augmented")
    description = Column(Text)
    configuration = Column(JSON, default=dict)
    metadata = Column(JSON, default=dict)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Foreign Keys
    organization_id = Column(PGUUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    
    # Relationships
    evaluations = relationship("RAGEvaluation", back_populates="rag_system")


class RAGEvaluation(Base):
    __tablename__ = "rag_evaluations"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    evaluation_name = Column(String(255), nullable=False)
    status = Column(SQLEnum(RAGEvaluationStatusEnum), default=RAGEvaluationStatusEnum.PENDING)
    
    # Evaluation data
    queries_count = Column(Integer, default=0)
    completed_queries = Column(Integer, default=0)
    
    # Metrics
    retrieval_precision = Column(Float)
    retrieval_recall = Column(Float)
    retrieval_f1 = Column(Float)
    generation_faithfulness = Column(Float)
    generation_relevancy = Column(Float)
    generation_coherence = Column(Float)
    overall_score = Column(Float)
    
    # Results and metadata
    evaluation_results = Column(JSON, default=dict)
    metrics_summary = Column(JSON, default=dict)
    configuration = Column(JSON, default=dict)
    
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Foreign Keys
    organization_id = Column(PGUUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    rag_system_id = Column(PGUUID(as_uuid=True), ForeignKey("rag_systems.id"), nullable=False)
    created_by = Column(PGUUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Relationships
    rag_system = relationship("RAGSystem", back_populates="evaluations")


# LLM Evaluation Models
class LLMModel(Base):
    __tablename__ = "llm_models"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(String(255), nullable=False, unique=True, index=True)
    model_name = Column(String(255), nullable=False)
    model_type = Column(String(100), default="text_generation")
    provider = Column(String(100), nullable=False)
    description = Column(Text)
    configuration = Column(JSON, default=dict)
    metadata = Column(JSON, default=dict)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Foreign Keys
    organization_id = Column(PGUUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    
    # Relationships
    evaluations = relationship("LLMEvaluation", back_populates="llm_model")


class LLMEvaluation(Base):
    __tablename__ = "llm_evaluations"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    evaluation_name = Column(String(255), nullable=False)
    status = Column(SQLEnum(EvaluationStatusEnum), default=EvaluationStatusEnum.PENDING)
    evaluation_framework = Column(String(100), nullable=False)
    task_type = Column(String(100), nullable=False)
    
    # Evaluation data
    inputs_count = Column(Integer, default=0)
    completed_evaluations = Column(Integer, default=0)
    
    # Content quality metrics
    factual_accuracy = Column(Float)
    completeness = Column(Float)
    relevancy = Column(Float)
    logical_consistency = Column(Float)
    
    # Language quality metrics
    fluency = Column(Float)
    grammar = Column(Float)
    clarity = Column(Float)
    conciseness = Column(Float)
    
    # Semantic metrics
    bert_score = Column(Float)
    semantic_coherence = Column(Float)
    
    # Safety metrics
    toxicity_score = Column(Float)
    bias_score = Column(Float)
    harmful_content_score = Column(Float)
    
    # Overall metrics
    overall_score = Column(Float)
    
    # Results and metadata
    evaluation_results = Column(JSON, default=dict)
    metrics_breakdown = Column(JSON, default=dict)
    configuration = Column(JSON, default=dict)
    
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Foreign Keys
    organization_id = Column(PGUUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    llm_model_id = Column(PGUUID(as_uuid=True), ForeignKey("llm_models.id"), nullable=False)
    created_by = Column(PGUUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Relationships
    llm_model = relationship("LLMModel", back_populates="evaluations")


# Agent Evaluation Models
class AgentType(str, Enum):
    CONVERSATIONAL = "conversational"
    RAG = "rag" 
    TOOL_USING = "tool_using"
    CODE_GENERATION = "code_generation"
    MULTIMODAL = "multimodal"


class AgentEvaluationSystem(Base):
    __tablename__ = "agent_evaluation_systems"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id = Column(String(255), nullable=False, unique=True, index=True)
    agent_name = Column(String(255), nullable=False)
    agent_type = Column(SQLEnum(AgentType), nullable=False)
    description = Column(Text)
    configuration = Column(JSON, default=dict)
    metadata = Column(JSON, default=dict)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Foreign Keys
    organization_id = Column(PGUUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    
    # Relationships
    evaluations = relationship("AgentEvaluationResult", back_populates="agent_system")


class AgentEvaluationResult(Base):
    __tablename__ = "agent_evaluation_results"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    evaluation_name = Column(String(255), nullable=False)
    status = Column(SQLEnum(EvaluationStatusEnum), default=EvaluationStatusEnum.PENDING)
    
    # Task performance metrics
    task_completion_rate = Column(Float)
    response_accuracy = Column(Float)
    response_relevance = Column(Float)
    response_helpfulness = Column(Float)
    
    # Conversation quality metrics
    conversation_coherence = Column(Float)
    context_awareness = Column(Float)
    response_consistency = Column(Float)
    
    # Technical metrics
    avg_response_time = Column(Float)
    error_rate = Column(Float)
    resource_efficiency = Column(Float)
    
    # Overall metrics
    overall_performance_score = Column(Float)
    user_satisfaction_score = Column(Float)
    
    # Detailed results
    evaluation_results = Column(JSON, default=dict)
    performance_breakdown = Column(JSON, default=dict)
    configuration = Column(JSON, default=dict)
    
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Foreign Keys
    organization_id = Column(PGUUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    agent_system_id = Column(PGUUID(as_uuid=True), ForeignKey("agent_evaluation_systems.id"), nullable=False)
    created_by = Column(PGUUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Relationships
    agent_system = relationship("AgentEvaluationSystem", back_populates="evaluations")


# Model Statistics Models
class ModelStatisticsSession(Base):
    __tablename__ = "model_statistics_sessions"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_name = Column(String(255), nullable=False)
    model_type = Column(String(50), nullable=False)  # "classification" or "regression"
    description = Column(Text)
    
    # Dataset information
    test_samples_count = Column(Integer)
    training_samples_count = Column(Integer)
    feature_count = Column(Integer)
    class_count = Column(Integer)  # For classification
    
    # Classification metrics
    accuracy = Column(Float)
    precision_macro = Column(Float)
    precision_micro = Column(Float)
    precision_weighted = Column(Float)
    recall_macro = Column(Float)
    recall_micro = Column(Float)
    recall_weighted = Column(Float)
    f1_macro = Column(Float)
    f1_micro = Column(Float)
    f1_weighted = Column(Float)
    roc_auc_macro = Column(Float)
    matthews_corrcoef = Column(Float)
    cohen_kappa = Column(Float)
    
    # Regression metrics
    mae = Column(Float)
    mse = Column(Float)
    rmse = Column(Float)
    r2_score = Column(Float)
    adjusted_r2 = Column(Float)
    explained_variance = Column(Float)
    
    # Statistical analysis
    cross_validation_results = Column(JSON, default=dict)
    learning_curves_data = Column(JSON, default=dict)
    statistical_tests = Column(JSON, default=dict)
    
    # Metadata
    detailed_results = Column(JSON, default=dict)
    configuration = Column(JSON, default=dict)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    
    # Foreign Keys
    organization_id = Column(PGUUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    model_id = Column(PGUUID(as_uuid=True), ForeignKey("models.id"), nullable=False)
    created_by = Column(PGUUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Relationships
    model = relationship("Model", foreign_keys=[model_id])


# What-If Analysis Models
class WhatIfAnalysisSession(Base):
    __tablename__ = "what_if_analysis_sessions"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_name = Column(String(255), nullable=False)
    description = Column(Text)
    analysis_type = Column(String(100), nullable=False)  # "scenario", "optimization", "sensitivity"
    
    # Base instance and scenarios
    base_instance = Column(JSON, nullable=False)
    scenarios_data = Column(JSON, default=list)
    scenario_results = Column(JSON, default=list)
    
    # Analysis results
    feature_impact_analysis = Column(JSON, default=dict)
    sensitivity_analysis = Column(JSON, default=dict)
    optimization_results = Column(JSON, default=dict)
    
    # Decision tree extraction
    surrogate_tree_accuracy = Column(Float)
    decision_rules = Column(JSON, default=list)
    tree_visualization_data = Column(JSON, default=dict)
    
    # Metadata
    configuration = Column(JSON, default=dict)
    processing_time_ms = Column(Integer)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    
    # Foreign Keys
    organization_id = Column(PGUUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    model_id = Column(PGUUID(as_uuid=True), ForeignKey("models.id"), nullable=False)
    created_by = Column(PGUUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Relationships
    model = relationship("Model", foreign_keys=[model_id])


# Advanced Explainability Models
class AdvancedExplanationSession(Base):
    __tablename__ = "advanced_explanation_sessions"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_name = Column(String(255), nullable=False)
    description = Column(Text)
    explanation_method = Column(String(100), nullable=False)  # "anchor", "ale", "prototype", "ice", etc.
    
    # Input data
    input_data = Column(JSON, nullable=False)
    feature_names = Column(JSON, default=list)
    parameters = Column(JSON, default=dict)
    
    # Results based on method
    explanation_results = Column(JSON, default=dict)
    
    # Anchor explanations
    anchor_rules = Column(JSON, default=list)
    anchor_precision = Column(Float)
    anchor_coverage = Column(Float)
    
    # ALE plot data
    ale_plot_data = Column(JSON, default=dict)
    
    # Prototype analysis
    prototype_data = Column(JSON, default=list)
    similarity_scores = Column(JSON, default=list)
    
    # ICE plot data
    ice_lines = Column(JSON, default=list)
    partial_dependence = Column(JSON, default=list)
    
    # Counterfactual data
    counterfactuals = Column(JSON, default=list)
    counterfactual_quality_metrics = Column(JSON, default=dict)
    
    # Permutation importance
    permutation_importance_scores = Column(JSON, default=dict)
    eli5_analysis = Column(JSON, default=dict)
    
    # Metadata
    processing_time_ms = Column(Integer)
    configuration = Column(JSON, default=dict)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    
    # Foreign Keys
    organization_id = Column(PGUUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    model_id = Column(PGUUID(as_uuid=True), ForeignKey("models.id"), nullable=False)
    created_by = Column(PGUUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Relationships
    model = relationship("Model", foreign_keys=[model_id])


# WebSocket Connection Tracking Models
class WebSocketConnection(Base):
    __tablename__ = "websocket_connections"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    connection_id = Column(String(255), nullable=False, unique=True, index=True)
    user_agent = Column(String(500))
    ip_address = Column(String(45))
    
    # Connection details
    subscriptions = Column(JSON, default=list)
    is_active = Column(Boolean, default=True)
    last_activity_at = Column(DateTime, default=datetime.utcnow)
    
    # Statistics
    messages_sent = Column(Integer, default=0)
    messages_received = Column(Integer, default=0)
    connection_duration_seconds = Column(Integer, default=0)
    
    connected_at = Column(DateTime, default=datetime.utcnow)
    disconnected_at = Column(DateTime)
    
    # Foreign Keys
    user_id = Column(PGUUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    organization_id = Column(PGUUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    
    # Relationships
    user = relationship("User", foreign_keys=[user_id])
    events = relationship("WebSocketEvent", back_populates="connection")


class WebSocketEventTypeEnum(str, Enum):
    MODEL_PERFORMANCE = "model_performance"
    SYSTEM_HEALTH = "system_health"
    EVALUATION_PROGRESS = "evaluation_progress"
    ALERT_NOTIFICATION = "alert_notification"
    DATA_DRIFT_DETECTION = "data_drift_detection"
    FAIRNESS_VIOLATION = "fairness_violation"
    ANOMALY_DETECTION = "anomaly_detection"
    BATCH_JOB_STATUS = "batch_job_status"
    RESOURCE_USAGE = "resource_usage"
    ERROR_NOTIFICATION = "error_notification"


class WebSocketEvent(Base):
    __tablename__ = "websocket_events"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_id = Column(String(255), nullable=False, index=True)
    event_type = Column(SQLEnum(WebSocketEventTypeEnum), nullable=False)
    severity = Column(String(20), default="info")  # info, warning, error, critical
    
    # Event data
    event_data = Column(JSON, default=dict)
    message = Column(Text)
    
    # Targeting
    broadcast_to_all = Column(Boolean, default=False)
    target_user_ids = Column(JSON, default=list)
    target_model_ids = Column(JSON, default=list)
    
    # Delivery tracking
    total_recipients = Column(Integer, default=0)
    successful_deliveries = Column(Integer, default=0)
    failed_deliveries = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Foreign Keys
    connection_id = Column(PGUUID(as_uuid=True), ForeignKey("websocket_connections.id"), nullable=True)
    organization_id = Column(PGUUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    
    # Relationships
    connection = relationship("WebSocketConnection", back_populates="events")


# Data Export Models
class ExportJobStatusEnum(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DataExportJob(Base):
    __tablename__ = "data_export_jobs"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_name = Column(String(255), nullable=False)
    export_type = Column(String(50), nullable=False)  # "pdf", "excel", "csv", "json"
    export_format = Column(String(20), nullable=False)
    status = Column(SQLEnum(ExportJobStatusEnum), default=ExportJobStatusEnum.PENDING)
    
    # Export configuration
    data_source = Column(String(100), nullable=False)  # "evaluation", "model_stats", "monitoring", etc.
    source_id = Column(PGUUID(as_uuid=True), nullable=False)  # ID of the source object
    export_filters = Column(JSON, default=dict)
    export_options = Column(JSON, default=dict)
    
    # File details
    filename = Column(String(255))
    file_path = Column(String(500))
    file_size_bytes = Column(Integer)
    download_url = Column(String(500))
    expires_at = Column(DateTime)
    
    # Processing details
    total_records = Column(Integer, default=0)
    processed_records = Column(Integer, default=0)
    progress_percentage = Column(Float, default=0.0)
    processing_time_ms = Column(Integer)
    error_message = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Foreign Keys
    organization_id = Column(PGUUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    requested_by = Column(PGUUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Relationships
    user = relationship("User", foreign_keys=[requested_by])


# Cache Management Models
class CacheEntry(Base):
    __tablename__ = "cache_entries"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    cache_key = Column(String(255), nullable=False, unique=True, index=True)
    cache_category = Column(String(100), nullable=False, index=True)  # "model_stats", "evaluations", etc.
    
    # Cache data
    cached_data = Column(JSON, nullable=False)
    data_size_bytes = Column(Integer)
    data_hash = Column(String(64), index=True)
    
    # Cache management
    ttl_seconds = Column(Integer, default=3600)  # Time to live
    access_count = Column(Integer, default=0)
    last_accessed_at = Column(DateTime, default=datetime.utcnow)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    
    # Foreign Keys
    organization_id = Column(PGUUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)


# Audit Trail Models
class AuditActionEnum(str, Enum):
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"
    EXPORT = "export"
    LOGIN = "login"
    LOGOUT = "logout"


class AuditLog(Base):
    __tablename__ = "audit_logs"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Action details
    action = Column(SQLEnum(AuditActionEnum), nullable=False)
    resource_type = Column(String(100), nullable=False)  # "model", "evaluation", "user", etc.
    resource_id = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=False)
    
    # Request details
    request_method = Column(String(10))  # HTTP method
    request_path = Column(String(500))
    request_params = Column(JSON, default=dict)
    request_body_hash = Column(String(64))  # Hash of request body for privacy
    
    # Response details
    response_status = Column(Integer)
    processing_time_ms = Column(Integer)
    
    # User and session info
    user_agent = Column(String(500))
    ip_address = Column(String(45))
    session_id = Column(String(255))
    
    # Additional context
    metadata = Column(JSON, default=dict)
    tags = Column(JSON, default=list)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Foreign Keys
    user_id = Column(PGUUID(as_uuid=True), ForeignKey("users.id"), nullable=True)  # Nullable for system actions
    organization_id = Column(PGUUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    
    # Relationships
    user = relationship("User", foreign_keys=[user_id])


# Model Drift Detection Models
class ModelDriftSeverityEnum(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ModelPerformanceBaseline(Base):
    __tablename__ = "model_performance_baselines"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    baseline_id = Column(String(255), nullable=False, unique=True, index=True)
    baseline_period = Column(String(100), nullable=False)
    sample_size = Column(Integer, nullable=False)
    
    # Classification metrics
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    roc_auc = Column(Float)
    
    # Regression metrics
    mae = Column(Float)
    mse = Column(Float)
    rmse = Column(Float)
    r2 = Column(Float)
    
    # Performance metrics
    prediction_latency = Column(Float)
    throughput = Column(Float)
    error_rate = Column(Float)
    
    # Additional baseline data
    baseline_data = Column(JSON, default=dict)
    
    # Timestamps
    established_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Foreign Keys
    model_id = Column(PGUUID(as_uuid=True), ForeignKey("models.id"), nullable=False)
    organization_id = Column(PGUUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    created_by = Column(PGUUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Relationships
    model = relationship("Model", foreign_keys=[model_id])
    organization = relationship("Organization")
    creator = relationship("User", foreign_keys=[created_by])
    drift_detections = relationship("ModelDriftDetection", back_populates="baseline")


class ModelDriftDetection(Base):
    __tablename__ = "model_drift_detections"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    drift_session_id = Column(String(255), nullable=False, unique=True, index=True)
    detection_period = Column(String(100), nullable=False)
    current_sample_size = Column(Integer, nullable=False)
    baseline_sample_size = Column(Integer, nullable=False)
    
    # Drift detection results
    drift_detected = Column(Boolean, default=False)
    drift_severity = Column(SQLEnum(ModelDriftSeverityEnum), default=ModelDriftSeverityEnum.NONE)
    drift_confidence = Column(Float, default=0.0)
    
    # Performance changes (JSON storage for flexibility)
    performance_degradation = Column(JSON, default=dict)
    statistical_significance = Column(JSON, default=dict)
    
    # Root cause analysis
    primary_causes = Column(JSON, default=list)
    data_drift_correlation = Column(Float)
    seasonal_pattern_detected = Column(Boolean, default=False)
    concept_drift_suspected = Column(Boolean, default=False)
    
    # Impact and recommendations
    business_impact_score = Column(Float, default=0.0)
    recommendations = Column(JSON, default=list)
    
    # Additional analysis data
    analysis_metadata = Column(JSON, default=dict)
    
    # Timestamps
    detected_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Foreign Keys
    model_id = Column(PGUUID(as_uuid=True), ForeignKey("models.id"), nullable=False)
    baseline_id = Column(PGUUID(as_uuid=True), ForeignKey("model_performance_baselines.id"), nullable=False)
    organization_id = Column(PGUUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    created_by = Column(PGUUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Relationships
    model = relationship("Model", foreign_keys=[model_id])
    baseline = relationship("ModelPerformanceBaseline", back_populates="drift_detections")
    organization = relationship("Organization")
    creator = relationship("User", foreign_keys=[created_by])
    impact_analyses = relationship("ModelDriftImpactAnalysis", back_populates="drift_detection")


class ModelDriftImpactAnalysis(Base):
    __tablename__ = "model_drift_impact_analyses"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Performance impact
    baseline_performance = Column(JSON, default=dict)
    current_performance = Column(JSON, default=dict)
    performance_delta = Column(JSON, default=dict)
    performance_trend = Column(JSON, default=dict)
    
    # Business impact
    estimated_cost_impact = Column(Float, default=0.0)
    affected_predictions = Column(Integer, default=0)
    false_positive_increase = Column(Float, default=0.0)
    false_negative_increase = Column(Float, default=0.0)
    
    # Technical impact
    latency_impact = Column(Float, default=0.0)
    throughput_impact = Column(Float, default=0.0)
    resource_utilization_change = Column(Float, default=0.0)
    
    # Confidence intervals
    performance_confidence_intervals = Column(JSON, default=dict)
    
    # Projections
    projected_impact_30d = Column(JSON, default=dict)
    retraining_urgency_score = Column(Float, default=0.0)
    
    # Timestamps
    analyzed_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Foreign Keys
    drift_detection_id = Column(PGUUID(as_uuid=True), ForeignKey("model_drift_detections.id"), nullable=False)
    model_id = Column(PGUUID(as_uuid=True), ForeignKey("models.id"), nullable=False)
    organization_id = Column(PGUUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    created_by = Column(PGUUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Relationships
    drift_detection = relationship("ModelDriftDetection", back_populates="impact_analyses")
    model = relationship("Model", foreign_keys=[model_id])
    organization = relationship("Organization")
    creator = relationship("User", foreign_keys=[created_by])


# Forward references
UserResponse.update_forward_refs()
RoleResponse.update_forward_refs()
AgentConfigurationResponse.update_forward_refs()
AgentEvaluationResponse.update_forward_refs()