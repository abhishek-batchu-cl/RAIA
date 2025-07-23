"""
RAIA Platform Configuration Management
Centralized configuration using Pydantic Settings for type safety and validation
"""

import secrets
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union

from pydantic import (
    BaseSettings, 
    PostgresDsn, 
    validator, 
    Field,
    AnyHttpUrl,
    EmailStr
)


class Settings(BaseSettings):
    """
    Application settings with environment variable support and validation.
    """
    
    # Application Settings
    APP_NAME: str = "RAIA Platform"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = Field(default="development", regex="^(development|staging|production)$")
    DEBUG: bool = Field(default=False)
    PORT: int = Field(default=8000, ge=1, le=65535)
    
    # Database Configuration
    DATABASE_URL: PostgresDsn
    DATABASE_POOL_SIZE: int = Field(default=20, ge=1, le=100)
    DATABASE_POOL_OVERFLOW: int = Field(default=0, ge=0, le=50)
    DATABASE_POOL_TIMEOUT: int = Field(default=30, ge=1)
    DATABASE_ECHO: bool = Field(default=False)
    
    # Redis Configuration
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_PASSWORD: Optional[str] = None
    REDIS_MAX_CONNECTIONS: int = Field(default=100, ge=1)
    REDIS_RETRY_ON_TIMEOUT: bool = True
    REDIS_SOCKET_TIMEOUT: int = Field(default=5, ge=1)
    
    # Authentication & Security
    JWT_SECRET_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, ge=1)
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7, ge=1)
    
    # Password Security
    PASSWORD_MIN_LENGTH: int = Field(default=8, ge=8)
    PASSWORD_HASH_SCHEMES: List[str] = ["bcrypt"]
    
    # CORS Settings
    CORS_ORIGINS: List[AnyHttpUrl] = []
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]
    
    # Trusted Hosts (for production)
    TRUSTED_HOSTS: List[str] = ["*"]
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS: int = Field(default=100, ge=1)
    RATE_LIMIT_PERIOD: int = Field(default=60, ge=1)  # seconds
    
    # LLM API Keys
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    COHERE_API_KEY: Optional[str] = None
    HUGGINGFACE_API_KEY: Optional[str] = None
    
    # Vector Database Configuration
    CHROMADB_URL: str = "http://localhost:8001"
    CHROMADB_API_KEY: Optional[str] = None
    VECTOR_DB_PATH: str = "./data/chroma"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    
    # Local Model Configuration
    LOCAL_MODEL_BASE_URL: Optional[str] = "http://localhost:11434"
    LOCAL_MODELS: List[str] = ["llama2", "mistral", "codellama"]
    
    # Time Series Database (InfluxDB)
    INFLUXDB_URL: str = "http://localhost:8086"
    INFLUXDB_TOKEN: Optional[str] = None
    INFLUXDB_ORG: str = "raia"
    INFLUXDB_BUCKET: str = "metrics"
    
    # Object Storage (S3/MinIO)
    S3_ENDPOINT_URL: Optional[str] = None
    S3_ACCESS_KEY_ID: Optional[str] = None
    S3_SECRET_ACCESS_KEY: Optional[str] = None
    S3_BUCKET_NAME: str = "raia-storage"
    S3_REGION: str = "us-east-1"
    
    # Message Queue (Kafka)
    KAFKA_BOOTSTRAP_SERVERS: List[str] = ["localhost:9092"]
    KAFKA_GROUP_ID: str = "raia-platform"
    KAFKA_AUTO_OFFSET_RESET: str = "latest"
    
    # Monitoring & Observability
    PROMETHEUS_ENABLED: bool = True
    PROMETHEUS_NAMESPACE: str = "raia"
    STRUCTURED_LOGGING: bool = True
    LOG_LEVEL: str = Field(default="INFO", regex="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    
    # Background Tasks
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"
    CELERY_TASK_SERIALIZER: str = "json"
    CELERY_RESULT_SERIALIZER: str = "json"
    CELERY_TIMEZONE: str = "UTC"
    
    # File Processing
    MAX_UPLOAD_SIZE: int = Field(default=100 * 1024 * 1024, ge=1)  # 100MB
    ALLOWED_UPLOAD_EXTENSIONS: List[str] = [".pdf", ".txt", ".docx", ".csv", ".json"]
    DOCUMENT_PROCESSING_CHUNK_SIZE: int = Field(default=1000, ge=100)
    DOCUMENT_PROCESSING_OVERLAP: int = Field(default=200, ge=0)
    
    # ML Model Configuration
    MODEL_CACHE_TTL: int = Field(default=3600, ge=60)  # seconds
    MODEL_INFERENCE_TIMEOUT: int = Field(default=30, ge=1)  # seconds
    MAX_PREDICTION_BATCH_SIZE: int = Field(default=1000, ge=1)
    
    # Data Drift Detection
    DRIFT_DETECTION_ENABLED: bool = True
    DRIFT_DETECTION_THRESHOLD: float = Field(default=0.05, ge=0.0, le=1.0)
    DRIFT_DETECTION_WINDOW_SIZE: int = Field(default=1000, ge=100)
    
    # Fairness Analysis
    FAIRNESS_ANALYSIS_ENABLED: bool = True
    FAIRNESS_THRESHOLD: float = Field(default=0.8, ge=0.0, le=1.0)
    PROTECTED_ATTRIBUTES: List[str] = ["race", "gender", "age", "religion"]
    
    # Email Configuration
    SMTP_SERVER: Optional[str] = None
    SMTP_PORT: int = Field(default=587, ge=1, le=65535)
    SMTP_USERNAME: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    SMTP_USE_TLS: bool = True
    EMAIL_FROM: Optional[EmailStr] = None
    
    # Notification Services
    SLACK_WEBHOOK_URL: Optional[AnyHttpUrl] = None
    DISCORD_WEBHOOK_URL: Optional[AnyHttpUrl] = None
    TEAMS_WEBHOOK_URL: Optional[AnyHttpUrl] = None
    
    # Feature Flags
    ENABLE_AGENT_EVALUATION: bool = True
    ENABLE_MODEL_EXPLAINABILITY: bool = True
    ENABLE_DATA_DRIFT_MONITORING: bool = True
    ENABLE_FAIRNESS_ANALYSIS: bool = True
    ENABLE_REAL_TIME_MONITORING: bool = True
    ENABLE_CUSTOM_DASHBOARDS: bool = True
    
    # Performance Settings
    ASYNC_POOL_SIZE: int = Field(default=100, ge=1)
    REQUEST_TIMEOUT: int = Field(default=300, ge=1)  # seconds
    WEBSOCKET_HEARTBEAT_INTERVAL: int = Field(default=30, ge=1)
    
    # Caching Configuration
    CACHE_TTL_DEFAULT: int = Field(default=3600, ge=1)
    CACHE_TTL_MODELS: int = Field(default=7200, ge=1)
    CACHE_TTL_EVALUATIONS: int = Field(default=1800, ge=1)
    CACHE_TTL_EXPLANATIONS: int = Field(default=3600, ge=1)
    
    @validator("DATABASE_URL", pre=True)
    def validate_database_url(cls, v: str) -> str:
        """Validate and potentially modify database URL"""
        if v.startswith("postgresql://"):
            # Convert to asyncpg-compatible URL
            return v.replace("postgresql://", "postgresql+asyncpg://", 1)
        return v
    
    @validator("CORS_ORIGINS", pre=True)
    def validate_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        """Parse CORS origins from string or list"""
        if isinstance(v, str) and v:
            return [origin.strip() for origin in v.split(",")]
        elif isinstance(v, list):
            return v
        return []
    
    @validator("TRUSTED_HOSTS", pre=True)
    def validate_trusted_hosts(cls, v: Union[str, List[str]]) -> List[str]:
        """Parse trusted hosts from string or list"""
        if isinstance(v, str) and v:
            return [host.strip() for host in v.split(",")]
        elif isinstance(v, list):
            return v
        return ["*"]
    
    @validator("KAFKA_BOOTSTRAP_SERVERS", pre=True)
    def validate_kafka_servers(cls, v: Union[str, List[str]]) -> List[str]:
        """Parse Kafka bootstrap servers from string or list"""
        if isinstance(v, str):
            return [server.strip() for server in v.split(",")]
        return v
    
    @validator("PROTECTED_ATTRIBUTES", pre=True) 
    def validate_protected_attributes(cls, v: Union[str, List[str]]) -> List[str]:
        """Parse protected attributes from string or list"""
        if isinstance(v, str):
            return [attr.strip() for attr in v.split(",")]
        return v
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration dictionary"""
        return {
            "url": self.DATABASE_URL,
            "pool_size": self.DATABASE_POOL_SIZE,
            "max_overflow": self.DATABASE_POOL_OVERFLOW,
            "pool_timeout": self.DATABASE_POOL_TIMEOUT,
            "echo": self.DATABASE_ECHO and self.ENVIRONMENT == "development",
        }
    
    def get_redis_config(self) -> Dict[str, Any]:
        """Get Redis configuration dictionary"""
        return {
            "url": self.REDIS_URL,
            "password": self.REDIS_PASSWORD,
            "max_connections": self.REDIS_MAX_CONNECTIONS,
            "retry_on_timeout": self.REDIS_RETRY_ON_TIMEOUT,
            "socket_timeout": self.REDIS_SOCKET_TIMEOUT,
        }
    
    def get_celery_config(self) -> Dict[str, Any]:
        """Get Celery configuration dictionary"""
        return {
            "broker_url": self.CELERY_BROKER_URL,
            "result_backend": self.CELERY_RESULT_BACKEND,
            "task_serializer": self.CELERY_TASK_SERIALIZER,
            "result_serializer": self.CELERY_RESULT_SERIALIZER,
            "timezone": self.CELERY_TIMEZONE,
            "include": [
                "app.services.evaluation_service",
                "app.services.explanation_service",
                "app.services.data_drift_service",
            ],
        }
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached application settings.
    Uses lru_cache to avoid re-reading environment variables on every call.
    """
    return Settings()


# Export settings for easy import
settings = get_settings()