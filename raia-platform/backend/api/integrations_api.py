# Integration API for External Systems
import os
import uuid
import json
import logging
import asyncio
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from enum import Enum

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, validator, SecretStr
from sqlalchemy.orm import Session
from sqlalchemy import Column, String, DateTime, Text, Boolean, JSON, UUID as PG_UUID, Integer, Float, ForeignKey, func, desc
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

# External service clients
import aiohttp
import boto3
import requests
from google.cloud import storage as gcs
from azure.storage.blob import BlobServiceClient
import docker
import kubernetes
from kubernetes import client as k8s_client

# Database and message queue
import redis
import pymongo
from sqlalchemy import create_engine
import psycopg2

# Authentication and security
from jose import jwt
import secrets

Base = declarative_base()
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/integrations", tags=["integrations"])

# Enums
class IntegrationType(str, Enum):
    # Cloud Platforms
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    
    # ML Platforms
    HUGGING_FACE = "hugging_face"
    MLFLOW = "mlflow"
    WANDB = "wandb"
    TENSORBOARD = "tensorboard"
    
    # Data Sources
    SNOWFLAKE = "snowflake"
    DATABRICKS = "databricks"
    BIGQUERY = "bigquery"
    REDSHIFT = "redshift"
    POSTGRESQL = "postgresql"
    MONGODB = "mongodb"
    
    # Storage
    S3 = "s3"
    GCS = "gcs"
    AZURE_BLOB = "azure_blob"
    HDFS = "hdfs"
    
    # Messaging
    SLACK = "slack"
    TEAMS = "teams"
    DISCORD = "discord"
    
    # DevOps
    GITHUB = "github"
    GITLAB = "gitlab"
    JENKINS = "jenkins"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    
    # Monitoring
    DATADOG = "datadog"
    NEW_RELIC = "new_relic"
    PROMETHEUS = "prometheus"
    GRAFANA = "grafana"
    
    # Business Intelligence
    TABLEAU = "tableau"
    POWER_BI = "power_bi"
    LOOKER = "looker"
    
    # Custom
    WEBHOOK = "webhook"
    REST_API = "rest_api"
    GRAPHQL = "graphql"

class IntegrationStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    TESTING = "testing"

class WebhookEvent(str, Enum):
    MODEL_DEPLOYED = "model.deployed"
    MODEL_FAILED = "model.failed"
    EXPERIMENT_COMPLETED = "experiment.completed"
    DATASET_UPLOADED = "dataset.uploaded"
    ALERT_TRIGGERED = "alert.triggered"
    REPORT_GENERATED = "report.generated"
    USER_REGISTERED = "user.registered"

# Database Models
class Integration(Base):
    """External system integration configuration"""
    __tablename__ = "integrations"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    
    # Integration configuration
    integration_type = Column(String(100), nullable=False)
    provider = Column(String(100))  # AWS, Google, etc.
    
    # Connection details
    config = Column(JSON, nullable=False)  # Connection parameters
    credentials = Column(JSON)  # Encrypted credentials
    
    # Features and capabilities
    capabilities = Column(JSON)  # What this integration can do
    supported_operations = Column(JSON)  # List of supported operations
    
    # Status and health
    status = Column(String(50), default=IntegrationStatus.INACTIVE)
    last_test_time = Column(DateTime)
    last_test_result = Column(JSON)
    health_check_interval = Column(Integer, default=300)  # seconds
    
    # Usage tracking
    request_count = Column(Integer, default=0)
    success_count = Column(Integer, default=0)
    failure_count = Column(Integer, default=0)
    last_used = Column(DateTime)
    
    # Rate limiting
    rate_limit_per_hour = Column(Integer)
    current_hour_requests = Column(Integer, default=0)
    current_hour_start = Column(DateTime)
    
    # Metadata
    tags = Column(JSON)
    metadata = Column(JSON)
    
    # Ownership
    created_by = Column(String(255), nullable=False)
    organization_id = Column(String(255))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    webhooks = relationship("Webhook", back_populates="integration", cascade="all, delete-orphan")
    sync_jobs = relationship("SyncJob", back_populates="integration", cascade="all, delete-orphan")

class Webhook(Base):
    """Webhook configuration for event notifications"""
    __tablename__ = "webhooks"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    integration_id = Column(PG_UUID(as_uuid=True), ForeignKey('integrations.id'))
    
    # Webhook configuration
    name = Column(String(255), nullable=False)
    url = Column(String(1000), nullable=False)
    secret = Column(String(255))  # For signature verification
    
    # Event configuration
    events = Column(JSON, nullable=False)  # List of events to subscribe to
    filters = Column(JSON)  # Event filters
    
    # Request configuration
    headers = Column(JSON)  # Additional headers
    timeout_seconds = Column(Integer, default=30)
    retry_count = Column(Integer, default=3)
    retry_delay_seconds = Column(Integer, default=60)
    
    # Status
    is_active = Column(Boolean, default=True)
    
    # Usage stats
    delivery_count = Column(Integer, default=0)
    success_count = Column(Integer, default=0)
    failure_count = Column(Integer, default=0)
    last_delivery = Column(DateTime)
    last_success = Column(DateTime)
    
    # Ownership
    created_by = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    integration = relationship("Integration", back_populates="webhooks")
    deliveries = relationship("WebhookDelivery", back_populates="webhook", cascade="all, delete-orphan")

class WebhookDelivery(Base):
    """Webhook delivery attempts"""
    __tablename__ = "webhook_deliveries"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    webhook_id = Column(PG_UUID(as_uuid=True), ForeignKey('webhooks.id'), nullable=False)
    
    # Event details
    event_type = Column(String(100), nullable=False)
    event_data = Column(JSON, nullable=False)
    
    # Delivery details
    request_id = Column(String(255), unique=True)
    attempt_count = Column(Integer, default=0)
    max_attempts = Column(Integer, default=3)
    
    # Request/Response
    request_headers = Column(JSON)
    request_body = Column(Text)
    response_status = Column(Integer)
    response_headers = Column(JSON)
    response_body = Column(Text)
    
    # Status and timing
    status = Column(String(50))  # pending, success, failed, expired
    delivered_at = Column(DateTime)
    next_attempt_at = Column(DateTime)
    
    # Error tracking
    error_message = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    webhook = relationship("Webhook", back_populates="deliveries")

class SyncJob(Base):
    """Data synchronization jobs"""
    __tablename__ = "sync_jobs"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    integration_id = Column(PG_UUID(as_uuid=True), ForeignKey('integrations.id'), nullable=False)
    
    # Job configuration
    name = Column(String(255), nullable=False)
    sync_type = Column(String(100))  # import, export, bidirectional
    source_config = Column(JSON)
    destination_config = Column(JSON)
    
    # Scheduling
    schedule = Column(String(255))  # Cron expression
    is_scheduled = Column(Boolean, default=False)
    last_run = Column(DateTime)
    next_run = Column(DateTime)
    
    # Status
    status = Column(String(50), default="inactive")
    
    # Statistics
    total_runs = Column(Integer, default=0)
    successful_runs = Column(Integer, default=0)
    failed_runs = Column(Integer, default=0)
    
    # Ownership
    created_by = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    integration = relationship("Integration", back_populates="sync_jobs")

# Pydantic Models
class IntegrationCreate(BaseModel):
    name: str
    description: Optional[str] = None
    integration_type: IntegrationType
    provider: Optional[str] = None
    config: Dict[str, Any]
    credentials: Optional[Dict[str, Any]] = {}
    capabilities: Optional[List[str]] = []
    rate_limit_per_hour: Optional[int] = None
    tags: Optional[List[str]] = []

class IntegrationUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    credentials: Optional[Dict[str, Any]] = None
    status: Optional[IntegrationStatus] = None
    rate_limit_per_hour: Optional[int] = None
    tags: Optional[List[str]] = None

class IntegrationResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    integration_type: str
    provider: Optional[str]
    status: str
    capabilities: Optional[List[str]]
    request_count: int
    success_count: int
    failure_count: int
    last_test_time: Optional[datetime]
    created_at: datetime
    
    class Config:
        orm_mode = True

class WebhookCreate(BaseModel):
    name: str
    url: str
    secret: Optional[str] = None
    events: List[WebhookEvent]
    filters: Optional[Dict[str, Any]] = {}
    headers: Optional[Dict[str, str]] = {}
    timeout_seconds: int = 30
    retry_count: int = 3

class WebhookResponse(BaseModel):
    id: str
    integration_id: str
    name: str
    url: str
    events: List[str]
    is_active: bool
    delivery_count: int
    success_count: int
    failure_count: int
    created_at: datetime
    
    class Config:
        orm_mode = True

class SyncJobCreate(BaseModel):
    name: str
    sync_type: str = "import"
    source_config: Dict[str, Any]
    destination_config: Dict[str, Any]
    schedule: Optional[str] = None

# Integration service
class IntegrationService:
    """Service for managing external integrations"""
    
    def __init__(self, db: Session):
        self.db = db
        
        # Integration clients
        self.clients = {}
        
        # Webhook delivery queue
        self.webhook_queue = asyncio.Queue()
        
        # Integration handlers
        self.integration_handlers = {
            IntegrationType.AWS: self._handle_aws,
            IntegrationType.GCP: self._handle_gcp,
            IntegrationType.AZURE: self._handle_azure,
            IntegrationType.HUGGING_FACE: self._handle_hugging_face,
            IntegrationType.MLFLOW: self._handle_mlflow,
            IntegrationType.WANDB: self._handle_wandb,
            IntegrationType.SNOWFLAKE: self._handle_snowflake,
            IntegrationType.POSTGRESQL: self._handle_postgresql,
            IntegrationType.MONGODB: self._handle_mongodb,
            IntegrationType.S3: self._handle_s3,
            IntegrationType.GCS: self._handle_gcs,
            IntegrationType.SLACK: self._handle_slack,
            IntegrationType.GITHUB: self._handle_github,
            IntegrationType.DOCKER: self._handle_docker,
            IntegrationType.KUBERNETES: self._handle_kubernetes,
            IntegrationType.WEBHOOK: self._handle_webhook,
            IntegrationType.REST_API: self._handle_rest_api
        }
        
        # Start background tasks
        asyncio.create_task(self._webhook_delivery_worker())
    
    async def create_integration(self, integration_data: IntegrationCreate, user_id: str) -> Integration:
        """Create a new integration"""
        
        # Encrypt sensitive credentials
        encrypted_credentials = self._encrypt_credentials(integration_data.credentials)
        
        integration = Integration(
            name=integration_data.name,
            description=integration_data.description,
            integration_type=integration_data.integration_type,
            provider=integration_data.provider,
            config=integration_data.config,
            credentials=encrypted_credentials,
            capabilities=integration_data.capabilities,
            rate_limit_per_hour=integration_data.rate_limit_per_hour,
            tags=integration_data.tags,
            created_by=user_id,
            organization_id=self._get_user_org(user_id)
        )
        
        self.db.add(integration)
        self.db.commit()
        self.db.refresh(integration)
        
        # Test the integration
        await self._test_integration(integration)
        
        logger.info(f"Created integration {integration.name} (ID: {integration.id})")
        return integration
    
    async def test_integration(self, integration_id: str, user_id: str) -> Dict[str, Any]:
        """Test an integration connection"""
        
        integration = self._get_integration_by_id(integration_id, user_id)
        if not integration:
            raise HTTPException(status_code=404, detail="Integration not found")
        
        return await self._test_integration(integration)
    
    async def _test_integration(self, integration: Integration) -> Dict[str, Any]:
        """Test integration connectivity"""
        
        try:
            handler = self.integration_handlers.get(IntegrationType(integration.integration_type))
            if not handler:
                raise ValueError(f"No handler for integration type: {integration.integration_type}")
            
            # Test connection
            result = await handler(integration, "test_connection", {})
            
            # Update integration status
            integration.status = IntegrationStatus.ACTIVE if result.get('success') else IntegrationStatus.ERROR
            integration.last_test_time = datetime.utcnow()
            integration.last_test_result = result
            
            self.db.commit()
            
            return result
            
        except Exception as e:
            integration.status = IntegrationStatus.ERROR
            integration.last_test_time = datetime.utcnow()
            integration.last_test_result = {'success': False, 'error': str(e)}
            
            self.db.commit()
            
            return {'success': False, 'error': str(e)}
    
    async def execute_integration_operation(self, integration_id: str, operation: str, params: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Execute an operation on an integration"""
        
        integration = self._get_integration_by_id(integration_id, user_id)
        if not integration:
            raise HTTPException(status_code=404, detail="Integration not found")
        
        if integration.status != IntegrationStatus.ACTIVE:
            raise HTTPException(status_code=400, detail="Integration is not active")
        
        # Check rate limiting
        if not self._check_rate_limit(integration):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        try:
            handler = self.integration_handlers.get(IntegrationType(integration.integration_type))
            if not handler:
                raise ValueError(f"No handler for integration type: {integration.integration_type}")
            
            # Execute operation
            result = await handler(integration, operation, params)
            
            # Update statistics
            integration.request_count += 1
            integration.last_used = datetime.utcnow()
            
            if result.get('success'):
                integration.success_count += 1
            else:
                integration.failure_count += 1
            
            self.db.commit()
            
            return result
            
        except Exception as e:
            integration.failure_count += 1
            self.db.commit()
            
            logger.error(f"Integration operation failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def create_webhook(self, integration_id: str, webhook_data: WebhookCreate, user_id: str) -> Webhook:
        """Create a webhook"""
        
        integration = self._get_integration_by_id(integration_id, user_id)
        if not integration:
            raise HTTPException(status_code=404, detail="Integration not found")
        
        webhook = Webhook(
            integration_id=integration.id,
            name=webhook_data.name,
            url=webhook_data.url,
            secret=webhook_data.secret or secrets.token_urlsafe(32),
            events=webhook_data.events,
            filters=webhook_data.filters,
            headers=webhook_data.headers,
            timeout_seconds=webhook_data.timeout_seconds,
            retry_count=webhook_data.retry_count,
            created_by=user_id
        )
        
        self.db.add(webhook)
        self.db.commit()
        self.db.refresh(webhook)
        
        logger.info(f"Created webhook {webhook.name} (ID: {webhook.id})")
        return webhook
    
    async def trigger_webhook(self, event_type: str, event_data: Dict[str, Any], filters: Optional[Dict[str, Any]] = None):
        """Trigger webhooks for an event"""
        
        # Find matching webhooks
        webhooks = self.db.query(Webhook).filter(
            Webhook.is_active == True,
            Webhook.events.contains([event_type])
        ).all()
        
        for webhook in webhooks:
            # Check filters
            if filters and not self._match_filters(event_data, webhook.filters):
                continue
            
            # Create delivery record
            delivery = WebhookDelivery(
                webhook_id=webhook.id,
                event_type=event_type,
                event_data=event_data,
                request_id=f"wh_{uuid.uuid4().hex[:12]}",
                max_attempts=webhook.retry_count
            )
            
            self.db.add(delivery)
            self.db.commit()
            self.db.refresh(delivery)
            
            # Queue for delivery
            await self.webhook_queue.put(delivery.id)
    
    async def _webhook_delivery_worker(self):
        """Background worker for webhook delivery"""
        
        while True:
            try:
                # Get delivery from queue
                delivery_id = await self.webhook_queue.get()
                
                delivery = self.db.query(WebhookDelivery).filter(
                    WebhookDelivery.id == delivery_id
                ).first()
                
                if delivery:
                    await self._deliver_webhook(delivery)
                    
            except Exception as e:
                logger.error(f"Webhook delivery worker error: {str(e)}")
                await asyncio.sleep(1)
    
    async def _deliver_webhook(self, delivery: WebhookDelivery):
        """Deliver a single webhook"""
        
        webhook = delivery.webhook
        
        try:
            delivery.attempt_count += 1
            
            # Prepare payload
            payload = {
                'event': delivery.event_type,
                'timestamp': delivery.created_at.isoformat(),
                'data': delivery.event_data,
                'request_id': delivery.request_id
            }
            
            # Prepare headers
            headers = {
                'Content-Type': 'application/json',
                'User-Agent': 'RAIA-Platform-Webhook/1.0',
                'X-RAIA-Event': delivery.event_type,
                'X-RAIA-Request-ID': delivery.request_id
            }
            
            if webhook.headers:
                headers.update(webhook.headers)
            
            # Add signature if secret is configured
            if webhook.secret:
                signature = self._generate_webhook_signature(
                    json.dumps(payload, sort_keys=True), 
                    webhook.secret
                )
                headers['X-RAIA-Signature'] = signature
            
            # Make request
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=webhook.timeout_seconds)) as session:
                async with session.post(
                    webhook.url,
                    json=payload,
                    headers=headers
                ) as response:
                    
                    delivery.response_status = response.status
                    delivery.response_headers = dict(response.headers)
                    delivery.response_body = await response.text()
                    
                    if 200 <= response.status < 300:
                        # Success
                        delivery.status = 'success'
                        delivery.delivered_at = datetime.utcnow()
                        
                        webhook.success_count += 1
                        webhook.last_success = datetime.utcnow()
                    else:
                        # HTTP error
                        delivery.status = 'failed'
                        delivery.error_message = f"HTTP {response.status}: {delivery.response_body}"
                        
                        webhook.failure_count += 1
            
            webhook.delivery_count += 1
            webhook.last_delivery = datetime.utcnow()
            
        except Exception as e:
            # Network/other error
            delivery.status = 'failed'
            delivery.error_message = str(e)
            webhook.failure_count += 1
        
        # Schedule retry if needed
        if (delivery.status == 'failed' and 
            delivery.attempt_count < delivery.max_attempts):
            
            # Exponential backoff
            delay_seconds = (2 ** delivery.attempt_count) * 60
            delivery.next_attempt_at = datetime.utcnow() + timedelta(seconds=delay_seconds)
            delivery.status = 'pending'
            
            # Re-queue for retry
            await asyncio.sleep(delay_seconds)
            await self.webhook_queue.put(delivery.id)
        
        self.db.commit()
    
    # Integration handlers
    async def _handle_aws(self, integration: Integration, operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle AWS operations"""
        
        config = integration.config
        credentials = self._decrypt_credentials(integration.credentials)
        
        if operation == "test_connection":
            try:
                # Test S3 connection
                import boto3
                s3 = boto3.client(
                    's3',
                    aws_access_key_id=credentials.get('access_key_id'),
                    aws_secret_access_key=credentials.get('secret_access_key'),
                    region_name=config.get('region', 'us-east-1')
                )
                s3.list_buckets()
                return {'success': True, 'message': 'AWS connection successful'}
            except Exception as e:
                return {'success': False, 'error': str(e)}
        
        elif operation == "list_buckets":
            try:
                s3 = boto3.client('s3', **credentials)
                response = s3.list_buckets()
                buckets = [bucket['Name'] for bucket in response['Buckets']]
                return {'success': True, 'buckets': buckets}
            except Exception as e:
                return {'success': False, 'error': str(e)}
        
        elif operation == "upload_file":
            # Implementation for file upload
            return {'success': True, 'message': 'File upload initiated'}
        
        return {'success': False, 'error': f'Unsupported operation: {operation}'}
    
    async def _handle_gcp(self, integration: Integration, operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Google Cloud operations"""
        
        if operation == "test_connection":
            try:
                # Test GCS connection
                from google.cloud import storage
                client = storage.Client()
                list(client.list_buckets())
                return {'success': True, 'message': 'GCP connection successful'}
            except Exception as e:
                return {'success': False, 'error': str(e)}
        
        return {'success': False, 'error': f'Unsupported operation: {operation}'}
    
    async def _handle_azure(self, integration: Integration, operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Azure operations"""
        
        if operation == "test_connection":
            try:
                # Test Azure Blob Storage connection
                from azure.storage.blob import BlobServiceClient
                credentials = self._decrypt_credentials(integration.credentials)
                blob_service = BlobServiceClient(
                    account_url=f"https://{credentials.get('account_name')}.blob.core.windows.net",
                    credential=credentials.get('account_key')
                )
                list(blob_service.list_containers())
                return {'success': True, 'message': 'Azure connection successful'}
            except Exception as e:
                return {'success': False, 'error': str(e)}
        
        return {'success': False, 'error': f'Unsupported operation: {operation}'}
    
    async def _handle_hugging_face(self, integration: Integration, operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Hugging Face operations"""
        
        if operation == "test_connection":
            try:
                credentials = self._decrypt_credentials(integration.credentials)
                headers = {'Authorization': f"Bearer {credentials.get('api_token')}"}
                
                async with aiohttp.ClientSession() as session:
                    async with session.get('https://huggingface.co/api/whoami', headers=headers) as response:
                        if response.status == 200:
                            return {'success': True, 'message': 'Hugging Face connection successful'}
                        else:
                            return {'success': False, 'error': f'HTTP {response.status}'}
            except Exception as e:
                return {'success': False, 'error': str(e)}
        
        return {'success': False, 'error': f'Unsupported operation: {operation}'}
    
    async def _handle_mlflow(self, integration: Integration, operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MLflow operations"""
        
        if operation == "test_connection":
            try:
                import mlflow
                mlflow.set_tracking_uri(integration.config.get('tracking_uri'))
                experiments = mlflow.list_experiments()
                return {'success': True, 'message': 'MLflow connection successful', 'experiments': len(experiments)}
            except Exception as e:
                return {'success': False, 'error': str(e)}
        
        return {'success': False, 'error': f'Unsupported operation: {operation}'}
    
    async def _handle_wandb(self, integration: Integration, operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Weights & Biases operations"""
        
        if operation == "test_connection":
            try:
                credentials = self._decrypt_credentials(integration.credentials)
                headers = {'Authorization': f"Bearer {credentials.get('api_key')}"}
                
                async with aiohttp.ClientSession() as session:
                    async with session.get('https://api.wandb.ai/graphql', headers=headers) as response:
                        if response.status == 200:
                            return {'success': True, 'message': 'W&B connection successful'}
                        else:
                            return {'success': False, 'error': f'HTTP {response.status}'}
            except Exception as e:
                return {'success': False, 'error': str(e)}
        
        return {'success': False, 'error': f'Unsupported operation: {operation}'}
    
    async def _handle_snowflake(self, integration: Integration, operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Snowflake operations"""
        
        if operation == "test_connection":
            try:
                import snowflake.connector
                config = integration.config
                credentials = self._decrypt_credentials(integration.credentials)
                
                conn = snowflake.connector.connect(
                    user=credentials.get('username'),
                    password=credentials.get('password'),
                    account=config.get('account'),
                    warehouse=config.get('warehouse'),
                    database=config.get('database'),
                    schema=config.get('schema')
                )
                
                cursor = conn.cursor()
                cursor.execute("SELECT CURRENT_VERSION()")
                cursor.close()
                conn.close()
                
                return {'success': True, 'message': 'Snowflake connection successful'}
            except Exception as e:
                return {'success': False, 'error': str(e)}
        
        return {'success': False, 'error': f'Unsupported operation: {operation}'}
    
    async def _handle_postgresql(self, integration: Integration, operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle PostgreSQL operations"""
        
        if operation == "test_connection":
            try:
                import psycopg2
                config = integration.config
                credentials = self._decrypt_credentials(integration.credentials)
                
                conn = psycopg2.connect(
                    host=config.get('host'),
                    port=config.get('port', 5432),
                    database=config.get('database'),
                    user=credentials.get('username'),
                    password=credentials.get('password')
                )
                
                cursor = conn.cursor()
                cursor.execute("SELECT version()")
                cursor.close()
                conn.close()
                
                return {'success': True, 'message': 'PostgreSQL connection successful'}
            except Exception as e:
                return {'success': False, 'error': str(e)}
        
        return {'success': False, 'error': f'Unsupported operation: {operation}'}
    
    async def _handle_mongodb(self, integration: Integration, operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MongoDB operations"""
        
        if operation == "test_connection":
            try:
                import pymongo
                config = integration.config
                credentials = self._decrypt_credentials(integration.credentials)
                
                client = pymongo.MongoClient(
                    host=config.get('host'),
                    port=config.get('port', 27017),
                    username=credentials.get('username'),
                    password=credentials.get('password'),
                    authSource=config.get('auth_database', 'admin')
                )
                
                # Test connection
                client.server_info()
                client.close()
                
                return {'success': True, 'message': 'MongoDB connection successful'}
            except Exception as e:
                return {'success': False, 'error': str(e)}
        
        return {'success': False, 'error': f'Unsupported operation: {operation}'}
    
    async def _handle_s3(self, integration: Integration, operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle S3 operations"""
        return await self._handle_aws(integration, operation, params)
    
    async def _handle_gcs(self, integration: Integration, operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Google Cloud Storage operations"""
        return await self._handle_gcp(integration, operation, params)
    
    async def _handle_slack(self, integration: Integration, operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Slack operations"""
        
        if operation == "test_connection":
            try:
                credentials = self._decrypt_credentials(integration.credentials)
                headers = {'Authorization': f"Bearer {credentials.get('bot_token')}"}
                
                async with aiohttp.ClientSession() as session:
                    async with session.post('https://slack.com/api/auth.test', headers=headers) as response:
                        data = await response.json()
                        if data.get('ok'):
                            return {'success': True, 'message': 'Slack connection successful'}
                        else:
                            return {'success': False, 'error': data.get('error', 'Unknown error')}
            except Exception as e:
                return {'success': False, 'error': str(e)}
        
        elif operation == "send_message":
            try:
                credentials = self._decrypt_credentials(integration.credentials)
                headers = {'Authorization': f"Bearer {credentials.get('bot_token')}"}
                
                payload = {
                    'channel': params.get('channel'),
                    'text': params.get('message'),
                    'username': params.get('username', 'RAIA Platform')
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post('https://slack.com/api/chat.postMessage', headers=headers, json=payload) as response:
                        data = await response.json()
                        return {'success': data.get('ok', False), 'data': data}
            except Exception as e:
                return {'success': False, 'error': str(e)}
        
        return {'success': False, 'error': f'Unsupported operation: {operation}'}
    
    async def _handle_github(self, integration: Integration, operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle GitHub operations"""
        
        if operation == "test_connection":
            try:
                credentials = self._decrypt_credentials(integration.credentials)
                headers = {'Authorization': f"token {credentials.get('access_token')}"}
                
                async with aiohttp.ClientSession() as session:
                    async with session.get('https://api.github.com/user', headers=headers) as response:
                        if response.status == 200:
                            return {'success': True, 'message': 'GitHub connection successful'}
                        else:
                            return {'success': False, 'error': f'HTTP {response.status}'}
            except Exception as e:
                return {'success': False, 'error': str(e)}
        
        return {'success': False, 'error': f'Unsupported operation: {operation}'}
    
    async def _handle_docker(self, integration: Integration, operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Docker operations"""
        
        if operation == "test_connection":
            try:
                import docker
                client = docker.from_env()
                client.ping()
                return {'success': True, 'message': 'Docker connection successful'}
            except Exception as e:
                return {'success': False, 'error': str(e)}
        
        return {'success': False, 'error': f'Unsupported operation: {operation}'}
    
    async def _handle_kubernetes(self, integration: Integration, operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Kubernetes operations"""
        
        if operation == "test_connection":
            try:
                from kubernetes import client, config
                config.load_kube_config()
                v1 = client.CoreV1Api()
                v1.list_node()
                return {'success': True, 'message': 'Kubernetes connection successful'}
            except Exception as e:
                return {'success': False, 'error': str(e)}
        
        return {'success': False, 'error': f'Unsupported operation: {operation}'}
    
    async def _handle_webhook(self, integration: Integration, operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle generic webhook operations"""
        
        if operation == "test_connection":
            try:
                config = integration.config
                url = config.get('url')
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        if response.status < 400:
                            return {'success': True, 'message': 'Webhook endpoint reachable'}
                        else:
                            return {'success': False, 'error': f'HTTP {response.status}'}
            except Exception as e:
                return {'success': False, 'error': str(e)}
        
        return {'success': False, 'error': f'Unsupported operation: {operation}'}
    
    async def _handle_rest_api(self, integration: Integration, operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle REST API operations"""
        
        if operation == "test_connection":
            try:
                config = integration.config
                base_url = config.get('base_url')
                
                headers = {}
                credentials = self._decrypt_credentials(integration.credentials)
                
                if credentials.get('api_key'):
                    headers['Authorization'] = f"Bearer {credentials.get('api_key')}"
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{base_url}/health", headers=headers) as response:
                        if response.status < 400:
                            return {'success': True, 'message': 'REST API connection successful'}
                        else:
                            return {'success': False, 'error': f'HTTP {response.status}'}
            except Exception as e:
                return {'success': False, 'error': str(e)}
        
        return {'success': False, 'error': f'Unsupported operation: {operation}'}
    
    # Helper methods
    def _get_integration_by_id(self, integration_id: str, user_id: str) -> Optional[Integration]:
        """Get integration by ID with access control"""
        
        return self.db.query(Integration).filter(
            Integration.id == integration_id,
            (
                (Integration.created_by == user_id) |
                (Integration.organization_id == self._get_user_org(user_id))
            )
        ).first()
    
    def _get_user_org(self, user_id: str) -> Optional[str]:
        """Get user's organization ID"""
        return "default_org"
    
    def _encrypt_credentials(self, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt sensitive credentials"""
        # In a real implementation, this would use proper encryption
        # For demo purposes, just return as-is
        return credentials
    
    def _decrypt_credentials(self, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt credentials"""
        # In a real implementation, this would decrypt the credentials
        return credentials or {}
    
    def _check_rate_limit(self, integration: Integration) -> bool:
        """Check if integration is within rate limits"""
        
        if not integration.rate_limit_per_hour:
            return True
        
        now = datetime.utcnow()
        current_hour = now.replace(minute=0, second=0, microsecond=0)
        
        # Reset counter if we're in a new hour
        if not integration.current_hour_start or integration.current_hour_start < current_hour:
            integration.current_hour_requests = 0
            integration.current_hour_start = current_hour
        
        return integration.current_hour_requests < integration.rate_limit_per_hour
    
    def _match_filters(self, event_data: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if event data matches webhook filters"""
        
        if not filters:
            return True
        
        # Simple filter matching logic
        for key, expected_value in filters.items():
            if key in event_data and event_data[key] != expected_value:
                return False
        
        return True
    
    def _generate_webhook_signature(self, payload: str, secret: str) -> str:
        """Generate webhook signature for verification"""
        
        signature = hmac.new(
            secret.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return f"sha256={signature}"

# Dependency injection
def get_db():
    pass

def get_current_user():
    return "current_user_id"

# API Endpoints
@router.post("/", response_model=IntegrationResponse)
async def create_integration(
    integration_data: IntegrationCreate,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Create a new integration"""
    
    service = IntegrationService(db)
    integration = await service.create_integration(integration_data, current_user)
    return integration

@router.get("/", response_model=List[IntegrationResponse])
async def get_integrations(
    skip: int = 0,
    limit: int = 100,
    integration_type: Optional[str] = None,
    status: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Get integrations"""
    
    service = IntegrationService(db)
    
    query = service.db.query(Integration).filter(
        (Integration.created_by == current_user) |
        (Integration.organization_id == service._get_user_org(current_user))
    )
    
    if integration_type:
        query = query.filter(Integration.integration_type == integration_type)
    if status:
        query = query.filter(Integration.status == status)
    
    integrations = query.order_by(desc(Integration.created_at)).offset(skip).limit(limit).all()
    return integrations

@router.get("/{integration_id}", response_model=IntegrationResponse)
async def get_integration(
    integration_id: str,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Get a specific integration"""
    
    service = IntegrationService(db)
    integration = service._get_integration_by_id(integration_id, current_user)
    if not integration:
        raise HTTPException(status_code=404, detail="Integration not found")
    
    return integration

@router.post("/{integration_id}/test")
async def test_integration(
    integration_id: str,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Test integration connection"""
    
    service = IntegrationService(db)
    result = await service.test_integration(integration_id, current_user)
    return result

@router.post("/{integration_id}/execute")
async def execute_operation(
    integration_id: str,
    operation: str,
    params: Dict[str, Any] = {},
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Execute operation on integration"""
    
    service = IntegrationService(db)
    result = await service.execute_integration_operation(integration_id, operation, params, current_user)
    return result

@router.post("/{integration_id}/webhooks", response_model=WebhookResponse)
async def create_webhook(
    integration_id: str,
    webhook_data: WebhookCreate,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Create a webhook"""
    
    service = IntegrationService(db)
    webhook = await service.create_webhook(integration_id, webhook_data, current_user)
    return webhook

@router.get("/{integration_id}/webhooks", response_model=List[WebhookResponse])
async def get_webhooks(
    integration_id: str,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Get webhooks for integration"""
    
    service = IntegrationService(db)
    integration = service._get_integration_by_id(integration_id, current_user)
    if not integration:
        raise HTTPException(status_code=404, detail="Integration not found")
    
    webhooks = service.db.query(Webhook).filter(Webhook.integration_id == integration.id).all()
    return webhooks

@router.post("/webhooks/trigger")
async def trigger_webhook_event(
    event_type: WebhookEvent,
    event_data: Dict[str, Any],
    filters: Optional[Dict[str, Any]] = None,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Trigger webhook event (for testing)"""
    
    service = IntegrationService(db)
    await service.trigger_webhook(event_type, event_data, filters)
    return {"message": "Webhook event triggered"}

@router.delete("/{integration_id}")
async def delete_integration(
    integration_id: str,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Delete an integration"""
    
    service = IntegrationService(db)
    integration = service._get_integration_by_id(integration_id, current_user)
    if not integration:
        raise HTTPException(status_code=404, detail="Integration not found")
    
    if integration.created_by != current_user:
        raise HTTPException(status_code=403, detail="Permission denied")
    
    # Delete integration and related data
    service.db.delete(integration)
    service.db.commit()
    
    return {"message": "Integration deleted successfully"}