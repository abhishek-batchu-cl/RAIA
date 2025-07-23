# Model Management API
import os
import uuid
import json
import shutil
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, validator
from sqlalchemy.orm import Session
from sqlalchemy import Column, String, DateTime, Text, Boolean, JSON, UUID as PG_UUID, Integer, Float, ForeignKey, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

# ML libraries
import joblib
import pickle
import numpy as np
import pandas as pd
try:
    import torch
    import tensorflow as tf
    import onnx
    PYTORCH_AVAILABLE = hasattr(torch, 'save')
    TENSORFLOW_AVAILABLE = hasattr(tf, 'saved_model')
    ONNX_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    TENSORFLOW_AVAILABLE = False
    ONNX_AVAILABLE = False

# Model validation and analysis
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score

Base = declarative_base()
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/models", tags=["models"])

# Database Models
class Model(Base):
    """Model metadata and tracking"""
    __tablename__ = "models"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    display_name = Column(String(255))
    description = Column(Text)
    
    # Model details
    model_type = Column(String(100), nullable=False)  # classification, regression, clustering, etc.
    algorithm = Column(String(100))  # random_forest, neural_network, etc.
    framework = Column(String(50))  # sklearn, pytorch, tensorflow, etc.
    version = Column(String(50), default="1.0.0")
    
    # File information
    file_path = Column(String(1000))
    file_size_bytes = Column(Integer)
    file_format = Column(String(50))  # pkl, joblib, onnx, pt, h5, etc.
    checksum = Column(String(128))
    
    # Performance metrics
    metrics = Column(JSON)  # {"accuracy": 0.95, "precision": 0.92, ...}
    hyperparameters = Column(JSON)
    feature_names = Column(JSON)  # List of feature names
    target_names = Column(JSON)  # List of target/class names
    
    # Training information
    training_dataset_id = Column(String(255))
    training_start_time = Column(DateTime)
    training_end_time = Column(DateTime)
    training_duration_seconds = Column(Integer)
    
    # Status and deployment
    status = Column(String(50), default="draft")  # draft, trained, validated, deployed, archived
    is_active = Column(Boolean, default=True)
    deployment_status = Column(String(50), default="not_deployed")  # not_deployed, staging, production
    deployment_url = Column(String(1000))
    
    # Metadata
    tags = Column(JSON)  # List of tags
    metadata = Column(JSON)  # Additional metadata
    
    # Ownership and permissions
    created_by = Column(String(255), nullable=False)
    organization_id = Column(String(255))
    is_public = Column(Boolean, default=False)
    access_level = Column(String(50), default="private")  # private, organization, public
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    deployed_at = Column(DateTime)
    archived_at = Column(DateTime)
    
    # Relationships
    experiments = relationship("ModelExperiment", back_populates="model")
    evaluations = relationship("ModelEvaluation", back_populates="model")
    deployments = relationship("ModelDeployment", back_populates="model")

class ModelExperiment(Base):
    """Track model experiments and runs"""
    __tablename__ = "model_experiments"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(PG_UUID(as_uuid=True), ForeignKey('models.id'), nullable=False)
    experiment_name = Column(String(255), nullable=False)
    run_id = Column(String(255), unique=True)
    
    # Experiment configuration
    config = Column(JSON)  # Hyperparameters, data splits, etc.
    environment = Column(JSON)  # Python version, library versions, etc.
    
    # Results
    metrics = Column(JSON)
    artifacts = Column(JSON)  # Paths to saved artifacts
    logs = Column(Text)
    
    # Status
    status = Column(String(50), default="running")  # running, completed, failed
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime)
    
    # Ownership
    created_by = Column(String(255), nullable=False)
    
    # Relationships
    model = relationship("Model", back_populates="experiments")

class ModelEvaluation(Base):
    """Model evaluation results"""
    __tablename__ = "model_evaluations"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(PG_UUID(as_uuid=True), ForeignKey('models.id'), nullable=False)
    
    # Evaluation details
    evaluation_type = Column(String(100))  # validation, test, production
    dataset_id = Column(String(255))
    dataset_name = Column(String(255))
    
    # Metrics
    metrics = Column(JSON)
    confusion_matrix = Column(JSON)
    feature_importance = Column(JSON)
    
    # Evaluation metadata
    evaluation_config = Column(JSON)
    sample_size = Column(Integer)
    evaluation_time = Column(DateTime, default=datetime.utcnow)
    
    # Ownership
    created_by = Column(String(255), nullable=False)
    
    # Relationships
    model = relationship("Model", back_populates="evaluations")

class ModelDeployment(Base):
    """Model deployment tracking"""
    __tablename__ = "model_deployments"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(PG_UUID(as_uuid=True), ForeignKey('models.id'), nullable=False)
    
    # Deployment details
    environment = Column(String(100))  # staging, production
    deployment_type = Column(String(100))  # api, batch, edge
    endpoint_url = Column(String(1000))
    
    # Configuration
    config = Column(JSON)
    scaling_config = Column(JSON)
    
    # Status
    status = Column(String(50), default="deploying")  # deploying, active, inactive, failed
    health_status = Column(String(50), default="unknown")  # healthy, unhealthy, unknown
    
    # Performance tracking
    request_count = Column(Integer, default=0)
    avg_response_time_ms = Column(Float, default=0.0)
    error_rate = Column(Float, default=0.0)
    last_health_check = Column(DateTime)
    
    # Timestamps
    deployed_at = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Ownership
    created_by = Column(String(255), nullable=False)
    
    # Relationships
    model = relationship("Model", back_populates="deployments")

# Pydantic Models
class ModelCreate(BaseModel):
    name: str
    display_name: Optional[str] = None
    description: Optional[str] = None
    model_type: str
    algorithm: Optional[str] = None
    framework: Optional[str] = None
    version: str = "1.0.0"
    tags: Optional[List[str]] = []
    metadata: Optional[Dict[str, Any]] = {}
    is_public: bool = False
    access_level: str = "private"

class ModelUpdate(BaseModel):
    display_name: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    status: Optional[str] = None
    is_active: Optional[bool] = None
    is_public: Optional[bool] = None
    access_level: Optional[str] = None

class ModelResponse(BaseModel):
    id: str
    name: str
    display_name: Optional[str]
    description: Optional[str]
    model_type: str
    algorithm: Optional[str]
    framework: Optional[str]
    version: str
    status: str
    deployment_status: str
    metrics: Optional[Dict[str, Any]]
    tags: Optional[List[str]]
    created_by: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True

class ModelExperimentCreate(BaseModel):
    experiment_name: str
    config: Dict[str, Any]
    environment: Optional[Dict[str, Any]] = {}

class ModelEvaluationCreate(BaseModel):
    evaluation_type: str = "validation"
    dataset_id: Optional[str] = None
    dataset_name: Optional[str] = None
    evaluation_config: Optional[Dict[str, Any]] = {}

class ModelDeploymentCreate(BaseModel):
    environment: str
    deployment_type: str = "api"
    config: Optional[Dict[str, Any]] = {}
    scaling_config: Optional[Dict[str, Any]] = {}

# Dependency injection
def get_db():
    # This would be implemented to return the database session
    pass

def get_current_user():
    # This would be implemented to return the current user
    return "current_user_id"

# Model Management Service
class ModelManagementService:
    """Service for managing ML models"""
    
    def __init__(self, db: Session, storage_path: str = "/tmp/raia_models"):
        self.db = db
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Model format handlers
        self.format_handlers = {
            'pickle': self._handle_pickle,
            'joblib': self._handle_joblib,
            'onnx': self._handle_onnx,
            'pytorch': self._handle_pytorch,
            'tensorflow': self._handle_tensorflow
        }
    
    async def create_model(self, model_data: ModelCreate, user_id: str) -> Model:
        """Create a new model record"""
        
        model = Model(
            name=model_data.name,
            display_name=model_data.display_name or model_data.name,
            description=model_data.description,
            model_type=model_data.model_type,
            algorithm=model_data.algorithm,
            framework=model_data.framework,
            version=model_data.version,
            tags=model_data.tags,
            metadata=model_data.metadata,
            is_public=model_data.is_public,
            access_level=model_data.access_level,
            created_by=user_id,
            organization_id=self._get_user_org(user_id)
        )
        
        self.db.add(model)
        self.db.commit()
        self.db.refresh(model)
        
        logger.info(f"Created model {model.name} (ID: {model.id})")
        return model
    
    async def upload_model_file(self, model_id: str, file: UploadFile, user_id: str) -> Dict[str, Any]:
        """Upload and store model file"""
        
        model = self._get_model_by_id(model_id, user_id)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Create model directory
        model_dir = self.storage_path / str(model.id)
        model_dir.mkdir(exist_ok=True)
        
        # Save file
        file_path = model_dir / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Calculate checksum
        import hashlib
        checksum = hashlib.sha256(file_path.read_bytes()).hexdigest()
        
        # Detect file format
        file_format = self._detect_file_format(file.filename, file_path)
        
        # Update model record
        model.file_path = str(file_path)
        model.file_size_bytes = file_path.stat().st_size
        model.file_format = file_format
        model.checksum = checksum
        model.status = "uploaded"
        model.updated_at = datetime.utcnow()
        
        # Extract model metadata if possible
        try:
            metadata = await self._extract_model_metadata(file_path, file_format)
            model.hyperparameters = metadata.get('hyperparameters')
            model.feature_names = metadata.get('feature_names')
            model.target_names = metadata.get('target_names')
        except Exception as e:
            logger.warning(f"Could not extract metadata from model file: {str(e)}")
        
        self.db.commit()
        
        return {
            'success': True,
            'file_path': str(file_path),
            'file_size': file_path.stat().st_size,
            'checksum': checksum,
            'format': file_format
        }
    
    async def get_models(self, user_id: str, skip: int = 0, limit: int = 100, 
                        filters: Optional[Dict[str, Any]] = None) -> List[Model]:
        """Get models with filtering and pagination"""
        
        query = self.db.query(Model).filter(
            (Model.created_by == user_id) | 
            (Model.is_public == True) |
            (Model.organization_id == self._get_user_org(user_id))
        )
        
        # Apply filters
        if filters:
            if filters.get('model_type'):
                query = query.filter(Model.model_type == filters['model_type'])
            if filters.get('framework'):
                query = query.filter(Model.framework == filters['framework'])
            if filters.get('status'):
                query = query.filter(Model.status == filters['status'])
            if filters.get('tags'):
                # Filter by tags (JSON array contains)
                for tag in filters['tags']:
                    query = query.filter(Model.tags.contains([tag]))
        
        return query.offset(skip).limit(limit).all()
    
    async def get_model(self, model_id: str, user_id: str) -> Model:
        """Get a specific model"""
        
        model = self._get_model_by_id(model_id, user_id)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        return model
    
    async def update_model(self, model_id: str, model_data: ModelUpdate, user_id: str) -> Model:
        """Update model metadata"""
        
        model = self._get_model_by_id(model_id, user_id)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        if model.created_by != user_id:
            raise HTTPException(status_code=403, detail="Permission denied")
        
        # Update fields
        for field, value in model_data.dict(exclude_unset=True).items():
            setattr(model, field, value)
        
        model.updated_at = datetime.utcnow()
        self.db.commit()
        
        return model
    
    async def delete_model(self, model_id: str, user_id: str) -> Dict[str, Any]:
        """Delete a model and its files"""
        
        model = self._get_model_by_id(model_id, user_id)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        if model.created_by != user_id:
            raise HTTPException(status_code=403, detail="Permission denied")
        
        # Check if model is deployed
        if model.deployment_status != "not_deployed":
            raise HTTPException(
                status_code=400, 
                detail="Cannot delete deployed model. Undeploy first."
            )
        
        # Delete files
        if model.file_path:
            model_dir = Path(model.file_path).parent
            if model_dir.exists():
                shutil.rmtree(model_dir)
        
        # Delete database records (cascading)
        self.db.delete(model)
        self.db.commit()
        
        logger.info(f"Deleted model {model.name} (ID: {model.id})")
        
        return {'success': True, 'message': 'Model deleted successfully'}
    
    async def evaluate_model(self, model_id: str, eval_data: ModelEvaluationCreate, 
                           user_id: str, test_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Evaluate a model's performance"""
        
        model = self._get_model_by_id(model_id, user_id)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        if not model.file_path or not Path(model.file_path).exists():
            raise HTTPException(status_code=400, detail="Model file not found")
        
        try:
            # Load the model
            loaded_model = await self._load_model(model.file_path, model.file_format)
            
            # Prepare test data (this would come from the frontend or be loaded from dataset_id)
            if test_data is None:
                # In a real implementation, this would load from the dataset_id
                raise HTTPException(status_code=400, detail="Test data required for evaluation")
            
            X_test = np.array(test_data['features'])
            y_test = np.array(test_data['targets'])
            
            # Make predictions
            if hasattr(loaded_model, 'predict'):
                predictions = loaded_model.predict(X_test)
            else:
                raise HTTPException(status_code=400, detail="Model does not support prediction")
            
            # Calculate metrics based on model type
            metrics = {}
            if model.model_type == 'classification':
                metrics = {
                    'accuracy': float(accuracy_score(y_test, predictions)),
                    'precision': float(precision_score(y_test, predictions, average='weighted')),
                    'recall': float(recall_score(y_test, predictions, average='weighted')),
                    'f1_score': float(f1_score(y_test, predictions, average='weighted'))
                }
                
                # Confusion matrix
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(y_test, predictions).tolist()
                
            elif model.model_type == 'regression':
                metrics = {
                    'mse': float(mean_squared_error(y_test, predictions)),
                    'rmse': float(np.sqrt(mean_squared_error(y_test, predictions))),
                    'r2_score': float(r2_score(y_test, predictions))
                }
                cm = None
            
            # Save evaluation
            evaluation = ModelEvaluation(
                model_id=model.id,
                evaluation_type=eval_data.evaluation_type,
                dataset_id=eval_data.dataset_id,
                dataset_name=eval_data.dataset_name,
                metrics=metrics,
                confusion_matrix=cm,
                evaluation_config=eval_data.evaluation_config,
                sample_size=len(X_test),
                created_by=user_id
            )
            
            self.db.add(evaluation)
            
            # Update model metrics
            model.metrics = metrics
            model.updated_at = datetime.utcnow()
            
            self.db.commit()
            
            return {
                'success': True,
                'evaluation_id': str(evaluation.id),
                'metrics': metrics,
                'confusion_matrix': cm,
                'sample_size': len(X_test)
            }
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")
    
    # Private methods
    def _get_model_by_id(self, model_id: str, user_id: str) -> Optional[Model]:
        """Get model by ID with access control"""
        
        return self.db.query(Model).filter(
            Model.id == model_id,
            (
                (Model.created_by == user_id) |
                (Model.is_public == True) |
                (Model.organization_id == self._get_user_org(user_id))
            )
        ).first()
    
    def _get_user_org(self, user_id: str) -> Optional[str]:
        """Get user's organization ID"""
        # This would be implemented to return the user's organization
        return "default_org"
    
    def _detect_file_format(self, filename: str, file_path: Path) -> str:
        """Detect model file format"""
        
        extension = filename.split('.')[-1].lower()
        
        if extension in ['pkl', 'pickle']:
            return 'pickle'
        elif extension in ['joblib', 'jl']:
            return 'joblib'
        elif extension == 'onnx':
            return 'onnx'
        elif extension in ['pt', 'pth']:
            return 'pytorch'
        elif extension in ['h5', 'hdf5']:
            return 'tensorflow'
        elif extension == 'pb':
            return 'tensorflow'
        else:
            # Try to detect by content
            try:
                with open(file_path, 'rb') as f:
                    header = f.read(16)
                    if b'ONNX' in header:
                        return 'onnx'
                    elif b'tensorflow' in header.lower():
                        return 'tensorflow'
            except:
                pass
            
            return 'unknown'
    
    async def _extract_model_metadata(self, file_path: Path, file_format: str) -> Dict[str, Any]:
        """Extract metadata from model file"""
        
        metadata = {}
        
        try:
            if file_format == 'pickle':
                with open(file_path, 'rb') as f:
                    model = pickle.load(f)
            elif file_format == 'joblib':
                model = joblib.load(file_path)
            else:
                return metadata
            
            # Extract sklearn model info
            if hasattr(model, 'get_params'):
                metadata['hyperparameters'] = model.get_params()
            
            if hasattr(model, 'feature_names_in_'):
                metadata['feature_names'] = list(model.feature_names_in_)
            
            if hasattr(model, 'classes_'):
                metadata['target_names'] = list(model.classes_)
                
        except Exception as e:
            logger.warning(f"Could not extract model metadata: {str(e)}")
        
        return metadata
    
    async def _load_model(self, file_path: str, file_format: str):
        """Load model from file"""
        
        path = Path(file_path)
        
        if file_format == 'pickle':
            with open(path, 'rb') as f:
                return pickle.load(f)
        elif file_format == 'joblib':
            return joblib.load(path)
        elif file_format == 'onnx' and ONNX_AVAILABLE:
            return onnx.load(path)
        elif file_format == 'pytorch' and PYTORCH_AVAILABLE:
            return torch.load(path)
        elif file_format == 'tensorflow' and TENSORFLOW_AVAILABLE:
            return tf.saved_model.load(str(path))
        else:
            raise ValueError(f"Unsupported model format: {file_format}")

# API Endpoints
@router.post("/", response_model=ModelResponse)
async def create_model(
    model_data: ModelCreate,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Create a new model"""
    
    service = ModelManagementService(db)
    model = await service.create_model(model_data, current_user)
    return model

@router.post("/{model_id}/upload")
async def upload_model_file(
    model_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Upload model file"""
    
    service = ModelManagementService(db)
    result = await service.upload_model_file(model_id, file, current_user)
    return result

@router.get("/", response_model=List[ModelResponse])
async def get_models(
    skip: int = 0,
    limit: int = 100,
    model_type: Optional[str] = None,
    framework: Optional[str] = None,
    status: Optional[str] = None,
    tags: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Get models with filtering"""
    
    filters = {}
    if model_type:
        filters['model_type'] = model_type
    if framework:
        filters['framework'] = framework
    if status:
        filters['status'] = status
    if tags:
        filters['tags'] = tags.split(',')
    
    service = ModelManagementService(db)
    models = await service.get_models(current_user, skip, limit, filters)
    return models

@router.get("/{model_id}", response_model=ModelResponse)
async def get_model(
    model_id: str,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Get a specific model"""
    
    service = ModelManagementService(db)
    model = await service.get_model(model_id, current_user)
    return model

@router.put("/{model_id}", response_model=ModelResponse)
async def update_model(
    model_id: str,
    model_data: ModelUpdate,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Update model metadata"""
    
    service = ModelManagementService(db)
    model = await service.update_model(model_id, model_data, current_user)
    return model

@router.delete("/{model_id}")
async def delete_model(
    model_id: str,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Delete a model"""
    
    service = ModelManagementService(db)
    result = await service.delete_model(model_id, current_user)
    return result

@router.post("/{model_id}/evaluate")
async def evaluate_model(
    model_id: str,
    eval_data: ModelEvaluationCreate,
    test_data: Optional[Dict[str, Any]] = None,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Evaluate model performance"""
    
    service = ModelManagementService(db)
    result = await service.evaluate_model(model_id, eval_data, current_user, test_data)
    return result

@router.get("/{model_id}/download")
async def download_model(
    model_id: str,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Download model file"""
    
    service = ModelManagementService(db)
    model = await service.get_model(model_id, current_user)
    
    if not model.file_path or not Path(model.file_path).exists():
        raise HTTPException(status_code=404, detail="Model file not found")
    
    return FileResponse(
        path=model.file_path,
        filename=f"{model.name}_{model.version}.{model.file_format}",
        media_type='application/octet-stream'
    )

@router.get("/{model_id}/experiments")
async def get_model_experiments(
    model_id: str,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Get model experiments"""
    
    service = ModelManagementService(db)
    model = await service.get_model(model_id, current_user)
    
    return {
        'model_id': model_id,
        'experiments': [
            {
                'id': str(exp.id),
                'experiment_name': exp.experiment_name,
                'run_id': exp.run_id,
                'status': exp.status,
                'metrics': exp.metrics,
                'start_time': exp.start_time,
                'end_time': exp.end_time
            }
            for exp in model.experiments
        ]
    }

@router.get("/{model_id}/evaluations")
async def get_model_evaluations(
    model_id: str,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Get model evaluations"""
    
    service = ModelManagementService(db)
    model = await service.get_model(model_id, current_user)
    
    return {
        'model_id': model_id,
        'evaluations': [
            {
                'id': str(eval.id),
                'evaluation_type': eval.evaluation_type,
                'metrics': eval.metrics,
                'confusion_matrix': eval.confusion_matrix,
                'sample_size': eval.sample_size,
                'evaluation_time': eval.evaluation_time
            }
            for eval in model.evaluations
        ]
    }

@router.get("/{model_id}/deployments")
async def get_model_deployments(
    model_id: str,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Get model deployments"""
    
    service = ModelManagementService(db)
    model = await service.get_model(model_id, current_user)
    
    return {
        'model_id': model_id,
        'deployments': [
            {
                'id': str(dep.id),
                'environment': dep.environment,
                'status': dep.status,
                'endpoint_url': dep.endpoint_url,
                'deployed_at': dep.deployed_at,
                'request_count': dep.request_count,
                'avg_response_time_ms': dep.avg_response_time_ms,
                'error_rate': dep.error_rate
            }
            for dep in model.deployments
        ]
    }