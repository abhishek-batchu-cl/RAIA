# Model Versioning and Experiment Tracking Service
import os
import json
import uuid
import hashlib
import shutil
import pickle
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from sqlalchemy.orm import Session
from sqlalchemy import Column, String, DateTime, Text, Boolean, JSON, UUID as PG_UUID, Integer, Float, ForeignKey, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import logging
import numpy as np
import pandas as pd

# Version control
import git
from packaging import version

# ML metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Model serialization
import joblib
import torch
import tensorflow as tf

# Experiment tracking
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

Base = declarative_base()
logger = logging.getLogger(__name__)

@dataclass
class VersionConfig:
    """Configuration for a model version"""
    version_number: str
    version_tag: str
    commit_message: str
    parent_version: Optional[str] = None
    model_file_path: str = None
    training_config: Dict[str, Any] = None
    dataset_info: Dict[str, Any] = None
    is_production: bool = False

@dataclass
class ExperimentConfig:
    """Configuration for an experiment"""
    name: str
    description: str
    objective_metric: str
    objective_direction: str  # 'maximize' or 'minimize'
    tags: List[str] = None
    hyperparameter_space: Dict[str, Any] = None
    max_runs: int = 100
    timeout_hours: int = 24

@dataclass
class RunConfig:
    """Configuration for an experiment run"""
    experiment_id: str
    hyperparameters: Dict[str, Any]
    dataset_config: Dict[str, Any]
    model_type: str
    notes: Optional[str] = None

@dataclass
class VersionComparison:
    """Comparison between model versions"""
    versions: List[str]
    metrics_comparison: Dict[str, List[float]]
    feature_differences: Dict[str, List[str]]
    hyperparameter_differences: Dict[str, List[Any]]
    performance_delta: Dict[str, float]
    recommendation: str

class ModelVersion(Base):
    """Store model versions"""
    __tablename__ = "model_versions"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    version_id = Column(String(255), unique=True, nullable=False)
    model_id = Column(String(255), nullable=False)
    version_number = Column(String(50), nullable=False)
    version_tag = Column(String(100))
    status = Column(String(50), default='active')  # active, archived, deprecated, production
    
    # Version control
    parent_version = Column(String(255))
    commit_message = Column(Text)
    commit_hash = Column(String(100))
    
    # Model artifacts
    model_file_path = Column(String(1000))
    model_file_size = Column(Integer)
    model_checksum = Column(String(255))
    framework = Column(String(100))
    framework_version = Column(String(50))
    
    # Training configuration
    training_config = Column(JSON)
    features = Column(JSON)
    target_variable = Column(String(255))
    
    # Performance metrics
    performance_metrics = Column(JSON)
    validation_metrics = Column(JSON)
    test_metrics = Column(JSON)
    
    # Dataset information
    dataset_info = Column(JSON)
    data_version = Column(String(100))
    
    # Deployment info
    deployment_status = Column(String(50))
    deployment_endpoint = Column(String(500))
    deployed_at = Column(DateTime)
    
    # Metadata
    created_by = Column(String(255))
    organization_id = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    experiments = relationship("ExperimentRun", back_populates="model_version")

class Experiment(Base):
    """Store experiments"""
    __tablename__ = "experiments"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    experiment_id = Column(String(255), unique=True, nullable=False)
    name = Column(String(500), nullable=False)
    description = Column(Text)
    
    # Experiment configuration
    status = Column(String(50), default='created')  # created, running, completed, failed, paused
    objective_metric = Column(String(100))
    objective_direction = Column(String(20))  # maximize, minimize
    tags = Column(JSON)
    
    # Tracking
    total_runs = Column(Integer, default=0)
    completed_runs = Column(Integer, default=0)
    failed_runs = Column(Integer, default=0)
    best_run_id = Column(String(255))
    baseline_run_id = Column(String(255))
    
    # Results
    best_metric_value = Column(Float)
    avg_metric_value = Column(Float)
    metric_std_dev = Column(Float)
    
    # Timing
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    last_run_at = Column(DateTime)
    
    # User info
    created_by = Column(String(255))
    organization_id = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    runs = relationship("ExperimentRun", back_populates="experiment")

class ExperimentRun(Base):
    """Store individual experiment runs"""
    __tablename__ = "experiment_runs"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(String(255), unique=True, nullable=False)
    experiment_id = Column(String(255), ForeignKey('experiments.experiment_id'))
    run_number = Column(Integer)
    
    # Run configuration
    status = Column(String(50), default='created')  # created, running, completed, failed
    hyperparameters = Column(JSON)
    model_type = Column(String(100))
    
    # Metrics
    metrics = Column(JSON)
    system_metrics = Column(JSON)  # CPU, memory, GPU usage
    
    # Artifacts
    model_path = Column(String(1000))
    logs_path = Column(String(1000))
    plots_paths = Column(JSON)
    
    # Version control
    git_commit = Column(String(100))
    git_branch = Column(String(100))
    git_dirty = Column(Boolean)
    
    # Dataset
    dataset_config = Column(JSON)
    dataset_hash = Column(String(255))
    
    # Timing
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    duration_seconds = Column(Float)
    
    # Notes
    notes = Column(Text)
    error_message = Column(Text)
    
    # Model version link
    model_version_id = Column(String(255), ForeignKey('model_versions.version_id'))
    
    # Relationships
    experiment = relationship("Experiment", back_populates="runs")
    model_version = relationship("ModelVersion", back_populates="experiments")

class ExperimentTrackingService:
    """Service for model versioning and experiment tracking"""
    
    def __init__(self, db_session: Session = None, 
                 model_storage_path: str = "/tmp/raia_models",
                 experiment_storage_path: str = "/tmp/raia_experiments"):
        self.db = db_session
        self.model_storage_path = model_storage_path
        self.experiment_storage_path = experiment_storage_path
        
        # Ensure directories exist
        os.makedirs(model_storage_path, exist_ok=True)
        os.makedirs(experiment_storage_path, exist_ok=True)
        
        # MLflow integration
        if MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(f"file://{experiment_storage_path}/mlflow")

    async def create_model_version(self, model_id: str, config: VersionConfig,
                                 model_object: Any = None,
                                 metrics: Dict[str, float] = None,
                                 created_by: str = None,
                                 organization_id: str = None) -> Dict[str, Any]:
        """Create a new model version"""
        
        version_id = f"v_{uuid.uuid4().hex[:8]}"
        
        # Validate version number
        if not self._is_valid_version(config.version_number):
            return {
                'success': False,
                'error': 'Invalid version number format. Use semantic versioning (e.g., v1.2.3)'
            }
        
        # Check if version already exists
        if self.db:
            existing = self.db.query(ModelVersion).filter(
                ModelVersion.model_id == model_id,
                ModelVersion.version_number == config.version_number
            ).first()
            
            if existing:
                return {
                    'success': False,
                    'error': f'Version {config.version_number} already exists for this model'
                }
        
        # Save model file if provided
        model_file_path = None
        model_file_size = None
        model_checksum = None
        
        if model_object:
            model_file_path = os.path.join(
                self.model_storage_path,
                model_id,
                config.version_number,
                'model.pkl'
            )
            
            os.makedirs(os.path.dirname(model_file_path), exist_ok=True)
            
            # Save model
            joblib.dump(model_object, model_file_path)
            
            # Calculate file size and checksum
            model_file_size = os.path.getsize(model_file_path)
            model_checksum = self._calculate_checksum(model_file_path)
        
        # Get git information
        git_info = self._get_git_info()
        
        # Detect framework
        framework_info = self._detect_framework(model_object) if model_object else {}
        
        # Create version record
        version_record = ModelVersion(
            version_id=version_id,
            model_id=model_id,
            version_number=config.version_number,
            version_tag=config.version_tag,
            parent_version=config.parent_version,
            commit_message=config.commit_message,
            commit_hash=git_info.get('commit_hash'),
            model_file_path=model_file_path,
            model_file_size=model_file_size,
            model_checksum=model_checksum,
            framework=framework_info.get('framework'),
            framework_version=framework_info.get('version'),
            training_config=config.training_config,
            features=config.training_config.get('features') if config.training_config else None,
            target_variable=config.training_config.get('target') if config.training_config else None,
            performance_metrics=metrics,
            dataset_info=config.dataset_info,
            created_by=created_by,
            organization_id=organization_id,
            status='production' if config.is_production else 'active'
        )
        
        if self.db:
            # If marking as production, update other versions
            if config.is_production:
                self.db.query(ModelVersion).filter(
                    ModelVersion.model_id == model_id,
                    ModelVersion.status == 'production'
                ).update({'status': 'active'})
            
            self.db.add(version_record)
            self.db.commit()
        
        return {
            'success': True,
            'version_id': version_id,
            'version_number': config.version_number,
            'model_path': model_file_path,
            'checksum': model_checksum
        }

    async def create_experiment(self, config: ExperimentConfig,
                              created_by: str = None,
                              organization_id: str = None) -> Dict[str, Any]:
        """Create a new experiment"""
        
        experiment_id = f"exp_{uuid.uuid4().hex[:8]}"
        
        # Create experiment record
        experiment_record = Experiment(
            experiment_id=experiment_id,
            name=config.name,
            description=config.description,
            objective_metric=config.objective_metric,
            objective_direction=config.objective_direction,
            tags=config.tags,
            status='created',
            created_by=created_by,
            organization_id=organization_id
        )
        
        if self.db:
            self.db.add(experiment_record)
            self.db.commit()
        
        # Create MLflow experiment if available
        if MLFLOW_AVAILABLE:
            mlflow.create_experiment(
                experiment_id,
                artifact_location=os.path.join(self.experiment_storage_path, experiment_id)
            )
        
        return {
            'success': True,
            'experiment_id': experiment_id,
            'message': f'Experiment "{config.name}" created successfully'
        }

    async def run_experiment(self, run_config: RunConfig,
                           model_train_func: callable,
                           data: Any) -> Dict[str, Any]:
        """Run a single experiment"""
        
        run_id = f"run_{uuid.uuid4().hex[:8]}"
        start_time = datetime.utcnow()
        
        # Get experiment
        experiment = None
        if self.db:
            experiment = self.db.query(Experiment).filter(
                Experiment.experiment_id == run_config.experiment_id
            ).first()
            
            if not experiment:
                return {
                    'success': False,
                    'error': f'Experiment {run_config.experiment_id} not found'
                }
            
            # Update experiment status
            experiment.status = 'running'
            experiment.total_runs += 1
            
            # Get run number
            run_number = experiment.total_runs
        else:
            run_number = 1
        
        # Create run record
        run_record = ExperimentRun(
            run_id=run_id,
            experiment_id=run_config.experiment_id,
            run_number=run_number,
            status='running',
            hyperparameters=run_config.hyperparameters,
            model_type=run_config.model_type,
            dataset_config=run_config.dataset_config,
            started_at=start_time,
            notes=run_config.notes
        )
        
        if self.db:
            self.db.add(run_record)
            self.db.commit()
        
        # Start MLflow run if available
        if MLFLOW_AVAILABLE:
            mlflow.set_experiment(run_config.experiment_id)
            mlflow.start_run(run_name=f"run_{run_number}")
            
            # Log hyperparameters
            for key, value in run_config.hyperparameters.items():
                mlflow.log_param(key, value)
        
        try:
            # Monitor system metrics
            system_metrics = self._get_system_metrics()
            
            # Train model
            logger.info(f"Starting training for run {run_id}")
            model, metrics = await model_train_func(
                data=data,
                hyperparameters=run_config.hyperparameters,
                model_type=run_config.model_type
            )
            
            # Calculate additional metrics if needed
            if 'accuracy' not in metrics and hasattr(model, 'predict'):
                # Assume classification task
                y_pred = model.predict(data.get('X_test'))
                y_true = data.get('y_test')
                
                metrics.update({
                    'accuracy': accuracy_score(y_true, y_pred),
                    'precision': precision_score(y_true, y_pred, average='weighted'),
                    'recall': recall_score(y_true, y_pred, average='weighted'),
                    'f1_score': f1_score(y_true, y_pred, average='weighted')
                })
            
            # Save model
            model_path = os.path.join(
                self.experiment_storage_path,
                run_config.experiment_id,
                run_id,
                'model.pkl'
            )
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            joblib.dump(model, model_path)
            
            # Log metrics to MLflow
            if MLFLOW_AVAILABLE:
                for key, value in metrics.items():
                    mlflow.log_metric(key, value)
                mlflow.log_artifact(model_path)
            
            # Update run record
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            if self.db:
                run_record.status = 'completed'
                run_record.metrics = metrics
                run_record.system_metrics = system_metrics
                run_record.model_path = model_path
                run_record.completed_at = end_time
                run_record.duration_seconds = duration
                
                # Update experiment
                self._update_experiment_stats(experiment, metrics)
                
                self.db.commit()
            
            return {
                'success': True,
                'run_id': run_id,
                'metrics': metrics,
                'duration_seconds': duration,
                'model_path': model_path
            }
            
        except Exception as e:
            logger.error(f"Run {run_id} failed: {str(e)}")
            
            if self.db:
                run_record.status = 'failed'
                run_record.error_message = str(e)
                run_record.completed_at = datetime.utcnow()
                
                experiment.failed_runs += 1
                
                self.db.commit()
            
            if MLFLOW_AVAILABLE:
                mlflow.end_run(status='FAILED')
            
            return {
                'success': False,
                'run_id': run_id,
                'error': str(e)
            }
        
        finally:
            if MLFLOW_AVAILABLE:
                mlflow.end_run()

    async def compare_versions(self, version_ids: List[str]) -> VersionComparison:
        """Compare multiple model versions"""
        
        if len(version_ids) < 2:
            raise ValueError("At least 2 versions required for comparison")
        
        versions = []
        if self.db:
            versions = self.db.query(ModelVersion).filter(
                ModelVersion.version_id.in_(version_ids)
            ).all()
        
        if len(versions) != len(version_ids):
            raise ValueError("Some versions not found")
        
        # Extract metrics
        metrics_comparison = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc']:
            values = []
            for v in versions:
                if v.performance_metrics and metric in v.performance_metrics:
                    values.append(v.performance_metrics[metric])
                else:
                    values.append(None)
            metrics_comparison[metric] = values
        
        # Extract features
        feature_differences = {}
        all_features = set()
        for v in versions:
            if v.features:
                all_features.update(v.features)
        
        for feature in all_features:
            feature_differences[feature] = [
                feature in (v.features or []) for v in versions
            ]
        
        # Extract hyperparameters
        hyperparameter_differences = {}
        all_params = set()
        for v in versions:
            if v.training_config and 'hyperparameters' in v.training_config:
                all_params.update(v.training_config['hyperparameters'].keys())
        
        for param in all_params:
            values = []
            for v in versions:
                if v.training_config and 'hyperparameters' in v.training_config:
                    values.append(v.training_config['hyperparameters'].get(param))
                else:
                    values.append(None)
            hyperparameter_differences[param] = values
        
        # Calculate performance delta
        performance_delta = {}
        baseline_idx = 0  # Use first version as baseline
        
        for metric, values in metrics_comparison.items():
            if values[baseline_idx] is not None:
                deltas = []
                for i, val in enumerate(values):
                    if val is not None:
                        delta = ((val - values[baseline_idx]) / values[baseline_idx]) * 100
                        deltas.append(delta)
                    else:
                        deltas.append(None)
                performance_delta[metric] = deltas
        
        # Generate recommendation
        best_version_idx = 0
        best_score = 0
        
        # Simple scoring based on key metrics
        for i, v in enumerate(versions):
            score = 0
            if v.performance_metrics:
                score += v.performance_metrics.get('accuracy', 0) * 0.3
                score += v.performance_metrics.get('f1_score', 0) * 0.3
                score += v.performance_metrics.get('auc', 0) * 0.4
            
            if score > best_score:
                best_score = score
                best_version_idx = i
        
        recommendation = f"Version {versions[best_version_idx].version_number} shows the best overall performance"
        
        return VersionComparison(
            versions=[v.version_id for v in versions],
            metrics_comparison=metrics_comparison,
            feature_differences=feature_differences,
            hyperparameter_differences=hyperparameter_differences,
            performance_delta=performance_delta,
            recommendation=recommendation
        )

    async def promote_to_production(self, version_id: str) -> Dict[str, Any]:
        """Promote a model version to production"""
        
        if not self.db:
            return {'success': False, 'error': 'Database not available'}
        
        # Get version
        version = self.db.query(ModelVersion).filter(
            ModelVersion.version_id == version_id
        ).first()
        
        if not version:
            return {'success': False, 'error': 'Version not found'}
        
        # Update current production version
        self.db.query(ModelVersion).filter(
            ModelVersion.model_id == version.model_id,
            ModelVersion.status == 'production'
        ).update({'status': 'active'})
        
        # Set new production version
        version.status = 'production'
        version.deployment_status = 'deployed'
        version.deployed_at = datetime.utcnow()
        
        self.db.commit()
        
        return {
            'success': True,
            'message': f'Version {version.version_number} promoted to production'
        }

    async def get_version_history(self, model_id: str) -> List[Dict[str, Any]]:
        """Get version history for a model"""
        
        if not self.db:
            return []
        
        versions = self.db.query(ModelVersion).filter(
            ModelVersion.model_id == model_id
        ).order_by(ModelVersion.created_at.desc()).all()
        
        history = []
        for v in versions:
            history.append({
                'version_id': v.version_id,
                'version_number': v.version_number,
                'version_tag': v.version_tag,
                'status': v.status,
                'created_at': v.created_at.isoformat(),
                'created_by': v.created_by,
                'commit_message': v.commit_message,
                'performance_metrics': v.performance_metrics,
                'is_production': v.status == 'production'
            })
        
        return history

    async def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """Get comprehensive experiment results"""
        
        if not self.db:
            return {}
        
        experiment = self.db.query(Experiment).filter(
            Experiment.experiment_id == experiment_id
        ).first()
        
        if not experiment:
            return {'error': 'Experiment not found'}
        
        runs = self.db.query(ExperimentRun).filter(
            ExperimentRun.experiment_id == experiment_id
        ).order_by(ExperimentRun.started_at.desc()).all()
        
        # Aggregate results
        results = {
            'experiment_id': experiment.experiment_id,
            'name': experiment.name,
            'status': experiment.status,
            'objective': {
                'metric': experiment.objective_metric,
                'direction': experiment.objective_direction
            },
            'summary': {
                'total_runs': experiment.total_runs,
                'completed_runs': experiment.completed_runs,
                'failed_runs': experiment.failed_runs,
                'best_metric_value': experiment.best_metric_value,
                'avg_metric_value': experiment.avg_metric_value,
                'metric_std_dev': experiment.metric_std_dev
            },
            'runs': []
        }
        
        for run in runs:
            results['runs'].append({
                'run_id': run.run_id,
                'run_number': run.run_number,
                'status': run.status,
                'metrics': run.metrics,
                'hyperparameters': run.hyperparameters,
                'duration_seconds': run.duration_seconds,
                'started_at': run.started_at.isoformat() if run.started_at else None
            })
        
        # Hyperparameter importance analysis
        if len(runs) > 5:  # Need sufficient runs
            importance = self._analyze_hyperparameter_importance(runs, experiment.objective_metric)
            results['hyperparameter_importance'] = importance
        
        return results

    # Helper methods
    def _is_valid_version(self, version_number: str) -> bool:
        """Validate version number format"""
        try:
            # Remove 'v' prefix if present
            if version_number.startswith('v'):
                version_number = version_number[1:]
            
            # Parse version
            version.parse(version_number)
            return True
        except:
            return False

    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate file checksum"""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _get_git_info(self) -> Dict[str, Any]:
        """Get git repository information"""
        try:
            repo = git.Repo(search_parent_directories=True)
            return {
                'commit_hash': repo.head.object.hexsha,
                'branch': repo.active_branch.name,
                'dirty': repo.is_dirty()
            }
        except:
            return {}

    def _detect_framework(self, model_object: Any) -> Dict[str, str]:
        """Detect ML framework from model object"""
        
        if hasattr(model_object, '_estimator_type'):  # Scikit-learn
            import sklearn
            return {
                'framework': 'scikit-learn',
                'version': sklearn.__version__
            }
        elif hasattr(model_object, 'layers'):  # Keras/TensorFlow
            return {
                'framework': 'tensorflow',
                'version': tf.__version__
            }
        elif hasattr(model_object, 'parameters'):  # PyTorch
            return {
                'framework': 'pytorch',
                'version': torch.__version__
            }
        else:
            return {
                'framework': 'unknown',
                'version': 'unknown'
            }

    def _get_system_metrics(self) -> Dict[str, float]:
        """Get current system metrics"""
        import psutil
        
        return {
            'cpu_usage': psutil.cpu_percent(interval=1),
            'memory_usage': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent
        }

    def _update_experiment_stats(self, experiment: Experiment, metrics: Dict[str, float]):
        """Update experiment statistics"""
        
        objective_metric = experiment.objective_metric
        if objective_metric not in metrics:
            return
        
        metric_value = metrics[objective_metric]
        
        # Update best run
        if experiment.best_metric_value is None:
            experiment.best_metric_value = metric_value
            experiment.best_run_id = experiment.runs[-1].run_id
        else:
            if experiment.objective_direction == 'maximize':
                if metric_value > experiment.best_metric_value:
                    experiment.best_metric_value = metric_value
                    experiment.best_run_id = experiment.runs[-1].run_id
            else:  # minimize
                if metric_value < experiment.best_metric_value:
                    experiment.best_metric_value = metric_value
                    experiment.best_run_id = experiment.runs[-1].run_id
        
        # Update averages
        completed_runs = [r for r in experiment.runs if r.status == 'completed']
        if completed_runs:
            values = []
            for run in completed_runs:
                if run.metrics and objective_metric in run.metrics:
                    values.append(run.metrics[objective_metric])
            
            if values:
                experiment.avg_metric_value = np.mean(values)
                experiment.metric_std_dev = np.std(values)
        
        experiment.completed_runs = len(completed_runs)
        experiment.last_run_at = datetime.utcnow()

    def _analyze_hyperparameter_importance(self, runs: List[ExperimentRun], 
                                         objective_metric: str) -> Dict[str, float]:
        """Analyze hyperparameter importance using correlation"""
        
        # Extract data
        data = []
        for run in runs:
            if run.status == 'completed' and run.metrics and objective_metric in run.metrics:
                row = {'metric': run.metrics[objective_metric]}
                row.update(run.hyperparameters or {})
                data.append(row)
        
        if len(data) < 5:
            return {}
        
        df = pd.DataFrame(data)
        
        # Calculate correlations
        correlations = {}
        metric_col = df['metric']
        
        for col in df.columns:
            if col != 'metric':
                try:
                    # Convert to numeric if possible
                    if df[col].dtype == 'object':
                        df[col] = pd.Categorical(df[col]).codes
                    
                    corr = metric_col.corr(df[col])
                    if not np.isnan(corr):
                        correlations[col] = abs(corr)
                except:
                    pass
        
        # Normalize to 0-1
        if correlations:
            max_corr = max(correlations.values())
            if max_corr > 0:
                correlations = {k: v/max_corr for k, v in correlations.items()}
        
        return correlations

# Factory function
def create_experiment_tracking_service(db_session: Session = None,
                                     model_storage_path: str = "/tmp/raia_models",
                                     experiment_storage_path: str = "/tmp/raia_experiments") -> ExperimentTrackingService:
    """Create and return an ExperimentTrackingService instance"""
    return ExperimentTrackingService(db_session, model_storage_path, experiment_storage_path)