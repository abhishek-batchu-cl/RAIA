# Experiment Tracking API
import os
import uuid
import json
import shutil
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, validator
from sqlalchemy.orm import Session
from sqlalchemy import Column, String, DateTime, Text, Boolean, JSON, UUID as PG_UUID, Integer, Float, ForeignKey, func, desc
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

# ML experiment tracking
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

# Plotting and visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder

# Scientific computing
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import learning_curve, validation_curve

Base = declarative_base()
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/experiments", tags=["experiments"])

# Database Models
class Experiment(Base):
    """Experiment tracking and metadata"""
    __tablename__ = "experiments"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    
    # Experiment configuration
    experiment_type = Column(String(100))  # training, hyperparameter_tuning, comparison, etc.
    objective = Column(String(255))  # What we're trying to optimize
    dataset_id = Column(String(255))
    dataset_name = Column(String(255))
    
    # Status and lifecycle
    status = Column(String(50), default="created")  # created, running, completed, failed, cancelled
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    duration_seconds = Column(Integer)
    
    # Configuration and environment
    config = Column(JSON)  # Experiment configuration
    environment = Column(JSON)  # Environment details (Python version, packages, etc.)
    git_commit = Column(String(255))  # Git commit hash
    
    # Results summary
    best_metric_value = Column(Float)
    best_metric_name = Column(String(100))
    total_runs = Column(Integer, default=0)
    successful_runs = Column(Integer, default=0)
    failed_runs = Column(Integer, default=0)
    
    # MLflow integration
    mlflow_experiment_id = Column(String(255))  # MLflow experiment ID
    
    # Metadata
    tags = Column(JSON)
    notes = Column(Text)
    
    # Ownership
    created_by = Column(String(255), nullable=False)
    organization_id = Column(String(255))
    is_public = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    runs = relationship("ExperimentRun", back_populates="experiment", cascade="all, delete-orphan")
    comparisons = relationship("ExperimentComparison", back_populates="experiment")

class ExperimentRun(Base):
    """Individual experiment run"""
    __tablename__ = "experiment_runs"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    experiment_id = Column(PG_UUID(as_uuid=True), ForeignKey('experiments.id'), nullable=False)
    
    # Run identification
    run_name = Column(String(255))
    run_number = Column(Integer)  # Sequential run number within experiment
    mlflow_run_id = Column(String(255))  # MLflow run ID
    
    # Configuration
    parameters = Column(JSON)  # Hyperparameters and configuration
    model_config = Column(JSON)  # Model architecture/configuration
    
    # Status and execution
    status = Column(String(50), default="running")  # running, completed, failed, cancelled
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime)
    duration_seconds = Column(Integer)
    
    # Metrics and results
    metrics = Column(JSON)  # All metrics collected during the run
    final_metrics = Column(JSON)  # Final/best metrics
    step_metrics = Column(JSON)  # Step-by-step metrics for plotting
    
    # Model artifacts
    model_path = Column(String(1000))  # Path to saved model
    artifacts = Column(JSON)  # List of artifact paths
    
    # System information
    system_metrics = Column(JSON)  # CPU, memory, GPU usage
    logs = Column(Text)  # Execution logs
    error_message = Column(Text)
    
    # Evaluation results
    validation_metrics = Column(JSON)
    test_metrics = Column(JSON)
    cross_validation_scores = Column(JSON)
    
    # Feature engineering
    feature_importance = Column(JSON)
    feature_selection = Column(JSON)
    preprocessing_steps = Column(JSON)
    
    # Version control
    code_version = Column(String(255))  # Git commit, version tag, etc.
    data_version = Column(String(255))  # Dataset version
    
    # Ownership
    created_by = Column(String(255), nullable=False)
    
    # Relationships
    experiment = relationship("Experiment", back_populates="runs")

class ExperimentComparison(Base):
    """Compare multiple experiments or runs"""
    __tablename__ = "experiment_comparisons"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    
    # Comparison configuration
    experiment_id = Column(PG_UUID(as_uuid=True), ForeignKey('experiments.id'))
    run_ids = Column(JSON)  # List of run IDs being compared
    comparison_type = Column(String(100))  # metrics, hyperparameters, models, etc.
    
    # Comparison results
    results = Column(JSON)  # Structured comparison results
    visualizations = Column(JSON)  # Generated charts/plots
    insights = Column(JSON)  # AI-generated insights
    
    # Configuration
    metrics_to_compare = Column(JSON)
    comparison_config = Column(JSON)
    
    # Ownership
    created_by = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    experiment = relationship("Experiment", back_populates="comparisons")

# Pydantic Models
class ExperimentCreate(BaseModel):
    name: str
    description: Optional[str] = None
    experiment_type: str = "training"
    objective: Optional[str] = None
    dataset_id: Optional[str] = None
    dataset_name: Optional[str] = None
    config: Optional[Dict[str, Any]] = {}
    tags: Optional[List[str]] = []
    is_public: bool = False

class ExperimentUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    notes: Optional[str] = None
    tags: Optional[List[str]] = None
    is_public: Optional[bool] = None

class ExperimentRunCreate(BaseModel):
    run_name: Optional[str] = None
    parameters: Dict[str, Any]
    model_config: Optional[Dict[str, Any]] = {}
    notes: Optional[str] = None

class ExperimentRunUpdate(BaseModel):
    status: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    final_metrics: Optional[Dict[str, Any]] = None
    step_metrics: Optional[Dict[str, Any]] = None
    logs: Optional[str] = None
    error_message: Optional[str] = None

class ExperimentResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    experiment_type: str
    status: str
    total_runs: int
    successful_runs: int
    best_metric_value: Optional[float]
    best_metric_name: Optional[str]
    created_by: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True

class ExperimentRunResponse(BaseModel):
    id: str
    experiment_id: str
    run_name: Optional[str]
    run_number: int
    status: str
    parameters: Dict[str, Any]
    final_metrics: Optional[Dict[str, Any]]
    start_time: datetime
    end_time: Optional[datetime]
    duration_seconds: Optional[int]
    
    class Config:
        orm_mode = True

# Dependency injection
def get_db():
    pass

def get_current_user():
    return "current_user_id"

# Experiment Tracking Service
class ExperimentTrackingService:
    """Service for ML experiment tracking and management"""
    
    def __init__(self, db: Session, mlflow_tracking_uri: str = "file:///tmp/mlruns"):
        self.db = db
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.client = MlflowClient(tracking_uri=mlflow_tracking_uri)
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        # Storage for artifacts
        self.storage_path = Path("/tmp/raia_experiments")
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    async def create_experiment(self, experiment_data: ExperimentCreate, user_id: str) -> Experiment:
        """Create a new experiment"""
        
        # Create MLflow experiment
        mlflow_exp_name = f"{experiment_data.name}_{uuid.uuid4().hex[:8]}"
        mlflow_experiment_id = mlflow.create_experiment(mlflow_exp_name)
        
        # Create database record
        experiment = Experiment(
            name=experiment_data.name,
            description=experiment_data.description,
            experiment_type=experiment_data.experiment_type,
            objective=experiment_data.objective,
            dataset_id=experiment_data.dataset_id,
            dataset_name=experiment_data.dataset_name,
            config=experiment_data.config,
            tags=experiment_data.tags,
            is_public=experiment_data.is_public,
            mlflow_experiment_id=str(mlflow_experiment_id),
            created_by=user_id,
            organization_id=self._get_user_org(user_id)
        )
        
        self.db.add(experiment)
        self.db.commit()
        self.db.refresh(experiment)
        
        logger.info(f"Created experiment {experiment.name} (ID: {experiment.id})")
        return experiment
    
    async def start_run(self, experiment_id: str, run_data: ExperimentRunCreate, user_id: str) -> ExperimentRun:
        """Start a new experiment run"""
        
        experiment = self._get_experiment_by_id(experiment_id, user_id)
        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found")
        
        # Get next run number
        run_number = self.db.query(func.count(ExperimentRun.id)).filter(
            ExperimentRun.experiment_id == experiment.id
        ).scalar() + 1
        
        # Start MLflow run
        mlflow.set_experiment(experiment.mlflow_experiment_id)
        mlflow_run = mlflow.start_run(
            run_name=run_data.run_name or f"run_{run_number}"
        )
        
        # Log parameters to MLflow
        for key, value in run_data.parameters.items():
            mlflow.log_param(key, value)
        
        # Create database record
        run = ExperimentRun(
            experiment_id=experiment.id,
            run_name=run_data.run_name or f"run_{run_number}",
            run_number=run_number,
            mlflow_run_id=mlflow_run.info.run_id,
            parameters=run_data.parameters,
            model_config=run_data.model_config,
            status="running",
            created_by=user_id
        )
        
        self.db.add(run)
        
        # Update experiment
        experiment.total_runs += 1
        experiment.status = "running"
        if not experiment.start_time:
            experiment.start_time = datetime.utcnow()
        
        self.db.commit()
        self.db.refresh(run)
        
        logger.info(f"Started run {run.run_name} for experiment {experiment.name}")
        return run
    
    async def log_metrics(self, run_id: str, metrics: Dict[str, Any], step: Optional[int] = None) -> Dict[str, Any]:
        """Log metrics for a run"""
        
        run = self.db.query(ExperimentRun).filter(ExperimentRun.id == run_id).first()
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        
        # Log to MLflow
        with mlflow.start_run(run_id=run.mlflow_run_id):
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value, step=step)
        
        # Update database
        current_metrics = run.metrics or {}
        current_metrics.update(metrics)
        run.metrics = current_metrics
        
        # Update step metrics for plotting
        if step is not None:
            step_metrics = run.step_metrics or {}
            if str(step) not in step_metrics:
                step_metrics[str(step)] = {}
            step_metrics[str(step)].update(metrics)
            run.step_metrics = step_metrics
        
        self.db.commit()
        
        return {'success': True, 'logged_metrics': list(metrics.keys())}
    
    async def log_artifact(self, run_id: str, artifact_path: str, artifact_data: bytes, artifact_type: str = "file") -> Dict[str, Any]:
        """Log an artifact for a run"""
        
        run = self.db.query(ExperimentRun).filter(ExperimentRun.id == run_id).first()
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        
        # Create artifact directory
        artifact_dir = self.storage_path / str(run.experiment_id) / str(run.id) / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        
        # Save artifact
        full_path = artifact_dir / artifact_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_bytes(artifact_data)
        
        # Log to MLflow
        with mlflow.start_run(run_id=run.mlflow_run_id):
            mlflow.log_artifact(str(full_path), artifact_path=artifact_path)
        
        # Update database
        artifacts = run.artifacts or []
        artifacts.append({
            'path': artifact_path,
            'full_path': str(full_path),
            'type': artifact_type,
            'size_bytes': len(artifact_data),
            'created_at': datetime.utcnow().isoformat()
        })
        run.artifacts = artifacts
        
        self.db.commit()
        
        return {'success': True, 'artifact_path': artifact_path}
    
    async def finish_run(self, run_id: str, final_metrics: Optional[Dict[str, Any]] = None, status: str = "completed") -> Dict[str, Any]:
        """Finish an experiment run"""
        
        run = self.db.query(ExperimentRun).filter(ExperimentRun.id == run_id).first()
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        
        # End MLflow run
        with mlflow.start_run(run_id=run.mlflow_run_id):
            mlflow.end_run()
        
        # Update database
        run.status = status
        run.end_time = datetime.utcnow()
        if run.start_time:
            run.duration_seconds = int((run.end_time - run.start_time).total_seconds())
        
        if final_metrics:
            run.final_metrics = final_metrics
        
        # Update experiment stats
        experiment = run.experiment
        if status == "completed":
            experiment.successful_runs += 1
            
            # Update best metric if applicable
            if final_metrics and experiment.objective:
                metric_value = final_metrics.get(experiment.objective)
                if metric_value is not None:
                    if (experiment.best_metric_value is None or 
                        metric_value > experiment.best_metric_value):
                        experiment.best_metric_value = metric_value
                        experiment.best_metric_name = experiment.objective
        else:
            experiment.failed_runs += 1
        
        # Check if experiment is complete
        if not self.db.query(ExperimentRun).filter(
            ExperimentRun.experiment_id == experiment.id,
            ExperimentRun.status == "running"
        ).first():
            experiment.status = "completed"
            experiment.end_time = datetime.utcnow()
            if experiment.start_time:
                experiment.duration_seconds = int(
                    (experiment.end_time - experiment.start_time).total_seconds()
                )
        
        self.db.commit()
        
        return {'success': True, 'status': status, 'duration_seconds': run.duration_seconds}
    
    async def get_experiments(self, user_id: str, skip: int = 0, limit: int = 100, filters: Optional[Dict[str, Any]] = None) -> List[Experiment]:
        """Get experiments with filtering"""
        
        query = self.db.query(Experiment).filter(
            (Experiment.created_by == user_id) |
            (Experiment.is_public == True) |
            (Experiment.organization_id == self._get_user_org(user_id))
        )
        
        if filters:
            if filters.get('status'):
                query = query.filter(Experiment.status == filters['status'])
            if filters.get('experiment_type'):
                query = query.filter(Experiment.experiment_type == filters['experiment_type'])
        
        return query.order_by(desc(Experiment.created_at)).offset(skip).limit(limit).all()
    
    async def get_experiment_runs(self, experiment_id: str, user_id: str) -> List[ExperimentRun]:
        """Get all runs for an experiment"""
        
        experiment = self._get_experiment_by_id(experiment_id, user_id)
        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found")
        
        return self.db.query(ExperimentRun).filter(
            ExperimentRun.experiment_id == experiment.id
        ).order_by(desc(ExperimentRun.created_at)).all()
    
    async def compare_runs(self, run_ids: List[str], user_id: str, metrics_to_compare: Optional[List[str]] = None) -> Dict[str, Any]:
        """Compare multiple experiment runs"""
        
        runs = self.db.query(ExperimentRun).filter(
            ExperimentRun.id.in_(run_ids)
        ).all()
        
        if len(runs) != len(run_ids):
            raise HTTPException(status_code=404, detail="Some runs not found")
        
        # Verify user has access to all runs
        for run in runs:
            experiment = run.experiment
            if not (experiment.created_by == user_id or experiment.is_public or experiment.organization_id == self._get_user_org(user_id)):
                raise HTTPException(status_code=403, detail="Access denied to some runs")
        
        # Gather comparison data
        comparison_data = {
            'runs': [],
            'parameter_comparison': {},
            'metric_comparison': {},
            'charts': []
        }
        
        all_metrics = set()
        all_parameters = set()
        
        # Collect data from all runs
        for run in runs:
            run_data = {
                'id': str(run.id),
                'name': run.run_name,
                'parameters': run.parameters or {},
                'final_metrics': run.final_metrics or {},
                'duration_seconds': run.duration_seconds,
                'status': run.status
            }
            comparison_data['runs'].append(run_data)
            
            if run.final_metrics:
                all_metrics.update(run.final_metrics.keys())
            if run.parameters:
                all_parameters.update(run.parameters.keys())
        
        # Filter metrics to compare
        if metrics_to_compare:
            all_metrics = all_metrics.intersection(set(metrics_to_compare))
        
        # Create parameter comparison table
        for param in all_parameters:
            comparison_data['parameter_comparison'][param] = [
                run['parameters'].get(param, None) for run in comparison_data['runs']
            ]
        
        # Create metric comparison table
        for metric in all_metrics:
            values = [run['final_metrics'].get(metric, None) for run in comparison_data['runs']]
            comparison_data['metric_comparison'][metric] = {
                'values': values,
                'best_run': comparison_data['runs'][np.nanargmax(values)]['name'] if any(v is not None for v in values) else None,
                'worst_run': comparison_data['runs'][np.nanargmin(values)]['name'] if any(v is not None for v in values) else None
            }
        
        # Generate visualization data
        if all_metrics:
            # Metric comparison chart
            chart_data = []
            for i, run in enumerate(comparison_data['runs']):
                for metric in all_metrics:
                    value = run['final_metrics'].get(metric)
                    if value is not None:
                        chart_data.append({
                            'run': run['name'],
                            'metric': metric,
                            'value': value
                        })
            
            comparison_data['charts'].append({
                'type': 'metric_comparison',
                'data': chart_data
            })
        
        return comparison_data
    
    async def generate_experiment_report(self, experiment_id: str, user_id: str) -> Dict[str, Any]:
        """Generate comprehensive experiment report"""
        
        experiment = self._get_experiment_by_id(experiment_id, user_id)
        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found")
        
        runs = await self.get_experiment_runs(experiment_id, user_id)
        
        # Generate report data
        report = {
            'experiment': {
                'name': experiment.name,
                'description': experiment.description,
                'type': experiment.experiment_type,
                'status': experiment.status,
                'total_runs': experiment.total_runs,
                'successful_runs': experiment.successful_runs,
                'failed_runs': experiment.failed_runs,
                'best_metric': experiment.best_metric_value,
                'duration': experiment.duration_seconds
            },
            'summary_statistics': {},
            'best_runs': [],
            'insights': [],
            'charts': []
        }
        
        if runs:
            # Get successful runs with metrics
            successful_runs = [r for r in runs if r.status == 'completed' and r.final_metrics]
            
            if successful_runs:
                # Summary statistics
                all_metrics = set()
                for run in successful_runs:
                    all_metrics.update(run.final_metrics.keys())
                
                for metric in all_metrics:
                    values = [r.final_metrics.get(metric) for r in successful_runs if r.final_metrics.get(metric) is not None]
                    if values:
                        report['summary_statistics'][metric] = {
                            'mean': float(np.mean(values)),
                            'std': float(np.std(values)),
                            'min': float(np.min(values)),
                            'max': float(np.max(values)),
                            'count': len(values)
                        }
                
                # Best runs (top 5 by objective metric)
                if experiment.objective and experiment.objective in all_metrics:
                    best_runs = sorted(
                        successful_runs,
                        key=lambda r: r.final_metrics.get(experiment.objective, -float('inf')),
                        reverse=True
                    )[:5]
                    
                    report['best_runs'] = [
                        {
                            'run_name': run.run_name,
                            'run_id': str(run.id),
                            'metrics': run.final_metrics,
                            'parameters': run.parameters,
                            'duration_seconds': run.duration_seconds
                        }
                        for run in best_runs
                    ]
                
                # Generate insights
                report['insights'] = self._generate_experiment_insights(experiment, successful_runs)
        
        return report
    
    # Private methods
    def _get_experiment_by_id(self, experiment_id: str, user_id: str) -> Optional[Experiment]:
        """Get experiment by ID with access control"""
        
        return self.db.query(Experiment).filter(
            Experiment.id == experiment_id,
            (
                (Experiment.created_by == user_id) |
                (Experiment.is_public == True) |
                (Experiment.organization_id == self._get_user_org(user_id))
            )
        ).first()
    
    def _get_user_org(self, user_id: str) -> Optional[str]:
        """Get user's organization ID"""
        return "default_org"
    
    def _generate_experiment_insights(self, experiment: Experiment, runs: List[ExperimentRun]) -> List[Dict[str, Any]]:
        """Generate AI-powered insights from experiment results"""
        
        insights = []
        
        if len(runs) < 2:
            return insights
        
        try:
            # Parameter sensitivity analysis
            all_params = set()
            for run in runs:
                if run.parameters:
                    all_params.update(run.parameters.keys())
            
            for param in all_params:
                param_values = []
                metric_values = []
                
                for run in runs:
                    if (run.parameters and param in run.parameters and 
                        run.final_metrics and experiment.objective in run.final_metrics):
                        param_values.append(run.parameters[param])
                        metric_values.append(run.final_metrics[experiment.objective])
                
                if len(param_values) > 3:
                    # Calculate correlation
                    try:
                        correlation = np.corrcoef(param_values, metric_values)[0, 1]
                        if abs(correlation) > 0.5:
                            insights.append({
                                'type': 'parameter_sensitivity',
                                'parameter': param,
                                'correlation': float(correlation),
                                'description': f"Parameter '{param}' shows {'strong positive' if correlation > 0.5 else 'strong negative'} correlation with {experiment.objective}"
                            })
                    except:
                        pass
            
            # Performance trends
            if len(runs) > 5:
                recent_runs = sorted(runs, key=lambda r: r.start_time)[-5:]
                old_runs = sorted(runs, key=lambda r: r.start_time)[:-5]
                
                if recent_runs and old_runs and experiment.objective:
                    recent_scores = [r.final_metrics.get(experiment.objective) for r in recent_runs 
                                   if r.final_metrics and experiment.objective in r.final_metrics]
                    old_scores = [r.final_metrics.get(experiment.objective) for r in old_runs 
                                if r.final_metrics and experiment.objective in r.final_metrics]
                    
                    if recent_scores and old_scores:
                        trend = np.mean(recent_scores) - np.mean(old_scores)
                        if abs(trend) > 0.01:  # Only report significant trends
                            insights.append({
                                'type': 'performance_trend',
                                'trend': 'improving' if trend > 0 else 'declining',
                                'magnitude': float(abs(trend)),
                                'description': f"Recent runs show {'improving' if trend > 0 else 'declining'} performance trend"
                            })
            
        except Exception as e:
            logger.warning(f"Failed to generate insights: {str(e)}")
        
        return insights

# API Endpoints
@router.post("/", response_model=ExperimentResponse)
async def create_experiment(
    experiment_data: ExperimentCreate,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Create a new experiment"""
    
    service = ExperimentTrackingService(db)
    experiment = await service.create_experiment(experiment_data, current_user)
    return experiment

@router.get("/", response_model=List[ExperimentResponse])
async def get_experiments(
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None,
    experiment_type: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Get experiments with filtering"""
    
    filters = {}
    if status:
        filters['status'] = status
    if experiment_type:
        filters['experiment_type'] = experiment_type
    
    service = ExperimentTrackingService(db)
    experiments = await service.get_experiments(current_user, skip, limit, filters)
    return experiments

@router.get("/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(
    experiment_id: str,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Get a specific experiment"""
    
    service = ExperimentTrackingService(db)
    experiment = service._get_experiment_by_id(experiment_id, current_user)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return experiment

@router.put("/{experiment_id}", response_model=ExperimentResponse)
async def update_experiment(
    experiment_id: str,
    experiment_data: ExperimentUpdate,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Update experiment metadata"""
    
    service = ExperimentTrackingService(db)
    experiment = service._get_experiment_by_id(experiment_id, current_user)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    if experiment.created_by != current_user:
        raise HTTPException(status_code=403, detail="Permission denied")
    
    # Update fields
    for field, value in experiment_data.dict(exclude_unset=True).items():
        setattr(experiment, field, value)
    
    experiment.updated_at = datetime.utcnow()
    service.db.commit()
    
    return experiment

@router.post("/{experiment_id}/runs", response_model=ExperimentRunResponse)
async def start_run(
    experiment_id: str,
    run_data: ExperimentRunCreate,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Start a new experiment run"""
    
    service = ExperimentTrackingService(db)
    run = await service.start_run(experiment_id, run_data, current_user)
    return run

@router.get("/{experiment_id}/runs", response_model=List[ExperimentRunResponse])
async def get_experiment_runs(
    experiment_id: str,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Get all runs for an experiment"""
    
    service = ExperimentTrackingService(db)
    runs = await service.get_experiment_runs(experiment_id, current_user)
    return runs

@router.post("/runs/{run_id}/metrics")
async def log_metrics(
    run_id: str,
    metrics: Dict[str, Any],
    step: Optional[int] = None,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Log metrics for a run"""
    
    service = ExperimentTrackingService(db)
    result = await service.log_metrics(run_id, metrics, step)
    return result

@router.post("/runs/{run_id}/artifacts")
async def log_artifact(
    run_id: str,
    file: UploadFile = File(...),
    artifact_type: str = "file",
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Log an artifact for a run"""
    
    service = ExperimentTrackingService(db)
    artifact_data = await file.read()
    result = await service.log_artifact(run_id, file.filename, artifact_data, artifact_type)
    return result

@router.post("/runs/{run_id}/finish")
async def finish_run(
    run_id: str,
    final_metrics: Optional[Dict[str, Any]] = None,
    status: str = "completed",
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Finish an experiment run"""
    
    service = ExperimentTrackingService(db)
    result = await service.finish_run(run_id, final_metrics, status)
    return result

@router.post("/compare")
async def compare_runs(
    run_ids: List[str],
    metrics_to_compare: Optional[List[str]] = None,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Compare multiple experiment runs"""
    
    service = ExperimentTrackingService(db)
    comparison = await service.compare_runs(run_ids, current_user, metrics_to_compare)
    return comparison

@router.get("/{experiment_id}/report")
async def generate_experiment_report(
    experiment_id: str,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Generate comprehensive experiment report"""
    
    service = ExperimentTrackingService(db)
    report = await service.generate_experiment_report(experiment_id, current_user)
    return report

@router.delete("/{experiment_id}")
async def delete_experiment(
    experiment_id: str,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Delete an experiment and all its runs"""
    
    service = ExperimentTrackingService(db)
    experiment = service._get_experiment_by_id(experiment_id, current_user)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    if experiment.created_by != current_user:
        raise HTTPException(status_code=403, detail="Permission denied")
    
    # Clean up MLflow experiment
    try:
        service.client.delete_experiment(experiment.mlflow_experiment_id)
    except Exception as e:
        logger.warning(f"Failed to delete MLflow experiment: {str(e)}")
    
    # Clean up files
    experiment_dir = service.storage_path / str(experiment.id)
    if experiment_dir.exists():
        shutil.rmtree(experiment_dir)
    
    # Delete database records (cascading)
    service.db.delete(experiment)
    service.db.commit()
    
    logger.info(f"Deleted experiment {experiment.name} (ID: {experiment.id})")
    
    return {'success': True, 'message': 'Experiment deleted successfully'}