# ML Model Registry Service - Core Business Logic
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc, func
from datetime import datetime, timedelta
import json
import joblib
import pickle
import boto3
from pathlib import Path
import numpy as np
from .models import Model, ModelVersion, ModelPrediction, ModelPerformanceMetric, ModelDriftReport, ModelExperiment, ModelApprovalWorkflow
from ..exceptions import ModelNotFoundError, ValidationError, ModelAlreadyExistsError

class ModelRegistryService:
    """Core service for ML model registry operations"""
    
    def __init__(self, db: Session, file_storage_client=None):
        self.db = db
        self.storage_client = file_storage_client or self._init_storage_client()
    
    def _init_storage_client(self):
        """Initialize file storage client (S3, MinIO, etc.)"""
        return boto3.client('s3')
    
    # ========================================================================================
    # MODEL CRUD OPERATIONS
    # ========================================================================================
    
    def create_model(self, model_data: Dict[str, Any], user_id: str) -> Model:
        """Create a new ML model in the registry"""
        
        # Validate required fields
        required_fields = ['name', 'display_name', 'model_type', 'algorithm']
        for field in required_fields:
            if field not in model_data:
                raise ValidationError(f"Required field '{field}' is missing")
        
        # Check if model name already exists
        existing_model = self.db.query(Model).filter(Model.name == model_data['name']).first()
        if existing_model:
            raise ModelAlreadyExistsError(f"Model with name '{model_data['name']}' already exists")
        
        # Create model instance
        model = Model(
            name=model_data['name'],
            display_name=model_data['display_name'],
            description=model_data.get('description'),
            model_type=model_data['model_type'],
            algorithm=model_data['algorithm'],
            framework=model_data.get('framework'),
            version=model_data.get('version', '1.0.0'),
            feature_names=model_data.get('feature_names', []),
            feature_types=model_data.get('feature_types', {}),
            target_column=model_data.get('target_column'),
            training_parameters=model_data.get('training_parameters', {}),
            business_objective=model_data.get('business_objective'),
            use_cases=model_data.get('use_cases', []),
            stakeholders=model_data.get('stakeholders', []),
            tags=model_data.get('tags', []),
            metadata=model_data.get('metadata', {}),
            created_by=user_id
        )
        
        self.db.add(model)
        self.db.commit()
        self.db.refresh(model)
        
        # Create initial version
        self._create_initial_version(model, user_id)
        
        return model
    
    def get_model(self, model_id: str) -> Model:
        """Get model by ID"""
        model = self.db.query(Model).filter(Model.id == model_id).first()
        if not model:
            raise ModelNotFoundError(f"Model with ID {model_id} not found")
        return model
    
    def get_models(self, 
                  user_id: str = None,
                  model_type: str = None, 
                  status: str = None,
                  tags: List[str] = None,
                  search_query: str = None,
                  page: int = 1, 
                  page_size: int = 20,
                  sort_by: str = 'created_at',
                  sort_order: str = 'desc') -> Dict[str, Any]:
        """Get paginated list of models with filtering and searching"""
        
        query = self.db.query(Model)
        
        # Apply filters
        if user_id:
            query = query.filter(Model.created_by == user_id)
        
        if model_type:
            query = query.filter(Model.model_type == model_type)
            
        if status:
            query = query.filter(Model.status == status)
            
        if tags:
            # Filter models that have ALL specified tags
            for tag in tags:
                query = query.filter(Model.tags.any(tag))
        
        if search_query:
            search_filter = or_(
                Model.name.ilike(f'%{search_query}%'),
                Model.display_name.ilike(f'%{search_query}%'),
                Model.description.ilike(f'%{search_query}%')
            )
            query = query.filter(search_filter)
        
        # Apply sorting
        sort_column = getattr(Model, sort_by, Model.created_at)
        if sort_order.lower() == 'desc':
            query = query.order_by(desc(sort_column))
        else:
            query = query.order_by(asc(sort_column))
        
        # Get total count
        total_count = query.count()
        
        # Apply pagination
        offset = (page - 1) * page_size
        models = query.offset(offset).limit(page_size).all()
        
        return {
            'models': models,
            'total_count': total_count,
            'page': page,
            'page_size': page_size,
            'total_pages': (total_count + page_size - 1) // page_size
        }
    
    def update_model(self, model_id: str, model_data: Dict[str, Any], user_id: str) -> Model:
        """Update an existing model"""
        model = self.get_model(model_id)
        
        # Update allowed fields
        updatable_fields = [
            'display_name', 'description', 'status', 'business_objective',
            'use_cases', 'stakeholders', 'tags', 'metadata'
        ]
        
        for field in updatable_fields:
            if field in model_data:
                setattr(model, field, model_data[field])
        
        model.updated_at = datetime.utcnow()
        
        self.db.commit()
        self.db.refresh(model)
        
        return model
    
    def delete_model(self, model_id: str, user_id: str) -> bool:
        """Delete a model (soft delete by changing status)"""
        model = self.get_model(model_id)
        
        # Only allow deletion if model is not in production
        if model.status == 'production':
            raise ValidationError("Cannot delete a model in production status")
        
        model.status = 'deleted'
        model.updated_at = datetime.utcnow()
        
        self.db.commit()
        return True
    
    # ========================================================================================
    # MODEL VERSIONING
    # ========================================================================================
    
    def create_model_version(self, model_id: str, version_data: Dict[str, Any], user_id: str) -> ModelVersion:
        """Create a new version of an existing model"""
        model = self.get_model(model_id)
        
        # Check if version already exists
        existing_version = self.db.query(ModelVersion).filter(
            and_(ModelVersion.model_id == model_id, ModelVersion.version == version_data['version'])
        ).first()
        
        if existing_version:
            raise ValidationError(f"Version {version_data['version']} already exists for this model")
        
        # Get previous version for comparison
        previous_version = self.db.query(ModelVersion).filter(
            ModelVersion.model_id == model_id
        ).order_by(desc(ModelVersion.created_at)).first()
        
        version = ModelVersion(
            model_id=model_id,
            version=version_data['version'],
            changelog=version_data.get('changelog'),
            performance_improvements=version_data.get('performance_improvements', {}),
            breaking_changes=version_data.get('breaking_changes', False),
            deployment_environment=version_data.get('deployment_environment'),
            deployment_config=version_data.get('deployment_config', {}),
            previous_version_id=previous_version.id if previous_version else None,
            champion_challenger_status=version_data.get('champion_challenger_status', 'challenger'),
            created_by=user_id
        )
        
        self.db.add(version)
        
        # Update model to latest version if specified
        if version_data.get('is_latest', False):
            # Set all other versions as not latest
            self.db.query(Model).filter(Model.id == model_id).update({'is_latest': False})
            model.version = version_data['version']
            model.is_latest = True
        
        self.db.commit()
        self.db.refresh(version)
        
        return version
    
    def get_model_versions(self, model_id: str) -> List[ModelVersion]:
        """Get all versions of a model"""
        return self.db.query(ModelVersion).filter(
            ModelVersion.model_id == model_id
        ).order_by(desc(ModelVersion.created_at)).all()
    
    def deploy_model_version(self, model_id: str, version: str, environment: str, user_id: str) -> ModelVersion:
        """Deploy a specific model version to an environment"""
        model_version = self.db.query(ModelVersion).filter(
            and_(ModelVersion.model_id == model_id, ModelVersion.version == version)
        ).first()
        
        if not model_version:
            raise ModelNotFoundError(f"Model version {version} not found")
        
        model_version.deployed_at = datetime.utcnow()
        model_version.deployment_environment = environment
        
        # Update model deployment status
        model = self.get_model(model_id)
        model.deployment_status = 'deployed'
        model.status = environment  # 'staging' or 'production'
        
        self.db.commit()
        self.db.refresh(model_version)
        
        return model_version
    
    # ========================================================================================
    # MODEL PERFORMANCE TRACKING
    # ========================================================================================
    
    def record_performance_metric(self, model_id: str, metric_data: Dict[str, Any]) -> ModelPerformanceMetric:
        """Record a performance metric for a model"""
        metric = ModelPerformanceMetric(
            model_id=model_id,
            model_version=metric_data.get('model_version'),
            metric_name=metric_data['metric_name'],
            metric_value=metric_data['metric_value'],
            metric_category=metric_data.get('metric_category', 'performance'),
            measurement_date=metric_data.get('measurement_date', datetime.utcnow()),
            measurement_period=metric_data.get('measurement_period', 'daily'),
            dataset_name=metric_data.get('dataset_name'),
            sample_size=metric_data.get('sample_size'),
            data_period_start=metric_data.get('data_period_start'),
            data_period_end=metric_data.get('data_period_end'),
            threshold_value=metric_data.get('threshold_value'),
            baseline_value=metric_data.get('baseline_value'),
            metadata=metric_data.get('metadata', {})
        )
        
        # Calculate trend and alerts
        if metric.baseline_value and metric.metric_value:
            metric.change_percentage = ((metric.metric_value - metric.baseline_value) / metric.baseline_value) * 100
            
            if metric.change_percentage > 5:
                metric.trend_direction = 'improving'
            elif metric.change_percentage < -5:
                metric.trend_direction = 'degrading'
            else:
                metric.trend_direction = 'stable'
        
        # Check for threshold violations
        if metric.threshold_value:
            if metric.metric_name in ['accuracy', 'precision', 'recall', 'f1_score']:
                # Higher is better
                if metric.metric_value < metric.threshold_value:
                    metric.alert_triggered = True
                    metric.alert_severity = 'high' if metric.metric_value < metric.threshold_value * 0.9 else 'medium'
            else:
                # Lower is better (e.g., error rate)
                if metric.metric_value > metric.threshold_value:
                    metric.alert_triggered = True
                    metric.alert_severity = 'high' if metric.metric_value > metric.threshold_value * 1.1 else 'medium'
        
        self.db.add(metric)
        self.db.commit()
        self.db.refresh(metric)
        
        return metric
    
    def get_performance_metrics(self, 
                               model_id: str,
                               metric_names: List[str] = None,
                               date_from: datetime = None,
                               date_to: datetime = None) -> List[ModelPerformanceMetric]:
        """Get performance metrics for a model"""
        query = self.db.query(ModelPerformanceMetric).filter(
            ModelPerformanceMetric.model_id == model_id
        )
        
        if metric_names:
            query = query.filter(ModelPerformanceMetric.metric_name.in_(metric_names))
        
        if date_from:
            query = query.filter(ModelPerformanceMetric.measurement_date >= date_from)
        
        if date_to:
            query = query.filter(ModelPerformanceMetric.measurement_date <= date_to)
        
        return query.order_by(ModelPerformanceMetric.measurement_date).all()
    
    def get_model_performance_summary(self, model_id: str) -> Dict[str, Any]:
        """Get performance summary for a model"""
        model = self.get_model(model_id)
        
        # Get latest metrics
        latest_metrics = self.db.query(ModelPerformanceMetric).filter(
            ModelPerformanceMetric.model_id == model_id
        ).order_by(desc(ModelPerformanceMetric.measurement_date)).limit(10).all()
        
        # Get performance trends (last 30 days)
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        trend_metrics = self.get_performance_metrics(
            model_id, date_from=thirty_days_ago
        )
        
        # Calculate summary statistics
        summary = {
            'model_id': model_id,
            'model_name': model.name,
            'current_status': model.status,
            'latest_metrics': {
                metric.metric_name: {
                    'value': float(metric.metric_value),
                    'measurement_date': metric.measurement_date.isoformat(),
                    'trend': metric.trend_direction,
                    'alert_triggered': metric.alert_triggered
                }
                for metric in latest_metrics
            },
            'performance_trends': self._calculate_performance_trends(trend_metrics),
            'alerts': [
                metric for metric in latest_metrics if metric.alert_triggered
            ]
        }
        
        return summary
    
    def _calculate_performance_trends(self, metrics: List[ModelPerformanceMetric]) -> Dict[str, Any]:
        """Calculate performance trends from metric history"""
        trends = {}
        
        # Group metrics by name
        metric_groups = {}
        for metric in metrics:
            if metric.metric_name not in metric_groups:
                metric_groups[metric.metric_name] = []
            metric_groups[metric.metric_name].append(metric)
        
        for metric_name, metric_list in metric_groups.items():
            if len(metric_list) < 2:
                continue
                
            # Sort by date
            metric_list.sort(key=lambda x: x.measurement_date)
            
            values = [float(m.metric_value) for m in metric_list]
            dates = [m.measurement_date.isoformat() for m in metric_list]
            
            # Calculate trend
            if len(values) >= 2:
                recent_avg = np.mean(values[-7:]) if len(values) >= 7 else np.mean(values[-3:])
                older_avg = np.mean(values[:7]) if len(values) >= 14 else np.mean(values[:-3]) if len(values) > 3 else values[0]
                
                trend_percentage = ((recent_avg - older_avg) / older_avg) * 100 if older_avg != 0 else 0
                
                trends[metric_name] = {
                    'values': values,
                    'dates': dates,
                    'trend_percentage': round(trend_percentage, 2),
                    'trend_direction': 'improving' if trend_percentage > 1 else 'degrading' if trend_percentage < -1 else 'stable',
                    'current_value': values[-1],
                    'min_value': min(values),
                    'max_value': max(values),
                    'avg_value': round(np.mean(values), 4)
                }
        
        return trends
    
    # ========================================================================================
    # MODEL PREDICTIONS AND TRACKING
    # ========================================================================================
    
    def record_prediction(self, prediction_data: Dict[str, Any]) -> ModelPrediction:
        """Record a model prediction"""
        prediction = ModelPrediction(
            model_id=prediction_data['model_id'],
            model_version=prediction_data.get('model_version'),
            input_data=prediction_data['input_data'],
            prediction=prediction_data['prediction'],
            prediction_probabilities=prediction_data.get('prediction_probabilities'),
            confidence_score=prediction_data.get('confidence_score'),
            shap_values=prediction_data.get('shap_values'),
            lime_explanation=prediction_data.get('lime_explanation'),
            feature_importance=prediction_data.get('feature_importance'),
            prediction_context=prediction_data.get('prediction_context', {}),
            protected_attributes=prediction_data.get('protected_attributes'),
            fairness_metrics=prediction_data.get('fairness_metrics'),
            api_endpoint=prediction_data.get('api_endpoint'),
            request_id=prediction_data.get('request_id'),
            response_time_ms=prediction_data.get('response_time_ms'),
            user_id=prediction_data.get('user_id')
        )
        
        self.db.add(prediction)
        self.db.commit()
        self.db.refresh(prediction)
        
        return prediction
    
    def get_model_predictions(self, 
                             model_id: str,
                             date_from: datetime = None,
                             date_to: datetime = None,
                             limit: int = 1000) -> List[ModelPrediction]:
        """Get predictions for a model"""
        query = self.db.query(ModelPrediction).filter(
            ModelPrediction.model_id == model_id
        )
        
        if date_from:
            query = query.filter(ModelPrediction.predicted_at >= date_from)
        
        if date_to:
            query = query.filter(ModelPrediction.predicted_at <= date_to)
        
        return query.order_by(desc(ModelPrediction.predicted_at)).limit(limit).all()
    
    def update_prediction_feedback(self, prediction_id: str, actual_outcome: Any) -> ModelPrediction:
        """Update prediction with actual outcome for accuracy calculation"""
        prediction = self.db.query(ModelPrediction).filter(
            ModelPrediction.id == prediction_id
        ).first()
        
        if not prediction:
            raise ModelNotFoundError(f"Prediction with ID {prediction_id} not found")
        
        prediction.actual_outcome = actual_outcome
        prediction.feedback_received = True
        
        # Calculate prediction error
        if prediction.prediction and actual_outcome:
            try:
                pred_value = prediction.prediction.get('value') if isinstance(prediction.prediction, dict) else prediction.prediction
                actual_value = actual_outcome.get('value') if isinstance(actual_outcome, dict) else actual_outcome
                
                if isinstance(pred_value, (int, float)) and isinstance(actual_value, (int, float)):
                    prediction.prediction_error = abs(pred_value - actual_value)
            except:
                pass  # Skip error calculation if values are not numeric
        
        self.db.commit()
        self.db.refresh(prediction)
        
        return prediction
    
    # ========================================================================================
    # MODEL EXPERIMENTS AND A/B TESTING
    # ========================================================================================
    
    def create_experiment(self, experiment_data: Dict[str, Any], user_id: str) -> ModelExperiment:
        """Create a new model experiment"""
        experiment = ModelExperiment(
            model_id=experiment_data['model_id'],
            experiment_name=experiment_data['experiment_name'],
            experiment_type=experiment_data.get('experiment_type', 'hyperparameter_tuning'),
            parameters=experiment_data.get('parameters', {}),
            parameter_search_space=experiment_data.get('parameter_search_space', {}),
            optimization_metric=experiment_data.get('optimization_metric', 'accuracy'),
            optimization_direction=experiment_data.get('optimization_direction', 'maximize'),
            notes=experiment_data.get('notes'),
            tags=experiment_data.get('tags', []),
            created_by=user_id
        )
        
        self.db.add(experiment)
        self.db.commit()
        self.db.refresh(experiment)
        
        return experiment
    
    def start_experiment(self, experiment_id: str) -> ModelExperiment:
        """Start an experiment"""
        experiment = self.db.query(ModelExperiment).filter(
            ModelExperiment.id == experiment_id
        ).first()
        
        if not experiment:
            raise ModelNotFoundError(f"Experiment with ID {experiment_id} not found")
        
        experiment.status = 'running'
        experiment.started_at = datetime.utcnow()
        
        self.db.commit()
        self.db.refresh(experiment)
        
        return experiment
    
    def complete_experiment(self, experiment_id: str, results: Dict[str, Any]) -> ModelExperiment:
        """Complete an experiment with results"""
        experiment = self.db.query(ModelExperiment).filter(
            ModelExperiment.id == experiment_id
        ).first()
        
        if not experiment:
            raise ModelNotFoundError(f"Experiment with ID {experiment_id} not found")
        
        experiment.status = 'completed'
        experiment.completed_at = datetime.utcnow()
        experiment.best_parameters = results.get('best_parameters', {})
        experiment.best_score = results.get('best_score')
        experiment.all_results = results.get('all_results', {})
        experiment.convergence_history = results.get('convergence_history', [])
        experiment.compute_resources_used = results.get('compute_resources_used', {})
        experiment.estimated_cost = results.get('estimated_cost')
        
        # Calculate duration
        if experiment.started_at:
            duration = experiment.completed_at - experiment.started_at
            experiment.duration_minutes = int(duration.total_seconds() / 60)
        
        self.db.commit()
        self.db.refresh(experiment)
        
        return experiment
    
    # ========================================================================================
    # MODEL APPROVAL WORKFLOW
    # ========================================================================================
    
    def initiate_approval_workflow(self, model_id: str, user_id: str) -> ModelApprovalWorkflow:
        """Initiate approval workflow for a model"""
        model = self.get_model(model_id)
        
        # Check if there's already an active workflow
        existing_workflow = self.db.query(ModelApprovalWorkflow).filter(
            and_(
                ModelApprovalWorkflow.model_id == model_id,
                ModelApprovalWorkflow.overall_status.in_(['pending', 'needs_revision'])
            )
        ).first()
        
        if existing_workflow:
            raise ValidationError("An approval workflow is already active for this model")
        
        workflow = ModelApprovalWorkflow(
            model_id=model_id,
            workflow_stage='validation',
            created_by=user_id
        )
        
        self.db.add(workflow)
        self.db.commit()
        self.db.refresh(workflow)
        
        return workflow
    
    def update_approval_workflow(self, workflow_id: str, review_data: Dict[str, Any], user_id: str) -> ModelApprovalWorkflow:
        """Update approval workflow with review results"""
        workflow = self.db.query(ModelApprovalWorkflow).filter(
            ModelApprovalWorkflow.id == workflow_id
        ).first()
        
        if not workflow:
            raise ModelNotFoundError(f"Approval workflow with ID {workflow_id} not found")
        
        # Update review fields based on review type
        review_type = review_data.get('review_type')
        if review_type == 'technical':
            workflow.technical_review_completed = True
            workflow.technical_review_result = review_data.get('result')
            workflow.technical_comments = review_data.get('comments')
            workflow.technical_reviewer = user_id
        elif review_type == 'business':
            workflow.business_review_completed = True
            workflow.business_review_result = review_data.get('result')
            workflow.business_comments = review_data.get('comments')
            workflow.business_reviewer = user_id
        elif review_type == 'security':
            workflow.security_review_completed = True
            workflow.security_review_result = review_data.get('result')
            workflow.security_comments = review_data.get('comments')
            workflow.security_reviewer = user_id
        elif review_type == 'compliance':
            workflow.compliance_review_completed = True
            workflow.compliance_review_result = review_data.get('result')
            workflow.compliance_comments = review_data.get('comments')
            workflow.compliance_reviewer = user_id
        
        # Check if all required reviews are completed
        reviews_completed = all([
            not workflow.technical_review_required or workflow.technical_review_completed,
            not workflow.business_review_required or workflow.business_review_completed,
            not workflow.security_review_required or workflow.security_review_completed,
            not workflow.compliance_review_required or workflow.compliance_review_completed
        ])
        
        if reviews_completed:
            # Determine overall result
            all_results = [
                workflow.technical_review_result,
                workflow.business_review_result,
                workflow.security_review_result,
                workflow.compliance_review_result
            ]
            
            if any(result == 'fail' for result in all_results if result):
                workflow.overall_status = 'rejected'
                workflow.final_decision = 'rejected'
            elif any(result == 'conditional' for result in all_results if result):
                workflow.overall_status = 'needs_revision'
                workflow.final_decision = 'conditional'
            else:
                workflow.overall_status = 'approved'
                workflow.final_decision = 'approved'
            
            workflow.workflow_completed_at = datetime.utcnow()
            workflow.approved_by = user_id
            workflow.approved_at = datetime.utcnow()
        
        self.db.commit()
        self.db.refresh(workflow)
        
        return workflow
    
    def _create_initial_version(self, model: Model, user_id: str):
        """Create initial version when model is created"""
        initial_version = ModelVersion(
            model_id=model.id,
            version="1.0.0",
            changelog="Initial version",
            champion_challenger_status='champion',
            created_by=user_id
        )
        
        self.db.add(initial_version)
        self.db.commit()
    
    # ========================================================================================
    # UTILITY METHODS
    # ========================================================================================
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """Get overall model registry statistics"""
        total_models = self.db.query(func.count(Model.id)).scalar()
        
        models_by_type = self.db.query(
            Model.model_type, 
            func.count(Model.id)
        ).group_by(Model.model_type).all()
        
        models_by_status = self.db.query(
            Model.status, 
            func.count(Model.id)
        ).group_by(Model.status).all()
        
        production_models = self.db.query(func.count(Model.id)).filter(
            Model.status == 'production'
        ).scalar()
        
        return {
            'total_models': total_models,
            'production_models': production_models,
            'models_by_type': dict(models_by_type),
            'models_by_status': dict(models_by_status)
        }