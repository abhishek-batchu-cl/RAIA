"""
Model Monitoring API Endpoints
Comprehensive real-time model monitoring and alerting endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

from app.core.database import get_database
from app.core.auth import get_current_user
from app.models.schemas import User
from app.services.model_monitoring_service import model_monitoring_service
from app.core.exceptions import RAIAException

router = APIRouter()

@router.post("/models/{model_id}/register")
async def register_model_for_monitoring(
    model_id: str,
    model_name: str,
    model_version: str,
    model_type: str,
    performance_metrics: List[str],
    baseline_performance: Dict[str, float],
    monitoring_config: Optional[Dict[str, Any]] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Register a model for continuous monitoring
    
    Args:
        model_id: Unique model identifier
        model_name: Human-readable model name
        model_version: Model version
        model_type: Type of model (classification, regression)
        performance_metrics: List of metrics to monitor
        baseline_performance: Baseline performance values
        monitoring_config: Optional monitoring configuration
    
    Returns:
        Registration status and monitoring setup
    """
    try:
        result = await model_monitoring_service.register_model_for_monitoring(
            model_id=model_id,
            model_name=model_name,
            model_version=model_version,
            model_type=model_type,
            performance_metrics=performance_metrics,
            baseline_performance=baseline_performance,
            monitoring_config=monitoring_config
        )
        
        if result['status'] == 'error':
            raise HTTPException(status_code=400, detail=result['message'])
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to register model for monitoring: {str(e)}")

@router.post("/models/{model_id}/predictions")
async def log_prediction(
    model_id: str,
    prediction_data: Dict[str, Any],
    ground_truth: Optional[Any] = None,
    prediction_time: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Log a model prediction for monitoring
    
    Args:
        model_id: Model identifier
        prediction_data: Prediction details including prediction, probability, features, etc.
        ground_truth: True label (if available)
        prediction_time: When prediction was made (ISO format)
    
    Returns:
        Logging status
    """
    try:
        # Parse prediction time if provided
        parsed_time = None
        if prediction_time:
            try:
                parsed_time = datetime.fromisoformat(prediction_time.replace('Z', '+00:00'))
            except ValueError:
                parsed_time = datetime.utcnow()
        
        result = await model_monitoring_service.log_prediction(
            model_id=model_id,
            prediction_data=prediction_data,
            ground_truth=ground_truth,
            prediction_time=parsed_time
        )
        
        if result['status'] == 'error':
            raise HTTPException(status_code=400, detail=result['message'])
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to log prediction: {str(e)}")

@router.get("/models/{model_id}/health")
async def get_model_health(
    model_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get comprehensive model health score and metrics
    
    Args:
        model_id: Model identifier
    
    Returns:
        Model health metrics and score
    """
    try:
        result = await model_monitoring_service.calculate_model_health(model_id)
        
        if result.get('status') == 'error':
            raise HTTPException(status_code=404, detail=result['message'])
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model health: {str(e)}")

@router.get("/models/{model_id}/performance")
async def get_performance_metrics(
    model_id: str,
    time_window_hours: int = Query(default=24, ge=1, le=8760),
    aggregation: str = Query(default="mean", regex="^(mean|min|max|latest)$"),
    current_user: User = Depends(get_current_user)
):
    """
    Get performance metrics for a model over a time window
    
    Args:
        model_id: Model identifier
        time_window_hours: Time window in hours for metrics
        aggregation: Aggregation method (mean, min, max, latest)
    
    Returns:
        Performance metrics data with trends
    """
    try:
        time_window = timedelta(hours=time_window_hours)
        
        result = await model_monitoring_service.get_performance_metrics(
            model_id=model_id,
            time_window=time_window,
            aggregation=aggregation
        )
        
        if result['status'] == 'error':
            raise HTTPException(status_code=400, detail=result['message'])
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")

@router.post("/models/{model_id}/alerts")
async def create_alert_rule(
    model_id: str,
    alert_name: str,
    condition: Dict[str, Any],
    action: Dict[str, Any],
    enabled: bool = True,
    current_user: User = Depends(get_current_user)
):
    """
    Create an alert rule for a model
    
    Args:
        model_id: Model identifier
        alert_name: Name of the alert rule
        condition: Alert condition definition
        action: Action to take when alert triggers
        enabled: Whether the alert is enabled
    
    Returns:
        Alert rule creation status
    """
    try:
        result = await model_monitoring_service.create_alert_rule(
            model_id=model_id,
            alert_name=alert_name,
            condition=condition,
            action=action,
            enabled=enabled
        )
        
        if result['status'] == 'error':
            raise HTTPException(status_code=400, detail=result['message'])
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create alert rule: {str(e)}")

@router.get("/models/{model_id}/alerts")
async def get_alert_rules(
    model_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get all alert rules for a model
    
    Args:
        model_id: Model identifier
    
    Returns:
        List of alert rules
    """
    try:
        if model_id not in model_monitoring_service.alert_rules:
            return JSONResponse(content={
                'status': 'success',
                'model_id': model_id,
                'alert_rules': [],
                'total_rules': 0
            })
        
        alert_rules = model_monitoring_service.alert_rules[model_id]
        
        return JSONResponse(content={
            'status': 'success',
            'model_id': model_id,
            'alert_rules': alert_rules,
            'total_rules': len(alert_rules)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get alert rules: {str(e)}")

@router.put("/models/{model_id}/alerts/{alert_id}")
async def update_alert_rule(
    model_id: str,
    alert_id: str,
    enabled: Optional[bool] = None,
    condition: Optional[Dict[str, Any]] = None,
    action: Optional[Dict[str, Any]] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Update an existing alert rule
    
    Args:
        model_id: Model identifier
        alert_id: Alert rule identifier
        enabled: Whether the alert is enabled
        condition: Updated alert condition
        action: Updated alert action
    
    Returns:
        Update status
    """
    try:
        if model_id not in model_monitoring_service.alert_rules:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Find and update the alert rule
        alert_found = False
        for alert_rule in model_monitoring_service.alert_rules[model_id]:
            if alert_rule['alert_id'] == alert_id:
                if enabled is not None:
                    alert_rule['enabled'] = enabled
                if condition is not None:
                    alert_rule['condition'] = condition
                if action is not None:
                    alert_rule['action'] = action
                alert_found = True
                break
        
        if not alert_found:
            raise HTTPException(status_code=404, detail="Alert rule not found")
        
        return JSONResponse(content={
            'status': 'success',
            'model_id': model_id,
            'alert_id': alert_id,
            'message': 'Alert rule updated successfully'
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update alert rule: {str(e)}")

@router.delete("/models/{model_id}/alerts/{alert_id}")
async def delete_alert_rule(
    model_id: str,
    alert_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Delete an alert rule
    
    Args:
        model_id: Model identifier
        alert_id: Alert rule identifier
    
    Returns:
        Deletion status
    """
    try:
        if model_id not in model_monitoring_service.alert_rules:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Find and remove the alert rule
        initial_count = len(model_monitoring_service.alert_rules[model_id])
        model_monitoring_service.alert_rules[model_id] = [
            rule for rule in model_monitoring_service.alert_rules[model_id]
            if rule['alert_id'] != alert_id
        ]
        final_count = len(model_monitoring_service.alert_rules[model_id])
        
        if initial_count == final_count:
            raise HTTPException(status_code=404, detail="Alert rule not found")
        
        return JSONResponse(content={
            'status': 'success',
            'model_id': model_id,
            'alert_id': alert_id,
            'message': 'Alert rule deleted successfully'
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete alert rule: {str(e)}")

@router.get("/monitoring/summary")
async def get_monitoring_summary(
    current_user: User = Depends(get_current_user)
):
    """
    Get summary of all monitored models
    
    Returns:
        Summary of monitoring status across all models
    """
    try:
        result = await model_monitoring_service.get_monitoring_summary()
        
        if result['status'] == 'error':
            raise HTTPException(status_code=500, detail=result['message'])
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get monitoring summary: {str(e)}")

@router.get("/models/{model_id}/predictions")
async def get_prediction_logs(
    model_id: str,
    limit: int = Query(default=100, ge=1, le=1000),
    start_time: Optional[str] = Query(default=None),
    end_time: Optional[str] = Query(default=None),
    current_user: User = Depends(get_current_user)
):
    """
    Get prediction logs for a model
    
    Args:
        model_id: Model identifier
        limit: Maximum number of predictions to return
        start_time: Start time filter (ISO format)
        end_time: End time filter (ISO format)
    
    Returns:
        List of prediction logs
    """
    try:
        if model_id not in model_monitoring_service.prediction_logs:
            return JSONResponse(content={
                'status': 'success',
                'model_id': model_id,
                'predictions': [],
                'total_predictions': 0,
                'filters_applied': {'limit': limit, 'start_time': start_time, 'end_time': end_time}
            })
        
        predictions = list(model_monitoring_service.prediction_logs[model_id])
        
        # Apply time filters if provided
        if start_time or end_time:
            filtered_predictions = []
            for pred in predictions:
                pred_time = pred['timestamp']
                
                if start_time:
                    try:
                        start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                        if pred_time < start_dt:
                            continue
                    except ValueError:
                        pass
                
                if end_time:
                    try:
                        end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                        if pred_time > end_dt:
                            continue
                    except ValueError:
                        pass
                
                filtered_predictions.append(pred)
            
            predictions = filtered_predictions
        
        # Apply limit
        predictions = sorted(predictions, key=lambda x: x['timestamp'], reverse=True)[:limit]
        
        # Convert timestamps to ISO strings for JSON serialization
        for pred in predictions:
            pred['timestamp'] = pred['timestamp'].isoformat()
        
        return JSONResponse(content={
            'status': 'success',
            'model_id': model_id,
            'predictions': predictions,
            'total_predictions': len(predictions),
            'filters_applied': {'limit': limit, 'start_time': start_time, 'end_time': end_time}
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get prediction logs: {str(e)}")

@router.delete("/models/{model_id}/monitoring")
async def stop_model_monitoring(
    model_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Stop monitoring for a model and cleanup resources
    
    Args:
        model_id: Model identifier
    
    Returns:
        Cleanup status
    """
    try:
        if model_id not in model_monitoring_service.monitored_models:
            raise HTTPException(status_code=404, detail="Model not registered for monitoring")
        
        # Stop monitoring task
        if model_id in model_monitoring_service._monitoring_tasks:
            task = model_monitoring_service._monitoring_tasks[model_id]
            task.cancel()
            del model_monitoring_service._monitoring_tasks[model_id]
        
        # Clean up data
        del model_monitoring_service.monitored_models[model_id]
        if model_id in model_monitoring_service.performance_history:
            del model_monitoring_service.performance_history[model_id]
        if model_id in model_monitoring_service.alert_rules:
            del model_monitoring_service.alert_rules[model_id]
        if model_id in model_monitoring_service.model_health_cache:
            del model_monitoring_service.model_health_cache[model_id]
        if model_id in model_monitoring_service.prediction_logs:
            del model_monitoring_service.prediction_logs[model_id]
        
        return JSONResponse(content={
            'status': 'success',
            'model_id': model_id,
            'message': 'Model monitoring stopped and resources cleaned up'
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop model monitoring: {str(e)}")

@router.get("/monitoring/config")
async def get_monitoring_configuration():
    """
    Get current monitoring configuration and thresholds
    
    Returns:
        Monitoring configuration details
    """
    config = {
        'performance_thresholds': model_monitoring_service.performance_thresholds,
        'monitoring_interval_seconds': model_monitoring_service.monitoring_interval,
        'health_score_components': {
            'prediction_volume': {'weight': 0.2, 'description': 'Model usage frequency'},
            'error_rate': {'weight': 0.25, 'description': 'Prediction error frequency'},
            'performance': {'weight': 0.25, 'description': 'Model accuracy vs baseline'},
            'latency': {'weight': 0.15, 'description': 'Prediction response time'},
            'freshness': {'weight': 0.15, 'description': 'Time since last prediction'}
        },
        'alert_condition_types': {
            'threshold': {
                'description': 'Trigger when metric crosses threshold',
                'operators': ['gt', 'lt', 'gte', 'lte', 'eq'],
                'example': {
                    'type': 'threshold',
                    'metric': 'health_score',
                    'operator': 'lt',
                    'threshold': 0.7
                }
            },
            'status': {
                'description': 'Trigger on specific health status',
                'valid_statuses': ['healthy', 'warning', 'critical'],
                'example': {
                    'type': 'status',
                    'status': 'critical'
                }
            }
        },
        'supported_metrics': [
            'accuracy', 'precision', 'recall', 'f1_score', 'auc_roc',
            'mse', 'mae', 'r2', 'latency_ms', 'error_rate', 'health_score'
        ]
    }
    
    return JSONResponse(content=config)
