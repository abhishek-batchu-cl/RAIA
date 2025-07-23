"""
Advanced Model Monitoring Service
Real-time model performance tracking, alerting, and health monitoring
"""

import asyncio
import json
import logging
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import uuid
from collections import defaultdict, deque

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import get_database

logger = logging.getLogger(__name__)
settings = get_settings()

class ModelMonitoringService:
    """
    Comprehensive model monitoring service for production ML models
    """
    
    def __init__(self):
        self.monitored_models = {}
        self.performance_history = defaultdict(lambda: deque(maxlen=1000))
        self.alert_rules = defaultdict(list)
        self.model_health_cache = {}
        self.prediction_logs = defaultdict(lambda: deque(maxlen=10000))
        self.performance_thresholds = {
            'accuracy_drop_threshold': 0.05,
            'latency_threshold_ms': 1000,
            'error_rate_threshold': 0.01,
            'throughput_threshold_rps': 10,
            'memory_threshold_mb': 2000,
            'cpu_threshold_percent': 80
        }
        self.monitoring_interval = 60  # seconds
        self._monitoring_tasks = {}
        
    async def register_model_for_monitoring(
        self,
        model_id: str,
        model_name: str,
        model_version: str,
        model_type: str,
        performance_metrics: List[str],
        baseline_performance: Dict[str, float],
        monitoring_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
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
            config = monitoring_config or {}
            
            model_info = {
                'model_id': model_id,
                'model_name': model_name,
                'model_version': model_version,
                'model_type': model_type,
                'performance_metrics': performance_metrics,
                'baseline_performance': baseline_performance,
                'registration_time': datetime.utcnow(),
                'last_health_check': datetime.utcnow(),
                'monitoring_status': 'active',
                'prediction_count': 0,
                'error_count': 0,
                'alert_count': 0,
                'config': {
                    'monitoring_interval': config.get('monitoring_interval', self.monitoring_interval),
                    'alert_thresholds': config.get('alert_thresholds', {}),
                    'enable_drift_detection': config.get('enable_drift_detection', True),
                    'enable_performance_tracking': config.get('enable_performance_tracking', True),
                    'enable_resource_monitoring': config.get('enable_resource_monitoring', True),
                    'retention_days': config.get('retention_days', 30)
                }
            }
            
            # Store model info
            self.monitored_models[model_id] = model_info
            
            # Initialize performance history with baseline
            self.performance_history[model_id].append({
                'timestamp': datetime.utcnow(),
                'metrics': baseline_performance,
                'type': 'baseline',
                'sample_count': 0
            })
            
            # Start monitoring task
            if model_id not in self._monitoring_tasks:
                self._monitoring_tasks[model_id] = asyncio.create_task(
                    self._continuous_monitoring_loop(model_id)
                )
            
            logger.info(f"Model {model_id} registered for monitoring")
            
            return {
                'status': 'success',
                'model_id': model_id,
                'monitoring_status': 'active',
                'baseline_performance': baseline_performance,
                'monitoring_config': model_info['config']
            }
            
        except Exception as e:
            logger.error(f"Failed to register model {model_id} for monitoring: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'model_id': model_id
            }
    
    async def log_prediction(
        self,
        model_id: str,
        prediction_data: Dict[str, Any],
        ground_truth: Optional[Any] = None,
        prediction_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Log a model prediction for monitoring
        
        Args:
            model_id: Model identifier
            prediction_data: Prediction details
            ground_truth: True label (if available)
            prediction_time: When prediction was made
        
        Returns:
            Logging status
        """
        try:
            if model_id not in self.monitored_models:
                raise ValueError(f"Model {model_id} not registered for monitoring")
            
            log_entry = {
                'prediction_id': str(uuid.uuid4()),
                'timestamp': prediction_time or datetime.utcnow(),
                'model_id': model_id,
                'prediction': prediction_data.get('prediction'),
                'prediction_probability': prediction_data.get('probability'),
                'features': prediction_data.get('features', {}),
                'ground_truth': ground_truth,
                'latency_ms': prediction_data.get('latency_ms'),
                'request_id': prediction_data.get('request_id'),
                'user_id': prediction_data.get('user_id')
            }
            
            # Store prediction log
            self.prediction_logs[model_id].append(log_entry)
            
            # Update model stats
            self.monitored_models[model_id]['prediction_count'] += 1
            self.monitored_models[model_id]['last_prediction_time'] = log_entry['timestamp']
            
            # Check for immediate alerts
            await self._check_prediction_alerts(model_id, log_entry)
            
            return {
                'status': 'success',
                'prediction_id': log_entry['prediction_id'],
                'logged_at': log_entry['timestamp'].isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to log prediction for model {model_id}: {e}")
            # Update error count
            if model_id in self.monitored_models:
                self.monitored_models[model_id]['error_count'] += 1
            
            return {
                'status': 'error',
                'message': str(e),
                'model_id': model_id
            }
    
    async def calculate_model_health(self, model_id: str) -> Dict[str, Any]:
        """
        Calculate comprehensive model health score
        
        Args:
            model_id: Model identifier
        
        Returns:
            Model health metrics and score
        """
        try:
            if model_id not in self.monitored_models:
                raise ValueError(f"Model {model_id} not registered for monitoring")
            
            model_info = self.monitored_models[model_id]
            current_time = datetime.utcnow()
            
            # Get recent predictions (last hour)
            recent_predictions = [
                pred for pred in self.prediction_logs[model_id]
                if (current_time - pred['timestamp']).seconds < 3600
            ]
            
            # Get recent performance data
            recent_performance = [
                perf for perf in self.performance_history[model_id]
                if (current_time - perf['timestamp']).seconds < 3600
            ]
            
            health_metrics = {
                'model_id': model_id,
                'health_score': 0.0,
                'status': 'unknown',
                'last_updated': current_time.isoformat(),
                'metrics': {}
            }
            
            scores = []
            
            # 1. Prediction Volume Health (20%)
            prediction_volume = len(recent_predictions)
            expected_volume = 100  # Expected predictions per hour
            volume_score = min(1.0, prediction_volume / expected_volume)
            scores.append(('prediction_volume', volume_score, 0.2))
            
            health_metrics['metrics']['prediction_volume'] = {
                'current': prediction_volume,
                'expected': expected_volume,
                'score': volume_score,
                'status': 'healthy' if volume_score > 0.8 else 'warning' if volume_score > 0.5 else 'critical'
            }
            
            # 2. Error Rate Health (25%)
            total_predictions = model_info['prediction_count']
            error_count = model_info['error_count']
            error_rate = error_count / max(1, total_predictions)
            error_score = max(0, 1 - (error_rate / self.performance_thresholds['error_rate_threshold']))
            scores.append(('error_rate', error_score, 0.25))
            
            health_metrics['metrics']['error_rate'] = {
                'current': error_rate,
                'threshold': self.performance_thresholds['error_rate_threshold'],
                'score': error_score,
                'total_errors': error_count,
                'status': 'healthy' if error_rate < 0.01 else 'warning' if error_rate < 0.05 else 'critical'
            }
            
            # 3. Performance Health (25%)
            performance_score = 1.0
            if recent_performance:
                latest_perf = recent_performance[-1]['metrics']
                baseline_perf = model_info['baseline_performance']
                
                for metric in model_info['performance_metrics']:
                    if metric in latest_perf and metric in baseline_perf:
                        current_val = latest_perf[metric]
                        baseline_val = baseline_perf[metric]
                        
                        # Calculate performance degradation
                        if baseline_val > 0:
                            degradation = abs(current_val - baseline_val) / baseline_val
                            performance_score = min(performance_score, max(0, 1 - degradation))
            
            scores.append(('performance', performance_score, 0.25))
            
            health_metrics['metrics']['performance'] = {
                'score': performance_score,
                'recent_samples': len(recent_performance),
                'status': 'healthy' if performance_score > 0.9 else 'warning' if performance_score > 0.7 else 'critical'
            }
            
            # 4. Latency Health (15%)
            latencies = [pred.get('latency_ms', 0) for pred in recent_predictions if pred.get('latency_ms')]
            if latencies:
                avg_latency = np.mean(latencies)
                latency_score = max(0, 1 - (avg_latency / self.performance_thresholds['latency_threshold_ms']))
            else:
                avg_latency = 0
                latency_score = 1.0
                
            scores.append(('latency', latency_score, 0.15))
            
            health_metrics['metrics']['latency'] = {
                'current_avg_ms': avg_latency,
                'threshold_ms': self.performance_thresholds['latency_threshold_ms'],
                'score': latency_score,
                'p95_ms': np.percentile(latencies, 95) if latencies else 0,
                'status': 'healthy' if avg_latency < 500 else 'warning' if avg_latency < 1000 else 'critical'
            }
            
            # 5. Freshness Health (15%)
            last_prediction_time = model_info.get('last_prediction_time', model_info['registration_time'])
            time_since_last = (current_time - last_prediction_time).seconds
            freshness_score = max(0, 1 - (time_since_last / 3600))  # Degrade over 1 hour
            scores.append(('freshness', freshness_score, 0.15))
            
            health_metrics['metrics']['freshness'] = {
                'last_prediction_ago_seconds': time_since_last,
                'score': freshness_score,
                'status': 'healthy' if time_since_last < 300 else 'warning' if time_since_last < 1800 else 'critical'
            }
            
            # Calculate weighted overall health score
            overall_score = sum(score * weight for _, score, weight in scores)
            health_metrics['health_score'] = overall_score
            
            # Determine overall status
            if overall_score >= 0.9:
                health_metrics['status'] = 'healthy'
            elif overall_score >= 0.7:
                health_metrics['status'] = 'warning'
            else:
                health_metrics['status'] = 'critical'
            
            # Cache the result
            self.model_health_cache[model_id] = health_metrics
            
            return health_metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate health for model {model_id}: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'model_id': model_id
            }
    
    async def get_performance_metrics(
        self,
        model_id: str,
        time_window: timedelta = timedelta(hours=24),
        aggregation: str = 'mean'
    ) -> Dict[str, Any]:
        """
        Get performance metrics for a model over a time window
        
        Args:
            model_id: Model identifier
            time_window: Time window for metrics
            aggregation: Aggregation method (mean, min, max, latest)
        
        Returns:
            Performance metrics data
        """
        try:
            if model_id not in self.monitored_models:
                raise ValueError(f"Model {model_id} not registered for monitoring")
            
            current_time = datetime.utcnow()
            cutoff_time = current_time - time_window
            
            # Filter performance history by time window
            recent_history = [
                perf for perf in self.performance_history[model_id]
                if perf['timestamp'] >= cutoff_time
            ]
            
            if not recent_history:
                return {
                    'status': 'success',
                    'model_id': model_id,
                    'metrics': {},
                    'time_window_hours': time_window.total_seconds() / 3600,
                    'sample_count': 0,
                    'message': 'No performance data available for the specified time window'
                }
            
            # Aggregate metrics
            all_metrics = {}
            for perf in recent_history:
                for metric, value in perf['metrics'].items():
                    if metric not in all_metrics:
                        all_metrics[metric] = []
                    all_metrics[metric].append(value)
            
            # Calculate aggregated values
            aggregated_metrics = {}
            for metric, values in all_metrics.items():
                if aggregation == 'mean':
                    aggregated_metrics[metric] = {
                        'value': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values))
                    }
                elif aggregation == 'latest':
                    aggregated_metrics[metric] = {
                        'value': float(values[-1]),
                        'previous': float(values[-2]) if len(values) > 1 else None,
                        'trend': 'improving' if len(values) > 1 and values[-1] > values[-2] else 'degrading'
                    }
                elif aggregation == 'min':
                    aggregated_metrics[metric] = {'value': float(np.min(values))}
                elif aggregation == 'max':
                    aggregated_metrics[metric] = {'value': float(np.max(values))}
            
            # Calculate trends
            trends = {}
            if len(recent_history) > 1:
                for metric in all_metrics:
                    values = all_metrics[metric]
                    if len(values) > 3:
                        # Simple linear trend
                        x = np.arange(len(values))
                        slope = np.polyfit(x, values, 1)[0]
                        trends[metric] = {
                            'slope': float(slope),
                            'direction': 'improving' if slope > 0 else 'degrading' if slope < 0 else 'stable',
                            'confidence': 'high' if abs(slope) > 0.01 else 'low'
                        }
            
            return {
                'status': 'success',
                'model_id': model_id,
                'metrics': aggregated_metrics,
                'trends': trends,
                'time_window_hours': time_window.total_seconds() / 3600,
                'sample_count': len(recent_history),
                'aggregation_method': aggregation,
                'baseline_comparison': self._compare_to_baseline(model_id, aggregated_metrics)
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance metrics for model {model_id}: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'model_id': model_id
            }
    
    async def create_alert_rule(
        self,
        model_id: str,
        alert_name: str,
        condition: Dict[str, Any],
        action: Dict[str, Any],
        enabled: bool = True
    ) -> Dict[str, Any]:
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
            alert_rule = {
                'alert_id': str(uuid.uuid4()),
                'model_id': model_id,
                'alert_name': alert_name,
                'condition': condition,
                'action': action,
                'enabled': enabled,
                'created_at': datetime.utcnow(),
                'last_triggered': None,
                'trigger_count': 0
            }
            
            self.alert_rules[model_id].append(alert_rule)
            
            logger.info(f"Alert rule '{alert_name}' created for model {model_id}")
            
            return {
                'status': 'success',
                'alert_id': alert_rule['alert_id'],
                'model_id': model_id,
                'alert_name': alert_name,
                'enabled': enabled
            }
            
        except Exception as e:
            logger.error(f"Failed to create alert rule for model {model_id}: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'model_id': model_id
            }
    
    async def _continuous_monitoring_loop(self, model_id: str):
        """
        Continuous monitoring loop for a model
        """
        try:
            while model_id in self.monitored_models:
                # Calculate current health
                health_metrics = await self.calculate_model_health(model_id)
                
                # Check alert conditions
                await self._check_alert_conditions(model_id, health_metrics)
                
                # Update last health check time
                self.monitored_models[model_id]['last_health_check'] = datetime.utcnow()
                
                # Wait for next monitoring cycle
                monitoring_interval = self.monitored_models[model_id]['config']['monitoring_interval']
                await asyncio.sleep(monitoring_interval)
                
        except Exception as e:
            logger.error(f"Monitoring loop error for model {model_id}: {e}")
            if model_id in self.monitored_models:
                self.monitored_models[model_id]['monitoring_status'] = 'error'
    
    async def _check_prediction_alerts(self, model_id: str, prediction_log: Dict[str, Any]):
        """
        Check for alerts based on individual predictions
        """
        try:
            # Check latency alerts
            if prediction_log.get('latency_ms', 0) > self.performance_thresholds['latency_threshold_ms']:
                await self._trigger_alert(model_id, 'high_latency', {
                    'prediction_id': prediction_log['prediction_id'],
                    'latency_ms': prediction_log['latency_ms'],
                    'threshold_ms': self.performance_thresholds['latency_threshold_ms']
                })
            
        except Exception as e:
            logger.error(f"Error checking prediction alerts: {e}")
    
    async def _check_alert_conditions(self, model_id: str, health_metrics: Dict[str, Any]):
        """
        Check if any alert conditions are met
        """
        try:
            for alert_rule in self.alert_rules[model_id]:
                if not alert_rule['enabled']:
                    continue
                
                condition = alert_rule['condition']
                
                # Check condition
                if self._evaluate_alert_condition(condition, health_metrics):
                    await self._trigger_alert(model_id, alert_rule['alert_name'], {
                        'alert_id': alert_rule['alert_id'],
                        'health_metrics': health_metrics,
                        'condition': condition
                    })
                    
                    alert_rule['last_triggered'] = datetime.utcnow()
                    alert_rule['trigger_count'] += 1
            
        except Exception as e:
            logger.error(f"Error checking alert conditions: {e}")
    
    def _evaluate_alert_condition(self, condition: Dict[str, Any], health_metrics: Dict[str, Any]) -> bool:
        """
        Evaluate if an alert condition is met
        """
        try:
            condition_type = condition.get('type', 'threshold')
            
            if condition_type == 'threshold':
                metric_path = condition['metric']
                operator = condition['operator']  # 'gt', 'lt', 'eq', 'gte', 'lte'
                threshold = condition['threshold']
                
                # Navigate to metric value
                value = health_metrics
                for key in metric_path.split('.'):
                    if isinstance(value, dict) and key in value:
                        value = value[key]
                    else:
                        return False
                
                # Evaluate condition
                if operator == 'gt':
                    return value > threshold
                elif operator == 'lt':
                    return value < threshold
                elif operator == 'gte':
                    return value >= threshold
                elif operator == 'lte':
                    return value <= threshold
                elif operator == 'eq':
                    return value == threshold
                    
            elif condition_type == 'status':
                status = health_metrics.get('status', 'unknown')
                target_status = condition['status']
                return status == target_status
                
            return False
            
        except Exception as e:
            logger.error(f"Error evaluating alert condition: {e}")
            return False
    
    async def _trigger_alert(self, model_id: str, alert_type: str, context: Dict[str, Any]):
        """
        Trigger an alert
        """
        try:
            alert = {
                'alert_id': str(uuid.uuid4()),
                'model_id': model_id,
                'alert_type': alert_type,
                'timestamp': datetime.utcnow(),
                'context': context,
                'severity': self._determine_alert_severity(alert_type, context),
                'status': 'active'
            }
            
            # Log alert
            logger.warning(f"ALERT: {alert_type} for model {model_id}: {context}")
            
            # Here you would typically send notifications (email, Slack, webhook, etc.)
            # For now, we'll just increment the alert count
            self.monitored_models[model_id]['alert_count'] += 1
            
        except Exception as e:
            logger.error(f"Error triggering alert: {e}")
    
    def _determine_alert_severity(self, alert_type: str, context: Dict[str, Any]) -> str:
        """
        Determine alert severity based on type and context
        """
        severity_map = {
            'high_latency': 'warning',
            'performance_degradation': 'critical',
            'high_error_rate': 'critical',
            'model_offline': 'critical',
            'low_prediction_volume': 'warning',
            'health_critical': 'critical',
            'health_warning': 'warning'
        }
        return severity_map.get(alert_type, 'info')
    
    def _compare_to_baseline(self, model_id: str, current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare current metrics to baseline
        """
        try:
            if model_id not in self.monitored_models:
                return {}
            
            baseline = self.monitored_models[model_id]['baseline_performance']
            comparison = {}
            
            for metric, current_data in current_metrics.items():
                if metric in baseline:
                    baseline_value = baseline[metric]
                    current_value = current_data.get('value', 0)
                    
                    if baseline_value > 0:
                        change_percent = ((current_value - baseline_value) / baseline_value) * 100
                    else:
                        change_percent = 0
                    
                    comparison[metric] = {
                        'baseline_value': baseline_value,
                        'current_value': current_value,
                        'change_percent': change_percent,
                        'change_direction': 'improvement' if change_percent > 0 else 'degradation' if change_percent < 0 else 'stable'
                    }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing to baseline: {e}")
            return {}
    
    async def get_monitoring_summary(self) -> Dict[str, Any]:
        """
        Get summary of all monitored models
        """
        try:
            summary = {
                'total_models': len(self.monitored_models),
                'healthy_models': 0,
                'warning_models': 0,
                'critical_models': 0,
                'models': []
            }
            
            for model_id, model_info in self.monitored_models.items():
                health = self.model_health_cache.get(model_id, {'status': 'unknown', 'health_score': 0})
                
                model_summary = {
                    'model_id': model_id,
                    'model_name': model_info['model_name'],
                    'status': health['status'],
                    'health_score': health['health_score'],
                    'prediction_count': model_info['prediction_count'],
                    'error_count': model_info['error_count'],
                    'alert_count': model_info['alert_count'],
                    'last_prediction': model_info.get('last_prediction_time', model_info['registration_time']).isoformat()
                }
                
                summary['models'].append(model_summary)
                
                # Update counters
                if health['status'] == 'healthy':
                    summary['healthy_models'] += 1
                elif health['status'] == 'warning':
                    summary['warning_models'] += 1
                elif health['status'] == 'critical':
                    summary['critical_models'] += 1
            
            return {
                'status': 'success',
                'summary': summary,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting monitoring summary: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }

# Global service instance
model_monitoring_service = ModelMonitoringService()