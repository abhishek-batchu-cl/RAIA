# Real-time Model Monitoring Engine
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, func
from datetime import datetime, timedelta
import json
import logging
from collections import deque, defaultdict
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import statistics
from scipy import stats
from scipy.spatial.distance import wasserstein_distance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import requests
import warnings

from .models import (
    MonitoringConfiguration, ModelMonitoringMetric, DataDriftReport,
    PerformanceDegradationAlert, ModelHealthScore, RealTimeMonitoringStream
)
from ..model_registry.models import Model
from ..exceptions import MonitoringError, ValidationError

logger = logging.getLogger(__name__)

class RealTimeMonitoringEngine:
    """Real-time model monitoring and alerting engine"""
    
    def __init__(self, db: Session, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        
        # Monitoring state
        self.is_running = False
        self.monitoring_tasks = {}
        self.metric_buffers = defaultdict(lambda: deque(maxlen=10000))  # Rolling buffers
        self.alert_cache = {}  # Prevent duplicate alerts
        
        # Performance tracking
        self.performance_baselines = {}
        self.drift_detectors = {}
        
        # Threading and async
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.monitoring_loop = None
        
        # Initialize alert handlers
        self.alert_handlers = {
            'email': self._send_email_alert,
            'slack': self._send_slack_alert,
            'webhook': self._send_webhook_alert
        }
    
    # ========================================================================================
    # CORE MONITORING ENGINE
    # ========================================================================================
    
    async def start_monitoring(self):
        """Start the real-time monitoring engine"""
        if self.is_running:
            logger.warning("Monitoring engine is already running")
            return
        
        self.is_running = True
        logger.info("Starting real-time monitoring engine...")
        
        # Load active monitoring configurations
        await self._load_monitoring_configurations()
        
        # Start monitoring loop
        self.monitoring_loop = asyncio.create_task(self._monitoring_loop())
        
        # Start real-time stream processing
        asyncio.create_task(self._process_realtime_stream())
        
        logger.info("Real-time monitoring engine started successfully")
    
    async def stop_monitoring(self):
        """Stop the monitoring engine"""
        if not self.is_running:
            return
        
        logger.info("Stopping real-time monitoring engine...")
        self.is_running = False
        
        # Cancel monitoring loop
        if self.monitoring_loop:
            self.monitoring_loop.cancel()
        
        # Cancel all monitoring tasks
        for task in self.monitoring_tasks.values():
            task.cancel()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Real-time monitoring engine stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                # Process each active monitoring configuration
                active_configs = self.db.query(MonitoringConfiguration).filter(
                    MonitoringConfiguration.is_active == True
                ).all()
                
                tasks = []
                for config in active_configs:
                    task = asyncio.create_task(self._process_monitoring_config(config))
                    tasks.append(task)
                
                # Wait for all monitoring tasks to complete
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                
                # Sleep until next monitoring cycle
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(30)  # Wait longer on error
    
    async def _process_monitoring_config(self, config: MonitoringConfiguration):
        """Process a single monitoring configuration"""
        try:
            model_id = str(config.model_id)
            
            # Check if it's time to run monitoring for this config
            last_run = getattr(config, '_last_run', None)
            if last_run and (datetime.utcnow() - last_run).total_seconds() < config.monitoring_interval_seconds:
                return
            
            # Update last run time
            config._last_run = datetime.utcnow()
            
            # Performance monitoring
            if 'performance' in config.performance_metrics_enabled:
                await self._monitor_performance(config)
            
            # Drift monitoring
            if config.drift_detection_enabled:
                await self._monitor_drift(config)
            
            # Fairness monitoring
            if config.fairness_monitoring_enabled:
                await self._monitor_fairness(config)
            
            # Safety monitoring
            if config.safety_monitoring_enabled:
                await self._monitor_safety(config)
            
            # Update model health score
            await self._update_model_health_score(config)
            
        except Exception as e:
            logger.error(f"Error processing monitoring config {config.id}: {str(e)}")
    
    # ========================================================================================
    # PERFORMANCE MONITORING
    # ========================================================================================
    
    async def _monitor_performance(self, config: MonitoringConfiguration):
        """Monitor model performance metrics"""
        model_id = str(config.model_id)
        
        # Get recent predictions and performance data
        performance_data = await self._get_recent_performance_data(model_id, config.batch_size)
        
        if not performance_data:
            return
        
        # Calculate performance metrics
        metrics = await self._calculate_performance_metrics(performance_data)
        
        # Store metrics in database
        await self._store_performance_metrics(model_id, config.id, metrics)
        
        # Check for performance degradation
        await self._check_performance_thresholds(model_id, config, metrics)
    
    async def _get_recent_performance_data(self, model_id: str, batch_size: int) -> List[Dict[str, Any]]:
        """Get recent performance data for a model"""
        try:
            # Query recent predictions with actual outcomes
            # This is a simplified query - in practice, you'd join with prediction and outcome tables
            recent_data = []
            
            # Simulate getting performance data
            # In production, this would query actual prediction logs
            for i in range(min(batch_size, 100)):
                recent_data.append({
                    'prediction_id': f'pred_{i}',
                    'predicted_value': np.random.random(),
                    'actual_value': np.random.random(),
                    'response_time_ms': np.random.randint(10, 1000),
                    'confidence_score': np.random.random(),
                    'timestamp': datetime.utcnow() - timedelta(minutes=np.random.randint(0, 60))
                })
            
            return recent_data
            
        except Exception as e:
            logger.error(f"Error getting performance data for model {model_id}: {str(e)}")
            return []
    
    async def _calculate_performance_metrics(self, performance_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate performance metrics from recent data"""
        if not performance_data:
            return {}
        
        metrics = {}
        
        try:
            # Extract values
            predicted_values = [d['predicted_value'] for d in performance_data if 'predicted_value' in d]
            actual_values = [d['actual_value'] for d in performance_data if 'actual_value' in d]
            response_times = [d['response_time_ms'] for d in performance_data if 'response_time_ms' in d]
            confidence_scores = [d['confidence_score'] for d in performance_data if 'confidence_score' in d]
            
            # Accuracy (for classification) or MAE (for regression)
            if predicted_values and actual_values and len(predicted_values) == len(actual_values):
                # Assume binary classification for simplicity
                pred_binary = [1 if p > 0.5 else 0 for p in predicted_values]
                actual_binary = [1 if a > 0.5 else 0 for a in actual_values]
                
                metrics['accuracy'] = accuracy_score(actual_binary, pred_binary)
                
                # Additional metrics
                if len(set(actual_binary)) > 1:  # Avoid division by zero
                    metrics['precision'] = precision_score(actual_binary, pred_binary, average='weighted', zero_division=0)
                    metrics['recall'] = recall_score(actual_binary, pred_binary, average='weighted', zero_division=0)
                    metrics['f1_score'] = f1_score(actual_binary, pred_binary, average='weighted', zero_division=0)
                
                # Mean Absolute Error
                mae = np.mean([abs(p - a) for p, a in zip(predicted_values, actual_values)])
                metrics['mae'] = mae
            
            # Latency metrics
            if response_times:
                metrics['avg_response_time'] = np.mean(response_times)
                metrics['p95_response_time'] = np.percentile(response_times, 95)
                metrics['p99_response_time'] = np.percentile(response_times, 99)
            
            # Throughput (predictions per unit time)
            if performance_data:
                time_span = max(d['timestamp'] for d in performance_data) - min(d['timestamp'] for d in performance_data)
                if time_span.total_seconds() > 0:
                    metrics['throughput_rps'] = len(performance_data) / time_span.total_seconds()
            
            # Confidence metrics
            if confidence_scores:
                metrics['avg_confidence'] = np.mean(confidence_scores)
                metrics['confidence_std'] = np.std(confidence_scores)
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
        
        return metrics
    
    async def _store_performance_metrics(self, model_id: str, config_id: str, metrics: Dict[str, float]):
        """Store performance metrics in database"""
        try:
            timestamp = datetime.utcnow()
            
            for metric_name, value in metrics.items():
                metric = ModelMonitoringMetric(
                    model_id=model_id,
                    monitoring_config_id=config_id,
                    metric_name=metric_name,
                    metric_category='performance',
                    metric_type='gauge',
                    metric_value=value,
                    timestamp=timestamp,
                    time_window_start=timestamp - timedelta(minutes=5),
                    time_window_end=timestamp
                )
                
                self.db.add(metric)
            
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Error storing performance metrics: {str(e)}")
            self.db.rollback()
    
    async def _check_performance_thresholds(self, model_id: str, config: MonitoringConfiguration, metrics: Dict[str, float]):
        """Check if performance metrics exceed thresholds"""
        if not config.performance_thresholds:
            return
        
        thresholds = config.performance_thresholds
        violations = []
        
        for metric_name, value in metrics.items():
            if metric_name in thresholds:
                threshold_config = thresholds[metric_name]
                
                if isinstance(threshold_config, dict):
                    min_threshold = threshold_config.get('min')
                    max_threshold = threshold_config.get('max')
                    
                    if min_threshold is not None and value < min_threshold:
                        violations.append({
                            'metric': metric_name,
                            'value': value,
                            'threshold': min_threshold,
                            'type': 'below_minimum'
                        })
                    
                    if max_threshold is not None and value > max_threshold:
                        violations.append({
                            'metric': metric_name,
                            'value': value,
                            'threshold': max_threshold,
                            'type': 'above_maximum'
                        })
        
        # Trigger alerts for violations
        if violations:
            await self._trigger_performance_alert(model_id, config, violations)
    
    # ========================================================================================
    # DRIFT MONITORING
    # ========================================================================================
    
    async def _monitor_drift(self, config: MonitoringConfiguration):
        """Monitor for data and concept drift"""
        model_id = str(config.model_id)
        
        # Get reference and comparison data
        reference_data = await self._get_reference_data(model_id, config)
        comparison_data = await self._get_comparison_data(model_id, config)
        
        if not reference_data or not comparison_data:
            logger.debug(f"Insufficient data for drift monitoring on model {model_id}")
            return
        
        # Detect drift
        drift_results = await self._detect_drift(reference_data, comparison_data, config)
        
        # Store drift report
        if drift_results:
            await self._store_drift_report(model_id, config, drift_results)
    
    async def _get_reference_data(self, model_id: str, config: MonitoringConfiguration) -> Optional[pd.DataFrame]:
        """Get reference data for drift detection"""
        try:
            # This would typically query a reference dataset or training data
            # For simulation, create reference data
            np.random.seed(42)  # Ensure consistency
            n_samples = config.reference_window_size
            
            reference_data = pd.DataFrame({
                'feature_1': np.random.normal(0, 1, n_samples),
                'feature_2': np.random.exponential(1, n_samples),
                'feature_3': np.random.uniform(-1, 1, n_samples),
                'prediction': np.random.random(n_samples)
            })
            
            return reference_data
            
        except Exception as e:
            logger.error(f"Error getting reference data: {str(e)}")
            return None
    
    async def _get_comparison_data(self, model_id: str, config: MonitoringConfiguration) -> Optional[pd.DataFrame]:
        """Get recent data for drift comparison"""
        try:
            # This would typically query recent prediction data
            # For simulation, create comparison data with some drift
            n_samples = config.comparison_window_size
            
            # Simulate drift by shifting distributions
            comparison_data = pd.DataFrame({
                'feature_1': np.random.normal(0.2, 1.1, n_samples),  # Slight shift in mean and variance
                'feature_2': np.random.exponential(1.2, n_samples),  # Different scale
                'feature_3': np.random.uniform(-0.8, 1.2, n_samples),  # Different range
                'prediction': np.random.random(n_samples)
            })
            
            return comparison_data
            
        except Exception as e:
            logger.error(f"Error getting comparison data: {str(e)}")
            return None
    
    async def _detect_drift(self, reference_data: pd.DataFrame, comparison_data: pd.DataFrame, 
                           config: MonitoringConfiguration) -> Optional[Dict[str, Any]]:
        """Detect drift between reference and comparison data"""
        try:
            method = config.drift_detection_method
            threshold = float(config.drift_threshold)
            
            drift_results = {
                'drift_detected': False,
                'drift_score': 0.0,
                'p_value': 1.0,
                'test_statistic': 0.0,
                'feature_drift_scores': {},
                'most_drifted_features': [],
                'method': method
            }
            
            feature_drift_scores = {}
            p_values = []
            
            # Test each feature for drift
            for column in reference_data.columns:
                if column in comparison_data.columns:
                    ref_values = reference_data[column].dropna()
                    comp_values = comparison_data[column].dropna()
                    
                    if len(ref_values) == 0 or len(comp_values) == 0:
                        continue
                    
                    # Perform statistical test based on method
                    if method == 'ks_test':
                        statistic, p_value = stats.ks_2samp(ref_values, comp_values)
                        drift_score = statistic
                    elif method == 'wasserstein':
                        drift_score = wasserstein_distance(ref_values, comp_values)
                        # Convert to p-value approximation
                        p_value = 1 - min(1.0, drift_score)
                    elif method == 'psi':
                        # Population Stability Index
                        drift_score = self._calculate_psi(ref_values, comp_values)
                        p_value = 1 - min(1.0, drift_score / 0.25)  # PSI > 0.25 indicates significant drift
                    else:
                        # Default to KS test
                        statistic, p_value = stats.ks_2samp(ref_values, comp_values)
                        drift_score = statistic
                    
                    feature_drift_scores[column] = {
                        'drift_score': float(drift_score),
                        'p_value': float(p_value),
                        'drift_detected': p_value < threshold
                    }
                    
                    p_values.append(p_value)
            
            # Overall drift assessment
            if feature_drift_scores:
                # Use minimum p-value (most significant drift)
                min_p_value = min(p_values)
                max_drift_score = max([f['drift_score'] for f in feature_drift_scores.values()])
                
                drift_results['drift_detected'] = min_p_value < threshold
                drift_results['p_value'] = float(min_p_value)
                drift_results['drift_score'] = float(max_drift_score)
                drift_results['feature_drift_scores'] = feature_drift_scores
                
                # Find most drifted features
                sorted_features = sorted(
                    feature_drift_scores.items(),
                    key=lambda x: x[1]['drift_score'],
                    reverse=True
                )
                drift_results['most_drifted_features'] = [f[0] for f in sorted_features[:5]]
            
            return drift_results
            
        except Exception as e:
            logger.error(f"Error detecting drift: {str(e)}")
            return None
    
    def _calculate_psi(self, reference: np.ndarray, comparison: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Population Stability Index (PSI)"""
        try:
            # Create bins based on reference data
            _, bin_edges = np.histogram(reference, bins=n_bins)
            
            # Calculate frequencies for both datasets
            ref_freq, _ = np.histogram(reference, bins=bin_edges)
            comp_freq, _ = np.histogram(comparison, bins=bin_edges)
            
            # Convert to proportions
            ref_prop = ref_freq / len(reference)
            comp_prop = comp_freq / len(comparison)
            
            # Avoid division by zero
            ref_prop = np.where(ref_prop == 0, 0.0001, ref_prop)
            comp_prop = np.where(comp_prop == 0, 0.0001, comp_prop)
            
            # Calculate PSI
            psi = np.sum((comp_prop - ref_prop) * np.log(comp_prop / ref_prop))
            
            return float(psi)
            
        except Exception as e:
            logger.error(f"Error calculating PSI: {str(e)}")
            return 0.0
    
    async def _store_drift_report(self, model_id: str, config: MonitoringConfiguration, drift_results: Dict[str, Any]):
        """Store drift detection report"""
        try:
            # Determine drift severity
            drift_score = drift_results['drift_score']
            if drift_score > 0.5:
                severity = 'severe'
            elif drift_score > 0.3:
                severity = 'high'
            elif drift_score > 0.1:
                severity = 'medium'
            else:
                severity = 'low'
            
            # Create drift report
            report = DataDriftReport(
                model_id=model_id,
                monitoring_config_id=config.id,
                drift_type='feature_drift',
                detection_method=config.drift_detection_method,
                timestamp=datetime.utcnow(),
                reference_period_start=datetime.utcnow() - timedelta(days=7),
                reference_period_end=datetime.utcnow() - timedelta(days=1),
                comparison_period_start=datetime.utcnow() - timedelta(hours=1),
                comparison_period_end=datetime.utcnow(),
                drift_detected=drift_results['drift_detected'],
                drift_score=drift_results['drift_score'],
                drift_severity=severity,
                p_value=drift_results['p_value'],
                test_statistic=drift_results.get('test_statistic', 0.0),
                significance_threshold=float(config.drift_threshold),
                feature_drift_scores=drift_results['feature_drift_scores'],
                most_drifted_features=drift_results['most_drifted_features'],
                reference_sample_size=config.reference_window_size,
                comparison_sample_size=config.comparison_window_size,
                recommended_actions=self._generate_drift_recommendations(drift_results),
                retrain_recommended=drift_results['drift_detected'] and severity in ['high', 'severe'],
                urgency_level=severity
            )
            
            self.db.add(report)
            self.db.commit()
            
            # Trigger alert if drift detected
            if drift_results['drift_detected']:
                await self._trigger_drift_alert(model_id, config, report)
            
        except Exception as e:
            logger.error(f"Error storing drift report: {str(e)}")
            self.db.rollback()
    
    def _generate_drift_recommendations(self, drift_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on drift results"""
        recommendations = []
        
        if not drift_results['drift_detected']:
            recommendations.append("No significant drift detected. Continue monitoring.")
            return recommendations
        
        drift_score = drift_results['drift_score']
        most_drifted = drift_results.get('most_drifted_features', [])
        
        if drift_score > 0.3:
            recommendations.append("Significant drift detected. Consider model retraining.")
            recommendations.append("Analyze root causes of drift in the data pipeline.")
        
        if most_drifted:
            recommendations.append(f"Focus analysis on features: {', '.join(most_drifted[:3])}")
        
        recommendations.append("Increase monitoring frequency temporarily.")
        recommendations.append("Review data quality and collection processes.")
        
        return recommendations
    
    # ========================================================================================
    # FAIRNESS MONITORING
    # ========================================================================================
    
    async def _monitor_fairness(self, config: MonitoringConfiguration):
        """Monitor fairness metrics"""
        if not config.protected_attributes or not config.fairness_metrics:
            return
        
        model_id = str(config.model_id)
        
        # Get recent predictions with protected attributes
        fairness_data = await self._get_fairness_data(model_id, config)
        
        if not fairness_data:
            return
        
        # Calculate fairness metrics
        fairness_metrics = await self._calculate_fairness_metrics(fairness_data, config)
        
        # Store and check thresholds
        await self._store_fairness_metrics(model_id, config.id, fairness_metrics)
        await self._check_fairness_thresholds(model_id, config, fairness_metrics)
    
    async def _get_fairness_data(self, model_id: str, config: MonitoringConfiguration) -> List[Dict[str, Any]]:
        """Get recent prediction data with protected attributes"""
        try:
            # Simulate fairness data
            data = []
            for i in range(100):
                data.append({
                    'prediction': np.random.random() > 0.5,
                    'actual': np.random.random() > 0.5,
                    'protected_attributes': {
                        'gender': np.random.choice(['male', 'female']),
                        'race': np.random.choice(['white', 'black', 'hispanic', 'asian']),
                        'age_group': np.random.choice(['young', 'middle', 'senior'])
                    }
                })
            return data
            
        except Exception as e:
            logger.error(f"Error getting fairness data: {str(e)}")
            return []
    
    async def _calculate_fairness_metrics(self, fairness_data: List[Dict[str, Any]], 
                                        config: MonitoringConfiguration) -> Dict[str, float]:
        """Calculate fairness metrics"""
        metrics = {}
        
        try:
            df = pd.DataFrame(fairness_data)
            
            for attr in config.protected_attributes:
                if attr not in df.iloc[0]['protected_attributes']:
                    continue
                
                # Extract protected attribute values
                attr_values = [d['protected_attributes'][attr] for d in fairness_data]
                predictions = [d['prediction'] for d in fairness_data]
                actuals = [d['actual'] for d in fairness_data]
                
                # Demographic Parity
                if 'demographic_parity' in config.fairness_metrics:
                    dp_score = self._calculate_demographic_parity(attr_values, predictions)
                    metrics[f'{attr}_demographic_parity'] = dp_score
                
                # Equalized Odds
                if 'equalized_odds' in config.fairness_metrics:
                    eo_score = self._calculate_equalized_odds(attr_values, predictions, actuals)
                    metrics[f'{attr}_equalized_odds'] = eo_score
            
        except Exception as e:
            logger.error(f"Error calculating fairness metrics: {str(e)}")
        
        return metrics
    
    def _calculate_demographic_parity(self, protected_attr: List[str], predictions: List[bool]) -> float:
        """Calculate demographic parity score"""
        try:
            groups = list(set(protected_attr))
            if len(groups) < 2:
                return 1.0
            
            group_rates = {}
            for group in groups:
                group_preds = [p for i, p in enumerate(predictions) if protected_attr[i] == group]
                if group_preds:
                    group_rates[group] = sum(group_preds) / len(group_preds)
            
            if len(group_rates) < 2:
                return 1.0
            
            rates = list(group_rates.values())
            return min(rates) / max(rates) if max(rates) > 0 else 1.0
            
        except Exception as e:
            logger.error(f"Error calculating demographic parity: {str(e)}")
            return 1.0
    
    def _calculate_equalized_odds(self, protected_attr: List[str], predictions: List[bool], 
                                actuals: List[bool]) -> float:
        """Calculate equalized odds score"""
        try:
            groups = list(set(protected_attr))
            if len(groups) < 2:
                return 1.0
            
            group_tpr = {}  # True Positive Rate
            group_fpr = {}  # False Positive Rate
            
            for group in groups:
                group_indices = [i for i, attr in enumerate(protected_attr) if attr == group]
                group_preds = [predictions[i] for i in group_indices]
                group_actuals = [actuals[i] for i in group_indices]
                
                # Calculate TPR and FPR
                tp = sum(1 for i in range(len(group_preds)) if group_preds[i] and group_actuals[i])
                fp = sum(1 for i in range(len(group_preds)) if group_preds[i] and not group_actuals[i])
                tn = sum(1 for i in range(len(group_preds)) if not group_preds[i] and not group_actuals[i])
                fn = sum(1 for i in range(len(group_preds)) if not group_preds[i] and group_actuals[i])
                
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                
                group_tpr[group] = tpr
                group_fpr[group] = fpr
            
            # Calculate equalized odds as minimum ratio
            tpr_values = list(group_tpr.values())
            fpr_values = list(group_fpr.values())
            
            tpr_ratio = min(tpr_values) / max(tpr_values) if max(tpr_values) > 0 else 1.0
            fpr_ratio = min(fpr_values) / max(fpr_values) if max(fpr_values) > 0 else 1.0
            
            return min(tpr_ratio, fpr_ratio)
            
        except Exception as e:
            logger.error(f"Error calculating equalized odds: {str(e)}")
            return 1.0
    
    async def _store_fairness_metrics(self, model_id: str, config_id: str, metrics: Dict[str, float]):
        """Store fairness metrics"""
        try:
            timestamp = datetime.utcnow()
            
            for metric_name, value in metrics.items():
                metric = ModelMonitoringMetric(
                    model_id=model_id,
                    monitoring_config_id=config_id,
                    metric_name=metric_name,
                    metric_category='fairness',
                    metric_type='gauge',
                    metric_value=value,
                    timestamp=timestamp
                )
                self.db.add(metric)
            
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Error storing fairness metrics: {str(e)}")
            self.db.rollback()
    
    async def _check_fairness_thresholds(self, model_id: str, config: MonitoringConfiguration, 
                                       metrics: Dict[str, float]):
        """Check fairness metric thresholds"""
        if not config.fairness_thresholds:
            return
        
        violations = []
        for metric_name, value in metrics.items():
            if metric_name in config.fairness_thresholds:
                threshold = config.fairness_thresholds[metric_name]
                if value < threshold:
                    violations.append({
                        'metric': metric_name,
                        'value': value,
                        'threshold': threshold,
                        'type': 'fairness_violation'
                    })
        
        if violations:
            await self._trigger_fairness_alert(model_id, config, violations)
    
    # ========================================================================================
    # SAFETY MONITORING
    # ========================================================================================
    
    async def _monitor_safety(self, config: MonitoringConfiguration):
        """Monitor safety metrics"""
        model_id = str(config.model_id)
        
        # Get recent model outputs
        safety_data = await self._get_safety_data(model_id, config)
        
        if not safety_data:
            return
        
        # Calculate safety metrics
        safety_metrics = await self._calculate_safety_metrics(safety_data, config)
        
        # Store and check thresholds
        await self._store_safety_metrics(model_id, config.id, safety_metrics)
        await self._check_safety_thresholds(model_id, config, safety_metrics)
    
    async def _get_safety_data(self, model_id: str, config: MonitoringConfiguration) -> List[Dict[str, Any]]:
        """Get recent model outputs for safety analysis"""
        try:
            # Simulate safety data
            outputs = []
            for i in range(50):
                # Simulate text outputs with varying safety levels
                toxic_words = ['hate', 'violence', 'harmful'] if np.random.random() < 0.05 else []
                bias_indicators = ['always', 'never', 'all'] if np.random.random() < 0.1 else []
                
                outputs.append({
                    'output_text': f"Sample output {i} with {' '.join(toxic_words + bias_indicators)}",
                    'toxicity_score': np.random.random() * 0.3 if not toxic_words else np.random.uniform(0.7, 1.0),
                    'bias_score': np.random.random() * 0.2 if not bias_indicators else np.random.uniform(0.6, 1.0),
                    'timestamp': datetime.utcnow() - timedelta(minutes=np.random.randint(0, 60))
                })
            
            return outputs
            
        except Exception as e:
            logger.error(f"Error getting safety data: {str(e)}")
            return []
    
    async def _calculate_safety_metrics(self, safety_data: List[Dict[str, Any]], 
                                      config: MonitoringConfiguration) -> Dict[str, float]:
        """Calculate safety metrics"""
        if not safety_data:
            return {}
        
        metrics = {}
        
        try:
            # Toxicity metrics
            toxicity_scores = [d.get('toxicity_score', 0) for d in safety_data]
            metrics['avg_toxicity_score'] = np.mean(toxicity_scores)
            metrics['high_toxicity_rate'] = sum(1 for score in toxicity_scores if score > config.toxicity_threshold) / len(toxicity_scores)
            
            # Bias metrics
            bias_scores = [d.get('bias_score', 0) for d in safety_data]
            metrics['avg_bias_score'] = np.mean(bias_scores)
            metrics['high_bias_rate'] = sum(1 for score in bias_scores if score > config.bias_threshold) / len(bias_scores)
            
            # Content safety
            metrics['content_safety_score'] = 1.0 - max(metrics['avg_toxicity_score'], metrics['avg_bias_score'])
            
        except Exception as e:
            logger.error(f"Error calculating safety metrics: {str(e)}")
        
        return metrics
    
    async def _store_safety_metrics(self, model_id: str, config_id: str, metrics: Dict[str, float]):
        """Store safety metrics"""
        try:
            timestamp = datetime.utcnow()
            
            for metric_name, value in metrics.items():
                metric = ModelMonitoringMetric(
                    model_id=model_id,
                    monitoring_config_id=config_id,
                    metric_name=metric_name,
                    metric_category='safety',
                    metric_type='gauge',
                    metric_value=value,
                    timestamp=timestamp
                )
                self.db.add(metric)
            
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Error storing safety metrics: {str(e)}")
            self.db.rollback()
    
    async def _check_safety_thresholds(self, model_id: str, config: MonitoringConfiguration, 
                                     metrics: Dict[str, float]):
        """Check safety metric thresholds"""
        violations = []
        
        # Check toxicity threshold
        if 'high_toxicity_rate' in metrics and metrics['high_toxicity_rate'] > 0.05:  # 5% threshold
            violations.append({
                'metric': 'high_toxicity_rate',
                'value': metrics['high_toxicity_rate'],
                'threshold': 0.05,
                'type': 'safety_violation'
            })
        
        # Check bias threshold
        if 'high_bias_rate' in metrics and metrics['high_bias_rate'] > 0.1:  # 10% threshold
            violations.append({
                'metric': 'high_bias_rate',
                'value': metrics['high_bias_rate'],
                'threshold': 0.1,
                'type': 'safety_violation'
            })
        
        if violations:
            await self._trigger_safety_alert(model_id, config, violations)
    
    # ========================================================================================
    # HEALTH SCORE CALCULATION
    # ========================================================================================
    
    async def _update_model_health_score(self, config: MonitoringConfiguration):
        """Update model health score"""
        model_id = str(config.model_id)
        
        # Get recent metrics
        recent_metrics = await self._get_recent_metrics(model_id)
        
        # Calculate component scores
        health_components = await self._calculate_health_components(recent_metrics)
        
        # Calculate overall health score
        overall_score = self._calculate_overall_health_score(health_components)
        
        # Store health score
        await self._store_health_score(model_id, health_components, overall_score)
    
    async def _get_recent_metrics(self, model_id: str) -> Dict[str, List[float]]:
        """Get recent metrics for health score calculation"""
        try:
            # Query recent metrics from last 24 hours
            recent_time = datetime.utcnow() - timedelta(hours=24)
            
            metrics = self.db.query(ModelMonitoringMetric).filter(
                and_(
                    ModelMonitoringMetric.model_id == model_id,
                    ModelMonitoringMetric.timestamp >= recent_time
                )
            ).all()
            
            # Group metrics by name
            metric_groups = defaultdict(list)
            for metric in metrics:
                metric_groups[metric.metric_name].append(metric.metric_value)
            
            return dict(metric_groups)
            
        except Exception as e:
            logger.error(f"Error getting recent metrics: {str(e)}")
            return {}
    
    async def _calculate_health_components(self, recent_metrics: Dict[str, List[float]]) -> Dict[str, float]:
        """Calculate health component scores"""
        components = {}
        
        try:
            # Performance score (based on accuracy and latency)
            performance_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            performance_values = []
            for metric in performance_metrics:
                if metric in recent_metrics and recent_metrics[metric]:
                    performance_values.extend(recent_metrics[metric])
            
            components['performance_score'] = np.mean(performance_values) if performance_values else 0.5
            
            # Reliability score (based on error rates and uptime)
            error_metrics = ['error_rate', 'failure_rate']
            error_values = []
            for metric in error_metrics:
                if metric in recent_metrics and recent_metrics[metric]:
                    error_values.extend(recent_metrics[metric])
            
            avg_error_rate = np.mean(error_values) if error_values else 0.0
            components['reliability_score'] = max(0.0, 1.0 - avg_error_rate)
            
            # Stability score (based on variance in performance)
            if 'accuracy' in recent_metrics and len(recent_metrics['accuracy']) > 1:
                accuracy_std = np.std(recent_metrics['accuracy'])
                components['stability_score'] = max(0.0, 1.0 - accuracy_std)
            else:
                components['stability_score'] = 0.5
            
            # Fairness score
            fairness_metrics = [m for m in recent_metrics.keys() if 'demographic_parity' in m or 'equalized_odds' in m]
            fairness_values = []
            for metric in fairness_metrics:
                fairness_values.extend(recent_metrics[metric])
            
            components['fairness_score'] = np.mean(fairness_values) if fairness_values else 1.0
            
            # Safety score
            safety_metrics = ['content_safety_score', 'avg_toxicity_score', 'avg_bias_score']
            safety_values = []
            for metric in safety_metrics:
                if metric in recent_metrics:
                    if 'safety_score' in metric:
                        safety_values.extend(recent_metrics[metric])
                    else:
                        # Invert toxicity and bias scores
                        safety_values.extend([1.0 - v for v in recent_metrics[metric]])
            
            components['safety_score'] = np.mean(safety_values) if safety_values else 1.0
            
        except Exception as e:
            logger.error(f"Error calculating health components: {str(e)}")
        
        return components
    
    def _calculate_overall_health_score(self, components: Dict[str, float]) -> float:
        """Calculate overall health score from components"""
        weights = {
            'performance_score': 0.3,
            'reliability_score': 0.25,
            'stability_score': 0.2,
            'fairness_score': 0.15,
            'safety_score': 0.1
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for component, weight in weights.items():
            if component in components:
                total_score += components[component] * weight
                total_weight += weight
        
        return (total_score / total_weight * 100) if total_weight > 0 else 50.0  # 0-100 scale
    
    async def _store_health_score(self, model_id: str, components: Dict[str, float], overall_score: float):
        """Store model health score"""
        try:
            # Determine health grade and status
            grade = self._calculate_health_grade(overall_score)
            status = self._determine_health_status(overall_score)
            
            health_score = ModelHealthScore(
                model_id=model_id,
                timestamp=datetime.utcnow(),
                calculation_period_start=datetime.utcnow() - timedelta(hours=24),
                calculation_period_end=datetime.utcnow(),
                overall_health_score=overall_score,
                health_grade=grade,
                health_status=status,
                performance_score=components.get('performance_score'),
                reliability_score=components.get('reliability_score'),
                stability_score=components.get('stability_score'),
                fairness_score=components.get('fairness_score'),
                safety_score=components.get('safety_score'),
                score_trend=self._determine_score_trend(model_id, overall_score),
                health_recommendations=self._generate_health_recommendations(components, overall_score)
            )
            
            self.db.add(health_score)
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Error storing health score: {str(e)}")
            self.db.rollback()
    
    def _calculate_health_grade(self, score: float) -> str:
        """Convert health score to grade"""
        if score >= 95:
            return 'A+'
        elif score >= 90:
            return 'A'
        elif score >= 85:
            return 'B+'
        elif score >= 80:
            return 'B'
        elif score >= 75:
            return 'C+'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'
    
    def _determine_health_status(self, score: float) -> str:
        """Determine health status from score"""
        if score >= 90:
            return 'excellent'
        elif score >= 80:
            return 'good'
        elif score >= 70:
            return 'fair'
        elif score >= 60:
            return 'poor'
        else:
            return 'critical'
    
    def _determine_score_trend(self, model_id: str, current_score: float) -> str:
        """Determine score trend"""
        try:
            # Get last 5 health scores
            recent_scores = self.db.query(ModelHealthScore).filter(
                ModelHealthScore.model_id == model_id
            ).order_by(desc(ModelHealthScore.timestamp)).limit(5).all()
            
            if len(recent_scores) < 2:
                return 'stable'
            
            scores = [s.overall_health_score for s in recent_scores]
            
            # Calculate trend
            if len(scores) >= 3:
                recent_avg = np.mean(scores[:2])
                older_avg = np.mean(scores[2:])
                
                if recent_avg > older_avg + 5:
                    return 'improving'
                elif recent_avg < older_avg - 5:
                    return 'declining'
                else:
                    return 'stable'
            else:
                return 'stable'
                
        except Exception as e:
            logger.error(f"Error determining score trend: {str(e)}")
            return 'stable'
    
    def _generate_health_recommendations(self, components: Dict[str, float], overall_score: float) -> List[str]:
        """Generate health recommendations"""
        recommendations = []
        
        if overall_score < 70:
            recommendations.append("Overall health is below acceptable threshold. Immediate attention required.")
        
        # Component-specific recommendations
        for component, score in components.items():
            if score < 0.7:
                if component == 'performance_score':
                    recommendations.append("Performance metrics are below target. Consider model retraining or optimization.")
                elif component == 'reliability_score':
                    recommendations.append("Reliability issues detected. Review error logs and system stability.")
                elif component == 'stability_score':
                    recommendations.append("Performance instability detected. Investigate data quality and model consistency.")
                elif component == 'fairness_score':
                    recommendations.append("Fairness metrics indicate potential bias. Review model predictions across demographic groups.")
                elif component == 'safety_score':
                    recommendations.append("Safety concerns detected. Review content filtering and safety measures.")
        
        if not recommendations:
            recommendations.append("Model health is good. Continue regular monitoring.")
        
        return recommendations
    
    # ========================================================================================
    # ALERT SYSTEM
    # ========================================================================================
    
    async def _trigger_performance_alert(self, model_id: str, config: MonitoringConfiguration, violations: List[Dict]):
        """Trigger performance degradation alert"""
        alert_key = f"performance_{model_id}_{hash(str(violations))}"
        
        # Prevent duplicate alerts
        if alert_key in self.alert_cache and (datetime.utcnow() - self.alert_cache[alert_key]).total_seconds() < 3600:
            return
        
        self.alert_cache[alert_key] = datetime.utcnow()
        
        # Create alert
        alert = PerformanceDegradationAlert(
            model_id=model_id,
            monitoring_config_id=config.id,
            alert_type='performance_drop',
            severity=self._determine_alert_severity(violations),
            timestamp=datetime.utcnow(),
            first_detected_at=datetime.utcnow(),
            title=f"Performance Degradation Detected - Model {model_id}",
            description=f"Performance metrics have exceeded thresholds: {violations}",
            affected_metrics=[v['metric'] for v in violations],
            current_values={v['metric']: v['value'] for v in violations},
            threshold_values={v['metric']: v['threshold'] for v in violations}
        )
        
        self.db.add(alert)
        self.db.commit()
        
        # Send notifications
        await self._send_alert_notifications(config, alert, violations)
    
    async def _trigger_drift_alert(self, model_id: str, config: MonitoringConfiguration, drift_report: DataDriftReport):
        """Trigger drift detection alert"""
        alert_key = f"drift_{model_id}_{drift_report.drift_score}"
        
        if alert_key in self.alert_cache and (datetime.utcnow() - self.alert_cache[alert_key]).total_seconds() < 3600:
            return
        
        self.alert_cache[alert_key] = datetime.utcnow()
        
        alert = PerformanceDegradationAlert(
            model_id=model_id,
            monitoring_config_id=config.id,
            alert_type='drift_detected',
            severity=drift_report.drift_severity,
            timestamp=datetime.utcnow(),
            first_detected_at=datetime.utcnow(),
            title=f"Data Drift Detected - Model {model_id}",
            description=f"Significant drift detected with score {drift_report.drift_score:.3f}",
            affected_metrics=drift_report.most_drifted_features,
            current_values={'drift_score': drift_report.drift_score},
            threshold_values={'drift_threshold': float(config.drift_threshold)}
        )
        
        self.db.add(alert)
        self.db.commit()
        
        await self._send_alert_notifications(config, alert, [{'type': 'drift', 'score': drift_report.drift_score}])
    
    async def _trigger_fairness_alert(self, model_id: str, config: MonitoringConfiguration, violations: List[Dict]):
        """Trigger fairness violation alert"""
        alert_key = f"fairness_{model_id}_{hash(str(violations))}"
        
        if alert_key in self.alert_cache and (datetime.utcnow() - self.alert_cache[alert_key]).total_seconds() < 3600:
            return
        
        self.alert_cache[alert_key] = datetime.utcnow()
        
        alert = PerformanceDegradationAlert(
            model_id=model_id,
            monitoring_config_id=config.id,
            alert_type='fairness_violation',
            severity='high',
            timestamp=datetime.utcnow(),
            first_detected_at=datetime.utcnow(),
            title=f"Fairness Violation Detected - Model {model_id}",
            description=f"Fairness metrics below acceptable thresholds: {violations}",
            affected_metrics=[v['metric'] for v in violations],
            current_values={v['metric']: v['value'] for v in violations},
            threshold_values={v['metric']: v['threshold'] for v in violations}
        )
        
        self.db.add(alert)
        self.db.commit()
        
        await self._send_alert_notifications(config, alert, violations)
    
    async def _trigger_safety_alert(self, model_id: str, config: MonitoringConfiguration, violations: List[Dict]):
        """Trigger safety violation alert"""
        alert_key = f"safety_{model_id}_{hash(str(violations))}"
        
        if alert_key in self.alert_cache and (datetime.utcnow() - self.alert_cache[alert_key]).total_seconds() < 3600:
            return
        
        self.alert_cache[alert_key] = datetime.utcnow()
        
        alert = PerformanceDegradationAlert(
            model_id=model_id,
            monitoring_config_id=config.id,
            alert_type='safety_issue',
            severity='critical',
            timestamp=datetime.utcnow(),
            first_detected_at=datetime.utcnow(),
            title=f"Safety Issue Detected - Model {model_id}",
            description=f"Safety violations detected: {violations}",
            affected_metrics=[v['metric'] for v in violations],
            current_values={v['metric']: v['value'] for v in violations},
            threshold_values={v['metric']: v['threshold'] for v in violations}
        )
        
        self.db.add(alert)
        self.db.commit()
        
        await self._send_alert_notifications(config, alert, violations)
    
    def _determine_alert_severity(self, violations: List[Dict]) -> str:
        """Determine alert severity based on violations"""
        if not violations:
            return 'low'
        
        max_deviation = 0.0
        for violation in violations:
            if 'value' in violation and 'threshold' in violation:
                deviation = abs(violation['value'] - violation['threshold']) / violation['threshold']
                max_deviation = max(max_deviation, deviation)
        
        if max_deviation > 0.5:
            return 'critical'
        elif max_deviation > 0.3:
            return 'high'
        elif max_deviation > 0.1:
            return 'medium'
        else:
            return 'low'
    
    async def _send_alert_notifications(self, config: MonitoringConfiguration, 
                                      alert: PerformanceDegradationAlert, violations: List[Dict]):
        """Send alert notifications through configured channels"""
        if not config.alert_enabled:
            return
        
        for channel in config.alert_channels or ['email']:
            if channel in self.alert_handlers:
                try:
                    await self.alert_handlers[channel](config, alert, violations)
                except Exception as e:
                    logger.error(f"Error sending {channel} alert: {str(e)}")
    
    async def _send_email_alert(self, config: MonitoringConfiguration, 
                              alert: PerformanceDegradationAlert, violations: List[Dict]):
        """Send email alert"""
        if not config.alert_recipients:
            return
        
        # This is a simplified implementation
        # In production, you'd use a proper email service
        logger.info(f"EMAIL ALERT: {alert.title} - {alert.description}")
        logger.info(f"Recipients: {config.alert_recipients}")
        logger.info(f"Violations: {violations}")
    
    async def _send_slack_alert(self, config: MonitoringConfiguration, 
                              alert: PerformanceDegradationAlert, violations: List[Dict]):
        """Send Slack alert"""
        logger.info(f"SLACK ALERT: {alert.title} - {alert.description}")
    
    async def _send_webhook_alert(self, config: MonitoringConfiguration, 
                                alert: PerformanceDegradationAlert, violations: List[Dict]):
        """Send webhook alert"""
        logger.info(f"WEBHOOK ALERT: {alert.title} - {alert.description}")
    
    # ========================================================================================
    # REAL-TIME STREAM PROCESSING
    # ========================================================================================
    
    async def _process_realtime_stream(self):
        """Process real-time monitoring stream"""
        while self.is_running:
            try:
                # Process recent stream data
                await self._process_stream_batch()
                await asyncio.sleep(5)  # Process every 5 seconds
                
            except Exception as e:
                logger.error(f"Error processing real-time stream: {str(e)}")
                await asyncio.sleep(10)
    
    async def _process_stream_batch(self):
        """Process a batch of stream data"""
        try:
            # Get recent stream data (last 5 minutes)
            recent_time = datetime.utcnow() - timedelta(minutes=5)
            
            stream_data = self.db.query(RealTimeMonitoringStream).filter(
                RealTimeMonitoringStream.timestamp >= recent_time
            ).all()
            
            if not stream_data:
                return
            
            # Group by model
            model_streams = defaultdict(list)
            for stream in stream_data:
                model_streams[str(stream.model_id)].append(stream)
            
            # Process each model's stream
            for model_id, streams in model_streams.items():
                await self._process_model_stream(model_id, streams)
                
        except Exception as e:
            logger.error(f"Error processing stream batch: {str(e)}")
    
    async def _process_model_stream(self, model_id: str, streams: List[RealTimeMonitoringStream]):
        """Process stream data for a single model"""
        try:
            # Calculate real-time metrics
            response_times = [s.response_time_ms for s in streams if s.response_time_ms]
            error_count = sum(1 for s in streams if s.error_occurred)
            throughput = len(streams) / 300 if streams else 0  # per second over 5 minutes
            
            # Store aggregated metrics
            timestamp = datetime.utcnow()
            
            if response_times:
                avg_response_time = ModelMonitoringMetric(
                    model_id=model_id,
                    metric_name='realtime_avg_response_time',
                    metric_category='performance',
                    metric_type='gauge',
                    metric_value=np.mean(response_times),
                    metric_unit='milliseconds',
                    timestamp=timestamp,
                    sample_size=len(response_times)
                )
                self.db.add(avg_response_time)
            
            error_rate_metric = ModelMonitoringMetric(
                model_id=model_id,
                metric_name='realtime_error_rate',
                metric_category='reliability',
                metric_type='gauge',
                metric_value=error_count / len(streams) if streams else 0,
                metric_unit='percentage',
                timestamp=timestamp,
                sample_size=len(streams)
            )
            self.db.add(error_rate_metric)
            
            throughput_metric = ModelMonitoringMetric(
                model_id=model_id,
                metric_name='realtime_throughput',
                metric_category='performance',
                metric_type='gauge',
                metric_value=throughput,
                metric_unit='rps',
                timestamp=timestamp,
                sample_size=len(streams)
            )
            self.db.add(throughput_metric)
            
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Error processing model stream for {model_id}: {str(e)}")
            self.db.rollback()
    
    # ========================================================================================
    # CONFIGURATION MANAGEMENT
    # ========================================================================================
    
    async def _load_monitoring_configurations(self):
        """Load active monitoring configurations"""
        try:
            active_configs = self.db.query(MonitoringConfiguration).filter(
                MonitoringConfiguration.is_active == True
            ).all()
            
            logger.info(f"Loaded {len(active_configs)} active monitoring configurations")
            
        except Exception as e:
            logger.error(f"Error loading monitoring configurations: {str(e)}")
    
    # ========================================================================================
    # PUBLIC API METHODS
    # ========================================================================================
    
    async def record_realtime_metric(self, model_id: str, metric_data: Dict[str, Any]):
        """Record a real-time metric"""
        try:
            stream_record = RealTimeMonitoringStream(
                model_id=model_id,
                timestamp=datetime.utcnow(),
                **metric_data
            )
            
            self.db.add(stream_record)
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Error recording real-time metric: {str(e)}")
            self.db.rollback()
    
    async def get_model_health_score(self, model_id: str) -> Optional[ModelHealthScore]:
        """Get latest health score for a model"""
        try:
            return self.db.query(ModelHealthScore).filter(
                ModelHealthScore.model_id == model_id
            ).order_by(desc(ModelHealthScore.timestamp)).first()
            
        except Exception as e:
            logger.error(f"Error getting health score: {str(e)}")
            return None
    
    async def get_active_alerts(self, model_id: str = None) -> List[PerformanceDegradationAlert]:
        """Get active alerts"""
        try:
            query = self.db.query(PerformanceDegradationAlert).filter(
                PerformanceDegradationAlert.status.in_(['open', 'acknowledged'])
            )
            
            if model_id:
                query = query.filter(PerformanceDegradationAlert.model_id == model_id)
            
            return query.order_by(desc(PerformanceDegradationAlert.timestamp)).all()
            
        except Exception as e:
            logger.error(f"Error getting active alerts: {str(e)}")
            return []


class MonitoringError(Exception):
    """Custom exception for monitoring errors"""
    pass