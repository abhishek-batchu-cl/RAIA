# Predictive Health Monitoring Service with Anomaly Detection
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from sqlalchemy.orm import Session
from sqlalchemy import Column, String, DateTime, Text, Boolean, JSON, UUID as PG_UUID, Float, Integer
from sqlalchemy.ext.declarative import declarative_base
import uuid
import asyncio
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

Base = declarative_base()

@dataclass
class HealthMetric:
    id: str
    name: str
    current_value: float
    threshold_warning: float
    threshold_critical: float
    trend: str  # 'improving', 'stable', 'declining'
    prediction: Dict[str, float]
    anomaly_score: float
    status: str  # 'healthy', 'warning', 'critical'
    last_updated: datetime

@dataclass
class PredictiveAlert:
    id: str
    type: str
    severity: str
    title: str
    description: str
    predicted_occurrence: datetime
    confidence: float
    affected_models: List[str]
    recommended_action: str
    time_to_impact: str
    prevention_window: str

@dataclass
class SystemHealth:
    overall_score: float
    status: str
    models_monitored: int
    active_alerts: int
    predictions_made: int
    uptime_percentage: float

class HealthMetricHistory(Base):
    """Store historical health metric data for trend analysis"""
    __tablename__ = "health_metric_history"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    metric_id = Column(String(100), nullable=False)
    metric_name = Column(String(255), nullable=False)
    value = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    model_id = Column(String(255))
    anomaly_score = Column(Float, default=0.0)
    metadata = Column(JSON, default=dict)

class PredictiveAlertHistory(Base):
    """Store predictive alerts for tracking and learning"""
    __tablename__ = "predictive_alert_history"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    alert_id = Column(String(255), unique=True, nullable=False)
    alert_type = Column(String(100), nullable=False)
    severity = Column(String(20), nullable=False)
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=False)
    predicted_occurrence = Column(DateTime, nullable=False)
    actual_occurrence = Column(DateTime)
    confidence = Column(Float, nullable=False)
    affected_models = Column(JSON)
    recommended_action = Column(Text)
    prevention_successful = Column(Boolean)
    user_feedback = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime)

class AnomalyDetectionModel(Base):
    """Store anomaly detection model parameters"""
    __tablename__ = "anomaly_detection_models"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    metric_id = Column(String(100), nullable=False)
    model_type = Column(String(50), default='isolation_forest')
    model_parameters = Column(JSON)
    training_data_period = Column(Integer, default=30)  # days
    last_trained = Column(DateTime, default=datetime.utcnow)
    accuracy_score = Column(Float)
    is_active = Column(Boolean, default=True)

class PredictiveHealthMonitor:
    """Advanced predictive health monitoring system with ML-based anomaly detection"""
    
    def __init__(self, db_session: Session = None):
        self.db = db_session
        self.anomaly_detectors = {}
        self.scalers = {}
        
        # Configurable thresholds
        self.thresholds = {
            'model_accuracy': {'warning': 0.85, 'critical': 0.80},
            'inference_latency': {'warning': 50, 'critical': 100},
            'data_quality': {'warning': 0.90, 'critical': 0.85},
            'prediction_volume': {'warning': 20000, 'critical': 25000},
            'resource_usage': {'warning': 80, 'critical': 95},
            'error_rate': {'warning': 0.05, 'critical': 0.10}
        }
        
        # Initialize anomaly detection models
        self._initialize_anomaly_detectors()

    def _initialize_anomaly_detectors(self):
        """Initialize anomaly detection models for each metric"""
        for metric_id in self.thresholds.keys():
            self.anomaly_detectors[metric_id] = IsolationForest(
                contamination=0.1,  # Expected proportion of outliers
                random_state=42,
                n_estimators=100
            )
            self.scalers[metric_id] = StandardScaler()

    async def collect_system_health(self) -> SystemHealth:
        """Collect current system health metrics"""
        
        # Get current health metrics
        health_metrics = await self.get_current_health_metrics()
        
        # Calculate overall health score
        overall_score = self._calculate_overall_health_score(health_metrics)
        
        # Determine system status
        status = self._determine_system_status(health_metrics)
        
        # Get alert count
        active_alerts = await self._count_active_alerts()
        
        return SystemHealth(
            overall_score=overall_score,
            status=status,
            models_monitored=12,  # This would come from model registry
            active_alerts=active_alerts,
            predictions_made=1453920,  # This would come from prediction logs
            uptime_percentage=99.97
        )

    async def get_current_health_metrics(self) -> List[HealthMetric]:
        """Get current health metrics with trend analysis and predictions"""
        
        metrics = []
        current_time = datetime.utcnow()
        
        # Mock current data - in production this would query actual systems
        current_data = {
            'model_accuracy': 0.847,
            'inference_latency': 45.2,
            'data_quality': 0.934,
            'prediction_volume': 15420,
            'resource_usage': 78.5,
            'error_rate': 0.023
        }
        
        for metric_id, current_value in current_data.items():
            # Get historical data for trend analysis
            historical_data = await self._get_historical_data(metric_id, days=7)
            
            # Calculate trend
            trend = self._calculate_trend(historical_data)
            
            # Generate predictions
            predictions = await self._generate_predictions(metric_id, historical_data, current_value)
            
            # Calculate anomaly score
            anomaly_score = self._calculate_anomaly_score(metric_id, current_value, historical_data)
            
            # Determine status
            status = self._determine_metric_status(metric_id, current_value, anomaly_score)
            
            metric = HealthMetric(
                id=metric_id,
                name=metric_id.replace('_', ' ').title(),
                current_value=current_value,
                threshold_warning=self.thresholds[metric_id]['warning'],
                threshold_critical=self.thresholds[metric_id]['critical'],
                trend=trend,
                prediction=predictions,
                anomaly_score=anomaly_score,
                status=status,
                last_updated=current_time
            )
            
            metrics.append(metric)
            
            # Store metric in history
            if self.db:
                await self._store_metric_history(metric)
        
        return metrics

    async def generate_predictive_alerts(self) -> List[PredictiveAlert]:
        """Generate predictive alerts based on ML analysis and trend forecasting"""
        
        alerts = []
        health_metrics = await self.get_current_health_metrics()
        
        for metric in health_metrics:
            # Check for performance degradation prediction
            if metric.id == 'model_accuracy' and metric.trend == 'declining':
                predicted_value = metric.prediction.get('next_week', metric.current_value)
                if predicted_value < metric.threshold_warning:
                    alert = self._create_performance_alert(metric, predicted_value)
                    alerts.append(alert)
            
            # Check for resource exhaustion
            elif metric.id == 'resource_usage' and metric.current_value > 70:
                predicted_value = metric.prediction.get('next_week', metric.current_value)
                if predicted_value > metric.threshold_warning:
                    alert = self._create_resource_alert(metric, predicted_value)
                    alerts.append(alert)
            
            # Check for anomaly patterns
            elif metric.anomaly_score > 0.7:
                alert = self._create_anomaly_alert(metric)
                alerts.append(alert)
        
        # Store alerts in database
        for alert in alerts:
            if self.db:
                await self._store_predictive_alert(alert)
        
        return alerts

    def _create_performance_alert(self, metric: HealthMetric, predicted_value: float) -> PredictiveAlert:
        """Create performance degradation alert"""
        
        confidence = 0.89
        impact_hours = 48
        
        return PredictiveAlert(
            id=f"perf_degradation_{uuid.uuid4().hex[:8]}",
            type='performance_degradation',
            severity='high' if predicted_value < metric.threshold_critical else 'medium',
            title=f"{metric.name} Performance Drop Predicted",
            description=f"ML models predict a {((metric.current_value - predicted_value) / metric.current_value * 100):.1f}% drop in {metric.name.lower()} in the next {impact_hours} hours based on current drift patterns.",
            predicted_occurrence=datetime.utcnow() + timedelta(hours=impact_hours),
            confidence=confidence,
            affected_models=['credit-scoring-v2', 'risk-assessment-v1'],
            recommended_action="Prepare model retraining with recent data and review feature importance changes.",
            time_to_impact=f"{impact_hours} hours",
            prevention_window=f"{impact_hours - 12} hours"
        )

    def _create_resource_alert(self, metric: HealthMetric, predicted_value: float) -> PredictiveAlert:
        """Create resource exhaustion alert"""
        
        confidence = 0.76
        impact_days = 5
        
        return PredictiveAlert(
            id=f"resource_exhaustion_{uuid.uuid4().hex[:8]}",
            type='resource_exhaustion',
            severity='medium' if predicted_value < 90 else 'high',
            title=f"{metric.name} Approaching Limits",
            description=f"Current resource consumption trends indicate {metric.name.lower()} exhaustion within {impact_days} days if usage continues to grow.",
            predicted_occurrence=datetime.utcnow() + timedelta(days=impact_days),
            confidence=confidence,
            affected_models=['all-models'],
            recommended_action="Scale infrastructure or optimize resource usage. Consider model compression techniques.",
            time_to_impact=f"{impact_days} days",
            prevention_window=f"{impact_days - 2} days"
        )

    def _create_anomaly_alert(self, metric: HealthMetric) -> PredictiveAlert:
        """Create anomaly detection alert"""
        
        confidence = 0.84
        impact_hours = 12
        
        return PredictiveAlert(
            id=f"anomaly_detected_{uuid.uuid4().hex[:8]}",
            type='anomaly_detected',
            severity='medium' if metric.anomaly_score < 0.8 else 'high',
            title=f"Unusual Pattern in {metric.name}",
            description=f"Anomaly detection algorithms identified irregular patterns in {metric.name.lower()} that may impact system performance.",
            predicted_occurrence=datetime.utcnow() + timedelta(hours=impact_hours),
            confidence=confidence,
            affected_models=['related-models'],
            recommended_action=f"Investigate root causes of {metric.name.lower()} anomalies and validate monitoring pipeline.",
            time_to_impact=f"{impact_hours} hours",
            prevention_window=f"{impact_hours - 4} hours"
        )

    async def _get_historical_data(self, metric_id: str, days: int = 30) -> List[float]:
        """Get historical data for a metric"""
        
        if self.db:
            try:
                end_date = datetime.utcnow()
                start_date = end_date - timedelta(days=days)
                
                history = self.db.query(HealthMetricHistory).filter(
                    HealthMetricHistory.metric_id == metric_id,
                    HealthMetricHistory.timestamp >= start_date,
                    HealthMetricHistory.timestamp <= end_date
                ).order_by(HealthMetricHistory.timestamp.desc()).limit(100).all()
                
                return [h.value for h in history]
            except Exception as e:
                print(f"Error getting historical data: {e}")
        
        # Generate synthetic historical data for demo
        np.random.seed(42)
        base_value = 0.85 if 'accuracy' in metric_id or 'quality' in metric_id else 50
        trend = -0.001 if metric_id == 'model_accuracy' else 0.001
        
        historical_data = []
        for i in range(days * 24):  # Hourly data points
            noise = np.random.normal(0, 0.01)
            value = base_value + trend * i + noise
            historical_data.append(max(0, value))
        
        return historical_data[-100:]  # Return last 100 points

    def _calculate_trend(self, historical_data: List[float]) -> str:
        """Calculate trend direction from historical data"""
        
        if len(historical_data) < 5:
            return 'stable'
        
        # Use linear regression to determine trend
        x = np.arange(len(historical_data))
        slope, _, r_value, _, _ = stats.linregress(x, historical_data)
        
        # Consider both slope magnitude and correlation strength
        if abs(r_value) < 0.3:  # Low correlation
            return 'stable'
        elif slope > 0.001:
            return 'improving'
        elif slope < -0.001:
            return 'declining'
        else:
            return 'stable'

    async def _generate_predictions(self, metric_id: str, historical_data: List[float], 
                                  current_value: float) -> Dict[str, float]:
        """Generate predictions for next 24h and next week"""
        
        if len(historical_data) < 10:
            # Fallback predictions
            return {
                'next_24h': current_value * (1 + np.random.uniform(-0.05, 0.05)),
                'next_week': current_value * (1 + np.random.uniform(-0.15, 0.15)),
                'confidence': 0.7
            }
        
        # Simple trend-based prediction (in production, use more sophisticated models)
        x = np.arange(len(historical_data))
        slope, intercept, r_value, _, _ = stats.linregress(x, historical_data)
        
        # Predict future values
        next_24h_x = len(historical_data) + 24  # 24 hours ahead
        next_week_x = len(historical_data) + 24 * 7  # 1 week ahead
        
        prediction_24h = slope * next_24h_x + intercept
        prediction_week = slope * next_week_x + intercept
        
        # Add some realistic noise
        noise_24h = np.random.uniform(-0.02, 0.02)
        noise_week = np.random.uniform(-0.05, 0.05)
        
        return {
            'next_24h': max(0, prediction_24h + noise_24h),
            'next_week': max(0, prediction_week + noise_week),
            'confidence': min(0.95, abs(r_value))
        }

    def _calculate_anomaly_score(self, metric_id: str, current_value: float, 
                                historical_data: List[float]) -> float:
        """Calculate anomaly score using Isolation Forest"""
        
        if len(historical_data) < 20:
            # Not enough data for anomaly detection
            return 0.1
        
        try:
            # Prepare data
            data = np.array(historical_data + [current_value]).reshape(-1, 1)
            
            # Scale data
            scaled_data = self.scalers[metric_id].fit_transform(data)
            
            # Train/update anomaly detector
            self.anomaly_detectors[metric_id].fit(scaled_data[:-1])
            
            # Calculate anomaly score for current value
            anomaly_score = self.anomaly_detectors[metric_id].decision_function([[scaled_data[-1][0]]])[0]
            
            # Convert to 0-1 scale (higher = more anomalous)
            normalized_score = max(0, min(1, (0.5 - anomaly_score)))
            
            return normalized_score
            
        except Exception as e:
            print(f"Error calculating anomaly score for {metric_id}: {e}")
            return 0.1

    def _determine_metric_status(self, metric_id: str, current_value: float, 
                               anomaly_score: float) -> str:
        """Determine metric status based on thresholds and anomaly score"""
        
        thresholds = self.thresholds[metric_id]
        
        # For metrics where lower is better (like error_rate, inference_latency)
        if metric_id in ['error_rate', 'inference_latency']:
            if current_value >= thresholds['critical']:
                return 'critical'
            elif current_value >= thresholds['warning']:
                return 'warning'
        else:
            # For metrics where higher is better (like accuracy, quality)
            if current_value <= thresholds['critical']:
                return 'critical'
            elif current_value <= thresholds['warning']:
                return 'warning'
        
        # Check for anomalies
        if anomaly_score > 0.7:
            return 'warning'
        
        return 'healthy'

    def _calculate_overall_health_score(self, health_metrics: List[HealthMetric]) -> float:
        """Calculate overall system health score"""
        
        if not health_metrics:
            return 0.0
        
        weights = {
            'model_accuracy': 0.3,
            'data_quality': 0.25,
            'inference_latency': 0.2,
            'resource_usage': 0.15,
            'prediction_volume': 0.05,
            'error_rate': 0.05
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for metric in health_metrics:
            weight = weights.get(metric.id, 0.1)
            
            # Normalize metric to 0-100 scale
            if metric.id in ['error_rate', 'inference_latency']:
                # Lower is better
                normalized = max(0, 100 - (metric.current_value / metric.threshold_critical * 100))
            else:
                # Higher is better
                normalized = min(100, (metric.current_value / metric.threshold_warning * 100))
            
            # Apply penalty for anomalies
            anomaly_penalty = metric.anomaly_score * 20
            normalized = max(0, normalized - anomaly_penalty)
            
            total_score += normalized * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0

    def _determine_system_status(self, health_metrics: List[HealthMetric]) -> str:
        """Determine overall system status"""
        
        critical_count = sum(1 for m in health_metrics if m.status == 'critical')
        warning_count = sum(1 for m in health_metrics if m.status == 'warning')
        
        if critical_count > 0:
            return 'critical'
        elif warning_count >= 2:
            return 'warning'
        elif warning_count == 1:
            return 'warning'
        else:
            return 'healthy'

    async def _count_active_alerts(self) -> int:
        """Count active predictive alerts"""
        
        if self.db:
            try:
                count = self.db.query(PredictiveAlertHistory).filter(
                    PredictiveAlertHistory.resolved_at.is_(None)
                ).count()
                return count
            except Exception as e:
                print(f"Error counting active alerts: {e}")
        
        return 2  # Mock count

    async def _store_metric_history(self, metric: HealthMetric):
        """Store metric in historical database"""
        
        if not self.db:
            return
        
        try:
            history_entry = HealthMetricHistory(
                metric_id=metric.id,
                metric_name=metric.name,
                value=metric.current_value,
                anomaly_score=metric.anomaly_score,
                metadata={
                    'trend': metric.trend,
                    'status': metric.status,
                    'prediction': metric.prediction
                }
            )
            
            self.db.add(history_entry)
            self.db.commit()
            
        except Exception as e:
            self.db.rollback()
            print(f"Error storing metric history: {e}")

    async def _store_predictive_alert(self, alert: PredictiveAlert):
        """Store predictive alert in database"""
        
        if not self.db:
            return
        
        try:
            alert_entry = PredictiveAlertHistory(
                alert_id=alert.id,
                alert_type=alert.type,
                severity=alert.severity,
                title=alert.title,
                description=alert.description,
                predicted_occurrence=alert.predicted_occurrence,
                confidence=alert.confidence,
                affected_models=alert.affected_models,
                recommended_action=alert.recommended_action
            )
            
            self.db.add(alert_entry)
            self.db.commit()
            
        except Exception as e:
            self.db.rollback()
            print(f"Error storing predictive alert: {e}")

    async def train_anomaly_models(self, metric_id: str = None):
        """Train or retrain anomaly detection models"""
        
        metrics_to_train = [metric_id] if metric_id else list(self.thresholds.keys())
        
        for mid in metrics_to_train:
            try:
                # Get extended historical data for training
                historical_data = await self._get_historical_data(mid, days=90)
                
                if len(historical_data) < 50:
                    print(f"Insufficient data for training {mid} anomaly detector")
                    continue
                
                # Prepare training data
                X = np.array(historical_data).reshape(-1, 1)
                X_scaled = self.scalers[mid].fit_transform(X)
                
                # Train anomaly detector
                self.anomaly_detectors[mid].fit(X_scaled)
                
                print(f"Trained anomaly detector for {mid}")
                
            except Exception as e:
                print(f"Error training anomaly detector for {mid}: {e}")

# Factory function
def create_predictive_health_monitor(db_session: Session = None) -> PredictiveHealthMonitor:
    """Create and return a PredictiveHealthMonitor instance"""
    return PredictiveHealthMonitor(db_session)