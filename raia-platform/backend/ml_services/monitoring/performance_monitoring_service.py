# Real-time Model Performance Monitoring Service
import os
import json
import uuid
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, asdict
from sqlalchemy.orm import Session
from sqlalchemy import Column, String, DateTime, Text, Boolean, JSON, UUID as PG_UUID, Integer, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import numpy as np
import pandas as pd

# Time series and monitoring
import threading
import time
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor

# Data drift detection
try:
    from alibi_detect import ChiSquareDrift, KSDrift, MMDDrift
    ALIBI_AVAILABLE = True
except ImportError:
    ALIBI_AVAILABLE = False

# Model performance metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats

# System monitoring
import psutil
import GPUtil

Base = declarative_base()
logger = logging.getLogger(__name__)

@dataclass
class PerformanceThreshold:
    """Performance threshold configuration"""
    metric_name: str
    warning_threshold: float
    critical_threshold: float
    direction: str  # 'above' or 'below'
    enabled: bool = True

@dataclass
class AlertConfig:
    """Alert configuration"""
    name: str
    severity: str  # 'info', 'warning', 'critical'
    conditions: List[Dict[str, Any]]
    notification_channels: List[str]
    cooldown_minutes: int = 5
    enabled: bool = True

@dataclass
class MonitoringConfig:
    """Monitoring configuration for a model"""
    model_id: str
    model_name: str
    monitoring_interval_seconds: int = 60
    data_retention_days: int = 30
    thresholds: List[PerformanceThreshold] = None
    alerts: List[AlertConfig] = None
    drift_detection_enabled: bool = True
    reference_data_path: str = None

@dataclass
class ModelMetrics:
    """Real-time model metrics"""
    model_id: str
    timestamp: datetime
    prediction_count: int
    average_response_time: float
    error_rate: float
    cpu_usage: float
    memory_usage: float
    gpu_usage: float = 0.0
    disk_usage: float = 0.0
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc: Optional[float] = None
    mse: Optional[float] = None
    mae: Optional[float] = None
    r2: Optional[float] = None
    drift_score: Optional[float] = None
    outlier_rate: Optional[float] = None
    missing_data_rate: Optional[float] = None

@dataclass
class Alert:
    """Alert data structure"""
    id: str
    model_id: str
    alert_type: str
    severity: str
    message: str
    timestamp: datetime
    acknowledged: bool = False
    resolved: bool = False
    metadata: Dict[str, Any] = None

class ModelMonitoringRecord(Base):
    """Store model monitoring data"""
    __tablename__ = "model_monitoring"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(String(255), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    
    # Performance metrics
    prediction_count = Column(Integer, default=0)
    average_response_time = Column(Float)
    error_rate = Column(Float)
    
    # System metrics
    cpu_usage = Column(Float)
    memory_usage = Column(Float)
    gpu_usage = Column(Float)
    disk_usage = Column(Float)
    
    # Model performance metrics
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    auc = Column(Float)
    mse = Column(Float)
    mae = Column(Float)
    r2 = Column(Float)
    
    # Data quality metrics
    drift_score = Column(Float)
    outlier_rate = Column(Float)
    missing_data_rate = Column(Float)
    
    # Additional metadata
    metadata = Column(JSON)

class ModelAlert(Base):
    """Store model alerts"""
    __tablename__ = "model_alerts"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    alert_id = Column(String(255), unique=True, nullable=False)
    model_id = Column(String(255), nullable=False, index=True)
    alert_type = Column(String(100), nullable=False)
    severity = Column(String(50), nullable=False)
    
    # Alert details
    message = Column(Text, nullable=False)
    acknowledged = Column(Boolean, default=False)
    resolved = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    acknowledged_at = Column(DateTime)
    resolved_at = Column(DateTime)
    
    # Metadata
    metadata = Column(JSON)
    acknowledged_by = Column(String(255))
    resolved_by = Column(String(255))

class PerformanceMonitoringService:
    """Service for real-time model performance monitoring"""
    
    def __init__(self, db_session: Session = None, 
                 monitoring_storage_path: str = "/tmp/raia_monitoring"):
        self.db = db_session
        self.monitoring_storage_path = monitoring_storage_path
        
        # In-memory storage for real-time metrics
        self.metrics_buffer = defaultdict(lambda: deque(maxlen=1000))
        self.alert_cooldowns = defaultdict(dict)
        self.active_monitors = {}
        self.drift_detectors = {}
        
        # Threading
        self.monitor_thread = None
        self.stop_monitoring = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # WebSocket connections for real-time updates
        self.websocket_clients = set()
        
        # Ensure directories exist
        os.makedirs(monitoring_storage_path, exist_ok=True)
        
        # Default thresholds
        self.default_thresholds = [
            PerformanceThreshold('accuracy', 0.85, 0.75, 'below'),
            PerformanceThreshold('error_rate', 0.05, 0.1, 'above'),
            PerformanceThreshold('average_response_time', 1.0, 2.0, 'above'),
            PerformanceThreshold('drift_score', 0.05, 0.1, 'above'),
            PerformanceThreshold('cpu_usage', 80.0, 90.0, 'above'),
            PerformanceThreshold('memory_usage', 85.0, 95.0, 'above')
        ]

    async def start_monitoring_model(self, config: MonitoringConfig) -> Dict[str, Any]:
        """Start monitoring a model"""
        
        model_id = config.model_id
        
        # Initialize drift detector if enabled
        if config.drift_detection_enabled and ALIBI_AVAILABLE:
            await self._initialize_drift_detector(config)
        
        # Store configuration
        self.active_monitors[model_id] = config
        
        # Start monitoring thread if not already running
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            self._start_monitoring_thread()
        
        logger.info(f"Started monitoring for model {model_id}")
        
        return {
            'success': True,
            'model_id': model_id,
            'message': f'Monitoring started for model {config.model_name}'
        }

    async def stop_monitoring_model(self, model_id: str) -> Dict[str, Any]:
        """Stop monitoring a model"""
        
        if model_id in self.active_monitors:
            del self.active_monitors[model_id]
            
            # Clean up drift detector
            if model_id in self.drift_detectors:
                del self.drift_detectors[model_id]
            
            logger.info(f"Stopped monitoring for model {model_id}")
            
            return {
                'success': True,
                'message': f'Monitoring stopped for model {model_id}'
            }
        else:
            return {
                'success': False,
                'error': f'Model {model_id} is not being monitored'
            }

    async def record_prediction(self, model_id: str, 
                              prediction_data: Dict[str, Any],
                              response_time: float,
                              error: bool = False) -> None:
        """Record a model prediction for monitoring"""
        
        timestamp = datetime.utcnow()
        
        # Update real-time metrics
        metrics_key = f"{model_id}_{timestamp.strftime('%Y%m%d_%H%M')}"
        
        if metrics_key not in self.metrics_buffer:
            self.metrics_buffer[metrics_key] = {
                'model_id': model_id,
                'timestamp': timestamp,
                'prediction_count': 0,
                'total_response_time': 0.0,
                'error_count': 0,
                'predictions': []
            }
        
        buffer = self.metrics_buffer[metrics_key]
        buffer['prediction_count'] += 1
        buffer['total_response_time'] += response_time
        
        if error:
            buffer['error_count'] += 1
        
        # Store prediction for data quality analysis
        buffer['predictions'].append({
            'input_data': prediction_data.get('input'),
            'prediction': prediction_data.get('output'),
            'actual': prediction_data.get('actual'),  # If available
            'timestamp': timestamp.isoformat()
        })
        
        # Check for drift if enabled
        if model_id in self.drift_detectors and prediction_data.get('input') is not None:
            await self._check_data_drift(model_id, prediction_data['input'])

    async def get_real_time_metrics(self, model_id: str, 
                                   time_window_minutes: int = 5) -> Dict[str, Any]:
        """Get real-time metrics for a model"""
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=time_window_minutes)
        
        # Aggregate metrics from buffer
        aggregated_metrics = {
            'model_id': model_id,
            'time_window_minutes': time_window_minutes,
            'timestamp': end_time.isoformat(),
            'prediction_count': 0,
            'average_response_time': 0.0,
            'error_rate': 0.0,
            'system_metrics': self._get_system_metrics()
        }
        
        total_predictions = 0
        total_response_time = 0.0
        total_errors = 0
        
        # Search through metrics buffer
        for key, buffer in self.metrics_buffer.items():
            if buffer['model_id'] == model_id and buffer['timestamp'] >= start_time:
                total_predictions += buffer['prediction_count']
                total_response_time += buffer['total_response_time']
                total_errors += buffer['error_count']
        
        if total_predictions > 0:
            aggregated_metrics.update({
                'prediction_count': total_predictions,
                'average_response_time': total_response_time / total_predictions,
                'error_rate': total_errors / total_predictions
            })
        
        # Add model performance metrics if available
        performance_metrics = await self._calculate_model_performance(model_id, start_time, end_time)
        aggregated_metrics.update(performance_metrics)
        
        # Add data quality metrics
        quality_metrics = await self._calculate_data_quality_metrics(model_id, start_time, end_time)
        aggregated_metrics.update(quality_metrics)
        
        return aggregated_metrics

    async def get_historical_metrics(self, model_id: str,
                                   start_time: datetime,
                                   end_time: datetime,
                                   aggregation_interval: str = '1h') -> List[Dict[str, Any]]:
        """Get historical metrics for a model"""
        
        if not self.db:
            return []
        
        # Query database for historical metrics
        records = self.db.query(ModelMonitoringRecord).filter(
            ModelMonitoringRecord.model_id == model_id,
            ModelMonitoringRecord.timestamp >= start_time,
            ModelMonitoringRecord.timestamp <= end_time
        ).order_by(ModelMonitoringRecord.timestamp).all()
        
        # Convert to time series data
        metrics_data = []
        for record in records:
            metrics_data.append({
                'timestamp': record.timestamp.isoformat(),
                'prediction_count': record.prediction_count,
                'average_response_time': record.average_response_time,
                'error_rate': record.error_rate,
                'cpu_usage': record.cpu_usage,
                'memory_usage': record.memory_usage,
                'gpu_usage': record.gpu_usage,
                'disk_usage': record.disk_usage,
                'accuracy': record.accuracy,
                'precision': record.precision,
                'recall': record.recall,
                'f1_score': record.f1_score,
                'auc': record.auc,
                'mse': record.mse,
                'mae': record.mae,
                'r2': record.r2,
                'drift_score': record.drift_score,
                'outlier_rate': record.outlier_rate,
                'missing_data_rate': record.missing_data_rate
            })
        
        return metrics_data

    async def get_active_alerts(self, model_id: str = None) -> List[Dict[str, Any]]:
        """Get active alerts"""
        
        if not self.db:
            return []
        
        query = self.db.query(ModelAlert).filter(
            ModelAlert.resolved == False
        )
        
        if model_id:
            query = query.filter(ModelAlert.model_id == model_id)
        
        alerts = query.order_by(ModelAlert.created_at.desc()).all()
        
        result = []
        for alert in alerts:
            result.append({
                'alert_id': alert.alert_id,
                'model_id': alert.model_id,
                'alert_type': alert.alert_type,
                'severity': alert.severity,
                'message': alert.message,
                'acknowledged': alert.acknowledged,
                'created_at': alert.created_at.isoformat(),
                'metadata': alert.metadata
            })
        
        return result

    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> Dict[str, Any]:
        """Acknowledge an alert"""
        
        if not self.db:
            return {'success': False, 'error': 'Database not available'}
        
        alert = self.db.query(ModelAlert).filter(
            ModelAlert.alert_id == alert_id
        ).first()
        
        if not alert:
            return {'success': False, 'error': 'Alert not found'}
        
        alert.acknowledged = True
        alert.acknowledged_by = acknowledged_by
        alert.acknowledged_at = datetime.utcnow()
        
        self.db.commit()
        
        return {
            'success': True,
            'message': f'Alert {alert_id} acknowledged'
        }

    async def resolve_alert(self, alert_id: str, resolved_by: str) -> Dict[str, Any]:
        """Resolve an alert"""
        
        if not self.db:
            return {'success': False, 'error': 'Database not available'}
        
        alert = self.db.query(ModelAlert).filter(
            ModelAlert.alert_id == alert_id
        ).first()
        
        if not alert:
            return {'success': False, 'error': 'Alert not found'}
        
        alert.resolved = True
        alert.resolved_by = resolved_by
        alert.resolved_at = datetime.utcnow()
        
        self.db.commit()
        
        return {
            'success': True,
            'message': f'Alert {alert_id} resolved'
        }

    async def update_thresholds(self, model_id: str, 
                              thresholds: List[PerformanceThreshold]) -> Dict[str, Any]:
        """Update performance thresholds for a model"""
        
        if model_id not in self.active_monitors:
            return {'success': False, 'error': f'Model {model_id} is not being monitored'}
        
        config = self.active_monitors[model_id]
        config.thresholds = thresholds
        
        logger.info(f"Updated thresholds for model {model_id}")
        
        return {
            'success': True,
            'message': f'Thresholds updated for model {model_id}'
        }

    # Private methods
    def _start_monitoring_thread(self):
        """Start the monitoring thread"""
        self.stop_monitoring.clear()
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("Monitoring thread started")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while not self.stop_monitoring.is_set():
            try:
                # Process metrics for all active monitors
                for model_id, config in self.active_monitors.items():
                    asyncio.run(self._process_model_metrics(model_id, config))
                
                # Sleep for monitoring interval
                time.sleep(min([config.monitoring_interval_seconds 
                              for config in self.active_monitors.values()] or [60]))
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(60)  # Wait before retrying

    async def _process_model_metrics(self, model_id: str, config: MonitoringConfig):
        """Process metrics for a single model"""
        
        # Get current metrics
        metrics = await self.get_real_time_metrics(model_id, config.monitoring_interval_seconds // 60)
        
        # Store in database if available
        if self.db:
            record = ModelMonitoringRecord(
                model_id=model_id,
                timestamp=datetime.utcnow(),
                prediction_count=metrics.get('prediction_count', 0),
                average_response_time=metrics.get('average_response_time'),
                error_rate=metrics.get('error_rate', 0),
                cpu_usage=metrics['system_metrics'].get('cpu_usage'),
                memory_usage=metrics['system_metrics'].get('memory_usage'),
                gpu_usage=metrics['system_metrics'].get('gpu_usage', 0),
                disk_usage=metrics['system_metrics'].get('disk_usage'),
                accuracy=metrics.get('accuracy'),
                precision=metrics.get('precision'),
                recall=metrics.get('recall'),
                f1_score=metrics.get('f1_score'),
                auc=metrics.get('auc'),
                mse=metrics.get('mse'),
                mae=metrics.get('mae'),
                r2=metrics.get('r2'),
                drift_score=metrics.get('drift_score'),
                outlier_rate=metrics.get('outlier_rate'),
                missing_data_rate=metrics.get('missing_data_rate')
            )
            
            self.db.add(record)
            self.db.commit()
        
        # Check thresholds and generate alerts
        await self._check_thresholds(model_id, metrics, config)
        
        # Broadcast to WebSocket clients
        await self._broadcast_metrics(model_id, metrics)

    async def _check_thresholds(self, model_id: str, metrics: Dict[str, Any], 
                               config: MonitoringConfig):
        """Check performance thresholds and generate alerts"""
        
        thresholds = config.thresholds or self.default_thresholds
        
        for threshold in thresholds:
            if not threshold.enabled:
                continue
            
            metric_value = metrics.get(threshold.metric_name)
            if metric_value is None:
                # Check system metrics
                metric_value = metrics.get('system_metrics', {}).get(threshold.metric_name)
            
            if metric_value is None:
                continue
            
            # Check threshold
            alert_triggered = False
            severity = None
            
            if threshold.direction == 'above':
                if metric_value >= threshold.critical_threshold:
                    alert_triggered = True
                    severity = 'critical'
                elif metric_value >= threshold.warning_threshold:
                    alert_triggered = True
                    severity = 'warning'
            else:  # below
                if metric_value <= threshold.critical_threshold:
                    alert_triggered = True
                    severity = 'critical'
                elif metric_value <= threshold.warning_threshold:
                    alert_triggered = True
                    severity = 'warning'
            
            if alert_triggered:
                # Check cooldown
                cooldown_key = f"{model_id}_{threshold.metric_name}_{severity}"
                last_alert_time = self.alert_cooldowns.get(cooldown_key)
                
                if last_alert_time is None or \
                   (datetime.utcnow() - last_alert_time).total_seconds() > 300:  # 5 minutes cooldown
                    
                    await self._create_alert(
                        model_id=model_id,
                        alert_type='threshold_breach',
                        severity=severity,
                        message=f"{threshold.metric_name} is {threshold.direction} threshold: {metric_value:.4f}",
                        metadata={
                            'metric_name': threshold.metric_name,
                            'metric_value': metric_value,
                            'threshold_value': threshold.critical_threshold if severity == 'critical' else threshold.warning_threshold,
                            'direction': threshold.direction
                        }
                    )
                    
                    self.alert_cooldowns[cooldown_key] = datetime.utcnow()

    async def _create_alert(self, model_id: str, alert_type: str, 
                           severity: str, message: str, metadata: Dict[str, Any] = None):
        """Create an alert"""
        
        alert_id = f"alert_{uuid.uuid4().hex[:8]}"
        
        alert = Alert(
            id=alert_id,
            model_id=model_id,
            alert_type=alert_type,
            severity=severity,
            message=message,
            timestamp=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        # Store in database if available
        if self.db:
            db_alert = ModelAlert(
                alert_id=alert_id,
                model_id=model_id,
                alert_type=alert_type,
                severity=severity,
                message=message,
                metadata=metadata
            )
            
            self.db.add(db_alert)
            self.db.commit()
        
        logger.warning(f"Alert created for model {model_id}: {message}")

    async def _calculate_model_performance(self, model_id: str, 
                                         start_time: datetime, 
                                         end_time: datetime) -> Dict[str, Any]:
        """Calculate model performance metrics"""
        
        # Collect predictions with actual values
        predictions_with_actuals = []
        
        for key, buffer in self.metrics_buffer.items():
            if (buffer['model_id'] == model_id and 
                buffer['timestamp'] >= start_time and 
                buffer['timestamp'] <= end_time):
                
                for pred in buffer['predictions']:
                    if pred.get('actual') is not None:
                        predictions_with_actuals.append(pred)
        
        if not predictions_with_actuals:
            return {}
        
        # Extract y_true and y_pred
        y_true = [pred['actual'] for pred in predictions_with_actuals]
        y_pred = [pred['prediction'] for pred in predictions_with_actuals]
        
        metrics = {}
        
        try:
            # Determine if classification or regression based on data type
            if all(isinstance(val, (int, bool)) for val in y_true):
                # Classification metrics
                metrics.update({
                    'accuracy': accuracy_score(y_true, y_pred),
                    'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                    'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
                })
                
                # AUC for binary classification
                if len(set(y_true)) == 2:
                    try:
                        metrics['auc'] = roc_auc_score(y_true, y_pred)
                    except:
                        pass
            else:
                # Regression metrics
                metrics.update({
                    'mse': mean_squared_error(y_true, y_pred),
                    'mae': mean_absolute_error(y_true, y_pred),
                    'r2': r2_score(y_true, y_pred)
                })
        
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
        
        return metrics

    async def _calculate_data_quality_metrics(self, model_id: str,
                                            start_time: datetime,
                                            end_time: datetime) -> Dict[str, Any]:
        """Calculate data quality metrics"""
        
        all_inputs = []
        
        # Collect input data
        for key, buffer in self.metrics_buffer.items():
            if (buffer['model_id'] == model_id and 
                buffer['timestamp'] >= start_time and 
                buffer['timestamp'] <= end_time):
                
                for pred in buffer['predictions']:
                    input_data = pred.get('input_data')
                    if input_data is not None:
                        all_inputs.append(input_data)
        
        if not all_inputs:
            return {}
        
        metrics = {}
        
        try:
            # Convert to DataFrame for analysis
            if isinstance(all_inputs[0], dict):
                df = pd.DataFrame(all_inputs)
            else:
                df = pd.DataFrame({'feature': all_inputs})
            
            # Missing data rate
            missing_rate = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
            metrics['missing_data_rate'] = missing_rate
            
            # Outlier detection using IQR method
            outlier_count = 0
            total_values = 0
            
            for col in df.select_dtypes(include=[np.number]).columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                outlier_count += outliers
                total_values += len(df[col])
            
            if total_values > 0:
                metrics['outlier_rate'] = outlier_count / total_values
            
        except Exception as e:
            logger.error(f"Error calculating data quality metrics: {str(e)}")
        
        return metrics

    def _get_system_metrics(self) -> Dict[str, float]:
        """Get current system metrics"""
        
        metrics = {
            'cpu_usage': psutil.cpu_percent(interval=1),
            'memory_usage': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'gpu_usage': 0.0
        }
        
        # GPU metrics if available
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_usage = sum(gpu.load * 100 for gpu in gpus) / len(gpus)
                metrics['gpu_usage'] = gpu_usage
        except:
            pass
        
        return metrics

    async def _initialize_drift_detector(self, config: MonitoringConfig):
        """Initialize data drift detector"""
        
        if not ALIBI_AVAILABLE:
            logger.warning("Alibi Detect not available, drift detection disabled")
            return
        
        if not config.reference_data_path or not os.path.exists(config.reference_data_path):
            logger.warning(f"Reference data not found for model {config.model_id}")
            return
        
        try:
            # Load reference data
            reference_data = pd.read_csv(config.reference_data_path)
            X_ref = reference_data.values
            
            # Initialize drift detector (using Chi-Square for categorical, KS for numerical)
            if X_ref.dtype in ['object', 'category']:
                detector = ChiSquareDrift(X_ref, p_val=0.05)
            else:
                detector = KSDrift(X_ref, p_val=0.05)
            
            self.drift_detectors[config.model_id] = detector
            logger.info(f"Drift detector initialized for model {config.model_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize drift detector: {str(e)}")

    async def _check_data_drift(self, model_id: str, input_data: Any):
        """Check for data drift"""
        
        if model_id not in self.drift_detectors:
            return
        
        try:
            detector = self.drift_detectors[model_id]
            
            # Convert input to appropriate format
            if isinstance(input_data, dict):
                X_test = list(input_data.values())
            else:
                X_test = input_data
            
            X_test = np.array(X_test).reshape(1, -1)
            
            # Run drift detection
            drift_result = detector.predict(X_test)
            
            if drift_result['data']['is_drift']:
                drift_score = drift_result['data']['p_val']
                
                await self._create_alert(
                    model_id=model_id,
                    alert_type='data_drift',
                    severity='warning' if drift_score > 0.01 else 'critical',
                    message=f"Data drift detected with p-value: {drift_score:.6f}",
                    metadata={
                        'drift_score': drift_score,
                        'drift_threshold': 0.05
                    }
                )
        
        except Exception as e:
            logger.error(f"Error checking data drift: {str(e)}")

    async def _broadcast_metrics(self, model_id: str, metrics: Dict[str, Any]):
        """Broadcast metrics to WebSocket clients"""
        
        # This would integrate with WebSocket server
        # For now, we'll just log the broadcast
        logger.debug(f"Broadcasting metrics for model {model_id}")

    def stop(self):
        """Stop the monitoring service"""
        self.stop_monitoring.set()
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        self.executor.shutdown(wait=True)
        logger.info("Performance monitoring service stopped")

# Factory function
def create_performance_monitoring_service(db_session: Session = None,
                                        monitoring_storage_path: str = "/tmp/raia_monitoring") -> PerformanceMonitoringService:
    """Create and return a PerformanceMonitoringService instance"""
    return PerformanceMonitoringService(db_session, monitoring_storage_path)