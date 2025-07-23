# Monitoring & Alerts API
import os
import uuid
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable
from enum import Enum

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, validator
from sqlalchemy.orm import Session
from sqlalchemy import Column, String, DateTime, Text, Boolean, JSON, UUID as PG_UUID, Integer, Float, ForeignKey, func, desc, and_
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

# Monitoring and metrics libraries
import prometheus_client
from prometheus_client import Counter, Gauge, Histogram, Summary, CollectorRegistry, generate_latest
import psutil
import asyncio
import aioredis
from contextlib import asynccontextmanager

# Alerting
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import json

# Time series data
from collections import deque, defaultdict
import numpy as np
import pandas as pd

Base = declarative_base()
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/monitoring", tags=["monitoring"])

# Enums
class MetricType(str, Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

class AlertSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class AlertStatus(str, Enum):
    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"
    SUPPRESSED = "suppressed"

class MonitorType(str, Enum):
    HEALTH_CHECK = "health_check"
    PERFORMANCE = "performance"
    ERROR_RATE = "error_rate"
    RESOURCE_USAGE = "resource_usage"
    CUSTOM_METRIC = "custom_metric"
    SLA_COMPLIANCE = "sla_compliance"

class NotificationChannel(str, Enum):
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    PAGERDUTY = "pagerduty"

# Database Models
class Monitor(Base):
    """Monitoring configuration"""
    __tablename__ = "monitors"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    
    # Monitor configuration
    monitor_type = Column(String(100), nullable=False)
    target_resource = Column(String(255))  # model, experiment, dataset, etc.
    target_resource_id = Column(String(255))
    
    # Monitoring parameters
    config = Column(JSON)  # Monitor-specific configuration
    check_interval_seconds = Column(Integer, default=60)
    timeout_seconds = Column(Integer, default=30)
    
    # Thresholds and conditions
    conditions = Column(JSON)  # Alert conditions
    severity = Column(String(50), default=AlertSeverity.MEDIUM)
    
    # Status and lifecycle
    is_enabled = Column(Boolean, default=True)
    last_check_time = Column(DateTime)
    last_check_status = Column(String(50))
    last_check_value = Column(Float)
    
    # Notification settings
    notification_channels = Column(JSON)  # List of notification channels
    escalation_rules = Column(JSON)  # Escalation configuration
    
    # Metadata
    tags = Column(JSON)
    
    # Ownership
    created_by = Column(String(255), nullable=False)
    organization_id = Column(String(255))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    alerts = relationship("Alert", back_populates="monitor", cascade="all, delete-orphan")
    metrics = relationship("Metric", back_populates="monitor", cascade="all, delete-orphan")

class Alert(Base):
    """Alert instances"""
    __tablename__ = "alerts"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    monitor_id = Column(PG_UUID(as_uuid=True), ForeignKey('monitors.id'), nullable=False)
    
    # Alert identification
    alert_name = Column(String(255), nullable=False)
    alert_message = Column(Text, nullable=False)
    fingerprint = Column(String(255), index=True)  # Unique identifier for deduplication
    
    # Alert details
    severity = Column(String(50), nullable=False)
    status = Column(String(50), default=AlertStatus.ACTIVE)
    
    # Alert data
    trigger_value = Column(Float)
    threshold_value = Column(Float)
    additional_data = Column(JSON)
    
    # Lifecycle
    triggered_at = Column(DateTime, default=datetime.utcnow)
    acknowledged_at = Column(DateTime)
    acknowledged_by = Column(String(255))
    resolved_at = Column(DateTime)
    resolved_by = Column(String(255))
    
    # Notification tracking
    notifications_sent = Column(JSON)  # Track which notifications were sent
    last_notification = Column(DateTime)
    notification_count = Column(Integer, default=0)
    
    # Escalation
    escalation_level = Column(Integer, default=0)
    escalated_at = Column(DateTime)
    
    # Context
    labels = Column(JSON)  # Key-value labels
    annotations = Column(JSON)  # Additional annotations
    
    # Relationships
    monitor = relationship("Monitor", back_populates="alerts")

class Metric(Base):
    """Metrics data points"""
    __tablename__ = "metrics"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    monitor_id = Column(PG_UUID(as_uuid=True), ForeignKey('monitors.id'))
    
    # Metric identification
    metric_name = Column(String(255), nullable=False, index=True)
    metric_type = Column(String(50), nullable=False)
    
    # Metric data
    value = Column(Float, nullable=False)
    labels = Column(JSON)  # Key-value labels
    
    # Timestamp
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    monitor = relationship("Monitor", back_populates="metrics")

class NotificationChannel(Base):
    """Notification channel configurations"""
    __tablename__ = "notification_channels"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    channel_type = Column(String(100), nullable=False)
    
    # Channel configuration
    config = Column(JSON, nullable=False)  # Channel-specific settings
    
    # Status
    is_enabled = Column(Boolean, default=True)
    last_used = Column(DateTime)
    success_count = Column(Integer, default=0)
    failure_count = Column(Integer, default=0)
    
    # Ownership
    created_by = Column(String(255), nullable=False)
    organization_id = Column(String(255))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Pydantic Models
class MonitorCreate(BaseModel):
    name: str
    description: Optional[str] = None
    monitor_type: MonitorType
    target_resource: Optional[str] = None
    target_resource_id: Optional[str] = None
    config: Dict[str, Any] = {}
    check_interval_seconds: int = 60
    timeout_seconds: int = 30
    conditions: Dict[str, Any] = {}
    severity: AlertSeverity = AlertSeverity.MEDIUM
    notification_channels: List[str] = []
    tags: List[str] = []

class MonitorUpdate(BaseModel):
    description: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    check_interval_seconds: Optional[int] = None
    conditions: Optional[Dict[str, Any]] = None
    severity: Optional[AlertSeverity] = None
    is_enabled: Optional[bool] = None
    notification_channels: Optional[List[str]] = None
    tags: Optional[List[str]] = None

class MonitorResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    monitor_type: str
    target_resource: Optional[str]
    check_interval_seconds: int
    severity: str
    is_enabled: bool
    last_check_time: Optional[datetime]
    last_check_status: Optional[str]
    created_by: str
    created_at: datetime
    
    class Config:
        orm_mode = True

class AlertResponse(BaseModel):
    id: str
    monitor_id: str
    alert_name: str
    alert_message: str
    severity: str
    status: str
    trigger_value: Optional[float]
    threshold_value: Optional[float]
    triggered_at: datetime
    acknowledged_at: Optional[datetime]
    resolved_at: Optional[datetime]
    
    class Config:
        orm_mode = True

class MetricCreate(BaseModel):
    metric_name: str
    metric_type: MetricType
    value: float
    labels: Optional[Dict[str, str]] = {}
    timestamp: Optional[datetime] = None

class NotificationChannelCreate(BaseModel):
    name: str
    channel_type: str  # email, slack, webhook, etc.
    config: Dict[str, Any]

# Dependency injection
def get_db():
    pass

def get_current_user():
    return "current_user_id"

# Global monitoring service instance
monitoring_service = None

# Monitoring & Alerting Service
class MonitoringService:
    """Service for monitoring and alerting"""
    
    def __init__(self, db: Session):
        self.db = db
        
        # Prometheus metrics registry
        self.registry = CollectorRegistry()
        
        # Metrics collectors
        self.system_metrics = {
            'cpu_usage': Gauge('system_cpu_usage_percent', 'System CPU usage percentage', registry=self.registry),
            'memory_usage': Gauge('system_memory_usage_percent', 'System memory usage percentage', registry=self.registry),
            'disk_usage': Gauge('system_disk_usage_percent', 'System disk usage percentage', registry=self.registry),
            'api_requests': Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'], registry=self.registry),
            'api_duration': Histogram('api_request_duration_seconds', 'API request duration', ['method', 'endpoint'], registry=self.registry),
            'model_predictions': Counter('model_predictions_total', 'Total model predictions', ['model_id'], registry=self.registry),
            'model_latency': Histogram('model_prediction_duration_seconds', 'Model prediction latency', ['model_id'], registry=self.registry),
            'experiment_runs': Counter('experiment_runs_total', 'Total experiment runs', ['experiment_id', 'status'], registry=self.registry),
            'dataset_downloads': Counter('dataset_downloads_total', 'Total dataset downloads', ['dataset_id'], registry=self.registry)
        }
        
        # Active monitors
        self.active_monitors = {}
        
        # WebSocket connections for real-time updates
        self.websocket_connections = set()
        
        # In-memory time series data (for demo purposes)
        self.time_series_data = defaultdict(lambda: deque(maxlen=1000))
        
        # Background tasks
        self.monitoring_task = None
        
        # Monitor type handlers
        self.monitor_handlers = {
            MonitorType.HEALTH_CHECK: self._check_health,
            MonitorType.PERFORMANCE: self._check_performance,
            MonitorType.ERROR_RATE: self._check_error_rate,
            MonitorType.RESOURCE_USAGE: self._check_resource_usage,
            MonitorType.CUSTOM_METRIC: self._check_custom_metric,
            MonitorType.SLA_COMPLIANCE: self._check_sla_compliance
        }
        
        # Notification handlers
        self.notification_handlers = {
            'email': self._send_email_notification,
            'slack': self._send_slack_notification,
            'webhook': self._send_webhook_notification,
            'sms': self._send_sms_notification,
            'pagerduty': self._send_pagerduty_notification
        }
    
    async def start_monitoring(self):
        """Start the monitoring background task"""
        
        if not self.monitoring_task:
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Monitoring service started")
    
    async def stop_monitoring(self):
        """Stop the monitoring background task"""
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            logger.info("Monitoring service stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        
        try:
            while True:
                # Collect system metrics
                await self._collect_system_metrics()
                
                # Check all active monitors
                monitors = self.db.query(Monitor).filter(Monitor.is_enabled == True).all()
                
                for monitor in monitors:
                    try:
                        await self._check_monitor(monitor)
                    except Exception as e:
                        logger.error(f"Error checking monitor {monitor.name}: {str(e)}")
                
                # Clean up old metrics data
                await self._cleanup_old_metrics()
                
                # Wait before next iteration
                await asyncio.sleep(10)  # Check every 10 seconds
                
        except asyncio.CancelledError:
            logger.info("Monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Monitoring loop error: {str(e)}")
    
    async def _collect_system_metrics(self):
        """Collect system-level metrics"""
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.system_metrics['cpu_usage'].set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.system_metrics['memory_usage'].set(memory.percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.system_metrics['disk_usage'].set(disk_percent)
            
            # Store in time series
            now = datetime.utcnow()
            self.time_series_data['system.cpu_usage'].append((now, cpu_percent))
            self.time_series_data['system.memory_usage'].append((now, memory.percent))
            self.time_series_data['system.disk_usage'].append((now, disk_percent))
            
            # Store metrics in database
            metrics_to_store = [
                Metric(metric_name='system.cpu_usage', metric_type=MetricType.GAUGE, 
                      value=cpu_percent, timestamp=now),
                Metric(metric_name='system.memory_usage', metric_type=MetricType.GAUGE, 
                      value=memory.percent, timestamp=now),
                Metric(metric_name='system.disk_usage', metric_type=MetricType.GAUGE, 
                      value=disk_percent, timestamp=now)
            ]
            
            for metric in metrics_to_store:
                self.db.add(metric)
            
            self.db.commit()
            
            # Broadcast to WebSocket connections
            await self._broadcast_metrics({
                'cpu_usage': cpu_percent,
                'memory_usage': memory.percent,
                'disk_usage': disk_percent,
                'timestamp': now.isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {str(e)}")
    
    async def _check_monitor(self, monitor: Monitor):
        """Check a specific monitor"""
        
        now = datetime.utcnow()
        
        # Check if it's time to run this monitor
        if (monitor.last_check_time and 
            (now - monitor.last_check_time).total_seconds() < monitor.check_interval_seconds):
            return
        
        try:
            # Get the appropriate handler
            handler = self.monitor_handlers.get(MonitorType(monitor.monitor_type))
            if not handler:
                logger.warning(f"No handler for monitor type: {monitor.monitor_type}")
                return
            
            # Execute the check
            result = await handler(monitor)
            
            # Update monitor status
            monitor.last_check_time = now
            monitor.last_check_status = result.get('status', 'unknown')
            monitor.last_check_value = result.get('value')
            
            # Evaluate conditions and create alerts if needed
            await self._evaluate_conditions(monitor, result)
            
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Error checking monitor {monitor.name}: {str(e)}")
            monitor.last_check_time = now
            monitor.last_check_status = 'error'
            self.db.commit()
    
    async def _evaluate_conditions(self, monitor: Monitor, result: Dict[str, Any]):
        """Evaluate monitor conditions and create alerts"""
        
        conditions = monitor.conditions
        value = result.get('value')
        
        if not conditions or value is None:
            return
        
        alert_triggered = False
        alert_message = ""
        
        # Check threshold conditions
        if 'threshold' in conditions:
            threshold_config = conditions['threshold']
            threshold_value = threshold_config.get('value')
            operator = threshold_config.get('operator', '>')
            
            if operator == '>' and value > threshold_value:
                alert_triggered = True
                alert_message = f"{monitor.name} value {value} exceeds threshold {threshold_value}"
            elif operator == '<' and value < threshold_value:
                alert_triggered = True
                alert_message = f"{monitor.name} value {value} below threshold {threshold_value}"
            elif operator == '>=' and value >= threshold_value:
                alert_triggered = True
                alert_message = f"{monitor.name} value {value} exceeds or equals threshold {threshold_value}"
            elif operator == '<=' and value <= threshold_value:
                alert_triggered = True
                alert_message = f"{monitor.name} value {value} below or equals threshold {threshold_value}"
            elif operator == '==' and value == threshold_value:
                alert_triggered = True
                alert_message = f"{monitor.name} value equals {threshold_value}"
            elif operator == '!=' and value != threshold_value:
                alert_triggered = True
                alert_message = f"{monitor.name} value {value} does not equal expected {threshold_value}"
        
        # Check rate conditions (change over time)
        if 'rate' in conditions and not alert_triggered:
            rate_config = conditions['rate']
            time_window = rate_config.get('time_window_minutes', 5)
            threshold_change = rate_config.get('threshold_change')
            
            # Get historical data
            cutoff_time = datetime.utcnow() - timedelta(minutes=time_window)
            historical_metrics = self.db.query(Metric).filter(
                Metric.monitor_id == monitor.id,
                Metric.timestamp >= cutoff_time
            ).order_by(Metric.timestamp.desc()).limit(10).all()
            
            if len(historical_metrics) >= 2:
                old_value = historical_metrics[-1].value
                rate_of_change = ((value - old_value) / old_value) * 100 if old_value != 0 else 0
                
                if abs(rate_of_change) > threshold_change:
                    alert_triggered = True
                    alert_message = f"{monitor.name} rate of change {rate_of_change:.2f}% exceeds threshold {threshold_change}%"
        
        if alert_triggered:
            await self._create_alert(monitor, alert_message, value, conditions.get('threshold', {}).get('value'))
    
    async def _create_alert(self, monitor: Monitor, message: str, trigger_value: float, threshold_value: Optional[float]):
        """Create a new alert"""
        
        # Create fingerprint for deduplication
        fingerprint = f"{monitor.id}:{message}"
        
        # Check if we already have an active alert with the same fingerprint
        existing_alert = self.db.query(Alert).filter(
            Alert.monitor_id == monitor.id,
            Alert.fingerprint == fingerprint,
            Alert.status == AlertStatus.ACTIVE
        ).first()
        
        if existing_alert:
            # Update existing alert
            existing_alert.trigger_value = trigger_value
            existing_alert.additional_data = {'last_seen': datetime.utcnow().isoformat()}
        else:
            # Create new alert
            alert = Alert(
                monitor_id=monitor.id,
                alert_name=monitor.name,
                alert_message=message,
                fingerprint=fingerprint,
                severity=monitor.severity,
                trigger_value=trigger_value,
                threshold_value=threshold_value,
                labels={'monitor_type': monitor.monitor_type, 'target_resource': monitor.target_resource},
                annotations={'description': message, 'monitor_id': str(monitor.id)}
            )
            
            self.db.add(alert)
            self.db.commit()
            self.db.refresh(alert)
            
            # Send notifications
            await self._send_alert_notifications(alert, monitor)
            
            logger.info(f"Created alert: {message}")
        
        self.db.commit()
    
    async def _send_alert_notifications(self, alert: Alert, monitor: Monitor):
        """Send notifications for an alert"""
        
        if not monitor.notification_channels:
            return
        
        notifications_sent = []
        
        for channel_id in monitor.notification_channels:
            try:
                channel = self.db.query(NotificationChannel).filter(
                    NotificationChannel.id == channel_id,
                    NotificationChannel.is_enabled == True
                ).first()
                
                if channel:
                    handler = self.notification_handlers.get(channel.channel_type)
                    if handler:
                        await handler(alert, channel)
                        notifications_sent.append({
                            'channel_id': channel_id,
                            'channel_type': channel.channel_type,
                            'sent_at': datetime.utcnow().isoformat()
                        })
                        
                        channel.success_count += 1
                        channel.last_used = datetime.utcnow()
                    else:
                        logger.warning(f"No handler for notification channel type: {channel.channel_type}")
                
            except Exception as e:
                logger.error(f"Error sending notification to channel {channel_id}: {str(e)}")
                if channel:
                    channel.failure_count += 1
        
        # Update alert with notification info
        alert.notifications_sent = notifications_sent
        alert.last_notification = datetime.utcnow()
        alert.notification_count += 1
        
        self.db.commit()
    
    # Monitor type handlers
    async def _check_health(self, monitor: Monitor) -> Dict[str, Any]:
        """Health check monitor"""
        
        config = monitor.config
        url = config.get('url')
        expected_status = config.get('expected_status', 200)
        
        if not url:
            return {'status': 'error', 'message': 'No URL configured'}
        
        try:
            import aiohttp
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=monitor.timeout_seconds)) as session:
                async with session.get(url) as response:
                    status_ok = response.status == expected_status
                    return {
                        'status': 'healthy' if status_ok else 'unhealthy',
                        'value': 1 if status_ok else 0,
                        'response_code': response.status,
                        'response_time': 0  # Would measure actual response time
                    }
        except Exception as e:
            return {'status': 'unhealthy', 'value': 0, 'error': str(e)}
    
    async def _check_performance(self, monitor: Monitor) -> Dict[str, Any]:
        """Performance monitor"""
        
        config = monitor.config
        metric_name = config.get('metric_name')
        
        # Get recent performance data
        # This would integrate with actual performance metrics
        
        return {'status': 'ok', 'value': 0.95}  # Mock value
    
    async def _check_error_rate(self, monitor: Monitor) -> Dict[str, Any]:
        """Error rate monitor"""
        
        config = monitor.config
        time_window = config.get('time_window_minutes', 5)
        
        # Calculate error rate from logs/metrics
        # Mock implementation
        
        return {'status': 'ok', 'value': 0.02}  # 2% error rate
    
    async def _check_resource_usage(self, monitor: Monitor) -> Dict[str, Any]:
        """Resource usage monitor"""
        
        config = monitor.config
        resource_type = config.get('resource_type', 'cpu')
        
        if resource_type == 'cpu':
            value = psutil.cpu_percent()
        elif resource_type == 'memory':
            value = psutil.virtual_memory().percent
        elif resource_type == 'disk':
            value = (psutil.disk_usage('/').used / psutil.disk_usage('/').total) * 100
        else:
            value = 0
        
        return {'status': 'ok', 'value': value}
    
    async def _check_custom_metric(self, monitor: Monitor) -> Dict[str, Any]:
        """Custom metric monitor"""
        
        config = monitor.config
        metric_name = config.get('metric_name')
        
        # Get custom metric value
        # This would query the metrics database or external system
        
        return {'status': 'ok', 'value': 1.0}
    
    async def _check_sla_compliance(self, monitor: Monitor) -> Dict[str, Any]:
        """SLA compliance monitor"""
        
        config = monitor.config
        sla_target = config.get('sla_target', 99.9)
        
        # Calculate SLA compliance
        # Mock implementation
        
        return {'status': 'ok', 'value': 99.95}
    
    # Notification handlers
    async def _send_email_notification(self, alert: Alert, channel: NotificationChannel):
        """Send email notification"""
        
        config = channel.config
        smtp_server = config.get('smtp_server')
        smtp_port = config.get('smtp_port', 587)
        username = config.get('username')
        password = config.get('password')
        recipients = config.get('recipients', [])
        
        if not all([smtp_server, username, password, recipients]):
            logger.warning("Incomplete email configuration")
            return
        
        # Mock email sending
        logger.info(f"Sending email alert: {alert.alert_message} to {recipients}")
    
    async def _send_slack_notification(self, alert: Alert, channel: NotificationChannel):
        """Send Slack notification"""
        
        config = channel.config
        webhook_url = config.get('webhook_url')
        
        if not webhook_url:
            logger.warning("No Slack webhook URL configured")
            return
        
        # Mock Slack notification
        logger.info(f"Sending Slack alert: {alert.alert_message}")
    
    async def _send_webhook_notification(self, alert: Alert, channel: NotificationChannel):
        """Send webhook notification"""
        
        config = channel.config
        url = config.get('url')
        
        if not url:
            logger.warning("No webhook URL configured")
            return
        
        # Mock webhook notification
        logger.info(f"Sending webhook alert: {alert.alert_message}")
    
    async def _send_sms_notification(self, alert: Alert, channel: NotificationChannel):
        """Send SMS notification"""
        
        # Mock SMS notification
        logger.info(f"Sending SMS alert: {alert.alert_message}")
    
    async def _send_pagerduty_notification(self, alert: Alert, channel: NotificationChannel):
        """Send PagerDuty notification"""
        
        # Mock PagerDuty notification
        logger.info(f"Sending PagerDuty alert: {alert.alert_message}")
    
    async def _cleanup_old_metrics(self):
        """Clean up old metrics data"""
        
        # Keep metrics for 30 days
        cutoff_date = datetime.utcnow() - timedelta(days=30)
        
        deleted_count = self.db.query(Metric).filter(
            Metric.timestamp < cutoff_date
        ).delete()
        
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old metrics")
            self.db.commit()
    
    async def _broadcast_metrics(self, data: Dict[str, Any]):
        """Broadcast metrics to WebSocket connections"""
        
        if not self.websocket_connections:
            return
        
        message = json.dumps({
            'type': 'metrics_update',
            'data': data
        })
        
        # Send to all connected clients
        disconnected = set()
        for websocket in self.websocket_connections:
            try:
                await websocket.send_text(message)
            except:
                disconnected.add(websocket)
        
        # Remove disconnected clients
        self.websocket_connections -= disconnected
    
    # Public API methods
    async def create_monitor(self, monitor_data: MonitorCreate, user_id: str) -> Monitor:
        """Create a new monitor"""
        
        monitor = Monitor(
            name=monitor_data.name,
            description=monitor_data.description,
            monitor_type=monitor_data.monitor_type,
            target_resource=monitor_data.target_resource,
            target_resource_id=monitor_data.target_resource_id,
            config=monitor_data.config,
            check_interval_seconds=monitor_data.check_interval_seconds,
            timeout_seconds=monitor_data.timeout_seconds,
            conditions=monitor_data.conditions,
            severity=monitor_data.severity,
            notification_channels=monitor_data.notification_channels,
            tags=monitor_data.tags,
            created_by=user_id,
            organization_id=self._get_user_org(user_id)
        )
        
        self.db.add(monitor)
        self.db.commit()
        self.db.refresh(monitor)
        
        logger.info(f"Created monitor {monitor.name} (ID: {monitor.id})")
        return monitor
    
    async def create_notification_channel(self, channel_data: NotificationChannelCreate, user_id: str) -> NotificationChannel:
        """Create a new notification channel"""
        
        channel = NotificationChannel(
            name=channel_data.name,
            channel_type=channel_data.channel_type,
            config=channel_data.config,
            created_by=user_id,
            organization_id=self._get_user_org(user_id)
        )
        
        self.db.add(channel)
        self.db.commit()
        self.db.refresh(channel)
        
        logger.info(f"Created notification channel {channel.name} (ID: {channel.id})")
        return channel
    
    async def get_metrics_data(self, metric_names: List[str], start_time: datetime, end_time: datetime) -> Dict[str, List[Dict]]:
        """Get metrics data for specified time range"""
        
        metrics_data = {}
        
        for metric_name in metric_names:
            metrics = self.db.query(Metric).filter(
                Metric.metric_name == metric_name,
                Metric.timestamp >= start_time,
                Metric.timestamp <= end_time
            ).order_by(Metric.timestamp).all()
            
            metrics_data[metric_name] = [
                {
                    'timestamp': metric.timestamp.isoformat(),
                    'value': metric.value,
                    'labels': metric.labels
                }
                for metric in metrics
            ]
        
        return metrics_data
    
    async def acknowledge_alert(self, alert_id: str, user_id: str) -> Dict[str, Any]:
        """Acknowledge an alert"""
        
        alert = self.db.query(Alert).filter(Alert.id == alert_id).first()
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_at = datetime.utcnow()
        alert.acknowledged_by = user_id
        
        self.db.commit()
        
        return {"message": "Alert acknowledged"}
    
    async def resolve_alert(self, alert_id: str, user_id: str) -> Dict[str, Any]:
        """Resolve an alert"""
        
        alert = self.db.query(Alert).filter(Alert.id == alert_id).first()
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.utcnow()
        alert.resolved_by = user_id
        
        self.db.commit()
        
        return {"message": "Alert resolved"}
    
    def _get_user_org(self, user_id: str) -> Optional[str]:
        """Get user's organization ID"""
        return "default_org"

# Initialize global service
async def get_monitoring_service(db: Session = Depends(get_db)) -> MonitoringService:
    """Get monitoring service instance"""
    global monitoring_service
    if not monitoring_service:
        monitoring_service = MonitoringService(db)
        await monitoring_service.start_monitoring()
    return monitoring_service

# API Endpoints
@router.post("/monitors", response_model=MonitorResponse)
async def create_monitor(
    monitor_data: MonitorCreate,
    service: MonitoringService = Depends(get_monitoring_service),
    current_user: str = Depends(get_current_user)
):
    """Create a new monitor"""
    
    monitor = await service.create_monitor(monitor_data, current_user)
    return monitor

@router.get("/monitors", response_model=List[MonitorResponse])
async def get_monitors(
    skip: int = 0,
    limit: int = 100,
    monitor_type: Optional[str] = None,
    is_enabled: Optional[bool] = None,
    service: MonitoringService = Depends(get_monitoring_service),
    current_user: str = Depends(get_current_user)
):
    """Get monitors"""
    
    query = service.db.query(Monitor).filter(
        (Monitor.created_by == current_user) |
        (Monitor.organization_id == service._get_user_org(current_user))
    )
    
    if monitor_type:
        query = query.filter(Monitor.monitor_type == monitor_type)
    if is_enabled is not None:
        query = query.filter(Monitor.is_enabled == is_enabled)
    
    monitors = query.order_by(desc(Monitor.created_at)).offset(skip).limit(limit).all()
    return monitors

@router.put("/monitors/{monitor_id}", response_model=MonitorResponse)
async def update_monitor(
    monitor_id: str,
    monitor_data: MonitorUpdate,
    service: MonitoringService = Depends(get_monitoring_service),
    current_user: str = Depends(get_current_user)
):
    """Update a monitor"""
    
    monitor = service.db.query(Monitor).filter(Monitor.id == monitor_id).first()
    if not monitor:
        raise HTTPException(status_code=404, detail="Monitor not found")
    
    if monitor.created_by != current_user:
        raise HTTPException(status_code=403, detail="Permission denied")
    
    # Update fields
    for field, value in monitor_data.dict(exclude_unset=True).items():
        setattr(monitor, field, value)
    
    monitor.updated_at = datetime.utcnow()
    service.db.commit()
    
    return monitor

@router.get("/alerts", response_model=List[AlertResponse])
async def get_alerts(
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None,
    severity: Optional[str] = None,
    service: MonitoringService = Depends(get_monitoring_service),
    current_user: str = Depends(get_current_user)
):
    """Get alerts"""
    
    query = service.db.query(Alert).join(Monitor).filter(
        (Monitor.created_by == current_user) |
        (Monitor.organization_id == service._get_user_org(current_user))
    )
    
    if status:
        query = query.filter(Alert.status == status)
    if severity:
        query = query.filter(Alert.severity == severity)
    
    alerts = query.order_by(desc(Alert.triggered_at)).offset(skip).limit(limit).all()
    return alerts

@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    service: MonitoringService = Depends(get_monitoring_service),
    current_user: str = Depends(get_current_user)
):
    """Acknowledge an alert"""
    
    result = await service.acknowledge_alert(alert_id, current_user)
    return result

@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(
    alert_id: str,
    service: MonitoringService = Depends(get_monitoring_service),
    current_user: str = Depends(get_current_user)
):
    """Resolve an alert"""
    
    result = await service.resolve_alert(alert_id, current_user)
    return result

@router.post("/channels")
async def create_notification_channel(
    channel_data: NotificationChannelCreate,
    service: MonitoringService = Depends(get_monitoring_service),
    current_user: str = Depends(get_current_user)
):
    """Create a notification channel"""
    
    channel = await service.create_notification_channel(channel_data, current_user)
    return channel

@router.get("/metrics")
async def get_metrics(
    metric_names: List[str],
    start_time: datetime,
    end_time: datetime,
    service: MonitoringService = Depends(get_monitoring_service),
    current_user: str = Depends(get_current_user)
):
    """Get metrics data"""
    
    data = await service.get_metrics_data(metric_names, start_time, end_time)
    return data

@router.post("/metrics")
async def record_metric(
    metric_data: MetricCreate,
    service: MonitoringService = Depends(get_monitoring_service),
    current_user: str = Depends(get_current_user)
):
    """Record a custom metric"""
    
    metric = Metric(
        metric_name=metric_data.metric_name,
        metric_type=metric_data.metric_type,
        value=metric_data.value,
        labels=metric_data.labels,
        timestamp=metric_data.timestamp or datetime.utcnow()
    )
    
    service.db.add(metric)
    service.db.commit()
    
    return {"message": "Metric recorded"}

@router.get("/prometheus")
async def get_prometheus_metrics(
    service: MonitoringService = Depends(get_monitoring_service)
):
    """Get metrics in Prometheus format"""
    
    return generate_latest(service.registry).decode('utf-8')

@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    service: MonitoringService = Depends(get_monitoring_service)
):
    """WebSocket endpoint for real-time monitoring data"""
    
    await websocket.accept()
    service.websocket_connections.add(websocket)
    
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        service.websocket_connections.discard(websocket)

@router.get("/dashboard")
async def get_monitoring_dashboard(
    service: MonitoringService = Depends(get_monitoring_service),
    current_user: str = Depends(get_current_user)
):
    """Get monitoring dashboard data"""
    
    # Get current system metrics
    system_stats = {
        'cpu_usage': psutil.cpu_percent(),
        'memory_usage': psutil.virtual_memory().percent,
        'disk_usage': (psutil.disk_usage('/').used / psutil.disk_usage('/').total) * 100
    }
    
    # Get active alerts count
    active_alerts_count = service.db.query(Alert).filter(Alert.status == AlertStatus.ACTIVE).count()
    
    # Get monitor counts
    total_monitors = service.db.query(Monitor).count()
    enabled_monitors = service.db.query(Monitor).filter(Monitor.is_enabled == True).count()
    
    return {
        'system_stats': system_stats,
        'active_alerts_count': active_alerts_count,
        'total_monitors': total_monitors,
        'enabled_monitors': enabled_monitors,
        'timestamp': datetime.utcnow().isoformat()
    }