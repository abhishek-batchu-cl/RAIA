"""
Real-time WebSocket Monitoring System
Provides real-time updates for model performance, system health, and evaluation metrics
"""

from fastapi import WebSocket, WebSocketDisconnect, Depends
from typing import Dict, List, Optional, Any
import asyncio
import json
import logging
from datetime import datetime
import uuid
from dataclasses import dataclass, asdict
from enum import Enum

from app.core.auth import get_current_active_user_ws
from app.models.schemas import User

logger = logging.getLogger(__name__)

class EventType(Enum):
    """WebSocket event types"""
    MODEL_PERFORMANCE = "model_performance"
    SYSTEM_HEALTH = "system_health"
    EVALUATION_PROGRESS = "evaluation_progress"
    ALERT_NOTIFICATION = "alert_notification"
    DATA_DRIFT_DETECTION = "data_drift_detection"
    FAIRNESS_VIOLATION = "fairness_violation"
    ANOMALY_DETECTION = "anomaly_detection"
    BATCH_JOB_STATUS = "batch_job_status"
    RESOURCE_USAGE = "resource_usage"
    ERROR_NOTIFICATION = "error_notification"

@dataclass
class WebSocketEvent:
    """WebSocket event structure"""
    event_type: EventType
    timestamp: str
    data: Dict[str, Any]
    event_id: str
    user_id: Optional[str] = None
    model_id: Optional[str] = None
    severity: str = "info"  # info, warning, error, critical

class ConnectionManager:
    """Manages WebSocket connections and message broadcasting"""
    
    def __init__(self):
        # Active connections by connection ID
        self.active_connections: Dict[str, WebSocket] = {}
        # User connections mapping
        self.user_connections: Dict[str, List[str]] = {}
        # Model subscriptions
        self.model_subscriptions: Dict[str, List[str]] = {}
        # Event type subscriptions
        self.event_subscriptions: Dict[EventType, List[str]] = {}
        # Connection metadata
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
    
    async def connect(
        self,
        websocket: WebSocket,
        connection_id: str,
        user: User,
        subscriptions: Optional[List[str]] = None
    ):
        """Accept new WebSocket connection"""
        await websocket.accept()
        
        # Store connection
        self.active_connections[connection_id] = websocket
        
        # Map user to connection
        user_id = str(user.id)
        if user_id not in self.user_connections:
            self.user_connections[user_id] = []
        self.user_connections[user_id].append(connection_id)
        
        # Store connection metadata
        self.connection_metadata[connection_id] = {
            "user_id": user_id,
            "user_email": user.email,
            "connected_at": datetime.now().isoformat(),
            "subscriptions": subscriptions or []
        }
        
        # Set up subscriptions
        if subscriptions:
            await self.subscribe_to_events(connection_id, subscriptions)
        
        logger.info(f"WebSocket connection established: {connection_id} for user {user.email}")
        
        # Send connection confirmation
        await self.send_to_connection(connection_id, {
            "event_type": "connection_established",
            "connection_id": connection_id,
            "timestamp": datetime.now().isoformat(),
            "message": "WebSocket connection established successfully"
        })
    
    async def disconnect(self, connection_id: str):
        """Handle WebSocket disconnection"""
        if connection_id in self.active_connections:
            # Remove from active connections
            del self.active_connections[connection_id]
            
            # Get connection metadata
            metadata = self.connection_metadata.get(connection_id, {})
            user_id = metadata.get("user_id")
            
            # Remove from user connections
            if user_id and user_id in self.user_connections:
                if connection_id in self.user_connections[user_id]:
                    self.user_connections[user_id].remove(connection_id)
                if not self.user_connections[user_id]:
                    del self.user_connections[user_id]
            
            # Remove from subscriptions
            for model_connections in self.model_subscriptions.values():
                if connection_id in model_connections:
                    model_connections.remove(connection_id)
            
            for event_connections in self.event_subscriptions.values():
                if connection_id in event_connections:
                    event_connections.remove(connection_id)
            
            # Clean up metadata
            if connection_id in self.connection_metadata:
                del self.connection_metadata[connection_id]
            
            logger.info(f"WebSocket connection disconnected: {connection_id}")
    
    async def subscribe_to_events(self, connection_id: str, subscriptions: List[str]):
        """Subscribe connection to specific events or models"""
        for subscription in subscriptions:
            if subscription.startswith("model:"):
                # Model-specific subscription
                model_id = subscription.replace("model:", "")
                if model_id not in self.model_subscriptions:
                    self.model_subscriptions[model_id] = []
                if connection_id not in self.model_subscriptions[model_id]:
                    self.model_subscriptions[model_id].append(connection_id)
            
            elif subscription.startswith("event:"):
                # Event type subscription
                event_type_str = subscription.replace("event:", "")
                try:
                    event_type = EventType(event_type_str)
                    if event_type not in self.event_subscriptions:
                        self.event_subscriptions[event_type] = []
                    if connection_id not in self.event_subscriptions[event_type]:
                        self.event_subscriptions[event_type].append(connection_id)
                except ValueError:
                    logger.warning(f"Invalid event type subscription: {event_type_str}")
    
    async def send_to_connection(self, connection_id: str, message: Dict[str, Any]):
        """Send message to specific connection"""
        if connection_id in self.active_connections:
            try:
                websocket = self.active_connections[connection_id]
                await websocket.send_text(json.dumps(message, default=str))
            except Exception as e:
                logger.error(f"Failed to send message to connection {connection_id}: {e}")
                await self.disconnect(connection_id)
    
    async def send_to_user(self, user_id: str, message: Dict[str, Any]):
        """Send message to all connections of a specific user"""
        if user_id in self.user_connections:
            for connection_id in self.user_connections[user_id]:
                await self.send_to_connection(connection_id, message)
    
    async def broadcast_event(self, event: WebSocketEvent):
        """Broadcast event to relevant subscribers"""
        message = asdict(event)
        message["event_type"] = event.event_type.value  # Convert enum to string
        
        # Send to event type subscribers
        if event.event_type in self.event_subscriptions:
            for connection_id in self.event_subscriptions[event.event_type]:
                await self.send_to_connection(connection_id, message)
        
        # Send to model subscribers if model_id is specified
        if event.model_id and event.model_id in self.model_subscriptions:
            for connection_id in self.model_subscriptions[event.model_id]:
                await self.send_to_connection(connection_id, message)
        
        # Send to user if user_id is specified
        if event.user_id:
            await self.send_to_user(event.user_id, message)
    
    async def broadcast_to_all(self, message: Dict[str, Any]):
        """Broadcast message to all active connections"""
        for connection_id in list(self.active_connections.keys()):
            await self.send_to_connection(connection_id, message)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        return {
            "total_connections": len(self.active_connections),
            "unique_users": len(self.user_connections),
            "model_subscriptions": len(self.model_subscriptions),
            "event_subscriptions": len(self.event_subscriptions),
            "active_connection_ids": list(self.active_connections.keys())
        }

# Global connection manager instance
connection_manager = ConnectionManager()

class MonitoringService:
    """Service for generating and broadcasting monitoring events"""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}
    
    async def start_model_monitoring(self, model_id: str, monitoring_config: Dict[str, Any]):
        """Start continuous monitoring for a specific model"""
        if model_id in self.monitoring_tasks:
            # Stop existing monitoring task
            self.monitoring_tasks[model_id].cancel()
        
        # Start new monitoring task
        self.monitoring_tasks[model_id] = asyncio.create_task(
            self._model_monitoring_loop(model_id, monitoring_config)
        )
        logger.info(f"Started monitoring for model: {model_id}")
    
    async def stop_model_monitoring(self, model_id: str):
        """Stop monitoring for a specific model"""
        if model_id in self.monitoring_tasks:
            self.monitoring_tasks[model_id].cancel()
            del self.monitoring_tasks[model_id]
            logger.info(f"Stopped monitoring for model: {model_id}")
    
    async def _model_monitoring_loop(self, model_id: str, config: Dict[str, Any]):
        """Continuous monitoring loop for a model"""
        interval = config.get("interval", 30)  # seconds
        
        try:
            while True:
                # Generate mock performance data (replace with actual monitoring logic)
                performance_data = await self._get_model_performance(model_id)
                
                event = WebSocketEvent(
                    event_type=EventType.MODEL_PERFORMANCE,
                    timestamp=datetime.now().isoformat(),
                    data=performance_data,
                    event_id=str(uuid.uuid4()),
                    model_id=model_id,
                    severity="info"
                )
                
                await self.connection_manager.broadcast_event(event)
                
                # Check for anomalies
                anomaly_data = await self._check_model_anomalies(model_id, performance_data)
                if anomaly_data:
                    anomaly_event = WebSocketEvent(
                        event_type=EventType.ANOMALY_DETECTION,
                        timestamp=datetime.now().isoformat(),
                        data=anomaly_data,
                        event_id=str(uuid.uuid4()),
                        model_id=model_id,
                        severity="warning"
                    )
                    await self.connection_manager.broadcast_event(anomaly_event)
                
                await asyncio.sleep(interval)
        
        except asyncio.CancelledError:
            logger.info(f"Monitoring loop cancelled for model: {model_id}")
        except Exception as e:
            logger.error(f"Error in monitoring loop for model {model_id}: {e}")
    
    async def _get_model_performance(self, model_id: str) -> Dict[str, Any]:
        """Get current model performance metrics"""
        # This would integrate with actual model monitoring services
        # For now, return mock data
        import random
        
        return {
            "model_id": model_id,
            "accuracy": round(random.uniform(0.85, 0.95), 4),
            "precision": round(random.uniform(0.80, 0.90), 4),
            "recall": round(random.uniform(0.75, 0.85), 4),
            "f1_score": round(random.uniform(0.78, 0.88), 4),
            "prediction_count": random.randint(100, 1000),
            "avg_response_time": round(random.uniform(0.1, 0.5), 3),
            "error_rate": round(random.uniform(0.01, 0.05), 4),
            "resource_usage": {
                "cpu_percent": round(random.uniform(20, 80), 2),
                "memory_percent": round(random.uniform(30, 70), 2),
                "gpu_percent": round(random.uniform(0, 90), 2)
            }
        }
    
    async def _check_model_anomalies(self, model_id: str, performance_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check for performance anomalies"""
        # Simple anomaly detection based on thresholds
        anomalies = []
        
        if performance_data.get("accuracy", 1.0) < 0.7:
            anomalies.append({
                "type": "low_accuracy",
                "value": performance_data["accuracy"],
                "threshold": 0.7,
                "message": "Model accuracy below threshold"
            })
        
        if performance_data.get("error_rate", 0.0) > 0.1:
            anomalies.append({
                "type": "high_error_rate",
                "value": performance_data["error_rate"],
                "threshold": 0.1,
                "message": "Model error rate above threshold"
            })
        
        if performance_data.get("avg_response_time", 0.0) > 1.0:
            anomalies.append({
                "type": "slow_response",
                "value": performance_data["avg_response_time"],
                "threshold": 1.0,
                "message": "Model response time above threshold"
            })
        
        if anomalies:
            return {
                "model_id": model_id,
                "anomalies": anomalies,
                "timestamp": datetime.now().isoformat()
            }
        
        return None
    
    async def broadcast_system_health(self):
        """Broadcast system health status"""
        health_data = await self._get_system_health()
        
        event = WebSocketEvent(
            event_type=EventType.SYSTEM_HEALTH,
            timestamp=datetime.now().isoformat(),
            data=health_data,
            event_id=str(uuid.uuid4()),
            severity="info"
        )
        
        await self.connection_manager.broadcast_event(event)
    
    async def _get_system_health(self) -> Dict[str, Any]:
        """Get system health metrics"""
        import psutil
        import random
        
        return {
            "cpu_usage": psutil.cpu_percent(interval=1),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "active_connections": len(self.connection_manager.active_connections),
            "active_models": len(self.monitoring_tasks),
            "uptime_seconds": random.randint(3600, 86400),  # Mock uptime
            "status": "healthy"
        }
    
    async def notify_evaluation_progress(
        self,
        evaluation_id: str,
        progress: float,
        status: str,
        details: Dict[str, Any],
        user_id: Optional[str] = None
    ):
        """Notify about evaluation progress"""
        event = WebSocketEvent(
            event_type=EventType.EVALUATION_PROGRESS,
            timestamp=datetime.now().isoformat(),
            data={
                "evaluation_id": evaluation_id,
                "progress": progress,
                "status": status,
                "details": details
            },
            event_id=str(uuid.uuid4()),
            user_id=user_id,
            severity="info"
        )
        
        await self.connection_manager.broadcast_event(event)
    
    async def notify_data_drift(
        self,
        model_id: str,
        drift_data: Dict[str, Any],
        severity: str = "warning"
    ):
        """Notify about data drift detection"""
        event = WebSocketEvent(
            event_type=EventType.DATA_DRIFT_DETECTION,
            timestamp=datetime.now().isoformat(),
            data=drift_data,
            event_id=str(uuid.uuid4()),
            model_id=model_id,
            severity=severity
        )
        
        await self.connection_manager.broadcast_event(event)
    
    async def notify_fairness_violation(
        self,
        model_id: str,
        fairness_data: Dict[str, Any],
        severity: str = "warning"
    ):
        """Notify about fairness violation"""
        event = WebSocketEvent(
            event_type=EventType.FAIRNESS_VIOLATION,
            timestamp=datetime.now().isoformat(),
            data=fairness_data,
            event_id=str(uuid.uuid4()),
            model_id=model_id,
            severity=severity
        )
        
        await self.connection_manager.broadcast_event(event)
    
    async def notify_batch_job_status(
        self,
        job_id: str,
        status: str,
        progress: float,
        details: Dict[str, Any],
        user_id: Optional[str] = None
    ):
        """Notify about batch job status"""
        event = WebSocketEvent(
            event_type=EventType.BATCH_JOB_STATUS,
            timestamp=datetime.now().isoformat(),
            data={
                "job_id": job_id,
                "status": status,
                "progress": progress,
                "details": details
            },
            event_id=str(uuid.uuid4()),
            user_id=user_id,
            severity="info"
        )
        
        await self.connection_manager.broadcast_event(event)

# Global monitoring service instance
monitoring_service = MonitoringService(connection_manager)

# Background task for system health monitoring
async def system_health_monitoring_task():
    """Background task for system health monitoring"""
    while True:
        try:
            await monitoring_service.broadcast_system_health()
            await asyncio.sleep(60)  # Every minute
        except Exception as e:
            logger.error(f"Error in system health monitoring: {e}")
            await asyncio.sleep(60)