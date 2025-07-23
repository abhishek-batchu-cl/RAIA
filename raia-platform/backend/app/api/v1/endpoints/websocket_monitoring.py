"""
WebSocket Monitoring API Endpoints
Real-time monitoring and notifications via WebSocket connections
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, Query, Form
from typing import Dict, List, Optional, Any
import json
import uuid
import asyncio

from app.core.auth import get_current_active_user_ws, get_current_active_user
from app.models.schemas import User
from app.websockets.monitoring import connection_manager, monitoring_service, EventType

router = APIRouter()

@router.websocket("/connect")
async def websocket_monitoring_endpoint(
    websocket: WebSocket,
    token: str = Query(...),
    subscriptions: Optional[str] = Query(None)  # Comma-separated list
):
    """WebSocket endpoint for real-time monitoring"""
    connection_id = str(uuid.uuid4())
    user = None
    
    try:
        # Authenticate user via token
        # This would need to be implemented based on your auth system
        # For now, we'll simulate authentication
        user = await get_current_active_user_ws(token)
        
        # Parse subscriptions
        subscription_list = []
        if subscriptions:
            subscription_list = [s.strip() for s in subscriptions.split(",")]
        
        # Accept connection
        await connection_manager.connect(websocket, connection_id, user, subscription_list)
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle different message types
                await handle_websocket_message(connection_id, message, user)
            
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await connection_manager.send_to_connection(connection_id, {
                    "error": "Invalid JSON format"
                })
            except Exception as e:
                await connection_manager.send_to_connection(connection_id, {
                    "error": f"Message processing error: {str(e)}"
                })
    
    except Exception as e:
        if websocket.client_state.value == 1:  # CONNECTING
            await websocket.close(code=1008, reason=f"Authentication failed: {str(e)}")
        return
    
    finally:
        await connection_manager.disconnect(connection_id)

async def handle_websocket_message(connection_id: str, message: Dict[str, Any], user: User):
    """Handle incoming WebSocket messages"""
    message_type = message.get("type")
    
    if message_type == "subscribe":
        # Subscribe to additional events/models
        subscriptions = message.get("subscriptions", [])
        await connection_manager.subscribe_to_events(connection_id, subscriptions)
        
        await connection_manager.send_to_connection(connection_id, {
            "type": "subscription_confirmed",
            "subscriptions": subscriptions
        })
    
    elif message_type == "start_model_monitoring":
        # Start monitoring for a specific model
        model_id = message.get("model_id")
        config = message.get("config", {})
        
        if model_id:
            await monitoring_service.start_model_monitoring(model_id, config)
            await connection_manager.send_to_connection(connection_id, {
                "type": "monitoring_started",
                "model_id": model_id
            })
    
    elif message_type == "stop_model_monitoring":
        # Stop monitoring for a specific model
        model_id = message.get("model_id")
        
        if model_id:
            await monitoring_service.stop_model_monitoring(model_id)
            await connection_manager.send_to_connection(connection_id, {
                "type": "monitoring_stopped",
                "model_id": model_id
            })
    
    elif message_type == "ping":
        # Respond to ping with pong
        await connection_manager.send_to_connection(connection_id, {
            "type": "pong",
            "timestamp": message.get("timestamp")
        })
    
    else:
        await connection_manager.send_to_connection(connection_id, {
            "error": f"Unknown message type: {message_type}"
        })

@router.post("/broadcast")
async def broadcast_message(
    message: str = Form(...),
    event_type: str = Form("system"),
    severity: str = Form("info"),
    current_user: User = Depends(get_current_active_user)
):
    """Broadcast a message to all connected clients"""
    try:
        # Parse message if it's JSON
        try:
            message_data = json.loads(message)
        except json.JSONDecodeError:
            message_data = {"message": message}
        
        broadcast_data = {
            "type": "broadcast",
            "event_type": event_type,
            "severity": severity,
            "data": message_data,
            "sender": current_user.email,
            "timestamp": json.dumps({"now": True}, default=str)
        }
        
        await connection_manager.broadcast_to_all(broadcast_data)
        
        return {
            "status": "success",
            "message": "Message broadcasted successfully",
            "recipients": len(connection_manager.active_connections)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/notify-user")
async def notify_specific_user(
    user_id: str = Form(...),
    message: str = Form(...),
    event_type: str = Form("notification"),
    severity: str = Form("info"),
    current_user: User = Depends(get_current_active_user)
):
    """Send notification to a specific user"""
    try:
        # Parse message if it's JSON
        try:
            message_data = json.loads(message)
        except json.JSONDecodeError:
            message_data = {"message": message}
        
        notification_data = {
            "type": "user_notification",
            "event_type": event_type,
            "severity": severity,
            "data": message_data,
            "sender": current_user.email,
            "timestamp": json.dumps({"now": True}, default=str)
        }
        
        await connection_manager.send_to_user(user_id, notification_data)
        
        # Check if user has active connections
        user_connections = connection_manager.user_connections.get(user_id, [])
        
        return {
            "status": "success",
            "message": "Notification sent successfully",
            "user_id": user_id,
            "active_connections": len(user_connections)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/start-monitoring")
async def start_model_monitoring(
    model_id: str = Form(...),
    interval: int = Form(30),
    config: Optional[str] = Form(None),  # JSON string
    current_user: User = Depends(get_current_active_user)
):
    """Start real-time monitoring for a model"""
    try:
        monitoring_config = {
            "interval": interval,
            "user_id": str(current_user.id)
        }
        
        if config:
            additional_config = json.loads(config)
            monitoring_config.update(additional_config)
        
        await monitoring_service.start_model_monitoring(model_id, monitoring_config)
        
        return {
            "status": "success",
            "message": f"Monitoring started for model {model_id}",
            "model_id": model_id,
            "config": monitoring_config
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/stop-monitoring")
async def stop_model_monitoring(
    model_id: str = Form(...),
    current_user: User = Depends(get_current_active_user)
):
    """Stop real-time monitoring for a model"""
    try:
        await monitoring_service.stop_model_monitoring(model_id)
        
        return {
            "status": "success",
            "message": f"Monitoring stopped for model {model_id}",
            "model_id": model_id
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/connections")
async def get_connection_stats(
    current_user: User = Depends(get_current_active_user)
):
    """Get WebSocket connection statistics"""
    try:
        stats = connection_manager.get_connection_stats()
        
        # Add detailed connection info for admin users
        if hasattr(current_user, 'is_admin') and current_user.is_admin:
            detailed_connections = []
            for conn_id, metadata in connection_manager.connection_metadata.items():
                detailed_connections.append({
                    "connection_id": conn_id,
                    "user_id": metadata.get("user_id"),
                    "user_email": metadata.get("user_email"),
                    "connected_at": metadata.get("connected_at"),
                    "subscriptions": metadata.get("subscriptions", [])
                })
            stats["detailed_connections"] = detailed_connections
        
        return {
            "status": "success",
            "connection_stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/active-monitoring")
async def get_active_monitoring(
    current_user: User = Depends(get_current_active_user)
):
    """Get list of actively monitored models"""
    try:
        active_models = list(monitoring_service.monitoring_tasks.keys())
        
        return {
            "status": "success",
            "active_monitoring": {
                "model_count": len(active_models),
                "model_ids": active_models
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/simulate-event")
async def simulate_monitoring_event(
    event_type: str = Form(...),
    model_id: Optional[str] = Form(None),
    severity: str = Form("info"),
    data: str = Form("{}"),  # JSON string
    current_user: User = Depends(get_current_active_user)
):
    """Simulate a monitoring event for testing purposes"""
    try:
        # Parse event data
        event_data = json.loads(data)
        
        # Validate event type
        try:
            event_type_enum = EventType(event_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid event type: {event_type}")
        
        # Simulate specific event types
        if event_type_enum == EventType.MODEL_PERFORMANCE:
            event_data.update({
                "accuracy": 0.92,
                "precision": 0.89,
                "recall": 0.87,
                "f1_score": 0.88,
                "prediction_count": 1245,
                "avg_response_time": 0.23
            })
        
        elif event_type_enum == EventType.DATA_DRIFT_DETECTION:
            event_data.update({
                "drift_detected": True,
                "drift_score": 0.78,
                "affected_features": ["feature_1", "feature_3", "feature_7"],
                "drift_magnitude": "moderate"
            })
        
        elif event_type_enum == EventType.ANOMALY_DETECTION:
            event_data.update({
                "anomaly_type": "performance_degradation",
                "confidence": 0.85,
                "affected_metrics": ["accuracy", "response_time"]
            })
        
        # Broadcast the simulated event
        from app.websockets.monitoring import WebSocketEvent
        import uuid
        from datetime import datetime
        
        event = WebSocketEvent(
            event_type=event_type_enum,
            timestamp=datetime.now().isoformat(),
            data=event_data,
            event_id=str(uuid.uuid4()),
            model_id=model_id,
            severity=severity
        )
        
        await connection_manager.broadcast_event(event)
        
        return {
            "status": "success",
            "message": "Event simulated and broadcasted",
            "event_id": event.event_id,
            "recipients": len(connection_manager.active_connections)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/event-types")
async def get_available_event_types(
    current_user: User = Depends(get_current_active_user)
):
    """Get list of available WebSocket event types"""
    return {
        "status": "success",
        "event_types": [
            {
                "value": event_type.value,
                "description": event_type.name.replace("_", " ").title()
            }
            for event_type in EventType
        ]
    }

@router.get("/subscription-formats")
async def get_subscription_formats(
    current_user: User = Depends(get_current_active_user)
):
    """Get information about WebSocket subscription formats"""
    return {
        "status": "success",
        "subscription_formats": {
            "model_specific": {
                "format": "model:<model_id>",
                "example": "model:my_classifier_v2",
                "description": "Subscribe to events for a specific model"
            },
            "event_specific": {
                "format": "event:<event_type>",
                "example": "event:model_performance",
                "description": "Subscribe to specific event types"
            },
            "user_specific": {
                "format": "user:<user_id>",
                "example": "user:123",
                "description": "Subscribe to user-specific notifications"
            }
        },
        "available_events": [event_type.value for event_type in EventType]
    }