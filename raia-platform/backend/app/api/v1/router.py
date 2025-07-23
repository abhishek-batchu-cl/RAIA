"""
RAIA Platform - Unified API Router
Main API router combining all platform features
"""

from fastapi import APIRouter, Depends

from app.api.v1.endpoints import (
    auth,
    users,
    organizations,
    models,
    agent_evaluation,
    model_explainability,
    data_management,
    monitoring,
    analytics,
    dashboards,
    chat,
    alerts,
    rag_evaluation,
    llm_evaluation,
    advanced_explainability,
    what_if_analysis,
    model_statistics,
    websocket_monitoring
)
from app.core.auth import get_current_active_user
from app.models.schemas import User

# Main API router
api_router = APIRouter()

# Public endpoints (no authentication required)
api_router.include_router(
    auth.router,
    prefix="/auth",
    tags=["Authentication"]
)

# Protected endpoints (authentication required)
# User and organization management
api_router.include_router(
    users.router,
    prefix="/users",
    tags=["User Management"],
    dependencies=[Depends(get_current_active_user)]
)

api_router.include_router(
    organizations.router,
    prefix="/organizations", 
    tags=["Organization Management"],
    dependencies=[Depends(get_current_active_user)]
)

# Model management
api_router.include_router(
    models.router,
    prefix="/models",
    tags=["Model Management"],
    dependencies=[Depends(get_current_active_user)]
)

# Agent evaluation features
api_router.include_router(
    agent_evaluation.router,
    prefix="/agent-evaluation",
    tags=["Agent Evaluation"],
    dependencies=[Depends(get_current_active_user)]
)

# Model explainability features
api_router.include_router(
    model_explainability.router,
    prefix="/model-explainability", 
    tags=["Model Explainability"],
    dependencies=[Depends(get_current_active_user)]
)

# Data management
api_router.include_router(
    data_management.router,
    prefix="/data",
    tags=["Data Management"],
    dependencies=[Depends(get_current_active_user)]
)

# Monitoring and system health
api_router.include_router(
    monitoring.router,
    prefix="/monitoring",
    tags=["Monitoring"],
    dependencies=[Depends(get_current_active_user)]
)

# Analytics and reporting
api_router.include_router(
    analytics.router,
    prefix="/analytics", 
    tags=["Analytics & Reporting"],
    dependencies=[Depends(get_current_active_user)]
)

# Dashboard features
api_router.include_router(
    dashboards.router,
    prefix="/dashboards",
    tags=["Dashboards"],
    dependencies=[Depends(get_current_active_user)]
)

# Chat interface
api_router.include_router(
    chat.router,
    prefix="/chat",
    tags=["Chat Interface"], 
    dependencies=[Depends(get_current_active_user)]
)

# Alerts and notifications
api_router.include_router(
    alerts.router,
    prefix="/alerts",
    tags=["Alerts & Notifications"],
    dependencies=[Depends(get_current_active_user)]
)

# Advanced AI Evaluation Services
# RAG evaluation
api_router.include_router(
    rag_evaluation.router,
    prefix="/rag-evaluation",
    tags=["RAG Evaluation"],
    dependencies=[Depends(get_current_active_user)]
)

# LLM evaluation
api_router.include_router(
    llm_evaluation.router,
    prefix="/llm-evaluation",
    tags=["LLM Evaluation"],
    dependencies=[Depends(get_current_active_user)]
)

# Advanced explainability
api_router.include_router(
    advanced_explainability.router,
    prefix="/advanced-explainability",
    tags=["Advanced Explainability"],
    dependencies=[Depends(get_current_active_user)]
)

# What-if analysis
api_router.include_router(
    what_if_analysis.router,
    prefix="/what-if-analysis",
    tags=["What-If Analysis"],
    dependencies=[Depends(get_current_active_user)]
)

# Model statistics
api_router.include_router(
    model_statistics.router,
    prefix="/model-statistics",
    tags=["Model Statistics"],
    dependencies=[Depends(get_current_active_user)]
)

# WebSocket monitoring
api_router.include_router(
    websocket_monitoring.router,
    prefix="/websocket",
    tags=["WebSocket Monitoring"],
    dependencies=[Depends(get_current_active_user)]
)

# Enterprise Dashboard (Executive-level insights)
from app.api.v1.endpoints import enterprise_dashboard

api_router.include_router(
    enterprise_dashboard.router,
    prefix="/enterprise-dashboard",
    tags=["Enterprise Dashboard"],
    dependencies=[Depends(get_current_active_user)]
)

# Data Export Services
from app.api.v1.endpoints import data_export

api_router.include_router(
    data_export.router,
    prefix="/data-export",
    tags=["Data Export"],
    dependencies=[Depends(get_current_active_user)]
)

# Cache Management
from app.api.v1.endpoints import cache_management

api_router.include_router(
    cache_management.router,
    prefix="/cache",
    tags=["Cache Management"],
    dependencies=[Depends(get_current_active_user)]
)