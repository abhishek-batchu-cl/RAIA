"""
RAIA Platform - Unified Backend API
Responsible AI Analytics Platform combining Agent Evaluation and Model Explainability
"""

import logging
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict

import structlog
import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.database import create_tables, get_database
from app.core.exceptions import RAIAException
from app.core.middleware import (
    LoggingMiddleware,
    RateLimitMiddleware,
    SecurityHeadersMiddleware,
)
from app.core.monitoring import setup_monitoring, metrics
from app.services.monitoring_service import MonitoringService

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("Starting RAIA Platform backend", version="1.0.0")
    
    try:
        # Initialize database
        logger.info("Initializing database...")
        await create_tables()
        logger.info("Database initialized successfully")
        
        # Setup monitoring
        logger.info("Setting up monitoring...")
        setup_monitoring(app)
        logger.info("Monitoring setup complete")
        
        # Initialize services
        monitoring_service = MonitoringService()
        await monitoring_service.initialize()
        app.state.monitoring_service = monitoring_service
        
        logger.info("RAIA Platform backend started successfully")
        metrics.app_startup_total.inc()
        
        yield
        
    except Exception as e:
        logger.error("Failed to start application", error=str(e))
        raise
    
    # Shutdown
    logger.info("Shutting down RAIA Platform backend")
    if hasattr(app.state, 'monitoring_service'):
        await app.state.monitoring_service.cleanup()
    logger.info("RAIA Platform backend shutdown complete")


# Initialize FastAPI application
app = FastAPI(
    title="RAIA Platform API",
    description="""
    ## Responsible AI Analytics Platform
    
    A unified enterprise-scale platform combining comprehensive AI agent evaluation 
    and model explainability capabilities.
    
    ### Features
    - **Agent Evaluation**: Multi-model testing, RAG pipelines, real-time chat
    - **Model Explainability**: SHAP, LIME, fairness analysis, drift detection
    - **Enterprise**: Multi-tenancy, RBAC, monitoring, compliance reporting
    
    ### Authentication
    All endpoints require authentication except for health checks and public documentation.
    Use JWT tokens with the `Authorization: Bearer <token>` header.
    """,
    version="1.0.0",
    openapi_url="/api/v1/openapi.json" if settings.ENVIRONMENT != "production" else None,
    docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT != "production" else None,
    lifespan=lifespan,
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if settings.ENVIRONMENT == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.TRUSTED_HOSTS
    )

app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(LoggingMiddleware)

# Include API routes
app.include_router(api_router, prefix="/api/v1")


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint for load balancers and monitoring systems.
    """
    try:
        # Check database connectivity
        async with get_database() as db:
            await db.execute("SELECT 1")
        
        return {
            "status": "healthy",
            "service": "raia-platform-backend",
            "version": "1.0.0",
            "timestamp": metrics.get_current_timestamp(),
            "database": "connected",
            "environment": settings.ENVIRONMENT
        }
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "service": "raia-platform-backend",
                "error": str(e)
            }
        )


@app.get("/health/ready", tags=["Health"])
async def readiness_check():
    """
    Readiness check for Kubernetes deployments.
    """
    return {
        "status": "ready",
        "service": "raia-platform-backend",
        "timestamp": metrics.get_current_timestamp()
    }


@app.get("/health/live", tags=["Health"])
async def liveness_check():
    """
    Liveness check for Kubernetes deployments.
    """
    return {
        "status": "alive",
        "service": "raia-platform-backend",
        "timestamp": metrics.get_current_timestamp()
    }


@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """
    Prometheus metrics endpoint.
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.exception_handler(RAIAException)
async def raia_exception_handler(request: Request, exc: RAIAException):
    """
    Handle custom RAIA exceptions.
    """
    logger.error(
        "RAIA exception occurred",
        path=request.url.path,
        method=request.method,
        error=exc.message,
        code=exc.code
    )
    
    metrics.api_errors_total.labels(
        endpoint=request.url.path,
        method=request.method,
        error_type=exc.__class__.__name__
    ).inc()
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.message,
            "code": exc.code,
            "type": exc.__class__.__name__,
            "timestamp": metrics.get_current_timestamp()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    Handle general exceptions.
    """
    logger.error(
        "Unhandled exception occurred",
        path=request.url.path,
        method=request.method,
        error=str(exc),
        exc_info=True
    )
    
    metrics.api_errors_total.labels(
        endpoint=request.url.path,
        method=request.method,
        error_type="InternalServerError"
    ).inc()
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "code": "INTERNAL_ERROR",
            "type": "InternalServerError",
            "timestamp": metrics.get_current_timestamp()
        }
    )


@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint with platform information.
    """
    return {
        "service": "RAIA Platform API",
        "version": "1.0.0",
        "description": "Responsible AI Analytics Platform",
        "docs_url": "/docs" if settings.ENVIRONMENT != "production" else None,
        "health_url": "/health",
        "api_prefix": "/api/v1",
        "features": [
            "Agent Evaluation",
            "Model Explainability", 
            "Data Management",
            "Real-time Monitoring",
            "Enterprise Security"
        ]
    }


if __name__ == "__main__":
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["default"]["fmt"] = "%(asctime)s [%(name)s] %(levelprefix)s %(message)s"
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=settings.ENVIRONMENT == "development",
        log_config=log_config,
        access_log=True,
    )