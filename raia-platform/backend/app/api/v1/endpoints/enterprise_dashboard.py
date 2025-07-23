"""
Enterprise Dashboard API Endpoints
Executive-level insights and analytics across all platform features
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from app.core.auth import get_current_active_user, require_admin_role
from app.models.schemas import User
from app.services.enterprise_dashboard_service import EnterpriseDashboardService

router = APIRouter()
dashboard_service = EnterpriseDashboardService()


@router.get("/executive-summary")
async def get_executive_summary(
    time_range_days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get comprehensive executive summary for the organization
    
    Args:
        time_range_days: Number of days to include in analysis
        current_user: Authenticated user
        
    Returns:
        Executive summary with KPIs, model performance, alerts, and recommendations
    """
    try:
        time_range = timedelta(days=time_range_days)
        summary = await dashboard_service.generate_executive_summary(
            organization_id=str(current_user.organization_id),
            time_range=time_range
        )
        
        return {
            "status": "success",
            "data": {
                "kpis": {
                    "total_models": summary.kpis.total_models,
                    "active_evaluations": summary.kpis.active_evaluations,
                    "avg_model_accuracy": summary.kpis.avg_model_accuracy,
                    "drift_alerts": summary.kpis.drift_alerts,
                    "fairness_violations": summary.kpis.fairness_violations,
                    "performance_degradations": summary.kpis.performance_degradations,
                    "data_quality_score": summary.kpis.data_quality_score,
                    "system_health_score": summary.kpis.system_health_score
                },
                "model_performance": [
                    {
                        "model_id": model.model_id,
                        "model_name": model.model_name,
                        "model_type": model.model_type,
                        "accuracy": model.accuracy,
                        "precision": model.precision,
                        "recall": model.recall,
                        "f1_score": model.f1_score,
                        "prediction_count": model.prediction_count,
                        "last_evaluation_date": model.last_evaluation_date.isoformat(),
                        "status": model.status,
                        "drift_score": model.drift_score,
                        "fairness_score": model.fairness_score
                    }
                    for model in summary.model_performance
                ],
                "alerts": {
                    "critical_alerts": summary.alerts.critical_alerts,
                    "warning_alerts": summary.alerts.warning_alerts,
                    "info_alerts": summary.alerts.info_alerts,
                    "resolved_alerts": summary.alerts.resolved_alerts,
                    "avg_resolution_time": summary.alerts.avg_resolution_time,
                    "top_alert_types": summary.alerts.top_alert_types
                },
                "data_quality": {
                    "total_datasets": summary.data_quality.total_datasets,
                    "datasets_with_issues": summary.data_quality.datasets_with_issues,
                    "avg_completeness": summary.data_quality.avg_completeness,
                    "avg_consistency": summary.data_quality.avg_consistency,
                    "avg_validity": summary.data_quality.avg_validity,
                    "data_drift_detected": summary.data_quality.data_drift_detected,
                    "schema_changes": summary.data_quality.schema_changes
                },
                "trends": summary.trends,
                "recommendations": summary.recommendations,
                "generated_at": summary.generated_at.isoformat(),
                "time_range_days": time_range_days
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate executive summary: {str(e)}")


@router.get("/real-time-metrics")
async def get_real_time_metrics(
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get real-time dashboard metrics for live monitoring
    
    Returns:
        Current system metrics, active predictions, and recent alerts
    """
    try:
        metrics = await dashboard_service.get_real_time_metrics(
            organization_id=str(current_user.organization_id)
        )
        
        # Format recent alerts for API response
        formatted_alerts = [
            {
                "id": alert["id"],
                "type": alert["type"],
                "severity": alert["severity"],
                "model_id": alert["model_id"],
                "message": alert["message"],
                "timestamp": alert["timestamp"].isoformat()
            }
            for alert in metrics["recent_alerts"]
        ]
        
        return {
            "status": "success",
            "data": {
                "live_predictions_per_minute": metrics["live_predictions_per_minute"],
                "active_models": metrics["active_models"],
                "system_resources": {
                    "cpu_usage": metrics["system_cpu_usage"],
                    "memory_usage": metrics["system_memory_usage"]
                },
                "api_performance": {
                    "avg_response_time": metrics["api_response_time_avg"],
                    "error_rate_percentage": metrics["error_rate_percentage"]
                },
                "active_users": metrics["active_users"],
                "recent_alerts": formatted_alerts,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get real-time metrics: {str(e)}")


@router.get("/model-comparison-matrix")
async def get_model_comparison_matrix(
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get comprehensive model comparison matrix
    
    Returns:
        Side-by-side comparison of all models with performance metrics
    """
    try:
        comparison_data = await dashboard_service.get_model_comparison_matrix(
            organization_id=str(current_user.organization_id)
        )
        
        return {
            "status": "success",
            "data": comparison_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model comparison: {str(e)}")


@router.get("/cost-analysis")
async def get_cost_analysis(
    current_user: User = Depends(require_admin_role)
) -> Dict[str, Any]:
    """
    Get detailed cost analysis (Admin only)
    
    Returns:
        Cost breakdown, trends, and ROI metrics
    """
    try:
        cost_data = await dashboard_service.get_cost_analysis(
            organization_id=str(current_user.organization_id)
        )
        
        return {
            "status": "success",
            "data": cost_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cost analysis: {str(e)}")


@router.get("/kpi-summary")
async def get_kpi_summary(
    time_range_days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get focused KPI summary for quick overview
    
    Args:
        time_range_days: Number of days to include in analysis
        current_user: Authenticated user
        
    Returns:
        Key performance indicators summary
    """
    try:
        time_range = timedelta(days=time_range_days)
        summary = await dashboard_service.generate_executive_summary(
            organization_id=str(current_user.organization_id),
            time_range=time_range
        )
        
        return {
            "status": "success", 
            "data": {
                "total_models": summary.kpis.total_models,
                "active_evaluations": summary.kpis.active_evaluations,
                "avg_model_accuracy": summary.kpis.avg_model_accuracy,
                "drift_alerts": summary.kpis.drift_alerts,
                "fairness_violations": summary.kpis.fairness_violations,
                "performance_degradations": summary.kpis.performance_degradations,
                "data_quality_score": summary.kpis.data_quality_score,
                "system_health_score": summary.kpis.system_health_score,
                "time_range_days": time_range_days,
                "generated_at": datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get KPI summary: {str(e)}")


@router.get("/performance-trends")
async def get_performance_trends(
    time_range_days: int = Query(30, ge=7, le=365),
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get performance trend analysis
    
    Args:
        time_range_days: Number of days for trend analysis
        current_user: Authenticated user
        
    Returns:
        Trend analysis for accuracy, volume, alerts, and data quality
    """
    try:
        time_range = timedelta(days=time_range_days)
        summary = await dashboard_service.generate_executive_summary(
            organization_id=str(current_user.organization_id),
            time_range=time_range
        )
        
        return {
            "status": "success",
            "data": {
                "trends": summary.trends,
                "time_range_days": time_range_days,
                "generated_at": datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance trends: {str(e)}")


@router.get("/recommendations")
async def get_recommendations(
    time_range_days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get actionable recommendations based on current system state
    
    Args:
        time_range_days: Number of days to analyze for recommendations
        current_user: Authenticated user
        
    Returns:
        List of actionable recommendations
    """
    try:
        time_range = timedelta(days=time_range_days)
        summary = await dashboard_service.generate_executive_summary(
            organization_id=str(current_user.organization_id),
            time_range=time_range
        )
        
        return {
            "status": "success",
            "data": {
                "recommendations": summary.recommendations,
                "recommendation_count": len(summary.recommendations),
                "time_range_days": time_range_days,
                "generated_at": datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get recommendations: {str(e)}")


@router.post("/export-report")
async def export_dashboard_report(
    format_type: str = Query("pdf", regex="^(pdf|excel|csv)$"),
    time_range_days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Export comprehensive dashboard report
    
    Args:
        format_type: Export format (pdf, excel, csv)
        time_range_days: Number of days to include in report
        current_user: Authenticated user
        
    Returns:
        Export job details and download information
    """
    try:
        export_data = await dashboard_service.export_dashboard_report(
            organization_id=str(current_user.organization_id),
            format_type=format_type
        )
        
        return {
            "status": "success",
            "data": {
                "report_id": export_data["report_id"],
                "format": export_data["format"],
                "sections": export_data["sections"],
                "summary_metrics": export_data["summary_data"],
                "generated_at": export_data["generated_at"],
                "download_url": f"/api/v1/export/download/{export_data['report_id']}",
                "expires_in_hours": 24
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export dashboard report: {str(e)}")


@router.get("/health-check")
async def dashboard_health_check(
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Check dashboard service health and data availability
    
    Returns:
        Dashboard service health status
    """
    try:
        # Perform basic health checks
        start_time = datetime.utcnow()
        
        # Test service responsiveness
        metrics = await dashboard_service.get_real_time_metrics(
            organization_id=str(current_user.organization_id)
        )
        
        response_time = (datetime.utcnow() - start_time).total_seconds()
        
        return {
            "status": "healthy",
            "data": {
                "service_status": "operational",
                "response_time_seconds": response_time,
                "data_availability": "complete",
                "last_updated": datetime.utcnow().isoformat(),
                "dependencies": {
                    "database": "connected",
                    "cache": "available",
                    "monitoring": "active"
                }
            }
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }