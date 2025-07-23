"""
Data Export API Endpoints
Comprehensive data export functionality supporting PDF, Excel, and CSV formats
"""

from fastapi import APIRouter, Depends, HTTPException, Form, Query, Path
from fastapi.responses import FileResponse
from typing import Dict, List, Optional, Any
import os

from app.core.auth import get_current_active_user
from app.models.schemas import User
from app.services.data_export_service import DataExportService

router = APIRouter()
export_service = DataExportService()


@router.post("/evaluation-report")
async def export_evaluation_report(
    evaluation_id: str = Form(...),
    export_format: str = Form(..., regex="^(pdf|excel|csv)$"),
    include_raw_data: bool = Form(False),
    include_visualizations: bool = Form(True),
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Export evaluation results report
    
    Args:
        evaluation_id: ID of the evaluation to export
        export_format: Export format (pdf, excel, csv)
        include_raw_data: Include raw evaluation data
        include_visualizations: Include charts and graphs
        current_user: Authenticated user
        
    Returns:
        Export job details with download information
    """
    try:
        # Simulate fetching evaluation data
        # In real implementation, this would query the database
        evaluation_data = {
            "report_type": "Model Evaluation Report",
            "evaluation_id": evaluation_id,
            "model_info": {
                "name": "Customer Churn Predictor",
                "type": "Classification",
                "version": "1.2.3",
                "created_at": "2024-01-15T10:30:00Z"
            },
            "performance_metrics": {
                "accuracy": 0.924,
                "precision": 0.891,
                "recall": 0.887,
                "f1_score": 0.889,
                "auc_score": 0.956,
                "log_loss": 0.142
            },
            "fairness_metrics": {
                "overall_score": 0.87,
                "bias_metrics": {
                    "gender_bias": 0.08,
                    "age_bias": 0.12,
                    "ethnicity_bias": 0.06
                }
            },
            "summary": "This evaluation shows strong model performance with accuracy above 92%. Some minor fairness concerns detected in age-related predictions.",
            "sections": [
                "Model Overview",
                "Performance Analysis", 
                "Fairness Assessment",
                "Recommendations"
            ]
        }
        
        if include_raw_data:
            evaluation_data["raw_data"] = [
                {"prediction": 0.89, "actual": 1, "feature_1": 0.45, "feature_2": 0.67},
                {"prediction": 0.23, "actual": 0, "feature_1": 0.12, "feature_2": 0.34},
                {"prediction": 0.78, "actual": 1, "feature_1": 0.89, "feature_2": 0.56}
            ]
        
        export_options = {
            "include_visualizations": include_visualizations,
            "user_id": str(current_user.id),
            "organization_id": str(current_user.organization_id)
        }
        
        export_result = await export_service.export_evaluation_results(
            evaluation_data=evaluation_data,
            export_format=export_format,
            export_options=export_options
        )
        
        return {
            "status": "success",
            "message": "Export completed successfully",
            "data": {
                "export_id": export_result["export_id"],
                "filename": export_result["filename"],
                "file_size_bytes": export_result["file_size_bytes"],
                "format": export_result["format"],
                "download_url": f"/api/v1/data-export/download/{export_result['export_id']}",
                "expires_at": export_result["expires_at"],
                "created_at": export_result["created_at"]
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@router.post("/model-performance-report")
async def export_model_performance_report(
    model_id: str = Form(...),
    export_format: str = Form(..., regex="^(pdf|excel|csv)$"),
    time_range_days: int = Form(30, ge=1, le=365),
    include_comparisons: bool = Form(True),
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Export comprehensive model performance report
    
    Args:
        model_id: ID of the model to export
        export_format: Export format (pdf, excel, csv)
        time_range_days: Time range for performance analysis
        include_comparisons: Include model comparisons
        current_user: Authenticated user
        
    Returns:
        Export job details with download information
    """
    try:
        # Simulate fetching model data
        model_data = {
            "model_id": model_id,
            "name": "Fraud Detection System",
            "type": "Binary Classification",
            "version": "2.1.0",
            "created_at": "2024-01-10T14:20:00Z",
            "owner": current_user.email
        }
        
        performance_metrics = {
            "current_accuracy": 0.953,
            "current_precision": 0.934,
            "current_recall": 0.912,
            "current_f1_score": 0.923,
            "avg_accuracy_30d": 0.948,
            "prediction_volume_30d": 45670,
            "avg_response_time_ms": 120,
            "error_rate_percentage": 0.02,
            "drift_score": 0.08,
            "fairness_score": 0.92
        }
        
        export_result = await export_service.export_model_performance_report(
            model_data=model_data,
            performance_metrics=performance_metrics,
            export_format=export_format,
            export_options={
                "time_range_days": time_range_days,
                "include_comparisons": include_comparisons,
                "user_id": str(current_user.id)
            }
        )
        
        return {
            "status": "success",
            "message": "Model performance report exported successfully",
            "data": {
                "export_id": export_result["export_id"],
                "filename": export_result["filename"],
                "file_size_bytes": export_result["file_size_bytes"],
                "format": export_result["format"],
                "download_url": f"/api/v1/data-export/download/{export_result['export_id']}",
                "expires_at": export_result["expires_at"],
                "model_id": model_id,
                "time_range_days": time_range_days
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model performance export failed: {str(e)}")


@router.post("/fairness-analysis-report")
async def export_fairness_analysis_report(
    analysis_id: str = Form(...),
    export_format: str = Form(..., regex="^(pdf|excel|csv)$"),
    include_mitigation_suggestions: bool = Form(True),
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Export fairness analysis report
    
    Args:
        analysis_id: ID of the fairness analysis to export
        export_format: Export format (pdf, excel, csv)
        include_mitigation_suggestions: Include bias mitigation recommendations
        current_user: Authenticated user
        
    Returns:
        Export job details with download information
    """
    try:
        fairness_data = {
            "analysis_id": analysis_id,
            "overall_score": 0.79,
            "bias_metrics": {
                "demographic_parity": 0.12,
                "equalized_odds": 0.08,
                "calibration": 0.15,
                "gender_bias": 0.09,
                "age_bias": 0.14,
                "ethnicity_bias": 0.07
            },
            "protected_attributes": ["gender", "age", "ethnicity"],
            "violations": [
                {
                    "attribute": "age",
                    "metric": "demographic_parity",
                    "score": 0.14,
                    "severity": "moderate"
                }
            ],
            "mitigation_suggestions": [
                "Apply demographic parity constraints during training",
                "Use reweighting techniques to balance protected groups",
                "Implement post-processing fairness corrections",
                "Regular monitoring of fairness metrics in production"
            ] if include_mitigation_suggestions else []
        }
        
        export_result = await export_service.export_fairness_analysis_report(
            fairness_data=fairness_data,
            export_format=export_format,
            export_options={
                "include_mitigation": include_mitigation_suggestions,
                "user_id": str(current_user.id)
            }
        )
        
        return {
            "status": "success",
            "message": "Fairness analysis report exported successfully",
            "data": {
                "export_id": export_result["export_id"],
                "filename": export_result["filename"],
                "file_size_bytes": export_result["file_size_bytes"],
                "format": export_result["format"],
                "download_url": f"/api/v1/data-export/download/{export_result['export_id']}",
                "expires_at": export_result["expires_at"],
                "analysis_id": analysis_id
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fairness analysis export failed: {str(e)}")


@router.post("/data-drift-report")
async def export_data_drift_report(
    drift_analysis_id: str = Form(...),
    export_format: str = Form(..., regex="^(pdf|excel|csv)$"),
    include_feature_analysis: bool = Form(True),
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Export data drift analysis report
    
    Args:
        drift_analysis_id: ID of the drift analysis to export
        export_format: Export format (pdf, excel, csv)
        include_feature_analysis: Include per-feature drift analysis
        current_user: Authenticated user
        
    Returns:
        Export job details with download information
    """
    try:
        drift_data = {
            "analysis_id": drift_analysis_id,
            "overall_drift_score": 0.28,
            "drift_detected": True,
            "reference_period": "2024-01-01 to 2024-01-15",
            "current_period": "2024-01-16 to 2024-01-30",
            "feature_drift": {
                "age": {"score": 0.15, "status": "stable"},
                "income": {"score": 0.42, "status": "drift_detected"},
                "credit_score": {"score": 0.08, "status": "stable"},
                "employment_length": {"score": 0.31, "status": "moderate_drift"}
            },
            "statistical_tests": {
                "kolmogorov_smirnov": {"p_value": 0.003, "significant": True},
                "chi_square": {"p_value": 0.012, "significant": True}
            },
            "impact_assessment": {
                "prediction_accuracy_change": -0.03,
                "model_reliability": "decreased",
                "recommended_action": "retrain_model"
            }
        }
        
        export_result = await export_service.export_data_drift_report(
            drift_data=drift_data,
            export_format=export_format,
            export_options={
                "include_feature_analysis": include_feature_analysis,
                "user_id": str(current_user.id)
            }
        )
        
        return {
            "status": "success",
            "message": "Data drift report exported successfully",
            "data": {
                "export_id": export_result["export_id"],
                "filename": export_result["filename"],
                "file_size_bytes": export_result["file_size_bytes"],
                "format": export_result["format"],
                "download_url": f"/api/v1/data-export/download/{export_result['export_id']}",
                "expires_at": export_result["expires_at"],
                "analysis_id": drift_analysis_id
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data drift export failed: {str(e)}")


@router.get("/download/{export_id}")
async def download_export_file(
    export_id: str = Path(...),
    current_user: User = Depends(get_current_active_user)
):
    """
    Download exported file
    
    Args:
        export_id: Export job ID
        current_user: Authenticated user
        
    Returns:
        File download response
    """
    try:
        # In real implementation, verify user has access to this export
        # and get file path from database
        
        # For demo, we'll check if file exists in export directory
        export_path = export_service.base_export_path
        
        # Find file with matching export_id in filename
        matching_files = [f for f in export_path.iterdir() if export_id in f.name or f.stem.endswith(export_id[-8:])]
        
        if not matching_files:
            raise HTTPException(status_code=404, detail="Export file not found or expired")
        
        file_path = matching_files[0]
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Export file not found")
        
        # Determine media type based on file extension
        media_types = {
            '.pdf': 'application/pdf',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.csv': 'text/csv'
        }
        
        media_type = media_types.get(file_path.suffix, 'application/octet-stream')
        
        return FileResponse(
            path=str(file_path),
            media_type=media_type,
            filename=file_path.name,
            headers={
                "Content-Disposition": f"attachment; filename={file_path.name}",
                "Cache-Control": "no-cache"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


@router.get("/status/{export_id}")
async def get_export_status(
    export_id: str = Path(...),
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get export job status
    
    Args:
        export_id: Export job ID
        current_user: Authenticated user
        
    Returns:
        Export job status and progress
    """
    try:
        status = await export_service.get_export_status(export_id)
        
        return {
            "status": "success",
            "data": status
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get export status: {str(e)}")


@router.get("/formats")
async def get_supported_formats(
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get supported export formats and their capabilities
    
    Returns:
        Available export formats and their features
    """
    return {
        "status": "success",
        "data": {
            "supported_formats": [
                {
                    "format": "pdf",
                    "name": "PDF Report",
                    "description": "Professional formatted report with charts and tables",
                    "features": ["formatted_layout", "charts", "tables", "branding"],
                    "file_extension": ".pdf",
                    "media_type": "application/pdf"
                },
                {
                    "format": "excel",
                    "name": "Excel Workbook", 
                    "description": "Multi-sheet workbook with data and analysis",
                    "features": ["multiple_sheets", "charts", "formulas", "formatting"],
                    "file_extension": ".xlsx",
                    "media_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                },
                {
                    "format": "csv",
                    "name": "CSV Data",
                    "description": "Raw data in comma-separated values format",
                    "features": ["raw_data", "lightweight", "universal_compatibility"],
                    "file_extension": ".csv",
                    "media_type": "text/csv"
                }
            ],
            "export_types": [
                {
                    "type": "evaluation_report",
                    "name": "Evaluation Report",
                    "description": "Comprehensive model evaluation results"
                },
                {
                    "type": "model_performance",
                    "name": "Model Performance Report",
                    "description": "Detailed model performance analysis over time"
                },
                {
                    "type": "fairness_analysis",
                    "name": "Fairness Analysis Report", 
                    "description": "Bias detection and fairness metrics analysis"
                },
                {
                    "type": "data_drift",
                    "name": "Data Drift Report",
                    "description": "Data distribution changes and drift analysis"
                }
            ]
        }
    }


@router.delete("/cleanup")
async def cleanup_expired_exports(
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Clean up expired export files (Admin only)
    
    Returns:
        Cleanup operation results
    """
    try:
        # Check if user has admin permissions
        # In real implementation, this would check user roles
        
        await export_service.cleanup_expired_exports()
        
        return {
            "status": "success",
            "message": "Expired exports cleaned up successfully",
            "timestamp": "2024-01-30T15:30:00Z"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")


@router.get("/usage-stats")
async def get_export_usage_stats(
    time_range_days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get export usage statistics for the organization
    
    Args:
        time_range_days: Time range for statistics
        current_user: Authenticated user
        
    Returns:
        Export usage statistics and trends
    """
    try:
        # Simulate usage statistics
        # In real implementation, this would query the database
        
        stats = {
            "total_exports": 147,
            "exports_by_format": {
                "pdf": 89,
                "excel": 45,
                "csv": 13
            },
            "exports_by_type": {
                "evaluation_report": 67,
                "model_performance": 34,
                "fairness_analysis": 28,
                "data_drift": 18
            },
            "total_file_size_gb": 2.4,
            "avg_export_time_seconds": 8.5,
            "most_active_users": [
                {"user_email": current_user.email, "export_count": 23},
                {"user_email": "analyst@company.com", "export_count": 18},
                {"user_email": "manager@company.com", "export_count": 15}
            ],
            "daily_trends": [
                {"date": "2024-01-23", "count": 8},
                {"date": "2024-01-24", "count": 12},
                {"date": "2024-01-25", "count": 6},
                {"date": "2024-01-26", "count": 15},
                {"date": "2024-01-27", "count": 9}
            ]
        }
        
        return {
            "status": "success",
            "data": {
                **stats,
                "time_range_days": time_range_days,
                "organization_id": str(current_user.organization_id),
                "generated_at": "2024-01-30T15:30:00Z"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get usage stats: {str(e)}")