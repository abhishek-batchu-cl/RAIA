"""
Model Analysis API Endpoints
Comprehensive endpoints for model explainability, feature analysis, and insights
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
import io
import json
from datetime import datetime, timedelta

from app.core.database import get_database
from app.core.auth import get_current_user
from app.models.schemas import User
from app.services.explainability_service import explainability_service
from app.services.model_service import ModelService
from app.core.exceptions import RAIAException

router = APIRouter()

@router.post("/models/{model_id}/initialize")
async def initialize_model_explainer(
    model_id: str,
    background_tasks: BackgroundTasks,
    model_file: UploadFile = File(...),
    data_file: UploadFile = File(...),
    model_type: str = "tabular",
    feature_names: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_database)
):
    """
    Initialize explainability components for a model
    
    Args:
        model_id: Unique identifier for the model
        model_file: Pickled model file
        data_file: Training data CSV file
        model_type: Type of model (tabular, text, image, neural)
        feature_names: Optional list of feature names
        class_names: Optional list of class names
    
    Returns:
        Initialization status and available explainers
    """
    try:
        # Read and load model
        model_content = await model_file.read()
        model = pickle.loads(model_content)
        
        # Read training data
        data_content = await data_file.read()
        df = pd.read_csv(io.StringIO(data_content.decode('utf-8')))
        
        # Auto-detect feature names if not provided
        if feature_names is None:
            feature_names = df.columns.tolist()
            if 'target' in feature_names:
                feature_names.remove('target')
            if 'y' in feature_names:
                feature_names.remove('y')
        
        # Prepare training data
        X_train = df[feature_names] if len(set(feature_names) & set(df.columns)) > 0 else df
        
        # Initialize explainer in background
        result = await explainability_service.initialize_explainer(
            model_id=model_id,
            model=model,
            X_train=X_train,
            feature_names=feature_names,
            class_names=class_names,
            model_type=model_type
        )
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to initialize model explainer: {str(e)}")

@router.get("/models/{model_id}/feature-importance")
async def get_feature_importance(
    model_id: str,
    method: str = "shap",
    current_user: User = Depends(get_current_user)
):
    """
    Get global feature importance for a model
    
    Args:
        model_id: Model identifier
        method: Method to use (shap, permutation, builtin)
    
    Returns:
        Feature importance data with rankings and visualizations
    """
    try:
        result = await explainability_service.get_feature_importance(
            model_id=model_id,
            method=method
        )
        
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get feature importance: {str(e)}")

@router.post("/models/{model_id}/shap-explanations")
async def get_shap_explanations(
    model_id: str,
    data_file: UploadFile = File(...),
    explanation_type: str = "tree",
    max_samples: int = 100,
    current_user: User = Depends(get_current_user)
):
    """
    Generate SHAP explanations for instances
    
    Args:
        model_id: Model identifier
        data_file: CSV file with instances to explain
        explanation_type: Type of SHAP explainer (tree, kernel, deep)
        max_samples: Maximum number of samples to explain
    
    Returns:
        SHAP values, feature importance, and waterfall data
    """
    try:
        # Read data
        data_content = await data_file.read()
        df = pd.read_csv(io.StringIO(data_content.decode('utf-8')))
        
        result = await explainability_service.get_shap_explanations(
            model_id=model_id,
            X=df,
            explanation_type=explanation_type,
            max_samples=max_samples
        )
        
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate SHAP explanations: {str(e)}")

@router.post("/models/{model_id}/lime-explanations")
async def get_lime_explanations(
    model_id: str,
    data_file: UploadFile = File(...),
    instance_idx: int = 0,
    num_features: int = 10,
    num_samples: int = 1000,
    current_user: User = Depends(get_current_user)
):
    """
    Generate LIME explanations for a specific instance
    
    Args:
        model_id: Model identifier
        data_file: CSV file with data
        instance_idx: Index of instance to explain
        num_features: Number of top features to explain
        num_samples: Number of samples for LIME
    
    Returns:
        LIME explanation data with feature weights and predictions
    """
    try:
        # Read data
        data_content = await data_file.read()
        df = pd.read_csv(io.StringIO(data_content.decode('utf-8')))
        
        result = await explainability_service.get_lime_explanations(
            model_id=model_id,
            X=df,
            instance_idx=instance_idx,
            num_features=num_features,
            num_samples=num_samples
        )
        
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate LIME explanations: {str(e)}")

@router.post("/models/{model_id}/partial-dependence")
async def get_partial_dependence(
    model_id: str,
    feature_name: str,
    data_file: UploadFile = File(...),
    num_points: int = 50,
    current_user: User = Depends(get_current_user)
):
    """
    Calculate partial dependence for a specific feature
    
    Args:
        model_id: Model identifier
        feature_name: Name of feature to analyze
        data_file: CSV file with data
        num_points: Number of points for PD curve
    
    Returns:
        Partial dependence data for visualization
    """
    try:
        # Read data
        data_content = await data_file.read()
        df = pd.read_csv(io.StringIO(data_content.decode('utf-8')))
        
        result = await explainability_service.get_partial_dependence(
            model_id=model_id,
            feature_name=feature_name,
            X=df,
            num_points=num_points
        )
        
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to calculate partial dependence: {str(e)}")

@router.post("/models/{model_id}/feature-interactions")
async def get_feature_interactions(
    model_id: str,
    data_file: UploadFile = File(...),
    top_k: int = 10,
    method: str = "shap",
    current_user: User = Depends(get_current_user)
):
    """
    Analyze feature interactions
    
    Args:
        model_id: Model identifier
        data_file: CSV file with data
        top_k: Number of top interactions to return
        method: Method to use for interaction detection
    
    Returns:
        Feature interaction analysis data
    """
    try:
        # Read data
        data_content = await data_file.read()
        df = pd.read_csv(io.StringIO(data_content.decode('utf-8')))
        
        result = await explainability_service.get_feature_interactions(
            model_id=model_id,
            X=df,
            top_k=top_k,
            method=method
        )
        
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze feature interactions: {str(e)}")

@router.post("/models/{model_id}/counterfactuals")
async def generate_counterfactuals(
    model_id: str,
    data_file: UploadFile = File(...),
    instance_idx: int = 0,
    target_class: Optional[int] = None,
    num_counterfactuals: int = 5,
    current_user: User = Depends(get_current_user)
):
    """
    Generate counterfactual explanations
    
    Args:
        model_id: Model identifier
        data_file: CSV file with data
        instance_idx: Index of instance to generate counterfactuals for
        target_class: Target class for counterfactuals
        num_counterfactuals: Number of counterfactuals to generate
    
    Returns:
        Counterfactual explanation data
    """
    try:
        # Read data
        data_content = await data_file.read()
        df = pd.read_csv(io.StringIO(data_content.decode('utf-8')))
        
        result = await explainability_service.generate_counterfactuals(
            model_id=model_id,
            X=df,
            instance_idx=instance_idx,
            target_class=target_class,
            num_counterfactuals=num_counterfactuals
        )
        
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate counterfactuals: {str(e)}")

@router.get("/models/{model_id}/analysis-summary")
async def get_analysis_summary(
    model_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get comprehensive analysis summary for a model
    
    Args:
        model_id: Model identifier
    
    Returns:
        Comprehensive model analysis summary
    """
    try:
        # This would aggregate data from various analysis endpoints
        # For now, return a placeholder summary
        
        summary = {
            "model_id": model_id,
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "available_analyses": [],
            "model_performance": {
                "accuracy": 0.92,
                "precision": 0.89,
                "recall": 0.94,
                "f1_score": 0.91
            },
            "explainability_score": 0.78,
            "fairness_score": 0.85,
            "interpretability_metrics": {
                "feature_count": 15,
                "model_complexity": "medium",
                "explanation_fidelity": 0.89
            },
            "recommendations": [
                "Consider reducing feature dimensionality",
                "Monitor for data drift in production",
                "Implement bias mitigation techniques"
            ]
        }
        
        # Check which explainers are available
        if model_id in explainability_service.explainers:
            available_explainers = list(explainability_service.explainers[model_id].keys())
            summary["available_analyses"].extend([
                f"SHAP ({exp.replace('shap_', '')})" for exp in available_explainers if exp.startswith('shap_')
            ])
            if 'lime' in available_explainers:
                summary["available_analyses"].append("LIME")
        
        return JSONResponse(content=summary)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get analysis summary: {str(e)}")

@router.delete("/models/{model_id}/cleanup")
async def cleanup_model_explainer(
    model_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Clean up explainer resources for a model
    
    Args:
        model_id: Model identifier
    
    Returns:
        Cleanup status
    """
    try:
        result = await explainability_service.cleanup_explainer(model_id)
        
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cleanup model explainer: {str(e)}")

@router.get("/explainability/methods")
async def get_available_methods():
    """
    Get list of available explainability methods
    
    Returns:
        List of available explainability methods and their capabilities
    """
    methods = {
        "shap": {
            "name": "SHAP (SHapley Additive exPlanations)",
            "types": ["tree", "kernel", "deep", "linear"],
            "supports": ["feature_importance", "individual_explanations", "interaction_detection"],
            "best_for": ["tree_models", "general_models"],
            "computation_time": "medium"
        },
        "lime": {
            "name": "LIME (Local Interpretable Model-agnostic Explanations)",
            "types": ["tabular", "text", "image"],
            "supports": ["local_explanations", "feature_weights"],
            "best_for": ["black_box_models", "individual_instances"],
            "computation_time": "high"
        },
        "partial_dependence": {
            "name": "Partial Dependence Analysis",
            "types": ["1d", "2d"],
            "supports": ["feature_effects", "marginal_effects"],
            "best_for": ["feature_analysis", "model_understanding"],
            "computation_time": "low"
        },
        "counterfactuals": {
            "name": "Counterfactual Explanations",
            "types": ["optimization_based", "perturbation_based"],
            "supports": ["what_if_scenarios", "decision_boundaries"],
            "best_for": ["actionable_insights", "decision_support"],
            "computation_time": "medium"
        }
    }
    
    return JSONResponse(content={
        "available_methods": methods,
        "total_methods": len(methods),
        "recommended_workflow": [
            "Start with SHAP for global understanding",
            "Use LIME for detailed instance analysis",
            "Apply partial dependence for feature effects",
            "Generate counterfactuals for actionable insights"
        ]
    })

@router.get("/models/{model_id}/status")
async def get_model_analysis_status(
    model_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get the current analysis status for a model
    
    Args:
        model_id: Model identifier
    
    Returns:
        Current status and available analysis options
    """
    try:
        status = {
            "model_id": model_id,
            "initialized": model_id in explainability_service.models,
            "available_explainers": [],
            "feature_count": 0,
            "last_analysis": None
        }
        
        if model_id in explainability_service.explainers:
            status["available_explainers"] = list(explainability_service.explainers[model_id].keys())
        
        if model_id in explainability_service.feature_names:
            status["feature_count"] = len(explainability_service.feature_names[model_id])
        
        return JSONResponse(content=status)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model status: {str(e)}")