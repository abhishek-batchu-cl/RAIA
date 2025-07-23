"""
Advanced Explainability API Endpoints
Comprehensive explainability with Alibi, Captum, ELI5, and prototype-based explanations
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from typing import Dict, List, Optional, Any
import json
import pandas as pd
from io import StringIO
import pickle
import base64

from app.core.auth import get_current_active_user
from app.models.schemas import User
from app.services.advanced_explainability_service import advanced_explainability_service

router = APIRouter()

@router.post("/initialize")
async def initialize_explainer(
    model_id: str = Form(...),
    model_data: UploadFile = File(...),  # Pickled model file
    training_data: UploadFile = File(...),  # CSV file with training data
    feature_names: str = Form(...),  # JSON string of feature names
    class_names: Optional[str] = Form(None),  # JSON string of class names
    model_type: str = Form("tabular"),
    categorical_features: Optional[str] = Form(None),  # JSON string of indices
    current_user: User = Depends(get_current_active_user)
):
    """Initialize advanced explainer for a model"""
    try:
        # Load model
        model_content = await model_data.read()
        model = pickle.loads(model_content)
        
        # Load training data
        training_content = await training_data.read()
        X_train = pd.read_csv(StringIO(training_content.decode('utf-8')))
        
        # Parse parameters
        feature_names_list = json.loads(feature_names)
        class_names_list = json.loads(class_names) if class_names else None
        categorical_features_list = json.loads(categorical_features) if categorical_features else None
        
        result = await advanced_explainability_service.initialize_advanced_explainer(
            model_id=model_id,
            model=model,
            X_train=X_train,
            feature_names=feature_names_list,
            class_names=class_names_list,
            model_type=model_type,
            categorical_features=categorical_features_list
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/anchor-explanations")
async def get_anchor_explanations(
    model_id: str = Form(...),
    instance_data: str = Form(...),  # JSON string of instance data
    instance_idx: int = Form(0),
    threshold: float = Form(0.95),
    max_anchor_size: int = Form(5),
    current_user: User = Depends(get_current_active_user)
):
    """Generate Anchor explanations for tabular data"""
    try:
        # Parse instance data
        instance_dict = json.loads(instance_data)
        X = pd.DataFrame([instance_dict] if isinstance(instance_dict, dict) else instance_dict)
        
        result = await advanced_explainability_service.get_anchor_explanations(
            model_id=model_id,
            X=X,
            instance_idx=instance_idx,
            threshold=threshold,
            max_anchor_size=max_anchor_size
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/ale-plots")
async def get_ale_plots(
    model_id: str = Form(...),
    data: UploadFile = File(...),  # CSV file with data
    features: Optional[str] = Form(None),  # JSON string of features to analyze
    num_quantiles: int = Form(50),
    current_user: User = Depends(get_current_active_user)
):
    """Generate Accumulated Local Effects (ALE) plots"""
    try:
        # Load data
        content = await data.read()
        X = pd.read_csv(StringIO(content.decode('utf-8')))
        
        features_list = json.loads(features) if features else None
        
        result = await advanced_explainability_service.get_ale_plots(
            model_id=model_id,
            X=X,
            features=features_list,
            num_quantiles=num_quantiles
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/permutation-importance")
async def get_advanced_permutation_importance(
    model_id: str = Form(...),
    data: UploadFile = File(...),  # CSV file with data
    target_data: Optional[UploadFile] = File(None),  # CSV file with targets
    n_repeats: int = Form(10),
    scoring: str = Form("accuracy"),
    current_user: User = Depends(get_current_active_user)
):
    """Get advanced permutation importance using ELI5"""
    try:
        # Load data
        content = await data.read()
        X = pd.read_csv(StringIO(content.decode('utf-8')))
        
        y = None
        if target_data:
            target_content = await target_data.read()
            y = pd.read_csv(StringIO(target_content.decode('utf-8'))).iloc[:, 0]
        
        result = await advanced_explainability_service.get_advanced_permutation_importance(
            model_id=model_id,
            X=X,
            y=y,
            n_repeats=n_repeats,
            scoring=scoring
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/prototype-explanations")
async def get_prototype_explanations(
    model_id: str = Form(...),
    instance_data: str = Form(...),  # JSON string of instance data
    instance_idx: int = Form(0),
    n_prototypes: int = Form(5),
    distance_metric: str = Form("euclidean"),
    current_user: User = Depends(get_current_active_user)
):
    """Generate prototype-based explanations"""
    try:
        # Parse instance data
        instance_dict = json.loads(instance_data)
        X = pd.DataFrame([instance_dict] if isinstance(instance_dict, dict) else instance_dict)
        
        result = await advanced_explainability_service.get_prototype_explanations(
            model_id=model_id,
            X=X,
            instance_idx=instance_idx,
            n_prototypes=n_prototypes,
            distance_metric=distance_metric
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/ice-plots")
async def get_ice_plots(
    model_id: str = Form(...),
    data: UploadFile = File(...),  # CSV file with data
    feature_name: str = Form(...),
    num_ice_lines: int = Form(50),
    num_points: int = Form(50),
    current_user: User = Depends(get_current_active_user)
):
    """Generate Individual Conditional Expectation (ICE) plots"""
    try:
        # Load data
        content = await data.read()
        X = pd.read_csv(StringIO(content.decode('utf-8')))
        
        result = await advanced_explainability_service.get_ice_plots(
            model_id=model_id,
            X=X,
            feature_name=feature_name,
            num_ice_lines=num_ice_lines,
            num_points=num_points
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/advanced-counterfactuals")
async def get_advanced_counterfactuals(
    model_id: str = Form(...),
    instance_data: str = Form(...),  # JSON string of instance data
    instance_idx: int = Form(0),
    target_class: Optional[int] = Form(None),
    num_counterfactuals: int = Form(5),
    method: str = Form("diverse"),
    current_user: User = Depends(get_current_active_user)
):
    """Generate advanced counterfactual explanations"""
    try:
        # Parse instance data
        instance_dict = json.loads(instance_data)
        X = pd.DataFrame([instance_dict] if isinstance(instance_dict, dict) else instance_dict)
        
        result = await advanced_explainability_service.get_advanced_counterfactuals(
            model_id=model_id,
            X=X,
            instance_idx=instance_idx,
            target_class=target_class,
            num_counterfactuals=num_counterfactuals,
            method=method
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/models/{model_id}/status")
async def get_explainer_status(
    model_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get explainer initialization status for a model"""
    try:
        if model_id in advanced_explainability_service.models:
            explainers = list(advanced_explainability_service.explainers.get(model_id, {}).keys())
            return {
                "status": "success",
                "model_id": model_id,
                "initialized": True,
                "available_explainers": explainers,
                "model_type": advanced_explainability_service.model_types.get(model_id, "unknown")
            }
        else:
            return {
                "status": "success",
                "model_id": model_id,
                "initialized": False,
                "available_explainers": []
            }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/models")
async def list_initialized_models(
    current_user: User = Depends(get_current_active_user)
):
    """List all initialized models for advanced explainability"""
    try:
        models = list(advanced_explainability_service.models.keys())
        model_info = []
        
        for model_id in models:
            explainers = list(advanced_explainability_service.explainers.get(model_id, {}).keys())
            model_type = advanced_explainability_service.model_types.get(model_id, "unknown")
            feature_count = len(advanced_explainability_service.feature_names.get(model_id, []))
            
            model_info.append({
                "model_id": model_id,
                "available_explainers": explainers,
                "model_type": model_type,
                "feature_count": feature_count
            })
        
        return {
            "status": "success",
            "models": model_info,
            "count": len(models)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/models/{model_id}")
async def cleanup_explainer(
    model_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Clean up explainer resources for a model"""
    try:
        result = await advanced_explainability_service.cleanup_advanced_explainer(model_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/methods/available")
async def get_available_methods(
    current_user: User = Depends(get_current_active_user)
):
    """Get list of available advanced explainability methods"""
    return {
        "status": "success",
        "methods": {
            "anchor_explanations": {
                "description": "Rule-based explanations with precision/coverage",
                "supports": ["tabular"],
                "requires": ["alibi"]
            },
            "ale_plots": {
                "description": "Accumulated Local Effects for feature analysis",
                "supports": ["tabular"],
                "requires": ["alibi"]
            },
            "permutation_importance": {
                "description": "Advanced permutation importance with ELI5",
                "supports": ["tabular"],
                "requires": ["eli5"]
            },
            "prototype_explanations": {
                "description": "Training data similarity analysis",
                "supports": ["tabular"],
                "requires": ["sklearn"]
            },
            "ice_plots": {
                "description": "Individual Conditional Expectation plots",
                "supports": ["tabular"],
                "requires": ["sklearn"]
            },
            "advanced_counterfactuals": {
                "description": "Multi-strategy counterfactual generation",
                "supports": ["tabular"],
                "requires": ["sklearn"]
            }
        },
        "counterfactual_methods": [
            "diverse", "minimal_change", "realistic", "sparse", "dense"
        ],
        "distance_metrics": [
            "euclidean", "manhattan", "cosine", "chebyshev"
        ]
    }