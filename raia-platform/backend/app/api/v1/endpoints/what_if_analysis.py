"""
What-If Analysis API Endpoints
Scenario analysis, decision tree extraction, and optimization capabilities
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from typing import Dict, List, Optional, Any
import json
import pandas as pd
from io import StringIO
import pickle

from app.core.auth import get_current_active_user
from app.models.schemas import User
from app.services.what_if_analysis_service import what_if_analysis_service

router = APIRouter()

@router.post("/register")
async def register_model_for_analysis(
    model_id: str = Form(...),
    model_data: UploadFile = File(...),  # Pickled model file
    training_features: UploadFile = File(...),  # CSV file with training features
    training_targets: UploadFile = File(...),  # CSV file with training targets
    feature_names: str = Form(...),  # JSON string of feature names
    class_names: Optional[str] = Form(None),  # JSON string of class names
    model_type: str = Form("classification"),
    current_user: User = Depends(get_current_active_user)
):
    """Register a model for what-if analysis"""
    try:
        # Load model
        model_content = await model_data.read()
        model = pickle.loads(model_content)
        
        # Load training data
        X_train_content = await training_features.read()
        X_train = pd.read_csv(StringIO(X_train_content.decode('utf-8')))
        
        y_train_content = await training_targets.read()
        y_train = pd.read_csv(StringIO(y_train_content.decode('utf-8'))).iloc[:, 0]
        
        # Parse parameters
        feature_names_list = json.loads(feature_names)
        class_names_list = json.loads(class_names) if class_names else None
        
        result = await what_if_analysis_service.register_model_for_analysis(
            model_id=model_id,
            model=model,
            X_train=X_train,
            y_train=y_train,
            feature_names=feature_names_list,
            class_names=class_names_list,
            model_type=model_type
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/what-if")
async def perform_what_if_analysis(
    model_id: str = Form(...),
    base_instance: str = Form(...),  # JSON string of base instance
    what_if_scenarios: str = Form(...),  # JSON string of scenarios
    include_confidence: bool = Form(True),
    include_feature_importance: bool = Form(True),
    current_user: User = Depends(get_current_active_user)
):
    """Perform what-if analysis on multiple scenarios"""
    try:
        # Parse JSON data
        base_instance_dict = json.loads(base_instance)
        scenarios_list = json.loads(what_if_scenarios)
        
        result = await what_if_analysis_service.perform_what_if_analysis(
            model_id=model_id,
            base_instance=base_instance_dict,
            what_if_scenarios=scenarios_list,
            include_confidence=include_confidence,
            include_feature_importance=include_feature_importance
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/decision-rules")
async def extract_decision_rules(
    model_id: str = Form(...),
    max_rules: int = Form(20),
    use_surrogate: bool = Form(True),
    min_support: float = Form(0.01),
    current_user: User = Depends(get_current_active_user)
):
    """Extract decision rules from the model"""
    try:
        result = await what_if_analysis_service.extract_decision_rules(
            model_id=model_id,
            max_rules=max_rules,
            use_surrogate=use_surrogate,
            min_support=min_support
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/optimization")
async def perform_optimization_analysis(
    model_id: str = Form(...),
    base_instance: str = Form(...),  # JSON string of base instance
    objective: str = Form("maximize"),  # maximize, minimize, or target
    target_feature: Optional[str] = Form(None),
    constraints: Optional[str] = Form(None),  # JSON string of constraints
    optimization_steps: int = Form(100),
    current_user: User = Depends(get_current_active_user)
):
    """Perform optimization analysis to find optimal feature values"""
    try:
        # Parse JSON data
        base_instance_dict = json.loads(base_instance)
        constraints_dict = json.loads(constraints) if constraints else None
        
        result = await what_if_analysis_service.perform_optimization_analysis(
            model_id=model_id,
            base_instance=base_instance_dict,
            objective=objective,
            target_feature=target_feature,
            constraints=constraints_dict,
            optimization_steps=optimization_steps
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/models/{model_id}/surrogate-info")
async def get_surrogate_info(
    model_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get information about the surrogate decision tree"""
    try:
        if model_id not in what_if_analysis_service.surrogate_trees:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found or no surrogate tree available")
        
        surrogate_info = what_if_analysis_service.surrogate_trees[model_id]
        if not surrogate_info:
            return {
                "status": "success",
                "model_id": model_id,
                "surrogate_available": False,
                "message": "Surrogate tree creation failed"
            }
        
        return {
            "status": "success",
            "model_id": model_id,
            "surrogate_available": True,
            "accuracy": surrogate_info["accuracy"],
            "depth": surrogate_info["depth"],
            "n_leaves": surrogate_info["n_leaves"],
            "n_nodes": surrogate_info["n_nodes"]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/batch-scenarios")
async def batch_what_if_analysis(
    model_id: str = Form(...),
    scenarios_data: UploadFile = File(...),  # CSV file with scenarios
    base_instance: str = Form(...),  # JSON string of base instance
    current_user: User = Depends(get_current_active_user)
):
    """Perform batch what-if analysis from uploaded scenarios file"""
    try:
        # Load scenarios data
        content = await scenarios_data.read()
        scenarios_df = pd.read_csv(StringIO(content.decode('utf-8')))
        scenarios_list = scenarios_df.to_dict('records')
        
        # Parse base instance
        base_instance_dict = json.loads(base_instance)
        
        result = await what_if_analysis_service.perform_what_if_analysis(
            model_id=model_id,
            base_instance=base_instance_dict,
            what_if_scenarios=scenarios_list,
            include_confidence=True,
            include_feature_importance=True
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/models/{model_id}/feature-ranges")
async def get_feature_ranges(
    model_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get feature value ranges from training data for constraint setting"""
    try:
        if model_id not in what_if_analysis_service.training_data:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        training_data = what_if_analysis_service.training_data[model_id]
        X_train = training_data['X']
        feature_names = what_if_analysis_service.feature_names[model_id]
        
        feature_ranges = {}
        for feature in feature_names:
            if feature in X_train.columns:
                col_data = X_train[feature]
                if col_data.dtype in ['int64', 'float64']:
                    feature_ranges[feature] = {
                        "type": "numerical",
                        "min": float(col_data.min()),
                        "max": float(col_data.max()),
                        "mean": float(col_data.mean()),
                        "std": float(col_data.std())
                    }
                else:
                    unique_values = col_data.unique().tolist()
                    feature_ranges[feature] = {
                        "type": "categorical",
                        "values": unique_values,
                        "count": len(unique_values)
                    }
        
        return {
            "status": "success",
            "model_id": model_id,
            "feature_ranges": feature_ranges
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/models")
async def list_registered_models(
    current_user: User = Depends(get_current_active_user)
):
    """List all registered models for what-if analysis"""
    try:
        models = list(what_if_analysis_service.models.keys())
        model_info = []
        
        for model_id in models:
            training_data = what_if_analysis_service.training_data.get(model_id, {})
            feature_names = what_if_analysis_service.feature_names.get(model_id, [])
            class_names = what_if_analysis_service.class_names.get(model_id, [])
            surrogate_info = what_if_analysis_service.surrogate_trees.get(model_id)
            
            model_info.append({
                "model_id": model_id,
                "model_type": training_data.get('model_type', 'unknown'),
                "feature_count": len(feature_names),
                "class_count": len(class_names),
                "training_samples": len(training_data.get('X', [])),
                "surrogate_available": surrogate_info is not None,
                "surrogate_accuracy": surrogate_info.get('accuracy', 0) if surrogate_info else 0
            })
        
        return {
            "status": "success",
            "models": model_info,
            "count": len(models)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/models/{model_id}")
async def cleanup_model(
    model_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Clean up model resources for what-if analysis"""
    try:
        result = await what_if_analysis_service.cleanup_model(model_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/optimization/objectives")
async def get_optimization_objectives(
    current_user: User = Depends(get_current_active_user)
):
    """Get available optimization objectives"""
    return {
        "status": "success",
        "objectives": {
            "maximize": "Maximize model prediction/probability",
            "minimize": "Minimize model prediction/probability",
            "target": "Target specific prediction value"
        },
        "constraint_types": {
            "min": "Minimum value constraint",
            "max": "Maximum value constraint",
            "fixed": "Fixed value constraint",
            "categorical": "Allowed categorical values"
        }
    }

@router.get("/analysis-types/available")
async def get_available_analysis_types(
    current_user: User = Depends(get_current_active_user)
):
    """Get list of available analysis types"""
    return {
        "status": "success",
        "analysis_types": {
            "what_if": "Scenario-based analysis with feature changes",
            "decision_rules": "Extract interpretable IF-THEN rules",
            "optimization": "Find optimal feature values for objectives",
            "sensitivity": "Analyze feature sensitivity around instances"
        },
        "supported_models": [
            "sklearn classifiers", "sklearn regressors", 
            "tree-based models", "ensemble models"
        ]
    }