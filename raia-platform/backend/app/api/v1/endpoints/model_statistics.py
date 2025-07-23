"""
Model Statistics API Endpoints
Comprehensive classification and regression statistics with detailed analysis
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from typing import Dict, List, Optional, Any
import json
import pandas as pd
from io import StringIO
import pickle
import numpy as np

from app.core.auth import get_current_active_user
from app.models.schemas import User
from app.services.model_statistics_service import model_statistics_service

router = APIRouter()

@router.post("/register")
async def register_model(
    model_id: str = Form(...),
    model_data: UploadFile = File(...),  # Pickled model file
    model_type: str = Form(...),  # "classification" or "regression"
    feature_names: Optional[str] = Form(None),  # JSON string of feature names
    class_names: Optional[str] = Form(None),  # JSON string of class names
    current_user: User = Depends(get_current_active_user)
):
    """Register a model for statistical analysis"""
    try:
        # Load model
        model_content = await model_data.read()
        model = pickle.loads(model_content)
        
        # Parse parameters
        feature_names_list = json.loads(feature_names) if feature_names else None
        class_names_list = json.loads(class_names) if class_names else None
        
        result = await model_statistics_service.register_model(
            model_id=model_id,
            model=model,
            model_type=model_type,
            feature_names=feature_names_list,
            class_names=class_names_list
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/classification/calculate")
async def calculate_classification_statistics(
    model_id: str = Form(...),
    test_features: UploadFile = File(...),  # CSV file with test features
    test_labels: UploadFile = File(...),  # CSV file with test labels
    train_features: Optional[UploadFile] = File(None),  # Optional training features
    train_labels: Optional[UploadFile] = File(None),  # Optional training labels
    sample_weights: Optional[UploadFile] = File(None),  # Optional sample weights
    current_user: User = Depends(get_current_active_user)
):
    """Calculate comprehensive classification statistics"""
    try:
        # Load test data
        X_test_content = await test_features.read()
        X_test = pd.read_csv(StringIO(X_test_content.decode('utf-8')))
        
        y_test_content = await test_labels.read()
        y_test = pd.read_csv(StringIO(y_test_content.decode('utf-8'))).iloc[:, 0]
        
        # Load optional training data
        X_train = None
        y_train = None
        if train_features and train_labels:
            X_train_content = await train_features.read()
            X_train = pd.read_csv(StringIO(X_train_content.decode('utf-8')))
            
            y_train_content = await train_labels.read()
            y_train = pd.read_csv(StringIO(y_train_content.decode('utf-8'))).iloc[:, 0]
        
        # Load optional sample weights
        sample_weight = None
        if sample_weights:
            weights_content = await sample_weights.read()
            sample_weight = pd.read_csv(StringIO(weights_content.decode('utf-8'))).iloc[:, 0].values
        
        result = await model_statistics_service.calculate_classification_statistics(
            model_id=model_id,
            X_test=X_test,
            y_test=y_test,
            X_train=X_train,
            y_train=y_train,
            sample_weight=sample_weight
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/regression/calculate")
async def calculate_regression_statistics(
    model_id: str = Form(...),
    test_features: UploadFile = File(...),  # CSV file with test features
    test_targets: UploadFile = File(...),  # CSV file with test targets
    train_features: Optional[UploadFile] = File(None),  # Optional training features
    train_targets: Optional[UploadFile] = File(None),  # Optional training targets
    sample_weights: Optional[UploadFile] = File(None),  # Optional sample weights
    current_user: User = Depends(get_current_active_user)
):
    """Calculate comprehensive regression statistics"""
    try:
        # Load test data
        X_test_content = await test_features.read()
        X_test = pd.read_csv(StringIO(X_test_content.decode('utf-8')))
        
        y_test_content = await test_targets.read()
        y_test = pd.read_csv(StringIO(y_test_content.decode('utf-8'))).iloc[:, 0]
        
        # Load optional training data
        X_train = None
        y_train = None
        if train_features and train_targets:
            X_train_content = await train_features.read()
            X_train = pd.read_csv(StringIO(X_train_content.decode('utf-8')))
            
            y_train_content = await train_targets.read()
            y_train = pd.read_csv(StringIO(y_train_content.decode('utf-8'))).iloc[:, 0]
        
        # Load optional sample weights
        sample_weight = None
        if sample_weights:
            weights_content = await sample_weights.read()
            sample_weight = pd.read_csv(StringIO(weights_content.decode('utf-8'))).iloc[:, 0].values
        
        result = await model_statistics_service.calculate_regression_statistics(
            model_id=model_id,
            X_test=X_test,
            y_test=y_test,
            X_train=X_train,
            y_train=y_train,
            sample_weight=sample_weight
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/compare")
async def compare_models(
    model_comparisons: str = Form(...),  # JSON string of model comparison data
    comparison_metric: str = Form("auto"),
    current_user: User = Depends(get_current_active_user)
):
    """Compare multiple models statistically"""
    try:
        comparisons_data = json.loads(model_comparisons)
        
        result = await model_statistics_service.compare_models(
            model_comparisons=comparisons_data,
            comparison_metric=comparison_metric
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/models/{model_id}/info")
async def get_model_info(
    model_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get information about a registered model"""
    try:
        if model_id not in model_statistics_service.models:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        model_info = model_statistics_service.models[model_id]
        return {
            "status": "success",
            "model_id": model_id,
            "model_type": model_info["model_type"],
            "feature_names": model_info["feature_names"],
            "class_names": model_info["class_names"],
            "features_count": len(model_info["feature_names"]),
            "registered_at": model_info["registered_at"]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/cache/{cache_key}")
async def get_cached_statistics(
    cache_key: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get cached statistics"""
    try:
        result = await model_statistics_service.get_cached_statistics(cache_key)
        if result is None:
            raise HTTPException(status_code=404, detail="Cache entry not found")
        
        return {
            "status": "success",
            "cache_key": cache_key,
            "statistics": result.__dict__ if hasattr(result, '__dict__') else result
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/cache")
async def clear_cache(
    model_id: Optional[str] = None,
    current_user: User = Depends(get_current_active_user)
):
    """Clear statistics cache"""
    try:
        result = await model_statistics_service.clear_cache(model_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/models")
async def list_models(
    current_user: User = Depends(get_current_active_user)
):
    """List all registered models"""
    try:
        models = list(model_statistics_service.models.keys())
        model_info = []
        
        for model_id in models:
            info = model_statistics_service.models[model_id]
            model_info.append({
                "model_id": model_id,
                "model_type": info["model_type"],
                "features_count": len(info["feature_names"]),
                "classes_count": len(info["class_names"]),
                "registered_at": info["registered_at"]
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
    """Clean up model resources"""
    try:
        result = await model_statistics_service.cleanup_model(model_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/metrics/classification")
async def get_classification_metrics_info(
    current_user: User = Depends(get_current_active_user)
):
    """Get information about available classification metrics"""
    return {
        "status": "success",
        "basic_metrics": [
            "accuracy", "balanced_accuracy", "precision", "recall", "f1_score"
        ],
        "advanced_metrics": [
            "matthews_corrcoef", "cohen_kappa", "log_loss", "roc_auc"
        ],
        "averaging_methods": [
            "macro", "micro", "weighted"
        ],
        "probability_metrics": [
            "roc_auc", "log_loss", "calibration_error"
        ],
        "statistical_tests": [
            "accuracy_vs_random", "prediction_independence", "mcnemar_test"
        ]
    }

@router.get("/metrics/regression")
async def get_regression_metrics_info(
    current_user: User = Depends(get_current_active_user)
):
    """Get information about available regression metrics"""
    return {
        "status": "success",
        "basic_metrics": [
            "mae", "mse", "rmse", "r2", "explained_variance"
        ],
        "advanced_metrics": [
            "adjusted_r2", "mape", "msle", "mean_bias_error", "mean_relative_error"
        ],
        "residual_analysis": [
            "normality_tests", "heteroscedasticity_tests", "residual_distribution"
        ],
        "normality_tests": [
            "shapiro_wilk", "dagostino_pearson", "kolmogorov_smirnov"
        ],
        "heteroscedasticity_tests": [
            "breusch_pagan", "variance_ratio"
        ]
    }

@router.post("/batch-analysis")
async def batch_statistical_analysis(
    models_data: UploadFile = File(...),  # CSV file with model information
    test_data: UploadFile = File(...),  # CSV file with test data
    analysis_type: str = Form("classification"),
    current_user: User = Depends(get_current_active_user)
):
    """Perform batch statistical analysis on multiple models"""
    try:
        # Load models information
        models_content = await models_data.read()
        models_df = pd.read_csv(StringIO(models_content.decode('utf-8')))
        
        # Load test data
        test_content = await test_data.read()
        test_df = pd.read_csv(StringIO(test_content.decode('utf-8')))
        
        results = []
        for _, model_row in models_df.iterrows():
            model_id = model_row['model_id']
            
            if model_id in model_statistics_service.models:
                try:
                    if analysis_type == "classification":
                        # Assume test_df has features and a 'target' column
                        X_test = test_df.drop('target', axis=1)
                        y_test = test_df['target']
                        
                        result = await model_statistics_service.calculate_classification_statistics(
                            model_id=model_id,
                            X_test=X_test,
                            y_test=y_test
                        )
                    else:
                        # Regression analysis
                        X_test = test_df.drop('target', axis=1)
                        y_test = test_df['target']
                        
                        result = await model_statistics_service.calculate_regression_statistics(
                            model_id=model_id,
                            X_test=X_test,
                            y_test=y_test
                        )
                    
                    results.append({
                        "model_id": model_id,
                        "analysis_result": result
                    })
                
                except Exception as e:
                    results.append({
                        "model_id": model_id,
                        "error": str(e)
                    })
        
        return {
            "status": "success",
            "analysis_type": analysis_type,
            "results": results,
            "models_analyzed": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))