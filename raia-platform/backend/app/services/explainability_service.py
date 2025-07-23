"""
Advanced Model Explainability Service
Comprehensive implementation of SHAP, LIME, and other explainability methods
"""

import asyncio
import json
import logging
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import uuid

import numpy as np
import pandas as pd
import shap
import lime
import lime.lime_tabular
import lime.lime_text
import lime.lime_image
from captum.attr import (
    IntegratedGradients, 
    GradientShap, 
    DeepLift, 
    LayerConductance,
    Saliency
)
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from app.core.config import get_settings
from app.core.database import get_database

logger = logging.getLogger(__name__)
settings = get_settings()

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

class ExplainabilityService:
    """
    Comprehensive model explainability service supporting multiple explanation methods
    """
    
    def __init__(self):
        self.explainers = {}
        self.models = {}
        self.preprocessors = {}
        self.feature_names = {}
        self.class_names = {}
        
    async def initialize_explainer(
        self, 
        model_id: str,
        model: BaseEstimator,
        X_train: pd.DataFrame,
        feature_names: List[str],
        class_names: Optional[List[str]] = None,
        model_type: str = "tabular"
    ) -> Dict[str, Any]:
        """
        Initialize explainers for a given model
        
        Args:
            model_id: Unique identifier for the model
            model: Trained model object
            X_train: Training data for background/baseline
            feature_names: List of feature names
            class_names: List of class names for classification
            model_type: Type of model (tabular, text, image, neural)
        
        Returns:
            Dictionary with initialization status and explainer info
        """
        try:
            # Store model and metadata
            self.models[model_id] = model
            self.feature_names[model_id] = feature_names
            self.class_names[model_id] = class_names or []
            
            explainers = {}
            
            # Initialize SHAP explainers
            if hasattr(model, 'predict'):
                try:
                    # Try TreeExplainer first (for tree-based models)
                    if hasattr(model, 'estimators_') or 'forest' in str(type(model)).lower():
                        explainers['shap_tree'] = shap.TreeExplainer(model)
                        logger.info(f"Initialized SHAP TreeExplainer for model {model_id}")
                    
                    # KernelExplainer for general models
                    background_sample = shap.sample(X_train, min(100, len(X_train)))
                    explainers['shap_kernel'] = shap.KernelExplainer(
                        model.predict_proba if hasattr(model, 'predict_proba') else model.predict,
                        background_sample
                    )
                    logger.info(f"Initialized SHAP KernelExplainer for model {model_id}")
                    
                except Exception as e:
                    logger.warning(f"Failed to initialize SHAP explainers: {e}")
            
            # Initialize LIME explainer
            if model_type == "tabular":
                try:
                    explainers['lime'] = lime.lime_tabular.LimeTabularExplainer(
                        X_train.values,
                        feature_names=feature_names,
                        class_names=class_names,
                        mode='classification' if hasattr(model, 'predict_proba') else 'regression',
                        discretize_continuous=True,
                        random_state=42
                    )
                    logger.info(f"Initialized LIME TabularExplainer for model {model_id}")
                except Exception as e:
                    logger.warning(f"Failed to initialize LIME explainer: {e}")
            
            self.explainers[model_id] = explainers
            
            return {
                "status": "success",
                "model_id": model_id,
                "explainers_initialized": list(explainers.keys()),
                "feature_count": len(feature_names),
                "model_type": model_type
            }
            
        except Exception as e:
            logger.error(f"Failed to initialize explainers for model {model_id}: {e}")
            return {
                "status": "error",
                "message": str(e),
                "model_id": model_id
            }
    
    async def get_shap_explanations(
        self,
        model_id: str,
        X: pd.DataFrame,
        explanation_type: str = "tree",
        max_samples: int = 100
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanations for given instances
        
        Args:
            model_id: Model identifier
            X: Input data to explain
            explanation_type: Type of SHAP explainer (tree, kernel, deep)
            max_samples: Maximum number of samples to explain
        
        Returns:
            Dictionary with SHAP values and visualization data
        """
        try:
            if model_id not in self.explainers:
                raise ValueError(f"Model {model_id} not initialized")
            
            # Limit samples for performance
            X_sample = X.head(max_samples) if len(X) > max_samples else X
            
            explainer_key = f"shap_{explanation_type}"
            if explainer_key not in self.explainers[model_id]:
                raise ValueError(f"SHAP {explanation_type} explainer not available for model {model_id}")
            
            explainer = self.explainers[model_id][explainer_key]
            
            # Generate SHAP values
            if explanation_type == "tree":
                shap_values = explainer.shap_values(X_sample)
                expected_value = explainer.expected_value
            else:  # kernel explainer
                shap_values = explainer.shap_values(X_sample, nsamples=50)
                expected_value = explainer.expected_value
            
            # Handle multi-class case
            if isinstance(shap_values, list):
                # Multi-class classification
                shap_data = []
                for class_idx, class_shap in enumerate(shap_values):
                    shap_data.append({
                        "class": self.class_names[model_id][class_idx] if self.class_names[model_id] else f"Class_{class_idx}",
                        "shap_values": class_shap.tolist(),
                        "expected_value": expected_value[class_idx] if isinstance(expected_value, np.ndarray) else expected_value
                    })
            else:
                # Binary classification or regression
                shap_data = [{
                    "class": "prediction",
                    "shap_values": shap_values.tolist(),
                    "expected_value": expected_value
                }]
            
            # Generate feature importance summary
            if isinstance(shap_values, list):
                # Use first class for feature importance
                importance_values = np.abs(shap_values[0]).mean(axis=0)
            else:
                importance_values = np.abs(shap_values).mean(axis=0)
            
            feature_importance = [
                {
                    "feature": feature,
                    "importance": float(importance),
                    "rank": rank + 1
                }
                for rank, (feature, importance) in enumerate(
                    zip(self.feature_names[model_id], importance_values)
                )
            ]
            feature_importance.sort(key=lambda x: x["importance"], reverse=True)
            
            # Generate waterfall data for first instance
            if len(X_sample) > 0:
                instance_idx = 0
                if isinstance(shap_values, list):
                    instance_shap = shap_values[0][instance_idx]
                    base_value = expected_value[0] if isinstance(expected_value, np.ndarray) else expected_value
                else:
                    instance_shap = shap_values[instance_idx]
                    base_value = expected_value
                
                waterfall_data = []
                for feature, value, shap_val in zip(
                    self.feature_names[model_id], 
                    X_sample.iloc[instance_idx].values,
                    instance_shap
                ):
                    waterfall_data.append({
                        "feature": feature,
                        "feature_value": float(value),
                        "shap_value": float(shap_val),
                        "contribution": "positive" if shap_val > 0 else "negative"
                    })
                
                waterfall_data.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
            else:
                waterfall_data = []
            
            return {
                "status": "success",
                "model_id": model_id,
                "explanation_type": explanation_type,
                "samples_explained": len(X_sample),
                "shap_data": shap_data,
                "feature_importance": feature_importance,
                "waterfall_data": waterfall_data,
                "metadata": {
                    "feature_names": self.feature_names[model_id],
                    "class_names": self.class_names[model_id],
                    "explanation_method": "SHAP"
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to generate SHAP explanations: {e}")
            return {
                "status": "error",
                "message": str(e),
                "model_id": model_id
            }
    
    async def get_lime_explanations(
        self,
        model_id: str,
        X: pd.DataFrame,
        instance_idx: int = 0,
        num_features: int = 10,
        num_samples: int = 1000
    ) -> Dict[str, Any]:
        """
        Generate LIME explanations for a specific instance
        
        Args:
            model_id: Model identifier
            X: Input data
            instance_idx: Index of instance to explain
            num_features: Number of top features to explain
            num_samples: Number of samples for LIME
        
        Returns:
            Dictionary with LIME explanation data
        """
        try:
            if model_id not in self.explainers or 'lime' not in self.explainers[model_id]:
                raise ValueError(f"LIME explainer not available for model {model_id}")
            
            explainer = self.explainers[model_id]['lime']
            model = self.models[model_id]
            
            if instance_idx >= len(X):
                raise ValueError(f"Instance index {instance_idx} out of range")
            
            instance = X.iloc[instance_idx].values
            
            # Generate explanation
            if hasattr(model, 'predict_proba'):
                explanation = explainer.explain_instance(
                    instance,
                    model.predict_proba,
                    num_features=num_features,
                    num_samples=num_samples
                )
            else:
                explanation = explainer.explain_instance(
                    instance,
                    model.predict,
                    num_features=num_features,
                    num_samples=num_samples
                )
            
            # Extract explanation data
            feature_explanations = []
            for feature_idx, weight in explanation.as_list():
                feature_name = self.feature_names[model_id][feature_idx] if isinstance(feature_idx, int) else str(feature_idx)
                feature_explanations.append({
                    "feature": feature_name,
                    "weight": float(weight),
                    "contribution": "positive" if weight > 0 else "negative",
                    "abs_weight": abs(float(weight))
                })
            
            # Sort by absolute weight
            feature_explanations.sort(key=lambda x: x["abs_weight"], reverse=True)
            
            # Get prediction information
            if hasattr(model, 'predict_proba'):
                prediction_proba = model.predict_proba([instance])[0]
                predicted_class = np.argmax(prediction_proba)
                prediction_info = {
                    "predicted_class": self.class_names[model_id][predicted_class] if self.class_names[model_id] else str(predicted_class),
                    "prediction_probability": float(prediction_proba[predicted_class]),
                    "all_probabilities": [float(p) for p in prediction_proba]
                }
            else:
                prediction_value = model.predict([instance])[0]
                prediction_info = {
                    "predicted_value": float(prediction_value),
                    "prediction_type": "regression"
                }
            
            return {
                "status": "success",
                "model_id": model_id,
                "instance_idx": instance_idx,
                "explanation_method": "LIME",
                "feature_explanations": feature_explanations,
                "prediction_info": prediction_info,
                "instance_values": {
                    feature: float(value) for feature, value in 
                    zip(self.feature_names[model_id], instance)
                },
                "lime_score": explanation.score,
                "local_prediction": explanation.local_pred[0] if hasattr(explanation, 'local_pred') else None
            }
            
        except Exception as e:
            logger.error(f"Failed to generate LIME explanations: {e}")
            return {
                "status": "error",
                "message": str(e),
                "model_id": model_id
            }
    
    async def get_feature_importance(
        self,
        model_id: str,
        method: str = "shap"
    ) -> Dict[str, Any]:
        """
        Get global feature importance using various methods
        
        Args:
            model_id: Model identifier
            method: Method to use (shap, permutation, builtin)
        
        Returns:
            Dictionary with feature importance data
        """
        try:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")
            
            model = self.models[model_id]
            feature_names = self.feature_names[model_id]
            
            if method == "builtin" and hasattr(model, 'feature_importances_'):
                # Use built-in feature importance (for tree-based models)
                importances = model.feature_importances_
                
            elif method == "shap" and model_id in self.explainers:
                # Use SHAP-based feature importance
                if 'shap_tree' in self.explainers[model_id]:
                    explainer = self.explainers[model_id]['shap_tree']
                    # Get mean absolute SHAP values as importance
                    importances = np.abs(explainer.shap_values(
                        self.models[f"{model_id}_X_train"][:100] if f"{model_id}_X_train" in self.models else None
                    )).mean(axis=0)
                else:
                    raise ValueError("SHAP explainer not available")
            else:
                raise ValueError(f"Method {method} not supported or not available")
            
            # Create feature importance list
            feature_importance = [
                {
                    "feature": feature,
                    "importance": float(importance),
                    "rank": rank + 1,
                    "percentage": float(importance / np.sum(importances) * 100)
                }
                for rank, (feature, importance) in enumerate(
                    zip(feature_names, importances)
                )
            ]
            
            # Sort by importance
            feature_importance.sort(key=lambda x: x["importance"], reverse=True)
            
            # Update ranks
            for rank, item in enumerate(feature_importance):
                item["rank"] = rank + 1
            
            return {
                "status": "success",
                "model_id": model_id,
                "method": method,
                "feature_importance": feature_importance,
                "total_features": len(feature_names),
                "importance_sum": float(np.sum(importances))
            }
            
        except Exception as e:
            logger.error(f"Failed to get feature importance: {e}")
            return {
                "status": "error",
                "message": str(e),
                "model_id": model_id
            }
    
    async def get_partial_dependence(
        self,
        model_id: str,
        feature_name: str,
        X: pd.DataFrame,
        num_points: int = 50
    ) -> Dict[str, Any]:
        """
        Calculate partial dependence for a specific feature
        
        Args:
            model_id: Model identifier
            feature_name: Name of feature to analyze
            X: Input data
            num_points: Number of points for PD curve
        
        Returns:
            Dictionary with partial dependence data
        """
        try:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")
            
            model = self.models[model_id]
            feature_names = self.feature_names[model_id]
            
            if feature_name not in feature_names:
                raise ValueError(f"Feature {feature_name} not found in model features")
            
            feature_idx = feature_names.index(feature_name)
            feature_values = X.iloc[:, feature_idx]
            
            # Create range of values for the feature
            if X[feature_name].dtype in ['int64', 'float64']:
                # Numerical feature
                min_val, max_val = feature_values.min(), feature_values.max()
                test_values = np.linspace(min_val, max_val, num_points)
            else:
                # Categorical feature
                test_values = feature_values.unique()[:num_points]
            
            # Calculate partial dependence
            pd_values = []
            baseline_data = X.copy()
            
            for test_val in test_values:
                # Replace feature with test value for all instances
                baseline_data.iloc[:, feature_idx] = test_val
                
                # Get predictions
                if hasattr(model, 'predict_proba'):
                    # Classification - use probability of positive class
                    predictions = model.predict_proba(baseline_data)
                    if predictions.shape[1] == 2:
                        avg_prediction = predictions[:, 1].mean()
                    else:
                        avg_prediction = predictions.max(axis=1).mean()
                else:
                    # Regression
                    predictions = model.predict(baseline_data)
                    avg_prediction = predictions.mean()
                
                pd_values.append({
                    "feature_value": float(test_val),
                    "partial_dependence": float(avg_prediction),
                    "feature_name": feature_name
                })
            
            # Sort by feature value if numerical
            if X[feature_name].dtype in ['int64', 'float64']:
                pd_values.sort(key=lambda x: x["feature_value"])
            
            return {
                "status": "success",
                "model_id": model_id,
                "feature_name": feature_name,
                "partial_dependence": pd_values,
                "feature_type": "numerical" if X[feature_name].dtype in ['int64', 'float64'] else "categorical",
                "num_points": len(pd_values)
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate partial dependence: {e}")
            return {
                "status": "error",
                "message": str(e),
                "model_id": model_id
            }
    
    async def get_feature_interactions(
        self,
        model_id: str,
        X: pd.DataFrame,
        top_k: int = 10,
        method: str = "shap"
    ) -> Dict[str, Any]:
        """
        Analyze feature interactions
        
        Args:
            model_id: Model identifier
            X: Input data
            top_k: Number of top interactions to return
            method: Method to use for interaction detection
        
        Returns:
            Dictionary with feature interaction data
        """
        try:
            if model_id not in self.explainers:
                raise ValueError(f"Model {model_id} not initialized")
            
            if method == "shap" and 'shap_tree' in self.explainers[model_id]:
                explainer = self.explainers[model_id]['shap_tree']
                
                # Get SHAP interaction values
                shap_interaction_values = explainer.shap_interaction_values(X.head(100))
                
                # Calculate interaction strengths
                interactions = []
                feature_names = self.feature_names[model_id]
                n_features = len(feature_names)
                
                for i in range(n_features):
                    for j in range(i + 1, n_features):
                        # Interaction strength is mean absolute interaction value
                        interaction_strength = np.abs(shap_interaction_values[:, i, j]).mean()
                        
                        interactions.append({
                            "feature_1": feature_names[i],
                            "feature_2": feature_names[j],
                            "interaction_strength": float(interaction_strength),
                            "feature_1_idx": i,
                            "feature_2_idx": j
                        })
                
                # Sort by interaction strength and take top k
                interactions.sort(key=lambda x: x["interaction_strength"], reverse=True)
                top_interactions = interactions[:top_k]
                
                return {
                    "status": "success",
                    "model_id": model_id,
                    "method": method,
                    "top_interactions": top_interactions,
                    "total_interactions": len(interactions)
                }
            
            else:
                raise ValueError(f"Interaction method {method} not supported")
                
        except Exception as e:
            logger.error(f"Failed to analyze feature interactions: {e}")
            return {
                "status": "error",
                "message": str(e),
                "model_id": model_id
            }
    
    async def generate_counterfactuals(
        self,
        model_id: str,
        X: pd.DataFrame,
        instance_idx: int = 0,
        target_class: Optional[int] = None,
        num_counterfactuals: int = 5
    ) -> Dict[str, Any]:
        """
        Generate counterfactual explanations
        
        Args:
            model_id: Model identifier
            X: Input data
            instance_idx: Index of instance to generate counterfactuals for
            target_class: Target class for counterfactuals
            num_counterfactuals: Number of counterfactuals to generate
        
        Returns:
            Dictionary with counterfactual data
        """
        try:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")
            
            model = self.models[model_id]
            feature_names = self.feature_names[model_id]
            
            if instance_idx >= len(X):
                raise ValueError(f"Instance index {instance_idx} out of range")
            
            original_instance = X.iloc[instance_idx].values
            original_prediction = model.predict([original_instance])[0]
            
            if hasattr(model, 'predict_proba'):
                original_proba = model.predict_proba([original_instance])[0]
                original_class = np.argmax(original_proba)
                
                # If no target class specified, use the opposite of current prediction
                if target_class is None:
                    if len(original_proba) == 2:
                        target_class = 1 - original_class
                    else:
                        # Multi-class: target the second most likely class
                        target_class = np.argsort(original_proba)[-2]
            
            # Simple counterfactual generation using perturbation
            counterfactuals = []
            
            for _ in range(num_counterfactuals * 10):  # Generate more and filter
                # Create perturbed instance
                perturbed_instance = original_instance.copy()
                
                # Randomly perturb 1-3 features
                n_perturb = np.random.randint(1, min(4, len(feature_names)))
                perturb_indices = np.random.choice(len(feature_names), n_perturb, replace=False)
                
                for idx in perturb_indices:
                    feature_values = X.iloc[:, idx].values
                    if X.dtypes[idx] in ['int64', 'float64']:
                        # Numerical feature: add random noise
                        std = feature_values.std()
                        perturbed_instance[idx] += np.random.normal(0, std * 0.5)
                    else:
                        # Categorical feature: random replacement
                        unique_vals = np.unique(feature_values)
                        perturbed_instance[idx] = np.random.choice(unique_vals)
                
                # Check if this achieves the target
                if hasattr(model, 'predict_proba'):
                    pred_proba = model.predict_proba([perturbed_instance])[0]
                    pred_class = np.argmax(pred_proba)
                    
                    if pred_class == target_class:
                        # Calculate changes
                        changes = []
                        for i, (orig, new) in enumerate(zip(original_instance, perturbed_instance)):
                            if orig != new:
                                changes.append({
                                    "feature": feature_names[i],
                                    "original_value": float(orig),
                                    "new_value": float(new),
                                    "change": float(new - orig) if X.dtypes[i] in ['int64', 'float64'] else "categorical"
                                })
                        
                        counterfactuals.append({
                            "counterfactual_id": len(counterfactuals),
                            "changes": changes,
                            "predicted_class": self.class_names[model_id][pred_class] if self.class_names[model_id] else str(pred_class),
                            "prediction_probability": float(pred_proba[pred_class]),
                            "distance_from_original": float(np.linalg.norm(perturbed_instance - original_instance))
                        })
                        
                        if len(counterfactuals) >= num_counterfactuals:
                            break
            
            return {
                "status": "success",
                "model_id": model_id,
                "instance_idx": instance_idx,
                "original_prediction": {
                    "class": self.class_names[model_id][original_class] if hasattr(model, 'predict_proba') and self.class_names[model_id] else str(original_prediction),
                    "probability": float(original_proba[original_class]) if hasattr(model, 'predict_proba') else float(original_prediction)
                },
                "target_class": self.class_names[model_id][target_class] if hasattr(model, 'predict_proba') and self.class_names[model_id] else str(target_class),
                "counterfactuals": counterfactuals,
                "num_generated": len(counterfactuals)
            }
            
        except Exception as e:
            logger.error(f"Failed to generate counterfactuals: {e}")
            return {
                "status": "error",
                "message": str(e),
                "model_id": model_id
            }
    
    async def cleanup_explainer(self, model_id: str) -> Dict[str, Any]:
        """
        Clean up explainer resources for a model
        
        Args:
            model_id: Model identifier
        
        Returns:
            Dictionary with cleanup status
        """
        try:
            removed_components = []
            
            if model_id in self.explainers:
                del self.explainers[model_id]
                removed_components.append("explainers")
            
            if model_id in self.models:
                del self.models[model_id]
                removed_components.append("model")
            
            if model_id in self.feature_names:
                del self.feature_names[model_id]
                removed_components.append("feature_names")
            
            if model_id in self.class_names:
                del self.class_names[model_id]
                removed_components.append("class_names")
            
            return {
                "status": "success",
                "model_id": model_id,
                "removed_components": removed_components
            }
            
        except Exception as e:
            logger.error(f"Failed to cleanup explainer: {e}")
            return {
                "status": "error",
                "message": str(e),
                "model_id": model_id
            }

# Global service instance
explainability_service = ExplainabilityService()