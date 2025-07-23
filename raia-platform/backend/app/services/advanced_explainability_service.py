"""
Advanced Model Explainability Service
Comprehensive implementation with Alibi, advanced Captum, ELI5, and prototype-based explanations
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
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.inspection import permutation_importance
import joblib

# SHAP imports
import shap

# LIME imports
import lime
import lime.lime_tabular
import lime.lime_text
import lime.lime_image

# Advanced Captum imports
from captum.attr import (
    IntegratedGradients, 
    GradientShap, 
    DeepLift,
    DeepLiftShap,
    LayerConductance,
    LayerIntegratedGradients,
    LayerGradientXActivation,
    Saliency,
    InputXGradient,
    GuidedBackprop,
    GuidedGradCam,
    Occlusion,
    FeatureAblation,
    NoiseTunnel
)

# Alibi imports
try:
    from alibi.explainers import (
        AnchorTabular,
        AnchorText, 
        AnchorImage,
        ALE,
        CounterfactualProto,
        CEM,
        Counterfactual
    )
    ALIBI_AVAILABLE = True
except ImportError:
    ALIBI_AVAILABLE = False
    logging.warning("Alibi not available. Some advanced explainability features will be disabled.")

# ELI5 imports
try:
    import eli5
    from eli5.sklearn import PermutationImportance
    from eli5 import show_weights
    ELI5_AVAILABLE = True
except ImportError:
    ELI5_AVAILABLE = False
    logging.warning("ELI5 not available. Permutation importance features will be limited.")

# Additional scientific libraries
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import KMeans
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

class AdvancedExplainabilityService:
    """
    Advanced model explainability service with comprehensive explanation methods
    """
    
    def __init__(self):
        self.explainers = {}
        self.models = {}
        self.preprocessors = {}
        self.feature_names = {}
        self.class_names = {}
        self.training_data = {}
        self.model_types = {}
        
    async def initialize_advanced_explainer(
        self, 
        model_id: str,
        model: Union[BaseEstimator, nn.Module],
        X_train: pd.DataFrame,
        feature_names: List[str],
        class_names: Optional[List[str]] = None,
        model_type: str = "tabular",
        categorical_features: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Initialize advanced explainers for a given model
        
        Args:
            model_id: Unique identifier for the model
            model: Trained model object
            X_train: Training data for background/baseline
            feature_names: List of feature names
            class_names: List of class names for classification
            model_type: Type of model (tabular, text, image, neural)
            categorical_features: Indices of categorical features
        
        Returns:
            Dictionary with initialization status and explainer info
        """
        try:
            # Store model and metadata
            self.models[model_id] = model
            self.feature_names[model_id] = feature_names
            self.class_names[model_id] = class_names or []
            self.training_data[model_id] = X_train.copy()
            self.model_types[model_id] = model_type
            
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
                        categorical_features=categorical_features,
                        mode='classification' if hasattr(model, 'predict_proba') else 'regression',
                        discretize_continuous=True,
                        random_state=42
                    )
                    logger.info(f"Initialized LIME TabularExplainer for model {model_id}")
                except Exception as e:
                    logger.warning(f"Failed to initialize LIME explainer: {e}")
            
            # Initialize Alibi explainers
            if ALIBI_AVAILABLE and model_type == "tabular":
                try:
                    # AnchorTabular explainer
                    explainers['anchor_tabular'] = AnchorTabular(
                        predictor=model.predict_proba if hasattr(model, 'predict_proba') else model.predict,
                        feature_names=feature_names,
                        categorical_names={i: f"cat_{i}" for i in (categorical_features or [])}
                    )
                    explainers['anchor_tabular'].fit(X_train.values)
                    logger.info(f"Initialized Alibi AnchorTabular for model {model_id}")
                    
                    # ALE explainer
                    explainers['ale'] = ALE(
                        predictor=model.predict_proba if hasattr(model, 'predict_proba') else model.predict,
                        feature_names=feature_names
                    )
                    logger.info(f"Initialized Alibi ALE for model {model_id}")
                    
                except Exception as e:
                    logger.warning(f"Failed to initialize Alibi explainers: {e}")
            
            # Initialize ELI5 explainers
            if ELI5_AVAILABLE:
                try:
                    # Create permutation importance explainer
                    if hasattr(model, 'predict'):
                        explainers['eli5_permutation'] = PermutationImportance(
                            model, random_state=42, n_iter=10
                        )
                        # Fit on a sample of training data for performance
                        sample_size = min(1000, len(X_train))
                        sample_indices = np.random.choice(len(X_train), sample_size, replace=False)
                        X_sample = X_train.iloc[sample_indices]
                        
                        # Need target values for fitting
                        if hasattr(model, 'predict'):
                            y_sample = model.predict(X_sample)
                            explainers['eli5_permutation'].fit(X_sample, y_sample)
                            logger.info(f"Initialized ELI5 PermutationImportance for model {model_id}")
                        
                except Exception as e:
                    logger.warning(f"Failed to initialize ELI5 explainers: {e}")
            
            # Initialize Captum explainers for neural networks
            if isinstance(model, nn.Module):
                try:
                    explainers['captum_ig'] = IntegratedGradients(model)
                    explainers['captum_gs'] = GradientShap(model)
                    explainers['captum_dl'] = DeepLift(model)
                    explainers['captum_dls'] = DeepLiftShap(model)
                    explainers['captum_saliency'] = Saliency(model)
                    explainers['captum_input_grad'] = InputXGradient(model)
                    explainers['captum_occlusion'] = Occlusion(model)
                    explainers['captum_ablation'] = FeatureAblation(model)
                    explainers['captum_noise'] = NoiseTunnel(IntegratedGradients(model))
                    
                    logger.info(f"Initialized advanced Captum explainers for neural model {model_id}")
                    
                except Exception as e:
                    logger.warning(f"Failed to initialize Captum explainers: {e}")
            
            self.explainers[model_id] = explainers
            
            return {
                "status": "success",
                "model_id": model_id,
                "explainers_initialized": list(explainers.keys()),
                "feature_count": len(feature_names),
                "model_type": model_type,
                "has_alibi": ALIBI_AVAILABLE,
                "has_eli5": ELI5_AVAILABLE
            }
            
        except Exception as e:
            logger.error(f"Failed to initialize advanced explainers for model {model_id}: {e}")
            return {
                "status": "error",
                "message": str(e),
                "model_id": model_id
            }
    
    async def get_anchor_explanations(
        self,
        model_id: str,
        X: pd.DataFrame,
        instance_idx: int = 0,
        threshold: float = 0.95,
        max_anchor_size: int = 5
    ) -> Dict[str, Any]:
        """
        Generate Anchor explanations for tabular data
        
        Args:
            model_id: Model identifier
            X: Input data
            instance_idx: Index of instance to explain
            threshold: Precision threshold for anchors
            max_anchor_size: Maximum number of features in anchor
        
        Returns:
            Dictionary with anchor explanation data
        """
        try:
            if not ALIBI_AVAILABLE:
                raise ValueError("Alibi not available for Anchor explanations")
            
            if model_id not in self.explainers or 'anchor_tabular' not in self.explainers[model_id]:
                raise ValueError(f"AnchorTabular explainer not available for model {model_id}")
            
            explainer = self.explainers[model_id]['anchor_tabular']
            
            if instance_idx >= len(X):
                raise ValueError(f"Instance index {instance_idx} out of range")
            
            instance = X.iloc[instance_idx].values.reshape(1, -1)
            
            # Generate anchor explanation
            explanation = explainer.explain(
                instance, 
                threshold=threshold,
                max_anchor_size=max_anchor_size
            )
            
            # Extract anchor information
            anchor_features = []
            if hasattr(explanation, 'names') and explanation.names:
                for i, feature_condition in enumerate(explanation.names):
                    anchor_features.append({
                        "feature_condition": feature_condition,
                        "feature_index": i,
                        "precision": float(explanation.precision),
                        "coverage": float(explanation.coverage)
                    })
            
            # Get prediction information
            model = self.models[model_id]
            if hasattr(model, 'predict_proba'):
                prediction_proba = model.predict_proba(instance)[0]
                predicted_class = np.argmax(prediction_proba)
                prediction_info = {
                    "predicted_class": self.class_names[model_id][predicted_class] if self.class_names[model_id] else str(predicted_class),
                    "prediction_probability": float(prediction_proba[predicted_class]),
                    "all_probabilities": [float(p) for p in prediction_proba]
                }
            else:
                prediction_value = model.predict(instance)[0]
                prediction_info = {
                    "predicted_value": float(prediction_value),
                    "prediction_type": "regression"
                }
            
            return {
                "status": "success",
                "model_id": model_id,
                "instance_idx": instance_idx,
                "explanation_method": "Anchor",
                "anchor_features": anchor_features,
                "anchor_precision": float(explanation.precision),
                "anchor_coverage": float(explanation.coverage),
                "prediction_info": prediction_info,
                "instance_values": {
                    feature: float(value) for feature, value in 
                    zip(self.feature_names[model_id], instance[0])
                },
                "interpretation": self._interpret_anchor_explanation(anchor_features, explanation.precision)
            }
            
        except Exception as e:
            logger.error(f"Failed to generate Anchor explanations: {e}")
            return {
                "status": "error",
                "message": str(e),
                "model_id": model_id
            }
    
    async def get_ale_plots(
        self,
        model_id: str,
        X: pd.DataFrame,
        features: Optional[List[str]] = None,
        num_quantiles: int = 50
    ) -> Dict[str, Any]:
        """
        Generate Accumulated Local Effects (ALE) plots
        
        Args:
            model_id: Model identifier
            X: Input data
            features: List of features to analyze (if None, analyze all)
            num_quantiles: Number of quantiles for ALE plot
        
        Returns:
            Dictionary with ALE plot data
        """
        try:
            if not ALIBI_AVAILABLE:
                raise ValueError("Alibi not available for ALE plots")
            
            if model_id not in self.explainers or 'ale' not in self.explainers[model_id]:
                raise ValueError(f"ALE explainer not available for model {model_id}")
            
            explainer = self.explainers[model_id]['ale']
            feature_names = self.feature_names[model_id]
            
            if features is None:
                features = feature_names
            
            ale_results = []
            
            for feature in features:
                if feature not in feature_names:
                    logger.warning(f"Feature {feature} not found in model features")
                    continue
                
                feature_idx = feature_names.index(feature)
                
                # Generate ALE plot data
                ale_data = explainer.explain(
                    X.values,
                    features=[feature_idx],
                    n_bins=num_quantiles
                )
                
                # Extract ALE plot information
                ale_values = ale_data.ale[0]  # ALE values for the feature
                feature_values = ale_data.feature_values[0]  # Feature bin boundaries
                
                ale_plot_data = []
                for i, (fval, ale_val) in enumerate(zip(feature_values[:-1], ale_values)):
                    ale_plot_data.append({
                        "feature_value": float(fval),
                        "ale_value": float(ale_val),
                        "bin_index": i
                    })
                
                ale_results.append({
                    "feature": feature,
                    "feature_index": feature_idx,
                    "ale_data": ale_plot_data,
                    "feature_type": "numerical" if X[feature].dtype in ['int64', 'float64'] else "categorical",
                    "min_ale": float(np.min(ale_values)),
                    "max_ale": float(np.max(ale_values)),
                    "ale_range": float(np.max(ale_values) - np.min(ale_values))
                })
            
            return {
                "status": "success",
                "model_id": model_id,
                "explanation_method": "ALE",
                "ale_results": ale_results,
                "num_features_analyzed": len(ale_results),
                "num_quantiles": num_quantiles
            }
            
        except Exception as e:
            logger.error(f"Failed to generate ALE plots: {e}")
            return {
                "status": "error",
                "message": str(e),
                "model_id": model_id
            }
    
    async def get_advanced_permutation_importance(
        self,
        model_id: str,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        n_repeats: int = 10,
        scoring: str = "accuracy"
    ) -> Dict[str, Any]:
        """
        Get advanced permutation importance using ELI5
        
        Args:
            model_id: Model identifier
            X: Input data
            y: Target values (if None, use model predictions)
            n_repeats: Number of permutation repeats
            scoring: Scoring method for importance
        
        Returns:
            Dictionary with advanced permutation importance data
        """
        try:
            model = self.models[model_id]
            feature_names = self.feature_names[model_id]
            
            if y is None:
                # Use model predictions as target
                y = model.predict(X)
            
            # Standard sklearn permutation importance
            perm_importance = permutation_importance(
                model, X, y, 
                n_repeats=n_repeats,
                random_state=42,
                scoring=scoring
            )
            
            # Create importance results
            importance_results = []
            for i, feature in enumerate(feature_names):
                importance_results.append({
                    "feature": feature,
                    "importance_mean": float(perm_importance.importances_mean[i]),
                    "importance_std": float(perm_importance.importances_std[i]),
                    "importance_scores": [float(score) for score in perm_importance.importances[i]],
                    "rank": 0  # Will be updated after sorting
                })
            
            # Sort by importance and update ranks
            importance_results.sort(key=lambda x: x["importance_mean"], reverse=True)
            for rank, result in enumerate(importance_results):
                result["rank"] = rank + 1
            
            # ELI5 analysis if available
            eli5_results = None
            if ELI5_AVAILABLE and model_id in self.explainers and 'eli5_permutation' in self.explainers[model_id]:
                try:
                    eli5_explainer = self.explainers[model_id]['eli5_permutation']
                    eli5_weights = show_weights(eli5_explainer, feature_names=feature_names)
                    eli5_results = {
                        "available": True,
                        "html_explanation": eli5_weights.data if hasattr(eli5_weights, 'data') else str(eli5_weights)
                    }
                except Exception as e:
                    eli5_results = {
                        "available": False,
                        "error": str(e)
                    }
            else:
                eli5_results = {
                    "available": False,
                    "message": "ELI5 not available or not initialized"
                }
            
            # Statistical analysis
            total_importance = sum(result["importance_mean"] for result in importance_results)
            
            return {
                "status": "success",
                "model_id": model_id,
                "explanation_method": "Advanced Permutation Importance",
                "importance_results": importance_results,
                "total_importance": float(total_importance),
                "n_repeats": n_repeats,
                "scoring": scoring,
                "eli5_analysis": eli5_results,
                "statistical_summary": {
                    "top_5_features": [r["feature"] for r in importance_results[:5]],
                    "importance_concentration": float(sum(r["importance_mean"] for r in importance_results[:5]) / total_importance) if total_importance > 0 else 0,
                    "feature_count": len(importance_results)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate advanced permutation importance: {e}")
            return {
                "status": "error",
                "message": str(e),
                "model_id": model_id
            }
    
    async def get_prototype_explanations(
        self,
        model_id: str,
        X: pd.DataFrame,
        instance_idx: int = 0,
        n_prototypes: int = 5,
        distance_metric: str = "euclidean"
    ) -> Dict[str, Any]:
        """
        Generate prototype-based explanations
        
        Args:
            model_id: Model identifier
            X: Input data
            instance_idx: Index of instance to explain
            n_prototypes: Number of prototypes to find
            distance_metric: Distance metric for prototype selection
        
        Returns:
            Dictionary with prototype explanation data
        """
        try:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")
            
            model = self.models[model_id]
            training_data = self.training_data[model_id]
            feature_names = self.feature_names[model_id]
            
            if instance_idx >= len(X):
                raise ValueError(f"Instance index {instance_idx} out of range")
            
            target_instance = X.iloc[instance_idx].values
            
            # Get prediction for target instance
            if hasattr(model, 'predict_proba'):
                target_prediction = model.predict_proba([target_instance])[0]
                target_class = np.argmax(target_prediction)
            else:
                target_prediction = model.predict([target_instance])[0]
                target_class = None
            
            # Find training instances of same class (for classification)
            if target_class is not None:
                training_predictions = model.predict(training_data)
                same_class_mask = training_predictions == target_class
                candidate_data = training_data[same_class_mask]
            else:
                candidate_data = training_data
            
            # Calculate distances to all training instances
            distances = pairwise_distances(
                [target_instance], 
                candidate_data,
                metric=distance_metric
            )[0]
            
            # Find closest prototypes
            prototype_indices = np.argsort(distances)[:n_prototypes]
            
            prototypes = []
            for i, proto_idx in enumerate(prototype_indices):
                prototype_data = candidate_data.iloc[proto_idx]
                proto_distance = distances[proto_idx]
                
                # Calculate feature-wise differences
                feature_differences = []
                for j, (feature, target_val, proto_val) in enumerate(zip(
                    feature_names, target_instance, prototype_data.values
                )):
                    difference = float(proto_val - target_val)
                    relative_diff = difference / target_val if target_val != 0 else difference
                    
                    feature_differences.append({
                        "feature": feature,
                        "target_value": float(target_val),
                        "prototype_value": float(proto_val),
                        "absolute_difference": abs(difference),
                        "relative_difference": float(relative_diff),
                        "feature_index": j
                    })
                
                # Sort by absolute difference
                feature_differences.sort(key=lambda x: x["absolute_difference"], reverse=True)
                
                prototypes.append({
                    "prototype_id": i + 1,
                    "original_index": int(proto_idx),
                    "distance": float(proto_distance),
                    "similarity_score": float(1 / (1 + proto_distance)),
                    "feature_differences": feature_differences,
                    "top_differing_features": [fd["feature"] for fd in feature_differences[:3]]
                })
            
            # Generate interpretation
            interpretation = self._interpret_prototype_explanations(
                prototypes, target_class, self.class_names[model_id]
            )
            
            return {
                "status": "success",
                "model_id": model_id,
                "instance_idx": instance_idx,
                "explanation_method": "Prototype Analysis",
                "target_prediction": {
                    "class": self.class_names[model_id][target_class] if target_class is not None and self.class_names[model_id] else str(target_class),
                    "probability": float(target_prediction[target_class]) if target_class is not None else float(target_prediction)
                },
                "prototypes": prototypes,
                "n_prototypes": len(prototypes),
                "distance_metric": distance_metric,
                "interpretation": interpretation
            }
            
        except Exception as e:
            logger.error(f"Failed to generate prototype explanations: {e}")
            return {
                "status": "error",
                "message": str(e),
                "model_id": model_id
            }
    
    async def get_ice_plots(
        self,
        model_id: str,
        X: pd.DataFrame,
        feature_name: str,
        num_ice_lines: int = 50,
        num_points: int = 50
    ) -> Dict[str, Any]:
        """
        Generate Individual Conditional Expectation (ICE) plots
        
        Args:
            model_id: Model identifier
            X: Input data
            feature_name: Name of feature to analyze
            num_ice_lines: Number of ICE lines to plot
            num_points: Number of points per ICE line
        
        Returns:
            Dictionary with ICE plot data
        """
        try:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")
            
            model = self.models[model_id]
            feature_names = self.feature_names[model_id]
            
            if feature_name not in feature_names:
                raise ValueError(f"Feature {feature_name} not found in model features")
            
            feature_idx = feature_names.index(feature_name)
            
            # Select subset of instances for ICE lines
            n_instances = min(num_ice_lines, len(X))
            ice_indices = np.random.choice(len(X), n_instances, replace=False)
            ice_data = X.iloc[ice_indices].copy()
            
            # Create range of values for the feature
            if X[feature_name].dtype in ['int64', 'float64']:
                # Numerical feature
                min_val, max_val = X[feature_name].min(), X[feature_name].max()
                feature_range = np.linspace(min_val, max_val, num_points)
            else:
                # Categorical feature
                feature_range = X[feature_name].unique()[:num_points]
            
            # Generate ICE lines
            ice_lines = []
            for instance_idx, original_idx in enumerate(ice_indices):
                ice_line_data = []
                baseline_data = ice_data.iloc[instance_idx:instance_idx+1].copy()
                
                for feature_val in feature_range:
                    # Replace feature value
                    baseline_data.iloc[0, feature_idx] = feature_val
                    
                    # Get prediction
                    if hasattr(model, 'predict_proba'):
                        # Classification - use probability of positive class
                        prediction = model.predict_proba(baseline_data)[0]
                        if len(prediction) == 2:
                            pred_value = prediction[1]  # Positive class probability
                        else:
                            pred_value = prediction.max()  # Max probability for multiclass
                    else:
                        # Regression
                        pred_value = model.predict(baseline_data)[0]
                    
                    ice_line_data.append({
                        "feature_value": float(feature_val),
                        "prediction": float(pred_value)
                    })
                
                ice_lines.append({
                    "instance_id": int(original_idx),
                    "ice_line": ice_line_data,
                    "original_feature_value": float(X.iloc[original_idx][feature_name])
                })
            
            # Calculate Partial Dependence (average of ICE lines)
            pd_data = []
            for i, feature_val in enumerate(feature_range):
                avg_prediction = np.mean([line["ice_line"][i]["prediction"] for line in ice_lines])
                pd_data.append({
                    "feature_value": float(feature_val),
                    "partial_dependence": float(avg_prediction)
                })
            
            # Calculate ICE variance (measure of interaction strength)
            ice_variance = []
            for i in range(len(feature_range)):
                predictions_at_value = [line["ice_line"][i]["prediction"] for line in ice_lines]
                variance = np.var(predictions_at_value)
                ice_variance.append({
                    "feature_value": float(feature_range[i]),
                    "variance": float(variance)
                })
            
            return {
                "status": "success",
                "model_id": model_id,
                "feature_name": feature_name,
                "explanation_method": "ICE Plot",
                "ice_lines": ice_lines,
                "partial_dependence": pd_data,
                "ice_variance": ice_variance,
                "num_instances": n_instances,
                "num_points": len(feature_range),
                "feature_type": "numerical" if X[feature_name].dtype in ['int64', 'float64'] else "categorical",
                "interpretation": self._interpret_ice_plot(ice_lines, pd_data, ice_variance, feature_name)
            }
            
        except Exception as e:
            logger.error(f"Failed to generate ICE plots: {e}")
            return {
                "status": "error",
                "message": str(e),
                "model_id": model_id
            }
    
    async def get_advanced_counterfactuals(
        self,
        model_id: str,
        X: pd.DataFrame,
        instance_idx: int = 0,
        target_class: Optional[int] = None,
        num_counterfactuals: int = 5,
        method: str = "diverse"
    ) -> Dict[str, Any]:
        """
        Generate advanced counterfactual explanations
        
        Args:
            model_id: Model identifier
            X: Input data
            instance_idx: Index of instance to generate counterfactuals for
            target_class: Target class for counterfactuals
            num_counterfactuals: Number of counterfactuals to generate
            method: Counterfactual generation method
        
        Returns:
            Dictionary with advanced counterfactual data
        """
        try:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")
            
            model = self.models[model_id]
            feature_names = self.feature_names[model_id]
            training_data = self.training_data[model_id]
            
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
            
            # Advanced counterfactual generation using optimization
            counterfactuals = []
            
            if method == "diverse":
                # Generate diverse counterfactuals using different strategies
                strategies = ["minimal_change", "realistic", "sparse", "dense"]
                cf_per_strategy = max(1, num_counterfactuals // len(strategies))
                
                for strategy in strategies:
                    strategy_counterfactuals = self._generate_counterfactuals_by_strategy(
                        original_instance, model, training_data, target_class, 
                        cf_per_strategy, strategy, feature_names
                    )
                    counterfactuals.extend(strategy_counterfactuals)
            
            else:
                counterfactuals = self._generate_counterfactuals_by_strategy(
                    original_instance, model, training_data, target_class,
                    num_counterfactuals, method, feature_names
                )
            
            # Limit to requested number
            counterfactuals = counterfactuals[:num_counterfactuals]
            
            # Calculate quality metrics for counterfactuals
            for cf in counterfactuals:
                cf["quality_metrics"] = self._calculate_counterfactual_quality(
                    original_instance, cf["counterfactual_values"], 
                    model, training_data
                )
            
            return {
                "status": "success",
                "model_id": model_id,
                "instance_idx": instance_idx,
                "explanation_method": "Advanced Counterfactuals",
                "original_prediction": {
                    "class": self.class_names[model_id][original_class] if hasattr(model, 'predict_proba') and self.class_names[model_id] else str(original_prediction),
                    "probability": float(original_proba[original_class]) if hasattr(model, 'predict_proba') else float(original_prediction)
                },
                "target_class": self.class_names[model_id][target_class] if hasattr(model, 'predict_proba') and self.class_names[model_id] else str(target_class),
                "counterfactuals": counterfactuals,
                "num_generated": len(counterfactuals),
                "generation_method": method,
                "interpretation": self._interpret_counterfactuals(counterfactuals, original_instance, feature_names)
            }
            
        except Exception as e:
            logger.error(f"Failed to generate advanced counterfactuals: {e}")
            return {
                "status": "error",
                "message": str(e),
                "model_id": model_id
            }
    
    def _generate_counterfactuals_by_strategy(
        self, 
        original_instance: np.ndarray,
        model: BaseEstimator,
        training_data: pd.DataFrame,
        target_class: int,
        num_cf: int,
        strategy: str,
        feature_names: List[str]
    ) -> List[Dict[str, Any]]:
        """Generate counterfactuals using specific strategy"""
        
        counterfactuals = []
        max_attempts = num_cf * 20  # Limit search attempts
        
        for attempt in range(max_attempts):
            if len(counterfactuals) >= num_cf:
                break
                
            # Create perturbed instance based on strategy
            if strategy == "minimal_change":
                # Minimize number of changed features
                n_changes = np.random.randint(1, min(3, len(feature_names)))
                perturbed_instance = self._minimal_change_perturbation(
                    original_instance, training_data, n_changes
                )
            
            elif strategy == "realistic":
                # Use training data distribution for realistic changes
                perturbed_instance = self._realistic_perturbation(
                    original_instance, training_data
                )
            
            elif strategy == "sparse":
                # Change only 1-2 features
                perturbed_instance = self._sparse_perturbation(
                    original_instance, training_data
                )
            
            elif strategy == "dense":
                # Change many features with small perturbations
                perturbed_instance = self._dense_perturbation(
                    original_instance, training_data
                )
            
            else:
                # Default random perturbation
                perturbed_instance = self._random_perturbation(
                    original_instance, training_data
                )
            
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
                                "change": float(new - orig) if training_data.dtypes.iloc[i] in ['int64', 'float64'] else "categorical",
                                "relative_change": float((new - orig) / orig) if orig != 0 and training_data.dtypes.iloc[i] in ['int64', 'float64'] else 0
                            })
                    
                    counterfactuals.append({
                        "counterfactual_id": len(counterfactuals),
                        "strategy": strategy,
                        "changes": changes,
                        "counterfactual_values": perturbed_instance.tolist(),
                        "predicted_class": target_class,
                        "prediction_probability": float(pred_proba[pred_class]),
                        "distance_from_original": float(np.linalg.norm(perturbed_instance - original_instance)),
                        "num_changes": len(changes)
                    })
        
        return counterfactuals
    
    def _minimal_change_perturbation(self, original: np.ndarray, training_data: pd.DataFrame, n_changes: int) -> np.ndarray:
        """Generate perturbation with minimal changes"""
        perturbed = original.copy()
        change_indices = np.random.choice(len(original), n_changes, replace=False)
        
        for idx in change_indices:
            if training_data.dtypes.iloc[idx] in ['int64', 'float64']:
                # Small numerical change
                std = training_data.iloc[:, idx].std()
                perturbed[idx] += np.random.normal(0, std * 0.1)
            else:
                # Random categorical replacement
                unique_vals = training_data.iloc[:, idx].unique()
                perturbed[idx] = np.random.choice(unique_vals)
        
        return perturbed
    
    def _realistic_perturbation(self, original: np.ndarray, training_data: pd.DataFrame) -> np.ndarray:
        """Generate realistic perturbation based on training data distribution"""
        # Sample a random training instance and blend with original
        random_instance = training_data.sample(1).values[0]
        blend_factor = np.random.uniform(0.1, 0.9)
        
        perturbed = original.copy()
        for i in range(len(original)):
            if training_data.dtypes.iloc[i] in ['int64', 'float64']:
                # Numerical: blend values
                perturbed[i] = original[i] * (1 - blend_factor) + random_instance[i] * blend_factor
            else:
                # Categorical: random choice between original and random
                if np.random.random() < blend_factor:
                    perturbed[i] = random_instance[i]
        
        return perturbed
    
    def _sparse_perturbation(self, original: np.ndarray, training_data: pd.DataFrame) -> np.ndarray:
        """Generate sparse perturbation (1-2 features)"""
        return self._minimal_change_perturbation(original, training_data, np.random.randint(1, 3))
    
    def _dense_perturbation(self, original: np.ndarray, training_data: pd.DataFrame) -> np.ndarray:
        """Generate dense perturbation (many small changes)"""
        perturbed = original.copy()
        n_changes = max(3, len(original) // 2)
        change_indices = np.random.choice(len(original), n_changes, replace=False)
        
        for idx in change_indices:
            if training_data.dtypes.iloc[idx] in ['int64', 'float64']:
                # Very small numerical change
                std = training_data.iloc[:, idx].std()
                perturbed[idx] += np.random.normal(0, std * 0.05)
            else:
                # Keep categorical unchanged for dense strategy
                pass
        
        return perturbed
    
    def _random_perturbation(self, original: np.ndarray, training_data: pd.DataFrame) -> np.ndarray:
        """Generate random perturbation"""
        perturbed = original.copy()
        n_changes = np.random.randint(1, min(5, len(original)))
        change_indices = np.random.choice(len(original), n_changes, replace=False)
        
        for idx in change_indices:
            if training_data.dtypes.iloc[idx] in ['int64', 'float64']:
                std = training_data.iloc[:, idx].std()
                perturbed[idx] += np.random.normal(0, std * 0.3)
            else:
                unique_vals = training_data.iloc[:, idx].unique()
                perturbed[idx] = np.random.choice(unique_vals)
        
        return perturbed
    
    def _calculate_counterfactual_quality(
        self,
        original: np.ndarray,
        counterfactual: List[float],
        model: BaseEstimator,
        training_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate quality metrics for counterfactual"""
        
        cf_array = np.array(counterfactual)
        
        # Distance metrics
        l2_distance = float(np.linalg.norm(cf_array - original))
        l1_distance = float(np.sum(np.abs(cf_array - original)))
        
        # Sparsity (number of changed features)
        sparsity = float(np.sum(cf_array != original))
        
        # Realism (distance to nearest training instance)
        distances_to_training = [
            np.linalg.norm(cf_array - training_data.iloc[i].values)
            for i in range(min(100, len(training_data)))  # Sample for performance
        ]
        realism_score = 1.0 / (1.0 + min(distances_to_training))
        
        # Prediction confidence
        if hasattr(model, 'predict_proba'):
            confidence = float(model.predict_proba([cf_array])[0].max())
        else:
            confidence = 1.0  # No confidence for regression
        
        return {
            "l2_distance": l2_distance,
            "l1_distance": l1_distance,
            "sparsity": sparsity,
            "realism_score": realism_score,
            "prediction_confidence": confidence,
            "overall_quality": (realism_score + (1.0 / (1.0 + l2_distance)) + confidence) / 3.0
        }
    
    def _interpret_anchor_explanation(self, anchor_features: List[Dict], precision: float) -> str:
        """Interpret anchor explanation results"""
        if not anchor_features:
            return "No anchor rules found for this instance."
        
        interpretation = f"Found anchor rule with {precision:.2%} precision. "
        interpretation += f"The prediction is stable when these {len(anchor_features)} conditions hold: "
        
        conditions = [af["feature_condition"] for af in anchor_features[:3]]
        interpretation += ", ".join(conditions)
        
        if len(anchor_features) > 3:
            interpretation += f" (and {len(anchor_features) - 3} other conditions)"
        
        return interpretation + "."
    
    def _interpret_prototype_explanations(
        self,
        prototypes: List[Dict],
        target_class: Optional[int],
        class_names: List[str]
    ) -> str:
        """Interpret prototype explanation results"""
        if not prototypes:
            return "No similar prototypes found in training data."
        
        class_name = class_names[target_class] if target_class is not None and class_names else "predicted class"
        
        interpretation = f"Found {len(prototypes)} similar instances from the training data with the same {class_name}. "
        
        # Analyze most common differing features
        all_diffs = []
        for proto in prototypes:
            all_diffs.extend([fd["feature"] for fd in proto["feature_differences"][:2]])
        
        from collections import Counter
        common_diffs = Counter(all_diffs).most_common(3)
        
        if common_diffs:
            diff_features = [feature for feature, count in common_diffs]
            interpretation += f"The most variable features compared to these prototypes are: {', '.join(diff_features)}."
        
        return interpretation
    
    def _interpret_ice_plot(
        self,
        ice_lines: List[Dict],
        pd_data: List[Dict],
        ice_variance: List[Dict],
        feature_name: str
    ) -> str:
        """Interpret ICE plot results"""
        # Analyze variance to detect interactions
        max_variance = max(iv["variance"] for iv in ice_variance)
        avg_variance = np.mean([iv["variance"] for iv in ice_variance])
        
        # Analyze PD curve shape
        pd_values = [pd["partial_dependence"] for pd in pd_data]
        pd_range = max(pd_values) - min(pd_values)
        
        interpretation = f"ICE plot analysis for {feature_name}: "
        
        if max_variance > 2 * avg_variance:
            interpretation += "Strong feature interactions detected - individual predictions vary significantly. "
        elif max_variance > 1.5 * avg_variance:
            interpretation += "Moderate feature interactions detected. "
        else:
            interpretation += "Low feature interactions - predictions are consistent across instances. "
        
        if pd_range > 0.1:
            interpretation += f"The feature has substantial impact on predictions (range: {pd_range:.3f}). "
        else:
            interpretation += "The feature has minimal impact on predictions. "
        
        return interpretation
    
    def _interpret_counterfactuals(
        self,
        counterfactuals: List[Dict],
        original_instance: np.ndarray,
        feature_names: List[str]
    ) -> str:
        """Interpret counterfactual explanation results"""
        if not counterfactuals:
            return "No valid counterfactuals could be generated for this instance."
        
        # Analyze common changes across counterfactuals
        all_changes = []
        for cf in counterfactuals:
            all_changes.extend([change["feature"] for change in cf["changes"]])
        
        from collections import Counter
        common_changes = Counter(all_changes).most_common(3)
        
        interpretation = f"Generated {len(counterfactuals)} counterfactual explanations. "
        
        if common_changes:
            important_features = [feature for feature, count in common_changes]
            interpretation += f"The most important features for changing the prediction are: {', '.join(important_features)}. "
        
        # Analyze average number of changes needed
        avg_changes = np.mean([cf["num_changes"] for cf in counterfactuals])
        interpretation += f"On average, {avg_changes:.1f} features need to be changed to flip the prediction."
        
        return interpretation
    
    async def cleanup_advanced_explainer(self, model_id: str) -> Dict[str, Any]:
        """
        Clean up advanced explainer resources for a model
        """
        try:
            removed_components = []
            
            for component_dict, component_name in [
                (self.explainers, "explainers"),
                (self.models, "models"),
                (self.feature_names, "feature_names"),
                (self.class_names, "class_names"),
                (self.training_data, "training_data"),
                (self.model_types, "model_types")
            ]:
                if model_id in component_dict:
                    del component_dict[model_id]
                    removed_components.append(component_name)
            
            return {
                "status": "success",
                "model_id": model_id,
                "removed_components": removed_components
            }
            
        except Exception as e:
            logger.error(f"Failed to cleanup advanced explainer: {e}")
            return {
                "status": "error",
                "message": str(e),
                "model_id": model_id
            }

# Global service instance
advanced_explainability_service = AdvancedExplainabilityService()