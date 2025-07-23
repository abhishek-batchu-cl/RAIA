"""
What-If Analysis and Decision Tree Extraction Service
Provides comprehensive scenario analysis and model introspection capabilities
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
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text, export_graphviz
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report
import joblib
import graphviz
from io import StringIO

# Additional analysis libraries
from scipy import stats
from scipy.optimize import minimize
from sklearn.neighbors import NearestNeighbors
from sklearn.inspection import partial_dependence
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

class WhatIfAnalysisService:
    """
    Comprehensive what-if analysis and decision tree extraction service
    """
    
    def __init__(self):
        self.models = {}
        self.surrogate_trees = {}
        self.feature_names = {}
        self.class_names = {}
        self.training_data = {}
        self.scalers = {}
        
    async def register_model_for_analysis(
        self,
        model_id: str,
        model: BaseEstimator,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        feature_names: List[str],
        class_names: Optional[List[str]] = None,
        model_type: str = "classification"
    ) -> Dict[str, Any]:
        """
        Register a model for what-if analysis
        
        Args:
            model_id: Unique identifier for the model
            model: Trained model object
            X_train: Training features
            y_train: Training targets
            feature_names: List of feature names
            class_names: List of class names (for classification)
            model_type: Type of model ("classification" or "regression")
        
        Returns:
            Dictionary with registration status
        """
        try:
            # Store model and metadata
            self.models[model_id] = model
            self.feature_names[model_id] = feature_names
            self.class_names[model_id] = class_names or []
            self.training_data[model_id] = {
                'X': X_train.copy(),
                'y': y_train.copy(),
                'model_type': model_type
            }
            
            # Create surrogate decision tree for interpretability
            surrogate_tree = await self._create_surrogate_tree(
                model, X_train, y_train, model_type
            )
            self.surrogate_trees[model_id] = surrogate_tree
            
            # Create feature scaler for what-if scenarios
            scaler = StandardScaler()
            scaler.fit(X_train)
            self.scalers[model_id] = scaler
            
            return {
                "status": "success",
                "model_id": model_id,
                "feature_count": len(feature_names),
                "training_samples": len(X_train),
                "model_type": model_type,
                "surrogate_accuracy": surrogate_tree.get('accuracy', 0.0) if surrogate_tree else 0.0
            }
            
        except Exception as e:
            logger.error(f"Failed to register model for what-if analysis: {e}")
            return {
                "status": "error",
                "message": str(e),
                "model_id": model_id
            }
    
    async def _create_surrogate_tree(
        self,
        model: BaseEstimator,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_type: str,
        max_depth: int = 10
    ) -> Dict[str, Any]:
        """
        Create a surrogate decision tree to approximate the original model
        """
        try:
            # Generate predictions from original model
            if model_type == "classification":
                y_pred = model.predict(X_train)
                surrogate = DecisionTreeClassifier(
                    max_depth=max_depth,
                    random_state=42,
                    min_samples_split=10,
                    min_samples_leaf=5
                )
            else:
                y_pred = model.predict(X_train)
                surrogate = DecisionTreeRegressor(
                    max_depth=max_depth,
                    random_state=42,
                    min_samples_split=10,
                    min_samples_leaf=5
                )
            
            # Train surrogate tree on original model's predictions
            surrogate.fit(X_train, y_pred)
            
            # Calculate surrogate accuracy
            surrogate_pred = surrogate.predict(X_train)
            if model_type == "classification":
                accuracy = accuracy_score(y_pred, surrogate_pred)
            else:
                accuracy = r2_score(y_pred, surrogate_pred)
            
            return {
                "tree": surrogate,
                "accuracy": accuracy,
                "depth": surrogate.get_depth(),
                "n_leaves": surrogate.get_n_leaves(),
                "n_nodes": surrogate.tree_.node_count
            }
            
        except Exception as e:
            logger.warning(f"Failed to create surrogate tree: {e}")
            return None
    
    async def perform_what_if_analysis(
        self,
        model_id: str,
        base_instance: Dict[str, Any],
        what_if_scenarios: List[Dict[str, Any]],
        include_confidence: bool = True,
        include_feature_importance: bool = True
    ) -> Dict[str, Any]:
        """
        Perform what-if analysis on multiple scenarios
        
        Args:
            model_id: Model identifier
            base_instance: Base instance for comparison
            what_if_scenarios: List of scenarios to analyze
            include_confidence: Whether to include prediction confidence
            include_feature_importance: Whether to include feature importance
        
        Returns:
            Dictionary with what-if analysis results
        """
        try:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")
            
            model = self.models[model_id]
            feature_names = self.feature_names[model_id]
            model_type = self.training_data[model_id]['model_type']
            
            # Convert base instance to DataFrame
            base_df = pd.DataFrame([base_instance])
            base_df = base_df[feature_names]  # Ensure correct order
            
            # Get base prediction
            base_prediction = self._get_model_prediction(model, base_df, model_type)
            
            # Analyze each scenario
            scenario_results = []
            for i, scenario in enumerate(what_if_scenarios):
                # Create scenario DataFrame
                scenario_instance = base_instance.copy()
                scenario_instance.update(scenario)
                scenario_df = pd.DataFrame([scenario_instance])
                scenario_df = scenario_df[feature_names]
                
                # Get scenario prediction
                scenario_prediction = self._get_model_prediction(model, scenario_df, model_type)
                
                # Calculate changes
                changes = []
                for feature in feature_names:
                    base_val = base_instance.get(feature, 0)
                    scenario_val = scenario_instance.get(feature, 0)
                    if base_val != scenario_val:
                        changes.append({
                            "feature": feature,
                            "base_value": base_val,
                            "scenario_value": scenario_val,
                            "change": scenario_val - base_val if isinstance(scenario_val, (int, float)) else "categorical",
                            "relative_change": ((scenario_val - base_val) / base_val * 100) if isinstance(scenario_val, (int, float)) and base_val != 0 else 0
                        })
                
                # Calculate prediction change
                pred_change = self._calculate_prediction_change(
                    base_prediction, scenario_prediction, model_type
                )
                
                # Feature importance for this scenario (if requested)
                feature_importance = None
                if include_feature_importance:
                    feature_importance = await self._calculate_scenario_feature_importance(
                        model, scenario_df, feature_names, model_type
                    )
                
                scenario_results.append({
                    "scenario_id": i + 1,
                    "scenario_name": scenario.get("_name", f"Scenario {i + 1}"),
                    "changes": changes,
                    "prediction": scenario_prediction,
                    "prediction_change": pred_change,
                    "feature_importance": feature_importance,
                    "impact_summary": self._summarize_scenario_impact(changes, pred_change)
                })
            
            # Generate overall analysis
            overall_analysis = self._analyze_scenario_patterns(scenario_results, model_type)
            
            # Create sensitivity analysis
            sensitivity_analysis = await self._perform_sensitivity_analysis(
                model, base_df, feature_names, model_type
            )
            
            return {
                "status": "success",
                "model_id": model_id,
                "base_instance": base_instance,
                "base_prediction": base_prediction,
                "scenario_results": scenario_results,
                "overall_analysis": overall_analysis,
                "sensitivity_analysis": sensitivity_analysis,
                "metadata": {
                    "num_scenarios": len(what_if_scenarios),
                    "model_type": model_type,
                    "features_analyzed": len(feature_names)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to perform what-if analysis: {e}")
            return {
                "status": "error",
                "message": str(e),
                "model_id": model_id
            }
    
    async def extract_decision_rules(
        self,
        model_id: str,
        max_rules: int = 20,
        use_surrogate: bool = True,
        min_support: float = 0.01
    ) -> Dict[str, Any]:
        """
        Extract decision rules from the model
        
        Args:
            model_id: Model identifier
            max_rules: Maximum number of rules to extract
            use_surrogate: Whether to use surrogate tree for rule extraction
            min_support: Minimum support for extracted rules
        
        Returns:
            Dictionary with extracted decision rules
        """
        try:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")
            
            model = self.models[model_id]
            feature_names = self.feature_names[model_id]
            class_names = self.class_names[model_id]
            training_data = self.training_data[model_id]
            
            rules = []
            
            # Extract rules from surrogate tree if available
            if use_surrogate and model_id in self.surrogate_trees:
                surrogate_info = self.surrogate_trees[model_id]
                if surrogate_info:
                    surrogate_tree = surrogate_info['tree']
                    
                    # Extract text representation
                    tree_text = export_text(
                        surrogate_tree,
                        feature_names=feature_names,
                        class_names=class_names
                    )
                    
                    # Parse tree rules
                    tree_rules = self._parse_tree_rules(
                        surrogate_tree, feature_names, class_names, min_support
                    )
                    rules.extend(tree_rules)
                    
                    # Generate graphviz visualization
                    dot_data = export_graphviz(
                        surrogate_tree,
                        feature_names=feature_names,
                        class_names=class_names,
                        filled=True,
                        rounded=True,
                        special_characters=True
                    )
                    
                    surrogate_viz = {
                        "text_representation": tree_text,
                        "graphviz_dot": dot_data,
                        "accuracy": surrogate_info['accuracy'],
                        "depth": surrogate_info['depth'],
                        "n_leaves": surrogate_info['n_leaves']
                    }
                else:
                    surrogate_viz = None
            else:
                surrogate_viz = None
            
            # Extract rules from tree-based models directly
            if hasattr(model, 'tree_') or hasattr(model, 'estimators_'):
                direct_rules = self._extract_direct_tree_rules(
                    model, feature_names, class_names, min_support
                )
                rules.extend(direct_rules)
            
            # Generate rule-based insights
            rule_insights = self._analyze_decision_rules(rules, training_data)
            
            # Create rule coverage analysis
            coverage_analysis = self._analyze_rule_coverage(
                rules, training_data['X'], training_data['y']
            )
            
            # Limit to max_rules
            rules = sorted(rules, key=lambda x: x.get('confidence', 0), reverse=True)[:max_rules]
            
            return {
                "status": "success",
                "model_id": model_id,
                "decision_rules": rules,
                "rule_insights": rule_insights,
                "coverage_analysis": coverage_analysis,
                "surrogate_visualization": surrogate_viz,
                "metadata": {
                    "total_rules_extracted": len(rules),
                    "min_support_threshold": min_support,
                    "extraction_method": "surrogate" if use_surrogate else "direct"
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to extract decision rules: {e}")
            return {
                "status": "error",
                "message": str(e),
                "model_id": model_id
            }
    
    async def perform_optimization_analysis(
        self,
        model_id: str,
        base_instance: Dict[str, Any],
        objective: str = "maximize",
        target_feature: Optional[str] = None,
        constraints: Optional[Dict[str, Dict[str, float]]] = None,
        optimization_steps: int = 100
    ) -> Dict[str, Any]:
        """
        Perform optimization analysis to find optimal feature values
        
        Args:
            model_id: Model identifier
            base_instance: Starting point for optimization
            objective: "maximize", "minimize", or "target"
            target_feature: Target feature for optimization
            constraints: Feature constraints {feature: {"min": val, "max": val}}
            optimization_steps: Number of optimization steps
        
        Returns:
            Dictionary with optimization results
        """
        try:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")
            
            model = self.models[model_id]
            feature_names = self.feature_names[model_id]
            training_data = self.training_data[model_id]
            model_type = training_data['model_type']
            
            # Set up optimization bounds
            bounds = []
            feature_ranges = {}
            
            for feature in feature_names:
                if constraints and feature in constraints:
                    # Use user-defined constraints
                    min_val = constraints[feature].get("min")
                    max_val = constraints[feature].get("max")
                else:
                    # Use training data range
                    col_data = training_data['X'][feature]
                    if col_data.dtype in ['int64', 'float64']:
                        min_val = float(col_data.min())
                        max_val = float(col_data.max())
                    else:
                        # Categorical - use mode
                        min_val = max_val = float(col_data.mode().iloc[0])
                
                bounds.append((min_val, max_val))
                feature_ranges[feature] = {"min": min_val, "max": max_val}
            
            # Define objective function
            def objective_function(x):
                instance_dict = {feature: val for feature, val in zip(feature_names, x)}
                instance_df = pd.DataFrame([instance_dict])
                
                if model_type == "classification":
                    if hasattr(model, 'predict_proba'):
                        prob = model.predict_proba(instance_df)[0]
                        if objective == "maximize":
                            return -prob.max()  # Negative for maximization
                        else:
                            return prob.max()
                    else:
                        pred = model.predict(instance_df)[0]
                        return -pred if objective == "maximize" else pred
                else:
                    pred = model.predict(instance_df)[0]
                    return -pred if objective == "maximize" else pred
            
            # Initial point
            x0 = [base_instance.get(feature, 0) for feature in feature_names]
            
            # Perform optimization
            result = minimize(
                objective_function,
                x0,
                bounds=bounds,
                method='L-BFGS-B',
                options={'maxiter': optimization_steps}
            )
            
            # Extract optimal values
            optimal_instance = {
                feature: float(val) for feature, val in zip(feature_names, result.x)
            }
            
            # Calculate changes from base instance
            changes = []
            for feature in feature_names:
                base_val = base_instance.get(feature, 0)
                optimal_val = optimal_instance[feature]
                if abs(optimal_val - base_val) > 1e-6:
                    changes.append({
                        "feature": feature,
                        "base_value": base_val,
                        "optimal_value": optimal_val,
                        "change": optimal_val - base_val,
                        "relative_change": ((optimal_val - base_val) / base_val * 100) if base_val != 0 else 0
                    })
            
            # Get predictions for base and optimal instances
            base_df = pd.DataFrame([base_instance])[feature_names]
            optimal_df = pd.DataFrame([optimal_instance])
            
            base_pred = self._get_model_prediction(model, base_df, model_type)
            optimal_pred = self._get_model_prediction(model, optimal_df, model_type)
            
            # Calculate improvement
            improvement = self._calculate_prediction_change(base_pred, optimal_pred, model_type)
            
            # Generate optimization path (sample points along the path)
            optimization_path = []
            for i in range(0, optimization_steps, optimization_steps // 10):
                if i < len(result.func) if hasattr(result, 'func') else 0:
                    # This would require storing intermediate results
                    # For now, we'll create a simplified path
                    alpha = i / optimization_steps
                    interpolated = {
                        feature: base_instance.get(feature, 0) * (1 - alpha) + optimal_instance[feature] * alpha
                        for feature in feature_names
                    }
                    interpolated_df = pd.DataFrame([interpolated])
                    interpolated_pred = self._get_model_prediction(model, interpolated_df, model_type)
                    optimization_path.append({
                        "step": i,
                        "values": interpolated,
                        "prediction": interpolated_pred
                    })
            
            return {
                "status": "success",
                "model_id": model_id,
                "optimization_objective": objective,
                "base_instance": base_instance,
                "optimal_instance": optimal_instance,
                "base_prediction": base_pred,
                "optimal_prediction": optimal_pred,
                "improvement": improvement,
                "changes": changes,
                "optimization_path": optimization_path,
                "convergence_info": {
                    "success": result.success,
                    "iterations": result.nit,
                    "function_evaluations": result.nfev,
                    "final_objective_value": float(result.fun)
                },
                "feature_ranges": feature_ranges
            }
            
        except Exception as e:
            logger.error(f"Failed to perform optimization analysis: {e}")
            return {
                "status": "error",
                "message": str(e),
                "model_id": model_id
            }
    
    def _get_model_prediction(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        model_type: str
    ) -> Dict[str, Any]:
        """Get structured model prediction"""
        
        if model_type == "classification":
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)[0]
                predicted_class = int(np.argmax(proba))
                class_names = self.class_names.get(id(model), [])
                
                return {
                    "type": "classification",
                    "predicted_class": predicted_class,
                    "class_name": class_names[predicted_class] if class_names else f"Class_{predicted_class}",
                    "probabilities": [float(p) for p in proba],
                    "confidence": float(proba[predicted_class])
                }
            else:
                prediction = model.predict(X)[0]
                return {
                    "type": "classification",
                    "predicted_class": int(prediction),
                    "class_name": f"Class_{int(prediction)}",
                    "confidence": 1.0
                }
        else:
            prediction = model.predict(X)[0]
            return {
                "type": "regression",
                "predicted_value": float(prediction)
            }
    
    def _calculate_prediction_change(
        self,
        base_pred: Dict[str, Any],
        scenario_pred: Dict[str, Any],
        model_type: str
    ) -> Dict[str, Any]:
        """Calculate change between predictions"""
        
        if model_type == "classification":
            base_conf = base_pred.get("confidence", 0)
            scenario_conf = scenario_pred.get("confidence", 0)
            
            class_changed = base_pred.get("predicted_class") != scenario_pred.get("predicted_class")
            confidence_change = scenario_conf - base_conf
            
            return {
                "class_changed": class_changed,
                "confidence_change": confidence_change,
                "base_class": base_pred.get("class_name", "Unknown"),
                "scenario_class": scenario_pred.get("class_name", "Unknown"),
                "magnitude": "high" if abs(confidence_change) > 0.3 else "medium" if abs(confidence_change) > 0.1 else "low"
            }
        else:
            base_val = base_pred.get("predicted_value", 0)
            scenario_val = scenario_pred.get("predicted_value", 0)
            
            absolute_change = scenario_val - base_val
            relative_change = (absolute_change / base_val * 100) if base_val != 0 else 0
            
            return {
                "absolute_change": absolute_change,
                "relative_change": relative_change,
                "base_value": base_val,
                "scenario_value": scenario_val,
                "magnitude": "high" if abs(relative_change) > 20 else "medium" if abs(relative_change) > 5 else "low"
            }
    
    async def _calculate_scenario_feature_importance(
        self,
        model: BaseEstimator,
        instance_df: pd.DataFrame,
        feature_names: List[str],
        model_type: str
    ) -> List[Dict[str, Any]]:
        """Calculate feature importance for specific scenario"""
        
        importance_scores = []
        base_pred = self._get_model_prediction(model, instance_df, model_type)
        
        for feature in feature_names:
            # Create perturbed instance
            perturbed_df = instance_df.copy()
            original_val = perturbed_df[feature].iloc[0]
            
            # Perturb feature (use mean from training data)
            if instance_df[feature].dtype in ['int64', 'float64']:
                perturbed_df[feature] = perturbed_df[feature].mean()
            else:
                # For categorical, use mode or different category
                perturbed_df[feature] = "OTHER_CATEGORY"
            
            # Get prediction change
            perturbed_pred = self._get_model_prediction(model, perturbed_df, model_type)
            
            if model_type == "classification":
                importance = abs(
                    perturbed_pred.get("confidence", 0) - base_pred.get("confidence", 0)
                )
            else:
                base_val = base_pred.get("predicted_value", 0)
                perturbed_val = perturbed_pred.get("predicted_value", 0)
                importance = abs(perturbed_val - base_val)
            
            importance_scores.append({
                "feature": feature,
                "importance": importance,
                "original_value": original_val
            })
        
        # Sort by importance
        importance_scores.sort(key=lambda x: x["importance"], reverse=True)
        
        # Add ranks
        for i, score in enumerate(importance_scores):
            score["rank"] = i + 1
        
        return importance_scores
    
    def _summarize_scenario_impact(
        self,
        changes: List[Dict[str, Any]],
        pred_change: Dict[str, Any]
    ) -> str:
        """Generate summary of scenario impact"""
        
        if not changes:
            return "No feature changes made."
        
        summary = f"Changed {len(changes)} features. "
        
        # Identify major changes
        major_changes = [c for c in changes if abs(c.get("relative_change", 0)) > 20]
        if major_changes:
            summary += f"Major changes in {len(major_changes)} features: "
            summary += ", ".join([c["feature"] for c in major_changes[:3]])
            if len(major_changes) > 3:
                summary += f" and {len(major_changes) - 3} others"
            summary += ". "
        
        # Add prediction impact
        magnitude = pred_change.get("magnitude", "low")
        if pred_change.get("class_changed"):
            summary += f"Prediction class changed with {magnitude} confidence impact."
        else:
            summary += f"Prediction confidence changed with {magnitude} magnitude."
        
        return summary
    
    def _analyze_scenario_patterns(
        self,
        scenario_results: List[Dict[str, Any]],
        model_type: str
    ) -> Dict[str, Any]:
        """Analyze patterns across scenarios"""
        
        # Feature change frequency
        feature_change_counts = {}
        for scenario in scenario_results:
            for change in scenario["changes"]:
                feature = change["feature"]
                feature_change_counts[feature] = feature_change_counts.get(feature, 0) + 1
        
        most_changed_features = sorted(
            feature_change_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        # Impact distribution
        if model_type == "classification":
            class_changes = sum(1 for s in scenario_results 
                              if s["prediction_change"].get("class_changed", False))
            high_impact = sum(1 for s in scenario_results 
                            if s["prediction_change"].get("magnitude") == "high")
        else:
            class_changes = 0
            high_impact = sum(1 for s in scenario_results 
                            if s["prediction_change"].get("magnitude") == "high")
        
        return {
            "most_changed_features": [
                {"feature": feature, "change_frequency": count} 
                for feature, count in most_changed_features
            ],
            "impact_distribution": {
                "class_changes": class_changes,
                "high_impact_scenarios": high_impact,
                "total_scenarios": len(scenario_results),
                "change_rate": class_changes / len(scenario_results) if scenario_results else 0
            }
        }
    
    async def _perform_sensitivity_analysis(
        self,
        model: BaseEstimator,
        base_df: pd.DataFrame,
        feature_names: List[str],
        model_type: str,
        perturbation_size: float = 0.1
    ) -> Dict[str, Any]:
        """Perform sensitivity analysis around base instance"""
        
        sensitivities = []
        base_pred = self._get_model_prediction(model, base_df, model_type)
        
        for feature in feature_names:
            if base_df[feature].dtype in ['int64', 'float64']:
                original_val = base_df[feature].iloc[0]
                
                # Test positive and negative perturbations
                perturbations = [
                    original_val * (1 + perturbation_size),
                    original_val * (1 - perturbation_size)
                ]
                
                max_sensitivity = 0
                for perturbed_val in perturbations:
                    perturbed_df = base_df.copy()
                    perturbed_df[feature] = perturbed_val
                    
                    perturbed_pred = self._get_model_prediction(model, perturbed_df, model_type)
                    
                    if model_type == "classification":
                        sensitivity = abs(
                            perturbed_pred.get("confidence", 0) - base_pred.get("confidence", 0)
                        )
                    else:
                        base_val = base_pred.get("predicted_value", 0)
                        perturbed_val_pred = perturbed_pred.get("predicted_value", 0)
                        sensitivity = abs(perturbed_val_pred - base_val)
                    
                    max_sensitivity = max(max_sensitivity, sensitivity)
                
                sensitivities.append({
                    "feature": feature,
                    "sensitivity": max_sensitivity,
                    "feature_value": original_val
                })
        
        # Sort by sensitivity
        sensitivities.sort(key=lambda x: x["sensitivity"], reverse=True)
        
        # Add ranks and categories
        for i, sens in enumerate(sensitivities):
            sens["rank"] = i + 1
            sens["sensitivity_category"] = (
                "high" if sens["sensitivity"] > 0.1 else
                "medium" if sens["sensitivity"] > 0.05 else
                "low"
            )
        
        return {
            "feature_sensitivities": sensitivities,
            "most_sensitive_features": [s["feature"] for s in sensitivities[:5]],
            "perturbation_size": perturbation_size
        }
    
    def _parse_tree_rules(
        self,
        tree: DecisionTreeClassifier,
        feature_names: List[str],
        class_names: List[str],
        min_support: float
    ) -> List[Dict[str, Any]]:
        """Parse rules from decision tree"""
        
        rules = []
        
        # Get tree structure
        n_nodes = tree.tree_.node_count
        children_left = tree.tree_.children_left
        children_right = tree.tree_.children_right
        feature = tree.tree_.feature
        threshold = tree.tree_.threshold
        value = tree.tree_.value
        n_node_samples = tree.tree_.n_node_samples
        
        def extract_rules_recursive(node_id, conditions, support):
            # Check if leaf node
            if children_left[node_id] != children_right[node_id]:
                # Internal node - add conditions and recurse
                feature_name = feature_names[feature[node_id]]
                thresh = threshold[node_id]
                
                # Left child (<=)
                left_conditions = conditions + [f"{feature_name} <= {thresh:.3f}"]
                left_support = n_node_samples[children_left[node_id]] / n_node_samples[0]
                extract_rules_recursive(children_left[node_id], left_conditions, left_support)
                
                # Right child (>)
                right_conditions = conditions + [f"{feature_name} > {thresh:.3f}"]
                right_support = n_node_samples[children_right[node_id]] / n_node_samples[0]
                extract_rules_recursive(children_right[node_id], right_conditions, right_support)
            else:
                # Leaf node - create rule
                if support >= min_support:
                    # Get class prediction
                    class_counts = value[node_id][0]
                    predicted_class = np.argmax(class_counts)
                    confidence = class_counts[predicted_class] / np.sum(class_counts)
                    
                    class_name = class_names[predicted_class] if class_names else f"Class_{predicted_class}"
                    
                    rule = {
                        "rule_id": len(rules),
                        "conditions": conditions,
                        "prediction": class_name,
                        "confidence": float(confidence),
                        "support": float(support),
                        "rule_text": f"IF {' AND '.join(conditions)} THEN {class_name} (confidence: {confidence:.3f})"
                    }
                    rules.append(rule)
        
        # Start extraction from root
        extract_rules_recursive(0, [], 1.0)
        
        return rules
    
    def _extract_direct_tree_rules(
        self,
        model: BaseEstimator,
        feature_names: List[str],
        class_names: List[str],
        min_support: float
    ) -> List[Dict[str, Any]]:
        """Extract rules directly from tree-based models"""
        
        rules = []
        
        if hasattr(model, 'tree_'):
            # Single decision tree
            tree_rules = self._parse_tree_rules(model, feature_names, class_names, min_support)
            rules.extend(tree_rules)
        
        elif hasattr(model, 'estimators_'):
            # Ensemble model - extract from first few trees
            max_trees = min(5, len(model.estimators_))
            for i, estimator in enumerate(model.estimators_[:max_trees]):
                tree_rules = self._parse_tree_rules(
                    estimator, feature_names, class_names, min_support
                )
                # Add tree identifier to rules
                for rule in tree_rules:
                    rule['source_tree'] = i
                    rule['rule_id'] = f"tree_{i}_{rule['rule_id']}"
                rules.extend(tree_rules)
        
        return rules
    
    def _analyze_decision_rules(
        self,
        rules: List[Dict[str, Any]],
        training_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze extracted decision rules"""
        
        if not rules:
            return {"message": "No rules extracted"}
        
        # Rule statistics
        avg_conditions = np.mean([len(rule["conditions"]) for rule in rules])
        avg_confidence = np.mean([rule["confidence"] for rule in rules])
        avg_support = np.mean([rule["support"] for rule in rules])
        
        # Feature usage frequency
        feature_usage = {}
        for rule in rules:
            for condition in rule["conditions"]:
                feature = condition.split()[0]  # First word is feature name
                feature_usage[feature] = feature_usage.get(feature, 0) + 1
        
        most_used_features = sorted(
            feature_usage.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        # Rule complexity distribution
        complexity_dist = {}
        for rule in rules:
            n_conditions = len(rule["conditions"])
            complexity_dist[n_conditions] = complexity_dist.get(n_conditions, 0) + 1
        
        return {
            "rule_statistics": {
                "total_rules": len(rules),
                "avg_conditions_per_rule": float(avg_conditions),
                "avg_confidence": float(avg_confidence),
                "avg_support": float(avg_support)
            },
            "feature_usage": [
                {"feature": feature, "usage_count": count}
                for feature, count in most_used_features
            ],
            "complexity_distribution": [
                {"num_conditions": n_cond, "rule_count": count}
                for n_cond, count in sorted(complexity_dist.items())
            ],
            "high_confidence_rules": [
                rule for rule in rules if rule["confidence"] > 0.9
            ][:5]
        }
    
    def _analyze_rule_coverage(
        self,
        rules: List[Dict[str, Any]],
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, Any]:
        """Analyze how well rules cover the training data"""
        
        if not rules:
            return {"message": "No rules to analyze coverage"}
        
        total_samples = len(X)
        covered_samples = set()
        
        rule_coverage = []
        for rule in rules:
            # Count samples covered by this rule
            rule_covered = 0
            for idx in range(len(X)):
                # Check if instance satisfies all conditions
                satisfies_rule = True
                for condition in rule["conditions"]:
                    if not self._check_condition(X.iloc[idx], condition):
                        satisfies_rule = False
                        break
                
                if satisfies_rule:
                    rule_covered += 1
                    covered_samples.add(idx)
            
            rule_coverage.append({
                "rule_id": rule["rule_id"],
                "samples_covered": rule_covered,
                "coverage_percentage": (rule_covered / total_samples) * 100
            })
        
        return {
            "total_samples": total_samples,
            "samples_covered_by_any_rule": len(covered_samples),
            "overall_coverage_percentage": (len(covered_samples) / total_samples) * 100,
            "rule_coverage": rule_coverage
        }
    
    def _check_condition(self, instance: pd.Series, condition: str) -> bool:
        """Check if instance satisfies a condition"""
        try:
            parts = condition.split()
            if len(parts) >= 3:
                feature = parts[0]
                operator = parts[1]
                threshold = float(parts[2])
                
                if feature in instance.index:
                    value = instance[feature]
                    if operator == "<=":
                        return value <= threshold
                    elif operator == ">":
                        return value > threshold
                    elif operator == "==":
                        return value == threshold
                    elif operator == "!=":
                        return value != threshold
        except:
            pass
        
        return False
    
    async def cleanup_model(self, model_id: str) -> Dict[str, Any]:
        """
        Clean up model resources
        """
        try:
            removed_components = []
            
            for component_dict, component_name in [
                (self.models, "models"),
                (self.surrogate_trees, "surrogate_trees"),
                (self.feature_names, "feature_names"),
                (self.class_names, "class_names"),
                (self.training_data, "training_data"),
                (self.scalers, "scalers")
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
            logger.error(f"Failed to cleanup what-if analysis model: {e}")
            return {
                "status": "error",
                "message": str(e),
                "model_id": model_id
            }

# Global service instance
what_if_analysis_service = WhatIfAnalysisService()