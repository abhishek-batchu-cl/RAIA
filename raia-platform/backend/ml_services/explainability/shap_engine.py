# SHAP Explainability Engine - Core SHAP Implementation
import shap
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
import joblib
import pickle
from pathlib import Path
import json
from datetime import datetime
import logging
from ..exceptions import ModelLoadError, ExplainabilityError, ValidationError

logger = logging.getLogger(__name__)

class SHAPExplainabilityEngine:
    """Core SHAP explainability engine for model interpretability"""
    
    def __init__(self, model_storage_path: str = None):
        self.model_storage_path = model_storage_path or "models/"
        self.explainer_cache = {}
        
        # Initialize SHAP with specific configurations
        shap.initjs()
        
    def load_model(self, model_path: str, model_type: str = 'scikit-learn'):
        """Load ML model for SHAP analysis"""
        try:
            if model_type == 'scikit-learn':
                return joblib.load(model_path)
            elif model_type == 'pickle':
                with open(model_path, 'rb') as f:
                    return pickle.load(f)
            elif model_type == 'tensorflow':
                import tensorflow as tf
                return tf.keras.models.load_model(model_path)
            elif model_type == 'pytorch':
                import torch
                return torch.load(model_path)
            else:
                raise ModelLoadError(f"Unsupported model type: {model_type}")
        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {str(e)}")
    
    def create_explainer(self, 
                        model: Any, 
                        background_data: pd.DataFrame,
                        model_type: str = 'tabular',
                        feature_names: List[str] = None) -> shap.Explainer:
        """Create appropriate SHAP explainer based on model and data type"""
        
        try:
            # Convert background data to numpy if it's a DataFrame
            if isinstance(background_data, pd.DataFrame):
                X_background = background_data.values
                feature_names = feature_names or list(background_data.columns)
            else:
                X_background = background_data
            
            # Choose appropriate explainer based on model type
            if hasattr(model, 'predict_proba'):
                # For classification models with probability output
                if model_type in ['tree', 'ensemble']:
                    # Tree-based models (Random Forest, XGBoost, etc.)
                    explainer = shap.TreeExplainer(model)
                else:
                    # General classifier with probability
                    explainer = shap.Explainer(model.predict_proba, X_background, feature_names=feature_names)
            
            elif hasattr(model, 'predict'):
                # For regression models or classifiers without probability
                if model_type in ['tree', 'ensemble']:
                    explainer = shap.TreeExplainer(model)
                else:
                    explainer = shap.Explainer(model.predict, X_background, feature_names=feature_names)
            
            elif model_type == 'deep':
                # Deep learning models
                explainer = shap.DeepExplainer(model, X_background)
                
            elif model_type == 'kernel':
                # Model-agnostic kernel explainer (slower but works with any model)
                explainer = shap.KernelExplainer(model.predict, X_background)
                
            else:
                # Default to Kernel explainer for unknown model types
                logger.warning(f"Unknown model type {model_type}, using KernelExplainer")
                explainer = shap.KernelExplainer(model.predict, X_background)
            
            return explainer
            
        except Exception as e:
            raise ExplainabilityError(f"Failed to create SHAP explainer: {str(e)}")
    
    def calculate_shap_values(self,
                             model_id: str,
                             model: Any,
                             instances: Union[pd.DataFrame, np.ndarray],
                             background_data: pd.DataFrame,
                             model_type: str = 'tabular',
                             feature_names: List[str] = None,
                             max_evals: int = 1000) -> Dict[str, Any]:
        """Calculate SHAP values for given instances"""
        
        # Create or get cached explainer
        cache_key = f"{model_id}_{model_type}"
        if cache_key not in self.explainer_cache:
            self.explainer_cache[cache_key] = self.create_explainer(
                model, background_data, model_type, feature_names
            )
        
        explainer = self.explainer_cache[cache_key]
        
        try:
            # Convert instances to appropriate format
            if isinstance(instances, pd.DataFrame):
                X_explain = instances.values
                feature_names = feature_names or list(instances.columns)
            else:
                X_explain = instances
            
            # Calculate SHAP values
            if isinstance(explainer, shap.TreeExplainer):
                # Tree explainers are fast and don't need max_evals
                shap_values = explainer.shap_values(X_explain)
            else:
                # Other explainers might be slower, use max_evals
                shap_values = explainer.shap_values(X_explain, nsamples=max_evals)
            
            # Handle different output formats
            if isinstance(shap_values, list):
                # Multi-class classification
                # Convert to format expected by frontend
                shap_results = {}
                for class_idx, class_shap_values in enumerate(shap_values):
                    shap_results[f'class_{class_idx}'] = self._format_shap_values(
                        class_shap_values, feature_names, instances
                    )
            else:
                # Binary classification or regression
                shap_results = self._format_shap_values(shap_values, feature_names, instances)
            
            # Calculate additional SHAP metrics
            explanation_metrics = self._calculate_explanation_metrics(shap_values, X_explain)
            
            # Generate summary statistics
            summary_stats = self._generate_shap_summary(shap_values, feature_names)
            
            return {
                'model_id': model_id,
                'shap_values': shap_results,
                'expected_value': float(explainer.expected_value) if hasattr(explainer, 'expected_value') else None,
                'feature_names': feature_names,
                'explanation_metrics': explanation_metrics,
                'summary_statistics': summary_stats,
                'calculation_timestamp': datetime.utcnow().isoformat(),
                'explainer_type': type(explainer).__name__,
                'num_instances': len(X_explain),
                'num_features': X_explain.shape[1] if len(X_explain.shape) > 1 else 1
            }
            
        except Exception as e:
            raise ExplainabilityError(f"Failed to calculate SHAP values: {str(e)}")
    
    def _format_shap_values(self, 
                           shap_values: np.ndarray, 
                           feature_names: List[str], 
                           instances: Union[pd.DataFrame, np.ndarray]) -> Dict[str, Any]:
        """Format SHAP values for frontend consumption"""
        
        if len(shap_values.shape) == 1:
            # Single instance
            return {
                'instance_0': {
                    'feature_contributions': [
                        {
                            'feature_name': feature_names[i] if feature_names else f'feature_{i}',
                            'feature_value': float(instances[i]) if hasattr(instances, '__getitem__') else None,
                            'shap_value': float(shap_values[i]),
                            'abs_shap_value': float(abs(shap_values[i])),
                            'contribution_type': 'positive' if shap_values[i] > 0 else 'negative'
                        }
                        for i in range(len(shap_values))
                    ],
                    'total_contribution': float(np.sum(shap_values))
                }
            }
        else:
            # Multiple instances
            formatted_results = {}
            for instance_idx in range(shap_values.shape[0]):
                instance_shap = shap_values[instance_idx]
                instance_data = instances[instance_idx] if hasattr(instances, '__getitem__') else None
                
                formatted_results[f'instance_{instance_idx}'] = {
                    'feature_contributions': [
                        {
                            'feature_name': feature_names[i] if feature_names else f'feature_{i}',
                            'feature_value': float(instance_data[i]) if instance_data is not None else None,
                            'shap_value': float(instance_shap[i]),
                            'abs_shap_value': float(abs(instance_shap[i])),
                            'contribution_type': 'positive' if instance_shap[i] > 0 else 'negative'
                        }
                        for i in range(len(instance_shap))
                    ],
                    'total_contribution': float(np.sum(instance_shap))
                }
            
            return formatted_results
    
    def _calculate_explanation_metrics(self, shap_values: np.ndarray, instances: np.ndarray) -> Dict[str, Any]:
        """Calculate additional metrics about the explanations"""
        
        if isinstance(shap_values, list):
            # Multi-class case - use first class for general metrics
            shap_array = shap_values[0]
        else:
            shap_array = shap_values
        
        if len(shap_array.shape) == 1:
            shap_array = shap_array.reshape(1, -1)
        
        metrics = {}
        
        # Feature importance ranking
        mean_abs_shap = np.mean(np.abs(shap_array), axis=0)
        feature_importance_ranking = np.argsort(mean_abs_shap)[::-1]
        
        metrics['feature_importance_ranking'] = feature_importance_ranking.tolist()
        metrics['mean_absolute_shap_values'] = mean_abs_shap.tolist()
        
        # Explanation complexity (how many features have significant impact)
        significant_features = np.sum(mean_abs_shap > np.mean(mean_abs_shap) * 0.1)
        metrics['explanation_complexity'] = int(significant_features)
        
        # Explanation concentration (how concentrated the explanation is)
        # Higher values mean fewer features explain most of the prediction
        sorted_abs_shap = np.sort(mean_abs_shap)[::-1]
        cumulative_shap = np.cumsum(sorted_abs_shap)
        total_shap = np.sum(mean_abs_shap)
        
        # Find how many top features explain 80% of the total contribution
        features_for_80_percent = np.argmax(cumulative_shap >= 0.8 * total_shap) + 1
        metrics['features_for_80_percent_explanation'] = int(features_for_80_percent)
        
        # Stability metrics (for multiple instances)
        if shap_array.shape[0] > 1:
            # Coefficient of variation for each feature across instances
            feature_stability = np.std(shap_array, axis=0) / (np.mean(np.abs(shap_array), axis=0) + 1e-8)
            metrics['feature_stability'] = feature_stability.tolist()
            metrics['avg_feature_stability'] = float(np.mean(feature_stability))
        
        return metrics
    
    def _generate_shap_summary(self, shap_values: np.ndarray, feature_names: List[str] = None) -> Dict[str, Any]:
        """Generate summary statistics for SHAP values"""
        
        if isinstance(shap_values, list):
            # Multi-class case
            summary = {'type': 'multiclass', 'num_classes': len(shap_values)}
            
            for class_idx, class_shap in enumerate(shap_values):
                if len(class_shap.shape) == 1:
                    class_shap = class_shap.reshape(1, -1)
                
                class_summary = self._calculate_single_class_summary(class_shap, feature_names)
                summary[f'class_{class_idx}'] = class_summary
        else:
            # Binary classification or regression
            if len(shap_values.shape) == 1:
                shap_values = shap_values.reshape(1, -1)
            
            summary = self._calculate_single_class_summary(shap_values, feature_names)
            summary['type'] = 'single_class'
        
        return summary
    
    def _calculate_single_class_summary(self, shap_values: np.ndarray, feature_names: List[str] = None) -> Dict[str, Any]:
        """Calculate summary for a single class or regression"""
        
        # Overall statistics
        mean_shap = np.mean(shap_values, axis=0)
        std_shap = np.std(shap_values, axis=0)
        min_shap = np.min(shap_values, axis=0)
        max_shap = np.max(shap_values, axis=0)
        
        # Feature importance (mean absolute SHAP values)
        feature_importance = np.mean(np.abs(shap_values), axis=0)
        
        # Create feature summary
        feature_summaries = []
        for i in range(len(mean_shap)):
            feature_name = feature_names[i] if feature_names else f'feature_{i}'
            feature_summaries.append({
                'feature_name': feature_name,
                'importance': float(feature_importance[i]),
                'mean_contribution': float(mean_shap[i]),
                'std_contribution': float(std_shap[i]),
                'min_contribution': float(min_shap[i]),
                'max_contribution': float(max_shap[i]),
                'avg_positive_contribution': float(np.mean(shap_values[shap_values[:, i] > 0, i])) if np.any(shap_values[:, i] > 0) else 0.0,
                'avg_negative_contribution': float(np.mean(shap_values[shap_values[:, i] < 0, i])) if np.any(shap_values[:, i] < 0) else 0.0,
                'contribution_frequency': float(np.mean(np.abs(shap_values[:, i]) > 0.001))  # How often this feature has meaningful contribution
            })
        
        # Sort by importance
        feature_summaries.sort(key=lambda x: x['importance'], reverse=True)
        
        return {
            'feature_summaries': feature_summaries,
            'top_5_features': feature_summaries[:5],
            'total_absolute_contribution': float(np.sum(feature_importance)),
            'explanation_spread': float(np.std(feature_importance)),
            'num_instances_analyzed': int(shap_values.shape[0]),
            'num_features': int(shap_values.shape[1])
        }
    
    def generate_waterfall_data(self, 
                               shap_values: Dict[str, Any], 
                               instance_id: str = 'instance_0') -> Dict[str, Any]:
        """Generate data for SHAP waterfall plots"""
        
        if instance_id not in shap_values:
            raise ValidationError(f"Instance {instance_id} not found in SHAP values")
        
        instance_data = shap_values[instance_id]
        contributions = instance_data['feature_contributions']
        
        # Sort by absolute contribution for waterfall
        sorted_contributions = sorted(contributions, key=lambda x: abs(x['shap_value']), reverse=True)
        
        # Prepare waterfall data
        waterfall_data = []
        cumulative_value = shap_values.get('expected_value', 0)
        
        for contrib in sorted_contributions:
            waterfall_data.append({
                'feature_name': contrib['feature_name'],
                'contribution': contrib['shap_value'],
                'cumulative_value': cumulative_value + contrib['shap_value'],
                'feature_value': contrib['feature_value'],
                'type': contrib['contribution_type']
            })
            cumulative_value += contrib['shap_value']
        
        return {
            'waterfall_data': waterfall_data,
            'base_value': shap_values.get('expected_value', 0),
            'final_prediction': cumulative_value,
            'total_contribution': instance_data['total_contribution']
        }
    
    def generate_force_plot_data(self, 
                                shap_values: Dict[str, Any],
                                instance_id: str = 'instance_0',
                                max_features: int = 20) -> Dict[str, Any]:
        """Generate data for SHAP force plots"""
        
        if instance_id not in shap_values:
            raise ValidationError(f"Instance {instance_id} not found in SHAP values")
        
        instance_data = shap_values[instance_id]
        contributions = instance_data['feature_contributions']
        
        # Separate positive and negative contributions
        positive_contribs = [c for c in contributions if c['shap_value'] > 0]
        negative_contribs = [c for c in contributions if c['shap_value'] < 0]
        
        # Sort by absolute value and limit features
        positive_contribs.sort(key=lambda x: x['shap_value'], reverse=True)
        negative_contribs.sort(key=lambda x: x['shap_value'])
        
        positive_contribs = positive_contribs[:max_features//2]
        negative_contribs = negative_contribs[:max_features//2]
        
        return {
            'base_value': shap_values.get('expected_value', 0),
            'positive_contributions': positive_contribs,
            'negative_contributions': negative_contribs,
            'final_prediction': shap_values.get('expected_value', 0) + instance_data['total_contribution']
        }
    
    def calculate_global_feature_importance(self,
                                          model_id: str,
                                          model: Any,
                                          sample_data: pd.DataFrame,
                                          background_data: pd.DataFrame,
                                          model_type: str = 'tabular',
                                          sample_size: int = 1000) -> Dict[str, Any]:
        """Calculate global feature importance using SHAP values"""
        
        # Sample data if too large
        if len(sample_data) > sample_size:
            sample_indices = np.random.choice(len(sample_data), sample_size, replace=False)
            sample_data = sample_data.iloc[sample_indices]
        
        # Calculate SHAP values for the sample
        shap_results = self.calculate_shap_values(
            model_id, model, sample_data, background_data, model_type
        )
        
        # Extract global importance from summary statistics
        summary_stats = shap_results['summary_statistics']
        
        if summary_stats.get('type') == 'multiclass':
            # For multiclass, aggregate importance across all classes
            all_features = {}
            
            for class_key, class_data in summary_stats.items():
                if class_key.startswith('class_'):
                    for feature_data in class_data['feature_summaries']:
                        feature_name = feature_data['feature_name']
                        if feature_name not in all_features:
                            all_features[feature_name] = {
                                'feature_name': feature_name,
                                'total_importance': 0,
                                'class_contributions': {}
                            }
                        
                        all_features[feature_name]['total_importance'] += feature_data['importance']
                        all_features[feature_name]['class_contributions'][class_key] = feature_data
            
            # Sort by total importance
            global_importance = sorted(all_features.values(), 
                                     key=lambda x: x['total_importance'], 
                                     reverse=True)
        else:
            # Single class case
            global_importance = summary_stats['feature_summaries']
        
        return {
            'model_id': model_id,
            'global_feature_importance': global_importance,
            'analysis_date': datetime.utcnow().isoformat(),
            'sample_size': len(sample_data),
            'importance_type': 'shap_based'
        }
    
    def explain_prediction_difference(self,
                                    model_id: str,
                                    model: Any,
                                    instance1: pd.DataFrame,
                                    instance2: pd.DataFrame,
                                    background_data: pd.DataFrame,
                                    model_type: str = 'tabular') -> Dict[str, Any]:
        """Explain the difference between two predictions using SHAP"""
        
        # Calculate SHAP values for both instances
        combined_instances = pd.concat([instance1, instance2], ignore_index=True)
        
        shap_results = self.calculate_shap_values(
            model_id, model, combined_instances, background_data, model_type
        )
        
        # Extract SHAP values for each instance
        if 'instance_0' in shap_results['shap_values'] and 'instance_1' in shap_results['shap_values']:
            shap_1 = shap_results['shap_values']['instance_0']['feature_contributions']
            shap_2 = shap_results['shap_values']['instance_1']['feature_contributions']
        else:
            raise ExplainabilityError("Could not find SHAP values for both instances")
        
        # Calculate differences
        differences = []
        for i in range(len(shap_1)):
            feature_name = shap_1[i]['feature_name']
            shap_diff = shap_1[i]['shap_value'] - shap_2[i]['shap_value']
            value_diff = shap_1[i]['feature_value'] - shap_2[i]['feature_value']
            
            differences.append({
                'feature_name': feature_name,
                'shap_difference': shap_diff,
                'value_difference': value_diff,
                'abs_shap_difference': abs(shap_diff),
                'instance1_shap': shap_1[i]['shap_value'],
                'instance2_shap': shap_2[i]['shap_value'],
                'instance1_value': shap_1[i]['feature_value'],
                'instance2_value': shap_2[i]['feature_value']
            })
        
        # Sort by absolute SHAP difference
        differences.sort(key=lambda x: x['abs_shap_difference'], reverse=True)
        
        # Calculate total prediction difference
        total_shap_diff = sum([d['shap_difference'] for d in differences])
        
        return {
            'model_id': model_id,
            'feature_differences': differences,
            'top_5_differences': differences[:5],
            'total_prediction_difference': total_shap_diff,
            'analysis_timestamp': datetime.utcnow().isoformat()
        }
    
    def clear_cache(self, model_id: str = None):
        """Clear explainer cache"""
        if model_id:
            # Clear specific model cache
            keys_to_remove = [key for key in self.explainer_cache.keys() if key.startswith(model_id)]
            for key in keys_to_remove:
                del self.explainer_cache[key]
        else:
            # Clear all cache
            self.explainer_cache.clear()
        
        logger.info(f"Cleared SHAP explainer cache for model_id: {model_id}")