# Advanced Explainability Features - Comprehensive XAI Toolkit
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform
import warnings
from datetime import datetime
import json
import logging
from concurrent.futures import ThreadPoolExecutor
import itertools

from .shap_engine import SHAPExplainabilityEngine
from .lime_engine import LIMEExplainabilityEngine
from ..exceptions import ExplainabilityError, ValidationError

logger = logging.getLogger(__name__)

class AdvancedExplainabilityEngine:
    """Comprehensive explainable AI toolkit with advanced analysis features"""
    
    def __init__(self):
        self.shap_engine = SHAPExplainabilityEngine()
        self.lime_engine = LIMEExplainabilityEngine()
        self.surrogate_models = {}
        self.feature_interactions_cache = {}
        
    # ========================================================================================
    # WHAT-IF ANALYSIS
    # ========================================================================================
    
    def what_if_analysis(self,
                        model: Any,
                        base_instance: Union[pd.DataFrame, np.ndarray],
                        feature_ranges: Dict[str, Union[List, Tuple, np.ndarray]],
                        feature_names: List[str] = None,
                        n_samples: int = 100,
                        analysis_type: str = 'grid') -> Dict[str, Any]:
        """
        Perform comprehensive what-if analysis
        
        Args:
            model: Trained model
            base_instance: Base instance for perturbation
            feature_ranges: Dictionary of feature names to their possible values/ranges
            feature_names: List of feature names
            n_samples: Number of samples to generate for analysis
            analysis_type: 'grid', 'random', or 'systematic'
        """
        
        try:
            if isinstance(base_instance, pd.DataFrame):
                base_array = base_instance.values.flatten()
                if feature_names is None:
                    feature_names = list(base_instance.columns)
            else:
                base_array = base_instance.flatten() if base_instance.ndim > 1 else base_instance
                if feature_names is None:
                    feature_names = [f'feature_{i}' for i in range(len(base_array))]
            
            # Generate what-if scenarios
            scenarios = self._generate_scenarios(
                base_array, feature_ranges, feature_names, n_samples, analysis_type
            )
            
            # Get predictions for all scenarios
            predictions = []
            for scenario in scenarios:
                if hasattr(model, 'predict_proba'):
                    pred_proba = model.predict_proba([scenario])[0]
                    pred = model.predict([scenario])[0]
                    predictions.append({
                        'prediction': pred,
                        'probabilities': pred_proba.tolist() if hasattr(pred_proba, 'tolist') else pred_proba,
                        'confidence': np.max(pred_proba) if hasattr(pred_proba, '__len__') else None
                    })
                else:
                    pred = model.predict([scenario])[0]
                    predictions.append({
                        'prediction': pred,
                        'probabilities': None,
                        'confidence': None
                    })
            
            # Analyze prediction changes
            analysis_results = self._analyze_prediction_changes(
                base_array, scenarios, predictions, feature_names, feature_ranges
            )
            
            # Calculate feature sensitivity
            sensitivity_analysis = self._calculate_feature_sensitivity(
                scenarios, predictions, feature_names, feature_ranges
            )
            
            # Generate recommendations
            recommendations = self._generate_what_if_recommendations(
                analysis_results, sensitivity_analysis, feature_ranges
            )
            
            return {
                'base_instance': base_array.tolist(),
                'base_prediction': predictions[0] if scenarios else None,
                'scenarios': [
                    {
                        'scenario_id': i,
                        'features': scenario.tolist(),
                        'changed_features': self._get_changed_features(base_array, scenario, feature_names),
                        'prediction': pred
                    }
                    for i, (scenario, pred) in enumerate(zip(scenarios, predictions))
                ],
                'analysis_results': analysis_results,
                'sensitivity_analysis': sensitivity_analysis,
                'recommendations': recommendations,
                'feature_names': feature_names,
                'analysis_metadata': {
                    'analysis_type': analysis_type,
                    'n_samples': len(scenarios),
                    'feature_ranges': feature_ranges,
                    'timestamp': datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"What-if analysis failed: {str(e)}")
            raise ExplainabilityError(f"What-if analysis failed: {str(e)}")
    
    def _generate_scenarios(self,
                           base_instance: np.ndarray,
                           feature_ranges: Dict[str, Union[List, Tuple, np.ndarray]],
                           feature_names: List[str],
                           n_samples: int,
                           analysis_type: str) -> List[np.ndarray]:
        """Generate what-if scenarios"""
        
        scenarios = []
        
        if analysis_type == 'grid':
            # Grid search over feature ranges
            scenarios = self._generate_grid_scenarios(
                base_instance, feature_ranges, feature_names, n_samples
            )
        elif analysis_type == 'random':
            # Random sampling from feature ranges
            scenarios = self._generate_random_scenarios(
                base_instance, feature_ranges, feature_names, n_samples
            )
        elif analysis_type == 'systematic':
            # Systematic perturbation of each feature
            scenarios = self._generate_systematic_scenarios(
                base_instance, feature_ranges, feature_names
            )
        
        return scenarios
    
    def _generate_grid_scenarios(self,
                               base_instance: np.ndarray,
                               feature_ranges: Dict[str, Union[List, Tuple, np.ndarray]],
                               feature_names: List[str],
                               n_samples: int) -> List[np.ndarray]:
        """Generate grid-based scenarios"""
        
        scenarios = []
        
        # Create grid for each feature in ranges
        feature_grids = {}
        for feature_name, feature_range in feature_ranges.items():
            if feature_name in feature_names:
                if isinstance(feature_range, (list, tuple)):
                    if len(feature_range) == 2:
                        # Range [min, max]
                        feature_grids[feature_name] = np.linspace(
                            feature_range[0], feature_range[1], min(10, int(n_samples**0.5))
                        )
                    else:
                        # Discrete values
                        feature_grids[feature_name] = np.array(feature_range)
                else:
                    feature_grids[feature_name] = np.array(feature_range)
        
        # Generate all combinations
        if feature_grids:
            feature_combinations = list(itertools.product(*feature_grids.values()))
            feature_names_in_ranges = list(feature_grids.keys())
            
            for combination in feature_combinations[:n_samples]:
                scenario = base_instance.copy()
                for i, feature_name in enumerate(feature_names_in_ranges):
                    if feature_name in feature_names:
                        feature_idx = feature_names.index(feature_name)
                        scenario[feature_idx] = combination[i]
                scenarios.append(scenario)
        
        return scenarios
    
    def _generate_random_scenarios(self,
                                 base_instance: np.ndarray,
                                 feature_ranges: Dict[str, Union[List, Tuple, np.ndarray]],
                                 feature_names: List[str],
                                 n_samples: int) -> List[np.ndarray]:
        """Generate random scenarios"""
        
        scenarios = []
        
        for _ in range(n_samples):
            scenario = base_instance.copy()
            
            # Randomly select features to modify
            features_to_modify = np.random.choice(
                list(feature_ranges.keys()),
                size=min(len(feature_ranges), np.random.randint(1, len(feature_ranges) + 1)),
                replace=False
            )
            
            for feature_name in features_to_modify:
                if feature_name in feature_names:
                    feature_idx = feature_names.index(feature_name)
                    feature_range = feature_ranges[feature_name]
                    
                    if isinstance(feature_range, (list, tuple)) and len(feature_range) == 2:
                        # Random value in range
                        new_value = np.random.uniform(feature_range[0], feature_range[1])
                    else:
                        # Random choice from discrete values
                        new_value = np.random.choice(feature_range)
                    
                    scenario[feature_idx] = new_value
            
            scenarios.append(scenario)
        
        return scenarios
    
    def _generate_systematic_scenarios(self,
                                     base_instance: np.ndarray,
                                     feature_ranges: Dict[str, Union[List, Tuple, np.ndarray]],
                                     feature_names: List[str]) -> List[np.ndarray]:
        """Generate systematic scenarios (one feature at a time)"""
        
        scenarios = []
        
        for feature_name, feature_range in feature_ranges.items():
            if feature_name in feature_names:
                feature_idx = feature_names.index(feature_name)
                
                if isinstance(feature_range, (list, tuple)) and len(feature_range) == 2:
                    # Create range of values
                    values = np.linspace(feature_range[0], feature_range[1], 10)
                else:
                    # Use discrete values
                    values = feature_range
                
                for value in values:
                    scenario = base_instance.copy()
                    scenario[feature_idx] = value
                    scenarios.append(scenario)
        
        return scenarios
    
    def _analyze_prediction_changes(self,
                                  base_instance: np.ndarray,
                                  scenarios: List[np.ndarray],
                                  predictions: List[Dict],
                                  feature_names: List[str],
                                  feature_ranges: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how predictions change across scenarios"""
        
        base_prediction = predictions[0]['prediction'] if predictions else None
        
        # Calculate prediction changes
        prediction_changes = []
        for i, pred in enumerate(predictions[1:], 1):
            change = {
                'scenario_id': i,
                'prediction_change': pred['prediction'] - base_prediction if base_prediction is not None else 0,
                'absolute_change': abs(pred['prediction'] - base_prediction) if base_prediction is not None else 0
            }
            
            if pred['confidence'] is not None and predictions[0]['confidence'] is not None:
                change['confidence_change'] = pred['confidence'] - predictions[0]['confidence']
            
            prediction_changes.append(change)
        
        # Find most impactful scenarios
        if prediction_changes:
            most_impactful = max(prediction_changes, key=lambda x: x['absolute_change'])
            least_impactful = min(prediction_changes, key=lambda x: x['absolute_change'])
            
            # Statistical analysis
            changes = [pc['prediction_change'] for pc in prediction_changes]
            abs_changes = [pc['absolute_change'] for pc in prediction_changes]
            
            return {
                'prediction_changes': prediction_changes,
                'most_impactful_scenario': most_impactful,
                'least_impactful_scenario': least_impactful,
                'statistics': {
                    'mean_change': np.mean(changes),
                    'std_change': np.std(changes),
                    'max_absolute_change': np.max(abs_changes),
                    'min_absolute_change': np.min(abs_changes),
                    'median_change': np.median(changes)
                }
            }
        
        return {}
    
    def _calculate_feature_sensitivity(self,
                                     scenarios: List[np.ndarray],
                                     predictions: List[Dict],
                                     feature_names: List[str],
                                     feature_ranges: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate feature sensitivity from scenarios"""
        
        sensitivity_scores = {}
        
        for feature_name in feature_ranges.keys():
            if feature_name in feature_names:
                feature_idx = feature_names.index(feature_name)
                
                # Collect feature values and corresponding predictions
                feature_values = []
                pred_values = []
                
                for scenario, prediction in zip(scenarios, predictions):
                    feature_values.append(scenario[feature_idx])
                    pred_values.append(prediction['prediction'])
                
                if len(set(feature_values)) > 1:  # Feature varies across scenarios
                    # Calculate correlation between feature values and predictions
                    correlation = np.corrcoef(feature_values, pred_values)[0, 1]
                    if np.isnan(correlation):
                        correlation = 0.0
                    
                    # Calculate variance in predictions when feature changes
                    prediction_variance = np.var(pred_values)
                    
                    # Calculate sensitivity as combination of correlation and variance
                    sensitivity = abs(correlation) * prediction_variance
                    
                    sensitivity_scores[feature_name] = {
                        'sensitivity_score': float(sensitivity),
                        'correlation': float(correlation),
                        'prediction_variance': float(prediction_variance),
                        'feature_range_used': [min(feature_values), max(feature_values)]
                    }
        
        # Rank features by sensitivity
        ranked_features = sorted(
            sensitivity_scores.items(),
            key=lambda x: x[1]['sensitivity_score'],
            reverse=True
        )
        
        return {
            'feature_sensitivity': sensitivity_scores,
            'ranked_features': [f[0] for f in ranked_features],
            'most_sensitive_feature': ranked_features[0][0] if ranked_features else None,
            'least_sensitive_feature': ranked_features[-1][0] if ranked_features else None
        }
    
    # ========================================================================================
    # PARTIAL DEPENDENCE PLOTS
    # ========================================================================================
    
    def generate_partial_dependence_plots(self,
                                        model: Any,
                                        X: Union[pd.DataFrame, np.ndarray],
                                        feature_names: List[str] = None,
                                        features: List[Union[int, str]] = None,
                                        grid_resolution: int = 100,
                                        percentiles: Tuple[float, float] = (0.05, 0.95),
                                        kind: str = 'average') -> Dict[str, Any]:
        """
        Generate partial dependence plots for specified features
        
        Args:
            model: Trained model
            X: Training data
            feature_names: List of feature names
            features: Features to analyze (indices or names)
            grid_resolution: Number of points in the grid
            percentiles: Percentiles to use for feature range
            kind: 'average', 'individual', or 'both'
        """
        
        try:
            if isinstance(X, pd.DataFrame):
                X_array = X.values
                if feature_names is None:
                    feature_names = list(X.columns)
            else:
                X_array = X
                if feature_names is None:
                    feature_names = [f'feature_{i}' for i in range(X_array.shape[1])]
            
            # Determine features to analyze
            if features is None:
                # Use top 5 most important features (if available)
                try:
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                        top_indices = np.argsort(importances)[-5:][::-1]
                        features = top_indices.tolist()
                    else:
                        features = list(range(min(5, X_array.shape[1])))
                except:
                    features = list(range(min(5, X_array.shape[1])))
            
            # Convert feature names to indices if needed
            feature_indices = []
            for feature in features:
                if isinstance(feature, str) and feature in feature_names:
                    feature_indices.append(feature_names.index(feature))
                elif isinstance(feature, int):
                    feature_indices.append(feature)
            
            pdp_results = {}
            
            # Generate PDP for each feature
            for feature_idx in feature_indices:
                feature_name = feature_names[feature_idx]
                
                try:
                    # Calculate partial dependence
                    pdp_result = partial_dependence(
                        model, X_array, [feature_idx],
                        grid_resolution=grid_resolution,
                        percentiles=percentiles,
                        kind=kind
                    )
                    
                    # Extract results
                    if hasattr(pdp_result, 'average'):
                        # Newer sklearn versions
                        avg_predictions = pdp_result.average[0]
                        feature_values = pdp_result.grid_values[0]
                        individual_predictions = getattr(pdp_result, 'individual', None)
                    else:
                        # Older sklearn versions
                        avg_predictions = pdp_result['average'][0]
                        feature_values = pdp_result['grid_values'][0]
                        individual_predictions = pdp_result.get('individual', None)
                    
                    # Calculate additional statistics
                    pdp_analysis = self._analyze_partial_dependence(
                        feature_values, avg_predictions, X_array[:, feature_idx]
                    )
                    
                    pdp_results[feature_name] = {
                        'feature_values': feature_values.tolist(),
                        'average_predictions': avg_predictions.tolist(),
                        'individual_predictions': individual_predictions.tolist() if individual_predictions is not None else None,
                        'analysis': pdp_analysis,
                        'feature_index': feature_idx
                    }
                    
                except Exception as e:
                    logger.warning(f"Failed to generate PDP for feature {feature_name}: {str(e)}")
                    continue
            
            # Generate 2D PDPs for feature interactions
            interaction_pdps = self._generate_interaction_pdps(
                model, X_array, feature_indices[:3], feature_names, grid_resolution, percentiles
            )
            
            return {
                'partial_dependence_plots': pdp_results,
                'interaction_plots': interaction_pdps,
                'metadata': {
                    'grid_resolution': grid_resolution,
                    'percentiles': percentiles,
                    'kind': kind,
                    'n_features_analyzed': len(pdp_results),
                    'timestamp': datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"PDP generation failed: {str(e)}")
            raise ExplainabilityError(f"PDP generation failed: {str(e)}")
    
    def _analyze_partial_dependence(self,
                                  feature_values: np.ndarray,
                                  predictions: np.ndarray,
                                  original_feature_values: np.ndarray) -> Dict[str, Any]:
        """Analyze partial dependence results"""
        
        analysis = {}
        
        try:
            # Trend analysis
            analysis['trend'] = self._analyze_trend(feature_values, predictions)
            
            # Find peaks and valleys
            analysis['extrema'] = self._find_extrema(feature_values, predictions)
            
            # Calculate feature effect strength
            prediction_range = np.max(predictions) - np.min(predictions)
            analysis['effect_strength'] = float(prediction_range)
            
            # Monotonicity check
            analysis['monotonicity'] = self._check_monotonicity(predictions)
            
            # Curvature analysis
            analysis['curvature'] = self._analyze_curvature(feature_values, predictions)
            
            # Feature distribution analysis
            analysis['feature_distribution'] = {
                'mean': float(np.mean(original_feature_values)),
                'std': float(np.std(original_feature_values)),
                'min': float(np.min(original_feature_values)),
                'max': float(np.max(original_feature_values)),
                'percentiles': {
                    '25': float(np.percentile(original_feature_values, 25)),
                    '50': float(np.percentile(original_feature_values, 50)),
                    '75': float(np.percentile(original_feature_values, 75))
                }
            }
            
        except Exception as e:
            logger.warning(f"PDP analysis failed: {str(e)}")
        
        return analysis
    
    def _analyze_trend(self, feature_values: np.ndarray, predictions: np.ndarray) -> str:
        """Analyze overall trend in partial dependence"""
        
        # Calculate correlation
        correlation = np.corrcoef(feature_values, predictions)[0, 1]
        
        if np.isnan(correlation):
            return 'no_trend'
        elif correlation > 0.7:
            return 'strong_positive'
        elif correlation > 0.3:
            return 'moderate_positive'
        elif correlation > -0.3:
            return 'weak_or_no_trend'
        elif correlation > -0.7:
            return 'moderate_negative'
        else:
            return 'strong_negative'
    
    def _find_extrema(self, feature_values: np.ndarray, predictions: np.ndarray) -> Dict[str, Any]:
        """Find peaks and valleys in partial dependence"""
        
        extrema = {'peaks': [], 'valleys': []}
        
        try:
            # Find local maxima and minima
            for i in range(1, len(predictions) - 1):
                if predictions[i] > predictions[i-1] and predictions[i] > predictions[i+1]:
                    extrema['peaks'].append({
                        'feature_value': float(feature_values[i]),
                        'prediction': float(predictions[i])
                    })
                elif predictions[i] < predictions[i-1] and predictions[i] < predictions[i+1]:
                    extrema['valleys'].append({
                        'feature_value': float(feature_values[i]),
                        'prediction': float(predictions[i])
                    })
            
            # Find global maximum and minimum
            max_idx = np.argmax(predictions)
            min_idx = np.argmin(predictions)
            
            extrema['global_maximum'] = {
                'feature_value': float(feature_values[max_idx]),
                'prediction': float(predictions[max_idx])
            }
            extrema['global_minimum'] = {
                'feature_value': float(feature_values[min_idx]),
                'prediction': float(predictions[min_idx])
            }
            
        except Exception as e:
            logger.warning(f"Extrema finding failed: {str(e)}")
        
        return extrema
    
    def _check_monotonicity(self, predictions: np.ndarray) -> Dict[str, Any]:
        """Check if the partial dependence is monotonic"""
        
        # Calculate differences
        diffs = np.diff(predictions)
        
        # Check monotonicity
        is_increasing = np.all(diffs >= 0)
        is_decreasing = np.all(diffs <= 0)
        is_strictly_increasing = np.all(diffs > 0)
        is_strictly_decreasing = np.all(diffs < 0)
        
        # Calculate monotonicity strength
        positive_changes = np.sum(diffs > 0)
        negative_changes = np.sum(diffs < 0)
        total_changes = len(diffs)
        
        if total_changes > 0:
            monotonicity_strength = abs(positive_changes - negative_changes) / total_changes
        else:
            monotonicity_strength = 0
        
        return {
            'is_monotonic': is_increasing or is_decreasing,
            'is_increasing': is_increasing,
            'is_decreasing': is_decreasing,
            'is_strictly_increasing': is_strictly_increasing,
            'is_strictly_decreasing': is_strictly_decreasing,
            'monotonicity_strength': float(monotonicity_strength),
            'direction_changes': int(np.sum(np.diff(np.sign(diffs)) != 0))
        }
    
    def _analyze_curvature(self, feature_values: np.ndarray, predictions: np.ndarray) -> Dict[str, Any]:
        """Analyze curvature of partial dependence"""
        
        curvature_analysis = {}
        
        try:
            # Calculate second derivative (curvature)
            if len(predictions) >= 3:
                second_derivative = np.diff(predictions, n=2)
                
                curvature_analysis['average_curvature'] = float(np.mean(np.abs(second_derivative)))
                curvature_analysis['max_curvature'] = float(np.max(np.abs(second_derivative)))
                curvature_analysis['curvature_variance'] = float(np.var(second_derivative))
                
                # Identify regions of high curvature
                high_curvature_threshold = np.percentile(np.abs(second_derivative), 90)
                high_curvature_indices = np.where(np.abs(second_derivative) > high_curvature_threshold)[0]
                
                curvature_analysis['high_curvature_regions'] = [
                    {
                        'feature_value': float(feature_values[i+1]),
                        'curvature': float(second_derivative[i])
                    }
                    for i in high_curvature_indices
                ]
            
        except Exception as e:
            logger.warning(f"Curvature analysis failed: {str(e)}")
        
        return curvature_analysis
    
    def _generate_interaction_pdps(self,
                                 model: Any,
                                 X: np.ndarray,
                                 feature_indices: List[int],
                                 feature_names: List[str],
                                 grid_resolution: int,
                                 percentiles: Tuple[float, float]) -> Dict[str, Any]:
        """Generate 2D partial dependence plots for feature interactions"""
        
        interaction_pdps = {}
        
        # Generate all pairs of features
        feature_pairs = list(itertools.combinations(feature_indices[:3], 2))  # Limit to avoid too many plots
        
        for feature_pair in feature_pairs:
            feature1_idx, feature2_idx = feature_pair
            feature1_name = feature_names[feature1_idx]
            feature2_name = feature_names[feature2_idx]
            
            try:
                # Calculate 2D partial dependence
                pdp_result = partial_dependence(
                    model, X, feature_pair,
                    grid_resolution=grid_resolution,
                    percentiles=percentiles
                )
                
                # Extract results
                if hasattr(pdp_result, 'average'):
                    predictions_2d = pdp_result.average[0]
                    feature1_values = pdp_result.grid_values[0]
                    feature2_values = pdp_result.grid_values[1]
                else:
                    predictions_2d = pdp_result['average'][0]
                    feature1_values = pdp_result['grid_values'][0]
                    feature2_values = pdp_result['grid_values'][1]
                
                # Analyze interaction
                interaction_analysis = self._analyze_feature_interaction(
                    feature1_values, feature2_values, predictions_2d
                )
                
                pair_name = f"{feature1_name}_vs_{feature2_name}"
                interaction_pdps[pair_name] = {
                    'feature1_name': feature1_name,
                    'feature2_name': feature2_name,
                    'feature1_values': feature1_values.tolist(),
                    'feature2_values': feature2_values.tolist(),
                    'predictions_2d': predictions_2d.tolist(),
                    'interaction_analysis': interaction_analysis
                }
                
            except Exception as e:
                logger.warning(f"Failed to generate interaction PDP for {feature1_name} vs {feature2_name}: {str(e)}")
                continue
        
        return interaction_pdps
    
    def _analyze_feature_interaction(self,
                                   feature1_values: np.ndarray,
                                   feature2_values: np.ndarray,
                                   predictions_2d: np.ndarray) -> Dict[str, Any]:
        """Analyze feature interaction from 2D partial dependence"""
        
        analysis = {}
        
        try:
            # Calculate interaction strength
            # Interaction is measured as deviation from additive effects
            feature1_main_effect = np.mean(predictions_2d, axis=1)  # Average over feature2
            feature2_main_effect = np.mean(predictions_2d, axis=0)  # Average over feature1
            
            # Create additive prediction surface
            additive_surface = feature1_main_effect[:, np.newaxis] + feature2_main_effect[np.newaxis, :] - np.mean(predictions_2d)
            
            # Interaction is the difference between actual and additive
            interaction_surface = predictions_2d - additive_surface
            
            analysis['interaction_strength'] = float(np.std(interaction_surface))
            analysis['max_interaction'] = float(np.max(np.abs(interaction_surface)))
            analysis['interaction_variance'] = float(np.var(interaction_surface))
            
            # Find regions of strongest interaction
            max_interaction_idx = np.unravel_index(np.argmax(np.abs(interaction_surface)), interaction_surface.shape)
            analysis['strongest_interaction_region'] = {
                'feature1_value': float(feature1_values[max_interaction_idx[0]]),
                'feature2_value': float(feature2_values[max_interaction_idx[1]]),
                'interaction_value': float(interaction_surface[max_interaction_idx])
            }
            
            # Classify interaction type
            analysis['interaction_type'] = self._classify_interaction_type(interaction_surface)
            
        except Exception as e:
            logger.warning(f"Interaction analysis failed: {str(e)}")
        
        return analysis
    
    def _classify_interaction_type(self, interaction_surface: np.ndarray) -> str:
        """Classify the type of feature interaction"""
        
        # Calculate statistics of interaction surface
        mean_interaction = np.mean(interaction_surface)
        std_interaction = np.std(interaction_surface)
        
        if std_interaction < 0.01:
            return 'no_interaction'
        elif np.all(interaction_surface >= 0):
            return 'positive_synergy'
        elif np.all(interaction_surface <= 0):
            return 'negative_synergy'
        elif abs(mean_interaction) < std_interaction * 0.1:
            return 'mixed_interaction'
        elif mean_interaction > 0:
            return 'predominantly_positive'
        else:
            return 'predominantly_negative'
    
    # ========================================================================================
    # COMPREHENSIVE MODEL STATISTICS
    # ========================================================================================
    
    def calculate_comprehensive_model_stats(self,
                                          model: Any,
                                          X_train: Union[pd.DataFrame, np.ndarray],
                                          y_train: np.ndarray,
                                          X_test: Union[pd.DataFrame, np.ndarray] = None,
                                          y_test: np.ndarray = None,
                                          feature_names: List[str] = None,
                                          target_names: List[str] = None,
                                          is_classifier: bool = None) -> Dict[str, Any]:
        """
        Calculate comprehensive model statistics for both classification and regression
        
        Args:
            model: Trained model
            X_train: Training features
            y_train: Training targets
            X_test: Test features (optional)
            y_test: Test targets (optional)
            feature_names: List of feature names
            target_names: List of target class names (for classification)
            is_classifier: Whether model is classifier (auto-detected if None)
        """
        
        try:
            # Convert DataFrames to arrays
            if isinstance(X_train, pd.DataFrame):
                X_train_array = X_train.values
                if feature_names is None:
                    feature_names = list(X_train.columns)
            else:
                X_train_array = X_train
                if feature_names is None:
                    feature_names = [f'feature_{i}' for i in range(X_train_array.shape[1])]
            
            if X_test is not None:
                if isinstance(X_test, pd.DataFrame):
                    X_test_array = X_test.values
                else:
                    X_test_array = X_test
            else:
                X_test_array = None
            
            # Auto-detect if classifier
            if is_classifier is None:
                is_classifier = hasattr(model, 'predict_proba') or hasattr(model, 'decision_function')
            
            # Generate predictions
            y_train_pred = model.predict(X_train_array)
            if X_test_array is not None and y_test is not None:
                y_test_pred = model.predict(X_test_array)
            else:
                y_test_pred = None
            
            # Get prediction probabilities for classification
            if is_classifier and hasattr(model, 'predict_proba'):
                y_train_proba = model.predict_proba(X_train_array)
                if X_test_array is not None:
                    y_test_proba = model.predict_proba(X_test_array)
                else:
                    y_test_proba = None
            else:
                y_train_proba = None
                y_test_proba = None
            
            # Calculate statistics based on model type
            if is_classifier:
                stats = self._calculate_classification_stats(
                    y_train, y_train_pred, y_train_proba,
                    y_test, y_test_pred, y_test_proba,
                    target_names
                )
            else:
                stats = self._calculate_regression_stats(
                    y_train, y_train_pred,
                    y_test, y_test_pred
                )
            
            # Add feature importance analysis
            feature_importance_stats = self._calculate_feature_importance_stats(
                model, X_train_array, y_train, feature_names
            )
            stats['feature_importance'] = feature_importance_stats
            
            # Add model complexity stats
            complexity_stats = self._calculate_model_complexity_stats(model, X_train_array)
            stats['model_complexity'] = complexity_stats
            
            # Add cross-validation results
            cv_stats = self._calculate_cross_validation_stats(
                model, X_train_array, y_train, is_classifier
            )
            stats['cross_validation'] = cv_stats
            
            # Add prediction uncertainty analysis
            if y_train_proba is not None:
                uncertainty_stats = self._calculate_prediction_uncertainty_stats(
                    y_train_proba, y_test_proba
                )
                stats['prediction_uncertainty'] = uncertainty_stats
            
            # Add metadata
            stats['metadata'] = {
                'model_type': 'classifier' if is_classifier else 'regressor',
                'n_train_samples': len(y_train),
                'n_test_samples': len(y_test) if y_test is not None else 0,
                'n_features': X_train_array.shape[1],
                'feature_names': feature_names,
                'target_names': target_names,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Model statistics calculation failed: {str(e)}")
            raise ExplainabilityError(f"Model statistics calculation failed: {str(e)}")
    
    def _calculate_classification_stats(self,
                                      y_train: np.ndarray,
                                      y_train_pred: np.ndarray,
                                      y_train_proba: np.ndarray,
                                      y_test: np.ndarray,
                                      y_test_pred: np.ndarray,
                                      y_test_proba: np.ndarray,
                                      target_names: List[str]) -> Dict[str, Any]:
        """Calculate comprehensive classification statistics"""
        
        stats = {}
        
        # Training set statistics
        train_stats = {}
        train_stats['accuracy'] = float(accuracy_score(y_train, y_train_pred))
        train_stats['precision'] = precision_score(y_train, y_train_pred, average='weighted', zero_division=0)
        train_stats['recall'] = recall_score(y_train, y_train_pred, average='weighted', zero_division=0)
        train_stats['f1_score'] = f1_score(y_train, y_train_pred, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm_train = confusion_matrix(y_train, y_train_pred)
        train_stats['confusion_matrix'] = cm_train.tolist()
        
        # Classification report
        train_stats['classification_report'] = classification_report(
            y_train, y_train_pred, output_dict=True, zero_division=0
        )
        
        # ROC AUC (for binary and multiclass)
        if y_train_proba is not None:
            try:
                if len(np.unique(y_train)) == 2:
                    # Binary classification
                    train_stats['roc_auc'] = float(roc_auc_score(y_train, y_train_proba[:, 1]))
                else:
                    # Multiclass classification
                    train_stats['roc_auc'] = float(roc_auc_score(y_train, y_train_proba, multi_class='ovr', average='weighted'))
            except:
                train_stats['roc_auc'] = None
        
        stats['train_metrics'] = train_stats
        
        # Test set statistics (if available)
        if y_test is not None and y_test_pred is not None:
            test_stats = {}
            test_stats['accuracy'] = float(accuracy_score(y_test, y_test_pred))
            test_stats['precision'] = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
            test_stats['recall'] = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
            test_stats['f1_score'] = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
            
            # Confusion matrix
            cm_test = confusion_matrix(y_test, y_test_pred)
            test_stats['confusion_matrix'] = cm_test.tolist()
            
            # Classification report
            test_stats['classification_report'] = classification_report(
                y_test, y_test_pred, output_dict=True, zero_division=0
            )
            
            # ROC AUC
            if y_test_proba is not None:
                try:
                    if len(np.unique(y_test)) == 2:
                        test_stats['roc_auc'] = float(roc_auc_score(y_test, y_test_proba[:, 1]))
                        
                        # ROC Curve
                        fpr, tpr, thresholds = roc_curve(y_test, y_test_proba[:, 1])
                        test_stats['roc_curve'] = {
                            'fpr': fpr.tolist(),
                            'tpr': tpr.tolist(),
                            'thresholds': thresholds.tolist()
                        }
                        
                        # Precision-Recall Curve
                        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_test, y_test_proba[:, 1])
                        test_stats['precision_recall_curve'] = {
                            'precision': precision_curve.tolist(),
                            'recall': recall_curve.tolist(),
                            'thresholds': pr_thresholds.tolist()
                        }
                    else:
                        test_stats['roc_auc'] = float(roc_auc_score(y_test, y_test_proba, multi_class='ovr', average='weighted'))
                except:
                    test_stats['roc_auc'] = None
            
            stats['test_metrics'] = test_stats
            
            # Calculate overfitting metrics
            stats['overfitting_analysis'] = {
                'accuracy_gap': train_stats['accuracy'] - test_stats['accuracy'],
                'f1_gap': train_stats['f1_score'] - test_stats['f1_score'],
                'overfitting_score': max(0, (train_stats['accuracy'] - test_stats['accuracy']) / train_stats['accuracy'])
            }
        
        # Class balance analysis
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        class_balance = {
            'class_distribution': dict(zip(unique_classes.tolist(), class_counts.tolist())),
            'class_balance_ratio': float(np.min(class_counts) / np.max(class_counts)),
            'is_balanced': float(np.min(class_counts) / np.max(class_counts)) > 0.8
        }
        stats['class_balance'] = class_balance
        
        return stats
    
    def _calculate_regression_stats(self,
                                  y_train: np.ndarray,
                                  y_train_pred: np.ndarray,
                                  y_test: np.ndarray,
                                  y_test_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive regression statistics"""
        
        stats = {}
        
        # Training set statistics
        train_stats = {}
        train_stats['mse'] = float(mean_squared_error(y_train, y_train_pred))
        train_stats['rmse'] = float(np.sqrt(train_stats['mse']))
        train_stats['mae'] = float(mean_absolute_error(y_train, y_train_pred))
        train_stats['r2_score'] = float(r2_score(y_train, y_train_pred))
        
        # Additional regression metrics
        train_residuals = y_train - y_train_pred
        train_stats['mean_residual'] = float(np.mean(train_residuals))
        train_stats['std_residual'] = float(np.std(train_residuals))
        train_stats['mean_absolute_percentage_error'] = float(np.mean(np.abs(train_residuals / y_train)) * 100)
        
        # Residual analysis
        train_stats['residual_analysis'] = self._analyze_residuals(y_train, y_train_pred, train_residuals)
        
        stats['train_metrics'] = train_stats
        
        # Test set statistics (if available)
        if y_test is not None and y_test_pred is not None:
            test_stats = {}
            test_stats['mse'] = float(mean_squared_error(y_test, y_test_pred))
            test_stats['rmse'] = float(np.sqrt(test_stats['mse']))
            test_stats['mae'] = float(mean_absolute_error(y_test, y_test_pred))
            test_stats['r2_score'] = float(r2_score(y_test, y_test_pred))
            
            # Additional test metrics
            test_residuals = y_test - y_test_pred
            test_stats['mean_residual'] = float(np.mean(test_residuals))
            test_stats['std_residual'] = float(np.std(test_residuals))
            test_stats['mean_absolute_percentage_error'] = float(np.mean(np.abs(test_residuals / y_test)) * 100)
            
            # Residual analysis
            test_stats['residual_analysis'] = self._analyze_residuals(y_test, y_test_pred, test_residuals)
            
            stats['test_metrics'] = test_stats
            
            # Calculate overfitting metrics
            stats['overfitting_analysis'] = {
                'r2_gap': train_stats['r2_score'] - test_stats['r2_score'],
                'rmse_ratio': test_stats['rmse'] / train_stats['rmse'] if train_stats['rmse'] > 0 else float('inf'),
                'overfitting_score': max(0, (train_stats['r2_score'] - test_stats['r2_score']) / abs(train_stats['r2_score'])) if train_stats['r2_score'] != 0 else 0
            }
        
        # Target distribution analysis
        stats['target_distribution'] = {
            'mean': float(np.mean(y_train)),
            'std': float(np.std(y_train)),
            'min': float(np.min(y_train)),
            'max': float(np.max(y_train)),
            'skewness': float(stats.skew(y_train)),
            'kurtosis': float(stats.kurtosis(y_train))
        }
        
        return stats
    
    def _analyze_residuals(self, y_true: np.ndarray, y_pred: np.ndarray, residuals: np.ndarray) -> Dict[str, Any]:
        """Analyze residuals for regression models"""
        
        analysis = {}
        
        try:
            # Normality test of residuals
            _, shapiro_p_value = stats.shapiro(residuals[:5000])  # Limit sample size for performance
            analysis['residuals_normal'] = float(shapiro_p_value) > 0.05
            analysis['shapiro_p_value'] = float(shapiro_p_value)
            
            # Homoscedasticity check (residuals vs predictions)
            correlation_resid_pred = np.corrcoef(y_pred, np.abs(residuals))[0, 1]
            analysis['homoscedastic'] = abs(correlation_resid_pred) < 0.1  # Low correlation suggests homoscedasticity
            analysis['residual_pred_correlation'] = float(correlation_resid_pred) if not np.isnan(correlation_resid_pred) else 0.0
            
            # Outlier detection in residuals
            residual_std = np.std(residuals)
            outlier_threshold = 3 * residual_std
            outliers = np.abs(residuals) > outlier_threshold
            analysis['n_outliers'] = int(np.sum(outliers))
            analysis['outlier_percentage'] = float(np.sum(outliers) / len(residuals) * 100)
            
            # Residual distribution percentiles
            analysis['residual_percentiles'] = {
                '5': float(np.percentile(residuals, 5)),
                '25': float(np.percentile(residuals, 25)),
                '50': float(np.percentile(residuals, 50)),
                '75': float(np.percentile(residuals, 75)),
                '95': float(np.percentile(residuals, 95))
            }
            
        except Exception as e:
            logger.warning(f"Residual analysis failed: {str(e)}")
        
        return analysis
    
    def _calculate_feature_importance_stats(self,
                                          model: Any,
                                          X: np.ndarray,
                                          y: np.ndarray,
                                          feature_names: List[str]) -> Dict[str, Any]:
        """Calculate feature importance statistics"""
        
        importance_stats = {}
        
        try:
            # Model-specific feature importance
            if hasattr(model, 'feature_importances_'):
                importance_values = model.feature_importances_
                importance_stats['builtin_importance'] = {
                    'values': importance_values.tolist(),
                    'feature_ranking': np.argsort(importance_values)[::-1].tolist(),
                    'top_5_features': [feature_names[i] for i in np.argsort(importance_values)[::-1][:5]]
                }
            
            # Permutation importance (model-agnostic)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    perm_importance = permutation_importance(
                        model, X, y, n_repeats=5, random_state=42, n_jobs=1
                    )
                
                importance_stats['permutation_importance'] = {
                    'values': perm_importance.importances_mean.tolist(),
                    'std': perm_importance.importances_std.tolist(),
                    'feature_ranking': np.argsort(perm_importance.importances_mean)[::-1].tolist(),
                    'top_5_features': [feature_names[i] for i in np.argsort(perm_importance.importances_mean)[::-1][:5]]
                }
            except Exception as e:
                logger.warning(f"Permutation importance calculation failed: {str(e)}")
            
            # Statistical feature analysis
            importance_stats['feature_statistics'] = self._calculate_feature_statistics(X, feature_names)
            
        except Exception as e:
            logger.warning(f"Feature importance calculation failed: {str(e)}")
        
        return importance_stats
    
    def _calculate_feature_statistics(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Calculate statistical properties of features"""
        
        feature_stats = {}
        
        try:
            for i, feature_name in enumerate(feature_names):
                feature_values = X[:, i]
                
                feature_stats[feature_name] = {
                    'mean': float(np.mean(feature_values)),
                    'std': float(np.std(feature_values)),
                    'min': float(np.min(feature_values)),
                    'max': float(np.max(feature_values)),
                    'skewness': float(stats.skew(feature_values)),
                    'kurtosis': float(stats.kurtosis(feature_values)),
                    'n_unique': int(len(np.unique(feature_values))),
                    'missing_percentage': float(np.sum(np.isnan(feature_values)) / len(feature_values) * 100),
                    'percentiles': {
                        '25': float(np.percentile(feature_values, 25)),
                        '50': float(np.percentile(feature_values, 50)),
                        '75': float(np.percentile(feature_values, 75))
                    }
                }
        
        except Exception as e:
            logger.warning(f"Feature statistics calculation failed: {str(e)}")
        
        return feature_stats
    
    def _calculate_model_complexity_stats(self, model: Any, X: np.ndarray) -> Dict[str, Any]:
        """Calculate model complexity statistics"""
        
        complexity_stats = {}
        
        try:
            # Model-specific complexity measures
            if hasattr(model, 'n_estimators'):
                complexity_stats['n_estimators'] = int(model.n_estimators)
            
            if hasattr(model, 'max_depth'):
                complexity_stats['max_depth'] = int(model.max_depth) if model.max_depth else None
            
            if hasattr(model, 'n_features_in_'):
                complexity_stats['n_features'] = int(model.n_features_in_)
            else:
                complexity_stats['n_features'] = X.shape[1]
            
            # Parameter count estimation
            n_params = self._estimate_parameter_count(model, X)
            complexity_stats['estimated_parameters'] = n_params
            
            # VC dimension estimation (rough approximation)
            complexity_stats['estimated_vc_dimension'] = self._estimate_vc_dimension(model, X)
            
        except Exception as e:
            logger.warning(f"Model complexity calculation failed: {str(e)}")
        
        return complexity_stats
    
    def _estimate_parameter_count(self, model: Any, X: np.ndarray) -> int:
        """Estimate the number of parameters in the model"""
        
        try:
            # For tree-based models
            if hasattr(model, 'tree_'):
                return int(model.tree_.node_count)
            elif hasattr(model, 'estimators_'):
                if hasattr(model.estimators_[0], 'tree_'):
                    return sum(est.tree_.node_count for est in model.estimators_)
            
            # For linear models
            if hasattr(model, 'coef_'):
                if model.coef_.ndim == 1:
                    return len(model.coef_) + (1 if hasattr(model, 'intercept_') else 0)
                else:
                    return model.coef_.size + (model.intercept_.size if hasattr(model, 'intercept_') else 0)
            
            # Default approximation
            return X.shape[1]
            
        except:
            return X.shape[1]
    
    def _estimate_vc_dimension(self, model: Any, X: np.ndarray) -> int:
        """Estimate VC dimension of the model"""
        
        try:
            # Rough approximations for common model types
            if hasattr(model, 'coef_'):
                # Linear models: VC dimension ≈ number of parameters
                return X.shape[1] + 1
            elif hasattr(model, 'n_estimators'):
                # Ensemble models: much higher VC dimension
                return min(10000, X.shape[1] * 10)
            else:
                # Default approximation
                return X.shape[1] * 2
                
        except:
            return X.shape[1]
    
    def _calculate_cross_validation_stats(self,
                                        model: Any,
                                        X: np.ndarray,
                                        y: np.ndarray,
                                        is_classifier: bool,
                                        cv_folds: int = 5) -> Dict[str, Any]:
        """Calculate cross-validation statistics"""
        
        cv_stats = {}
        
        try:
            # Choose scoring metric based on model type
            if is_classifier:
                scoring_metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
            else:
                scoring_metrics = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
            
            for metric in scoring_metrics:
                try:
                    cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring=metric, n_jobs=1)
                    
                    # Convert negative scores to positive for error metrics
                    if metric.startswith('neg_'):
                        cv_scores = -cv_scores
                        metric_name = metric[4:]  # Remove 'neg_' prefix
                    else:
                        metric_name = metric
                    
                    cv_stats[metric_name] = {
                        'mean': float(np.mean(cv_scores)),
                        'std': float(np.std(cv_scores)),
                        'min': float(np.min(cv_scores)),
                        'max': float(np.max(cv_scores)),
                        'scores': cv_scores.tolist()
                    }
                    
                except Exception as e:
                    logger.warning(f"Cross-validation for {metric} failed: {str(e)}")
                    continue
        
        except Exception as e:
            logger.warning(f"Cross-validation calculation failed: {str(e)}")
        
        return cv_stats
    
    def _calculate_prediction_uncertainty_stats(self,
                                              y_train_proba: np.ndarray,
                                              y_test_proba: np.ndarray = None) -> Dict[str, Any]:
        """Calculate prediction uncertainty statistics"""
        
        uncertainty_stats = {}
        
        try:
            # Training set uncertainty
            train_max_proba = np.max(y_train_proba, axis=1)
            train_entropy = -np.sum(y_train_proba * np.log(y_train_proba + 1e-8), axis=1)
            
            uncertainty_stats['train_uncertainty'] = {
                'mean_confidence': float(np.mean(train_max_proba)),
                'std_confidence': float(np.std(train_max_proba)),
                'mean_entropy': float(np.mean(train_entropy)),
                'std_entropy': float(np.std(train_entropy)),
                'low_confidence_percentage': float(np.sum(train_max_proba < 0.6) / len(train_max_proba) * 100)
            }
            
            # Test set uncertainty (if available)
            if y_test_proba is not None:
                test_max_proba = np.max(y_test_proba, axis=1)
                test_entropy = -np.sum(y_test_proba * np.log(y_test_proba + 1e-8), axis=1)
                
                uncertainty_stats['test_uncertainty'] = {
                    'mean_confidence': float(np.mean(test_max_proba)),
                    'std_confidence': float(np.std(test_max_proba)),
                    'mean_entropy': float(np.mean(test_entropy)),
                    'std_entropy': float(np.std(test_entropy)),
                    'low_confidence_percentage': float(np.sum(test_max_proba < 0.6) / len(test_max_proba) * 100)
                }
        
        except Exception as e:
            logger.warning(f"Uncertainty calculation failed: {str(e)}")
        
        return uncertainty_stats
    
    # ========================================================================================
    # UTILITY METHODS
    # ========================================================================================
    
    def _get_changed_features(self,
                             base_instance: np.ndarray,
                             scenario: np.ndarray,
                             feature_names: List[str]) -> List[Dict[str, Any]]:
        """Get features that changed between base instance and scenario"""
        
        changed_features = []
        
        for i, (base_val, scenario_val) in enumerate(zip(base_instance, scenario)):
            if abs(base_val - scenario_val) > 1e-8:  # Numerical tolerance
                changed_features.append({
                    'feature_name': feature_names[i],
                    'feature_index': i,
                    'original_value': float(base_val),
                    'new_value': float(scenario_val),
                    'change': float(scenario_val - base_val),
                    'relative_change': float((scenario_val - base_val) / base_val) if abs(base_val) > 1e-8 else float('inf')
                })
        
        return changed_features
    
    def _generate_what_if_recommendations(self,
                                        analysis_results: Dict[str, Any],
                                        sensitivity_analysis: Dict[str, Any],
                                        feature_ranges: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on what-if analysis"""
        
        recommendations = []
        
        try:
            # Most sensitive feature recommendations
            if 'most_sensitive_feature' in sensitivity_analysis and sensitivity_analysis['most_sensitive_feature']:
                most_sensitive = sensitivity_analysis['most_sensitive_feature']
                recommendations.append(f"Focus on optimizing '{most_sensitive}' as it has the highest impact on predictions")
            
            # Impactful scenario recommendations
            if 'most_impactful_scenario' in analysis_results:
                recommendations.append("Analyze the most impactful scenario to understand extreme prediction changes")
            
            # Feature range recommendations
            for feature_name in feature_ranges.keys():
                if feature_name in sensitivity_analysis.get('feature_sensitivity', {}):
                    sensitivity = sensitivity_analysis['feature_sensitivity'][feature_name]
                    if sensitivity['sensitivity_score'] > 0.1:
                        recommendations.append(f"Monitor '{feature_name}' closely due to high sensitivity")
            
            # Statistical recommendations
            if 'statistics' in analysis_results:
                stats = analysis_results['statistics']
                if stats.get('std_change', 0) > stats.get('mean_change', 0):
                    recommendations.append("High variability in predictions suggests model instability in this region")
            
            if not recommendations:
                recommendations.append("No specific concerns identified. Model appears stable in the analyzed range.")
        
        except Exception as e:
            logger.warning(f"Recommendation generation failed: {str(e)}")
            recommendations.append("Unable to generate specific recommendations due to analysis limitations.")
        
        return recommendations


class ExplainabilityError(Exception):
    """Custom exception for explainability errors"""
    pass