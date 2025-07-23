"""
Comprehensive Fairness Analysis Service
Implementation of bias detection, fairness metrics, and mitigation strategies
"""

import asyncio
import json
import logging
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import uuid

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Fairness libraries
try:
    from fairlearn.metrics import (
        demographic_parity_difference, demographic_parity_ratio,
        equalized_odds_difference, equalized_odds_ratio,
        selection_rate, count, MetricFrame
    )
    from fairlearn.postprocessing import ThresholdOptimizer
    from fairlearn.preprocessing import CorrelationRemover
    from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
    FAIRLEARN_AVAILABLE = True
except ImportError:
    logger.warning("Fairlearn not available. Some fairness features will be limited.")
    FAIRLEARN_AVAILABLE = False

try:
    from aif360.datasets import BinaryLabelDataset
    from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
    from aif360.algorithms.preprocessing import Reweighing, DisparateImpactRemover
    from aif360.algorithms.inprocessing import PrejudiceRemover
    from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing, EqOddsPostprocessing
    AIF360_AVAILABLE = True
except ImportError:
    logger.warning("AIF360 not available. Some advanced fairness features will be limited.")
    AIF360_AVAILABLE = False

from app.core.config import get_settings
from app.core.database import get_database

logger = logging.getLogger(__name__)
settings = get_settings()

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

class FairnessService:
    """
    Comprehensive fairness analysis and bias mitigation service
    """
    
    def __init__(self):
        self.fairness_analyses = {}
        self.mitigation_strategies = {}
        self.fairness_thresholds = {
            'demographic_parity_difference': 0.1,
            'demographic_parity_ratio': 0.8,
            'equalized_odds_difference': 0.1,
            'equalized_odds_ratio': 0.8,
            'statistical_parity_difference': 0.1,
            'average_odds_difference': 0.1,
            'true_positive_rate_difference': 0.1,
            'false_positive_rate_difference': 0.1
        }
    
    async def analyze_fairness(
        self,
        model_id: str,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        sensitive_features: Union[str, List[str]],
        predictions: Optional[np.ndarray] = None,
        prediction_probabilities: Optional[np.ndarray] = None,
        protected_groups: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive fairness analysis
        
        Args:
            model_id: Unique model identifier
            model: Trained model
            X: Feature data
            y: True labels
            sensitive_features: Column name(s) for sensitive attributes
            predictions: Model predictions (if not provided, will generate)
            prediction_probabilities: Prediction probabilities
            protected_groups: Dictionary defining protected group values
        
        Returns:
            Comprehensive fairness analysis results
        """
        try:
            # Ensure sensitive_features is a list
            if isinstance(sensitive_features, str):
                sensitive_features = [sensitive_features]
            
            # Generate predictions if not provided
            if predictions is None:
                predictions = model.predict(X)
            
            if prediction_probabilities is None and hasattr(model, 'predict_proba'):
                prediction_probabilities = model.predict_proba(X)
            
            # Initialize results
            fairness_results = {
                'model_id': model_id,
                'analysis_timestamp': datetime.utcnow().isoformat(),
                'sensitive_features': sensitive_features,
                'total_samples': len(X),
                'overall_fairness_score': 0.0,
                'fairness_violations': [],
                'group_metrics': {},
                'bias_metrics': {},
                'recommendations': [],
                'protected_group_analysis': {}
            }
            
            # 1. Overall model performance
            overall_metrics = await self._calculate_overall_metrics(y, predictions, prediction_probabilities)
            fairness_results['overall_performance'] = overall_metrics
            
            # 2. Group-wise performance analysis
            for sensitive_attr in sensitive_features:
                group_analysis = await self._analyze_group_performance(
                    X, y, predictions, sensitive_attr, prediction_probabilities
                )
                fairness_results['group_metrics'][sensitive_attr] = group_analysis
            
            # 3. Fairness metrics calculation
            if FAIRLEARN_AVAILABLE:
                fairlearn_metrics = await self._calculate_fairlearn_metrics(
                    y, predictions, X[sensitive_features]
                )
                fairness_results['fairlearn_metrics'] = fairlearn_metrics
            
            # 4. AIF360 analysis if available
            if AIF360_AVAILABLE:
                aif360_metrics = await self._calculate_aif360_metrics(
                    X, y, predictions, sensitive_features[0]  # Use first sensitive feature
                )
                fairness_results['aif360_metrics'] = aif360_metrics
            
            # 5. Intersectional fairness analysis
            if len(sensitive_features) > 1:
                intersectional_analysis = await self._analyze_intersectional_fairness(
                    X, y, predictions, sensitive_features
                )
                fairness_results['intersectional_analysis'] = intersectional_analysis
            
            # 6. Bias detection
            bias_detection = await self._detect_bias_patterns(
                X, y, predictions, sensitive_features
            )
            fairness_results['bias_detection'] = bias_detection
            
            # 7. Calculate overall fairness score
            fairness_score = await self._calculate_fairness_score(fairness_results)
            fairness_results['overall_fairness_score'] = fairness_score
            
            # 8. Generate recommendations
            recommendations = await self._generate_fairness_recommendations(fairness_results)
            fairness_results['recommendations'] = recommendations
            
            # 9. Identify fairness violations
            violations = await self._identify_fairness_violations(fairness_results)
            fairness_results['fairness_violations'] = violations
            
            # Store analysis
            analysis_id = str(uuid.uuid4())
            self.fairness_analyses[analysis_id] = fairness_results
            fairness_results['analysis_id'] = analysis_id
            
            return {
                'status': 'success',
                **fairness_results
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze fairness for model {model_id}: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'model_id': model_id
            }
    
    async def _calculate_overall_metrics(
        self, 
        y_true: pd.Series, 
        y_pred: np.ndarray, 
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Calculate overall model performance metrics"""
        try:
            metrics = {}
            
            # Determine if classification or regression
            is_classification = len(np.unique(y_true)) < 50  # Heuristic
            
            if is_classification:
                metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
                metrics['precision'] = float(precision_score(y_true, y_pred, average='weighted'))
                metrics['recall'] = float(recall_score(y_true, y_pred, average='weighted'))
                metrics['f1_score'] = float(f1_score(y_true, y_pred, average='weighted'))
                
                if y_proba is not None and y_proba.shape[1] == 2:
                    metrics['auc_roc'] = float(roc_auc_score(y_true, y_proba[:, 1]))
                
                # Class distribution
                class_dist = pd.Series(y_true).value_counts(normalize=True)
                metrics['class_distribution'] = class_dist.to_dict()
                
            else:
                # Regression metrics
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                metrics['mse'] = float(mean_squared_error(y_true, y_pred))
                metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
                metrics['r2'] = float(r2_score(y_true, y_pred))
            
            metrics['prediction_distribution'] = {
                'mean': float(np.mean(y_pred)),
                'std': float(np.std(y_pred)),
                'min': float(np.min(y_pred)),
                'max': float(np.max(y_pred))
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating overall metrics: {e}")
            return {'error': str(e)}
    
    async def _analyze_group_performance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        predictions: np.ndarray,
        sensitive_attr: str,
        prediction_probabilities: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Analyze model performance across different groups"""
        try:
            if sensitive_attr not in X.columns:
                return {'error': f'Sensitive attribute {sensitive_attr} not found in data'}
            
            group_analysis = {
                'sensitive_attribute': sensitive_attr,
                'groups': {},
                'performance_gaps': {}
            }
            
            # Get unique groups
            groups = X[sensitive_attr].unique()
            
            # Calculate metrics for each group
            group_metrics = {}
            for group in groups:
                group_mask = X[sensitive_attr] == group
                group_y = y[group_mask]
                group_pred = predictions[group_mask]
                group_proba = prediction_probabilities[group_mask] if prediction_probabilities is not None else None
                
                if len(group_y) == 0:
                    continue
                
                # Calculate group-specific metrics
                group_perf = await self._calculate_overall_metrics(group_y, group_pred, group_proba)
                group_perf['sample_size'] = int(len(group_y))
                group_perf['population_ratio'] = float(len(group_y) / len(y))
                
                group_metrics[str(group)] = group_perf
            
            group_analysis['groups'] = group_metrics
            
            # Calculate performance gaps
            if len(group_metrics) >= 2:
                metric_names = ['accuracy', 'precision', 'recall', 'f1_score']
                gaps = {}
                
                group_names = list(group_metrics.keys())
                for metric in metric_names:
                    if metric in group_metrics[group_names[0]]:
                        values = [group_metrics[g].get(metric, 0) for g in group_names]
                        gaps[metric] = {
                            'max_difference': float(max(values) - min(values)),
                            'coefficient_of_variation': float(np.std(values) / np.mean(values)) if np.mean(values) > 0 else 0,
                            'group_values': {g: group_metrics[g].get(metric, 0) for g in group_names}
                        }
                
                group_analysis['performance_gaps'] = gaps
            
            return group_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing group performance: {e}")
            return {'error': str(e)}
    
    async def _calculate_fairlearn_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        sensitive_features: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate fairness metrics using Fairlearn"""
        if not FAIRLEARN_AVAILABLE:
            return {'error': 'Fairlearn not available'}
        
        try:
            fairlearn_metrics = {}
            
            # Demographic parity
            dp_diff = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features)
            dp_ratio = demographic_parity_ratio(y_true, y_pred, sensitive_features=sensitive_features)
            
            fairlearn_metrics['demographic_parity'] = {
                'difference': float(dp_diff),
                'ratio': float(dp_ratio),
                'fair': abs(dp_diff) <= self.fairness_thresholds['demographic_parity_difference']
            }
            
            # Equalized odds
            eo_diff = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_features)
            eo_ratio = equalized_odds_ratio(y_true, y_pred, sensitive_features=sensitive_features)
            
            fairlearn_metrics['equalized_odds'] = {
                'difference': float(eo_diff),
                'ratio': float(eo_ratio),
                'fair': abs(eo_diff) <= self.fairness_thresholds['equalized_odds_difference']
            }
            
            # Selection rate by group
            selection_rates = MetricFrame(
                metrics=selection_rate,
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive_features
            )
            
            fairlearn_metrics['selection_rates'] = {
                'by_group': selection_rates.by_group.to_dict(),
                'overall': float(selection_rates.overall),
                'difference': float(selection_rates.difference())
            }
            
            # Accuracy by group
            accuracy_frame = MetricFrame(
                metrics=accuracy_score,
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive_features
            )
            
            fairlearn_metrics['accuracy_by_group'] = {
                'by_group': accuracy_frame.by_group.to_dict(),
                'overall': float(accuracy_frame.overall),
                'difference': float(accuracy_frame.difference())
            }
            
            return fairlearn_metrics
            
        except Exception as e:
            logger.error(f"Error calculating Fairlearn metrics: {e}")
            return {'error': str(e)}
    
    async def _calculate_aif360_metrics(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        predictions: np.ndarray,
        sensitive_attr: str
    ) -> Dict[str, Any]:
        """Calculate fairness metrics using AIF360"""
        if not AIF360_AVAILABLE:
            return {'error': 'AIF360 not available'}
        
        try:
            # Prepare data for AIF360
            df = X.copy()
            df['target'] = y
            df['prediction'] = predictions
            
            # Create BinaryLabelDataset
            dataset = BinaryLabelDataset(
                favorable_label=1,
                unfavorable_label=0,
                df=df,
                label_names=['target'],
                protected_attribute_names=[sensitive_attr]
            )
            
            # Create predicted dataset
            dataset_pred = dataset.copy()
            dataset_pred.labels = predictions.reshape(-1, 1)
            
            # Calculate pre-processing metrics
            metric = BinaryLabelDatasetMetric(dataset, unprivileged_groups=[{sensitive_attr: 0}], privileged_groups=[{sensitive_attr: 1}])
            
            aif360_metrics = {
                'dataset_metrics': {
                    'statistical_parity_difference': float(metric.statistical_parity_difference()),
                    'disparate_impact': float(metric.disparate_impact()),
                    'consistency': float(metric.consistency())
                }
            }
            
            # Calculate post-processing metrics
            classified_metric = ClassificationMetric(
                dataset, dataset_pred,
                unprivileged_groups=[{sensitive_attr: 0}],
                privileged_groups=[{sensitive_attr: 1}]
            )
            
            aif360_metrics['classification_metrics'] = {
                'average_odds_difference': float(classified_metric.average_odds_difference()),
                'equal_opportunity_difference': float(classified_metric.equal_opportunity_difference()),
                'theil_index': float(classified_metric.theil_index()),
                'true_positive_rate_difference': float(classified_metric.true_positive_rate_difference()),
                'false_positive_rate_difference': float(classified_metric.false_positive_rate_difference())
            }
            
            return aif360_metrics
            
        except Exception as e:
            logger.error(f"Error calculating AIF360 metrics: {e}")
            return {'error': str(e)}
    
    async def _analyze_intersectional_fairness(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        predictions: np.ndarray,
        sensitive_features: List[str]
    ) -> Dict[str, Any]:
        """Analyze fairness across intersections of multiple sensitive attributes"""
        try:
            # Create intersectional groups
            X_intersect = X[sensitive_features].copy()
            X_intersect['intersection'] = X_intersect.apply(
                lambda row: '_'.join([f"{col}:{row[col]}" for col in sensitive_features]), 
                axis=1
            )
            
            # Analyze performance for each intersectional group
            intersect_analysis = {
                'intersectional_groups': {},
                'performance_matrix': {},
                'worst_performing_groups': [],
                'best_performing_groups': []
            }
            
            groups = X_intersect['intersection'].unique()
            group_performances = []
            
            for group in groups:
                group_mask = X_intersect['intersection'] == group
                group_y = y[group_mask]
                group_pred = predictions[group_mask]
                
                if len(group_y) < 10:  # Skip very small groups
                    continue
                
                group_metrics = await self._calculate_overall_metrics(group_y, group_pred)
                group_metrics['sample_size'] = int(len(group_y))
                group_metrics['group_name'] = group
                
                intersect_analysis['intersectional_groups'][group] = group_metrics
                
                if 'accuracy' in group_metrics:
                    group_performances.append((group, group_metrics['accuracy']))
            
            # Identify best and worst performing groups
            if group_performances:
                sorted_groups = sorted(group_performances, key=lambda x: x[1])
                intersect_analysis['worst_performing_groups'] = sorted_groups[:3]
                intersect_analysis['best_performing_groups'] = sorted_groups[-3:]
                
                # Performance spread
                accuracies = [perf[1] for perf in group_performances]
                intersect_analysis['performance_spread'] = {
                    'max_accuracy': float(max(accuracies)),
                    'min_accuracy': float(min(accuracies)),
                    'accuracy_range': float(max(accuracies) - min(accuracies)),
                    'std_accuracy': float(np.std(accuracies))
                }
            
            return intersect_analysis
            
        except Exception as e:
            logger.error(f"Error in intersectional fairness analysis: {e}")
            return {'error': str(e)}
    
    async def _detect_bias_patterns(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        predictions: np.ndarray,
        sensitive_features: List[str]
    ) -> Dict[str, Any]:
        """Detect various bias patterns in the model"""
        try:
            bias_patterns = {
                'feature_bias': {},
                'prediction_bias': {},
                'correlation_bias': {},
                'systematic_errors': []
            }
            
            # 1. Feature correlation with sensitive attributes
            for sensitive_attr in sensitive_features:
                if sensitive_attr in X.columns:
                    correlations = {}
                    
                    for feature in X.columns:
                        if feature != sensitive_attr and pd.api.types.is_numeric_dtype(X[feature]):
                            try:
                                corr = X[feature].corr(pd.get_dummies(X[sensitive_attr]).iloc[:, 0])
                                correlations[feature] = float(corr) if not pd.isna(corr) else 0
                            except:
                                correlations[feature] = 0
                    
                    # Sort by absolute correlation
                    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
                    bias_patterns['feature_bias'][sensitive_attr] = {
                        'high_correlation_features': sorted_corrs[:5],
                        'max_correlation': abs(sorted_corrs[0][1]) if sorted_corrs else 0
                    }
            
            # 2. Prediction bias analysis
            for sensitive_attr in sensitive_features:
                if sensitive_attr in X.columns:
                    # Calculate prediction rates by group
                    group_pred_rates = {}
                    groups = X[sensitive_attr].unique()
                    
                    for group in groups:
                        group_mask = X[sensitive_attr] == group
                        group_predictions = predictions[group_mask]
                        
                        if len(group_predictions) > 0:
                            positive_rate = np.mean(group_predictions == 1) if len(np.unique(predictions)) == 2 else np.mean(group_predictions)
                            group_pred_rates[str(group)] = float(positive_rate)
                    
                    bias_patterns['prediction_bias'][sensitive_attr] = group_pred_rates
            
            # 3. Systematic error detection
            for sensitive_attr in sensitive_features:
                if sensitive_attr in X.columns:
                    groups = X[sensitive_attr].unique()
                    
                    # Check for consistent over/under-prediction
                    for group in groups:
                        group_mask = X[sensitive_attr] == group
                        group_y = y[group_mask]
                        group_pred = predictions[group_mask]
                        
                        if len(group_y) > 10:
                            # For classification
                            if len(np.unique(y)) == 2:
                                false_positive_rate = np.mean((group_pred == 1) & (group_y == 0))
                                false_negative_rate = np.mean((group_pred == 0) & (group_y == 1))
                                
                                if false_positive_rate > 0.2:
                                    bias_patterns['systematic_errors'].append({
                                        'type': 'high_false_positive_rate',
                                        'group': f"{sensitive_attr}={group}",
                                        'rate': float(false_positive_rate)
                                    })
                                
                                if false_negative_rate > 0.2:
                                    bias_patterns['systematic_errors'].append({
                                        'type': 'high_false_negative_rate',
                                        'group': f"{sensitive_attr}={group}",
                                        'rate': float(false_negative_rate)
                                    })
            
            return bias_patterns
            
        except Exception as e:
            logger.error(f"Error detecting bias patterns: {e}")
            return {'error': str(e)}
    
    async def _calculate_fairness_score(self, fairness_results: Dict[str, Any]) -> float:
        """Calculate overall fairness score (0-1, higher is better)"""
        try:
            scores = []
            
            # From Fairlearn metrics
            if 'fairlearn_metrics' in fairness_results:
                fl_metrics = fairness_results['fairlearn_metrics']
                
                if 'demographic_parity' in fl_metrics:
                    dp_score = 1 - min(1, abs(fl_metrics['demographic_parity']['difference']) / 0.2)
                    scores.append(dp_score)
                
                if 'equalized_odds' in fl_metrics:
                    eo_score = 1 - min(1, abs(fl_metrics['equalized_odds']['difference']) / 0.2)
                    scores.append(eo_score)
            
            # From group performance gaps
            if 'group_metrics' in fairness_results:
                for attr, group_data in fairness_results['group_metrics'].items():
                    if 'performance_gaps' in group_data:
                        gaps = group_data['performance_gaps']
                        if 'accuracy' in gaps:
                            acc_gap_score = 1 - min(1, gaps['accuracy']['max_difference'] / 0.2)
                            scores.append(acc_gap_score)
            
            # From intersectional analysis
            if 'intersectional_analysis' in fairness_results:
                intersect = fairness_results['intersectional_analysis']
                if 'performance_spread' in intersect:
                    spread_score = 1 - min(1, intersect['performance_spread']['accuracy_range'] / 0.3)
                    scores.append(spread_score)
            
            # Default score if no metrics available
            if not scores:
                return 0.5
            
            return float(np.mean(scores))
            
        except Exception as e:
            logger.error(f"Error calculating fairness score: {e}")
            return 0.0
    
    async def _generate_fairness_recommendations(self, fairness_results: Dict[str, Any]) -> List[str]:
        """Generate actionable fairness recommendations"""
        recommendations = []
        
        try:
            fairness_score = fairness_results.get('overall_fairness_score', 0)
            
            # Overall assessment
            if fairness_score >= 0.8:
                recommendations.append("✅ Model shows good fairness across groups. Continue monitoring.")
            elif fairness_score >= 0.6:
                recommendations.append("⚠️ Model shows moderate fairness concerns. Consider improvements.")
            else:
                recommendations.append("🚨 Model shows significant fairness issues. Immediate action required.")
            
            # Specific recommendations based on violations
            violations = fairness_results.get('fairness_violations', [])
            if violations:
                recommendations.append(f"📋 {len(violations)} fairness violations detected. Review detailed analysis.")
            
            # Group performance recommendations
            if 'group_metrics' in fairness_results:
                for attr, group_data in fairness_results['group_metrics'].items():
                    if 'performance_gaps' in group_data:
                        gaps = group_data['performance_gaps']
                        if 'accuracy' in gaps and gaps['accuracy']['max_difference'] > 0.1:
                            recommendations.append(f"⚖️ Large accuracy gap detected for {attr}. Consider bias mitigation.")
            
            # Data recommendations
            if 'bias_detection' in fairness_results:
                bias = fairness_results['bias_detection']
                if 'systematic_errors' in bias and bias['systematic_errors']:
                    recommendations.append("🔍 Systematic prediction errors detected. Review training data quality.")
            
            # Mitigation strategies
            recommendations.extend([
                "📊 Implement fairness-aware ML techniques during training",
                "🎯 Use post-processing methods to adjust decision thresholds",
                "📈 Set up continuous fairness monitoring",
                "👥 Engage domain experts and affected communities in the review process"
            ])
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            recommendations.append("❌ Error generating recommendations. Manual review required.")
        
        return recommendations
    
    async def _identify_fairness_violations(self, fairness_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify specific fairness violations"""
        violations = []
        
        try:
            # Check Fairlearn metrics
            if 'fairlearn_metrics' in fairness_results:
                fl_metrics = fairness_results['fairlearn_metrics']
                
                if 'demographic_parity' in fl_metrics:
                    dp = fl_metrics['demographic_parity']
                    if abs(dp['difference']) > self.fairness_thresholds['demographic_parity_difference']:
                        violations.append({
                            'type': 'demographic_parity_violation',
                            'severity': 'high' if abs(dp['difference']) > 0.2 else 'medium',
                            'metric': 'Demographic Parity Difference',
                            'value': dp['difference'],
                            'threshold': self.fairness_thresholds['demographic_parity_difference'],
                            'description': f"Demographic parity difference of {dp['difference']:.3f} exceeds threshold"
                        })
                
                if 'equalized_odds' in fl_metrics:
                    eo = fl_metrics['equalized_odds']
                    if abs(eo['difference']) > self.fairness_thresholds['equalized_odds_difference']:
                        violations.append({
                            'type': 'equalized_odds_violation',
                            'severity': 'high' if abs(eo['difference']) > 0.2 else 'medium',
                            'metric': 'Equalized Odds Difference',
                            'value': eo['difference'],
                            'threshold': self.fairness_thresholds['equalized_odds_difference'],
                            'description': f"Equalized odds difference of {eo['difference']:.3f} exceeds threshold"
                        })
            
            # Check AIF360 metrics
            if 'aif360_metrics' in fairness_results:
                aif_metrics = fairness_results['aif360_metrics']
                
                if 'classification_metrics' in aif_metrics:
                    cm = aif_metrics['classification_metrics']
                    
                    if abs(cm.get('average_odds_difference', 0)) > 0.1:
                        violations.append({
                            'type': 'average_odds_violation',
                            'severity': 'medium',
                            'metric': 'Average Odds Difference',
                            'value': cm['average_odds_difference'],
                            'threshold': 0.1,
                            'description': f"Average odds difference exceeds acceptable range"
                        })
            
        except Exception as e:
            logger.error(f"Error identifying violations: {e}")
        
        return violations
    
    async def suggest_mitigation_strategies(
        self,
        analysis_id: str,
        X: pd.DataFrame,
        y: pd.Series,
        model: BaseEstimator,
        sensitive_features: List[str]
    ) -> Dict[str, Any]:
        """Suggest bias mitigation strategies based on analysis"""
        try:
            if analysis_id not in self.fairness_analyses:
                raise ValueError(f"Analysis {analysis_id} not found")
            
            analysis = self.fairness_analyses[analysis_id]
            
            strategies = {
                'preprocessing': [],
                'inprocessing': [],
                'postprocessing': [],
                'data_strategies': [],
                'recommended_approach': ''
            }
            
            # Preprocessing strategies
            strategies['preprocessing'].extend([
                {
                    'name': 'Correlation Remover',
                    'description': 'Remove correlation between features and sensitive attributes',
                    'suitable_for': 'High feature correlation with sensitive attributes',
                    'implementation_complexity': 'Low',
                    'performance_impact': 'Low-Medium'
                },
                {
                    'name': 'Reweighing',
                    'description': 'Adjust sample weights to balance representation',
                    'suitable_for': 'Imbalanced groups in training data',
                    'implementation_complexity': 'Low',
                    'performance_impact': 'Low'
                },
                {
                    'name': 'Disparate Impact Remover',
                    'description': 'Transform features to reduce disparate impact',
                    'suitable_for': 'Direct discrimination in features',
                    'implementation_complexity': 'Medium',
                    'performance_impact': 'Medium'
                }
            ])
            
            # In-processing strategies
            if hasattr(model, 'fit'):  # Trainable model
                strategies['inprocessing'].extend([
                    {
                        'name': 'Fairness Constraints',
                        'description': 'Add fairness constraints to model training',
                        'suitable_for': 'Training from scratch with fairness requirements',
                        'implementation_complexity': 'High',
                        'performance_impact': 'Medium-High'
                    },
                    {
                        'name': 'Adversarial Debiasing',
                        'description': 'Use adversarial training to reduce bias',
                        'suitable_for': 'Deep learning models',
                        'implementation_complexity': 'High',
                        'performance_impact': 'Medium'
                    }
                ])
            
            # Post-processing strategies
            strategies['postprocessing'].extend([
                {
                    'name': 'Threshold Optimization',
                    'description': 'Adjust decision thresholds per group',
                    'suitable_for': 'Different performance across groups',
                    'implementation_complexity': 'Medium',
                    'performance_impact': 'Low'
                },
                {
                    'name': 'Calibrated Equalized Odds',
                    'description': 'Post-process to achieve equalized odds',
                    'suitable_for': 'Classification with equalized odds violations',
                    'implementation_complexity': 'Medium',
                    'performance_impact': 'Low-Medium'
                }
            ])
            
            # Data strategies
            strategies['data_strategies'].extend([
                {
                    'name': 'Data Augmentation',
                    'description': 'Increase representation of underrepresented groups',
                    'suitable_for': 'Small group sizes',
                    'implementation_complexity': 'Medium',
                    'performance_impact': 'Variable'
                },
                {
                    'name': 'Synthetic Data Generation',
                    'description': 'Generate synthetic samples for balanced representation',
                    'suitable_for': 'Severe group imbalance',
                    'implementation_complexity': 'High',
                    'performance_impact': 'Variable'
                }
            ])
            
            # Recommend approach based on analysis
            fairness_score = analysis.get('overall_fairness_score', 0)
            violations = analysis.get('fairness_violations', [])
            
            if fairness_score < 0.5:
                strategies['recommended_approach'] = 'comprehensive'
                strategies['priority_actions'] = [
                    'Start with data audit and augmentation',
                    'Implement preprocessing techniques',
                    'Consider retraining with fairness constraints',
                    'Apply post-processing as backup'
                ]
            elif fairness_score < 0.7:
                strategies['recommended_approach'] = 'targeted'
                strategies['priority_actions'] = [
                    'Apply post-processing techniques',
                    'Adjust decision thresholds',
                    'Monitor and iterate'
                ]
            else:
                strategies['recommended_approach'] = 'monitoring'
                strategies['priority_actions'] = [
                    'Implement continuous monitoring',
                    'Set up fairness alerts',
                    'Regular fairness audits'
                ]
            
            return {
                'status': 'success',
                'analysis_id': analysis_id,
                'mitigation_strategies': strategies,
                'implementation_timeline': {
                    'immediate': ['Threshold optimization', 'Monitoring setup'],
                    'short_term': ['Data augmentation', 'Post-processing'],
                    'long_term': ['Model retraining', 'Systematic bias removal']
                }
            }
            
        except Exception as e:
            logger.error(f"Error suggesting mitigation strategies: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'analysis_id': analysis_id
            }

# Global service instance
fairness_service = FairnessService()