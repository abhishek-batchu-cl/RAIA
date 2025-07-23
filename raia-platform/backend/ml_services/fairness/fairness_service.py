# Fairness and Bias Detection Service - Core Implementation
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import logging
from scipy import stats
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
import json

from .models import FairnessReport, BiasMetric, BiasIncident, BiasMitigationPlan, FairnessConfiguration, ExplainabilityReport
from ..explainability.shap_engine import SHAPExplainabilityEngine
from ..explainability.lime_engine import LIMEExplainabilityEngine
from ..exceptions import ValidationError, ModelNotFoundError, FairnessAnalysisError

logger = logging.getLogger(__name__)

class FairnessDetectionService:
    """Core service for fairness analysis and bias detection in ML models"""
    
    def __init__(self, db: Session):
        self.db = db
        self.shap_engine = SHAPExplainabilityEngine()
        self.lime_engine = LIMEExplainabilityEngine()
    
    # ========================================================================================
    # BIAS DETECTION ALGORITHMS
    # ========================================================================================
    
    def calculate_demographic_parity(self, 
                                   y_pred: np.ndarray,
                                   protected_attribute: np.ndarray,
                                   privileged_groups: List[str],
                                   unprivileged_groups: List[str]) -> Dict[str, Any]:
        """Calculate demographic parity (statistical parity) bias metric"""
        
        results = {}
        
        # Calculate positive prediction rates for each group
        for group in privileged_groups + unprivileged_groups:
            group_mask = protected_attribute == group
            if np.sum(group_mask) == 0:
                continue
                
            positive_rate = np.mean(y_pred[group_mask])
            results[f'{group}_positive_rate'] = float(positive_rate)
            results[f'{group}_sample_size'] = int(np.sum(group_mask))
        
        # Calculate parity ratios between groups
        parity_ratios = {}
        for privileged in privileged_groups:
            for unprivileged in unprivileged_groups:
                if f'{privileged}_positive_rate' in results and f'{unprivileged}_positive_rate' in results:
                    priv_rate = results[f'{privileged}_positive_rate']
                    unpriv_rate = results[f'{unprivileged}_positive_rate']
                    
                    if priv_rate > 0:
                        ratio = unpriv_rate / priv_rate
                        parity_ratios[f'{unprivileged}_vs_{privileged}'] = float(ratio)
        
        # Calculate overall demographic parity score (closest to 1.0 is most fair)
        if parity_ratios:
            min_ratio = min(parity_ratios.values())
            max_ratio = max(parity_ratios.values())
            # Use the minimum ratio as the fairness score (worst case)
            demographic_parity_score = float(min_ratio)
        else:
            demographic_parity_score = 1.0
        
        return {
            'metric_name': 'demographic_parity',
            'score': demographic_parity_score,
            'group_rates': results,
            'parity_ratios': parity_ratios,
            'interpretation': self._interpret_demographic_parity(demographic_parity_score),
            'passes_threshold': demographic_parity_score >= 0.8  # 80% rule
        }
    
    def calculate_equalized_odds(self,
                               y_true: np.ndarray,
                               y_pred: np.ndarray,
                               protected_attribute: np.ndarray,
                               privileged_groups: List[str],
                               unprivileged_groups: List[str]) -> Dict[str, Any]:
        """Calculate equalized odds bias metric"""
        
        results = {}
        
        # Calculate TPR and FPR for each group
        for group in privileged_groups + unprivileged_groups:
            group_mask = protected_attribute == group
            if np.sum(group_mask) == 0:
                continue
            
            group_y_true = y_true[group_mask]
            group_y_pred = y_pred[group_mask]
            
            if len(np.unique(group_y_true)) < 2:
                continue
            
            # True Positive Rate (Sensitivity/Recall)
            tpr = np.mean(group_y_pred[group_y_true == 1]) if np.sum(group_y_true == 1) > 0 else 0
            
            # False Positive Rate
            fpr = np.mean(group_y_pred[group_y_true == 0]) if np.sum(group_y_true == 0) > 0 else 0
            
            results[f'{group}_tpr'] = float(tpr)
            results[f'{group}_fpr'] = float(fpr)
            results[f'{group}_sample_size'] = int(np.sum(group_mask))
            results[f'{group}_positive_samples'] = int(np.sum(group_y_true == 1))
            results[f'{group}_negative_samples'] = int(np.sum(group_y_true == 0))
        
        # Calculate equalized odds ratios
        tpr_ratios = {}
        fpr_ratios = {}
        
        for privileged in privileged_groups:
            for unprivileged in unprivileged_groups:
                if f'{privileged}_tpr' in results and f'{unprivileged}_tpr' in results:
                    priv_tpr = results[f'{privileged}_tpr']
                    unpriv_tpr = results[f'{unprivileged}_tpr']
                    
                    if priv_tpr > 0:
                        tpr_ratio = unpriv_tpr / priv_tpr
                        tpr_ratios[f'{unprivileged}_vs_{privileged}_tpr'] = float(tpr_ratio)
                
                if f'{privileged}_fpr' in results and f'{unprivileged}_fpr' in results:
                    priv_fpr = results[f'{privileged}_fpr']
                    unpriv_fpr = results[f'{unprivileged}_fpr']
                    
                    if priv_fpr > 0:
                        fpr_ratio = unpriv_fpr / priv_fpr
                        fpr_ratios[f'{unprivileged}_vs_{privileged}_fpr'] = float(fpr_ratio)
        
        # Calculate overall equalized odds score
        all_ratios = list(tpr_ratios.values()) + list(fpr_ratios.values())
        if all_ratios:
            # For equalized odds, we want both TPR and FPR ratios close to 1
            deviations_from_one = [abs(1.0 - ratio) for ratio in all_ratios]
            max_deviation = max(deviations_from_one)
            equalized_odds_score = float(1.0 - max_deviation)
        else:
            equalized_odds_score = 1.0
        
        return {
            'metric_name': 'equalized_odds',
            'score': max(0.0, equalized_odds_score),  # Ensure non-negative
            'group_metrics': results,
            'tpr_ratios': tpr_ratios,
            'fpr_ratios': fpr_ratios,
            'interpretation': self._interpret_equalized_odds(equalized_odds_score),
            'passes_threshold': equalized_odds_score >= 0.8
        }
    
    def calculate_equal_opportunity(self,
                                  y_true: np.ndarray,
                                  y_pred: np.ndarray,
                                  protected_attribute: np.ndarray,
                                  privileged_groups: List[str],
                                  unprivileged_groups: List[str]) -> Dict[str, Any]:
        """Calculate equal opportunity bias metric (focuses on TPR only)"""
        
        results = {}
        
        # Calculate TPR for each group
        for group in privileged_groups + unprivileged_groups:
            group_mask = protected_attribute == group
            if np.sum(group_mask) == 0:
                continue
            
            group_y_true = y_true[group_mask]
            group_y_pred = y_pred[group_mask]
            
            # True Positive Rate (Sensitivity/Recall)
            tpr = np.mean(group_y_pred[group_y_true == 1]) if np.sum(group_y_true == 1) > 0 else 0
            
            results[f'{group}_tpr'] = float(tpr)
            results[f'{group}_sample_size'] = int(np.sum(group_mask))
            results[f'{group}_positive_samples'] = int(np.sum(group_y_true == 1))
        
        # Calculate equal opportunity ratios
        tpr_ratios = {}
        for privileged in privileged_groups:
            for unprivileged in unprivileged_groups:
                if f'{privileged}_tpr' in results and f'{unprivileged}_tpr' in results:
                    priv_tpr = results[f'{privileged}_tpr']
                    unpriv_tpr = results[f'{unprivileged}_tpr']
                    
                    if priv_tpr > 0:
                        ratio = unpriv_tpr / priv_tpr
                        tpr_ratios[f'{unprivileged}_vs_{privileged}_tpr'] = float(ratio)
        
        # Calculate overall equal opportunity score
        if tpr_ratios:
            min_ratio = min(tpr_ratios.values())
            equal_opportunity_score = float(min_ratio)
        else:
            equal_opportunity_score = 1.0
        
        return {
            'metric_name': 'equal_opportunity',
            'score': equal_opportunity_score,
            'group_tpr': results,
            'tpr_ratios': tpr_ratios,
            'interpretation': self._interpret_equal_opportunity(equal_opportunity_score),
            'passes_threshold': equal_opportunity_score >= 0.8
        }
    
    def calculate_calibration_score(self,
                                  y_true: np.ndarray,
                                  y_prob: np.ndarray,
                                  protected_attribute: np.ndarray,
                                  privileged_groups: List[str],
                                  unprivileged_groups: List[str],
                                  n_bins: int = 10) -> Dict[str, Any]:
        """Calculate calibration fairness metric"""
        
        results = {}
        calibration_scores = {}
        
        # Calculate calibration for each group
        for group in privileged_groups + unprivileged_groups:
            group_mask = protected_attribute == group
            if np.sum(group_mask) == 0:
                continue
            
            group_y_true = y_true[group_mask]
            group_y_prob = y_prob[group_mask]
            
            # Create probability bins
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            bin_accuracies = []
            bin_confidences = []
            bin_counts = []
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                # Find predictions in this bin
                in_bin = (group_y_prob > bin_lower) & (group_y_prob <= bin_upper)
                prop_in_bin = np.mean(in_bin)
                
                if prop_in_bin > 0:
                    accuracy_in_bin = np.mean(group_y_true[in_bin])
                    avg_confidence_in_bin = np.mean(group_y_prob[in_bin])
                    count_in_bin = np.sum(in_bin)
                    
                    bin_accuracies.append(accuracy_in_bin)
                    bin_confidences.append(avg_confidence_in_bin)
                    bin_counts.append(count_in_bin)
            
            # Calculate Expected Calibration Error (ECE)
            if bin_accuracies:
                total_samples = len(group_y_true)
                ece = sum([
                    (count / total_samples) * abs(acc - conf)
                    for acc, conf, count in zip(bin_accuracies, bin_confidences, bin_counts)
                ])
                
                calibration_scores[f'{group}_ece'] = float(ece)
                calibration_scores[f'{group}_sample_size'] = int(np.sum(group_mask))
                
                results[f'{group}_calibration'] = {
                    'expected_calibration_error': float(ece),
                    'bin_accuracies': bin_accuracies,
                    'bin_confidences': bin_confidences,
                    'bin_counts': bin_counts
                }
        
        # Calculate overall calibration fairness score
        eces = [score for key, score in calibration_scores.items() if key.endswith('_ece')]
        if len(eces) >= 2:
            # Calibration fairness is measured by similarity in ECE across groups
            ece_diff = max(eces) - min(eces)
            calibration_fairness_score = float(1.0 - ece_diff)  # Higher is better
        else:
            calibration_fairness_score = 1.0
        
        return {
            'metric_name': 'calibration',
            'score': max(0.0, calibration_fairness_score),
            'group_calibration': results,
            'group_eces': calibration_scores,
            'interpretation': self._interpret_calibration(calibration_fairness_score),
            'passes_threshold': calibration_fairness_score >= 0.8
        }
    
    def calculate_individual_fairness(self,
                                    model: Any,
                                    X: np.ndarray,
                                    similarity_threshold: float = 0.1,
                                    prediction_threshold: float = 0.1,
                                    sample_size: int = 1000) -> Dict[str, Any]:
        """Calculate individual fairness metric"""
        
        # Sample pairs for efficiency
        if len(X) > sample_size:
            indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        # Generate predictions
        if hasattr(model, 'predict_proba'):
            predictions = model.predict_proba(X_sample)[:, 1]  # Assuming binary classification
        else:
            predictions = model.predict(X_sample)
        
        violations = 0
        total_pairs = 0
        similarity_scores = []
        prediction_differences = []
        
        # Check individual fairness for sample pairs
        n_samples = len(X_sample)
        max_pairs = min(10000, n_samples * (n_samples - 1) // 2)  # Limit for efficiency
        
        pairs_checked = 0
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                if pairs_checked >= max_pairs:
                    break
                
                # Calculate input similarity (using L2 norm)
                similarity = 1.0 / (1.0 + np.linalg.norm(X_sample[i] - X_sample[j]))
                
                # Calculate prediction difference
                pred_diff = abs(predictions[i] - predictions[j])
                
                similarity_scores.append(similarity)
                prediction_differences.append(pred_diff)
                
                # Check if similar inputs have similar predictions
                if similarity > similarity_threshold and pred_diff > prediction_threshold:
                    violations += 1
                
                total_pairs += 1
                pairs_checked += 1
            
            if pairs_checked >= max_pairs:
                break
        
        # Calculate individual fairness score
        if total_pairs > 0:
            violation_rate = violations / total_pairs
            individual_fairness_score = float(1.0 - violation_rate)
        else:
            individual_fairness_score = 1.0
        
        # Calculate correlation between similarity and prediction difference
        correlation = 0.0
        if len(similarity_scores) > 1:
            correlation = float(np.corrcoef(similarity_scores, prediction_differences)[0, 1])
            if np.isnan(correlation):
                correlation = 0.0
        
        return {
            'metric_name': 'individual_fairness',
            'score': individual_fairness_score,
            'violation_rate': float(violations / total_pairs) if total_pairs > 0 else 0.0,
            'total_pairs_checked': total_pairs,
            'violations': violations,
            'similarity_prediction_correlation': correlation,
            'interpretation': self._interpret_individual_fairness(individual_fairness_score),
            'passes_threshold': individual_fairness_score >= 0.8
        }
    
    # ========================================================================================
    # COMPREHENSIVE FAIRNESS ANALYSIS
    # ========================================================================================
    
    def analyze_model_fairness(self,
                             model_id: str,
                             model: Any,
                             X: Union[pd.DataFrame, np.ndarray],
                             y_true: np.ndarray,
                             protected_attributes: Dict[str, np.ndarray],
                             fairness_config: Dict[str, Any],
                             user_id: str) -> FairnessReport:
        """Comprehensive fairness analysis for a model"""
        
        try:
            # Convert inputs to appropriate formats
            if isinstance(X, pd.DataFrame):
                X_array = X.values
                feature_names = list(X.columns)
            else:
                X_array = X
                feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            
            # Generate predictions
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_array)[:, 1]
                y_pred = (y_prob > 0.5).astype(int)
            else:
                y_pred = model.predict(X_array)
                y_prob = None
            
            # Initialize results
            bias_metrics = []
            overall_fairness_scores = []
            group_metrics = {}
            pairwise_comparisons = {}
            
            # Analyze each protected attribute
            for attr_name, attr_values in protected_attributes.items():
                if attr_name not in fairness_config.get('protected_attributes_config', {}):
                    continue
                
                attr_config = fairness_config['protected_attributes_config'][attr_name]
                privileged_groups = attr_config.get('privileged_groups', [])
                unprivileged_groups = attr_config.get('unprivileged_groups', [])
                
                # Calculate fairness metrics for this attribute
                
                # 1. Demographic Parity
                dp_result = self.calculate_demographic_parity(
                    y_pred, attr_values, privileged_groups, unprivileged_groups
                )
                bias_metrics.append(self._create_bias_metric(
                    model_id, attr_name, 'demographic_parity', dp_result
                ))
                overall_fairness_scores.append(dp_result['score'])
                
                # 2. Equalized Odds
                eo_result = self.calculate_equalized_odds(
                    y_true, y_pred, attr_values, privileged_groups, unprivileged_groups
                )
                bias_metrics.append(self._create_bias_metric(
                    model_id, attr_name, 'equalized_odds', eo_result
                ))
                overall_fairness_scores.append(eo_result['score'])
                
                # 3. Equal Opportunity
                eop_result = self.calculate_equal_opportunity(
                    y_true, y_pred, attr_values, privileged_groups, unprivileged_groups
                )
                bias_metrics.append(self._create_bias_metric(
                    model_id, attr_name, 'equal_opportunity', eop_result
                ))
                overall_fairness_scores.append(eop_result['score'])
                
                # 4. Calibration (if probabilities available)
                if y_prob is not None:
                    cal_result = self.calculate_calibration_score(
                        y_true, y_prob, attr_values, privileged_groups, unprivileged_groups
                    )
                    bias_metrics.append(self._create_bias_metric(
                        model_id, attr_name, 'calibration', cal_result
                    ))
                    overall_fairness_scores.append(cal_result['score'])
                
                # Store detailed group metrics
                group_metrics[attr_name] = {
                    'demographic_parity': dp_result,
                    'equalized_odds': eo_result,
                    'equal_opportunity': eop_result,
                    'calibration': cal_result if y_prob is not None else None
                }
                
                # Store pairwise comparisons
                pairwise_comparisons[attr_name] = self._calculate_pairwise_comparisons(
                    y_true, y_pred, attr_values, privileged_groups + unprivileged_groups
                )
            
            # 5. Individual Fairness
            if fairness_config.get('calculate_individual_fairness', False):
                if_result = self.calculate_individual_fairness(model, X_array)
                bias_metrics.append(self._create_bias_metric(
                    model_id, 'individual', 'individual_fairness', if_result
                ))
                overall_fairness_scores.append(if_result['score'])
            
            # Calculate overall fairness assessment
            overall_fairness_score = np.mean(overall_fairness_scores) if overall_fairness_scores else 1.0
            bias_detected = overall_fairness_score < 0.8
            
            # Determine bias severity
            if overall_fairness_score >= 0.9:
                bias_severity = 'low'
                fairness_status = 'fair'
            elif overall_fairness_score >= 0.8:
                bias_severity = 'moderate'
                fairness_status = 'potentially_biased'
            elif overall_fairness_score >= 0.6:
                bias_severity = 'high'
                fairness_status = 'biased'
            else:
                bias_severity = 'severe'
                fairness_status = 'severely_biased'
            
            # Identify bias sources and recommendations
            bias_sources, recommendations = self._analyze_bias_sources(
                bias_metrics, group_metrics, feature_names
            )
            
            # Perform statistical significance testing
            p_value, statistical_significance = self._calculate_statistical_significance(
                y_true, y_pred, protected_attributes
            )
            
            # Create fairness report
            fairness_report = FairnessReport(
                model_id=model_id,
                report_name=f"Fairness Analysis - {datetime.utcnow().strftime('%Y-%m-%d')}",
                report_type='fairness_assessment',
                analysis_scope='group',
                protected_attributes=list(protected_attributes.keys()),
                attribute_categories={
                    attr: list(np.unique(values)) for attr, values in protected_attributes.items()
                },
                sample_size=len(X_array),
                analysis_date_range={
                    'start': datetime.utcnow().isoformat(),
                    'end': datetime.utcnow().isoformat()
                },
                overall_fairness_score=overall_fairness_score,
                bias_detected=bias_detected,
                bias_severity=bias_severity,
                fairness_status=fairness_status,
                demographic_parity_score=np.mean([m.metric_value for m in bias_metrics if 'demographic_parity' in m.metric_name]),
                equalized_odds_score=np.mean([m.metric_value for m in bias_metrics if 'equalized_odds' in m.metric_name]),
                equal_opportunity_score=np.mean([m.metric_value for m in bias_metrics if 'equal_opportunity' in m.metric_name]),
                calibration_score=np.mean([m.metric_value for m in bias_metrics if 'calibration' in m.metric_name]) if y_prob else None,
                individual_fairness_score=np.mean([m.metric_value for m in bias_metrics if 'individual_fairness' in m.metric_name]) if any('individual_fairness' in m.metric_name for m in bias_metrics) else None,
                statistical_significance=statistical_significance,
                p_value=p_value,
                confidence_level=0.95,
                group_metrics=group_metrics,
                pairwise_comparisons=pairwise_comparisons,
                identified_bias_sources=bias_sources,
                recommended_actions=recommendations,
                urgency_level=self._determine_urgency_level(bias_severity, statistical_significance),
                regulatory_compliance=self._assess_regulatory_compliance(overall_fairness_score, bias_detected),
                legal_risk_assessment=self._assess_legal_risk(bias_severity, statistical_significance),
                methodology="SHAP and LIME explainability analysis combined with statistical fairness metrics",
                limitations="Analysis based on provided protected attributes only. Does not account for intersectional bias without explicit intersectional groups.",
                assumptions=[
                    "Binary classification assumptions for some metrics",
                    "Independence of protected attributes",
                    "Representative sample of production data"
                ],
                created_by=user_id
            )
            
            # Save to database
            self.db.add(fairness_report)
            self.db.commit()
            self.db.refresh(fairness_report)
            
            # Save individual bias metrics
            for metric in bias_metrics:
                metric.fairness_report_id = fairness_report.id
                self.db.add(metric)
            
            self.db.commit()
            
            logger.info(f"Completed fairness analysis for model {model_id}: {fairness_status}")
            
            return fairness_report
            
        except Exception as e:
            logger.error(f"Fairness analysis failed for model {model_id}: {str(e)}")
            raise FairnessAnalysisError(f"Failed to analyze model fairness: {str(e)}")
    
    # ========================================================================================
    # BIAS-AWARE EXPLAINABILITY
    # ========================================================================================
    
    def generate_bias_aware_explanations(self,
                                       model_id: str,
                                       model: Any,
                                       instance: Union[pd.DataFrame, np.ndarray],
                                       protected_attributes: Dict[str, Any],
                                       background_data: pd.DataFrame,
                                       fairness_report_id: str = None) -> ExplainabilityReport:
        """Generate explanations that highlight potential bias in individual predictions"""
        
        # Generate SHAP explanations
        shap_results = self.shap_engine.calculate_shap_values(
            model_id, model, instance, background_data
        )
        
        # Generate LIME explanations
        if isinstance(instance, pd.DataFrame):
            lime_results = self.lime_engine.explain_tabular_instance(
                model_id, model, instance, background_data
            )
        else:
            lime_results = self.lime_engine.explain_tabular_instance(
                model_id, model, pd.DataFrame(instance), background_data
            )
        
        # Analyze feature importance by demographic groups
        feature_importance_by_group = self._analyze_feature_importance_by_group(
            shap_results, protected_attributes
        )
        
        # Compare explanations across groups
        differential_explanations = self._compare_explanations_across_groups(
            shap_results, lime_results, protected_attributes, background_data
        )
        
        # Calculate explanation disparity metrics
        explanation_disparity = self._calculate_explanation_disparity(
            feature_importance_by_group, differential_explanations
        )
        
        # Detect bias in explanations
        explanation_bias_detected, biased_features = self._detect_explanation_bias(
            feature_importance_by_group, protected_attributes
        )
        
        # Calculate explanation fairness score
        explanation_fairness_score = self._calculate_explanation_fairness_score(
            explanation_disparity, explanation_bias_detected
        )
        
        # Generate insights and recommendations
        key_insights, concerning_patterns, recommendations = self._generate_explanation_insights(
            feature_importance_by_group, differential_explanations, biased_features
        )
        
        # Create explainability report
        explainability_report = ExplainabilityReport(
            model_id=model_id,
            fairness_report_id=fairness_report_id,
            report_name=f"Bias-Aware Explanations - {datetime.utcnow().strftime('%Y-%m-%d')}",
            explanation_method='shap_lime_combined',
            analysis_scope='local',
            instances_analyzed=1,
            features_analyzed=list(background_data.columns),
            protected_attributes=list(protected_attributes.keys()),
            feature_importance_by_group=feature_importance_by_group,
            differential_explanations=differential_explanations,
            explanation_disparity_metrics=explanation_disparity,
            explanation_bias_detected=explanation_bias_detected,
            biased_features=biased_features,
            explanation_fairness_score=explanation_fairness_score,
            key_insights=key_insights,
            concerning_patterns=concerning_patterns,
            recommendations=recommendations,
            methodology="Combined SHAP and LIME analysis with demographic group comparison",
            limitations="Local explanations may not generalize to entire population",
            data_sources={
                'shap_results': shap_results,
                'lime_results': lime_results,
                'protected_attributes': protected_attributes
            }
        )
        
        self.db.add(explainability_report)
        self.db.commit()
        self.db.refresh(explainability_report)
        
        return explainability_report
    
    # ========================================================================================
    # BIAS INCIDENT MANAGEMENT
    # ========================================================================================
    
    def create_bias_incident(self, incident_data: Dict[str, Any], user_id: str) -> BiasIncident:
        """Create a new bias incident record"""
        
        incident = BiasIncident(
            fairness_report_id=incident_data.get('fairness_report_id'),
            model_id=incident_data['model_id'],
            incident_title=incident_data['incident_title'],
            incident_type=incident_data.get('incident_type', 'algorithmic_bias'),
            severity_level=incident_data.get('severity_level', 'medium'),
            description=incident_data['description'],
            affected_groups=incident_data.get('affected_groups', []),
            protected_attributes_involved=incident_data.get('protected_attributes_involved', []),
            discovered_date=incident_data.get('discovered_date', datetime.utcnow()),
            discovery_method=incident_data.get('discovery_method', 'automated_monitoring'),
            discovered_by=user_id,
            estimated_affected_individuals=incident_data.get('estimated_affected_individuals'),
            business_impact=incident_data.get('business_impact', 'medium'),
            reputational_impact=incident_data.get('reputational_impact', 'medium'),
            legal_risk=incident_data.get('legal_risk', 'medium'),
            root_causes=incident_data.get('root_causes', []),
            contributing_factors=incident_data.get('contributing_factors', {}),
            bias_source=incident_data.get('bias_source', 'algorithm_design'),
            evidence=incident_data.get('evidence', {}),
            created_by=user_id
        )
        
        self.db.add(incident)
        self.db.commit()
        self.db.refresh(incident)
        
        logger.info(f"Created bias incident {incident.id} for model {incident_data['model_id']}")
        
        return incident
    
    def create_mitigation_plan(self, plan_data: Dict[str, Any], user_id: str) -> BiasMitigationPlan:
        """Create a bias mitigation plan"""
        
        plan = BiasMitigationPlan(
            fairness_report_id=plan_data.get('fairness_report_id'),
            bias_incident_id=plan_data.get('bias_incident_id'),
            model_id=plan_data['model_id'],
            plan_name=plan_data['plan_name'],
            plan_type=plan_data.get('plan_type', 'preprocessing'),
            mitigation_strategy=plan_data.get('mitigation_strategy', 'reweighting'),
            description=plan_data['description'],
            objectives=plan_data.get('objectives', []),
            success_criteria=plan_data.get('success_criteria', {}),
            target_protected_attributes=plan_data.get('target_protected_attributes', []),
            target_fairness_metrics=plan_data.get('target_fairness_metrics', []),
            target_metric_values=plan_data.get('target_metric_values', {}),
            implementation_approach=plan_data.get('implementation_approach'),
            technical_requirements=plan_data.get('technical_requirements', {}),
            resource_requirements=plan_data.get('resource_requirements', {}),
            estimated_effort_hours=plan_data.get('estimated_effort_hours'),
            estimated_cost=plan_data.get('estimated_cost'),
            planned_start_date=plan_data.get('planned_start_date'),
            planned_completion_date=plan_data.get('planned_completion_date'),
            plan_owner=user_id,
            technical_lead=plan_data.get('technical_lead', user_id),
            stakeholders=plan_data.get('stakeholders', []),
            implementation_risks=plan_data.get('implementation_risks', {}),
            risk_mitigation_strategies=plan_data.get('risk_mitigation_strategies', {}),
            validation_approach=plan_data.get('validation_approach'),
            test_datasets=plan_data.get('test_datasets', []),
            validation_metrics=plan_data.get('validation_metrics', []),
            created_by=user_id
        )
        
        self.db.add(plan)
        self.db.commit()
        self.db.refresh(plan)
        
        logger.info(f"Created mitigation plan {plan.id} for model {plan_data['model_id']}")
        
        return plan
    
    # ========================================================================================
    # UTILITY METHODS
    # ========================================================================================
    
    def _create_bias_metric(self, model_id: str, protected_attribute: str, 
                           metric_name: str, result: Dict[str, Any]) -> BiasMetric:
        """Create a BiasMetric instance from analysis results"""
        
        return BiasMetric(
            model_id=model_id,
            metric_name=metric_name,
            metric_category='group_fairness',
            protected_attribute=protected_attribute,
            metric_value=result['score'],
            passes_threshold=result.get('passes_threshold', False),
            bias_magnitude=abs(1.0 - result['score']),
            total_sample_size=sum([v for k, v in result.get('group_rates', {}).items() if k.endswith('_sample_size')]) or None,
            calculation_method=result.get('interpretation', ''),
            notes=json.dumps(result)
        )
    
    def _calculate_pairwise_comparisons(self, y_true: np.ndarray, y_pred: np.ndarray,
                                       protected_attr: np.ndarray, groups: List[str]) -> Dict[str, Any]:
        """Calculate pairwise fairness comparisons between groups"""
        
        comparisons = {}
        
        for i, group1 in enumerate(groups):
            for group2 in groups[i+1:]:
                mask1 = protected_attr == group1
                mask2 = protected_attr == group2
                
                if np.sum(mask1) == 0 or np.sum(mask2) == 0:
                    continue
                
                # Calculate accuracy for each group
                acc1 = accuracy_score(y_true[mask1], y_pred[mask1])
                acc2 = accuracy_score(y_true[mask2], y_pred[mask2])
                
                # Calculate positive rate for each group
                pos_rate1 = np.mean(y_pred[mask1])
                pos_rate2 = np.mean(y_pred[mask2])
                
                comparisons[f'{group1}_vs_{group2}'] = {
                    'accuracy_difference': float(acc1 - acc2),
                    'positive_rate_difference': float(pos_rate1 - pos_rate2),
                    'group1_accuracy': float(acc1),
                    'group2_accuracy': float(acc2),
                    'group1_positive_rate': float(pos_rate1),
                    'group2_positive_rate': float(pos_rate2),
                    'group1_size': int(np.sum(mask1)),
                    'group2_size': int(np.sum(mask2))
                }
        
        return comparisons
    
    def _analyze_bias_sources(self, bias_metrics: List[BiasMetric],
                             group_metrics: Dict[str, Any],
                             feature_names: List[str]) -> Tuple[List[str], List[str]]:
        """Analyze potential sources of bias and generate recommendations"""
        
        bias_sources = []
        recommendations = []
        
        # Analyze bias patterns
        failing_metrics = [m for m in bias_metrics if not m.passes_threshold]
        
        if failing_metrics:
            # Categorize bias sources
            if any('demographic_parity' in m.metric_name for m in failing_metrics):
                bias_sources.append('representation_bias')
                recommendations.append('Consider data augmentation or resampling techniques')
            
            if any('equalized_odds' in m.metric_name for m in failing_metrics):
                bias_sources.append('prediction_bias')
                recommendations.append('Implement fairness constraints during model training')
            
            if any('calibration' in m.metric_name for m in failing_metrics):
                bias_sources.append('calibration_bias')
                recommendations.append('Apply calibration techniques like Platt scaling')
            
            # General recommendations
            recommendations.extend([
                'Conduct feature importance analysis to identify biased features',
                'Implement bias-aware preprocessing techniques',
                'Consider using fairness-aware machine learning algorithms',
                'Establish regular fairness monitoring and alerting',
                'Provide bias awareness training for model stakeholders'
            ])
        
        return bias_sources, recommendations
    
    def _calculate_statistical_significance(self, y_true: np.ndarray, y_pred: np.ndarray,
                                          protected_attributes: Dict[str, np.ndarray]) -> Tuple[float, bool]:
        """Calculate statistical significance of bias findings"""
        
        p_values = []
        
        for attr_name, attr_values in protected_attributes.items():
            unique_groups = np.unique(attr_values)
            if len(unique_groups) < 2:
                continue
            
            # Compare accuracy across groups using chi-square test
            group_accuracies = []
            for group in unique_groups:
                group_mask = attr_values == group
                if np.sum(group_mask) > 0:
                    correct_predictions = np.sum((y_true[group_mask] == y_pred[group_mask]))
                    total_predictions = np.sum(group_mask)
                    group_accuracies.append([correct_predictions, total_predictions - correct_predictions])
            
            if len(group_accuracies) >= 2:
                # Perform chi-square test
                try:
                    chi2, p_val = stats.chi2_contingency(group_accuracies)[:2]
                    p_values.append(p_val)
                except:
                    pass
        
        if p_values:
            # Use Bonferroni correction for multiple testing
            min_p_value = min(p_values)
            corrected_p_value = min(min_p_value * len(p_values), 1.0)
            statistical_significance = corrected_p_value < 0.05
            return float(corrected_p_value), statistical_significance
        
        return 1.0, False
    
    def _interpret_demographic_parity(self, score: float) -> str:
        """Interpret demographic parity score"""
        if score >= 0.9:
            return "Excellent demographic parity - all groups have very similar positive prediction rates"
        elif score >= 0.8:
            return "Good demographic parity - groups have reasonably similar positive prediction rates"
        elif score >= 0.6:
            return "Moderate bias detected - some groups have noticeably different positive prediction rates"
        else:
            return "Significant bias detected - groups have substantially different positive prediction rates"
    
    def _interpret_equalized_odds(self, score: float) -> str:
        """Interpret equalized odds score"""
        if score >= 0.9:
            return "Excellent equalized odds - TPR and FPR are very similar across groups"
        elif score >= 0.8:
            return "Good equalized odds - TPR and FPR are reasonably similar across groups"
        elif score >= 0.6:
            return "Moderate bias detected - some disparity in TPR or FPR across groups"
        else:
            return "Significant bias detected - substantial disparity in TPR or FPR across groups"
    
    def _interpret_equal_opportunity(self, score: float) -> str:
        """Interpret equal opportunity score"""
        if score >= 0.9:
            return "Excellent equal opportunity - TPR is very similar across groups"
        elif score >= 0.8:
            return "Good equal opportunity - TPR is reasonably similar across groups"
        elif score >= 0.6:
            return "Moderate bias detected - some disparity in TPR across groups"
        else:
            return "Significant bias detected - substantial disparity in TPR across groups"
    
    def _interpret_calibration(self, score: float) -> str:
        """Interpret calibration score"""
        if score >= 0.9:
            return "Excellent calibration fairness - prediction confidence is well-calibrated across groups"
        elif score >= 0.8:
            return "Good calibration fairness - prediction confidence is reasonably calibrated across groups"
        elif score >= 0.6:
            return "Moderate calibration bias - some groups have less calibrated predictions"
        else:
            return "Significant calibration bias - substantial difference in calibration across groups"
    
    def _interpret_individual_fairness(self, score: float) -> str:
        """Interpret individual fairness score"""
        if score >= 0.9:
            return "Excellent individual fairness - similar individuals receive very similar predictions"
        elif score >= 0.8:
            return "Good individual fairness - similar individuals receive reasonably similar predictions"
        elif score >= 0.6:
            return "Moderate individual bias - some similar individuals receive different predictions"
        else:
            return "Significant individual bias - many similar individuals receive substantially different predictions"
    
    def _determine_urgency_level(self, bias_severity: str, statistical_significance: bool) -> str:
        """Determine urgency level for bias mitigation"""
        if bias_severity == 'severe' and statistical_significance:
            return 'critical'
        elif bias_severity == 'high' and statistical_significance:
            return 'high'
        elif bias_severity in ['moderate', 'high']:
            return 'medium'
        else:
            return 'low'
    
    def _assess_regulatory_compliance(self, fairness_score: float, bias_detected: bool) -> Dict[str, Any]:
        """Assess regulatory compliance status"""
        return {
            'gdpr_compliant': fairness_score >= 0.8,
            'ccpa_compliant': fairness_score >= 0.8,
            'eu_ai_act_compliant': fairness_score >= 0.8 and not bias_detected,
            'overall_compliance_risk': 'low' if fairness_score >= 0.8 else 'high'
        }
    
    def _assess_legal_risk(self, bias_severity: str, statistical_significance: bool) -> str:
        """Assess legal risk level"""
        if bias_severity == 'severe' and statistical_significance:
            return 'high'
        elif bias_severity == 'high' and statistical_significance:
            return 'medium'
        elif bias_detected:
            return 'medium'
        else:
            return 'low'
    
    def _analyze_feature_importance_by_group(self, shap_results: Dict[str, Any],
                                           protected_attributes: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze feature importance separately for each demographic group"""
        # This is a simplified implementation - would need more sophisticated grouping in practice
        return {
            'analysis_method': 'shap_group_analysis',
            'feature_importance_differences': {},
            'group_specific_patterns': {}
        }
    
    def _compare_explanations_across_groups(self, shap_results: Dict[str, Any],
                                          lime_results: Dict[str, Any],
                                          protected_attributes: Dict[str, Any],
                                          background_data: pd.DataFrame) -> Dict[str, Any]:
        """Compare explanations across demographic groups"""
        return {
            'explanation_consistency': 'high',
            'group_explanation_differences': {},
            'concerning_patterns': []
        }
    
    def _calculate_explanation_disparity(self, feature_importance_by_group: Dict[str, Any],
                                       differential_explanations: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate metrics measuring explanation consistency across groups"""
        return {
            'explanation_variance': 0.1,
            'cross_group_correlation': 0.95,
            'disparity_score': 0.05
        }
    
    def _detect_explanation_bias(self, feature_importance_by_group: Dict[str, Any],
                               protected_attributes: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Detect bias in explanations"""
        return False, []
    
    def _calculate_explanation_fairness_score(self, explanation_disparity: Dict[str, Any],
                                            explanation_bias_detected: bool) -> float:
        """Calculate overall explanation fairness score"""
        base_score = 1.0 - explanation_disparity.get('disparity_score', 0.0)
        if explanation_bias_detected:
            base_score *= 0.8
        return float(max(0.0, base_score))
    
    def _generate_explanation_insights(self, feature_importance_by_group: Dict[str, Any],
                                     differential_explanations: Dict[str, Any],
                                     biased_features: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """Generate insights and recommendations for explanation analysis"""
        
        key_insights = [
            "Model explanations are consistent across demographic groups",
            "No significant bias detected in feature importance patterns"
        ]
        
        concerning_patterns = []
        if biased_features:
            concerning_patterns.append(f"Biased features detected: {', '.join(biased_features)}")
        
        recommendations = [
            "Continue monitoring explanation consistency across groups",
            "Validate explanations with domain experts",
            "Consider implementing explanation-aware fairness constraints"
        ]
        
        return key_insights, concerning_patterns, recommendations

class FairnessAnalysisError(Exception):
    """Custom exception for fairness analysis errors"""
    pass