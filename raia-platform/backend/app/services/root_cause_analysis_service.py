"""
Root Cause Analysis Service
Automated root cause analysis for ML model performance issues and anomalies
"""

import asyncio
import json
import logging
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import uuid
from collections import defaultdict, deque
from enum import Enum

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import get_database

logger = logging.getLogger(__name__)
settings = get_settings()

class AnalysisType(Enum):
    PERFORMANCE_DROP = "performance_drop"
    DATA_DRIFT = "data_drift"
    PREDICTION_ANOMALY = "prediction_anomaly"
    BIAS_DETECTION = "bias_detection"
    ERROR_PATTERN = "error_pattern"
    LATENCY_SPIKE = "latency_spike"
    SYSTEM_ISSUE = "system_issue"

class SeverityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RootCauseAnalysisService:
    """
    Advanced root cause analysis service for ML model issues
    """
    
    def __init__(self):
        self.analysis_cache = {}
        self.issue_patterns = {}
        self.anomaly_detectors = {}
        self.performance_baselines = {}
        self.diagnostic_rules = self._initialize_diagnostic_rules()
        self.investigation_history = defaultdict(list)
        
    def _initialize_diagnostic_rules(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize diagnostic rules for different types of issues
        """
        return {
            'performance_drop': {
                'threshold': 0.05,  # 5% performance drop
                'window_size': 24,  # hours
                'min_samples': 100,
                'checks': [
                    'data_quality_degradation',
                    'feature_distribution_shift',
                    'label_distribution_change',
                    'infrastructure_issues',
                    'model_staleness'
                ]
            },
            'data_drift': {
                'statistical_threshold': 0.05,
                'js_divergence_threshold': 0.1,
                'checks': [
                    'feature_drift_analysis',
                    'correlation_changes',
                    'new_categorical_values',
                    'missing_value_patterns',
                    'outlier_detection'
                ]
            },
            'prediction_anomaly': {
                'anomaly_threshold': 0.95,  # 95th percentile
                'batch_size_threshold': 0.1,  # 10% anomalous predictions
                'checks': [
                    'prediction_distribution_analysis',
                    'confidence_score_analysis',
                    'input_data_validation',
                    'model_behavior_consistency'
                ]
            },
            'bias_detection': {
                'fairness_threshold': 0.1,
                'demographic_parity_threshold': 0.1,
                'checks': [
                    'protected_attribute_analysis',
                    'intersectional_bias_check',
                    'historical_bias_trends',
                    'proxy_feature_detection'
                ]
            }
        }
    
    async def investigate_issue(
        self,
        issue_id: str,
        issue_type: AnalysisType,
        model_id: str,
        issue_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Investigate a specific issue using automated root cause analysis
        
        Args:
            issue_id: Unique identifier for the issue
            issue_type: Type of issue to investigate
            model_id: Model identifier
            issue_data: Data related to the issue
            context: Additional context information
        
        Returns:
            Comprehensive root cause analysis results
        """
        try:
            investigation_id = str(uuid.uuid4())
            
            investigation = {
                'investigation_id': investigation_id,
                'issue_id': issue_id,
                'issue_type': issue_type.value,
                'model_id': model_id,
                'start_time': datetime.utcnow(),
                'status': 'in_progress',
                'severity': SeverityLevel.MEDIUM.value,
                'root_causes': [],
                'contributing_factors': [],
                'evidence': {},
                'recommendations': [],
                'confidence_score': 0.0,
                'investigation_steps': []
            }
            
            # Step 1: Initial triage and severity assessment
            severity_assessment = await self._assess_issue_severity(
                issue_type, issue_data, model_id
            )
            investigation['severity'] = severity_assessment['severity']
            investigation['investigation_steps'].append({
                'step': 'severity_assessment',
                'timestamp': datetime.utcnow().isoformat(),
                'result': severity_assessment
            })
            
            # Step 2: Collect relevant data and metrics
            data_collection = await self._collect_investigation_data(
                issue_type, model_id, issue_data, context
            )
            investigation['evidence']['collected_data'] = data_collection
            investigation['investigation_steps'].append({
                'step': 'data_collection',
                'timestamp': datetime.utcnow().isoformat(),
                'result': {'data_sources': list(data_collection.keys())}
            })
            
            # Step 3: Run diagnostic checks based on issue type
            diagnostic_results = await self._run_diagnostic_checks(
                issue_type, data_collection, model_id
            )
            investigation['evidence']['diagnostic_results'] = diagnostic_results
            investigation['investigation_steps'].append({
                'step': 'diagnostic_analysis',
                'timestamp': datetime.utcnow().isoformat(),
                'result': diagnostic_results
            })
            
            # Step 4: Pattern matching against known issues
            pattern_analysis = await self._analyze_issue_patterns(
                issue_type, diagnostic_results, model_id
            )
            investigation['evidence']['pattern_analysis'] = pattern_analysis
            investigation['investigation_steps'].append({
                'step': 'pattern_analysis',
                'timestamp': datetime.utcnow().isoformat(),
                'result': pattern_analysis
            })
            
            # Step 5: Statistical analysis and hypothesis testing
            statistical_analysis = await self._perform_statistical_analysis(
                issue_type, data_collection, diagnostic_results
            )
            investigation['evidence']['statistical_analysis'] = statistical_analysis
            investigation['investigation_steps'].append({
                'step': 'statistical_analysis',
                'timestamp': datetime.utcnow().isoformat(),
                'result': statistical_analysis
            })
            
            # Step 6: Identify root causes
            root_cause_identification = await self._identify_root_causes(
                investigation['evidence'], issue_type
            )
            investigation['root_causes'] = root_cause_identification['root_causes']
            investigation['contributing_factors'] = root_cause_identification['contributing_factors']
            investigation['confidence_score'] = root_cause_identification['confidence_score']
            investigation['investigation_steps'].append({
                'step': 'root_cause_identification',
                'timestamp': datetime.utcnow().isoformat(),
                'result': root_cause_identification
            })
            
            # Step 7: Generate actionable recommendations
            recommendations = await self._generate_recommendations(
                investigation['root_causes'],
                investigation['contributing_factors'],
                issue_type,
                investigation['severity']
            )
            investigation['recommendations'] = recommendations
            investigation['investigation_steps'].append({
                'step': 'recommendation_generation',
                'timestamp': datetime.utcnow().isoformat(),
                'result': {'recommendation_count': len(recommendations)}
            })
            
            investigation['status'] = 'completed'
            investigation['end_time'] = datetime.utcnow()
            investigation['duration_seconds'] = (
                investigation['end_time'] - investigation['start_time']
            ).total_seconds()
            
            # Store investigation
            self.investigation_history[model_id].append(investigation)
            self.analysis_cache[investigation_id] = investigation
            
            return {
                'status': 'success',
                'investigation_id': investigation_id,
                **investigation
            }
            
        except Exception as e:
            logger.error(f"Failed to investigate issue {issue_id}: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'issue_id': issue_id,
                'investigation_id': investigation_id if 'investigation_id' in locals() else None
            }
    
    async def _assess_issue_severity(
        self,
        issue_type: AnalysisType,
        issue_data: Dict[str, Any],
        model_id: str
    ) -> Dict[str, Any]:
        """
        Assess the severity of an issue based on type and impact
        """
        try:
            severity_factors = {
                'performance_impact': 0,
                'user_impact': 0,
                'business_impact': 0,
                'frequency': 0,
                'trend': 0
            }
            
            # Performance impact assessment
            if issue_type == AnalysisType.PERFORMANCE_DROP:
                performance_drop = issue_data.get('performance_drop', 0)
                if performance_drop > 0.2:  # 20% drop
                    severity_factors['performance_impact'] = 4
                elif performance_drop > 0.1:  # 10% drop
                    severity_factors['performance_impact'] = 3
                elif performance_drop > 0.05:  # 5% drop
                    severity_factors['performance_impact'] = 2
                else:
                    severity_factors['performance_impact'] = 1
            
            # User impact assessment
            affected_users = issue_data.get('affected_users', 0)
            total_users = issue_data.get('total_users', 1)
            user_impact_ratio = affected_users / total_users
            
            if user_impact_ratio > 0.5:
                severity_factors['user_impact'] = 4
            elif user_impact_ratio > 0.2:
                severity_factors['user_impact'] = 3
            elif user_impact_ratio > 0.05:
                severity_factors['user_impact'] = 2
            else:
                severity_factors['user_impact'] = 1
            
            # Business impact (could be revenue, SLA, etc.)
            business_impact = issue_data.get('business_impact_score', 0)
            severity_factors['business_impact'] = min(4, max(1, int(business_impact)))
            
            # Frequency assessment
            frequency = issue_data.get('frequency', 'low')
            frequency_map = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
            severity_factors['frequency'] = frequency_map.get(frequency, 2)
            
            # Trend assessment (is it getting worse?)
            trend = issue_data.get('trend', 'stable')
            trend_map = {'improving': 1, 'stable': 2, 'worsening': 3, 'rapidly_worsening': 4}
            severity_factors['trend'] = trend_map.get(trend, 2)
            
            # Calculate overall severity score
            weights = {
                'performance_impact': 0.3,
                'user_impact': 0.25,
                'business_impact': 0.25,
                'frequency': 0.1,
                'trend': 0.1
            }
            
            overall_score = sum(
                severity_factors[factor] * weights[factor]
                for factor in severity_factors
            )
            
            # Map to severity levels
            if overall_score >= 3.5:
                severity = SeverityLevel.CRITICAL.value
            elif overall_score >= 2.5:
                severity = SeverityLevel.HIGH.value
            elif overall_score >= 1.5:
                severity = SeverityLevel.MEDIUM.value
            else:
                severity = SeverityLevel.LOW.value
            
            return {
                'severity': severity,
                'severity_score': overall_score,
                'severity_factors': severity_factors,
                'assessment_rationale': f"Overall severity score: {overall_score:.2f}"
            }
            
        except Exception as e:
            logger.error(f"Error assessing issue severity: {e}")
            return {
                'severity': SeverityLevel.MEDIUM.value,
                'severity_score': 2.0,
                'error': str(e)
            }
    
    async def _collect_investigation_data(
        self,
        issue_type: AnalysisType,
        model_id: str,
        issue_data: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Collect relevant data for investigation
        """
        try:
            collected_data = {}
            
            # Time range for data collection
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=7)  # Look back 7 days
            
            # Performance metrics
            collected_data['performance_metrics'] = {
                'accuracy_history': await self._get_performance_history(
                    model_id, 'accuracy', start_time, end_time
                ),
                'latency_history': await self._get_performance_history(
                    model_id, 'latency', start_time, end_time
                ),
                'error_rate_history': await self._get_performance_history(
                    model_id, 'error_rate', start_time, end_time
                ),
                'throughput_history': await self._get_performance_history(
                    model_id, 'throughput', start_time, end_time
                )
            }
            
            # Data quality metrics
            collected_data['data_quality'] = {
                'missing_values_trend': await self._get_missing_values_trend(model_id, start_time, end_time),
                'duplicate_rate_trend': await self._get_duplicate_trend(model_id, start_time, end_time),
                'schema_changes': await self._get_schema_changes(model_id, start_time, end_time),
                'data_volume_trend': await self._get_data_volume_trend(model_id, start_time, end_time)
            }
            
            # Feature statistics
            collected_data['feature_statistics'] = {
                'feature_distributions': await self._get_feature_distributions(model_id, start_time, end_time),
                'correlation_changes': await self._get_correlation_changes(model_id, start_time, end_time),
                'feature_importance_drift': await self._get_feature_importance_drift(model_id, start_time, end_time)
            }
            
            # System metrics
            collected_data['system_metrics'] = {
                'resource_utilization': await self._get_resource_utilization(model_id, start_time, end_time),
                'deployment_history': await self._get_deployment_history(model_id, start_time, end_time),
                'infrastructure_events': await self._get_infrastructure_events(model_id, start_time, end_time)
            }
            
            # Prediction patterns
            collected_data['prediction_patterns'] = {
                'prediction_distribution': await self._get_prediction_distribution(model_id, start_time, end_time),
                'confidence_score_trends': await self._get_confidence_trends(model_id, start_time, end_time),
                'anomalous_predictions': await self._get_anomalous_predictions(model_id, start_time, end_time)
            }
            
            # External factors
            collected_data['external_factors'] = {
                'seasonal_patterns': await self._analyze_seasonal_patterns(model_id, start_time, end_time),
                'upstream_dependencies': await self._check_upstream_dependencies(model_id, start_time, end_time),
                'business_events': context.get('business_events', []) if context else []
            }
            
            return collected_data
            
        except Exception as e:
            logger.error(f"Error collecting investigation data: {e}")
            return {'error': str(e)}
    
    async def _run_diagnostic_checks(
        self,
        issue_type: AnalysisType,
        collected_data: Dict[str, Any],
        model_id: str
    ) -> Dict[str, Any]:
        """
        Run diagnostic checks based on issue type
        """
        try:
            diagnostic_results = {}
            rules = self.diagnostic_rules.get(issue_type.value, {})
            checks = rules.get('checks', [])
            
            for check in checks:
                check_result = await self._run_individual_check(
                    check, collected_data, model_id, rules
                )
                diagnostic_results[check] = check_result
            
            return diagnostic_results
            
        except Exception as e:
            logger.error(f"Error running diagnostic checks: {e}")
            return {'error': str(e)}
    
    async def _run_individual_check(
        self,
        check_name: str,
        collected_data: Dict[str, Any],
        model_id: str,
        rules: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run an individual diagnostic check
        """
        try:
            if check_name == 'data_quality_degradation':
                return await self._check_data_quality_degradation(collected_data, rules)
            
            elif check_name == 'feature_distribution_shift':
                return await self._check_feature_distribution_shift(collected_data, rules)
            
            elif check_name == 'label_distribution_change':
                return await self._check_label_distribution_change(collected_data, rules)
            
            elif check_name == 'infrastructure_issues':
                return await self._check_infrastructure_issues(collected_data, rules)
            
            elif check_name == 'model_staleness':
                return await self._check_model_staleness(collected_data, model_id, rules)
            
            elif check_name == 'feature_drift_analysis':
                return await self._analyze_feature_drift(collected_data, rules)
            
            elif check_name == 'correlation_changes':
                return await self._analyze_correlation_changes(collected_data, rules)
            
            elif check_name == 'new_categorical_values':
                return await self._check_new_categorical_values(collected_data, rules)
            
            elif check_name == 'missing_value_patterns':
                return await self._analyze_missing_value_patterns(collected_data, rules)
            
            elif check_name == 'outlier_detection':
                return await self._detect_outliers(collected_data, rules)
            
            elif check_name == 'prediction_distribution_analysis':
                return await self._analyze_prediction_distribution(collected_data, rules)
            
            elif check_name == 'confidence_score_analysis':
                return await self._analyze_confidence_scores(collected_data, rules)
            
            elif check_name == 'input_data_validation':
                return await self._validate_input_data(collected_data, rules)
            
            elif check_name == 'model_behavior_consistency':
                return await self._check_model_behavior_consistency(collected_data, rules)
            
            elif check_name == 'protected_attribute_analysis':
                return await self._analyze_protected_attributes(collected_data, rules)
            
            elif check_name == 'intersectional_bias_check':
                return await self._check_intersectional_bias(collected_data, rules)
            
            elif check_name == 'historical_bias_trends':
                return await self._analyze_historical_bias_trends(collected_data, rules)
            
            elif check_name == 'proxy_feature_detection':
                return await self._detect_proxy_features(collected_data, rules)
            
            else:
                return {
                    'status': 'not_implemented',
                    'message': f'Check {check_name} not implemented yet'
                }
            
        except Exception as e:
            logger.error(f"Error in check {check_name}: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'check_name': check_name
            }
    
    async def _check_data_quality_degradation(self, collected_data: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check for data quality degradation
        """
        try:
            data_quality = collected_data.get('data_quality', {})
            
            issues_found = []
            severity_score = 0
            
            # Check missing values trend
            missing_trend = data_quality.get('missing_values_trend', {})
            if missing_trend.get('increasing_trend', False):
                increase_rate = missing_trend.get('increase_rate', 0)
                if increase_rate > 0.05:  # 5% increase
                    issues_found.append(f"Missing values increased by {increase_rate:.2%}")
                    severity_score += 2
                elif increase_rate > 0.02:  # 2% increase
                    issues_found.append(f"Moderate increase in missing values: {increase_rate:.2%}")
                    severity_score += 1
            
            # Check duplicate rate
            duplicate_trend = data_quality.get('duplicate_rate_trend', {})
            if duplicate_trend.get('increasing_trend', False):
                duplicate_rate = duplicate_trend.get('current_rate', 0)
                if duplicate_rate > 0.1:  # 10% duplicates
                    issues_found.append(f"High duplicate rate: {duplicate_rate:.2%}")
                    severity_score += 3
                elif duplicate_rate > 0.05:  # 5% duplicates
                    issues_found.append(f"Moderate duplicate rate: {duplicate_rate:.2%}")
                    severity_score += 1
            
            # Check schema changes
            schema_changes = data_quality.get('schema_changes', [])
            if schema_changes:
                issues_found.append(f"Schema changes detected: {len(schema_changes)} changes")
                severity_score += len(schema_changes)
            
            return {
                'status': 'completed',
                'issues_found': issues_found,
                'severity_score': min(severity_score, 10),  # Cap at 10
                'degradation_detected': len(issues_found) > 0,
                'details': {
                    'missing_values': missing_trend,
                    'duplicates': duplicate_trend,
                    'schema_changes': schema_changes
                }
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    async def _check_feature_distribution_shift(self, collected_data: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check for feature distribution shifts
        """
        try:
            feature_stats = collected_data.get('feature_statistics', {})
            distributions = feature_stats.get('feature_distributions', {})
            
            shifted_features = []
            shift_scores = {}
            
            for feature, dist_data in distributions.items():
                if 'drift_score' in dist_data:
                    drift_score = dist_data['drift_score']
                    threshold = rules.get('statistical_threshold', 0.05)
                    
                    if drift_score > threshold:
                        shifted_features.append(feature)
                        shift_scores[feature] = drift_score
            
            return {
                'status': 'completed',
                'shifted_features': shifted_features,
                'shift_scores': shift_scores,
                'distribution_shift_detected': len(shifted_features) > 0,
                'most_affected_feature': max(shift_scores.items(), key=lambda x: x[1])[0] if shift_scores else None
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    async def _identify_root_causes(
        self,
        evidence: Dict[str, Any],
        issue_type: AnalysisType
    ) -> Dict[str, Any]:
        """
        Identify root causes based on collected evidence
        """
        try:
            root_causes = []
            contributing_factors = []
            confidence_scores = []
            
            diagnostic_results = evidence.get('diagnostic_results', {})
            
            # Analyze diagnostic results to identify root causes
            for check_name, result in diagnostic_results.items():
                if result.get('status') == 'completed':
                    # Data quality issues
                    if check_name == 'data_quality_degradation' and result.get('degradation_detected', False):
                        root_cause = {
                            'cause': 'Data Quality Degradation',
                            'description': 'Deterioration in data quality metrics',
                            'evidence': result.get('issues_found', []),
                            'confidence': min(0.9, result.get('severity_score', 0) / 10),
                            'category': 'data'
                        }
                        root_causes.append(root_cause)
                        confidence_scores.append(root_cause['confidence'])
                    
                    # Feature distribution shift
                    if check_name == 'feature_distribution_shift' and result.get('distribution_shift_detected', False):
                        shifted_features = result.get('shifted_features', [])
                        if len(shifted_features) > 3:  # More than 3 features shifted - likely root cause
                            root_cause = {
                                'cause': 'Widespread Feature Distribution Shift',
                                'description': f'Distribution shift detected in {len(shifted_features)} features',
                                'evidence': shifted_features,
                                'confidence': 0.8,
                                'category': 'data_drift'
                            }
                            root_causes.append(root_cause)
                            confidence_scores.append(0.8)
                        elif len(shifted_features) > 0:  # Few features shifted - contributing factor
                            contributing_factor = {
                                'factor': 'Limited Feature Distribution Shift',
                                'description': f'Distribution shift in {len(shifted_features)} features',
                                'features': shifted_features,
                                'impact': 'medium'
                            }
                            contributing_factors.append(contributing_factor)
                    
                    # Infrastructure issues
                    if check_name == 'infrastructure_issues' and result.get('issues_detected', False):
                        root_cause = {
                            'cause': 'Infrastructure Issues',
                            'description': 'System or infrastructure problems detected',
                            'evidence': result.get('issues', []),
                            'confidence': 0.9,
                            'category': 'infrastructure'
                        }
                        root_causes.append(root_cause)
                        confidence_scores.append(0.9)
                    
                    # Model staleness
                    if check_name == 'model_staleness' and result.get('stale_model_detected', False):
                        days_since_training = result.get('days_since_training', 0)
                        if days_since_training > 90:  # More than 90 days - root cause
                            root_cause = {
                                'cause': 'Model Staleness',
                                'description': f'Model not retrained for {days_since_training} days',
                                'evidence': [f'Last training: {days_since_training} days ago'],
                                'confidence': 0.7,
                                'category': 'model'
                            }
                            root_causes.append(root_cause)
                            confidence_scores.append(0.7)
                        else:  # Contributing factor
                            contributing_factor = {
                                'factor': 'Model Age',
                                'description': f'Model is {days_since_training} days old',
                                'impact': 'low' if days_since_training < 30 else 'medium'
                            }
                            contributing_factors.append(contributing_factor)
            
            # Calculate overall confidence
            overall_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
            
            # If no clear root causes found, check for complex interactions
            if not root_causes:
                root_causes.append({
                    'cause': 'Complex Multi-factor Issue',
                    'description': 'Issue likely caused by interaction of multiple factors',
                    'evidence': [f'{len(contributing_factors)} contributing factors identified'],
                    'confidence': 0.5,
                    'category': 'complex'
                })
                overall_confidence = 0.5
            
            return {
                'root_causes': root_causes,
                'contributing_factors': contributing_factors,
                'confidence_score': overall_confidence,
                'analysis_summary': {
                    'total_root_causes': len(root_causes),
                    'total_contributing_factors': len(contributing_factors),
                    'primary_category': root_causes[0]['category'] if root_causes else 'unknown'
                }
            }
            
        except Exception as e:
            logger.error(f"Error identifying root causes: {e}")
            return {
                'root_causes': [],
                'contributing_factors': [],
                'confidence_score': 0.0,
                'error': str(e)
            }
    
    async def _generate_recommendations(
        self,
        root_causes: List[Dict[str, Any]],
        contributing_factors: List[Dict[str, Any]],
        issue_type: AnalysisType,
        severity: str
    ) -> List[Dict[str, Any]]:
        """
        Generate actionable recommendations based on root causes
        """
        try:
            recommendations = []
            
            for root_cause in root_causes:
                cause_type = root_cause.get('category', 'unknown')
                
                if cause_type == 'data':
                    recommendations.extend([
                        {
                            'priority': 'high',
                            'action': 'Implement Data Quality Monitoring',
                            'description': 'Set up automated data quality checks and alerts',
                            'timeline': 'immediate',
                            'effort': 'medium',
                            'impact': 'high'
                        },
                        {
                            'priority': 'high',
                            'action': 'Investigate Data Pipeline',
                            'description': 'Review data collection and preprocessing pipeline',
                            'timeline': 'immediate',
                            'effort': 'high',
                            'impact': 'high'
                        }
                    ])
                
                elif cause_type == 'data_drift':
                    recommendations.extend([
                        {
                            'priority': 'high',
                            'action': 'Retrain Model',
                            'description': 'Retrain model with recent data to adapt to distribution changes',
                            'timeline': 'short_term',
                            'effort': 'high',
                            'impact': 'high'
                        },
                        {
                            'priority': 'medium',
                            'action': 'Implement Drift Detection',
                            'description': 'Set up automated drift detection and alerting',
                            'timeline': 'medium_term',
                            'effort': 'medium',
                            'impact': 'medium'
                        }
                    ])
                
                elif cause_type == 'infrastructure':
                    recommendations.extend([
                        {
                            'priority': 'critical',
                            'action': 'Fix Infrastructure Issues',
                            'description': 'Address identified infrastructure problems immediately',
                            'timeline': 'immediate',
                            'effort': 'high',
                            'impact': 'high'
                        },
                        {
                            'priority': 'high',
                            'action': 'Improve Monitoring',
                            'description': 'Enhance infrastructure monitoring and alerting',
                            'timeline': 'short_term',
                            'effort': 'medium',
                            'impact': 'medium'
                        }
                    ])
                
                elif cause_type == 'model':
                    recommendations.extend([
                        {
                            'priority': 'high',
                            'action': 'Schedule Model Retraining',
                            'description': 'Establish regular model retraining schedule',
                            'timeline': 'short_term',
                            'effort': 'medium',
                            'impact': 'high'
                        },
                        {
                            'priority': 'medium',
                            'action': 'Implement Model Versioning',
                            'description': 'Set up proper model versioning and rollback capabilities',
                            'timeline': 'medium_term',
                            'effort': 'medium',
                            'impact': 'medium'
                        }
                    ])
            
            # Add general recommendations based on severity
            if severity in ['high', 'critical']:
                recommendations.extend([
                    {
                        'priority': 'high',
                        'action': 'Increase Monitoring Frequency',
                        'description': 'Temporarily increase monitoring frequency until issue is resolved',
                        'timeline': 'immediate',
                        'effort': 'low',
                        'impact': 'medium'
                    },
                    {
                        'priority': 'high',
                        'action': 'Set Up War Room',
                        'description': 'Establish dedicated team for issue resolution',
                        'timeline': 'immediate',
                        'effort': 'high',
                        'impact': 'high'
                    }
                ])
            
            # Sort by priority
            priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
            recommendations.sort(key=lambda x: priority_order.get(x['priority'], 99))
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return [{
                'priority': 'high',
                'action': 'Manual Investigation Required',
                'description': f'Automated analysis failed: {str(e)}',
                'timeline': 'immediate',
                'effort': 'high',
                'impact': 'unknown'
            }]
    
    # Placeholder methods for data collection (would integrate with actual monitoring systems)
    async def _get_performance_history(self, model_id: str, metric: str, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Get performance history for a metric"""
        # Placeholder - would integrate with actual monitoring system
        return {
            'metric': metric,
            'time_range': f'{start_time} to {end_time}',
            'data_points': 100,
            'trend': 'declining',
            'current_value': 0.85,
            'baseline_value': 0.90
        }
    
    async def _get_missing_values_trend(self, model_id: str, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Get missing values trend"""
        return {
            'increasing_trend': True,
            'increase_rate': 0.03,
            'current_rate': 0.08,
            'baseline_rate': 0.05
        }
    
    async def _get_duplicate_trend(self, model_id: str, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Get duplicate rate trend"""
        return {
            'increasing_trend': False,
            'current_rate': 0.02,
            'baseline_rate': 0.01
        }
    
    async def _get_schema_changes(self, model_id: str, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Get schema changes"""
        return []
    
    async def _get_data_volume_trend(self, model_id: str, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Get data volume trend"""
        return {
            'trend': 'stable',
            'current_volume': 10000,
            'average_volume': 9500
        }
    
    async def _get_feature_distributions(self, model_id: str, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Get feature distributions"""
        return {
            'feature1': {'drift_score': 0.02, 'distribution_type': 'normal'},
            'feature2': {'drift_score': 0.08, 'distribution_type': 'uniform'},
            'feature3': {'drift_score': 0.15, 'distribution_type': 'skewed'}
        }
    
    async def _get_correlation_changes(self, model_id: str, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Get correlation changes"""
        return {'significant_changes': []}
    
    async def _get_feature_importance_drift(self, model_id: str, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Get feature importance drift"""
        return {'drift_detected': False}
    
    async def _get_resource_utilization(self, model_id: str, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Get resource utilization"""
        return {
            'cpu_usage': {'average': 0.4, 'max': 0.8, 'trend': 'stable'},
            'memory_usage': {'average': 0.6, 'max': 0.9, 'trend': 'increasing'},
            'disk_usage': {'average': 0.3, 'max': 0.5, 'trend': 'stable'}
        }
    
    async def _get_deployment_history(self, model_id: str, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Get deployment history"""
        return []
    
    async def _get_infrastructure_events(self, model_id: str, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Get infrastructure events"""
        return []
    
    async def _get_prediction_distribution(self, model_id: str, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Get prediction distribution"""
        return {'distribution_shift': False}
    
    async def _get_confidence_trends(self, model_id: str, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Get confidence score trends"""
        return {'trend': 'stable', 'average_confidence': 0.8}
    
    async def _get_anomalous_predictions(self, model_id: str, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Get anomalous predictions"""
        return []
    
    async def _analyze_seasonal_patterns(self, model_id: str, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Analyze seasonal patterns"""
        return {'seasonal_effect_detected': False}
    
    async def _check_upstream_dependencies(self, model_id: str, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Check upstream dependencies"""
        return {'dependency_issues': []}
    
    # Additional placeholder check methods
    async def _check_label_distribution_change(self, collected_data: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
        return {'status': 'completed', 'distribution_change_detected': False}
    
    async def _check_infrastructure_issues(self, collected_data: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
        return {'status': 'completed', 'issues_detected': False, 'issues': []}
    
    async def _check_model_staleness(self, collected_data: Dict[str, Any], model_id: str, rules: Dict[str, Any]) -> Dict[str, Any]:
        return {'status': 'completed', 'stale_model_detected': True, 'days_since_training': 45}
    
    async def _analyze_feature_drift(self, collected_data: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
        return {'status': 'completed', 'drift_detected': False}
    
    async def _analyze_correlation_changes(self, collected_data: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
        return {'status': 'completed', 'significant_changes': False}
    
    async def _check_new_categorical_values(self, collected_data: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
        return {'status': 'completed', 'new_values_detected': False}
    
    async def _analyze_missing_value_patterns(self, collected_data: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
        return {'status': 'completed', 'pattern_changes_detected': False}
    
    async def _detect_outliers(self, collected_data: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
        return {'status': 'completed', 'outliers_detected': False}
    
    async def _analyze_prediction_distribution(self, collected_data: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
        return {'status': 'completed', 'distribution_anomaly_detected': False}
    
    async def _analyze_confidence_scores(self, collected_data: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
        return {'status': 'completed', 'confidence_anomaly_detected': False}
    
    async def _validate_input_data(self, collected_data: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
        return {'status': 'completed', 'validation_issues_detected': False}
    
    async def _check_model_behavior_consistency(self, collected_data: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
        return {'status': 'completed', 'inconsistency_detected': False}
    
    async def _analyze_protected_attributes(self, collected_data: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
        return {'status': 'completed', 'bias_detected': False}
    
    async def _check_intersectional_bias(self, collected_data: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
        return {'status': 'completed', 'intersectional_bias_detected': False}
    
    async def _analyze_historical_bias_trends(self, collected_data: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
        return {'status': 'completed', 'bias_trend_detected': False}
    
    async def _detect_proxy_features(self, collected_data: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
        return {'status': 'completed', 'proxy_features_detected': False}
    
    async def _analyze_issue_patterns(self, issue_type: AnalysisType, diagnostic_results: Dict[str, Any], model_id: str) -> Dict[str, Any]:
        """Analyze patterns against known issues"""
        return {'patterns_matched': [], 'similarity_scores': {}}
    
    async def _perform_statistical_analysis(self, issue_type: AnalysisType, collected_data: Dict[str, Any], diagnostic_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical analysis"""
        return {'statistical_tests': {}, 'p_values': {}, 'effect_sizes': {}}
    
    async def get_investigation_history(self, model_id: str, limit: int = 10) -> Dict[str, Any]:
        """
        Get investigation history for a model
        
        Args:
            model_id: Model identifier
            limit: Maximum number of investigations to return
        
        Returns:
            Investigation history
        """
        try:
            investigations = self.investigation_history.get(model_id, [])
            recent_investigations = sorted(
                investigations, 
                key=lambda x: x['start_time'], 
                reverse=True
            )[:limit]
            
            return {
                'status': 'success',
                'model_id': model_id,
                'investigations': recent_investigations,
                'total_investigations': len(investigations)
            }
            
        except Exception as e:
            logger.error(f"Error getting investigation history: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'model_id': model_id
            }

# Global service instance
root_cause_analysis_service = RootCauseAnalysisService()
