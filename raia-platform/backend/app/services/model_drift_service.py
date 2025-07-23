"""
Model Drift Analysis Service
Comprehensive model performance drift detection and impact analysis
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ModelPerformanceBaseline:
    """Baseline model performance metrics"""
    model_id: str
    model_type: str  # 'classification' or 'regression'
    baseline_period: str
    sample_size: int
    
    # Classification metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    roc_auc: Optional[float] = None
    
    # Regression metrics
    mae: Optional[float] = None
    mse: Optional[float] = None
    rmse: Optional[float] = None
    r2: Optional[float] = None
    
    # Additional metrics
    prediction_latency: Optional[float] = None
    throughput: Optional[float] = None
    error_rate: Optional[float] = None
    
    established_at: datetime = None


@dataclass
class ModelDriftDetection:
    """Model drift detection results"""
    model_id: str
    drift_session_id: str
    detection_period: str
    current_sample_size: int
    baseline_sample_size: int
    
    # Drift detection results
    drift_detected: bool
    drift_severity: str  # 'low', 'medium', 'high', 'critical'
    drift_confidence: float
    
    # Performance changes
    performance_degradation: Dict[str, float]  # metric -> % change
    statistical_significance: Dict[str, float]  # metric -> p-value
    
    # Root cause analysis
    primary_causes: List[str]
    data_drift_correlation: Optional[float] = None
    seasonal_pattern_detected: bool = False
    concept_drift_suspected: bool = False
    
    # Impact analysis
    business_impact_score: float
    recommendations: List[str]
    
    detected_at: datetime = None


@dataclass
class ModelDriftImpact:
    """Comprehensive drift impact analysis"""
    drift_session_id: str
    model_id: str
    
    # Performance impact
    baseline_performance: Dict[str, float]
    current_performance: Dict[str, float]
    performance_delta: Dict[str, float]
    performance_trend: Dict[str, List[float]]  # time series
    
    # Business impact
    estimated_cost_impact: float
    affected_predictions: int
    false_positive_increase: float
    false_negative_increase: float
    
    # Technical impact
    latency_impact: float
    throughput_impact: float
    resource_utilization_change: float
    
    # Confidence intervals
    performance_confidence_intervals: Dict[str, Tuple[float, float]]
    
    # Projection
    projected_impact_30d: Dict[str, float]
    retraining_urgency_score: float
    
    analyzed_at: datetime = None


class ModelDriftService:
    """
    Comprehensive model drift detection and analysis service
    """
    
    def __init__(self):
        self.logger = logger.bind(service="model_drift")
        
        # Drift detection thresholds
        self.drift_thresholds = {
            'low': 0.05,      # 5% performance drop
            'medium': 0.10,   # 10% performance drop
            'high': 0.15,     # 15% performance drop
            'critical': 0.25  # 25% performance drop
        }
        
        # Statistical significance threshold
        self.significance_threshold = 0.05
        
        # Minimum sample sizes for reliable detection
        self.min_baseline_samples = 1000
        self.min_current_samples = 500
    
    async def establish_baseline(
        self,
        model_id: str,
        model_type: str,
        historical_data: Dict[str, Any],
        period_days: int = 30
    ) -> ModelPerformanceBaseline:
        """
        Establish performance baseline for a model
        
        Args:
            model_id: Model identifier
            model_type: 'classification' or 'regression'
            historical_data: Historical predictions and actual values
            period_days: Baseline period in days
            
        Returns:
            ModelPerformanceBaseline object
        """
        
        self.logger.info(
            "Establishing model baseline",
            model_id=model_id,
            model_type=model_type,
            period_days=period_days
        )
        
        try:
            y_true = historical_data['actual_values']
            y_pred = historical_data['predictions']
            y_prob = historical_data.get('prediction_probabilities')
            
            sample_size = len(y_true)
            
            if sample_size < self.min_baseline_samples:
                self.logger.warning(
                    "Insufficient samples for baseline",
                    sample_size=sample_size,
                    min_required=self.min_baseline_samples
                )
            
            baseline = ModelPerformanceBaseline(
                model_id=model_id,
                model_type=model_type,
                baseline_period=f"last_{period_days}_days",
                sample_size=sample_size,
                established_at=datetime.utcnow()
            )
            
            if model_type == 'classification':
                baseline.accuracy = accuracy_score(y_true, y_pred)
                baseline.precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                baseline.recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                baseline.f1_score = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                
                if y_prob is not None and len(np.unique(y_true)) == 2:
                    baseline.roc_auc = roc_auc_score(y_true, y_prob)
                    
            elif model_type == 'regression':
                baseline.mae = mean_absolute_error(y_true, y_pred)
                baseline.mse = mean_squared_error(y_true, y_pred)
                baseline.rmse = np.sqrt(baseline.mse)
                baseline.r2 = r2_score(y_true, y_pred)
            
            # Performance metrics
            if 'latencies' in historical_data:
                baseline.prediction_latency = np.mean(historical_data['latencies'])
            
            if 'throughput' in historical_data:
                baseline.throughput = historical_data['throughput']
            
            if 'errors' in historical_data:
                baseline.error_rate = len(historical_data['errors']) / sample_size
            
            self.logger.info(
                "Baseline established successfully",
                model_id=model_id,
                sample_size=sample_size
            )
            
            return baseline
            
        except Exception as e:
            self.logger.error(
                "Failed to establish baseline",
                model_id=model_id,
                error=str(e)
            )
            raise
    
    async def detect_model_drift(
        self,
        model_id: str,
        baseline: ModelPerformanceBaseline,
        current_data: Dict[str, Any],
        data_drift_info: Optional[Dict[str, Any]] = None
    ) -> ModelDriftDetection:
        """
        Detect model performance drift
        
        Args:
            model_id: Model identifier
            baseline: Established performance baseline
            current_data: Current model performance data
            data_drift_info: Optional data drift analysis results
            
        Returns:
            ModelDriftDetection object
        """
        
        drift_session_id = f"drift_{model_id}_{int(datetime.utcnow().timestamp())}"
        
        self.logger.info(
            "Starting drift detection",
            model_id=model_id,
            drift_session_id=drift_session_id
        )
        
        try:
            y_true = current_data['actual_values']
            y_pred = current_data['predictions']
            y_prob = current_data.get('prediction_probabilities')
            
            current_sample_size = len(y_true)
            
            if current_sample_size < self.min_current_samples:
                self.logger.warning(
                    "Insufficient current samples",
                    current_sample_size=current_sample_size
                )
            
            # Calculate current performance metrics
            current_metrics = {}
            if baseline.model_type == 'classification':
                current_metrics['accuracy'] = accuracy_score(y_true, y_pred)
                current_metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                current_metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                current_metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                
                if y_prob is not None and len(np.unique(y_true)) == 2:
                    current_metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
                    
            elif baseline.model_type == 'regression':
                current_metrics['mae'] = mean_absolute_error(y_true, y_pred)
                current_metrics['mse'] = mean_squared_error(y_true, y_pred)
                current_metrics['rmse'] = np.sqrt(current_metrics['mse'])
                current_metrics['r2'] = r2_score(y_true, y_pred)
            
            # Calculate performance degradation
            performance_degradation = {}
            statistical_significance = {}
            
            for metric, current_value in current_metrics.items():
                baseline_value = getattr(baseline, metric, None)
                
                if baseline_value is not None and baseline_value > 0:
                    if baseline.model_type == 'regression' and metric in ['mae', 'mse', 'rmse']:
                        # For error metrics, increase is degradation
                        degradation = (current_value - baseline_value) / baseline_value
                    else:
                        # For performance metrics, decrease is degradation
                        degradation = (baseline_value - current_value) / baseline_value
                    
                    performance_degradation[metric] = degradation
                    
                    # Statistical significance test (simplified)
                    # In practice, you'd need historical variance data
                    p_value = self._calculate_significance(baseline_value, current_value, current_sample_size)
                    statistical_significance[metric] = p_value
            
            # Determine drift severity
            max_degradation = max(performance_degradation.values()) if performance_degradation else 0
            drift_detected, drift_severity = self._classify_drift_severity(max_degradation)
            
            # Calculate confidence
            drift_confidence = min(1.0, max_degradation / self.drift_thresholds['high'])
            
            # Root cause analysis
            primary_causes = self._identify_drift_causes(
                performance_degradation,
                statistical_significance,
                data_drift_info
            )
            
            # Data drift correlation
            data_drift_correlation = None
            if data_drift_info and 'overall_drift_score' in data_drift_info:
                data_drift_correlation = np.corrcoef([max_degradation, data_drift_info['overall_drift_score']])[0, 1]
            
            # Business impact assessment
            business_impact_score = self._calculate_business_impact(
                model_id,
                performance_degradation,
                current_sample_size
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                drift_severity,
                primary_causes,
                performance_degradation,
                business_impact_score
            )
            
            drift_detection = ModelDriftDetection(
                model_id=model_id,
                drift_session_id=drift_session_id,
                detection_period=f"current_vs_baseline",
                current_sample_size=current_sample_size,
                baseline_sample_size=baseline.sample_size,
                drift_detected=drift_detected,
                drift_severity=drift_severity,
                drift_confidence=drift_confidence,
                performance_degradation=performance_degradation,
                statistical_significance=statistical_significance,
                primary_causes=primary_causes,
                data_drift_correlation=data_drift_correlation,
                seasonal_pattern_detected=self._detect_seasonal_patterns(current_data),
                concept_drift_suspected=self._suspect_concept_drift(performance_degradation),
                business_impact_score=business_impact_score,
                recommendations=recommendations,
                detected_at=datetime.utcnow()
            )
            
            self.logger.info(
                "Drift detection completed",
                model_id=model_id,
                drift_detected=drift_detected,
                drift_severity=drift_severity
            )
            
            return drift_detection
            
        except Exception as e:
            self.logger.error(
                "Drift detection failed",
                model_id=model_id,
                error=str(e)
            )
            raise
    
    async def analyze_drift_impact(
        self,
        drift_detection: ModelDriftDetection,
        baseline: ModelPerformanceBaseline,
        historical_performance: List[Dict[str, Any]]
    ) -> ModelDriftImpact:
        """
        Comprehensive drift impact analysis
        
        Args:
            drift_detection: Drift detection results
            baseline: Performance baseline
            historical_performance: Time series of performance data
            
        Returns:
            ModelDriftImpact object
        """
        
        self.logger.info(
            "Starting drift impact analysis",
            drift_session_id=drift_detection.drift_session_id
        )
        
        try:
            # Extract baseline performance
            baseline_performance = {}
            current_performance = {}
            performance_delta = {}
            
            for metric, degradation in drift_detection.performance_degradation.items():
                baseline_value = getattr(baseline, metric, None)
                if baseline_value is not None:
                    baseline_performance[metric] = baseline_value
                    
                    if baseline.model_type == 'regression' and metric in ['mae', 'mse', 'rmse']:
                        current_value = baseline_value * (1 + degradation)
                    else:
                        current_value = baseline_value * (1 - degradation)
                    
                    current_performance[metric] = current_value
                    performance_delta[metric] = current_value - baseline_value
            
            # Performance trends
            performance_trend = self._extract_performance_trends(historical_performance)
            
            # Business impact estimation
            estimated_cost_impact = self._estimate_cost_impact(
                drift_detection.model_id,
                performance_delta,
                drift_detection.current_sample_size
            )
            
            affected_predictions = drift_detection.current_sample_size
            
            # Error rate changes
            fp_increase, fn_increase = self._estimate_error_changes(
                baseline_performance,
                current_performance,
                drift_detection.model_type
            )
            
            # Technical impact
            latency_impact = self._estimate_latency_impact(performance_delta)
            throughput_impact = self._estimate_throughput_impact(performance_delta)
            resource_impact = self._estimate_resource_impact(performance_delta)
            
            # Confidence intervals
            confidence_intervals = self._calculate_confidence_intervals(
                baseline_performance,
                current_performance,
                drift_detection.current_sample_size
            )
            
            # Future projections
            projected_impact = self._project_future_impact(
                performance_trend,
                drift_detection.drift_severity,
                days_ahead=30
            )
            
            # Retraining urgency
            retraining_urgency = self._calculate_retraining_urgency(
                drift_detection.drift_severity,
                drift_detection.business_impact_score,
                performance_delta
            )
            
            impact_analysis = ModelDriftImpact(
                drift_session_id=drift_detection.drift_session_id,
                model_id=drift_detection.model_id,
                baseline_performance=baseline_performance,
                current_performance=current_performance,
                performance_delta=performance_delta,
                performance_trend=performance_trend,
                estimated_cost_impact=estimated_cost_impact,
                affected_predictions=affected_predictions,
                false_positive_increase=fp_increase,
                false_negative_increase=fn_increase,
                latency_impact=latency_impact,
                throughput_impact=throughput_impact,
                resource_utilization_change=resource_impact,
                performance_confidence_intervals=confidence_intervals,
                projected_impact_30d=projected_impact,
                retraining_urgency_score=retraining_urgency,
                analyzed_at=datetime.utcnow()
            )
            
            self.logger.info(
                "Drift impact analysis completed",
                drift_session_id=drift_detection.drift_session_id,
                estimated_cost_impact=estimated_cost_impact,
                retraining_urgency=retraining_urgency
            )
            
            return impact_analysis
            
        except Exception as e:
            self.logger.error(
                "Impact analysis failed",
                drift_session_id=drift_detection.drift_session_id,
                error=str(e)
            )
            raise
    
    def _classify_drift_severity(self, max_degradation: float) -> Tuple[bool, str]:
        """Classify drift severity based on performance degradation"""
        
        if max_degradation >= self.drift_thresholds['critical']:
            return True, 'critical'
        elif max_degradation >= self.drift_thresholds['high']:
            return True, 'high'
        elif max_degradation >= self.drift_thresholds['medium']:
            return True, 'medium'
        elif max_degradation >= self.drift_thresholds['low']:
            return True, 'low'
        else:
            return False, 'none'
    
    def _calculate_significance(self, baseline_value: float, current_value: float, sample_size: int) -> float:
        """Calculate statistical significance of performance change"""
        # Simplified significance test - in practice use proper statistical tests
        # This would require historical variance data
        estimated_variance = baseline_value * 0.1  # Rough estimate
        z_score = abs(current_value - baseline_value) / (estimated_variance / np.sqrt(sample_size))
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        return p_value
    
    def _identify_drift_causes(
        self,
        performance_degradation: Dict[str, float],
        statistical_significance: Dict[str, float],
        data_drift_info: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Identify primary causes of model drift"""
        
        causes = []
        
        # Data drift correlation
        if data_drift_info and data_drift_info.get('drift_detected', False):
            causes.append("Data distribution drift detected")
        
        # Performance pattern analysis
        max_degradation = max(performance_degradation.values()) if performance_degradation else 0
        
        if max_degradation > 0.20:
            causes.append("Severe performance degradation")
        
        # Statistical significance
        significant_changes = [metric for metric, p_val in statistical_significance.items() 
                             if p_val < self.significance_threshold]
        
        if significant_changes:
            causes.append(f"Statistically significant changes in: {', '.join(significant_changes)}")
        
        if not causes:
            causes.append("Minor performance variation within normal range")
        
        return causes
    
    def _detect_seasonal_patterns(self, current_data: Dict[str, Any]) -> bool:
        """Detect seasonal patterns in performance"""
        # Simplified seasonal detection
        # In practice, would analyze timestamps and performance over time
        return False
    
    def _suspect_concept_drift(self, performance_degradation: Dict[str, float]) -> bool:
        """Determine if concept drift is suspected"""
        # Concept drift typically shows in accuracy/F1 degradation
        accuracy_drop = performance_degradation.get('accuracy', 0)
        f1_drop = performance_degradation.get('f1_score', 0)
        
        return max(accuracy_drop, f1_drop) > 0.15
    
    def _calculate_business_impact(
        self,
        model_id: str,
        performance_degradation: Dict[str, float],
        sample_size: int
    ) -> float:
        """Calculate business impact score"""
        
        # Weighted impact based on metric importance
        weights = {
            'accuracy': 0.3,
            'precision': 0.25,
            'recall': 0.25,
            'f1_score': 0.20,
            'r2': 0.4,
            'mae': 0.3,
            'mse': 0.3
        }
        
        weighted_impact = 0
        total_weight = 0
        
        for metric, degradation in performance_degradation.items():
            if metric in weights:
                weighted_impact += weights[metric] * abs(degradation)
                total_weight += weights[metric]
        
        if total_weight > 0:
            normalized_impact = weighted_impact / total_weight
        else:
            normalized_impact = 0
        
        # Scale by sample size impact
        volume_multiplier = min(2.0, np.log10(sample_size) / 3.0)
        
        return normalized_impact * volume_multiplier
    
    def _generate_recommendations(
        self,
        drift_severity: str,
        primary_causes: List[str],
        performance_degradation: Dict[str, float],
        business_impact_score: float
    ) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        if drift_severity == 'critical':
            recommendations.append("URGENT: Immediate model retraining required")
            recommendations.append("Consider rolling back to previous model version")
            recommendations.append("Implement enhanced monitoring")
            
        elif drift_severity == 'high':
            recommendations.append("Schedule model retraining within 1-2 weeks")
            recommendations.append("Increase data collection for affected segments")
            recommendations.append("Review feature engineering pipeline")
            
        elif drift_severity == 'medium':
            recommendations.append("Plan model retraining within 1 month")
            recommendations.append("Investigate data quality issues")
            recommendations.append("Consider ensemble approaches")
            
        elif drift_severity == 'low':
            recommendations.append("Continue monitoring, retraining not urgent")
            recommendations.append("Analyze root causes for preventive measures")
        
        # Specific recommendations based on causes
        for cause in primary_causes:
            if "Data distribution drift" in cause:
                recommendations.append("Review data preprocessing pipeline")
                recommendations.append("Update feature selection strategy")
            
            if "Statistically significant" in cause:
                recommendations.append("Collect more recent training data")
                recommendations.append("Consider online learning approaches")
        
        # Business impact based recommendations
        if business_impact_score > 0.5:
            recommendations.append("High business impact - prioritize remediation")
            recommendations.append("Consider A/B testing with updated model")
        
        return list(set(recommendations))  # Remove duplicates
    
    def _extract_performance_trends(self, historical_performance: List[Dict[str, Any]]) -> Dict[str, List[float]]:
        """Extract performance trends over time"""
        
        trends = {}
        
        for record in historical_performance:
            for metric, value in record.get('metrics', {}).items():
                if metric not in trends:
                    trends[metric] = []
                trends[metric].append(value)
        
        return trends
    
    def _estimate_cost_impact(self, model_id: str, performance_delta: Dict[str, float], sample_size: int) -> float:
        """Estimate cost impact of performance degradation"""
        
        # Simplified cost estimation
        # In practice, would integrate with business metrics
        
        base_cost_per_prediction = 0.001  # $0.001 per prediction
        
        # Cost multiplier based on performance degradation
        max_degradation = max([abs(delta) for delta in performance_delta.values()], default=0)
        cost_multiplier = 1 + (max_degradation * 2)  # Double cost for each 50% degradation
        
        return base_cost_per_prediction * sample_size * cost_multiplier
    
    def _estimate_error_changes(
        self,
        baseline_performance: Dict[str, float],
        current_performance: Dict[str, float],
        model_type: str
    ) -> Tuple[float, float]:
        """Estimate changes in false positive and false negative rates"""
        
        fp_increase = 0
        fn_increase = 0
        
        if model_type == 'classification':
            precision_change = current_performance.get('precision', 1) - baseline_performance.get('precision', 1)
            recall_change = current_performance.get('recall', 1) - baseline_performance.get('recall', 1)
            
            # Simplified estimation
            fp_increase = max(0, -precision_change)  # Lower precision = more FPs
            fn_increase = max(0, -recall_change)     # Lower recall = more FNs
        
        return fp_increase, fn_increase
    
    def _estimate_latency_impact(self, performance_delta: Dict[str, float]) -> float:
        """Estimate latency impact from performance changes"""
        # Simplified estimation - degraded models might be slower
        max_degradation = max([abs(delta) for delta in performance_delta.values()], default=0)
        return max_degradation * 0.1  # 10% latency increase per unit degradation
    
    def _estimate_throughput_impact(self, performance_delta: Dict[str, float]) -> float:
        """Estimate throughput impact"""
        latency_impact = self._estimate_latency_impact(performance_delta)
        return -latency_impact  # Inverse relationship
    
    def _estimate_resource_impact(self, performance_delta: Dict[str, float]) -> float:
        """Estimate resource utilization impact"""
        max_degradation = max([abs(delta) for delta in performance_delta.values()], default=0)
        return max_degradation * 0.05  # 5% resource increase per unit degradation
    
    def _calculate_confidence_intervals(
        self,
        baseline_performance: Dict[str, float],
        current_performance: Dict[str, float],
        sample_size: int
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for performance metrics"""
        
        confidence_intervals = {}
        confidence_level = 0.95
        z_score = 1.96  # 95% confidence
        
        for metric in baseline_performance.keys():
            baseline_val = baseline_performance[metric]
            current_val = current_performance.get(metric, baseline_val)
            
            # Simplified CI calculation
            estimated_std = abs(current_val - baseline_val) / 2
            margin_of_error = z_score * (estimated_std / np.sqrt(sample_size))
            
            ci_lower = current_val - margin_of_error
            ci_upper = current_val + margin_of_error
            
            confidence_intervals[metric] = (ci_lower, ci_upper)
        
        return confidence_intervals
    
    def _project_future_impact(
        self,
        performance_trend: Dict[str, List[float]],
        drift_severity: str,
        days_ahead: int = 30
    ) -> Dict[str, float]:
        """Project future performance impact"""
        
        projections = {}
        
        # Degradation rates by severity
        degradation_rates = {
            'critical': 0.02,  # 2% per day
            'high': 0.01,      # 1% per day
            'medium': 0.005,   # 0.5% per day
            'low': 0.001       # 0.1% per day
        }
        
        daily_degradation = degradation_rates.get(drift_severity, 0)
        
        for metric, values in performance_trend.items():
            if values:
                current_value = values[-1]
                projected_value = current_value * (1 - daily_degradation * days_ahead)
                projections[metric] = projected_value
        
        return projections
    
    def _calculate_retraining_urgency(
        self,
        drift_severity: str,
        business_impact_score: float,
        performance_delta: Dict[str, float]
    ) -> float:
        """Calculate urgency score for model retraining"""
        
        severity_scores = {
            'critical': 1.0,
            'high': 0.8,
            'medium': 0.6,
            'low': 0.3,
            'none': 0.0
        }
        
        severity_score = severity_scores.get(drift_severity, 0)
        
        # Combine factors
        urgency_score = (
            severity_score * 0.4 +
            business_impact_score * 0.4 +
            min(1.0, max([abs(delta) for delta in performance_delta.values()], default=0)) * 0.2
        )
        
        return min(1.0, urgency_score)