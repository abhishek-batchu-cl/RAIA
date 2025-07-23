"""
A/B Testing Service
Comprehensive A/B testing framework for ML models with statistical analysis
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import get_database

logger = logging.getLogger(__name__)
settings = get_settings()

class ExperimentStatus(Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    STOPPED = "stopped"

class TrafficAllocation(Enum):
    EQUAL = "equal"
    WEIGHTED = "weighted"
    RAMPED = "ramped"
    CHAMPION_CHALLENGER = "champion_challenger"

class ABTestingService:
    """
    Comprehensive A/B testing service for ML model experimentation
    """
    
    def __init__(self):
        self.experiments = {}
        self.experiment_data = defaultdict(lambda: defaultdict(list))
        self.variant_assignments = {}
        self.statistical_tests = {
            'ttest': stats.ttest_ind,
            'mannwhitney': stats.mannwhitneyu,
            'chi2': stats.chi2_contingency,
            'proportions_ztest': self._proportions_z_test
        }
        self.confidence_level = 0.95
        self.minimum_sample_size = 100
        self.minimum_effect_size = 0.05
        
    async def create_experiment(
        self,
        experiment_name: str,
        description: str,
        variants: Dict[str, Dict[str, Any]],
        success_metrics: List[str],
        traffic_allocation: Dict[str, float],
        allocation_method: TrafficAllocation = TrafficAllocation.EQUAL,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        minimum_sample_size: Optional[int] = None,
        significance_threshold: float = 0.05,
        experiment_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a new A/B testing experiment
        
        Args:
            experiment_name: Name of the experiment
            description: Experiment description
            variants: Dictionary of variant configurations
            success_metrics: List of metrics to track
            traffic_allocation: Traffic allocation percentages
            allocation_method: Method for traffic allocation
            start_date: When experiment should start
            end_date: When experiment should end
            minimum_sample_size: Minimum samples per variant
            significance_threshold: Statistical significance threshold
            experiment_config: Additional configuration
        
        Returns:
            Experiment creation result
        """
        try:
            experiment_id = str(uuid.uuid4())
            
            # Validate traffic allocation
            total_allocation = sum(traffic_allocation.values())
            if abs(total_allocation - 1.0) > 0.01:
                raise ValueError(f"Traffic allocation must sum to 1.0, got {total_allocation}")
            
            # Validate variants exist in allocation
            variant_names = set(variants.keys())
            allocation_variants = set(traffic_allocation.keys())
            if variant_names != allocation_variants:
                raise ValueError(f"Variants mismatch: {variant_names} vs {allocation_variants}")
            
            experiment = {
                'experiment_id': experiment_id,
                'experiment_name': experiment_name,
                'description': description,
                'status': ExperimentStatus.DRAFT,
                'variants': variants,
                'success_metrics': success_metrics,
                'traffic_allocation': traffic_allocation,
                'allocation_method': allocation_method,
                'start_date': start_date or datetime.utcnow(),
                'end_date': end_date,
                'minimum_sample_size': minimum_sample_size or self.minimum_sample_size,
                'significance_threshold': significance_threshold,
                'created_at': datetime.utcnow(),
                'created_by': 'system',  # Would be user_id in production
                'sample_counts': {variant: 0 for variant in variants.keys()},
                'conversion_counts': {variant: 0 for variant in variants.keys()},
                'statistical_power': 0.8,
                'effect_size': self.minimum_effect_size,
                'config': experiment_config or {}
            }
            
            # Calculate required sample size
            required_sample_size = await self._calculate_required_sample_size(
                experiment['statistical_power'],
                experiment['effect_size'],
                experiment['significance_threshold']
            )
            experiment['required_sample_size_per_variant'] = required_sample_size
            
            self.experiments[experiment_id] = experiment
            
            return {
                'status': 'success',
                'experiment_id': experiment_id,
                'experiment': experiment,
                'message': 'Experiment created successfully'
            }
            
        except Exception as e:
            logger.error(f"Failed to create experiment: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    async def start_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """
        Start an A/B test experiment
        
        Args:
            experiment_id: Experiment identifier
        
        Returns:
            Start status
        """
        try:
            if experiment_id not in self.experiments:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            experiment = self.experiments[experiment_id]
            
            if experiment['status'] != ExperimentStatus.DRAFT:
                raise ValueError(f"Cannot start experiment with status {experiment['status']}")
            
            # Validate experiment configuration
            validation_result = await self._validate_experiment_config(experiment)
            if not validation_result['valid']:
                raise ValueError(f"Experiment validation failed: {validation_result['message']}")
            
            experiment['status'] = ExperimentStatus.ACTIVE
            experiment['actual_start_date'] = datetime.utcnow()
            
            # Initialize tracking data structures
            for variant in experiment['variants']:
                self.experiment_data[experiment_id][variant] = []
            
            return {
                'status': 'success',
                'experiment_id': experiment_id,
                'message': 'Experiment started successfully',
                'start_time': experiment['actual_start_date'].isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to start experiment {experiment_id}: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'experiment_id': experiment_id
            }
    
    async def assign_variant(
        self,
        experiment_id: str,
        user_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Assign a user to a variant in an experiment
        
        Args:
            experiment_id: Experiment identifier
            user_id: User identifier
            context: Additional context for assignment
        
        Returns:
            Variant assignment result
        """
        try:
            if experiment_id not in self.experiments:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            experiment = self.experiments[experiment_id]
            
            if experiment['status'] != ExperimentStatus.ACTIVE:
                raise ValueError(f"Experiment {experiment_id} is not active")
            
            # Check if user is already assigned
            assignment_key = f"{experiment_id}:{user_id}"
            if assignment_key in self.variant_assignments:
                existing_assignment = self.variant_assignments[assignment_key]
                return {
                    'status': 'success',
                    'experiment_id': experiment_id,
                    'user_id': user_id,
                    'variant': existing_assignment['variant'],
                    'is_new_assignment': False,
                    'assignment_time': existing_assignment['assignment_time']
                }
            
            # Assign variant based on allocation method
            variant = await self._assign_variant_by_method(
                experiment, user_id, context
            )
            
            # Store assignment
            assignment = {
                'experiment_id': experiment_id,
                'user_id': user_id,
                'variant': variant,
                'assignment_time': datetime.utcnow(),
                'context': context or {}
            }
            
            self.variant_assignments[assignment_key] = assignment
            experiment['sample_counts'][variant] += 1
            
            return {
                'status': 'success',
                'experiment_id': experiment_id,
                'user_id': user_id,
                'variant': variant,
                'is_new_assignment': True,
                'assignment_time': assignment['assignment_time'].isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to assign variant: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'experiment_id': experiment_id,
                'user_id': user_id
            }
    
    async def record_event(
        self,
        experiment_id: str,
        user_id: str,
        event_type: str,
        event_data: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Record an event for experiment analysis
        
        Args:
            experiment_id: Experiment identifier
            user_id: User identifier
            event_type: Type of event (conversion, click, purchase, etc.)
            event_data: Event-specific data
            timestamp: When event occurred
        
        Returns:
            Event recording status
        """
        try:
            if experiment_id not in self.experiments:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            # Get user's variant assignment
            assignment_key = f"{experiment_id}:{user_id}"
            if assignment_key not in self.variant_assignments:
                raise ValueError(f"User {user_id} not assigned to experiment {experiment_id}")
            
            assignment = self.variant_assignments[assignment_key]
            variant = assignment['variant']
            
            # Record event
            event = {
                'event_id': str(uuid.uuid4()),
                'experiment_id': experiment_id,
                'user_id': user_id,
                'variant': variant,
                'event_type': event_type,
                'event_data': event_data,
                'timestamp': timestamp or datetime.utcnow()
            }
            
            self.experiment_data[experiment_id][variant].append(event)
            
            # Update conversion count if it's a success metric
            experiment = self.experiments[experiment_id]
            if event_type in experiment['success_metrics']:
                experiment['conversion_counts'][variant] += 1
            
            return {
                'status': 'success',
                'event_id': event['event_id'],
                'experiment_id': experiment_id,
                'variant': variant,
                'event_type': event_type
            }
            
        except Exception as e:
            logger.error(f"Failed to record event: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'experiment_id': experiment_id,
                'user_id': user_id
            }
    
    async def analyze_experiment(
        self,
        experiment_id: str,
        analysis_type: str = 'comprehensive'
    ) -> Dict[str, Any]:
        """
        Perform statistical analysis of an experiment
        
        Args:
            experiment_id: Experiment identifier
            analysis_type: Type of analysis (comprehensive, quick, bayesian)
        
        Returns:
            Complete experiment analysis results
        """
        try:
            if experiment_id not in self.experiments:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            experiment = self.experiments[experiment_id]
            
            analysis_results = {
                'experiment_id': experiment_id,
                'experiment_name': experiment['experiment_name'],
                'analysis_timestamp': datetime.utcnow().isoformat(),
                'experiment_status': experiment['status'].value,
                'analysis_type': analysis_type,
                'statistical_significance': {},
                'practical_significance': {},
                'variant_performance': {},
                'recommendations': [],
                'confidence_intervals': {},
                'effect_sizes': {},
                'statistical_power': {},
                'sample_ratios': {}
            }
            
            # Get experiment data for all variants
            variant_data = {}
            for variant in experiment['variants']:
                variant_events = self.experiment_data[experiment_id][variant]
                variant_data[variant] = {
                    'events': variant_events,
                    'sample_size': experiment['sample_counts'][variant],
                    'conversions': experiment['conversion_counts'][variant]
                }
            
            # 1. Descriptive Statistics
            for variant, data in variant_data.items():
                sample_size = data['sample_size']
                conversions = data['conversions']
                conversion_rate = conversions / sample_size if sample_size > 0 else 0
                
                analysis_results['variant_performance'][variant] = {
                    'sample_size': sample_size,
                    'conversions': conversions,
                    'conversion_rate': conversion_rate,
                    'confidence_interval': self._calculate_confidence_interval(
                        conversion_rate, sample_size
                    )
                }
            
            # 2. Statistical Significance Testing
            variants = list(experiment['variants'].keys())
            if len(variants) >= 2:
                # Pairwise comparisons
                for i, variant_a in enumerate(variants):
                    for variant_b in variants[i+1:]:
                        comparison_result = await self._compare_variants(
                            variant_data[variant_a],
                            variant_data[variant_b],
                            variant_a,
                            variant_b,
                            experiment['significance_threshold']
                        )
                        comparison_key = f"{variant_a}_vs_{variant_b}"
                        analysis_results['statistical_significance'][comparison_key] = comparison_result
            
            # 3. Effect Size Calculations
            if len(variants) >= 2:
                control_variant = variants[0]  # Assume first variant is control
                for variant in variants[1:]:
                    effect_size = await self._calculate_effect_size(
                        variant_data[control_variant],
                        variant_data[variant]
                    )
                    analysis_results['effect_sizes'][f"{control_variant}_to_{variant}"] = effect_size
            
            # 4. Statistical Power Analysis
            power_analysis = await self._calculate_statistical_power(
                experiment, variant_data
            )
            analysis_results['statistical_power'] = power_analysis
            
            # 5. Sample Ratio Mismatch Detection
            srm_results = await self._detect_sample_ratio_mismatch(
                experiment, variant_data
            )
            analysis_results['sample_ratios'] = srm_results
            
            # 6. Generate Recommendations
            recommendations = await self._generate_experiment_recommendations(
                experiment, analysis_results
            )
            analysis_results['recommendations'] = recommendations
            
            # 7. Determine experiment status
            experiment_conclusion = await self._determine_experiment_conclusion(
                analysis_results
            )
            analysis_results['conclusion'] = experiment_conclusion
            
            return {
                'status': 'success',
                **analysis_results
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze experiment {experiment_id}: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'experiment_id': experiment_id
            }
    
    async def _assign_variant_by_method(
        self, 
        experiment: Dict[str, Any], 
        user_id: str, 
        context: Optional[Dict[str, Any]]
    ) -> str:
        """
        Assign variant based on allocation method
        """
        allocation_method = experiment['allocation_method']
        traffic_allocation = experiment['traffic_allocation']
        
        if allocation_method == TrafficAllocation.EQUAL:
            # Hash-based assignment for consistent user experience
            hash_input = f"{experiment['experiment_id']}:{user_id}"
            hash_value = hash(hash_input) % 100
            cumulative = 0
            
            for variant, percentage in traffic_allocation.items():
                cumulative += percentage * 100
                if hash_value < cumulative:
                    return variant
            
            return list(traffic_allocation.keys())[-1]  # Fallback
        
        elif allocation_method == TrafficAllocation.WEIGHTED:
            # Weighted random assignment
            np.random.seed(hash(f"{experiment['experiment_id']}:{user_id}") % (2**32))
            return np.random.choice(
                list(traffic_allocation.keys()),
                p=list(traffic_allocation.values())
            )
        
        elif allocation_method == TrafficAllocation.CHAMPION_CHALLENGER:
            # 90% champion, 10% challenger (or custom allocation)
            variants = list(traffic_allocation.keys())
            champion = variants[0]
            challenger = variants[1] if len(variants) > 1 else variants[0]
            
            champion_percentage = traffic_allocation.get(champion, 0.9)
            hash_value = hash(f"{experiment['experiment_id']}:{user_id}") % 100
            
            return champion if hash_value < (champion_percentage * 100) else challenger
        
        else:
            # Default to equal allocation
            return await self._assign_variant_by_method(
                {**experiment, 'allocation_method': TrafficAllocation.EQUAL},
                user_id,
                context
            )
    
    async def _compare_variants(
        self,
        variant_a_data: Dict[str, Any],
        variant_b_data: Dict[str, Any],
        variant_a_name: str,
        variant_b_name: str,
        significance_threshold: float
    ) -> Dict[str, Any]:
        """
        Compare two variants statistically
        """
        try:
            # Extract data
            n_a = variant_a_data['sample_size']
            x_a = variant_a_data['conversions']
            n_b = variant_b_data['sample_size']
            x_b = variant_b_data['conversions']
            
            if n_a == 0 or n_b == 0:
                return {
                    'test_type': 'insufficient_data',
                    'p_value': None,
                    'significant': False,
                    'message': 'Insufficient data for comparison'
                }
            
            p_a = x_a / n_a
            p_b = x_b / n_b
            
            # Two-proportion z-test
            z_stat, p_value = self._proportions_z_test(
                x_a, n_a, x_b, n_b
            )
            
            is_significant = p_value < significance_threshold
            
            # Calculate confidence interval for difference
            diff = p_b - p_a
            se_diff = np.sqrt((p_a * (1 - p_a) / n_a) + (p_b * (1 - p_b) / n_b))
            z_critical = stats.norm.ppf(1 - significance_threshold / 2)
            ci_lower = diff - z_critical * se_diff
            ci_upper = diff + z_critical * se_diff
            
            return {
                'test_type': 'two_proportion_z_test',
                'variant_a': variant_a_name,
                'variant_b': variant_b_name,
                'variant_a_rate': p_a,
                'variant_b_rate': p_b,
                'difference': diff,
                'relative_difference': (diff / p_a) if p_a > 0 else 0,
                'z_statistic': z_stat,
                'p_value': p_value,
                'significant': is_significant,
                'confidence_interval': {
                    'lower': ci_lower,
                    'upper': ci_upper,
                    'level': 1 - significance_threshold
                },
                'sample_sizes': {'variant_a': n_a, 'variant_b': n_b}
            }
            
        except Exception as e:
            logger.error(f"Error comparing variants: {e}")
            return {
                'test_type': 'error',
                'error': str(e),
                'significant': False
            }
    
    def _proportions_z_test(self, x1: int, n1: int, x2: int, n2: int) -> Tuple[float, float]:
        """
        Two-proportion z-test
        """
        if n1 == 0 or n2 == 0:
            return 0.0, 1.0
        
        p1 = x1 / n1
        p2 = x2 / n2
        p_pool = (x1 + x2) / (n1 + n2)
        
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
        
        if se == 0:
            return 0.0, 1.0
        
        z = (p1 - p2) / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        return z, p_value
    
    def _calculate_confidence_interval(self, proportion: float, sample_size: int) -> Dict[str, float]:
        """
        Calculate confidence interval for a proportion
        """
        if sample_size == 0:
            return {'lower': 0, 'upper': 0, 'level': self.confidence_level}
        
        z_critical = stats.norm.ppf(1 - (1 - self.confidence_level) / 2)
        se = np.sqrt((proportion * (1 - proportion)) / sample_size)
        
        lower = max(0, proportion - z_critical * se)
        upper = min(1, proportion + z_critical * se)
        
        return {
            'lower': lower,
            'upper': upper,
            'level': self.confidence_level
        }
    
    async def _calculate_effect_size(
        self,
        control_data: Dict[str, Any],
        treatment_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate effect size (Cohen's h for proportions)
        """
        try:
            n_control = control_data['sample_size']
            x_control = control_data['conversions']
            n_treatment = treatment_data['sample_size']
            x_treatment = treatment_data['conversions']
            
            if n_control == 0 or n_treatment == 0:
                return {'cohens_h': 0, 'interpretation': 'insufficient_data'}
            
            p_control = x_control / n_control
            p_treatment = x_treatment / n_treatment
            
            # Cohen's h for proportions
            h = 2 * (np.arcsin(np.sqrt(p_treatment)) - np.arcsin(np.sqrt(p_control)))
            
            # Interpretation
            if abs(h) < 0.2:
                interpretation = 'small'
            elif abs(h) < 0.5:
                interpretation = 'medium'
            else:
                interpretation = 'large'
            
            return {
                'cohens_h': h,
                'absolute_difference': p_treatment - p_control,
                'relative_lift': ((p_treatment - p_control) / p_control) if p_control > 0 else 0,
                'interpretation': interpretation
            }
            
        except Exception as e:
            logger.error(f"Error calculating effect size: {e}")
            return {'cohens_h': 0, 'interpretation': 'error', 'error': str(e)}
    
    async def _calculate_required_sample_size(
        self,
        power: float,
        effect_size: float,
        alpha: float
    ) -> int:
        """
        Calculate required sample size for desired statistical power
        """
        try:
            # Using Cohen's formula for two-proportion test
            z_alpha = stats.norm.ppf(1 - alpha / 2)
            z_beta = stats.norm.ppf(power)
            
            # Approximate sample size calculation
            n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
            
            return max(int(n), self.minimum_sample_size)
            
        except Exception as e:
            logger.error(f"Error calculating sample size: {e}")
            return self.minimum_sample_size
    
    async def _calculate_statistical_power(
        self,
        experiment: Dict[str, Any],
        variant_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate current statistical power of the experiment
        """
        try:
            variants = list(experiment['variants'].keys())
            if len(variants) < 2:
                return {'power': 0, 'message': 'Need at least 2 variants'}
            
            # Use first two variants for power calculation
            variant_a = variants[0]
            variant_b = variants[1]
            
            n_a = variant_data[variant_a]['sample_size']
            x_a = variant_data[variant_a]['conversions']
            n_b = variant_data[variant_b]['sample_size']
            x_b = variant_data[variant_b]['conversions']
            
            if n_a == 0 or n_b == 0:
                return {'power': 0, 'message': 'Insufficient data'}
            
            p_a = x_a / n_a
            p_b = x_b / n_b
            
            # Calculate observed effect size
            observed_effect = 2 * (np.arcsin(np.sqrt(p_b)) - np.arcsin(np.sqrt(p_a)))
            
            # Power calculation using observed effect and current sample sizes
            z_alpha = stats.norm.ppf(1 - experiment['significance_threshold'] / 2)
            se = np.sqrt(2 * p_a * (1 - p_a) / min(n_a, n_b))  # Approximate
            
            if se > 0:
                z_beta = (abs(observed_effect) - z_alpha * se) / se
                power = stats.norm.cdf(z_beta)
            else:
                power = 0
            
            return {
                'current_power': max(0, min(1, power)),
                'target_power': experiment['statistical_power'],
                'observed_effect_size': observed_effect,
                'adequate_power': power >= experiment['statistical_power']
            }
            
        except Exception as e:
            logger.error(f"Error calculating statistical power: {e}")
            return {'power': 0, 'error': str(e)}
    
    async def _detect_sample_ratio_mismatch(
        self,
        experiment: Dict[str, Any],
        variant_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Detect sample ratio mismatch (SRM)
        """
        try:
            expected_allocation = experiment['traffic_allocation']
            observed_counts = {variant: data['sample_size'] for variant, data in variant_data.items()}
            total_observed = sum(observed_counts.values())
            
            if total_observed == 0:
                return {'srm_detected': False, 'message': 'No data'}
            
            # Calculate expected counts
            expected_counts = {
                variant: total_observed * allocation
                for variant, allocation in expected_allocation.items()
            }
            
            # Chi-square goodness of fit test
            observed_values = list(observed_counts.values())
            expected_values = list(expected_counts.values())
            
            chi2_stat, p_value = stats.chisquare(observed_values, expected_values)
            
            srm_detected = p_value < 0.01  # More stringent threshold for SRM
            
            return {
                'srm_detected': srm_detected,
                'chi2_statistic': chi2_stat,
                'p_value': p_value,
                'observed_counts': observed_counts,
                'expected_counts': expected_counts,
                'total_samples': total_observed,
                'severity': 'high' if p_value < 0.001 else 'medium' if srm_detected else 'low'
            }
            
        except Exception as e:
            logger.error(f"Error detecting SRM: {e}")
            return {'srm_detected': False, 'error': str(e)}
    
    async def _generate_experiment_recommendations(
        self,
        experiment: Dict[str, Any],
        analysis_results: Dict[str, Any]
    ) -> List[str]:
        """
        Generate actionable recommendations based on experiment analysis
        """
        recommendations = []
        
        try:
            # Check sample sizes
            min_sample_size = experiment['minimum_sample_size']
            for variant, performance in analysis_results['variant_performance'].items():
                if performance['sample_size'] < min_sample_size:
                    recommendations.append(
                        f"⚠️ Variant '{variant}' has insufficient sample size ({performance['sample_size']} < {min_sample_size}). Continue collecting data."
                    )
            
            # Check for SRM
            if analysis_results.get('sample_ratios', {}).get('srm_detected', False):
                recommendations.append(
                    "🚨 Sample Ratio Mismatch detected. Check traffic allocation implementation and data collection."
                )
            
            # Check statistical power
            power_info = analysis_results.get('statistical_power', {})
            if not power_info.get('adequate_power', True):
                recommendations.append(
                    f"📊 Statistical power is low ({power_info.get('current_power', 0):.2f}). Consider increasing sample size or effect size."
                )
            
            # Check for significant results
            significant_results = []
            for comparison, result in analysis_results.get('statistical_significance', {}).items():
                if result.get('significant', False):
                    significant_results.append(comparison)
            
            if significant_results:
                recommendations.append(
                    f"✅ Statistically significant results found in: {', '.join(significant_results)}"
                )
            else:
                recommendations.append(
                    "📈 No statistically significant differences detected yet. Continue monitoring or consider larger effect sizes."
                )
            
            # Effect size recommendations
            for comparison, effect in analysis_results.get('effect_sizes', {}).items():
                if effect['interpretation'] == 'large':
                    recommendations.append(
                        f"🎯 Large practical effect detected in {comparison}. Consider business impact."
                    )
                elif effect['interpretation'] == 'small':
                    recommendations.append(
                        f"💡 Small effect in {comparison}. Evaluate if improvement justifies implementation costs."
                    )
            
            # General recommendations
            recommendations.extend([
                "📋 Monitor key business metrics alongside statistical metrics",
                "🔍 Consider segmented analysis for different user groups",
                "⏰ Set up automated monitoring and alerts for experiment health"
            ])
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            recommendations.append("❌ Error generating recommendations. Manual review required.")
        
        return recommendations
    
    async def _determine_experiment_conclusion(
        self,
        analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Determine overall experiment conclusion
        """
        try:
            # Check for significant results
            significant_comparisons = [
                comp for comp, result in analysis_results.get('statistical_significance', {}).items()
                if result.get('significant', False)
            ]
            
            # Check sample adequacy
            adequate_samples = all(
                perf['sample_size'] >= 100  # Minimum threshold
                for perf in analysis_results.get('variant_performance', {}).values()
            )
            
            # Check for SRM
            srm_detected = analysis_results.get('sample_ratios', {}).get('srm_detected', False)
            
            if srm_detected:
                conclusion = 'invalid'
                reason = 'Sample ratio mismatch detected'
            elif not adequate_samples:
                conclusion = 'inconclusive'
                reason = 'Insufficient sample size'
            elif significant_comparisons:
                conclusion = 'significant'
                reason = f'Significant differences found: {significant_comparisons}'
            else:
                conclusion = 'no_difference'
                reason = 'No statistically significant differences detected'
            
            return {
                'conclusion': conclusion,
                'reason': reason,
                'confidence': 'high' if adequate_samples and not srm_detected else 'low',
                'recommended_action': self._get_recommended_action(conclusion)
            }
            
        except Exception as e:
            logger.error(f"Error determining conclusion: {e}")
            return {
                'conclusion': 'error',
                'reason': str(e),
                'confidence': 'low',
                'recommended_action': 'manual_review'
            }
    
    def _get_recommended_action(self, conclusion: str) -> str:
        """
        Get recommended action based on conclusion
        """
        action_map = {
            'significant': 'implement_winner',
            'no_difference': 'choose_based_on_other_factors',
            'inconclusive': 'continue_experiment',
            'invalid': 'fix_implementation_and_restart',
            'error': 'manual_review'
        }
        return action_map.get(conclusion, 'manual_review')
    
    async def _validate_experiment_config(self, experiment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate experiment configuration before starting
        """
        try:
            issues = []
            
            # Check traffic allocation
            total_allocation = sum(experiment['traffic_allocation'].values())
            if abs(total_allocation - 1.0) > 0.01:
                issues.append(f"Traffic allocation sums to {total_allocation}, should be 1.0")
            
            # Check variants
            if len(experiment['variants']) < 2:
                issues.append("Need at least 2 variants for A/B test")
            
            # Check success metrics
            if not experiment['success_metrics']:
                issues.append("Need at least one success metric")
            
            # Check dates
            if experiment['end_date'] and experiment['start_date'] >= experiment['end_date']:
                issues.append("End date must be after start date")
            
            return {
                'valid': len(issues) == 0,
                'issues': issues,
                'message': '; '.join(issues) if issues else 'Configuration is valid'
            }
            
        except Exception as e:
            return {
                'valid': False,
                'issues': [str(e)],
                'message': f"Validation error: {e}"
            }
    
    async def stop_experiment(
        self,
        experiment_id: str,
        reason: str = "Manual stop"
    ) -> Dict[str, Any]:
        """
        Stop an active experiment
        
        Args:
            experiment_id: Experiment identifier
            reason: Reason for stopping
        
        Returns:
            Stop status
        """
        try:
            if experiment_id not in self.experiments:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            experiment = self.experiments[experiment_id]
            
            if experiment['status'] not in [ExperimentStatus.ACTIVE, ExperimentStatus.PAUSED]:
                raise ValueError(f"Cannot stop experiment with status {experiment['status']}")
            
            experiment['status'] = ExperimentStatus.STOPPED
            experiment['stop_date'] = datetime.utcnow()
            experiment['stop_reason'] = reason
            
            return {
                'status': 'success',
                'experiment_id': experiment_id,
                'message': 'Experiment stopped successfully',
                'stop_time': experiment['stop_date'].isoformat(),
                'reason': reason
            }
            
        except Exception as e:
            logger.error(f"Failed to stop experiment {experiment_id}: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'experiment_id': experiment_id
            }
    
    async def list_experiments(
        self,
        status_filter: Optional[ExperimentStatus] = None
    ) -> Dict[str, Any]:
        """
        List all experiments with optional status filter
        
        Args:
            status_filter: Optional status to filter by
        
        Returns:
            List of experiments
        """
        try:
            experiments = []
            
            for exp_id, experiment in self.experiments.items():
                if status_filter is None or experiment['status'] == status_filter:
                    summary = {
                        'experiment_id': exp_id,
                        'experiment_name': experiment['experiment_name'],
                        'status': experiment['status'].value,
                        'created_at': experiment['created_at'].isoformat(),
                        'start_date': experiment['start_date'].isoformat() if experiment['start_date'] else None,
                        'variants': list(experiment['variants'].keys()),
                        'total_samples': sum(experiment['sample_counts'].values()),
                        'total_conversions': sum(experiment['conversion_counts'].values())
                    }
                    experiments.append(summary)
            
            return {
                'status': 'success',
                'experiments': experiments,
                'total_count': len(experiments),
                'filter_applied': status_filter.value if status_filter else None
            }
            
        except Exception as e:
            logger.error(f"Failed to list experiments: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }

# Global service instance
ab_testing_service = ABTestingService()
