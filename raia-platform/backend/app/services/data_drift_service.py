"""
Data Drift Detection Service
Comprehensive implementation using Evidently AI for drift monitoring
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
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

# Evidently imports
try:
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, RegressionPreset, ClassificationPreset
    from evidently.metrics import (
        DataDriftTable, DatasetDriftMetric, ColumnDriftMetric, 
        ColumnSummaryMetric, DatasetMissingValuesMetric,
        RegressionQualityMetric, ClassificationQualityMetric
    )
    from evidently.test_suite import TestSuite
    from evidently.tests import (
        TestNumberOfColumnsWithMissingValues,
        TestNumberOfRowsWithMissingValues,
        TestNumberOfConstantColumns,
        TestNumberOfDuplicatedRows,
        TestNumberOfDuplicatedColumns,
        TestColumnsType,
        TestNumberOfDriftedColumns,
        TestShareOfMissingValues,
        TestMeanInNSigmas,
        TestValueMin,
        TestValueMax,
        TestValueQuantile
    )
    EVIDENTLY_AVAILABLE = True
except ImportError:
    logger.warning("Evidently not available. Some drift detection features will be limited.")
    EVIDENTLY_AVAILABLE = False

from app.core.config import get_settings
from app.core.database import get_database

logger = logging.getLogger(__name__)
settings = get_settings()

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)

class DataDriftService:
    """
    Advanced data drift detection service using multiple statistical methods
    """
    
    def __init__(self):
        self.drift_reports = {}
        self.baseline_data = {}
        self.drift_thresholds = {
            'ks_test': 0.05,
            'jensen_shannon': 0.1,
            'wasserstein': 0.1,
            'population_stability_index': 0.1,
            'prediction_drift': 0.05
        }
        self.feature_thresholds = {}  # Per-feature custom thresholds
        
    async def register_baseline(
        self,
        model_id: str,
        baseline_data: pd.DataFrame,
        target_column: Optional[str] = None,
        prediction_column: Optional[str] = None,
        feature_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Register baseline/reference data for drift detection
        
        Args:
            model_id: Unique identifier for the model
            baseline_data: Reference dataset
            target_column: Name of target column
            prediction_column: Name of prediction column
            feature_columns: List of feature column names
        
        Returns:
            Registration status and baseline statistics
        """
        try:
            # Clean and validate data
            baseline_data = baseline_data.copy()
            
            # Auto-detect columns if not provided
            if feature_columns is None:
                feature_columns = [col for col in baseline_data.columns 
                                 if col not in [target_column, prediction_column]]
            
            # Calculate baseline statistics
            baseline_stats = {
                'data_shape': baseline_data.shape,
                'missing_values': baseline_data.isnull().sum().to_dict(),
                'data_types': baseline_data.dtypes.astype(str).to_dict(),
                'numeric_stats': {},
                'categorical_stats': {}
            }
            
            # Calculate statistics for numeric columns
            numeric_columns = baseline_data.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if col in feature_columns:
                    baseline_stats['numeric_stats'][col] = {
                        'mean': float(baseline_data[col].mean()),
                        'std': float(baseline_data[col].std()),
                        'min': float(baseline_data[col].min()),
                        'max': float(baseline_data[col].max()),
                        'quantiles': {
                            '25%': float(baseline_data[col].quantile(0.25)),
                            '50%': float(baseline_data[col].quantile(0.50)),
                            '75%': float(baseline_data[col].quantile(0.75))
                        }
                    }
            
            # Calculate statistics for categorical columns
            categorical_columns = baseline_data.select_dtypes(include=['object', 'category']).columns
            for col in categorical_columns:
                if col in feature_columns:
                    value_counts = baseline_data[col].value_counts()
                    baseline_stats['categorical_stats'][col] = {
                        'unique_values': int(baseline_data[col].nunique()),
                        'value_distribution': value_counts.to_dict(),
                        'top_values': value_counts.head(10).to_dict()
                    }
            
            # Store baseline data and metadata
            self.baseline_data[model_id] = {
                'data': baseline_data,
                'target_column': target_column,
                'prediction_column': prediction_column,
                'feature_columns': feature_columns,
                'baseline_stats': baseline_stats,
                'registration_time': datetime.utcnow()
            }
            
            return {
                'status': 'success',
                'model_id': model_id,
                'baseline_registered': True,
                'baseline_stats': baseline_stats,
                'feature_columns': feature_columns,
                'registration_time': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to register baseline for model {model_id}: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'model_id': model_id
            }
    
    async def detect_drift(
        self,
        model_id: str,
        current_data: pd.DataFrame,
        drift_methods: List[str] = ['ks_test', 'jensen_shannon', 'wasserstein'],
        include_prediction_drift: bool = True
    ) -> Dict[str, Any]:
        """
        Detect data drift between baseline and current data
        
        Args:
            model_id: Model identifier
            current_data: Current dataset to compare against baseline
            drift_methods: List of drift detection methods to use
            include_prediction_drift: Whether to detect prediction drift
        
        Returns:
            Comprehensive drift analysis results
        """
        try:
            if model_id not in self.baseline_data:
                raise ValueError(f"No baseline data registered for model {model_id}")
            
            baseline_info = self.baseline_data[model_id]
            baseline_data = baseline_info['data']
            feature_columns = baseline_info['feature_columns']
            
            # Ensure current data has the same columns
            missing_columns = set(feature_columns) - set(current_data.columns)
            if missing_columns:
                logger.warning(f"Missing columns in current data: {missing_columns}")
                feature_columns = [col for col in feature_columns if col in current_data.columns]
            
            drift_results = {
                'model_id': model_id,
                'detection_time': datetime.utcnow().isoformat(),
                'baseline_size': len(baseline_data),
                'current_size': len(current_data),
                'features_analyzed': len(feature_columns),
                'drift_detected': False,
                'overall_drift_score': 0.0,
                'feature_drift': {},
                'data_quality_issues': {},
                'recommendations': []
            }
            
            # 1. Statistical drift detection
            for feature in feature_columns:
                feature_drift = await self._detect_feature_drift(
                    baseline_data[feature],
                    current_data[feature],
                    feature_name=feature,
                    methods=drift_methods
                )
                drift_results['feature_drift'][feature] = feature_drift
                
                # Check if drift detected for this feature
                if any(result['drift_detected'] for result in feature_drift['test_results'].values()):
                    drift_results['drift_detected'] = True
            
            # 2. Prediction drift detection (if applicable)
            if include_prediction_drift and baseline_info['prediction_column']:
                pred_col = baseline_info['prediction_column']
                if pred_col in current_data.columns:
                    pred_drift = await self._detect_prediction_drift(
                        baseline_data[pred_col],
                        current_data[pred_col]
                    )
                    drift_results['prediction_drift'] = pred_drift
                    if pred_drift['drift_detected']:
                        drift_results['drift_detected'] = True
            
            # 3. Data quality analysis
            quality_issues = await self._analyze_data_quality(
                baseline_data[feature_columns],
                current_data[feature_columns]
            )
            drift_results['data_quality_issues'] = quality_issues
            
            # 4. Calculate overall drift score
            feature_scores = []
            for feature_data in drift_results['feature_drift'].values():
                if 'ks_test' in feature_data['test_results']:
                    feature_scores.append(1 - feature_data['test_results']['ks_test']['p_value'])
                elif 'jensen_shannon' in feature_data['test_results']:
                    feature_scores.append(feature_data['test_results']['jensen_shannon']['distance'])
            
            if feature_scores:
                drift_results['overall_drift_score'] = np.mean(feature_scores)
            
            # 5. Generate recommendations
            recommendations = await self._generate_drift_recommendations(drift_results)
            drift_results['recommendations'] = recommendations
            
            # 6. Use Evidently if available
            if EVIDENTLY_AVAILABLE:
                evidently_results = await self._run_evidently_analysis(
                    baseline_data[feature_columns],
                    current_data[feature_columns],
                    target_column=baseline_info['target_column']
                )
                drift_results['evidently_analysis'] = evidently_results
                
                # 7. Run comprehensive test suite
                test_results = await self._run_evidently_tests(
                    baseline_data[feature_columns],
                    current_data[feature_columns],
                    target_column=baseline_info['target_column']
                )
                drift_results['evidently_tests'] = test_results
            
            # Store results
            report_id = str(uuid.uuid4())
            self.drift_reports[report_id] = drift_results
            drift_results['report_id'] = report_id
            
            return drift_results
            
        except Exception as e:
            logger.error(f"Failed to detect drift for model {model_id}: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'model_id': model_id
            }
    
    async def _detect_feature_drift(
        self,
        baseline_feature: pd.Series,
        current_feature: pd.Series,
        feature_name: str,
        methods: List[str]
    ) -> Dict[str, Any]:
        """
        Detect drift for a single feature using multiple statistical tests
        """
        feature_drift = {
            'feature_name': feature_name,
            'feature_type': 'numerical' if pd.api.types.is_numeric_dtype(baseline_feature) else 'categorical',
            'test_results': {},
            'drift_detected': False,
            'severity': 'none'
        }
        
        try:
            is_numerical = pd.api.types.is_numeric_dtype(baseline_feature)
            
            # Remove NaN values for testing
            baseline_clean = baseline_feature.dropna()
            current_clean = current_feature.dropna()
            
            if len(baseline_clean) == 0 or len(current_clean) == 0:
                return feature_drift
            
            # Kolmogorov-Smirnov test
            if 'ks_test' in methods and is_numerical:
                ks_stat, ks_pvalue = stats.ks_2samp(baseline_clean, current_clean)
                feature_drift['test_results']['ks_test'] = {
                    'statistic': float(ks_stat),
                    'p_value': float(ks_pvalue),
                    'drift_detected': ks_pvalue < self.drift_thresholds['ks_test'],
                    'threshold': self.drift_thresholds['ks_test']
                }
                if ks_pvalue < self.drift_thresholds['ks_test']:
                    feature_drift['drift_detected'] = True
            
            # Jensen-Shannon divergence
            if 'jensen_shannon' in methods:
                js_distance = self._jensen_shannon_distance(baseline_clean, current_clean, is_numerical)
                feature_drift['test_results']['jensen_shannon'] = {
                    'distance': float(js_distance),
                    'drift_detected': js_distance > self.drift_thresholds['jensen_shannon'],
                    'threshold': self.drift_thresholds['jensen_shannon']
                }
                if js_distance > self.drift_thresholds['jensen_shannon']:
                    feature_drift['drift_detected'] = True
            
            # Wasserstein distance (for numerical features)
            if 'wasserstein' in methods and is_numerical:
                wasserstein_dist = stats.wasserstein_distance(baseline_clean, current_clean)
                # Normalize by the range of the baseline data
                baseline_range = baseline_clean.max() - baseline_clean.min()
                normalized_wasserstein = wasserstein_dist / baseline_range if baseline_range > 0 else wasserstein_dist
                
                feature_drift['test_results']['wasserstein'] = {
                    'distance': float(wasserstein_dist),
                    'normalized_distance': float(normalized_wasserstein),
                    'drift_detected': normalized_wasserstein > self.drift_thresholds['wasserstein'],
                    'threshold': self.drift_thresholds['wasserstein']
                }
                if normalized_wasserstein > self.drift_thresholds['wasserstein']:
                    feature_drift['drift_detected'] = True
            
            # Population Stability Index (PSI)
            if 'psi' in methods:
                psi_score = self._population_stability_index(baseline_clean, current_clean, is_numerical)
                feature_drift['test_results']['psi'] = {
                    'score': float(psi_score),
                    'drift_detected': psi_score > self.drift_thresholds['population_stability_index'],
                    'threshold': self.drift_thresholds['population_stability_index']
                }
                if psi_score > self.drift_thresholds['population_stability_index']:
                    feature_drift['drift_detected'] = True
            
            # Determine severity
            drift_scores = []
            for result in feature_drift['test_results'].values():
                if 'p_value' in result:
                    drift_scores.append(1 - result['p_value'])
                elif 'distance' in result:
                    drift_scores.append(result.get('normalized_distance', result['distance']))
                elif 'score' in result:
                    drift_scores.append(result['score'])
            
            if drift_scores:
                avg_score = np.mean(drift_scores)
                if avg_score > 0.7:
                    feature_drift['severity'] = 'high'
                elif avg_score > 0.3:
                    feature_drift['severity'] = 'medium'
                elif avg_score > 0.1:
                    feature_drift['severity'] = 'low'
        
        except Exception as e:
            logger.error(f"Error detecting drift for feature {feature_name}: {e}")
            feature_drift['error'] = str(e)
        
        return feature_drift
    
    def _jensen_shannon_distance(self, baseline: pd.Series, current: pd.Series, is_numerical: bool) -> float:
        """
        Calculate Jensen-Shannon divergence between two distributions
        """
        try:
            if is_numerical:
                # Create histograms for numerical data
                combined_min = min(baseline.min(), current.min())
                combined_max = max(baseline.max(), current.max())
                bins = np.linspace(combined_min, combined_max, 30)
                
                baseline_hist, _ = np.histogram(baseline, bins=bins, density=True)
                current_hist, _ = np.histogram(current, bins=bins, density=True)
                
                # Normalize to create probability distributions
                baseline_hist = baseline_hist / np.sum(baseline_hist)
                current_hist = current_hist / np.sum(current_hist)
            else:
                # For categorical data, use value counts
                all_categories = set(baseline.unique()) | set(current.unique())
                baseline_counts = baseline.value_counts()
                current_counts = current.value_counts()
                
                baseline_hist = np.array([baseline_counts.get(cat, 0) for cat in all_categories])
                current_hist = np.array([current_counts.get(cat, 0) for cat in all_categories])
                
                # Normalize
                baseline_hist = baseline_hist / np.sum(baseline_hist)
                current_hist = current_hist / np.sum(current_hist)
            
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            baseline_hist = baseline_hist + epsilon
            current_hist = current_hist + epsilon
            
            # Calculate Jensen-Shannon divergence
            m = 0.5 * (baseline_hist + current_hist)
            js_div = 0.5 * stats.entropy(baseline_hist, m) + 0.5 * stats.entropy(current_hist, m)
            
            return np.sqrt(js_div)
        
        except Exception as e:
            logger.error(f"Error calculating Jensen-Shannon distance: {e}")
            return 0.0
    
    def _population_stability_index(self, baseline: pd.Series, current: pd.Series, is_numerical: bool) -> float:
        """
        Calculate Population Stability Index (PSI)
        """
        try:
            if is_numerical:
                # Create quantile-based bins
                quantiles = np.linspace(0, 1, 11)  # 10 bins
                bin_edges = baseline.quantile(quantiles).values
                bin_edges[0] = -np.inf
                bin_edges[-1] = np.inf
                
                baseline_binned = pd.cut(baseline, bins=bin_edges, duplicates='drop')
                current_binned = pd.cut(current, bins=bin_edges, duplicates='drop')
                
                baseline_dist = baseline_binned.value_counts(normalize=True, sort=False)
                current_dist = current_binned.value_counts(normalize=True, sort=False)
            else:
                # For categorical data
                all_categories = set(baseline.unique()) | set(current.unique())
                baseline_dist = baseline.value_counts(normalize=True)
                current_dist = current.value_counts(normalize=True)
                
                # Ensure both distributions have the same categories
                for cat in all_categories:
                    if cat not in baseline_dist:
                        baseline_dist[cat] = 0
                    if cat not in current_dist:
                        current_dist[cat] = 0
                
                baseline_dist = baseline_dist.sort_index()
                current_dist = current_dist.sort_index()
            
            # Calculate PSI
            psi = 0
            for i in range(len(baseline_dist)):
                baseline_pct = baseline_dist.iloc[i]
                current_pct = current_dist.iloc[i]
                
                if baseline_pct == 0:
                    baseline_pct = 0.0001
                if current_pct == 0:
                    current_pct = 0.0001
                
                psi += (current_pct - baseline_pct) * np.log(current_pct / baseline_pct)
            
            return abs(psi)
        
        except Exception as e:
            logger.error(f"Error calculating PSI: {e}")
            return 0.0
    
    async def _detect_prediction_drift(self, baseline_preds: pd.Series, current_preds: pd.Series) -> Dict[str, Any]:
        """
        Detect drift in model predictions
        """
        try:
            # Statistical tests on prediction distribution
            is_classification = len(baseline_preds.unique()) < 50  # Heuristic
            
            if is_classification:
                # For classification: compare class distributions
                baseline_dist = baseline_preds.value_counts(normalize=True)
                current_dist = current_preds.value_counts(normalize=True)
                
                # Chi-square test
                all_classes = set(baseline_preds.unique()) | set(current_preds.unique())
                baseline_counts = [baseline_dist.get(cls, 0) * len(baseline_preds) for cls in all_classes]
                current_counts = [current_dist.get(cls, 0) * len(current_preds) for cls in all_classes]
                
                chi2_stat, chi2_pvalue = stats.chisquare(current_counts, baseline_counts)
                
                result = {
                    'prediction_type': 'classification',
                    'chi2_statistic': float(chi2_stat),
                    'chi2_p_value': float(chi2_pvalue),
                    'drift_detected': chi2_pvalue < self.drift_thresholds['prediction_drift'],
                    'class_distribution_baseline': baseline_dist.to_dict(),
                    'class_distribution_current': current_dist.to_dict()
                }
            else:
                # For regression: compare prediction distributions
                ks_stat, ks_pvalue = stats.ks_2samp(baseline_preds, current_preds)
                
                result = {
                    'prediction_type': 'regression',
                    'ks_statistic': float(ks_stat),
                    'ks_p_value': float(ks_pvalue),
                    'drift_detected': ks_pvalue < self.drift_thresholds['prediction_drift'],
                    'baseline_mean': float(baseline_preds.mean()),
                    'current_mean': float(current_preds.mean()),
                    'baseline_std': float(baseline_preds.std()),
                    'current_std': float(current_preds.std())
                }
            
            return result
        
        except Exception as e:
            logger.error(f"Error detecting prediction drift: {e}")
            return {'error': str(e), 'drift_detected': False}
    
    async def _analyze_data_quality(self, baseline_df: pd.DataFrame, current_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze data quality issues between baseline and current data
        """
        try:
            quality_issues = {
                'missing_values': {},
                'data_types': {},
                'duplicates': {},
                'outliers': {},
                'new_categories': {}
            }
            
            # Missing values analysis
            baseline_missing = baseline_df.isnull().sum()
            current_missing = current_df.isnull().sum()
            
            for col in baseline_df.columns:
                baseline_missing_pct = (baseline_missing[col] / len(baseline_df)) * 100
                current_missing_pct = (current_missing[col] / len(current_df)) * 100
                
                quality_issues['missing_values'][col] = {
                    'baseline_missing_pct': float(baseline_missing_pct),
                    'current_missing_pct': float(current_missing_pct),
                    'missing_increase': float(current_missing_pct - baseline_missing_pct),
                    'issue_detected': abs(current_missing_pct - baseline_missing_pct) > 5  # 5% threshold
                }
            
            # Data type consistency
            baseline_dtypes = baseline_df.dtypes
            current_dtypes = current_df.dtypes
            
            for col in baseline_df.columns:
                if col in current_df.columns:
                    quality_issues['data_types'][col] = {
                        'baseline_type': str(baseline_dtypes[col]),
                        'current_type': str(current_dtypes[col]),
                        'type_changed': baseline_dtypes[col] != current_dtypes[col]
                    }
            
            # Duplicate analysis
            baseline_duplicates = baseline_df.duplicated().sum()
            current_duplicates = current_df.duplicated().sum()
            
            quality_issues['duplicates'] = {
                'baseline_count': int(baseline_duplicates),
                'current_count': int(current_duplicates),
                'baseline_pct': float((baseline_duplicates / len(baseline_df)) * 100),
                'current_pct': float((current_duplicates / len(current_df)) * 100)
            }
            
            # New categorical values
            for col in baseline_df.select_dtypes(include=['object', 'category']).columns:
                if col in current_df.columns:
                    baseline_values = set(baseline_df[col].unique())
                    current_values = set(current_df[col].unique())
                    new_values = current_values - baseline_values
                    
                    quality_issues['new_categories'][col] = {
                        'new_values': list(new_values),
                        'new_values_count': len(new_values),
                        'has_new_values': len(new_values) > 0
                    }
            
            return quality_issues
        
        except Exception as e:
            logger.error(f"Error analyzing data quality: {e}")
            return {'error': str(e)}
    
    async def _run_evidently_analysis(
        self, 
        baseline_df: pd.DataFrame, 
        current_df: pd.DataFrame,
        target_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run Evidently AI analysis if available
        """
        if not EVIDENTLY_AVAILABLE:
            return {'error': 'Evidently not available'}
        
        try:
            # Create column mapping
            column_mapping = ColumnMapping()
            if target_column and target_column in baseline_df.columns:
                column_mapping.target = target_column
            
            # Create and run data drift report
            data_drift_report = Report(metrics=[
                DataDriftPreset(),
                DatasetDriftMetric(),
                DataDriftTable()
            ])
            
            data_drift_report.run(
                reference_data=baseline_df,
                current_data=current_df,
                column_mapping=column_mapping
            )
            
            # Extract results
            report_dict = data_drift_report.as_dict()
            
            # Simplify results for API response
            evidently_results = {
                'data_drift_detected': report_dict['metrics'][1]['result']['drift_detected'],
                'drift_score': report_dict['metrics'][1]['result']['drift_score'],
                'drifted_features_count': report_dict['metrics'][1]['result']['number_of_drifted_columns'],
                'total_features': report_dict['metrics'][1]['result']['number_of_columns'],
                'feature_drift_details': {}
            }
            
            # Extract individual feature drift details
            if 'metrics' in report_dict and len(report_dict['metrics']) > 2:
                drift_table = report_dict['metrics'][2]['result']
                if 'drift_by_columns' in drift_table:
                    for feature, drift_info in drift_table['drift_by_columns'].items():
                        evidently_results['feature_drift_details'][feature] = {
                            'drift_detected': drift_info['drift_detected'],
                            'drift_score': drift_info['drift_score'],
                            'stattest_name': drift_info['stattest_name']
                        }
            
            return evidently_results
        
        except Exception as e:
            logger.error(f"Error running Evidently analysis: {e}")
            return {'error': str(e)}
    
    async def _run_evidently_tests(
        self,
        baseline_df: pd.DataFrame,
        current_df: pd.DataFrame,
        target_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive Evidently test suite for enhanced validation
        """
        if not EVIDENTLY_AVAILABLE:
            return {'error': 'Evidently not available'}
        
        try:
            # Create column mapping
            column_mapping = ColumnMapping()
            if target_column and target_column in baseline_df.columns:
                column_mapping.target = target_column
            
            # Define comprehensive test suite
            test_suite = TestSuite(tests=[
                # Data quality tests
                TestNumberOfColumnsWithMissingValues(lt=5),
                TestNumberOfRowsWithMissingValues(lt=0.3),
                TestNumberOfConstantColumns(eq=0),
                TestNumberOfDuplicatedRows(lt=0.1),
                TestNumberOfDuplicatedColumns(eq=0),
                TestColumnsType(),
                
                # Drift tests
                TestNumberOfDriftedColumns(lt=0.5),  # Less than 50% of features drifted
                TestShareOfMissingValues(lt=0.3),
                
                # Statistical tests for numeric columns
                TestMeanInNSigmas(n_sigmas=3),
                TestValueMin(),
                TestValueMax(),
                TestValueQuantile(quantile=0.25),
                TestValueQuantile(quantile=0.75)
            ])
            
            # Run test suite
            test_suite.run(
                reference_data=baseline_df,
                current_data=current_df,
                column_mapping=column_mapping
            )
            
            # Extract test results
            test_dict = test_suite.as_dict()
            
            # Process test results
            test_results = {
                'tests_passed': 0,
                'tests_failed': 0,
                'tests_total': len(test_dict['tests']),
                'test_details': {},
                'overall_status': 'passed'
            }
            
            for test in test_dict['tests']:
                test_name = test['name']
                test_status = test['status']
                test_results['test_details'][test_name] = {
                    'status': test_status,
                    'description': test.get('description', ''),
                    'parameters': test.get('parameters', {})
                }
                
                if test_status == 'SUCCESS':
                    test_results['tests_passed'] += 1
                else:
                    test_results['tests_failed'] += 1
            
            # Determine overall status
            if test_results['tests_failed'] > 0:
                test_results['overall_status'] = 'failed'
                if test_results['tests_failed'] > test_results['tests_passed']:
                    test_results['overall_status'] = 'critical'
            
            return test_results
        
        except Exception as e:
            logger.error(f"Error running Evidently test suite: {e}")
            return {'error': str(e)}
    
    async def _generate_drift_recommendations(self, drift_results: Dict[str, Any]) -> List[str]:
        """
        Generate actionable recommendations based on drift analysis
        """
        recommendations = []
        
        try:
            # Check overall drift
            if drift_results['drift_detected']:
                recommendations.append("⚠️ Data drift detected. Consider retraining your model.")
                
                # Feature-specific recommendations
                high_drift_features = []
                for feature, drift_data in drift_results['feature_drift'].items():
                    if drift_data['severity'] == 'high':
                        high_drift_features.append(feature)
                
                if high_drift_features:
                    recommendations.append(f"🎯 High drift detected in features: {', '.join(high_drift_features)}. Investigate data collection process.")
                
                # Data quality recommendations
                if 'data_quality_issues' in drift_results:
                    quality = drift_results['data_quality_issues']
                    
                    # Missing values
                    if 'missing_values' in quality:
                        high_missing_features = [
                            feat for feat, info in quality['missing_values'].items()
                            if info.get('missing_increase', 0) > 10
                        ]
                        if high_missing_features:
                            recommendations.append(f"📊 Significant increase in missing values for: {', '.join(high_missing_features)}. Check data pipeline.")
                    
                    # New categorical values
                    if 'new_categories' in quality:
                        new_category_features = [
                            feat for feat, info in quality['new_categories'].items()
                            if info.get('has_new_values', False)
                        ]
                        if new_category_features:
                            recommendations.append(f"🆕 New categorical values detected in: {', '.join(new_category_features)}. Update preprocessing.")
            
            else:
                recommendations.append("✅ No significant data drift detected. Model performance should remain stable.")
            
            # Performance recommendations
            overall_score = drift_results.get('overall_drift_score', 0)
            if overall_score > 0.5:
                recommendations.append("🔄 Consider implementing automated retraining pipeline.")
                recommendations.append("📈 Monitor model performance metrics closely.")
            elif overall_score > 0.2:
                recommendations.append("👀 Schedule regular drift monitoring checks.")
            
            # Always include monitoring recommendation
            recommendations.append("⏰ Set up automated drift detection for continuous monitoring.")
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            recommendations.append("❌ Error generating recommendations. Please review drift results manually.")
        
        return recommendations
    
    async def get_drift_report(self, report_id: str) -> Dict[str, Any]:
        """
        Retrieve a previously generated drift report
        
        Args:
            report_id: Report identifier
            
        Returns:
            Drift report data
        """
        try:
            if report_id not in self.drift_reports:
                raise ValueError(f"Report {report_id} not found")
            
            return self.drift_reports[report_id]
        
        except Exception as e:
            logger.error(f"Error retrieving drift report: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def list_drift_reports(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """
        List all drift reports, optionally filtered by model
        
        Args:
            model_id: Optional model identifier filter
            
        Returns:
            List of drift reports
        """
        try:
            reports = []
            for report_id, report_data in self.drift_reports.items():
                if model_id is None or report_data.get('model_id') == model_id:
                    summary = {
                        'report_id': report_id,
                        'model_id': report_data.get('model_id'),
                        'detection_time': report_data.get('detection_time'),
                        'drift_detected': report_data.get('drift_detected', False),
                        'overall_drift_score': report_data.get('overall_drift_score', 0),
                        'features_analyzed': report_data.get('features_analyzed', 0)
                    }
                    reports.append(summary)
            
            return {
                'status': 'success',
                'reports': reports,
                'total_reports': len(reports)
            }
        
        except Exception as e:
            logger.error(f"Error listing drift reports: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def generate_html_report(
        self,
        model_id: str,
        current_data: pd.DataFrame,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive HTML drift report using Evidently
        """
        if not EVIDENTLY_AVAILABLE:
            return {'error': 'Evidently not available'}
        
        try:
            if model_id not in self.baseline_data:
                raise ValueError(f"No baseline data registered for model {model_id}")
            
            baseline_info = self.baseline_data[model_id]
            baseline_data = baseline_info['data']
            feature_columns = baseline_info['feature_columns']
            target_column = baseline_info['target_column']
            
            # Create comprehensive report
            report = Report(metrics=[
                DataDriftPreset(),
                TargetDriftPreset() if target_column else None,
                DatasetDriftMetric(),
                DataDriftTable(),
                DatasetMissingValuesMetric(),
                *[ColumnDriftMetric(column_name=col) for col in feature_columns[:10]]  # Limit to 10 for performance
            ])
            
            # Filter None values
            report.metrics = [m for m in report.metrics if m is not None]
            
            # Create column mapping
            column_mapping = ColumnMapping()
            if target_column and target_column in baseline_data.columns:
                column_mapping.target = target_column
            
            # Run report
            report.run(
                reference_data=baseline_data[feature_columns + ([target_column] if target_column else [])],
                current_data=current_data[feature_columns + ([target_column] if target_column and target_column in current_data.columns else [])],
                column_mapping=column_mapping
            )
            
            # Save HTML report
            if output_path is None:
                output_path = f"/tmp/drift_report_{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            
            report.save_html(output_path)
            
            return {
                'status': 'success',
                'report_path': output_path,
                'model_id': model_id,
                'generation_time': datetime.utcnow().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error generating HTML report: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def set_feature_thresholds(
        self,
        model_id: str,
        feature_thresholds: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Set custom drift thresholds for specific features
        
        Args:
            model_id: Model identifier
            feature_thresholds: Dict of {feature_name: {method: threshold}}
        """
        try:
            self.feature_thresholds[model_id] = feature_thresholds
            return {
                'status': 'success',
                'model_id': model_id,
                'thresholds_set': len(feature_thresholds)
            }
        except Exception as e:
            logger.error(f"Error setting feature thresholds: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def analyze_temporal_drift(
        self,
        model_id: str,
        time_series_data: List[Dict[str, Any]],
        window_size: int = 7
    ) -> Dict[str, Any]:
        """
        Analyze drift patterns over time using sliding window approach
        
        Args:
            model_id: Model identifier
            time_series_data: List of {timestamp, data} entries
            window_size: Size of sliding window for analysis
        """
        try:
            if model_id not in self.baseline_data:
                raise ValueError(f"No baseline data registered for model {model_id}")
            
            baseline_info = self.baseline_data[model_id]
            feature_columns = baseline_info['feature_columns']
            
            # Sort data by timestamp
            time_series_data.sort(key=lambda x: x['timestamp'])
            
            temporal_results = {
                'model_id': model_id,
                'analysis_time': datetime.utcnow().isoformat(),
                'window_size': window_size,
                'drift_timeline': [],
                'trend_analysis': {},
                'drift_velocity': {},
                'seasonal_patterns': {}
            }
            
            # Sliding window analysis
            for i in range(len(time_series_data) - window_size + 1):
                window_data = time_series_data[i:i + window_size]
                
                # Combine window data
                combined_df = pd.concat([
                    pd.DataFrame(entry['data']) for entry in window_data
                ], ignore_index=True)
                
                # Run drift detection on window
                drift_result = await self.detect_drift(
                    model_id=model_id,
                    current_data=combined_df,
                    drift_methods=['ks_test', 'jensen_shannon']
                )
                
                window_result = {
                    'window_start': window_data[0]['timestamp'],
                    'window_end': window_data[-1]['timestamp'],
                    'drift_score': drift_result.get('overall_drift_score', 0),
                    'drift_detected': drift_result.get('drift_detected', False),
                    'feature_drift_count': sum(
                        1 for f in drift_result.get('feature_drift', {}).values()
                        if f.get('drift_detected', False)
                    )
                }
                temporal_results['drift_timeline'].append(window_result)
            
            # Calculate drift velocity (rate of change)
            if len(temporal_results['drift_timeline']) > 1:
                scores = [w['drift_score'] for w in temporal_results['drift_timeline']]
                for i in range(1, len(scores)):
                    velocity = scores[i] - scores[i-1]
                    temporal_results['drift_velocity'][f'window_{i}'] = velocity
            
            return temporal_results
        
        except Exception as e:
            logger.error(f"Error analyzing temporal drift: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def track_model_performance_drift(
        self,
        model_id: str,
        predictions: pd.Series,
        actual_values: pd.Series,
        task_type: str = 'classification'
    ) -> Dict[str, Any]:
        """
        Track model performance degradation over time
        """
        try:
            if model_id not in self.baseline_data:
                raise ValueError(f"No baseline data registered for model {model_id}")
            
            baseline_info = self.baseline_data[model_id]
            
            # Calculate current performance metrics
            if task_type == 'classification':
                current_metrics = {
                    'accuracy': accuracy_score(actual_values, predictions),
                    'precision': precision_score(actual_values, predictions, average='weighted', zero_division=0),
                    'recall': recall_score(actual_values, predictions, average='weighted', zero_division=0),
                    'f1': f1_score(actual_values, predictions, average='weighted', zero_division=0)
                }
            else:  # regression
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                current_metrics = {
                    'mse': mean_squared_error(actual_values, predictions),
                    'mae': mean_absolute_error(actual_values, predictions),
                    'r2': r2_score(actual_values, predictions)
                }
            
            # Store performance metrics
            if 'performance_history' not in baseline_info:
                baseline_info['performance_history'] = []
            
            performance_entry = {
                'timestamp': datetime.utcnow().isoformat(),
                'metrics': current_metrics,
                'data_size': len(predictions)
            }
            baseline_info['performance_history'].append(performance_entry)
            
            # Analyze performance drift
            performance_drift = {
                'model_id': model_id,
                'current_metrics': current_metrics,
                'performance_drift_detected': False,
                'degradation_severity': 'none',
                'recommendations': []
            }
            
            # Compare with baseline if available
            if len(baseline_info['performance_history']) > 1:
                baseline_metrics = baseline_info['performance_history'][0]['metrics']
                
                # Calculate performance degradation
                degradation_scores = {}
                for metric, current_value in current_metrics.items():
                    baseline_value = baseline_metrics.get(metric, current_value)
                    if task_type == 'classification':
                        # Higher is better for classification metrics
                        degradation = (baseline_value - current_value) / baseline_value if baseline_value != 0 else 0
                    else:
                        # Lower is better for regression errors, higher for R2
                        if metric == 'r2':
                            degradation = (baseline_value - current_value) / baseline_value if baseline_value != 0 else 0
                        else:
                            degradation = (current_value - baseline_value) / baseline_value if baseline_value != 0 else 0
                    
                    degradation_scores[metric] = degradation
                
                # Determine overall degradation
                avg_degradation = np.mean(list(degradation_scores.values()))
                performance_drift['degradation_scores'] = degradation_scores
                performance_drift['average_degradation'] = avg_degradation
                
                if avg_degradation > 0.1:  # 10% degradation threshold
                    performance_drift['performance_drift_detected'] = True
                    if avg_degradation > 0.3:
                        performance_drift['degradation_severity'] = 'critical'
                        performance_drift['recommendations'].append("🚨 Critical performance degradation detected. Immediate model retraining required.")
                    elif avg_degradation > 0.2:
                        performance_drift['degradation_severity'] = 'high'
                        performance_drift['recommendations'].append("⚠️ High performance degradation. Schedule model retraining soon.")
                    else:
                        performance_drift['degradation_severity'] = 'medium'
                        performance_drift['recommendations'].append("📊 Moderate performance degradation detected. Monitor closely.")
            
            return performance_drift
        
        except Exception as e:
            logger.error(f"Error tracking performance drift: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def setup_automated_retraining_triggers(
        self,
        model_id: str,
        drift_threshold: float = 0.3,
        performance_threshold: float = 0.15,
        callback_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Setup automated triggers for model retraining based on drift severity
        """
        try:
            trigger_config = {
                'model_id': model_id,
                'drift_threshold': drift_threshold,
                'performance_threshold': performance_threshold,
                'callback_url': callback_url,
                'trigger_conditions': [
                    f"Overall drift score > {drift_threshold}",
                    f"Performance degradation > {performance_threshold * 100}%",
                    "Critical data quality issues detected"
                ],
                'created_at': datetime.utcnow().isoformat(),
                'triggers_fired': []
            }
            
            # Store trigger configuration
            if model_id not in self.baseline_data:
                raise ValueError(f"No baseline data registered for model {model_id}")
            
            self.baseline_data[model_id]['retraining_triggers'] = trigger_config
            
            return {
                'status': 'success',
                'trigger_config': trigger_config
            }
        
        except Exception as e:
            logger.error(f"Error setting up retraining triggers: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def check_retraining_triggers(self, drift_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if retraining should be triggered based on current drift results
        """
        try:
            model_id = drift_results.get('model_id')
            if not model_id or model_id not in self.baseline_data:
                return {'trigger_required': False}
            
            baseline_info = self.baseline_data[model_id]
            trigger_config = baseline_info.get('retraining_triggers')
            
            if not trigger_config:
                return {'trigger_required': False}
            
            trigger_result = {
                'trigger_required': False,
                'trigger_reasons': [],
                'severity': 'none',
                'recommended_actions': []
            }
            
            # Check drift threshold
            overall_score = drift_results.get('overall_drift_score', 0)
            if overall_score > trigger_config['drift_threshold']:
                trigger_result['trigger_required'] = True
                trigger_result['trigger_reasons'].append(f"Drift score ({overall_score:.3f}) exceeds threshold ({trigger_config['drift_threshold']})")
                trigger_result['severity'] = 'high'
            
            # Check for critical data quality issues
            quality_issues = drift_results.get('data_quality_issues', {})
            critical_issues = []
            
            for category, issues in quality_issues.items():
                if category == 'missing_values':
                    for feature, info in issues.items():
                        if info.get('missing_increase', 0) > 20:  # 20% increase
                            critical_issues.append(f"Critical missing values increase in {feature}")
                elif category == 'new_categories':
                    for feature, info in issues.items():
                        if info.get('new_values_count', 0) > 10:
                            critical_issues.append(f"Many new categories in {feature}")
            
            if critical_issues:
                trigger_result['trigger_required'] = True
                trigger_result['trigger_reasons'].extend(critical_issues)
                trigger_result['severity'] = 'critical'
            
            # Log trigger event
            if trigger_result['trigger_required']:
                trigger_event = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'reasons': trigger_result['trigger_reasons'],
                    'drift_score': overall_score,
                    'severity': trigger_result['severity']
                }
                trigger_config['triggers_fired'].append(trigger_event)
                
                # Generate recommendations
                if trigger_result['severity'] == 'critical':
                    trigger_result['recommended_actions'] = [
                        "🚨 Immediate model retraining required",
                        "📋 Review data collection pipeline",
                        "🔄 Update data preprocessing steps"
                    ]
                else:
                    trigger_result['recommended_actions'] = [
                        "⏰ Schedule model retraining",
                        "📊 Investigate drift patterns",
                        "🔍 Analyze feature importance changes"
                    ]
            
            return trigger_result
        
        except Exception as e:
            logger.error(f"Error checking retraining triggers: {e}")
            return {'status': 'error', 'message': str(e)}

# Global service instance
data_drift_service = DataDriftService()