"""
Comprehensive Model Statistics Service
Advanced classification and regression statistics with detailed analysis
"""

import asyncio
import json
import logging
import warnings
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    # Classification metrics
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, roc_curve, auc,
    classification_report, confusion_matrix, matthews_corrcoef,
    balanced_accuracy_score, cohen_kappa_score, log_loss,
    # Regression metrics
    mean_absolute_error, mean_squared_error, r2_score,
    explained_variance_score, max_error, mean_absolute_percentage_error,
    median_absolute_error, mean_squared_log_error
)
from sklearn.model_selection import (
    learning_curve, validation_curve, cross_val_score, 
    StratifiedKFold, KFold
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize
from scipy import stats
from scipy.stats import normaltest, shapiro, kstest
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

@dataclass
class ClassificationStats:
    """Comprehensive classification statistics"""
    # Basic metrics
    accuracy: float
    balanced_accuracy: float
    precision_macro: float
    precision_micro: float
    precision_weighted: float
    recall_macro: float
    recall_micro: float
    recall_weighted: float
    f1_macro: float
    f1_micro: float
    f1_weighted: float
    
    # Advanced metrics
    matthews_corrcoef: float
    cohen_kappa: float
    log_loss: float
    roc_auc_macro: Optional[float]
    roc_auc_micro: Optional[float]
    roc_auc_weighted: Optional[float]
    
    # Per-class metrics
    per_class_metrics: Dict[str, Dict[str, float]]
    confusion_matrix: List[List[int]]
    classification_report: Dict[str, Any]
    
    # Probability calibration
    calibration_metrics: Dict[str, Any]
    
    # Statistical tests
    statistical_tests: Dict[str, Any]

@dataclass
class RegressionStats:
    """Comprehensive regression statistics"""
    # Basic metrics
    mae: float  # Mean Absolute Error
    mse: float  # Mean Squared Error
    rmse: float  # Root Mean Squared Error
    r2: float  # R-squared
    adjusted_r2: float  # Adjusted R-squared
    explained_variance: float
    max_error: float
    median_absolute_error: float
    
    # Advanced metrics
    mape: float  # Mean Absolute Percentage Error
    msle: float  # Mean Squared Log Error
    mean_bias_error: float
    mean_relative_error: float
    
    # Statistical analysis
    residual_analysis: Dict[str, Any]
    normality_tests: Dict[str, Any]
    heteroscedasticity_tests: Dict[str, Any]
    
    # Distribution analysis
    prediction_distribution: Dict[str, Any]
    error_distribution: Dict[str, Any]

class ModelStatisticsService:
    """
    Comprehensive model statistics and analysis service
    """
    
    def __init__(self):
        self.models = {}
        self.statistics_cache = {}
        
    async def register_model(
        self,
        model_id: str,
        model: BaseEstimator,
        model_type: str,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Register a model for statistical analysis
        
        Args:
            model_id: Unique identifier for the model
            model: Trained model object
            model_type: "classification" or "regression"
            feature_names: List of feature names
            class_names: List of class names (for classification)
        
        Returns:
            Dictionary with registration status
        """
        try:
            self.models[model_id] = {
                "model": model,
                "model_type": model_type,
                "feature_names": feature_names or [],
                "class_names": class_names or [],
                "registered_at": datetime.now().isoformat()
            }
            
            return {
                "status": "success",
                "model_id": model_id,
                "model_type": model_type,
                "features_count": len(feature_names) if feature_names else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return {
                "status": "error",
                "message": str(e),
                "model_id": model_id
            }
    
    async def calculate_classification_statistics(
        self,
        model_id: str,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        X_train: Optional[pd.DataFrame] = None,
        y_train: Optional[pd.Series] = None,
        sample_weight: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive classification statistics
        
        Args:
            model_id: Model identifier
            X_test: Test features
            y_test: Test labels
            X_train: Training features (optional, for additional analysis)
            y_train: Training labels (optional, for additional analysis)
            sample_weight: Sample weights (optional)
        
        Returns:
            Dictionary with comprehensive classification statistics
        """
        try:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")
            
            model_info = self.models[model_id]
            model = model_info["model"]
            class_names = model_info["class_names"]
            
            if model_info["model_type"] != "classification":
                raise ValueError(f"Model {model_id} is not a classification model")
            
            # Get predictions
            y_pred = model.predict(X_test)
            y_prob = None
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)
            
            # Basic classification metrics
            accuracy = accuracy_score(y_test, y_pred, sample_weight=sample_weight)
            balanced_acc = balanced_accuracy_score(y_test, y_pred, sample_weight=sample_weight)
            
            # Precision, Recall, F1 with different averaging
            precision_macro = precision_score(y_test, y_pred, average='macro', sample_weight=sample_weight, zero_division=0)
            precision_micro = precision_score(y_test, y_pred, average='micro', sample_weight=sample_weight, zero_division=0)
            precision_weighted = precision_score(y_test, y_pred, average='weighted', sample_weight=sample_weight, zero_division=0)
            
            recall_macro = recall_score(y_test, y_pred, average='macro', sample_weight=sample_weight, zero_division=0)
            recall_micro = recall_score(y_test, y_pred, average='micro', sample_weight=sample_weight, zero_division=0)
            recall_weighted = recall_score(y_test, y_pred, average='weighted', sample_weight=sample_weight, zero_division=0)
            
            f1_macro = f1_score(y_test, y_pred, average='macro', sample_weight=sample_weight, zero_division=0)
            f1_micro = f1_score(y_test, y_pred, average='micro', sample_weight=sample_weight, zero_division=0)
            f1_weighted = f1_score(y_test, y_pred, average='weighted', sample_weight=sample_weight, zero_division=0)
            
            # Advanced metrics
            mcc = matthews_corrcoef(y_test, y_pred, sample_weight=sample_weight)
            kappa = cohen_kappa_score(y_test, y_pred, sample_weight=sample_weight)
            
            # Log loss (if probabilities available)
            logloss = None
            if y_prob is not None:
                try:
                    logloss = log_loss(y_test, y_prob, sample_weight=sample_weight)
                except ValueError:
                    logloss = None
            
            # ROC AUC scores
            roc_auc_macro = roc_auc_micro = roc_auc_weighted = None
            if y_prob is not None:
                try:
                    unique_classes = np.unique(y_test)
                    if len(unique_classes) == 2:
                        # Binary classification
                        roc_auc_macro = roc_auc_score(y_test, y_prob[:, 1], sample_weight=sample_weight)
                        roc_auc_micro = roc_auc_macro
                        roc_auc_weighted = roc_auc_macro
                    else:
                        # Multi-class classification
                        roc_auc_macro = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro', sample_weight=sample_weight)
                        roc_auc_micro = roc_auc_score(y_test, y_prob, multi_class='ovr', average='micro', sample_weight=sample_weight)
                        roc_auc_weighted = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted', sample_weight=sample_weight)
                except ValueError as e:
                    logger.warning(f"Could not calculate ROC AUC: {e}")
            
            # Per-class metrics
            per_class_metrics = {}
            unique_classes = np.unique(np.concatenate([y_test, y_pred]))
            for cls in unique_classes:
                cls_name = class_names[cls] if cls < len(class_names) else f"Class_{cls}"
                
                # Binary metrics for this class vs rest
                y_binary = (y_test == cls).astype(int)
                y_pred_binary = (y_pred == cls).astype(int)
                
                per_class_metrics[cls_name] = {
                    "precision": precision_score(y_binary, y_pred_binary, zero_division=0),
                    "recall": recall_score(y_binary, y_pred_binary, zero_division=0),
                    "f1_score": f1_score(y_binary, y_pred_binary, zero_division=0),
                    "support": int(np.sum(y_test == cls))
                }
                
                if y_prob is not None and len(unique_classes) == 2:
                    per_class_metrics[cls_name]["roc_auc"] = roc_auc_score(y_binary, y_prob[:, cls] if cls < y_prob.shape[1] else y_prob[:, 0])
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            cm_normalized = confusion_matrix(y_test, y_pred, normalize='true')
            
            # Classification report
            class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            
            # Probability calibration analysis
            calibration_metrics = {}
            if y_prob is not None:
                calibration_metrics = await self._analyze_probability_calibration(
                    y_test, y_prob, class_names
                )
            
            # Statistical significance tests
            statistical_tests = await self._perform_classification_statistical_tests(
                y_test, y_pred, y_prob
            )
            
            # Learning curves (if training data provided)
            learning_curves = None
            if X_train is not None and y_train is not None:
                learning_curves = await self._calculate_learning_curves(
                    model, X_train, y_train, "classification"
                )
            
            # Cross-validation analysis
            cv_analysis = await self._perform_cross_validation_analysis(
                model, X_test, y_test, "classification"
            )
            
            # Create comprehensive statistics object
            stats_obj = ClassificationStats(
                accuracy=accuracy,
                balanced_accuracy=balanced_acc,
                precision_macro=precision_macro,
                precision_micro=precision_micro,
                precision_weighted=precision_weighted,
                recall_macro=recall_macro,
                recall_micro=recall_micro,
                recall_weighted=recall_weighted,
                f1_macro=f1_macro,
                f1_micro=f1_micro,
                f1_weighted=f1_weighted,
                matthews_corrcoef=mcc,
                cohen_kappa=kappa,
                log_loss=logloss,
                roc_auc_macro=roc_auc_macro,
                roc_auc_micro=roc_auc_micro,
                roc_auc_weighted=roc_auc_weighted,
                per_class_metrics=per_class_metrics,
                confusion_matrix=cm.tolist(),
                classification_report=class_report,
                calibration_metrics=calibration_metrics,
                statistical_tests=statistical_tests
            )
            
            # Cache results
            cache_key = f"{model_id}_classification_{hash(str(X_test.values.tobytes()))}"
            self.statistics_cache[cache_key] = stats_obj
            
            result = {
                "status": "success",
                "model_id": model_id,
                "statistics": stats_obj.__dict__,
                "additional_analysis": {
                    "confusion_matrix_normalized": cm_normalized.tolist(),
                    "learning_curves": learning_curves,
                    "cross_validation": cv_analysis
                },
                "metadata": {
                    "test_samples": len(X_test),
                    "n_classes": len(unique_classes),
                    "class_distribution": {
                        class_names[cls] if cls < len(class_names) else f"Class_{cls}": int(count)
                        for cls, count in zip(*np.unique(y_test, return_counts=True))
                    },
                    "has_probabilities": y_prob is not None,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to calculate classification statistics: {e}")
            return {
                "status": "error",
                "message": str(e),
                "model_id": model_id
            }
    
    async def calculate_regression_statistics(
        self,
        model_id: str,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        X_train: Optional[pd.DataFrame] = None,
        y_train: Optional[pd.Series] = None,
        sample_weight: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive regression statistics
        
        Args:
            model_id: Model identifier
            X_test: Test features
            y_test: Test targets
            X_train: Training features (optional, for additional analysis)
            y_train: Training targets (optional, for additional analysis)
            sample_weight: Sample weights (optional)
        
        Returns:
            Dictionary with comprehensive regression statistics
        """
        try:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")
            
            model_info = self.models[model_id]
            model = model_info["model"]
            
            if model_info["model_type"] != "regression":
                raise ValueError(f"Model {model_id} is not a regression model")
            
            # Get predictions
            y_pred = model.predict(X_test)
            
            # Basic regression metrics
            mae = mean_absolute_error(y_test, y_pred, sample_weight=sample_weight)
            mse = mean_squared_error(y_test, y_pred, sample_weight=sample_weight)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred, sample_weight=sample_weight)
            
            # Adjusted R-squared
            n_samples = len(y_test)
            n_features = X_test.shape[1]
            adjusted_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
            
            explained_var = explained_variance_score(y_test, y_pred, sample_weight=sample_weight)
            max_err = max_error(y_test, y_pred)
            median_ae = median_absolute_error(y_test, y_pred, sample_weight=sample_weight)
            
            # Advanced metrics
            mape = mean_absolute_percentage_error(y_test, y_pred, sample_weight=sample_weight)
            
            # Mean Squared Log Error (handle negative values)
            msle = None
            try:
                if np.all(y_test >= 0) and np.all(y_pred >= 0):
                    msle = mean_squared_log_error(y_test, y_pred, sample_weight=sample_weight)
            except ValueError:
                msle = None
            
            # Mean Bias Error
            mean_bias_error = np.average(y_pred - y_test, weights=sample_weight)
            
            # Mean Relative Error
            mean_relative_error = np.average(
                np.abs(y_pred - y_test) / (np.abs(y_test) + 1e-8),
                weights=sample_weight
            )
            
            # Residual analysis
            residuals = y_pred - y_test
            residual_analysis = {
                "mean": float(np.mean(residuals)),
                "std": float(np.std(residuals)),
                "min": float(np.min(residuals)),
                "max": float(np.max(residuals)),
                "median": float(np.median(residuals)),
                "q25": float(np.percentile(residuals, 25)),
                "q75": float(np.percentile(residuals, 75)),
                "skewness": float(stats.skew(residuals)),
                "kurtosis": float(stats.kurtosis(residuals))
            }
            
            # Normality tests for residuals
            normality_tests = {}
            try:
                # Shapiro-Wilk test (for n < 5000)
                if len(residuals) <= 5000:
                    shapiro_stat, shapiro_p = shapiro(residuals)
                    normality_tests["shapiro_wilk"] = {
                        "statistic": float(shapiro_stat),
                        "p_value": float(shapiro_p),
                        "is_normal": shapiro_p > 0.05
                    }
                
                # D'Agostino-Pearson test
                dagostino_stat, dagostino_p = normaltest(residuals)
                normality_tests["dagostino_pearson"] = {
                    "statistic": float(dagostino_stat),
                    "p_value": float(dagostino_p),
                    "is_normal": dagostino_p > 0.05
                }
                
                # Kolmogorov-Smirnov test against normal distribution
                ks_stat, ks_p = kstest(residuals, 'norm', args=(np.mean(residuals), np.std(residuals)))
                normality_tests["kolmogorov_smirnov"] = {
                    "statistic": float(ks_stat),
                    "p_value": float(ks_p),
                    "is_normal": ks_p > 0.05
                }
                
            except Exception as e:
                logger.warning(f"Failed to perform normality tests: {e}")
            
            # Heteroscedasticity tests (Breusch-Pagan)
            heteroscedasticity_tests = {}
            try:
                # Simple Breusch-Pagan test
                residuals_squared = residuals ** 2
                correlation = np.corrcoef(y_pred, residuals_squared)[0, 1]
                heteroscedasticity_tests["breusch_pagan_correlation"] = {
                    "correlation": float(correlation),
                    "suggests_heteroscedasticity": abs(correlation) > 0.1
                }
                
                # Variance ratio test (split into groups)
                n_groups = 3
                group_size = len(y_pred) // n_groups
                group_variances = []
                
                for i in range(n_groups):
                    start_idx = i * group_size
                    end_idx = (i + 1) * group_size if i < n_groups - 1 else len(y_pred)
                    group_residuals = residuals[start_idx:end_idx]
                    if len(group_residuals) > 1:
                        group_variances.append(np.var(group_residuals))
                
                if len(group_variances) > 1:
                    variance_ratio = max(group_variances) / min(group_variances)
                    heteroscedasticity_tests["variance_ratio"] = {
                        "ratio": float(variance_ratio),
                        "suggests_heteroscedasticity": variance_ratio > 3.0
                    }
                
            except Exception as e:
                logger.warning(f"Failed to perform heteroscedasticity tests: {e}")
            
            # Prediction distribution analysis
            prediction_distribution = {
                "mean": float(np.mean(y_pred)),
                "std": float(np.std(y_pred)),
                "min": float(np.min(y_pred)),
                "max": float(np.max(y_pred)),
                "median": float(np.median(y_pred)),
                "q25": float(np.percentile(y_pred, 25)),
                "q75": float(np.percentile(y_pred, 75)),
                "skewness": float(stats.skew(y_pred)),
                "kurtosis": float(stats.kurtosis(y_pred))
            }
            
            # Error distribution analysis
            abs_errors = np.abs(residuals)
            error_distribution = {
                "mean_abs_error": float(np.mean(abs_errors)),
                "std_abs_error": float(np.std(abs_errors)),
                "median_abs_error": float(np.median(abs_errors)),
                "q90_abs_error": float(np.percentile(abs_errors, 90)),
                "q95_abs_error": float(np.percentile(abs_errors, 95)),
                "q99_abs_error": float(np.percentile(abs_errors, 99))
            }
            
            # Learning curves (if training data provided)
            learning_curves = None
            if X_train is not None and y_train is not None:
                learning_curves = await self._calculate_learning_curves(
                    model, X_train, y_train, "regression"
                )
            
            # Cross-validation analysis
            cv_analysis = await self._perform_cross_validation_analysis(
                model, X_test, y_test, "regression"
            )
            
            # Create comprehensive statistics object
            stats_obj = RegressionStats(
                mae=mae,
                mse=mse,
                rmse=rmse,
                r2=r2,
                adjusted_r2=adjusted_r2,
                explained_variance=explained_var,
                max_error=max_err,
                median_absolute_error=median_ae,
                mape=mape,
                msle=msle,
                mean_bias_error=mean_bias_error,
                mean_relative_error=mean_relative_error,
                residual_analysis=residual_analysis,
                normality_tests=normality_tests,
                heteroscedasticity_tests=heteroscedasticity_tests,
                prediction_distribution=prediction_distribution,
                error_distribution=error_distribution
            )
            
            # Cache results
            cache_key = f"{model_id}_regression_{hash(str(X_test.values.tobytes()))}"
            self.statistics_cache[cache_key] = stats_obj
            
            result = {
                "status": "success",
                "model_id": model_id,
                "statistics": stats_obj.__dict__,
                "additional_analysis": {
                    "learning_curves": learning_curves,
                    "cross_validation": cv_analysis,
                    "residual_vs_fitted": {
                        "fitted_values": y_pred.tolist(),
                        "residuals": residuals.tolist()
                    }
                },
                "metadata": {
                    "test_samples": len(X_test),
                    "target_distribution": {
                        "mean": float(np.mean(y_test)),
                        "std": float(np.std(y_test)),
                        "min": float(np.min(y_test)),
                        "max": float(np.max(y_test))
                    },
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to calculate regression statistics: {e}")
            return {
                "status": "error",
                "message": str(e),
                "model_id": model_id
            }
    
    async def _analyze_probability_calibration(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        class_names: List[str]
    ) -> Dict[str, Any]:
        """Analyze probability calibration"""
        
        calibration_analysis = {}
        
        try:
            if y_prob.shape[1] == 2:
                # Binary classification
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_true, y_prob[:, 1], n_bins=10
                )
                
                calibration_analysis["binary"] = {
                    "fraction_of_positives": fraction_of_positives.tolist(),
                    "mean_predicted_value": mean_predicted_value.tolist(),
                    "calibration_error": float(np.mean(np.abs(fraction_of_positives - mean_predicted_value))),
                    "max_calibration_error": float(np.max(np.abs(fraction_of_positives - mean_predicted_value)))
                }
            
            else:
                # Multi-class calibration
                calibration_errors = []
                for class_idx in range(y_prob.shape[1]):
                    y_binary = (y_true == class_idx).astype(int)
                    if np.sum(y_binary) > 0:  # Only if class exists in test set
                        fraction_of_positives, mean_predicted_value = calibration_curve(
                            y_binary, y_prob[:, class_idx], n_bins=10
                        )
                        
                        class_name = class_names[class_idx] if class_idx < len(class_names) else f"Class_{class_idx}"
                        calibration_errors.append({
                            "class": class_name,
                            "calibration_error": float(np.mean(np.abs(fraction_of_positives - mean_predicted_value)))
                        })
                
                calibration_analysis["multiclass"] = {
                    "per_class_errors": calibration_errors,
                    "average_calibration_error": float(np.mean([ce["calibration_error"] for ce in calibration_errors]))
                }
            
            # Reliability diagram data
            calibration_analysis["reliability_diagram_data"] = {
                "available": True,
                "description": "Use fraction_of_positives and mean_predicted_value to create reliability diagram"
            }
            
        except Exception as e:
            logger.warning(f"Failed to analyze probability calibration: {e}")
            calibration_analysis = {"error": str(e)}
        
        return calibration_analysis
    
    async def _perform_classification_statistical_tests(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Perform statistical significance tests for classification"""
        
        tests = {}
        
        try:
            # McNemar's test (requires paired data with another classifier)
            # For now, we'll implement a basic binomial test for accuracy
            n_correct = np.sum(y_true == y_pred)
            n_total = len(y_true)
            
            # Test if accuracy is significantly better than random (0.5 for binary, 1/n_classes for multiclass)
            n_classes = len(np.unique(y_true))
            null_accuracy = 1.0 / n_classes
            
            # Binomial test
            from scipy.stats import binom
            p_value = 1 - binom.cdf(n_correct - 1, n_total, null_accuracy)
            
            tests["accuracy_vs_random"] = {
                "test_type": "binomial",
                "null_hypothesis": f"Accuracy = {null_accuracy} (random guessing)",
                "observed_accuracy": float(n_correct / n_total),
                "p_value": float(p_value),
                "is_significant": p_value < 0.05
            }
            
            # Chi-square test for independence (if sufficient data)
            if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
                from scipy.stats import chi2_contingency
                
                contingency_table = pd.crosstab(y_true, y_pred)
                chi2, p_chi2, dof, expected = chi2_contingency(contingency_table)
                
                tests["prediction_independence"] = {
                    "test_type": "chi_square",
                    "null_hypothesis": "Predictions are independent of true labels",
                    "chi2_statistic": float(chi2),
                    "p_value": float(p_chi2),
                    "degrees_of_freedom": int(dof),
                    "is_significant": p_chi2 < 0.05
                }
            
        except Exception as e:
            logger.warning(f"Failed to perform statistical tests: {e}")
            tests["error"] = str(e)
        
        return tests
    
    async def _calculate_learning_curves(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        model_type: str,
        cv: int = 5
    ) -> Dict[str, Any]:
        """Calculate learning curves"""
        
        try:
            # Determine scoring method
            scoring = 'accuracy' if model_type == 'classification' else 'r2'
            
            # Calculate learning curves
            train_sizes, train_scores, val_scores = learning_curve(
                model, X, y,
                cv=cv,
                n_jobs=-1,
                train_sizes=np.linspace(0.1, 1.0, 10),
                scoring=scoring,
                random_state=42
            )
            
            return {
                "train_sizes": train_sizes.tolist(),
                "train_scores_mean": train_scores.mean(axis=1).tolist(),
                "train_scores_std": train_scores.std(axis=1).tolist(),
                "validation_scores_mean": val_scores.mean(axis=1).tolist(),
                "validation_scores_std": val_scores.std(axis=1).tolist(),
                "scoring_method": scoring,
                "cv_folds": cv
            }
            
        except Exception as e:
            logger.warning(f"Failed to calculate learning curves: {e}")
            return {"error": str(e)}
    
    async def _perform_cross_validation_analysis(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        model_type: str,
        cv: int = 5
    ) -> Dict[str, Any]:
        """Perform cross-validation analysis"""
        
        try:
            # Determine CV strategy and scoring
            if model_type == 'classification':
                cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
                scoring_methods = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
            else:
                cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=42)
                scoring_methods = ['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error']
            
            cv_results = {}
            for scoring in scoring_methods:
                try:
                    scores = cross_val_score(model, X, y, cv=cv_strategy, scoring=scoring, n_jobs=-1)
                    cv_results[scoring] = {
                        "scores": scores.tolist(),
                        "mean": float(scores.mean()),
                        "std": float(scores.std()),
                        "min": float(scores.min()),
                        "max": float(scores.max())
                    }
                except Exception as e:
                    logger.warning(f"Failed to calculate CV score for {scoring}: {e}")
            
            return {
                "cv_folds": cv,
                "results": cv_results,
                "cv_strategy": str(cv_strategy)
            }
            
        except Exception as e:
            logger.warning(f"Failed to perform cross-validation analysis: {e}")
            return {"error": str(e)}
    
    async def compare_models(
        self,
        model_comparisons: List[Dict[str, Any]],
        comparison_metric: str = "auto"
    ) -> Dict[str, Any]:
        """
        Compare multiple models statistically
        
        Args:
            model_comparisons: List of model comparison data
            comparison_metric: Metric to use for comparison
        
        Returns:
            Dictionary with model comparison results
        """
        try:
            if len(model_comparisons) < 2:
                raise ValueError("Need at least 2 models for comparison")
            
            # Determine comparison metric based on model type
            model_type = model_comparisons[0].get("model_type", "classification")
            if comparison_metric == "auto":
                comparison_metric = "accuracy" if model_type == "classification" else "r2"
            
            # Extract comparison data
            comparison_results = []
            for model_data in model_comparisons:
                model_id = model_data["model_id"]
                statistics = model_data["statistics"]
                
                if model_type == "classification":
                    metric_value = statistics.get(comparison_metric, 0)
                else:
                    metric_value = statistics.get(comparison_metric, 0)
                
                comparison_results.append({
                    "model_id": model_id,
                    "metric_value": metric_value,
                    "statistics": statistics
                })
            
            # Sort by metric value
            comparison_results.sort(key=lambda x: x["metric_value"], reverse=True)
            
            # Statistical significance testing (paired t-test if CV scores available)
            significance_tests = {}
            if len(comparison_results) == 2:
                # Pairwise comparison
                model1_id = comparison_results[0]["model_id"]
                model2_id = comparison_results[1]["model_id"]
                
                # Check if both have CV results
                cv1 = comparison_results[0]["statistics"].get("cross_validation", {}).get("results", {})
                cv2 = comparison_results[1]["statistics"].get("cross_validation", {}).get("results", {})
                
                if comparison_metric in cv1 and comparison_metric in cv2:
                    scores1 = cv1[comparison_metric]["scores"]
                    scores2 = cv2[comparison_metric]["scores"]
                    
                    if len(scores1) == len(scores2):
                        # Paired t-test
                        from scipy.stats import ttest_rel
                        t_stat, p_value = ttest_rel(scores1, scores2)
                        
                        significance_tests[f"{model1_id}_vs_{model2_id}"] = {
                            "test_type": "paired_t_test",
                            "t_statistic": float(t_stat),
                            "p_value": float(p_value),
                            "is_significant": p_value < 0.05,
                            "better_model": model1_id if t_stat > 0 else model2_id
                        }
            
            # Generate comparison summary
            best_model = comparison_results[0]
            worst_model = comparison_results[-1]
            
            performance_gap = best_model["metric_value"] - worst_model["metric_value"]
            relative_improvement = (performance_gap / worst_model["metric_value"] * 100) if worst_model["metric_value"] > 0 else 0
            
            return {
                "status": "success",
                "comparison_metric": comparison_metric,
                "model_rankings": comparison_results,
                "best_model": {
                    "model_id": best_model["model_id"],
                    "metric_value": best_model["metric_value"]
                },
                "performance_analysis": {
                    "performance_gap": float(performance_gap),
                    "relative_improvement_percent": float(relative_improvement),
                    "models_compared": len(comparison_results)
                },
                "significance_tests": significance_tests,
                "metadata": {
                    "comparison_timestamp": datetime.now().isoformat(),
                    "model_type": model_type
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to compare models: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def get_cached_statistics(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached statistics"""
        return self.statistics_cache.get(cache_key)
    
    async def clear_cache(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """Clear statistics cache"""
        try:
            if model_id:
                # Clear cache for specific model
                keys_to_remove = [key for key in self.statistics_cache.keys() if key.startswith(model_id)]
                for key in keys_to_remove:
                    del self.statistics_cache[key]
                return {
                    "status": "success",
                    "cleared_entries": len(keys_to_remove),
                    "model_id": model_id
                }
            else:
                # Clear all cache
                cache_size = len(self.statistics_cache)
                self.statistics_cache.clear()
                return {
                    "status": "success",
                    "cleared_entries": cache_size
                }
                
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def cleanup_model(self, model_id: str) -> Dict[str, Any]:
        """
        Clean up model resources
        """
        try:
            removed_components = []
            
            if model_id in self.models:
                del self.models[model_id]
                removed_components.append("model")
            
            # Clear related cache entries
            keys_to_remove = [key for key in self.statistics_cache.keys() if key.startswith(model_id)]
            for key in keys_to_remove:
                del self.statistics_cache[key]
            
            if keys_to_remove:
                removed_components.append("cache_entries")
            
            return {
                "status": "success",
                "model_id": model_id,
                "removed_components": removed_components,
                "cache_entries_removed": len(keys_to_remove)
            }
            
        except Exception as e:
            logger.error(f"Failed to cleanup model statistics: {e}")
            return {
                "status": "error",
                "message": str(e),
                "model_id": model_id
            }

# Global service instance
model_statistics_service = ModelStatisticsService()