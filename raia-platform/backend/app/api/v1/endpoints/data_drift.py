"""
Data Drift Detection API Endpoints
Comprehensive endpoints for drift monitoring and analysis
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, UploadFile, File, Query
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
import io
import json
from datetime import datetime, timedelta

from app.core.database import get_database
from app.core.auth import get_current_user
from app.models.schemas import User
from app.services.data_drift_service import data_drift_service
from app.core.exceptions import RAIAException

router = APIRouter()

@router.post("/models/{model_id}/baseline")
async def register_baseline_data(
    model_id: str,
    baseline_file: UploadFile = File(...),
    target_column: Optional[str] = None,
    prediction_column: Optional[str] = None,
    feature_columns: Optional[str] = None,  # JSON string of list
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_database)
):
    """
    Register baseline/reference data for drift detection
    
    Args:
        model_id: Unique identifier for the model
        baseline_file: CSV file with baseline data
        target_column: Name of target column
        prediction_column: Name of prediction column
        feature_columns: JSON string of feature column names
    
    Returns:
        Registration status and baseline statistics
    """
    try:
        # Read baseline data
        baseline_content = await baseline_file.read()
        baseline_df = pd.read_csv(io.StringIO(baseline_content.decode('utf-8')))
        
        # Parse feature columns if provided
        feature_column_list = None
        if feature_columns:
            try:
                feature_column_list = json.loads(feature_columns)
            except json.JSONDecodeError:
                feature_column_list = [col.strip() for col in feature_columns.split(',')]
        
        result = await data_drift_service.register_baseline(
            model_id=model_id,
            baseline_data=baseline_df,
            target_column=target_column,
            prediction_column=prediction_column,
            feature_columns=feature_column_list
        )
        
        if result['status'] == 'error':
            raise HTTPException(status_code=400, detail=result['message'])
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to register baseline data: {str(e)}")

@router.post("/models/{model_id}/drift-detection")
async def detect_data_drift(
    model_id: str,
    current_data_file: UploadFile = File(...),
    drift_methods: List[str] = Query(default=['ks_test', 'jensen_shannon', 'wasserstein']),
    include_prediction_drift: bool = Query(default=True),
    background_tasks: BackgroundTasks = None,
    current_user: User = Depends(get_current_user)
):
    """
    Detect data drift between baseline and current data
    
    Args:
        model_id: Model identifier
        current_data_file: CSV file with current data
        drift_methods: List of drift detection methods to use
        include_prediction_drift: Whether to detect prediction drift
    
    Returns:
        Comprehensive drift analysis results
    """
    try:
        # Read current data
        current_content = await current_data_file.read()
        current_df = pd.read_csv(io.StringIO(current_content.decode('utf-8')))
        
        # Run drift detection
        result = await data_drift_service.detect_drift(
            model_id=model_id,
            current_data=current_df,
            drift_methods=drift_methods,
            include_prediction_drift=include_prediction_drift
        )
        
        if result.get('status') == 'error':
            raise HTTPException(status_code=400, detail=result['message'])
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to detect data drift: {str(e)}")

@router.get("/models/{model_id}/drift-history")
async def get_drift_history(
    model_id: str,
    limit: int = Query(default=10, ge=1, le=100),
    current_user: User = Depends(get_current_user)
):
    """
    Get drift detection history for a model
    
    Args:
        model_id: Model identifier
        limit: Maximum number of reports to return
    
    Returns:
        List of drift detection reports
    """
    try:
        result = await data_drift_service.list_drift_reports(model_id=model_id)
        
        if result['status'] == 'error':
            raise HTTPException(status_code=400, detail=result['message'])
        
        # Apply limit
        reports = result['reports'][:limit]
        
        return JSONResponse(content={
            'status': 'success',
            'model_id': model_id,
            'reports': reports,
            'total_reports': len(reports),
            'limit_applied': limit
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get drift history: {str(e)}")

@router.get("/drift-reports/{report_id}")
async def get_drift_report_details(
    report_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get detailed drift report by ID
    
    Args:
        report_id: Report identifier
    
    Returns:
        Detailed drift report data
    """
    try:
        result = await data_drift_service.get_drift_report(report_id)
        
        if result.get('status') == 'error':
            raise HTTPException(status_code=404, detail=result['message'])
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get drift report: {str(e)}")

@router.get("/drift-reports")
async def list_all_drift_reports(
    limit: int = Query(default=50, ge=1, le=200),
    model_id: Optional[str] = Query(default=None),
    current_user: User = Depends(get_current_user)
):
    """
    List all drift reports across all models
    
    Args:
        limit: Maximum number of reports to return
        model_id: Optional filter by model ID
    
    Returns:
        List of all drift reports
    """
    try:
        result = await data_drift_service.list_drift_reports(model_id=model_id)
        
        if result['status'] == 'error':
            raise HTTPException(status_code=400, detail=result['message'])
        
        # Apply limit and sorting
        reports = sorted(
            result['reports'], 
            key=lambda x: x.get('detection_time', ''), 
            reverse=True
        )[:limit]
        
        return JSONResponse(content={
            'status': 'success',
            'reports': reports,
            'total_reports': len(reports),
            'filter_applied': {'model_id': model_id, 'limit': limit}
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list drift reports: {str(e)}")

@router.post("/models/{model_id}/drift-thresholds")
async def configure_drift_thresholds(
    model_id: str,
    thresholds: Dict[str, float],
    current_user: User = Depends(get_current_user)
):
    """
    Configure custom drift detection thresholds for a model
    
    Args:
        model_id: Model identifier
        thresholds: Dictionary of threshold values for different tests
    
    Returns:
        Configuration status
    """
    try:
        # Validate threshold values
        valid_methods = ['ks_test', 'jensen_shannon', 'wasserstein', 'population_stability_index', 'prediction_drift']
        
        for method, threshold in thresholds.items():
            if method not in valid_methods:
                raise HTTPException(status_code=400, detail=f"Invalid drift method: {method}")
            if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 1:
                raise HTTPException(status_code=400, detail=f"Invalid threshold value for {method}: {threshold}")
        
        # Update thresholds in service
        data_drift_service.drift_thresholds.update(thresholds)
        
        return JSONResponse(content={
            'status': 'success',
            'model_id': model_id,
            'updated_thresholds': thresholds,
            'current_thresholds': data_drift_service.drift_thresholds
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to configure drift thresholds: {str(e)}")

@router.get("/models/{model_id}/drift-thresholds")
async def get_drift_thresholds(
    model_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get current drift detection thresholds for a model
    
    Args:
        model_id: Model identifier
    
    Returns:
        Current threshold configuration
    """
    try:
        return JSONResponse(content={
            'status': 'success',
            'model_id': model_id,
            'drift_thresholds': data_drift_service.drift_thresholds,
            'available_methods': list(data_drift_service.drift_thresholds.keys())
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get drift thresholds: {str(e)}")

@router.post("/models/{model_id}/simulated-drift")
async def generate_simulated_drift(
    model_id: str,
    baseline_file: UploadFile = File(...),
    drift_magnitude: float = Query(default=0.3, ge=0.1, le=2.0),
    drift_features: Optional[List[str]] = Query(default=None),
    drift_type: str = Query(default="shift", regex="^(shift|scale|noise|missing)$"),
    current_user: User = Depends(get_current_user)
):
    """
    Generate simulated drift data for testing purposes
    
    Args:
        model_id: Model identifier
        baseline_file: CSV file with baseline data
        drift_magnitude: Magnitude of drift to introduce (0.1 = small, 2.0 = large)
        drift_features: List of features to apply drift to (None = random selection)
        drift_type: Type of drift (shift, scale, noise, missing)
    
    Returns:
        Simulated drift data and detection results
    """
    try:
        # Read baseline data
        baseline_content = await baseline_file.read()
        baseline_df = pd.read_csv(io.StringIO(baseline_content.decode('utf-8')))
        
        # Generate simulated drift
        drifted_df = baseline_df.copy()
        
        # Select features to drift
        numeric_features = baseline_df.select_dtypes(include=[np.number]).columns.tolist()
        if drift_features is None:
            # Randomly select 20-50% of numeric features
            n_features = max(1, int(len(numeric_features) * np.random.uniform(0.2, 0.5)))
            drift_features = np.random.choice(numeric_features, size=n_features, replace=False).tolist()
        
        applied_drifts = []
        
        for feature in drift_features:
            if feature in numeric_features:
                original_mean = baseline_df[feature].mean()
                original_std = baseline_df[feature].std()
                
                if drift_type == "shift":
                    # Add mean shift
                    shift = drift_magnitude * original_std
                    drifted_df[feature] = drifted_df[feature] + shift
                    applied_drifts.append({
                        'feature': feature,
                        'type': 'mean_shift',
                        'magnitude': float(shift),
                        'relative_magnitude': float(shift / original_std)
                    })
                
                elif drift_type == "scale":
                    # Change variance
                    scale_factor = 1 + drift_magnitude
                    drifted_df[feature] = (drifted_df[feature] - original_mean) * scale_factor + original_mean
                    applied_drifts.append({
                        'feature': feature,
                        'type': 'variance_change',
                        'scale_factor': float(scale_factor)
                    })
                
                elif drift_type == "noise":
                    # Add random noise
                    noise = np.random.normal(0, drift_magnitude * original_std, len(drifted_df))
                    drifted_df[feature] = drifted_df[feature] + noise
                    applied_drifts.append({
                        'feature': feature,
                        'type': 'added_noise',
                        'noise_std': float(drift_magnitude * original_std)
                    })
                
                elif drift_type == "missing":
                    # Introduce missing values
                    missing_rate = min(drift_magnitude, 0.5)  # Cap at 50%
                    missing_indices = np.random.choice(
                        len(drifted_df), 
                        size=int(len(drifted_df) * missing_rate), 
                        replace=False
                    )
                    drifted_df.loc[missing_indices, feature] = np.nan
                    applied_drifts.append({
                        'feature': feature,
                        'type': 'missing_values',
                        'missing_rate': float(missing_rate)
                    })
        
        # Run drift detection on simulated data
        drift_result = await data_drift_service.detect_drift(
            model_id=model_id,
            current_data=drifted_df,
            drift_methods=['ks_test', 'jensen_shannon', 'wasserstein'],
            include_prediction_drift=False
        )
        
        # Combine results
        simulation_result = {
            'status': 'success',
            'model_id': model_id,
            'simulation_parameters': {
                'drift_magnitude': drift_magnitude,
                'drift_type': drift_type,
                'target_features': drift_features,
                'applied_drifts': applied_drifts
            },
            'drift_detection_results': drift_result,
            'baseline_shape': baseline_df.shape,
            'drifted_shape': drifted_df.shape,
            'simulation_timestamp': datetime.utcnow().isoformat()
        }
        
        return JSONResponse(content=simulation_result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate simulated drift: {str(e)}")

@router.get("/drift-methods")
async def get_available_drift_methods():
    """
    Get information about available drift detection methods
    
    Returns:
        Comprehensive information about drift detection methods
    """
    methods_info = {
        'statistical_tests': {
            'ks_test': {
                'name': 'Kolmogorov-Smirnov Test',
                'description': 'Two-sample test for comparing distributions',
                'best_for': 'Numerical features with continuous distributions',
                'null_hypothesis': 'Two samples come from the same distribution',
                'p_value_threshold': 0.05,
                'pros': ['Non-parametric', 'No assumptions about distribution shape'],
                'cons': ['Sensitive to sample size', 'May miss subtle drifts']
            },
            'jensen_shannon': {
                'name': 'Jensen-Shannon Divergence',
                'description': 'Symmetric measure of similarity between probability distributions',
                'best_for': 'Both numerical and categorical features',
                'distance_threshold': 0.1,
                'pros': ['Bounded (0-1)', 'Symmetric', 'Works with any distribution'],
                'cons': ['May be less sensitive to tail differences']
            },
            'wasserstein': {
                'name': 'Wasserstein Distance (Earth Movers Distance)',
                'description': 'Minimum cost to transform one distribution into another',
                'best_for': 'Numerical features where order matters',
                'pros': ['Considers distance between values', 'Intuitive interpretation'],
                'cons': ['Only for numerical data', 'Computationally expensive']
            },
            'psi': {
                'name': 'Population Stability Index',
                'description': 'Measures change in distribution across bins',
                'best_for': 'Feature stability monitoring in credit scoring',
                'thresholds': {
                    'no_change': 0.1,
                    'small_change': 0.2,
                    'major_change': 0.25
                },
                'pros': ['Industry standard in finance', 'Easy to interpret'],
                'cons': ['Requires binning', 'Less sensitive to shape changes']
            }
        },
        'prediction_drift': {
            'classification': {
                'method': 'Chi-square test',
                'description': 'Tests for changes in class distribution'
            },
            'regression': {
                'method': 'Kolmogorov-Smirnov test',
                'description': 'Tests for changes in prediction distribution'
            }
        },
        'data_quality_checks': [
            'Missing value rates',
            'Data type consistency',
            'Duplicate detection',
            'New categorical values',
            'Outlier detection'
        ],
        'recommended_workflow': [
            '1. Register baseline data with representative sample',
            '2. Use KS test for initial numerical drift screening',
            '3. Apply Jensen-Shannon for comprehensive analysis',
            '4. Monitor prediction drift for model performance',
            '5. Set up regular automated drift checks',
            '6. Configure alerts based on business requirements'
        ]
    }
    
    return JSONResponse(content=methods_info)

@router.get("/models/{model_id}/baseline-status")
async def get_baseline_status(
    model_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get baseline data registration status for a model
    
    Args:
        model_id: Model identifier
    
    Returns:
        Baseline registration status and metadata
    """
    try:
        if model_id not in data_drift_service.baseline_data:
            return JSONResponse(content={
                'status': 'success',
                'model_id': model_id,
                'baseline_registered': False,
                'message': 'No baseline data registered for this model'
            })
        
        baseline_info = data_drift_service.baseline_data[model_id]
        
        return JSONResponse(content={
            'status': 'success',
            'model_id': model_id,
            'baseline_registered': True,
            'registration_time': baseline_info['registration_time'].isoformat(),
            'data_shape': baseline_info['data'].shape,
            'feature_columns': baseline_info['feature_columns'],
            'target_column': baseline_info['target_column'],
            'prediction_column': baseline_info['prediction_column'],
            'baseline_stats_summary': {
                'total_features': len(baseline_info['feature_columns']),
                'numerical_features': len(baseline_info['baseline_stats']['numeric_stats']),
                'categorical_features': len(baseline_info['baseline_stats']['categorical_stats']),
                'total_missing_values': sum(baseline_info['baseline_stats']['missing_values'].values())
            }
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get baseline status: {str(e)}")