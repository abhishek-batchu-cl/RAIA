# Time Series Analytics Service
import os
import json
import uuid
import asyncio
import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from sqlalchemy.orm import Session
from sqlalchemy import Column, String, DateTime, Text, Boolean, JSON, UUID as PG_UUID, Integer, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

# Time series analysis libraries
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

# Forecasting models
try:
    # Facebook Prophet
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    # Statistical models
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    # Deep learning models
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import MinMaxScaler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Anomaly detection
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Changepoint detection
try:
    import ruptures as rpt
    RUPTURES_AVAILABLE = True
except ImportError:
    RUPTURES_AVAILABLE = False

Base = declarative_base()
logger = logging.getLogger(__name__)

@dataclass
class TimeSeriesConfig:
    """Configuration for time series analysis"""
    name: str
    description: str
    data_source: Dict[str, Any]
    target_variable: str
    frequency: str  # 'H', 'D', 'W', 'M', 'Q', 'Y'
    seasonality_periods: List[int] = None  # e.g., [24, 168] for hourly data with daily and weekly seasonality
    outlier_detection: bool = True
    missing_value_strategy: str = 'interpolate'  # 'interpolate', 'forward_fill', 'drop'

@dataclass
class ForecastConfig:
    """Configuration for forecasting"""
    model_type: str  # 'arima', 'prophet', 'lstm', 'transformer', 'ensemble'
    horizon: int  # Number of periods to forecast
    confidence_intervals: List[float] = None  # e.g., [0.8, 0.95]
    hyperparameters: Dict[str, Any] = None
    retrain_frequency: str = 'weekly'  # 'daily', 'weekly', 'monthly'
    enable_holidays: bool = False
    enable_events: bool = False

@dataclass
class AnomalyDetectionConfig:
    """Configuration for anomaly detection"""
    methods: List[str] = None  # 'isolation_forest', 'one_class_svm', 'dbscan', 'statistical'
    contamination: float = 0.1  # Expected proportion of anomalies
    window_size: int = 24  # For rolling statistics
    threshold_std: float = 3.0  # For statistical method

class TimeSeriesModel(Base):
    """Store time series models and configurations"""
    __tablename__ = "timeseries_models"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(String(255), unique=True, nullable=False)
    name = Column(String(500), nullable=False)
    description = Column(Text)
    
    # Data configuration
    data_source_config = Column(JSON)
    target_variable = Column(String(255), nullable=False)
    frequency = Column(String(10), nullable=False)
    
    # Model configuration
    model_type = Column(String(100), nullable=False)
    model_config = Column(JSON)
    forecast_horizon = Column(Integer)
    
    # Performance metrics
    accuracy_metrics = Column(JSON)
    validation_scores = Column(JSON)
    
    # Status and metadata
    status = Column(String(50), default='created')  # created, training, ready, error
    training_progress = Column(Float, default=0.0)
    last_trained = Column(DateTime)
    next_retrain = Column(DateTime)
    
    # Seasonality and patterns
    seasonality_info = Column(JSON)
    trend_info = Column(JSON)
    changepoints = Column(JSON)
    
    # Anomaly detection
    anomaly_config = Column(JSON)
    anomaly_threshold = Column(Float)
    
    # File paths
    model_file_path = Column(String(1000))
    data_file_path = Column(String(1000))
    
    # User info
    created_by = Column(String(255))
    organization_id = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    forecasts = relationship("TimeSeriesForecast", back_populates="model")
    anomalies = relationship("TimeSeriesAnomaly", back_populates="model")

class TimeSeriesForecast(Base):
    """Store forecast results"""
    __tablename__ = "timeseries_forecasts"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    forecast_id = Column(String(255), unique=True, nullable=False)
    model_id = Column(String(255), ForeignKey('timeseries_models.model_id'), nullable=False)
    
    # Forecast details
    forecast_start = Column(DateTime, nullable=False)
    forecast_end = Column(DateTime, nullable=False)
    horizon_periods = Column(Integer, nullable=False)
    
    # Results
    forecast_values = Column(JSON)  # List of forecasted values
    confidence_intervals = Column(JSON)  # Upper and lower bounds
    forecast_dates = Column(JSON)  # Corresponding timestamps
    
    # Metrics
    point_forecasts = Column(JSON)
    interval_coverage = Column(Float)  # Actual coverage of confidence intervals
    forecast_accuracy = Column(JSON)  # MAE, MAPE, RMSE if actual values available
    
    # Metadata
    model_version = Column(String(100))
    parameters_used = Column(JSON)
    computation_time = Column(Float)
    
    # Status
    status = Column(String(50), default='generated')  # generated, validated, published
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    model = relationship("TimeSeriesModel", back_populates="forecasts")

class TimeSeriesAnomaly(Base):
    """Store detected anomalies"""
    __tablename__ = "timeseries_anomalies"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    anomaly_id = Column(String(255), unique=True, nullable=False)
    model_id = Column(String(255), ForeignKey('timeseries_models.model_id'), nullable=False)
    
    # Anomaly details
    timestamp = Column(DateTime, nullable=False)
    value = Column(Float, nullable=False)
    expected_value = Column(Float)
    deviation = Column(Float)
    anomaly_score = Column(Float, nullable=False)
    
    # Detection method
    detection_method = Column(String(100), nullable=False)
    threshold_used = Column(Float)
    
    # Classification
    anomaly_type = Column(String(100))  # 'point', 'contextual', 'collective'
    severity = Column(String(50))  # 'low', 'medium', 'high', 'critical'
    
    # Context
    window_start = Column(DateTime)
    window_end = Column(DateTime)
    related_features = Column(JSON)
    
    # Validation
    is_confirmed = Column(Boolean, default=False)
    confirmed_by = Column(String(255))
    confirmed_at = Column(DateTime)
    false_positive = Column(Boolean, default=False)
    
    # Actions taken
    alert_sent = Column(Boolean, default=False)
    action_required = Column(Boolean, default=True)
    resolution_notes = Column(Text)
    
    # Timestamps
    detected_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    model = relationship("TimeSeriesModel", back_populates="anomalies")

class TimeSeriesInsight(Base):
    """Store insights and patterns"""
    __tablename__ = "timeseries_insights"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    insight_id = Column(String(255), unique=True, nullable=False)
    model_id = Column(String(255), nullable=False)
    
    # Insight details
    insight_type = Column(String(100), nullable=False)  # 'trend', 'seasonality', 'pattern', 'correlation'
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=False)
    
    # Analysis results
    statistical_significance = Column(Float)
    confidence_level = Column(Float)
    supporting_data = Column(JSON)
    
    # Impact and importance
    importance_score = Column(Float)  # 0-1 scale
    business_impact = Column(String(100))  # 'low', 'medium', 'high', 'critical'
    recommended_actions = Column(JSON)  # List of suggested actions
    
    # Temporal context
    period_start = Column(DateTime)
    period_end = Column(DateTime)
    pattern_frequency = Column(String(100))  # How often this pattern occurs
    
    # Validation
    is_actionable = Column(Boolean, default=True)
    has_been_acted_upon = Column(Boolean, default=False)
    
    # Timestamps
    discovered_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)  # When this insight becomes stale

class TimeSeriesAnalyticsService:
    """Service for time series forecasting and analysis"""
    
    def __init__(self, db_session: Session = None,
                 storage_path: str = "/tmp/raia_timeseries"):
        self.db = db_session
        self.storage_path = storage_path
        
        # Ensure directories exist
        os.makedirs(storage_path, exist_ok=True)
        os.makedirs(f"{storage_path}/models", exist_ok=True)
        os.makedirs(f"{storage_path}/data", exist_ok=True)
        os.makedirs(f"{storage_path}/forecasts", exist_ok=True)
        
        # Initialize scalers and models cache
        self.model_cache = {}
        self.scaler_cache = {}

    async def create_timeseries_model(self, config: TimeSeriesConfig, 
                                    forecast_config: ForecastConfig,
                                    created_by: str = None,
                                    organization_id: str = None) -> Dict[str, Any]:
        """Create a new time series model"""
        
        model_id = f"ts_model_{uuid.uuid4().hex[:8]}"
        
        try:
            # Validate configuration
            validation_result = self._validate_config(config, forecast_config)
            if not validation_result['valid']:
                return {
                    'success': False,
                    'error': f"Invalid configuration: {validation_result['error']}"
                }
            
            # Create model record
            model_record = TimeSeriesModel(
                model_id=model_id,
                name=config.name,
                description=config.description,
                data_source_config=config.data_source,
                target_variable=config.target_variable,
                frequency=config.frequency,
                model_type=forecast_config.model_type,
                model_config=asdict(forecast_config),
                forecast_horizon=forecast_config.horizon,
                status='created',
                created_by=created_by,
                organization_id=organization_id
            )
            
            if self.db:
                self.db.add(model_record)
                self.db.commit()
            
            logger.info(f"Created time series model {model_id}")
            
            return {
                'success': True,
                'model_id': model_id,
                'message': f'Time series model "{config.name}" created successfully'
            }
            
        except Exception as e:
            logger.error(f"Error creating time series model: {str(e)}")
            return {
                'success': False,
                'error': f'Failed to create model: {str(e)}'
            }

    async def load_and_prepare_data(self, model_id: str) -> Dict[str, Any]:
        """Load and prepare time series data for analysis"""
        
        if not self.db:
            return {'success': False, 'error': 'Database not available'}
        
        model = self.db.query(TimeSeriesModel).filter(
            TimeSeriesModel.model_id == model_id
        ).first()
        
        if not model:
            return {'success': False, 'error': 'Model not found'}
        
        try:
            # Load data based on data source configuration
            data_source = model.data_source_config
            df = await self._load_data(data_source)
            
            if df is None or df.empty:
                return {'success': False, 'error': 'Failed to load data'}
            
            # Prepare time series data
            prepared_data = self._prepare_timeseries_data(
                df, 
                model.target_variable, 
                model.frequency
            )
            
            # Save prepared data
            data_path = os.path.join(
                self.storage_path, 
                'data', 
                f'{model_id}_data.parquet'
            )
            prepared_data.to_parquet(data_path)
            
            # Update model record
            model.data_file_path = data_path
            model.status = 'data_ready'
            
            if self.db:
                self.db.commit()
            
            # Generate initial insights
            insights = await self._generate_data_insights(model_id, prepared_data)
            
            return {
                'success': True,
                'data_shape': prepared_data.shape,
                'date_range': {
                    'start': prepared_data.index.min().isoformat(),
                    'end': prepared_data.index.max().isoformat()
                },
                'insights_generated': len(insights),
                'message': 'Data loaded and prepared successfully'
            }
            
        except Exception as e:
            logger.error(f"Error preparing data for model {model_id}: {str(e)}")
            
            if self.db:
                model.status = 'error'
                self.db.commit()
            
            return {
                'success': False,
                'error': f'Data preparation failed: {str(e)}'
            }

    async def train_model(self, model_id: str) -> Dict[str, Any]:
        """Train the time series forecasting model"""
        
        if not self.db:
            return {'success': False, 'error': 'Database not available'}
        
        model = self.db.query(TimeSeriesModel).filter(
            TimeSeriesModel.model_id == model_id
        ).first()
        
        if not model:
            return {'success': False, 'error': 'Model not found'}
        
        if not model.data_file_path or not os.path.exists(model.data_file_path):
            return {'success': False, 'error': 'Data not prepared. Run load_and_prepare_data first.'}
        
        try:
            # Update status
            model.status = 'training'
            model.training_progress = 0.0
            if self.db:
                self.db.commit()
            
            # Load prepared data
            df = pd.read_parquet(model.data_file_path)
            
            # Train model based on type
            model_config = model.model_config
            training_result = None
            
            if model.model_type == 'arima':
                training_result = await self._train_arima_model(df, model_config)
            elif model.model_type == 'prophet':
                training_result = await self._train_prophet_model(df, model_config)
            elif model.model_type == 'lstm':
                training_result = await self._train_lstm_model(df, model_config)
            elif model.model_type == 'ensemble':
                training_result = await self._train_ensemble_model(df, model_config)
            else:
                return {'success': False, 'error': f'Unsupported model type: {model.model_type}'}
            
            if not training_result['success']:
                model.status = 'error'
                if self.db:
                    self.db.commit()
                return training_result
            
            # Save trained model
            model_path = os.path.join(
                self.storage_path,
                'models',
                f'{model_id}_model.pkl'
            )
            
            # Store model and results
            import pickle
            with open(model_path, 'wb') as f:
                pickle.dump(training_result['model'], f)
            
            # Update model record
            model.model_file_path = model_path
            model.accuracy_metrics = training_result['metrics']
            model.validation_scores = training_result.get('validation_scores', {})
            model.seasonality_info = training_result.get('seasonality_info', {})
            model.trend_info = training_result.get('trend_info', {})
            model.status = 'ready'
            model.training_progress = 100.0
            model.last_trained = datetime.utcnow()
            
            # Calculate next retrain time
            retrain_freq = model_config.get('retrain_frequency', 'weekly')
            if retrain_freq == 'daily':
                model.next_retrain = datetime.utcnow() + timedelta(days=1)
            elif retrain_freq == 'weekly':
                model.next_retrain = datetime.utcnow() + timedelta(weeks=1)
            elif retrain_freq == 'monthly':
                model.next_retrain = datetime.utcnow() + timedelta(days=30)
            
            if self.db:
                self.db.commit()
            
            # Cache the model
            self.model_cache[model_id] = training_result['model']
            
            logger.info(f"Successfully trained model {model_id}")
            
            return {
                'success': True,
                'model_id': model_id,
                'training_time': training_result.get('training_time', 0),
                'accuracy_metrics': training_result['metrics'],
                'message': 'Model trained successfully'
            }
            
        except Exception as e:
            logger.error(f"Error training model {model_id}: {str(e)}")
            
            if self.db:
                model.status = 'error'
                model.training_progress = 0.0
                self.db.commit()
            
            return {
                'success': False,
                'error': f'Training failed: {str(e)}'
            }

    async def generate_forecast(self, model_id: str, 
                              custom_horizon: int = None) -> Dict[str, Any]:
        """Generate forecast for the specified model"""
        
        if not self.db:
            return {'success': False, 'error': 'Database not available'}
        
        model = self.db.query(TimeSeriesModel).filter(
            TimeSeriesModel.model_id == model_id
        ).first()
        
        if not model:
            return {'success': False, 'error': 'Model not found'}
        
        if model.status != 'ready':
            return {'success': False, 'error': f'Model not ready for forecasting. Status: {model.status}'}
        
        try:
            # Load trained model
            if model_id not in self.model_cache:
                if not model.model_file_path or not os.path.exists(model.model_file_path):
                    return {'success': False, 'error': 'Trained model file not found'}
                
                import pickle
                with open(model.model_file_path, 'rb') as f:
                    trained_model = pickle.load(f)
                self.model_cache[model_id] = trained_model
            
            trained_model = self.model_cache[model_id]
            
            # Load data
            df = pd.read_parquet(model.data_file_path)
            
            # Determine forecast horizon
            horizon = custom_horizon or model.forecast_horizon
            
            # Generate forecast based on model type
            forecast_result = None
            
            if model.model_type == 'arima':
                forecast_result = self._forecast_with_arima(trained_model, df, horizon)
            elif model.model_type == 'prophet':
                forecast_result = self._forecast_with_prophet(trained_model, df, horizon, model.frequency)
            elif model.model_type == 'lstm':
                forecast_result = self._forecast_with_lstm(trained_model, df, horizon)
            elif model.model_type == 'ensemble':
                forecast_result = self._forecast_with_ensemble(trained_model, df, horizon)
            
            if not forecast_result:
                return {'success': False, 'error': 'Forecast generation failed'}
            
            # Create forecast record
            forecast_id = f"forecast_{uuid.uuid4().hex[:8]}"
            forecast_record = TimeSeriesForecast(
                forecast_id=forecast_id,
                model_id=model_id,
                forecast_start=forecast_result['dates'][0],
                forecast_end=forecast_result['dates'][-1],
                horizon_periods=horizon,
                forecast_values=forecast_result['values'].tolist(),
                confidence_intervals=forecast_result.get('confidence_intervals'),
                forecast_dates=[d.isoformat() for d in forecast_result['dates']],
                point_forecasts=forecast_result.get('point_forecasts'),
                model_version=model.model_config.get('version', '1.0'),
                parameters_used=model.model_config,
                computation_time=forecast_result.get('computation_time', 0)
            )
            
            if self.db:
                self.db.add(forecast_record)
                self.db.commit()
            
            return {
                'success': True,
                'forecast_id': forecast_id,
                'forecast_data': {
                    'dates': [d.isoformat() for d in forecast_result['dates']],
                    'values': forecast_result['values'].tolist(),
                    'confidence_intervals': forecast_result.get('confidence_intervals'),
                    'point_forecasts': forecast_result.get('point_forecasts')
                },
                'horizon': horizon,
                'message': f'Forecast generated for {horizon} periods'
            }
            
        except Exception as e:
            logger.error(f"Error generating forecast for model {model_id}: {str(e)}")
            return {
                'success': False,
                'error': f'Forecast generation failed: {str(e)}'
            }

    async def detect_anomalies(self, model_id: str, 
                              anomaly_config: AnomalyDetectionConfig = None) -> Dict[str, Any]:
        """Detect anomalies in the time series data"""
        
        if not self.db:
            return {'success': False, 'error': 'Database not available'}
        
        model = self.db.query(TimeSeriesModel).filter(
            TimeSeriesModel.model_id == model_id
        ).first()
        
        if not model:
            return {'success': False, 'error': 'Model not found'}
        
        try:
            # Load data
            df = pd.read_parquet(model.data_file_path)
            
            # Use provided config or default
            if anomaly_config is None:
                anomaly_config = AnomalyDetectionConfig()
            
            detected_anomalies = []
            
            # Apply different detection methods
            methods = anomaly_config.methods or ['isolation_forest', 'statistical']
            
            for method in methods:
                if method == 'isolation_forest' and SKLEARN_AVAILABLE:
                    anomalies = self._detect_anomalies_isolation_forest(df, anomaly_config)
                elif method == 'statistical':
                    anomalies = self._detect_anomalies_statistical(df, anomaly_config)
                elif method == 'one_class_svm' and SKLEARN_AVAILABLE:
                    anomalies = self._detect_anomalies_one_class_svm(df, anomaly_config)
                else:
                    continue
                
                # Store anomalies in database
                for anomaly in anomalies:
                    anomaly_id = f"anomaly_{uuid.uuid4().hex[:8]}"
                    
                    anomaly_record = TimeSeriesAnomaly(
                        anomaly_id=anomaly_id,
                        model_id=model_id,
                        timestamp=anomaly['timestamp'],
                        value=anomaly['value'],
                        expected_value=anomaly.get('expected_value'),
                        deviation=anomaly.get('deviation'),
                        anomaly_score=anomaly['score'],
                        detection_method=method,
                        threshold_used=anomaly.get('threshold'),
                        anomaly_type=anomaly.get('type', 'point'),
                        severity=self._classify_anomaly_severity(anomaly['score'])
                    )
                    
                    if self.db:
                        self.db.add(anomaly_record)
                    
                    detected_anomalies.append({
                        'anomaly_id': anomaly_id,
                        'timestamp': anomaly['timestamp'].isoformat(),
                        'value': anomaly['value'],
                        'score': anomaly['score'],
                        'method': method,
                        'severity': anomaly_record.severity
                    })
            
            if self.db:
                self.db.commit()
            
            # Update model with anomaly configuration
            model.anomaly_config = asdict(anomaly_config)
            if self.db:
                self.db.commit()
            
            return {
                'success': True,
                'anomalies_detected': len(detected_anomalies),
                'anomalies': detected_anomalies,
                'detection_methods': methods,
                'message': f'Detected {len(detected_anomalies)} anomalies'
            }
            
        except Exception as e:
            logger.error(f"Error detecting anomalies for model {model_id}: {str(e)}")
            return {
                'success': False,
                'error': f'Anomaly detection failed: {str(e)}'
            }

    async def get_model_insights(self, model_id: str) -> Dict[str, Any]:
        """Get insights and analysis for a time series model"""
        
        if not self.db:
            return {'success': False, 'error': 'Database not available'}
        
        model = self.db.query(TimeSeriesModel).filter(
            TimeSeriesModel.model_id == model_id
        ).first()
        
        if not model:
            return {'success': False, 'error': 'Model not found'}
        
        try:
            # Load data
            df = pd.read_parquet(model.data_file_path)
            
            insights = []
            
            # Trend analysis
            trend_insights = self._analyze_trends(df, model.target_variable)
            insights.extend(trend_insights)
            
            # Seasonality analysis
            if model.seasonality_info:
                seasonality_insights = self._analyze_seasonality(df, model.seasonality_info)
                insights.extend(seasonality_insights)
            
            # Recent anomalies
            recent_anomalies = self.db.query(TimeSeriesAnomaly).filter(
                TimeSeriesAnomaly.model_id == model_id,
                TimeSeriesAnomaly.detected_at >= datetime.utcnow() - timedelta(days=30)
            ).all() if self.db else []
            
            if recent_anomalies:
                anomaly_insights = self._analyze_recent_anomalies(recent_anomalies)
                insights.extend(anomaly_insights)
            
            # Forecast accuracy if available
            recent_forecasts = self.db.query(TimeSeriesForecast).filter(
                TimeSeriesForecast.model_id == model_id
            ).order_by(TimeSeriesForecast.created_at.desc()).limit(5).all() if self.db else []
            
            if recent_forecasts:
                accuracy_insights = self._analyze_forecast_accuracy(recent_forecasts)
                insights.extend(accuracy_insights)
            
            # Store insights in database
            for insight_data in insights:
                insight_id = f"insight_{uuid.uuid4().hex[:8]}"
                
                insight_record = TimeSeriesInsight(
                    insight_id=insight_id,
                    model_id=model_id,
                    insight_type=insight_data['type'],
                    title=insight_data['title'],
                    description=insight_data['description'],
                    statistical_significance=insight_data.get('significance'),
                    confidence_level=insight_data.get('confidence'),
                    supporting_data=insight_data.get('data'),
                    importance_score=insight_data.get('importance', 0.5),
                    business_impact=insight_data.get('impact', 'medium'),
                    recommended_actions=insight_data.get('actions', [])
                )
                
                if self.db:
                    self.db.add(insight_record)
            
            if self.db:
                self.db.commit()
            
            return {
                'success': True,
                'insights': insights,
                'total_insights': len(insights),
                'message': f'Generated {len(insights)} insights'
            }
            
        except Exception as e:
            logger.error(f"Error generating insights for model {model_id}: {str(e)}")
            return {
                'success': False,
                'error': f'Insight generation failed: {str(e)}'
            }

    # Private helper methods
    async def _load_data(self, data_source: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Load data from various sources"""
        
        source_type = data_source.get('type', 'csv')
        
        try:
            if source_type == 'csv':
                file_path = data_source.get('file_path')
                if file_path and os.path.exists(file_path):
                    return pd.read_csv(file_path)
            elif source_type == 'database':
                # Implement database loading
                pass
            elif source_type == 'api':
                # Implement API data loading
                pass
        
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
        
        return None

    def _prepare_timeseries_data(self, df: pd.DataFrame, target_var: str, frequency: str) -> pd.DataFrame:
        """Prepare time series data for analysis"""
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            # Try to find datetime column
            datetime_cols = df.select_dtypes(include=['datetime64']).columns
            if len(datetime_cols) > 0:
                df = df.set_index(datetime_cols[0])
            else:
                # Assume first column is datetime
                df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
                df = df.set_index(df.columns[0])
        
        # Ensure target variable exists
        if target_var not in df.columns:
            raise ValueError(f"Target variable '{target_var}' not found in data")
        
        # Sort by index
        df = df.sort_index()
        
        # Handle missing values
        df[target_var] = df[target_var].interpolate(method='linear')
        
        # Resample to specified frequency if needed
        if frequency:
            df = df.resample(frequency).mean()
        
        # Remove any remaining NaN values
        df = df.dropna()
        
        return df

    async def _train_arima_model(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Train ARIMA model"""
        
        if not STATSMODELS_AVAILABLE:
            return {'success': False, 'error': 'statsmodels not available'}
        
        try:
            target_col = list(df.columns)[0]  # Assume first column is target
            data = df[target_col]
            
            # Auto-determine ARIMA parameters if not provided
            hyperparams = config.get('hyperparameters', {})
            p = hyperparams.get('p', 1)
            d = hyperparams.get('d', 1)
            q = hyperparams.get('q', 1)
            
            # Fit model
            model = ARIMA(data, order=(p, d, q))
            fitted_model = model.fit()
            
            # Calculate metrics
            residuals = fitted_model.resid
            metrics = {
                'aic': fitted_model.aic,
                'bic': fitted_model.bic,
                'rmse': np.sqrt(np.mean(residuals**2)),
                'mae': np.mean(np.abs(residuals))
            }
            
            return {
                'success': True,
                'model': fitted_model,
                'metrics': metrics,
                'training_time': 0  # Would track actual training time
            }
            
        except Exception as e:
            logger.error(f"ARIMA training error: {str(e)}")
            return {'success': False, 'error': str(e)}

    async def _train_prophet_model(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Train Facebook Prophet model"""
        
        if not PROPHET_AVAILABLE:
            return {'success': False, 'error': 'Prophet not available'}
        
        try:
            # Prepare data for Prophet
            target_col = list(df.columns)[0]
            prophet_df = pd.DataFrame({
                'ds': df.index,
                'y': df[target_col].values
            })
            
            # Initialize Prophet with hyperparameters
            hyperparams = config.get('hyperparameters', {})
            model = Prophet(
                growth=hyperparams.get('growth', 'linear'),
                seasonality_mode=hyperparams.get('seasonality_mode', 'additive'),
                changepoint_prior_scale=hyperparams.get('changepoint_prior_scale', 0.05),
                seasonality_prior_scale=hyperparams.get('seasonality_prior_scale', 10.0)
            )
            
            # Add custom seasonalities if specified
            if 'yearly_seasonality' in hyperparams:
                model.add_seasonality(name='yearly', period=365.25, fourier_order=hyperparams['yearly_seasonality'])
            
            # Fit model
            model.fit(prophet_df)
            
            # Generate in-sample predictions for validation
            forecast = model.predict(prophet_df)
            
            # Calculate metrics
            actual = prophet_df['y']
            predicted = forecast['yhat']
            
            metrics = {
                'mae': np.mean(np.abs(actual - predicted)),
                'mape': np.mean(np.abs((actual - predicted) / actual)) * 100,
                'rmse': np.sqrt(np.mean((actual - predicted)**2)),
                'r2': 1 - (np.sum((actual - predicted)**2) / np.sum((actual - np.mean(actual))**2))
            }
            
            # Extract seasonality info
            seasonality_info = {}
            if 'yearly' in forecast.columns:
                seasonality_info['yearly'] = np.std(forecast['yearly']) / np.std(actual)
            if 'weekly' in forecast.columns:
                seasonality_info['weekly'] = np.std(forecast['weekly']) / np.std(actual)
            if 'daily' in forecast.columns:
                seasonality_info['daily'] = np.std(forecast['daily']) / np.std(actual)
            
            # Extract trend info
            trend_info = {
                'trend_strength': np.std(forecast['trend']) / np.std(actual),
                'changepoints': [cp.isoformat() for cp in model.changepoints]
            }
            
            return {
                'success': True,
                'model': model,
                'metrics': metrics,
                'seasonality_info': seasonality_info,
                'trend_info': trend_info,
                'training_time': 0
            }
            
        except Exception as e:
            logger.error(f"Prophet training error: {str(e)}")
            return {'success': False, 'error': str(e)}

    async def _train_lstm_model(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Train LSTM model"""
        
        if not TORCH_AVAILABLE:
            return {'success': False, 'error': 'PyTorch not available'}
        
        try:
            # This is a simplified LSTM implementation
            # In practice, you'd want a more sophisticated architecture
            
            target_col = list(df.columns)[0]
            data = df[target_col].values.reshape(-1, 1)
            
            # Scale data
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data)
            
            # Create sequences
            sequence_length = config.get('hyperparameters', {}).get('sequence_length', 60)
            X, y = self._create_sequences(scaled_data, sequence_length)
            
            # Split data
            split_ratio = 0.8
            split_idx = int(len(X) * split_ratio)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Convert to tensors
            X_train = torch.FloatTensor(X_train)
            y_train = torch.FloatTensor(y_train)
            X_test = torch.FloatTensor(X_test)
            y_test = torch.FloatTensor(y_test)
            
            # Simple LSTM model
            class SimpleLSTM(nn.Module):
                def __init__(self, input_size=1, hidden_size=50, num_layers=1, output_size=1):
                    super(SimpleLSTM, self).__init__()
                    self.hidden_size = hidden_size
                    self.num_layers = num_layers
                    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                    self.fc = nn.Linear(hidden_size, output_size)
                
                def forward(self, x):
                    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
                    c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
                    out, _ = self.lstm(x, (h0, c0))
                    out = self.fc(out[:, -1, :])
                    return out
            
            # Initialize model
            hyperparams = config.get('hyperparameters', {})
            model = SimpleLSTM(
                input_size=1,
                hidden_size=hyperparams.get('hidden_size', 50),
                num_layers=hyperparams.get('num_layers', 1),
                output_size=1
            )
            
            # Training (simplified)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams.get('learning_rate', 0.001))
            
            epochs = hyperparams.get('epochs', 100)
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                outputs = model(X_train)
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()
            
            # Evaluate
            model.eval()
            with torch.no_grad():
                train_pred = model(X_train)
                test_pred = model(X_test)
            
            # Calculate metrics
            train_loss = criterion(train_pred, y_train).item()
            test_loss = criterion(test_pred, y_test).item()
            
            metrics = {
                'train_loss': train_loss,
                'test_loss': test_loss,
                'rmse': np.sqrt(test_loss)
            }
            
            # Package model with scaler
            model_package = {
                'model': model,
                'scaler': scaler,
                'sequence_length': sequence_length
            }
            
            return {
                'success': True,
                'model': model_package,
                'metrics': metrics,
                'training_time': 0
            }
            
        except Exception as e:
            logger.error(f"LSTM training error: {str(e)}")
            return {'success': False, 'error': str(e)}

    async def _train_ensemble_model(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Train ensemble model combining multiple approaches"""
        
        try:
            models = {}
            all_metrics = {}
            
            # Train individual models
            if STATSMODELS_AVAILABLE:
                arima_result = await self._train_arima_model(df, config)
                if arima_result['success']:
                    models['arima'] = arima_result['model']
                    all_metrics['arima'] = arima_result['metrics']
            
            if PROPHET_AVAILABLE:
                prophet_result = await self._train_prophet_model(df, config)
                if prophet_result['success']:
                    models['prophet'] = prophet_result['model']
                    all_metrics['prophet'] = prophet_result['metrics']
            
            if not models:
                return {'success': False, 'error': 'No models could be trained'}
            
            # Simple ensemble (equal weights)
            ensemble_package = {
                'models': models,
                'weights': {model: 1.0/len(models) for model in models.keys()},
                'individual_metrics': all_metrics
            }
            
            # Calculate ensemble metrics (simplified)
            metrics = {
                'ensemble_models': list(models.keys()),
                'individual_performance': all_metrics
            }
            
            return {
                'success': True,
                'model': ensemble_package,
                'metrics': metrics,
                'training_time': 0
            }
            
        except Exception as e:
            logger.error(f"Ensemble training error: {str(e)}")
            return {'success': False, 'error': str(e)}

    def _forecast_with_prophet(self, model, df: pd.DataFrame, horizon: int, frequency: str) -> Dict[str, Any]:
        """Generate forecast using Prophet model"""
        
        try:
            # Create future dataframe
            future = model.make_future_dataframe(periods=horizon, freq=frequency)
            forecast = model.predict(future)
            
            # Extract forecast portion
            forecast_data = forecast.tail(horizon)
            
            return {
                'dates': forecast_data['ds'].tolist(),
                'values': forecast_data['yhat'].values,
                'confidence_intervals': {
                    'lower': forecast_data['yhat_lower'].tolist(),
                    'upper': forecast_data['yhat_upper'].tolist()
                },
                'computation_time': 0
            }
            
        except Exception as e:
            logger.error(f"Prophet forecasting error: {str(e)}")
            return None

    def _detect_anomalies_statistical(self, df: pd.DataFrame, config: AnomalyDetectionConfig) -> List[Dict[str, Any]]:
        """Detect anomalies using statistical methods"""
        
        target_col = list(df.columns)[0]
        data = df[target_col]
        
        # Rolling statistics
        window = config.window_size
        rolling_mean = data.rolling(window=window).mean()
        rolling_std = data.rolling(window=window).std()
        
        # Z-score based detection
        z_scores = np.abs((data - rolling_mean) / rolling_std)
        threshold = config.threshold_std
        
        anomalies = []
        for idx, (timestamp, value) in enumerate(data.items()):
            if idx >= window and z_scores.iloc[idx] > threshold:
                anomalies.append({
                    'timestamp': timestamp,
                    'value': value,
                    'expected_value': rolling_mean.iloc[idx],
                    'deviation': value - rolling_mean.iloc[idx],
                    'score': z_scores.iloc[idx] / threshold,  # Normalize score
                    'threshold': threshold,
                    'type': 'point'
                })
        
        return anomalies

    def _detect_anomalies_isolation_forest(self, df: pd.DataFrame, config: AnomalyDetectionConfig) -> List[Dict[str, Any]]:
        """Detect anomalies using Isolation Forest"""
        
        if not SKLEARN_AVAILABLE:
            return []
        
        target_col = list(df.columns)[0]
        data = df[target_col].values.reshape(-1, 1)
        
        # Train Isolation Forest
        clf = IsolationForest(contamination=config.contamination, random_state=42)
        predictions = clf.fit_predict(data)
        scores = clf.decision_function(data)
        
        anomalies = []
        for idx, (timestamp, value) in enumerate(df[target_col].items()):
            if predictions[idx] == -1:  # Anomaly
                anomalies.append({
                    'timestamp': timestamp,
                    'value': value,
                    'score': abs(scores[idx]),  # Convert to positive score
                    'threshold': config.contamination,
                    'type': 'point'
                })
        
        return anomalies

    def _create_sequences(self, data: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:(i + sequence_length)])
            y.append(data[i + sequence_length])
        
        return np.array(X), np.array(y)

    def _classify_anomaly_severity(self, score: float) -> str:
        """Classify anomaly severity based on score"""
        
        if score >= 0.9:
            return 'critical'
        elif score >= 0.7:
            return 'high'
        elif score >= 0.5:
            return 'medium'
        else:
            return 'low'

    def _analyze_trends(self, df: pd.DataFrame, target_var: str) -> List[Dict[str, Any]]:
        """Analyze trends in the time series"""
        
        insights = []
        data = df[target_var]
        
        # Overall trend
        x = np.arange(len(data))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
        
        if p_value < 0.05:  # Statistically significant
            trend_direction = 'increasing' if slope > 0 else 'decreasing'
            insights.append({
                'type': 'trend',
                'title': f'Significant {trend_direction.capitalize()} Trend Detected',
                'description': f'The data shows a statistically significant {trend_direction} trend with R² = {r_value**2:.3f}',
                'significance': p_value,
                'confidence': 1 - p_value,
                'data': {
                    'slope': slope,
                    'r_squared': r_value**2,
                    'trend_direction': trend_direction
                },
                'importance': 0.8 if abs(r_value) > 0.5 else 0.5,
                'impact': 'high' if abs(slope) > data.std() else 'medium',
                'actions': [
                    f'Monitor {trend_direction} trend for business impact',
                    'Consider seasonal adjustments in forecasting',
                    'Investigate underlying causes of trend'
                ]
            })
        
        return insights

    def _analyze_seasonality(self, df: pd.DataFrame, seasonality_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze seasonality patterns"""
        
        insights = []
        
        for period, strength in seasonality_info.items():
            if strength > 0.3:  # Significant seasonality
                insights.append({
                    'type': 'seasonality',
                    'title': f'Strong {period.capitalize()} Seasonality Detected',
                    'description': f'{period.capitalize()} seasonal patterns contribute {strength*100:.1f}% to data variation',
                    'confidence': min(strength * 2, 1.0),
                    'data': {
                        'period': period,
                        'strength': strength
                    },
                    'importance': 0.7 if strength > 0.5 else 0.5,
                    'impact': 'high' if strength > 0.5 else 'medium',
                    'actions': [
                        f'Account for {period} patterns in forecasting',
                        f'Adjust business operations for {period} variations',
                        'Monitor for changes in seasonal patterns'
                    ]
                })
        
        return insights

    async def _generate_data_insights(self, model_id: str, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate initial insights from data analysis"""
        
        insights = []
        target_col = list(df.columns)[0]
        
        # Basic statistics insight
        insights.append({
            'type': 'statistics',
            'title': 'Data Summary Statistics',
            'description': f'Dataset contains {len(df)} observations with mean value {df[target_col].mean():.2f}',
            'confidence': 1.0,
            'data': {
                'count': len(df),
                'mean': df[target_col].mean(),
                'std': df[target_col].std(),
                'min': df[target_col].min(),
                'max': df[target_col].max()
            },
            'importance': 0.3,
            'impact': 'low',
            'actions': []
        })
        
        return insights

    def _validate_config(self, ts_config: TimeSeriesConfig, forecast_config: ForecastConfig) -> Dict[str, Any]:
        """Validate configuration parameters"""
        
        if not ts_config.name:
            return {'valid': False, 'error': 'Model name is required'}
        
        if not ts_config.target_variable:
            return {'valid': False, 'error': 'Target variable is required'}
        
        if not ts_config.frequency:
            return {'valid': False, 'error': 'Frequency is required'}
        
        if forecast_config.horizon <= 0:
            return {'valid': False, 'error': 'Forecast horizon must be positive'}
        
        valid_models = ['arima', 'prophet', 'lstm', 'transformer', 'ensemble']
        if forecast_config.model_type not in valid_models:
            return {'valid': False, 'error': f'Model type must be one of {valid_models}'}
        
        return {'valid': True}

# Factory function
def create_timeseries_analytics_service(db_session: Session = None,
                                       storage_path: str = "/tmp/raia_timeseries") -> TimeSeriesAnalyticsService:
    """Create and return a TimeSeriesAnalyticsService instance"""
    return TimeSeriesAnalyticsService(db_session, storage_path)