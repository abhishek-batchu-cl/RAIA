# Automated Model Retraining Service with Drift Detection
import os
import json
import uuid
import asyncio
import logging
import schedule
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass, asdict
from sqlalchemy.orm import Session
from sqlalchemy import Column, String, DateTime, Text, Boolean, JSON, UUID as PG_UUID, Integer, Float, ForeignKey, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import threading
import time
from concurrent.futures import ThreadPoolExecutor

# ML and data processing
import numpy as np
import pandas as pd
import pickle
import joblib

# Data drift detection
try:
    from alibi_detect import ChiSquareDrift, KSDrift, MMDDrift, TabularDrift
    from alibi_detect.saving import save_detector, load_detector
    ALIBI_AVAILABLE = True
except ImportError:
    ALIBI_AVAILABLE = False

# Data quality and validation
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Model training
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR

# Email notifications
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

Base = declarative_base()
logger = logging.getLogger(__name__)

@dataclass
class RetrainingConfig:
    """Configuration for automated retraining"""
    model_id: str
    model_name: str
    schedule_type: str  # 'time_based', 'performance_based', 'drift_based', 'hybrid'
    schedule_params: Dict[str, Any]
    data_source_config: Dict[str, Any]
    training_config: Dict[str, Any]
    validation_config: Dict[str, Any]
    notification_config: Dict[str, Any] = None
    enabled: bool = True

@dataclass
class DriftDetectionConfig:
    """Configuration for drift detection"""
    model_id: str
    reference_data_path: str
    drift_types: List[str]  # ['feature_drift', 'target_drift', 'prediction_drift']
    detection_method: str  # 'statistical', 'ml_based', 'ensemble'
    thresholds: Dict[str, float]
    monitoring_window: int  # hours
    min_samples_for_detection: int = 100

@dataclass
class RetrainingJob:
    """Retraining job specification"""
    job_id: str
    model_id: str
    trigger_type: str  # 'scheduled', 'drift_detected', 'performance_drop', 'manual'
    trigger_metadata: Dict[str, Any]
    data_config: Dict[str, Any]
    training_config: Dict[str, Any]
    priority: int = 1  # 1-5, higher number = higher priority
    max_duration_hours: int = 24
    created_at: datetime = None

class RetrainingSchedule(Base):
    """Store retraining schedules"""
    __tablename__ = "retraining_schedules"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    schedule_id = Column(String(255), unique=True, nullable=False)
    model_id = Column(String(255), nullable=False, index=True)
    model_name = Column(String(500), nullable=False)
    
    # Schedule configuration
    schedule_type = Column(String(50), nullable=False)  # time_based, performance_based, drift_based, hybrid
    schedule_params = Column(JSON)
    
    # Status
    status = Column(String(50), default='active')  # active, paused, disabled
    enabled = Column(Boolean, default=True)
    
    # Configuration
    data_source_config = Column(JSON)
    training_config = Column(JSON)
    validation_config = Column(JSON)
    notification_config = Column(JSON)
    
    # Tracking
    last_execution = Column(DateTime)
    next_execution = Column(DateTime)
    execution_count = Column(Integer, default=0)
    success_count = Column(Integer, default=0)
    failure_count = Column(Integer, default=0)
    
    # Metadata
    created_by = Column(String(255))
    organization_id = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    jobs = relationship("RetrainingJobRecord", back_populates="schedule")

class RetrainingJobRecord(Base):
    """Store retraining job records"""
    __tablename__ = "retraining_jobs"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(String(255), unique=True, nullable=False)
    schedule_id = Column(String(255), ForeignKey('retraining_schedules.schedule_id'))
    model_id = Column(String(255), nullable=False, index=True)
    
    # Job details
    trigger_type = Column(String(50), nullable=False)
    trigger_metadata = Column(JSON)
    status = Column(String(50), default='queued')  # queued, running, completed, failed, cancelled
    priority = Column(Integer, default=1)
    
    # Configuration
    data_config = Column(JSON)
    training_config = Column(JSON)
    
    # Execution details
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    duration_minutes = Column(Float)
    progress_percentage = Column(Float, default=0.0)
    
    # Results
    old_model_metrics = Column(JSON)
    new_model_metrics = Column(JSON)
    performance_improvement = Column(JSON)
    model_artifact_path = Column(String(1000))
    training_logs_path = Column(String(1000))
    
    # Error handling
    error_message = Column(Text)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    
    # Resource usage
    cpu_usage = Column(JSON)
    memory_usage = Column(JSON)
    training_samples = Column(Integer)
    validation_samples = Column(Integer)
    
    # Relationships
    schedule = relationship("RetrainingSchedule", back_populates="jobs")

class DriftDetection(Base):
    """Store drift detection results"""
    __tablename__ = "drift_detections"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    detection_id = Column(String(255), unique=True, nullable=False)
    model_id = Column(String(255), nullable=False, index=True)
    
    # Detection details
    drift_type = Column(String(50))  # feature_drift, target_drift, prediction_drift
    detection_method = Column(String(100))
    drift_score = Column(Float)
    p_value = Column(Float)
    threshold = Column(Float)
    is_drift_detected = Column(Boolean, default=False)
    
    # Affected features/columns
    features_analyzed = Column(JSON)
    features_with_drift = Column(JSON)
    drift_scores_by_feature = Column(JSON)
    
    # Data details
    reference_period = Column(JSON)
    current_period = Column(JSON)
    samples_analyzed = Column(Integer)
    
    # Actions taken
    retraining_triggered = Column(Boolean, default=False)
    alert_sent = Column(Boolean, default=False)
    
    # Metadata
    detection_timestamp = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON)

class AutomatedRetrainingService:
    """Service for automated model retraining with drift detection"""
    
    def __init__(self, db_session: Session = None,
                 storage_path: str = "/tmp/raia_retraining",
                 max_concurrent_jobs: int = 3):
        self.db = db_session
        self.storage_path = storage_path
        self.max_concurrent_jobs = max_concurrent_jobs
        
        # Job management
        self.job_queue = asyncio.Queue()
        self.active_jobs = {}
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_jobs)
        
        # Drift detectors
        self.drift_detectors = {}
        
        # Scheduling
        self.scheduler_thread = None
        self.stop_scheduler = threading.Event()
        
        # Monitoring
        self.monitoring_data = {}
        
        # Ensure directories exist
        os.makedirs(storage_path, exist_ok=True)
        os.makedirs(f"{storage_path}/models", exist_ok=True)
        os.makedirs(f"{storage_path}/data", exist_ok=True)
        os.makedirs(f"{storage_path}/logs", exist_ok=True)
        os.makedirs(f"{storage_path}/detectors", exist_ok=True)

    async def create_retraining_schedule(self, config: RetrainingConfig,
                                       created_by: str = None,
                                       organization_id: str = None) -> Dict[str, Any]:
        """Create a new retraining schedule"""
        
        schedule_id = f"schedule_{uuid.uuid4().hex[:8]}"
        
        # Validate configuration
        validation_result = self._validate_retraining_config(config)
        if not validation_result['valid']:
            return {
                'success': False,
                'error': f"Invalid configuration: {validation_result['error']}"
            }
        
        # Create schedule record
        schedule_record = RetrainingSchedule(
            schedule_id=schedule_id,
            model_id=config.model_id,
            model_name=config.model_name,
            schedule_type=config.schedule_type,
            schedule_params=config.schedule_params,
            data_source_config=config.data_source_config,
            training_config=config.training_config,
            validation_config=config.validation_config,
            notification_config=config.notification_config,
            enabled=config.enabled,
            created_by=created_by,
            organization_id=organization_id
        )
        
        # Calculate next execution time
        next_execution = self._calculate_next_execution(config)
        schedule_record.next_execution = next_execution
        
        if self.db:
            self.db.add(schedule_record)
            self.db.commit()
        
        # Start scheduler if not running
        if not self.scheduler_thread or not self.scheduler_thread.is_alive():
            self._start_scheduler()
        
        logger.info(f"Created retraining schedule {schedule_id} for model {config.model_id}")
        
        return {
            'success': True,
            'schedule_id': schedule_id,
            'next_execution': next_execution.isoformat() if next_execution else None,
            'message': f'Retraining schedule created for {config.model_name}'
        }

    async def setup_drift_detection(self, config: DriftDetectionConfig) -> Dict[str, Any]:
        """Setup drift detection for a model"""
        
        if not ALIBI_AVAILABLE:
            return {
                'success': False,
                'error': 'Alibi Detect library not available'
            }
        
        try:
            # Load reference data
            if not os.path.exists(config.reference_data_path):
                return {
                    'success': False,
                    'error': f'Reference data file not found: {config.reference_data_path}'
                }
            
            reference_data = pd.read_csv(config.reference_data_path)
            X_ref = reference_data.values
            
            # Initialize drift detectors based on configuration
            detectors = {}
            
            if 'feature_drift' in config.drift_types:
                if config.detection_method == 'statistical':
                    if X_ref.dtype in ['object', 'category']:
                        detector = ChiSquareDrift(
                            X_ref, 
                            p_val=config.thresholds.get('p_value', 0.05)
                        )
                    else:
                        detector = KSDrift(
                            X_ref,
                            p_val=config.thresholds.get('p_value', 0.05)
                        )
                elif config.detection_method == 'ml_based':
                    detector = TabularDrift(
                        X_ref,
                        p_val=config.thresholds.get('p_value', 0.05)
                    )
                else:  # ensemble
                    # Combine multiple detectors
                    detector = TabularDrift(X_ref, p_val=config.thresholds.get('p_value', 0.05))
                
                detectors['feature_drift'] = detector
                
                # Save detector
                detector_path = os.path.join(
                    self.storage_path, 
                    'detectors', 
                    f'{config.model_id}_feature_drift'
                )
                save_detector(detector, detector_path)
            
            # Store detectors
            self.drift_detectors[config.model_id] = {
                'config': config,
                'detectors': detectors,
                'last_check': datetime.utcnow()
            }
            
            logger.info(f"Drift detection setup completed for model {config.model_id}")
            
            return {
                'success': True,
                'model_id': config.model_id,
                'drift_types': config.drift_types,
                'message': 'Drift detection setup completed'
            }
            
        except Exception as e:
            logger.error(f"Failed to setup drift detection: {str(e)}")
            return {
                'success': False,
                'error': f'Failed to setup drift detection: {str(e)}'
            }

    async def check_drift(self, model_id: str, new_data: pd.DataFrame) -> Dict[str, Any]:
        """Check for drift in new data"""
        
        if model_id not in self.drift_detectors:
            return {
                'success': False,
                'error': f'Drift detection not setup for model {model_id}'
            }
        
        detector_info = self.drift_detectors[model_id]
        config = detector_info['config']
        detectors = detector_info['detectors']
        
        results = {
            'model_id': model_id,
            'detection_timestamp': datetime.utcnow().isoformat(),
            'drift_detected': False,
            'results_by_type': {}
        }
        
        try:
            X_new = new_data.values
            
            if len(X_new) < config.min_samples_for_detection:
                return {
                    'success': False,
                    'error': f'Insufficient samples for drift detection. Need at least {config.min_samples_for_detection}'
                }
            
            # Check each drift type
            for drift_type, detector in detectors.items():
                drift_result = detector.predict(X_new)
                
                is_drift = drift_result['data']['is_drift']
                p_value = drift_result['data']['p_val']
                
                # Get feature-level drift scores if available
                feature_scores = {}
                if 'feature_score' in drift_result['data']:
                    feature_scores = dict(zip(
                        new_data.columns,
                        drift_result['data']['feature_score']
                    ))
                
                drift_info = {
                    'is_drift_detected': bool(is_drift),
                    'p_value': float(p_value),
                    'threshold': config.thresholds.get('p_value', 0.05),
                    'feature_scores': feature_scores,
                    'samples_analyzed': len(X_new)
                }
                
                results['results_by_type'][drift_type] = drift_info
                
                if is_drift:
                    results['drift_detected'] = True
                    
                    # Store drift detection result
                    await self._store_drift_detection(
                        model_id, drift_type, drift_info, config
                    )
                    
                    # Trigger retraining if configured
                    if config.thresholds.get('auto_retrain', False):
                        await self._trigger_drift_based_retraining(
                            model_id, drift_type, drift_info
                        )
            
            return {
                'success': True,
                **results
            }
            
        except Exception as e:
            logger.error(f"Error during drift detection: {str(e)}")
            return {
                'success': False,
                'error': f'Drift detection failed: {str(e)}'
            }

    async def trigger_manual_retraining(self, model_id: str,
                                      reason: str = "Manual trigger",
                                      priority: int = 1,
                                      custom_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Trigger manual retraining for a model"""
        
        # Find active schedule for the model
        schedule = None
        if self.db:
            schedule = self.db.query(RetrainingSchedule).filter(
                RetrainingSchedule.model_id == model_id,
                RetrainingSchedule.status == 'active'
            ).first()
        
        if not schedule:
            return {
                'success': False,
                'error': f'No active retraining schedule found for model {model_id}'
            }
        
        # Create retraining job
        job = RetrainingJob(
            job_id=f"job_{uuid.uuid4().hex[:8]}",
            model_id=model_id,
            trigger_type='manual',
            trigger_metadata={
                'reason': reason,
                'triggered_by': 'user',
                'timestamp': datetime.utcnow().isoformat()
            },
            data_config=custom_config or schedule.data_source_config,
            training_config=custom_config or schedule.training_config,
            priority=priority
        )
        
        # Add to job queue
        await self.job_queue.put(job)
        
        logger.info(f"Manual retraining triggered for model {model_id}")
        
        return {
            'success': True,
            'job_id': job.job_id,
            'model_id': model_id,
            'message': f'Retraining job queued for model {model_id}'
        }

    async def get_retraining_status(self, model_id: str = None) -> Dict[str, Any]:
        """Get retraining status for models"""
        
        if not self.db:
            return {'error': 'Database not available'}
        
        query = self.db.query(RetrainingSchedule)
        if model_id:
            query = query.filter(RetrainingSchedule.model_id == model_id)
        
        schedules = query.all()
        
        # Get recent jobs
        job_query = self.db.query(RetrainingJobRecord)
        if model_id:
            job_query = job_query.filter(RetrainingJobRecord.model_id == model_id)
        
        recent_jobs = job_query.order_by(
            RetrainingJobRecord.started_at.desc()
        ).limit(10).all()
        
        return {
            'schedules': [self._schedule_to_dict(s) for s in schedules],
            'recent_jobs': [self._job_to_dict(j) for j in recent_jobs],
            'active_jobs': list(self.active_jobs.keys()),
            'queue_size': self.job_queue.qsize()
        }

    async def get_drift_alerts(self, model_id: str = None,
                             hours_back: int = 24) -> List[Dict[str, Any]]:
        """Get recent drift detection alerts"""
        
        if not self.db:
            return []
        
        start_time = datetime.utcnow() - timedelta(hours=hours_back)
        
        query = self.db.query(DriftDetection).filter(
            DriftDetection.detection_timestamp >= start_time
        )
        
        if model_id:
            query = query.filter(DriftDetection.model_id == model_id)
        
        detections = query.order_by(
            DriftDetection.detection_timestamp.desc()
        ).all()
        
        alerts = []
        for detection in detections:
            if detection.is_drift_detected:
                alerts.append({
                    'detection_id': detection.detection_id,
                    'model_id': detection.model_id,
                    'drift_type': detection.drift_type,
                    'drift_score': detection.drift_score,
                    'p_value': detection.p_value,
                    'threshold': detection.threshold,
                    'features_with_drift': detection.features_with_drift,
                    'retraining_triggered': detection.retraining_triggered,
                    'detection_timestamp': detection.detection_timestamp.isoformat(),
                    'samples_analyzed': detection.samples_analyzed
                })
        
        return alerts

    # Private methods
    def _validate_retraining_config(self, config: RetrainingConfig) -> Dict[str, Any]:
        """Validate retraining configuration"""
        
        if not config.model_id or not config.model_name:
            return {'valid': False, 'error': 'Model ID and name are required'}
        
        if config.schedule_type not in ['time_based', 'performance_based', 'drift_based', 'hybrid']:
            return {'valid': False, 'error': 'Invalid schedule type'}
        
        if not config.schedule_params:
            return {'valid': False, 'error': 'Schedule parameters are required'}
        
        if not config.data_source_config:
            return {'valid': False, 'error': 'Data source configuration is required'}
        
        if not config.training_config:
            return {'valid': False, 'error': 'Training configuration is required'}
        
        return {'valid': True}

    def _calculate_next_execution(self, config: RetrainingConfig) -> Optional[datetime]:
        """Calculate next execution time based on schedule type"""
        
        now = datetime.utcnow()
        
        if config.schedule_type == 'time_based':
            interval_hours = config.schedule_params.get('interval_hours', 24)
            return now + timedelta(hours=interval_hours)
        
        elif config.schedule_type == 'performance_based':
            # Check performance periodically
            check_interval_hours = config.schedule_params.get('check_interval_hours', 6)
            return now + timedelta(hours=check_interval_hours)
        
        elif config.schedule_type == 'drift_based':
            # Check for drift periodically
            check_interval_hours = config.schedule_params.get('check_interval_hours', 1)
            return now + timedelta(hours=check_interval_hours)
        
        elif config.schedule_type == 'hybrid':
            # Use the shortest interval among all configured triggers
            intervals = []
            if 'time_interval_hours' in config.schedule_params:
                intervals.append(config.schedule_params['time_interval_hours'])
            if 'performance_check_interval_hours' in config.schedule_params:
                intervals.append(config.schedule_params['performance_check_interval_hours'])
            if 'drift_check_interval_hours' in config.schedule_params:
                intervals.append(config.schedule_params['drift_check_interval_hours'])
            
            if intervals:
                return now + timedelta(hours=min(intervals))
        
        return None

    def _start_scheduler(self):
        """Start the scheduler thread"""
        self.stop_scheduler.clear()
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        logger.info("Retraining scheduler started")

    def _scheduler_loop(self):
        """Main scheduler loop"""
        while not self.stop_scheduler.is_set():
            try:
                # Check schedules
                asyncio.run(self._check_schedules())
                
                # Process job queue
                asyncio.run(self._process_job_queue())
                
                # Sleep for a minute before next check
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {str(e)}")
                time.sleep(60)

    async def _check_schedules(self):
        """Check all active schedules"""
        
        if not self.db:
            return
        
        now = datetime.utcnow()
        
        # Get schedules that need execution
        schedules = self.db.query(RetrainingSchedule).filter(
            RetrainingSchedule.status == 'active',
            RetrainingSchedule.enabled == True,
            RetrainingSchedule.next_execution <= now
        ).all()
        
        for schedule in schedules:
            try:
                await self._execute_schedule(schedule)
            except Exception as e:
                logger.error(f"Error executing schedule {schedule.schedule_id}: {str(e)}")

    async def _execute_schedule(self, schedule: RetrainingSchedule):
        """Execute a retraining schedule"""
        
        # Determine trigger type and check conditions
        should_trigger = False
        trigger_reason = ""
        trigger_metadata = {}
        
        if schedule.schedule_type == 'time_based':
            should_trigger = True
            trigger_reason = "Scheduled time-based retraining"
            
        elif schedule.schedule_type == 'performance_based':
            # Check performance metrics
            performance_check = await self._check_performance_threshold(
                schedule.model_id,
                schedule.schedule_params
            )
            should_trigger = performance_check['should_retrain']
            trigger_reason = performance_check['reason']
            trigger_metadata = performance_check['metadata']
            
        elif schedule.schedule_type == 'drift_based':
            # Check for drift
            drift_check = await self._check_drift_threshold(
                schedule.model_id,
                schedule.schedule_params
            )
            should_trigger = drift_check['should_retrain']
            trigger_reason = drift_check['reason']
            trigger_metadata = drift_check['metadata']
            
        elif schedule.schedule_type == 'hybrid':
            # Check all conditions
            checks = []
            
            # Time-based check
            if 'time_interval_hours' in schedule.schedule_params:
                time_elapsed = (datetime.utcnow() - (schedule.last_execution or datetime.min)).total_seconds() / 3600
                if time_elapsed >= schedule.schedule_params['time_interval_hours']:
                    checks.append({
                        'type': 'time_based',
                        'should_retrain': True,
                        'reason': 'Time interval reached'
                    })
            
            # Performance-based check
            if 'performance_threshold' in schedule.schedule_params:
                perf_check = await self._check_performance_threshold(
                    schedule.model_id, schedule.schedule_params
                )
                checks.append({
                    'type': 'performance_based',
                    **perf_check
                })
            
            # Drift-based check
            if 'drift_threshold' in schedule.schedule_params:
                drift_check = await self._check_drift_threshold(
                    schedule.model_id, schedule.schedule_params
                )
                checks.append({
                    'type': 'drift_based',
                    **drift_check
                })
            
            # Trigger if any condition is met
            triggered_checks = [c for c in checks if c['should_retrain']]
            if triggered_checks:
                should_trigger = True
                trigger_reason = '; '.join([c['reason'] for c in triggered_checks])
                trigger_metadata = {
                    'triggered_by': [c['type'] for c in triggered_checks],
                    'all_checks': checks
                }
        
        if should_trigger:
            # Create and queue retraining job
            job = RetrainingJob(
                job_id=f"job_{uuid.uuid4().hex[:8]}",
                model_id=schedule.model_id,
                trigger_type='scheduled',
                trigger_metadata={
                    'schedule_id': schedule.schedule_id,
                    'schedule_type': schedule.schedule_type,
                    'reason': trigger_reason,
                    **trigger_metadata
                },
                data_config=schedule.data_source_config,
                training_config=schedule.training_config,
                priority=schedule.schedule_params.get('priority', 1)
            )
            
            await self.job_queue.put(job)
            
            # Update schedule
            schedule.last_execution = datetime.utcnow()
            schedule.execution_count += 1
        
        # Calculate next execution time
        config = RetrainingConfig(
            model_id=schedule.model_id,
            model_name=schedule.model_name,
            schedule_type=schedule.schedule_type,
            schedule_params=schedule.schedule_params,
            data_source_config=schedule.data_source_config,
            training_config=schedule.training_config,
            validation_config=schedule.validation_config
        )
        
        schedule.next_execution = self._calculate_next_execution(config)
        
        if self.db:
            self.db.commit()

    async def _process_job_queue(self):
        """Process jobs from the queue"""
        
        # Check if we can start more jobs
        while (len(self.active_jobs) < self.max_concurrent_jobs and
               not self.job_queue.empty()):
            
            try:
                job = await asyncio.wait_for(self.job_queue.get(), timeout=1.0)
                
                # Submit job to executor
                future = self.executor.submit(self._execute_retraining_job, job)
                self.active_jobs[job.job_id] = {
                    'job': job,
                    'future': future,
                    'started_at': datetime.utcnow()
                }
                
                logger.info(f"Started retraining job {job.job_id}")
                
            except asyncio.TimeoutError:
                break
            except Exception as e:
                logger.error(f"Error starting job: {str(e)}")
        
        # Check completed jobs
        completed_jobs = []
        for job_id, job_info in self.active_jobs.items():
            if job_info['future'].done():
                completed_jobs.append(job_id)
                
                # Get result
                try:
                    result = job_info['future'].result()
                    logger.info(f"Job {job_id} completed: {result.get('status', 'unknown')}")
                except Exception as e:
                    logger.error(f"Job {job_id} failed: {str(e)}")
        
        # Remove completed jobs
        for job_id in completed_jobs:
            del self.active_jobs[job_id]

    def _execute_retraining_job(self, job: RetrainingJob) -> Dict[str, Any]:
        """Execute a retraining job"""
        
        job_id = job.job_id
        start_time = datetime.utcnow()
        
        # Create job record in database
        job_record = None
        if self.db:
            job_record = RetrainingJobRecord(
                job_id=job_id,
                model_id=job.model_id,
                trigger_type=job.trigger_type,
                trigger_metadata=job.trigger_metadata,
                data_config=job.data_config,
                training_config=job.training_config,
                status='running',
                started_at=start_time,
                priority=job.priority
            )
            
            self.db.add(job_record)
            self.db.commit()
        
        try:
            # Load current model for comparison
            old_model_path = job.training_config.get('current_model_path')
            old_model = None
            old_metrics = {}
            
            if old_model_path and os.path.exists(old_model_path):
                old_model = joblib.load(old_model_path)
                
                # Get validation data to calculate old metrics
                validation_data = self._load_validation_data(job.data_config)
                if validation_data is not None:
                    old_metrics = self._evaluate_model(old_model, validation_data)
            
            # Load training data
            training_data = self._load_training_data(job.data_config)
            if training_data is None:
                raise Exception("Failed to load training data")
            
            # Update progress
            self._update_job_progress(job_record, 25.0)
            
            # Train new model
            new_model = self._train_model(
                training_data,
                job.training_config,
                progress_callback=lambda p: self._update_job_progress(job_record, 25.0 + p * 0.5)
            )
            
            # Update progress
            self._update_job_progress(job_record, 75.0)
            
            # Evaluate new model
            validation_data = self._load_validation_data(job.data_config)
            new_metrics = self._evaluate_model(new_model, validation_data)
            
            # Save new model
            model_path = os.path.join(
                self.storage_path,
                'models',
                f'{job.model_id}_{job_id}_model.pkl'
            )
            joblib.dump(new_model, model_path)
            
            # Calculate improvement
            improvement = self._calculate_improvement(old_metrics, new_metrics)
            
            # Update progress
            self._update_job_progress(job_record, 100.0)
            
            # Update job record
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds() / 60
            
            if job_record:
                job_record.status = 'completed'
                job_record.completed_at = end_time
                job_record.duration_minutes = duration
                job_record.old_model_metrics = old_metrics
                job_record.new_model_metrics = new_metrics
                job_record.performance_improvement = improvement
                job_record.model_artifact_path = model_path
                
                # Update schedule success count
                schedule = self.db.query(RetrainingSchedule).filter(
                    RetrainingSchedule.model_id == job.model_id
                ).first()
                if schedule:
                    schedule.success_count += 1
                
                self.db.commit()
            
            return {
                'status': 'completed',
                'job_id': job_id,
                'duration_minutes': duration,
                'old_metrics': old_metrics,
                'new_metrics': new_metrics,
                'improvement': improvement,
                'model_path': model_path
            }
            
        except Exception as e:
            logger.error(f"Retraining job {job_id} failed: {str(e)}")
            
            # Update job record
            if job_record:
                job_record.status = 'failed'
                job_record.error_message = str(e)
                job_record.completed_at = datetime.utcnow()
                
                # Update schedule failure count
                schedule = self.db.query(RetrainingSchedule).filter(
                    RetrainingSchedule.model_id == job.model_id
                ).first()
                if schedule:
                    schedule.failure_count += 1
                
                self.db.commit()
            
            return {
                'status': 'failed',
                'job_id': job_id,
                'error': str(e)
            }

    def _load_training_data(self, data_config: Dict[str, Any]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Load training data based on configuration"""
        
        try:
            data_source = data_config.get('source_type', 'file')
            
            if data_source == 'file':
                file_path = data_config.get('file_path')
                if not file_path or not os.path.exists(file_path):
                    return None
                
                # Load data based on file type
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                elif file_path.endswith('.parquet'):
                    df = pd.read_parquet(file_path)
                else:
                    return None
                
                # Extract features and target
                target_column = data_config.get('target_column')
                feature_columns = data_config.get('feature_columns')
                
                if target_column not in df.columns:
                    return None
                
                if feature_columns:
                    X = df[feature_columns].values
                else:
                    X = df.drop(columns=[target_column]).values
                
                y = df[target_column].values
                
                return X, y
            
            # Add support for other data sources (database, API, etc.)
            
        except Exception as e:
            logger.error(f"Error loading training data: {str(e)}")
        
        return None

    def _load_validation_data(self, data_config: Dict[str, Any]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Load validation data"""
        
        # For now, use a portion of training data for validation
        # In practice, this should be separate validation data
        training_data = self._load_training_data(data_config)
        if training_data is None:
            return None
        
        X, y = training_data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X_val, y_val

    def _train_model(self, training_data: Tuple[np.ndarray, np.ndarray],
                    training_config: Dict[str, Any],
                    progress_callback: Callable[[float], None] = None) -> Any:
        """Train a new model"""
        
        X, y = training_data
        
        # Split data for training and validation
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Determine model type
        model_type = training_config.get('model_type', 'auto')
        
        if model_type == 'auto':
            # Auto-detect based on target variable
            unique_values = len(np.unique(y))
            if unique_values <= 20 and np.issubdtype(y.dtype, np.integer):
                model_type = 'classification'
            else:
                model_type = 'regression'
        
        # Initialize model
        hyperparameters = training_config.get('hyperparameters', {})
        
        if model_type == 'classification':
            if training_config.get('algorithm', 'random_forest') == 'random_forest':
                model = RandomForestClassifier(**hyperparameters)
            else:
                model = LogisticRegression(**hyperparameters)
        else:  # regression
            if training_config.get('algorithm', 'random_forest') == 'random_forest':
                model = RandomForestRegressor(**hyperparameters)
            else:
                model = LinearRegression(**hyperparameters)
        
        # Train model
        if progress_callback:
            progress_callback(0.0)
        
        model.fit(X_train, y_train)
        
        if progress_callback:
            progress_callback(100.0)
        
        return model

    def _evaluate_model(self, model: Any, validation_data: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        """Evaluate model performance"""
        
        X_val, y_val = validation_data
        y_pred = model.predict(X_val)
        
        metrics = {}
        
        # Determine if classification or regression
        unique_values = len(np.unique(y_val))
        
        if unique_values <= 20 and np.issubdtype(y_val.dtype, np.integer):
            # Classification metrics
            metrics['accuracy'] = accuracy_score(y_val, y_pred)
            metrics['precision'] = precision_score(y_val, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_val, y_pred, average='weighted', zero_division=0)
            metrics['f1_score'] = f1_score(y_val, y_pred, average='weighted', zero_division=0)
            
            # AUC for binary classification
            if unique_values == 2:
                try:
                    if hasattr(model, 'predict_proba'):
                        y_proba = model.predict_proba(X_val)[:, 1]
                        metrics['auc'] = roc_auc_score(y_val, y_proba)
                    else:
                        metrics['auc'] = roc_auc_score(y_val, y_pred)
                except:
                    pass
        else:
            # Regression metrics
            metrics['mse'] = mean_squared_error(y_val, y_pred)
            metrics['mae'] = mean_absolute_error(y_val, y_pred)
            metrics['r2'] = r2_score(y_val, y_pred)
        
        return metrics

    def _calculate_improvement(self, old_metrics: Dict[str, float],
                             new_metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate performance improvement"""
        
        if not old_metrics:
            return {'improvement': 'N/A - No baseline metrics'}
        
        improvement = {}
        
        for metric, new_value in new_metrics.items():
            if metric in old_metrics:
                old_value = old_metrics[metric]
                
                if metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc', 'r2']:
                    # Higher is better
                    change = new_value - old_value
                    percent_change = (change / old_value) * 100 if old_value != 0 else 0
                else:
                    # Lower is better (mse, mae)
                    change = old_value - new_value
                    percent_change = (change / old_value) * 100 if old_value != 0 else 0
                
                improvement[f'{metric}_change'] = change
                improvement[f'{metric}_percent_change'] = percent_change
        
        return improvement

    def _update_job_progress(self, job_record: RetrainingJobRecord, progress: float):
        """Update job progress in database"""
        
        if job_record and self.db:
            job_record.progress_percentage = progress
            self.db.commit()

    async def _check_performance_threshold(self, model_id: str,
                                         params: Dict[str, Any]) -> Dict[str, Any]:
        """Check if performance has dropped below threshold"""
        
        # This would integrate with monitoring service to get current performance
        # For now, return mock result
        return {
            'should_retrain': False,
            'reason': 'Performance above threshold',
            'metadata': {
                'current_accuracy': 0.89,
                'threshold': params.get('performance_threshold', 0.85)
            }
        }

    async def _check_drift_threshold(self, model_id: str,
                                   params: Dict[str, Any]) -> Dict[str, Any]:
        """Check if drift has exceeded threshold"""
        
        # This would use the drift detection system
        # For now, return mock result
        return {
            'should_retrain': False,
            'reason': 'No significant drift detected',
            'metadata': {
                'drift_score': 0.03,
                'threshold': params.get('drift_threshold', 0.1)
            }
        }

    async def _trigger_drift_based_retraining(self, model_id: str,
                                            drift_type: str,
                                            drift_info: Dict[str, Any]):
        """Trigger retraining based on drift detection"""
        
        await self.trigger_manual_retraining(
            model_id=model_id,
            reason=f"Drift detected: {drift_type} - p_value: {drift_info['p_value']:.6f}",
            priority=2
        )

    async def _store_drift_detection(self, model_id: str, drift_type: str,
                                   drift_info: Dict[str, Any],
                                   config: DriftDetectionConfig):
        """Store drift detection result in database"""
        
        if not self.db:
            return
        
        detection_record = DriftDetection(
            detection_id=f"drift_{uuid.uuid4().hex[:8]}",
            model_id=model_id,
            drift_type=drift_type,
            detection_method=config.detection_method,
            drift_score=drift_info.get('feature_scores', {}).get('average', 0),
            p_value=drift_info['p_value'],
            threshold=drift_info['threshold'],
            is_drift_detected=drift_info['is_drift_detected'],
            features_analyzed=list(drift_info.get('feature_scores', {}).keys()),
            features_with_drift=[
                k for k, v in drift_info.get('feature_scores', {}).items()
                if v > drift_info['threshold']
            ],
            drift_scores_by_feature=drift_info.get('feature_scores', {}),
            samples_analyzed=drift_info['samples_analyzed'],
            retraining_triggered=config.thresholds.get('auto_retrain', False)
        )
        
        self.db.add(detection_record)
        self.db.commit()

    def _schedule_to_dict(self, schedule: RetrainingSchedule) -> Dict[str, Any]:
        """Convert schedule record to dictionary"""
        
        return {
            'schedule_id': schedule.schedule_id,
            'model_id': schedule.model_id,
            'model_name': schedule.model_name,
            'schedule_type': schedule.schedule_type,
            'schedule_params': schedule.schedule_params,
            'status': schedule.status,
            'enabled': schedule.enabled,
            'last_execution': schedule.last_execution.isoformat() if schedule.last_execution else None,
            'next_execution': schedule.next_execution.isoformat() if schedule.next_execution else None,
            'execution_count': schedule.execution_count,
            'success_count': schedule.success_count,
            'failure_count': schedule.failure_count,
            'created_at': schedule.created_at.isoformat()
        }

    def _job_to_dict(self, job: RetrainingJobRecord) -> Dict[str, Any]:
        """Convert job record to dictionary"""
        
        return {
            'job_id': job.job_id,
            'model_id': job.model_id,
            'trigger_type': job.trigger_type,
            'trigger_metadata': job.trigger_metadata,
            'status': job.status,
            'priority': job.priority,
            'started_at': job.started_at.isoformat() if job.started_at else None,
            'completed_at': job.completed_at.isoformat() if job.completed_at else None,
            'duration_minutes': job.duration_minutes,
            'progress_percentage': job.progress_percentage,
            'old_model_metrics': job.old_model_metrics,
            'new_model_metrics': job.new_model_metrics,
            'performance_improvement': job.performance_improvement,
            'error_message': job.error_message,
            'retry_count': job.retry_count,
            'training_samples': job.training_samples,
            'validation_samples': job.validation_samples
        }

    def stop(self):
        """Stop the retraining service"""
        self.stop_scheduler.set()
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5)
        self.executor.shutdown(wait=True)
        logger.info("Automated retraining service stopped")

# Factory function
def create_automated_retraining_service(db_session: Session = None,
                                      storage_path: str = "/tmp/raia_retraining",
                                      max_concurrent_jobs: int = 3) -> AutomatedRetrainingService:
    """Create and return an AutomatedRetrainingService instance"""
    return AutomatedRetrainingService(db_session, storage_path, max_concurrent_jobs)