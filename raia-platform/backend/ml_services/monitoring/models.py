# Real-time Model Monitoring - Database Models
from sqlalchemy import Column, String, DateTime, Text, Boolean, DECIMAL, Integer, JSON, UUID, ForeignKey, Float, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, ARRAY
import uuid
from datetime import datetime

Base = declarative_base()

class MonitoringConfiguration(Base):
    """Configuration for model monitoring"""
    __tablename__ = "monitoring_configurations"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(PG_UUID(as_uuid=True), ForeignKey('models.id'), nullable=False)
    
    # Configuration details
    config_name = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    monitoring_type = Column(String(50))  # 'performance', 'drift', 'fairness', 'safety'
    
    # Monitoring frequency
    monitoring_interval_seconds = Column(Integer, default=300)  # 5 minutes default
    batch_size = Column(Integer, default=100)
    
    # Performance monitoring
    performance_metrics_enabled = Column(ARRAY(String), default=['accuracy', 'latency', 'throughput'])
    performance_thresholds = Column(JSON)  # Threshold values for alerts
    
    # Drift monitoring
    drift_detection_enabled = Column(Boolean, default=True)
    drift_detection_method = Column(String(50), default='ks_test')  # 'ks_test', 'psi', 'wasserstein'
    drift_threshold = Column(DECIMAL(6,4), default=0.05)
    reference_window_size = Column(Integer, default=1000)
    comparison_window_size = Column(Integer, default=100)
    
    # Fairness monitoring
    fairness_monitoring_enabled = Column(Boolean, default=False)
    protected_attributes = Column(ARRAY(String))
    fairness_metrics = Column(ARRAY(String), default=['demographic_parity', 'equalized_odds'])
    fairness_thresholds = Column(JSON)
    
    # Safety monitoring
    safety_monitoring_enabled = Column(Boolean, default=False)
    toxicity_threshold = Column(DECIMAL(4,3), default=0.1)
    bias_threshold = Column(DECIMAL(4,3), default=0.2)
    
    # Alert configuration
    alert_enabled = Column(Boolean, default=True)
    alert_channels = Column(ARRAY(String), default=['email'])  # 'email', 'slack', 'webhook'
    alert_recipients = Column(ARRAY(String))
    alert_severity_threshold = Column(String(20), default='medium')  # 'low', 'medium', 'high', 'critical'
    
    # Data sources
    prediction_data_source = Column(String(255))  # Database table or stream
    ground_truth_data_source = Column(String(255))
    feature_data_source = Column(String(255))
    
    # Retention policy
    data_retention_days = Column(Integer, default=90)
    
    # Metadata
    created_by = Column(PG_UUID(as_uuid=True), ForeignKey('users.id'))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ModelMonitoringMetric(Base):
    """Real-time monitoring metrics"""
    __tablename__ = "model_monitoring_metrics"
    __table_args__ = (
        Index('idx_model_timestamp', 'model_id', 'timestamp'),
        Index('idx_metric_name_timestamp', 'metric_name', 'timestamp'),
    )
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(PG_UUID(as_uuid=True), ForeignKey('models.id'), nullable=False)
    monitoring_config_id = Column(PG_UUID(as_uuid=True), ForeignKey('monitoring_configurations.id'))
    
    # Metric identification
    metric_name = Column(String(100), nullable=False, index=True)
    metric_category = Column(String(50), nullable=False, index=True)  # 'performance', 'drift', 'fairness', 'safety'
    metric_type = Column(String(50))  # 'counter', 'gauge', 'histogram', 'summary'
    
    # Metric value and metadata
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(20))  # 'seconds', 'percentage', 'count', 'bytes'
    
    # Time series data
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    time_window_start = Column(DateTime)
    time_window_end = Column(DateTime)
    aggregation_period = Column(String(20))  # '1m', '5m', '1h', '1d'
    
    # Context and segmentation
    segment = Column(String(100))  # User segment, geographic region, etc.
    model_version = Column(String(50))
    deployment_environment = Column(String(50))  # 'staging', 'production', 'canary'
    
    # Statistical metadata
    sample_size = Column(Integer)
    confidence_interval_lower = Column(Float)
    confidence_interval_upper = Column(Float)
    standard_error = Column(Float)
    
    # Threshold comparison
    threshold_value = Column(Float)
    threshold_exceeded = Column(Boolean, default=False)
    threshold_type = Column(String(20))  # 'upper', 'lower', 'range'
    
    # Additional metadata
    tags = Column(JSON)  # Flexible tagging system
    metadata = Column(JSON)  # Additional metric-specific data

class DataDriftReport(Base):
    """Data drift detection reports"""
    __tablename__ = "data_drift_reports"
    __table_args__ = (
        Index('idx_model_drift_timestamp', 'model_id', 'timestamp'),
    )
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(PG_UUID(as_uuid=True), ForeignKey('models.id'), nullable=False)
    monitoring_config_id = Column(PG_UUID(as_uuid=True), ForeignKey('monitoring_configurations.id'))
    
    # Drift detection metadata
    drift_type = Column(String(50), nullable=False)  # 'feature_drift', 'prediction_drift', 'concept_drift'
    detection_method = Column(String(100))  # 'ks_test', 'psi', 'wasserstein_distance', 'js_divergence'
    
    # Time periods
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    reference_period_start = Column(DateTime, nullable=False)
    reference_period_end = Column(DateTime, nullable=False)
    comparison_period_start = Column(DateTime, nullable=False)
    comparison_period_end = Column(DateTime, nullable=False)
    
    # Drift results
    drift_detected = Column(Boolean, nullable=False)
    drift_score = Column(Float, nullable=False)  # Overall drift magnitude
    drift_severity = Column(String(20))  # 'low', 'medium', 'high', 'severe'
    
    # Statistical analysis
    p_value = Column(Float)
    test_statistic = Column(Float)
    significance_threshold = Column(Float, default=0.05)
    
    # Feature-level drift
    feature_drift_scores = Column(JSON)  # Per-feature drift scores
    most_drifted_features = Column(ARRAY(String))  # Top drifted features
    feature_drift_details = Column(JSON)  # Detailed per-feature analysis
    
    # Data characteristics
    reference_sample_size = Column(Integer, nullable=False)
    comparison_sample_size = Column(Integer, nullable=False)
    
    # Distribution analysis
    reference_statistics = Column(JSON)  # Mean, std, percentiles for reference
    comparison_statistics = Column(JSON)  # Mean, std, percentiles for comparison
    distribution_shift_analysis = Column(JSON)  # Detailed distribution changes
    
    # Impact assessment
    predicted_performance_impact = Column(Float)  # Predicted impact on model performance
    confidence_in_prediction = Column(Float)  # Confidence in impact prediction
    
    # Recommendations
    recommended_actions = Column(ARRAY(String))
    retrain_recommended = Column(Boolean, default=False)
    urgency_level = Column(String(20))  # 'low', 'medium', 'high', 'immediate'
    
    # Alert status
    alert_triggered = Column(Boolean, default=False)
    alert_sent_at = Column(DateTime)
    alert_recipients = Column(ARRAY(String))

class PerformanceDegradationAlert(Base):
    """Performance degradation alerts and incidents"""
    __tablename__ = "performance_degradation_alerts"
    __table_args__ = (
        Index('idx_model_alert_timestamp', 'model_id', 'timestamp'),
        Index('idx_severity_status', 'severity', 'status'),
    )
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(PG_UUID(as_uuid=True), ForeignKey('models.id'), nullable=False)
    monitoring_config_id = Column(PG_UUID(as_uuid=True), ForeignKey('monitoring_configurations.id'))
    
    # Alert identification
    alert_type = Column(String(50), nullable=False)  # 'performance_drop', 'drift_detected', 'fairness_violation', 'safety_issue'
    severity = Column(String(20), nullable=False, index=True)  # 'low', 'medium', 'high', 'critical'
    status = Column(String(20), default='open', index=True)  # 'open', 'acknowledged', 'investigating', 'resolved', 'closed'
    
    # Timing
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    first_detected_at = Column(DateTime, nullable=False)
    last_detected_at = Column(DateTime)
    resolved_at = Column(DateTime)
    
    # Alert details
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    affected_metrics = Column(ARRAY(String))
    current_values = Column(JSON)  # Current metric values that triggered alert
    threshold_values = Column(JSON)  # Threshold values that were exceeded
    
    # Impact assessment
    estimated_affected_predictions = Column(Integer)
    estimated_business_impact = Column(String(20))  # 'low', 'medium', 'high'
    affected_user_segments = Column(ARRAY(String))
    
    # Root cause analysis
    potential_causes = Column(ARRAY(String))
    contributing_factors = Column(JSON)
    related_incidents = Column(ARRAY(PG_UUID(as_uuid=True)))
    
    # Resolution tracking
    assigned_to = Column(PG_UUID(as_uuid=True), ForeignKey('users.id'))
    investigation_notes = Column(Text)
    resolution_actions = Column(JSON)
    resolution_summary = Column(Text)
    
    # Prevention
    prevention_measures = Column(JSON)
    process_improvements = Column(ARRAY(String))
    
    # Notification tracking
    notifications_sent = Column(JSON)  # Track who was notified and when
    escalation_level = Column(Integer, default=0)
    
    # Metadata
    tags = Column(ARRAY(String))
    metadata = Column(JSON)

class ModelHealthScore(Base):
    """Aggregated model health scores"""
    __tablename__ = "model_health_scores"
    __table_args__ = (
        Index('idx_model_health_timestamp', 'model_id', 'timestamp'),
    )
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(PG_UUID(as_uuid=True), ForeignKey('models.id'), nullable=False)
    
    # Time information
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    calculation_period_start = Column(DateTime, nullable=False)
    calculation_period_end = Column(DateTime, nullable=False)
    
    # Overall health score
    overall_health_score = Column(Float, nullable=False)  # 0-100 scale
    health_grade = Column(String(10))  # 'A+', 'A', 'B+', 'B', 'C+', 'C', 'D', 'F'
    health_status = Column(String(20))  # 'excellent', 'good', 'fair', 'poor', 'critical'
    
    # Component scores
    performance_score = Column(Float)  # Accuracy, latency, throughput
    reliability_score = Column(Float)  # Uptime, error rates
    stability_score = Column(Float)  # Consistency over time
    drift_score = Column(Float)  # Data and concept drift
    fairness_score = Column(Float)  # Bias and fairness metrics
    safety_score = Column(Float)  # Safety and toxicity
    
    # Performance metrics contributing to score
    accuracy_score = Column(Float)
    latency_score = Column(Float)
    throughput_score = Column(Float)
    error_rate_score = Column(Float)
    
    # Trend analysis
    score_trend = Column(String(20))  # 'improving', 'stable', 'declining', 'volatile'
    trend_confidence = Column(Float)  # Confidence in trend assessment
    days_since_last_incident = Column(Integer)
    
    # Predictive indicators
    predicted_health_24h = Column(Float)  # Predicted health in 24 hours
    predicted_health_7d = Column(Float)  # Predicted health in 7 days
    risk_factors = Column(ARRAY(String))  # Identified risk factors
    
    # Recommendations
    health_recommendations = Column(ARRAY(String))
    priority_actions = Column(ARRAY(String))
    maintenance_suggestions = Column(JSON)
    
    # Calculation metadata
    metrics_included = Column(ARRAY(String))  # Which metrics contributed to score
    calculation_method = Column(String(100))  # How score was calculated
    confidence_interval = Column(JSON)  # Confidence bounds for the score
    
    # Comparison with baselines
    baseline_comparison = Column(JSON)  # Comparison with historical baselines
    peer_model_comparison = Column(JSON)  # Comparison with similar models

class MonitoringDashboard(Base):
    """Monitoring dashboard configurations"""
    __tablename__ = "monitoring_dashboards"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Dashboard identification
    name = Column(String(255), nullable=False)
    description = Column(Text)
    dashboard_type = Column(String(50))  # 'model_overview', 'performance', 'drift', 'fairness', 'safety'
    
    # Access control
    is_public = Column(Boolean, default=False)
    owner_id = Column(PG_UUID(as_uuid=True), ForeignKey('users.id'))
    shared_with = Column(ARRAY(PG_UUID(as_uuid=True)))  # User IDs with access
    
    # Dashboard configuration
    layout_config = Column(JSON)  # Dashboard layout and widget positions
    widget_configs = Column(JSON)  # Individual widget configurations
    filters = Column(JSON)  # Default filters
    time_range_default = Column(String(20), default='24h')  # '1h', '24h', '7d', '30d'
    
    # Models and metrics
    monitored_models = Column(ARRAY(PG_UUID(as_uuid=True)))  # Model IDs to monitor
    displayed_metrics = Column(ARRAY(String))  # Metrics to display
    
    # Refresh and updates
    auto_refresh_interval = Column(Integer, default=60)  # Seconds
    last_updated = Column(DateTime, default=datetime.utcnow)
    
    # Customization
    theme = Column(String(20), default='light')  # 'light', 'dark'
    color_scheme = Column(JSON)  # Custom colors for charts
    
    # Metadata
    tags = Column(ARRAY(String))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class MonitoringReport(Base):
    """Scheduled monitoring reports"""
    __tablename__ = "monitoring_reports"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Report identification
    report_name = Column(String(255), nullable=False)
    report_type = Column(String(50))  # 'daily', 'weekly', 'monthly', 'incident', 'custom'
    
    # Report scope
    model_ids = Column(ARRAY(PG_UUID(as_uuid=True)))  # Models included in report
    report_period_start = Column(DateTime, nullable=False)
    report_period_end = Column(DateTime, nullable=False)
    
    # Report content
    executive_summary = Column(Text)
    key_findings = Column(ARRAY(String))
    performance_highlights = Column(JSON)
    issues_identified = Column(JSON)
    recommendations = Column(ARRAY(String))
    
    # Metrics summary
    overall_health_summary = Column(JSON)
    performance_metrics_summary = Column(JSON)
    drift_analysis_summary = Column(JSON)
    fairness_metrics_summary = Column(JSON)
    safety_metrics_summary = Column(JSON)
    
    # Incidents and alerts
    incidents_during_period = Column(Integer, default=0)
    critical_alerts_count = Column(Integer, default=0)
    high_alerts_count = Column(Integer, default=0)
    incident_summaries = Column(JSON)
    
    # Trends and predictions
    trend_analysis = Column(JSON)
    predictive_insights = Column(JSON)
    risk_assessment = Column(JSON)
    
    # Actions and follow-ups
    action_items = Column(JSON)
    assigned_tasks = Column(JSON)
    follow_up_required = Column(Boolean, default=False)
    next_review_date = Column(DateTime)
    
    # Report generation
    generated_at = Column(DateTime, default=datetime.utcnow)
    generated_by = Column(PG_UUID(as_uuid=True), ForeignKey('users.id'))
    generation_method = Column(String(50))  # 'automatic', 'manual', 'scheduled'
    
    # Distribution
    recipients = Column(ARRAY(String))  # Email addresses
    distribution_channels = Column(ARRAY(String))  # 'email', 'slack', 'pdf'
    
    # Status
    status = Column(String(20), default='generated')  # 'generating', 'generated', 'sent', 'archived'
    
    # Metadata
    tags = Column(ARRAY(String))
    metadata = Column(JSON)

class MonitoringAuditLog(Base):
    """Audit log for monitoring system changes"""
    __tablename__ = "monitoring_audit_logs"
    __table_args__ = (
        Index('idx_audit_timestamp', 'timestamp'),
        Index('idx_audit_action_type', 'action_type'),
    )
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Action details
    action_type = Column(String(100), nullable=False, index=True)  # 'create', 'update', 'delete', 'alert_triggered'
    resource_type = Column(String(50), nullable=False)  # 'monitoring_config', 'alert', 'dashboard', 'report'
    resource_id = Column(PG_UUID(as_uuid=True), nullable=False)
    
    # Actor information
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey('users.id'))
    user_email = Column(String(255))
    source_ip = Column(String(45))  # IPv6 compatible
    user_agent = Column(String(500))
    
    # Change details
    changes_made = Column(JSON)  # Detailed change information
    previous_values = Column(JSON)  # Values before change
    new_values = Column(JSON)  # Values after change
    
    # Context
    model_id = Column(PG_UUID(as_uuid=True), ForeignKey('models.id'))
    session_id = Column(String(255))
    request_id = Column(String(255))
    
    # Timing
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    # Additional metadata
    tags = Column(ARRAY(String))
    metadata = Column(JSON)

class RealTimeMonitoringStream(Base):
    """Real-time streaming metrics for high-frequency monitoring"""
    __tablename__ = "realtime_monitoring_stream"
    __table_args__ = (
        Index('idx_realtime_model_timestamp', 'model_id', 'timestamp'),
        Index('idx_realtime_timestamp', 'timestamp'),
    )
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(PG_UUID(as_uuid=True), ForeignKey('models.id'), nullable=False)
    
    # Streaming data
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    prediction_id = Column(PG_UUID(as_uuid=True))  # Link to specific prediction if available
    
    # Real-time metrics
    response_time_ms = Column(Float)
    throughput_rps = Column(Float)  # Requests per second
    error_occurred = Column(Boolean, default=False)
    error_type = Column(String(100))
    
    # Input/output characteristics
    input_size_bytes = Column(Integer)
    output_size_bytes = Column(Integer)
    input_tokens = Column(Integer)  # For LLMs
    output_tokens = Column(Integer)  # For LLMs
    
    # Quality indicators (if available)
    confidence_score = Column(Float)
    prediction_value = Column(JSON)  # Actual prediction value
    
    # Context information
    user_segment = Column(String(100))
    geographic_region = Column(String(100))
    device_type = Column(String(50))
    model_version = Column(String(50))
    
    # Fairness attributes (if provided)
    protected_attributes = Column(JSON)
    
    # Processing pipeline
    preprocessing_time_ms = Column(Float)
    inference_time_ms = Column(Float)
    postprocessing_time_ms = Column(Float)
    
    # Resource utilization
    cpu_usage_percent = Column(Float)
    memory_usage_mb = Column(Float)
    gpu_usage_percent = Column(Float)
    
    # Additional metadata
    session_id = Column(String(255))
    batch_id = Column(String(255))
    metadata = Column(JSON)