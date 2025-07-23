# ML Model Registry - Database Models
from sqlalchemy import Column, String, DateTime, Text, Boolean, DECIMAL, Integer, JSON, UUID, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, ARRAY
import uuid
from datetime import datetime

Base = declarative_base()

class Model(Base):
    """Core model registry table storing ML model metadata and configuration"""
    __tablename__ = "models"
    
    # Primary identifiers
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, unique=True)
    display_name = Column(String(255), nullable=False)
    description = Column(Text)
    
    # Model classification
    model_type = Column(String(50), nullable=False)  # 'classification', 'regression', 'llm', 'computer_vision'
    algorithm = Column(String(100))  # 'random_forest', 'neural_network', 'gpt', 'bert'
    framework = Column(String(50))   # 'scikit-learn', 'tensorflow', 'pytorch', 'transformers'
    
    # Version control
    version = Column(String(20), nullable=False, default="1.0.0")
    is_latest = Column(Boolean, default=True)
    
    # Status and lifecycle
    status = Column(String(50), default='development')  # 'development', 'staging', 'production', 'retired'
    deployment_status = Column(String(50), default='not_deployed')  # 'not_deployed', 'deployed', 'failed'
    
    # Performance metrics
    accuracy = Column(DECIMAL(5,4))
    precision = Column(DECIMAL(5,4))
    recall = Column(DECIMAL(5,4))
    f1_score = Column(DECIMAL(5,4))
    auc_score = Column(DECIMAL(5,4))
    
    # Data schema information
    feature_names = Column(ARRAY(String))
    feature_types = Column(JSON)  # {'feature_name': 'categorical|numerical|text'}
    target_column = Column(String(100))
    
    # Model artifacts and storage
    model_path = Column(String(500))  # Path to serialized model file
    model_size_mb = Column(DECIMAL(10,2))
    training_dataset_id = Column(PG_UUID(as_uuid=True), ForeignKey('datasets.id'))
    
    # Training information
    training_started_at = Column(DateTime)
    training_completed_at = Column(DateTime)
    training_duration_minutes = Column(Integer)
    training_parameters = Column(JSON)  # Hyperparameters used
    
    # Business context
    business_objective = Column(Text)
    use_cases = Column(ARRAY(String))
    stakeholders = Column(ARRAY(String))
    
    # Governance and compliance
    approval_status = Column(String(50), default='pending')  # 'pending', 'approved', 'rejected'
    approved_by = Column(PG_UUID(as_uuid=True), ForeignKey('users.id'))
    approved_at = Column(DateTime)
    
    # Risk and fairness flags
    fairness_assessment_required = Column(Boolean, default=True)
    bias_risk_level = Column(String(20))  # 'low', 'medium', 'high'
    
    # Metadata and tags
    tags = Column(ARRAY(String))
    metadata = Column(JSON)
    
    # Audit fields
    created_by = Column(PG_UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    versions = relationship("ModelVersion", back_populates="model")
    predictions = relationship("ModelPrediction", back_populates="model")
    fairness_reports = relationship("FairnessReport", back_populates="model")
    performance_metrics = relationship("ModelPerformanceMetric", back_populates="model")
    explainability_reports = relationship("ExplainabilityReport", back_populates="model")

class ModelVersion(Base):
    """Model version tracking for MLOps and model evolution"""
    __tablename__ = "model_versions"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(PG_UUID(as_uuid=True), ForeignKey('models.id'), nullable=False)
    version = Column(String(20), nullable=False)
    
    # Version-specific information
    changelog = Column(Text)
    performance_improvements = Column(JSON)
    breaking_changes = Column(Boolean, default=False)
    
    # Deployment information
    deployed_at = Column(DateTime)
    deployment_environment = Column(String(50))  # 'staging', 'production'
    deployment_config = Column(JSON)
    
    # Performance comparison
    previous_version_id = Column(PG_UUID(as_uuid=True), ForeignKey('model_versions.id'))
    performance_delta = Column(JSON)  # Performance change vs previous version
    
    # A/B testing results
    ab_test_results = Column(JSON)
    champion_challenger_status = Column(String(20))  # 'champion', 'challenger', 'retired'
    
    # Rollback information
    can_rollback = Column(Boolean, default=True)
    rollback_strategy = Column(JSON)
    
    # Audit fields
    created_by = Column(PG_UUID(as_uuid=True), ForeignKey('users.id'))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    model = relationship("Model", back_populates="versions")
    previous_version = relationship("ModelVersion", remote_side=[id])

class ModelPrediction(Base):
    """Individual model predictions for auditing and analysis"""
    __tablename__ = "model_predictions"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(PG_UUID(as_uuid=True), ForeignKey('models.id'), nullable=False)
    model_version = Column(String(20))
    
    # Prediction data
    input_data = Column(JSON, nullable=False)  # Input features
    prediction = Column(JSON, nullable=False)  # Model output
    prediction_probabilities = Column(JSON)    # For classification models
    confidence_score = Column(DECIMAL(5,4))
    
    # Explainability data
    shap_values = Column(JSON)              # SHAP explanation values
    lime_explanation = Column(JSON)         # LIME explanation
    feature_importance = Column(JSON)       # Feature importance for this prediction
    
    # Context and metadata
    prediction_context = Column(JSON)       # Business context when prediction was made
    feedback_received = Column(Boolean, default=False)
    actual_outcome = Column(JSON)          # Ground truth if available
    prediction_error = Column(DECIMAL(10,4)) # Error if actual outcome known
    
    # Fairness and bias tracking
    protected_attributes = Column(JSON)     # Values of protected attributes
    fairness_metrics = Column(JSON)        # Individual fairness scores
    
    # System information
    api_endpoint = Column(String(255))      # Which API endpoint was used
    request_id = Column(String(100))        # Request tracking ID
    response_time_ms = Column(Integer)      # Prediction latency
    
    # Audit fields
    predicted_at = Column(DateTime, default=datetime.utcnow)
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey('users.id'))
    
    # Relationships
    model = relationship("Model", back_populates="predictions")

class ModelPerformanceMetric(Base):
    """Time-series performance metrics for model monitoring"""
    __tablename__ = "model_performance_metrics"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(PG_UUID(as_uuid=True), ForeignKey('models.id'), nullable=False)
    model_version = Column(String(20))
    
    # Metric information
    metric_name = Column(String(100), nullable=False)  # 'accuracy', 'precision', 'recall', etc.
    metric_value = Column(DECIMAL(10,6), nullable=False)
    metric_category = Column(String(50))  # 'performance', 'fairness', 'drift', 'latency'
    
    # Time-series data
    measurement_date = Column(DateTime, nullable=False, default=datetime.utcnow)
    measurement_period = Column(String(20))  # 'hourly', 'daily', 'weekly', 'monthly'
    
    # Data context
    dataset_name = Column(String(255))
    sample_size = Column(Integer)
    data_period_start = Column(DateTime)
    data_period_end = Column(DateTime)
    
    # Thresholds and alerts
    threshold_value = Column(DECIMAL(10,6))
    alert_triggered = Column(Boolean, default=False)
    alert_severity = Column(String(20))  # 'low', 'medium', 'high', 'critical'
    
    # Comparison metrics
    baseline_value = Column(DECIMAL(10,6))  # Original/baseline performance
    trend_direction = Column(String(20))    # 'improving', 'degrading', 'stable'
    change_percentage = Column(DECIMAL(8,4)) # Percentage change from baseline
    
    # Additional context
    metadata = Column(JSON)
    
    # Audit fields
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    model = relationship("Model", back_populates="performance_metrics")

class ModelDriftReport(Base):
    """Model and data drift detection reports"""
    __tablename__ = "model_drift_reports"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(PG_UUID(as_uuid=True), ForeignKey('models.id'), nullable=False)
    
    # Drift analysis
    drift_type = Column(String(50), nullable=False)  # 'data_drift', 'concept_drift', 'model_drift'
    drift_detected = Column(Boolean, nullable=False)
    drift_severity = Column(String(20))  # 'low', 'medium', 'high', 'critical'
    drift_score = Column(DECIMAL(5,4))
    
    # Statistical analysis
    statistical_test = Column(String(100))  # 'ks_test', 'psi', 'wasserstein'
    p_value = Column(DECIMAL(10,8))
    test_statistic = Column(DECIMAL(10,6))
    significance_threshold = Column(DECIMAL(5,4), default=0.05)
    
    # Feature-level drift
    feature_drift_scores = Column(JSON)    # Per-feature drift scores
    most_drifted_features = Column(ARRAY(String))
    
    # Time period analysis
    reference_period_start = Column(DateTime)
    reference_period_end = Column(DateTime)
    comparison_period_start = Column(DateTime)
    comparison_period_end = Column(DateTime)
    
    # Data characteristics
    reference_sample_size = Column(Integer)
    comparison_sample_size = Column(Integer)
    reference_distribution = Column(JSON)
    comparison_distribution = Column(JSON)
    
    # Recommendations
    recommended_actions = Column(ARRAY(String))
    retrain_recommended = Column(Boolean, default=False)
    urgency_level = Column(String(20))  # 'low', 'medium', 'high', 'immediate'
    
    # Alert information
    alert_sent = Column(Boolean, default=False)
    alert_recipients = Column(ARRAY(String))
    alert_sent_at = Column(DateTime)
    
    # Analysis metadata
    analysis_config = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class ModelExperiment(Base):
    """Model training experiments and hyperparameter optimization"""
    __tablename__ = "model_experiments"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(PG_UUID(as_uuid=True), ForeignKey('models.id'), nullable=False)
    
    # Experiment identification
    experiment_name = Column(String(255), nullable=False)
    experiment_type = Column(String(50))  # 'hyperparameter_tuning', 'feature_selection', 'architecture_search'
    
    # Experiment configuration
    parameters = Column(JSON)  # All hyperparameters and configurations tested
    parameter_search_space = Column(JSON)  # Search space for optimization
    optimization_metric = Column(String(100))  # Metric being optimized
    optimization_direction = Column(String(10))  # 'maximize', 'minimize'
    
    # Experiment execution
    status = Column(String(50), default='pending')  # 'pending', 'running', 'completed', 'failed', 'cancelled'
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    duration_minutes = Column(Integer)
    
    # Results
    best_parameters = Column(JSON)
    best_score = Column(DECIMAL(10,6))
    all_results = Column(JSON)  # All parameter combinations and their scores
    convergence_history = Column(JSON)  # Optimization convergence
    
    # Resource usage
    compute_resources_used = Column(JSON)  # CPU, GPU, memory usage
    estimated_cost = Column(DECIMAL(10,2))
    
    # Experiment metadata
    notes = Column(Text)
    tags = Column(ARRAY(String))
    
    # Audit fields
    created_by = Column(PG_UUID(as_uuid=True), ForeignKey('users.id'))
    created_at = Column(DateTime, default=datetime.utcnow)

class ModelApprovalWorkflow(Base):
    """Model governance and approval workflow"""
    __tablename__ = "model_approval_workflows"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(PG_UUID(as_uuid=True), ForeignKey('models.id'), nullable=False)
    
    # Workflow status
    workflow_stage = Column(String(50), nullable=False)  # 'validation', 'review', 'testing', 'approval'
    overall_status = Column(String(50), default='pending')  # 'pending', 'approved', 'rejected', 'needs_revision'
    
    # Review requirements
    technical_review_required = Column(Boolean, default=True)
    business_review_required = Column(Boolean, default=True)
    security_review_required = Column(Boolean, default=True)
    compliance_review_required = Column(Boolean, default=True)
    
    # Review completion status
    technical_review_completed = Column(Boolean, default=False)
    business_review_completed = Column(Boolean, default=False)
    security_review_completed = Column(Boolean, default=False)
    compliance_review_completed = Column(Boolean, default=False)
    
    # Reviewer assignments
    technical_reviewer = Column(PG_UUID(as_uuid=True), ForeignKey('users.id'))
    business_reviewer = Column(PG_UUID(as_uuid=True), ForeignKey('users.id'))
    security_reviewer = Column(PG_UUID(as_uuid=True), ForeignKey('users.id'))
    compliance_reviewer = Column(PG_UUID(as_uuid=True), ForeignKey('users.id'))
    
    # Review results
    technical_review_result = Column(String(20))  # 'pass', 'fail', 'conditional'
    business_review_result = Column(String(20))
    security_review_result = Column(String(20))
    compliance_review_result = Column(String(20))
    
    # Comments and feedback
    technical_comments = Column(Text)
    business_comments = Column(Text)
    security_comments = Column(Text)
    compliance_comments = Column(Text)
    
    # Decision and approval
    final_decision = Column(String(20))  # 'approved', 'rejected', 'conditional'
    decision_rationale = Column(Text)
    approved_by = Column(PG_UUID(as_uuid=True), ForeignKey('users.id'))
    approved_at = Column(DateTime)
    
    # Conditions and requirements
    approval_conditions = Column(ARRAY(String))
    conditions_met = Column(Boolean, default=False)
    
    # Audit fields
    workflow_started_at = Column(DateTime, default=datetime.utcnow)
    workflow_completed_at = Column(DateTime)
    created_by = Column(PG_UUID(as_uuid=True), ForeignKey('users.id'))