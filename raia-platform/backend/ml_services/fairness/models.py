# Fairness and Bias Detection - Database Models
from sqlalchemy import Column, String, DateTime, Text, Boolean, DECIMAL, Integer, JSON, UUID, ForeignKey, ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
import uuid
from datetime import datetime

Base = declarative_base()

class FairnessReport(Base):
    """Comprehensive fairness analysis reports for ML models"""
    __tablename__ = "fairness_reports"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(PG_UUID(as_uuid=True), ForeignKey('models.id'), nullable=False)
    
    # Report metadata
    report_name = Column(String(255), nullable=False)
    report_type = Column(String(50), nullable=False)  # 'bias_audit', 'fairness_assessment', 'impact_analysis'
    analysis_scope = Column(String(50))  # 'individual', 'group', 'intersectional'
    
    # Protected attributes analyzed
    protected_attributes = Column(ARRAY(String), nullable=False)  # ['race', 'gender', 'age']
    attribute_categories = Column(JSON)  # {'race': ['white', 'black', 'hispanic'], 'gender': ['male', 'female']}
    
    # Dataset information
    dataset_id = Column(PG_UUID(as_uuid=True), ForeignKey('datasets.id'))
    dataset_name = Column(String(255))
    sample_size = Column(Integer)
    analysis_date_range = Column(JSON)  # {'start': '2024-01-01', 'end': '2024-01-31'}
    
    # Overall fairness assessment
    overall_fairness_score = Column(DECIMAL(5,4))  # 0-1 scale
    bias_detected = Column(Boolean, default=False)
    bias_severity = Column(String(20))  # 'low', 'moderate', 'high', 'severe'
    fairness_status = Column(String(20))  # 'fair', 'potentially_biased', 'biased', 'severely_biased'
    
    # Fairness metrics summary
    demographic_parity_score = Column(DECIMAL(8,6))
    equalized_odds_score = Column(DECIMAL(8,6))
    equal_opportunity_score = Column(DECIMAL(8,6))
    calibration_score = Column(DECIMAL(8,6))
    individual_fairness_score = Column(DECIMAL(8,6))
    
    # Statistical significance
    statistical_significance = Column(Boolean, default=False)
    p_value = Column(DECIMAL(10,8))
    confidence_level = Column(DECIMAL(4,3), default=0.95)
    
    # Detailed analysis results
    group_metrics = Column(JSON)  # Detailed metrics for each group
    pairwise_comparisons = Column(JSON)  # All pairwise group comparisons
    intersectional_analysis = Column(JSON)  # Results for intersectional groups
    
    # Bias sources and explanations
    identified_bias_sources = Column(ARRAY(String))  # ['feature_bias', 'historical_bias', 'representation_bias']
    bias_explanations = Column(JSON)  # Detailed explanations for each bias source
    feature_bias_analysis = Column(JSON)  # Which features contribute most to bias
    
    # Recommendations and mitigations
    recommended_actions = Column(ARRAY(String))
    mitigation_strategies = Column(JSON)
    urgency_level = Column(String(20))  # 'low', 'medium', 'high', 'critical'
    
    # Compliance and regulatory
    regulatory_compliance = Column(JSON)  # GDPR, CCPA, EU AI Act compliance status
    legal_risk_assessment = Column(String(20))  # 'low', 'medium', 'high'
    
    # Report metadata
    methodology = Column(Text)
    limitations = Column(Text)
    assumptions = Column(ARRAY(String))
    
    # Audit trail
    created_by = Column(PG_UUID(as_uuid=True), ForeignKey('users.id'))
    reviewed_by = Column(PG_UUID(as_uuid=True), ForeignKey('users.id'))
    approved_by = Column(PG_UUID(as_uuid=True), ForeignKey('users.id'))
    created_at = Column(DateTime, default=datetime.utcnow)
    reviewed_at = Column(DateTime)
    approved_at = Column(DateTime)
    
    # Relationships
    model = relationship("Model", back_populates="fairness_reports")
    bias_incidents = relationship("BiasIncident", back_populates="fairness_report")
    mitigation_plans = relationship("BiasMitigationPlan", back_populates="fairness_report")

class BiasMetric(Base):
    """Individual bias metrics and measurements"""
    __tablename__ = "bias_metrics"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    fairness_report_id = Column(PG_UUID(as_uuid=True), ForeignKey('fairness_reports.id'), nullable=False)
    model_id = Column(PG_UUID(as_uuid=True), ForeignKey('models.id'), nullable=False)
    
    # Metric identification
    metric_name = Column(String(100), nullable=False)  # 'demographic_parity', 'equal_opportunity', etc.
    metric_category = Column(String(50))  # 'group_fairness', 'individual_fairness', 'causal_fairness'
    
    # Protected group information
    protected_attribute = Column(String(100), nullable=False)
    reference_group = Column(String(100))  # The baseline/privileged group
    comparison_group = Column(String(100))  # The group being compared
    
    # Metric values
    metric_value = Column(DECIMAL(10,6), nullable=False)
    reference_group_value = Column(DECIMAL(10,6))
    comparison_group_value = Column(DECIMAL(10,6))
    
    # Thresholds and assessment
    fairness_threshold = Column(DECIMAL(5,4), default=0.8)  # Minimum acceptable ratio
    passes_threshold = Column(Boolean)
    bias_magnitude = Column(DECIMAL(8,6))  # How far from fair (1.0)
    
    # Statistical analysis
    standard_error = Column(DECIMAL(8,6))
    confidence_interval_lower = Column(DECIMAL(8,6))
    confidence_interval_upper = Column(DECIMAL(8,6))
    statistical_significance = Column(Boolean)
    
    # Sample information
    total_sample_size = Column(Integer)
    reference_group_size = Column(Integer)
    comparison_group_size = Column(Integer)
    
    # Temporal tracking
    measurement_date = Column(DateTime, default=datetime.utcnow)
    measurement_period = Column(String(20))  # 'daily', 'weekly', 'monthly'
    
    # Metadata
    calculation_method = Column(Text)
    data_filters = Column(JSON)  # Any filters applied to data
    notes = Column(Text)
    
    # Relationships
    fairness_report = relationship("FairnessReport")

class BiasIncident(Base):
    """Documented bias incidents and their resolutions"""
    __tablename__ = "bias_incidents"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    fairness_report_id = Column(PG_UUID(as_uuid=True), ForeignKey('fairness_reports.id'))
    model_id = Column(PG_UUID(as_uuid=True), ForeignKey('models.id'), nullable=False)
    
    # Incident identification
    incident_title = Column(String(255), nullable=False)
    incident_type = Column(String(50))  # 'algorithmic_bias', 'data_bias', 'outcome_bias'
    severity_level = Column(String(20))  # 'low', 'medium', 'high', 'critical'
    
    # Incident details
    description = Column(Text, nullable=False)
    affected_groups = Column(ARRAY(String))
    protected_attributes_involved = Column(ARRAY(String))
    
    # Discovery information
    discovered_date = Column(DateTime, nullable=False)
    discovery_method = Column(String(100))  # 'automated_monitoring', 'manual_review', 'user_report'
    discovered_by = Column(PG_UUID(as_uuid=True), ForeignKey('users.id'))
    
    # Impact assessment
    estimated_affected_individuals = Column(Integer)
    business_impact = Column(String(20))  # 'low', 'medium', 'high'
    reputational_impact = Column(String(20))  # 'low', 'medium', 'high'
    legal_risk = Column(String(20))  # 'low', 'medium', 'high'
    
    # Root cause analysis
    root_causes = Column(ARRAY(String))
    contributing_factors = Column(JSON)
    bias_source = Column(String(100))  # 'training_data', 'feature_selection', 'algorithm_design'
    
    # Resolution tracking
    status = Column(String(50), default='open')  # 'open', 'investigating', 'resolved', 'closed'
    assigned_to = Column(PG_UUID(as_uuid=True), ForeignKey('users.id'))
    resolution_deadline = Column(DateTime)
    resolved_date = Column(DateTime)
    
    # Resolution details
    resolution_actions = Column(JSON)
    resolution_description = Column(Text)
    resolution_effectiveness = Column(String(20))  # 'effective', 'partially_effective', 'ineffective'
    
    # Follow-up monitoring
    monitoring_required = Column(Boolean, default=True)
    monitoring_frequency = Column(String(20))  # 'daily', 'weekly', 'monthly'
    next_review_date = Column(DateTime)
    
    # Documentation
    evidence = Column(JSON)  # Links to screenshots, data samples, etc.
    stakeholder_notifications = Column(JSON)  # Who was notified and when
    
    # Audit trail
    created_by = Column(PG_UUID(as_uuid=True), ForeignKey('users.id'))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    fairness_report = relationship("FairnessReport", back_populates="bias_incidents")
    mitigation_plans = relationship("BiasMitigationPlan", back_populates="bias_incident")

class BiasMitigationPlan(Base):
    """Bias mitigation strategies and implementation plans"""
    __tablename__ = "bias_mitigation_plans"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    fairness_report_id = Column(PG_UUID(as_uuid=True), ForeignKey('fairness_reports.id'))
    bias_incident_id = Column(PG_UUID(as_uuid=True), ForeignKey('bias_incidents.id'))
    model_id = Column(PG_UUID(as_uuid=True), ForeignKey('models.id'), nullable=False)
    
    # Plan identification
    plan_name = Column(String(255), nullable=False)
    plan_type = Column(String(50))  # 'preprocessing', 'inprocessing', 'postprocessing'
    mitigation_strategy = Column(String(100))  # 'reweighting', 'fairness_constraints', 'adversarial_debiasing'
    
    # Plan details
    description = Column(Text, nullable=False)
    objectives = Column(ARRAY(String))
    success_criteria = Column(JSON)
    
    # Target groups and metrics
    target_protected_attributes = Column(ARRAY(String))
    target_fairness_metrics = Column(ARRAY(String))
    target_metric_values = Column(JSON)  # Expected metric improvements
    
    # Implementation details
    implementation_approach = Column(Text)
    technical_requirements = Column(JSON)
    resource_requirements = Column(JSON)
    estimated_effort_hours = Column(Integer)
    estimated_cost = Column(DECIMAL(12,2))
    
    # Timeline
    planned_start_date = Column(DateTime)
    planned_completion_date = Column(DateTime)
    actual_start_date = Column(DateTime)
    actual_completion_date = Column(DateTime)
    
    # Status tracking
    status = Column(String(50), default='planned')  # 'planned', 'in_progress', 'testing', 'implemented', 'cancelled'
    progress_percentage = Column(Integer, default=0)
    
    # Team and responsibilities
    plan_owner = Column(PG_UUID(as_uuid=True), ForeignKey('users.id'))
    technical_lead = Column(PG_UUID(as_uuid=True), ForeignKey('users.id'))
    stakeholders = Column(ARRAY(PG_UUID(as_uuid=True)))
    
    # Risk assessment
    implementation_risks = Column(JSON)
    risk_mitigation_strategies = Column(JSON)
    
    # Testing and validation
    validation_approach = Column(Text)
    test_datasets = Column(ARRAY(String))
    validation_metrics = Column(ARRAY(String))
    
    # Results and effectiveness
    effectiveness_measured = Column(Boolean, default=False)
    pre_implementation_metrics = Column(JSON)
    post_implementation_metrics = Column(JSON)
    effectiveness_score = Column(DECIMAL(5,4))  # 0-1 scale
    
    # Side effects and trade-offs
    accuracy_impact = Column(DECIMAL(6,4))  # Change in model accuracy
    performance_impact = Column(DECIMAL(6,4))  # Change in other performance metrics
    unintended_consequences = Column(JSON)
    
    # Approval workflow
    approval_required = Column(Boolean, default=True)
    approved_by = Column(PG_UUID(as_uuid=True), ForeignKey('users.id'))
    approved_at = Column(DateTime)
    approval_conditions = Column(ARRAY(String))
    
    # Documentation
    implementation_notes = Column(Text)
    lessons_learned = Column(Text)
    recommendations = Column(Text)
    
    # Audit trail
    created_by = Column(PG_UUID(as_uuid=True), ForeignKey('users.id'))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    fairness_report = relationship("FairnessReport", back_populates="mitigation_plans")
    bias_incident = relationship("BiasIncident", back_populates="mitigation_plans")

class FairnessConfiguration(Base):
    """Configuration settings for fairness analysis"""
    __tablename__ = "fairness_configurations"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(PG_UUID(as_uuid=True), ForeignKey('models.id'))
    
    # Configuration details
    config_name = Column(String(255), nullable=False)
    is_default = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    
    # Protected attributes configuration
    protected_attributes_config = Column(JSON, nullable=False)
    # Example: {
    #   "race": {
    #     "privileged_groups": ["white"],
    #     "unprivileged_groups": ["black", "hispanic", "asian"],
    #     "intersectional": true
    #   },
    #   "gender": {
    #     "privileged_groups": ["male"],
    #     "unprivileged_groups": ["female"],
    #     "intersectional": true
    #   }
    # }
    
    # Fairness metrics configuration
    enabled_metrics = Column(ARRAY(String), nullable=False)
    metric_thresholds = Column(JSON)  # Threshold values for each metric
    
    # Analysis settings
    analysis_frequency = Column(String(20))  # 'daily', 'weekly', 'monthly'
    minimum_sample_size = Column(Integer, default=100)
    confidence_level = Column(DECIMAL(4,3), default=0.95)
    
    # Alert configuration
    alert_enabled = Column(Boolean, default=True)
    alert_thresholds = Column(JSON)  # When to trigger alerts
    alert_recipients = Column(ARRAY(String))  # Email addresses
    
    # Reporting settings
    auto_report_generation = Column(Boolean, default=True)
    report_frequency = Column(String(20))  # 'weekly', 'monthly', 'quarterly'
    report_recipients = Column(ARRAY(String))
    
    # Metadata
    description = Column(Text)
    tags = Column(ARRAY(String))
    
    # Audit trail
    created_by = Column(PG_UUID(as_uuid=True), ForeignKey('users.id'))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ExplainabilityReport(Base):
    """Explainability analysis reports linking to fairness"""
    __tablename__ = "explainability_reports"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(PG_UUID(as_uuid=True), ForeignKey('models.id'), nullable=False)
    fairness_report_id = Column(PG_UUID(as_uuid=True), ForeignKey('fairness_reports.id'))
    
    # Report metadata
    report_name = Column(String(255), nullable=False)
    explanation_method = Column(String(50))  # 'shap', 'lime', 'counterfactual'
    analysis_scope = Column(String(50))  # 'global', 'local', 'cohort'
    
    # Analysis configuration
    instances_analyzed = Column(Integer)
    features_analyzed = Column(ARRAY(String))
    protected_attributes = Column(ARRAY(String))
    
    # Results summary
    feature_importance_by_group = Column(JSON)  # Feature importance for each demographic group
    differential_explanations = Column(JSON)  # How explanations differ across groups
    explanation_disparity_metrics = Column(JSON)  # Metrics measuring explanation consistency
    
    # Bias in explanations
    explanation_bias_detected = Column(Boolean, default=False)
    biased_features = Column(ARRAY(String))  # Features showing bias in explanations
    explanation_fairness_score = Column(DECIMAL(5,4))
    
    # Key findings
    key_insights = Column(ARRAY(String))
    concerning_patterns = Column(ARRAY(String))
    recommendations = Column(ARRAY(String))
    
    # Technical details
    methodology = Column(Text)
    limitations = Column(Text)
    data_sources = Column(JSON)
    
    # Audit trail
    created_by = Column(PG_UUID(as_uuid=True), ForeignKey('users.id'))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    model = relationship("Model", back_populates="explainability_reports")