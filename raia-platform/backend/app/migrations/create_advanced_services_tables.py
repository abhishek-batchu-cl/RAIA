"""
Database Migration: Create Advanced Services Tables
Creates all database tables for the new advanced AI evaluation services
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
import uuid

# Migration metadata
revision = '003_advanced_services'
down_revision = '002_base_platform'
branch_labels = None
depends_on = None

def upgrade() -> None:
    """Create all advanced services tables"""
    
    # RAG Systems and Evaluations
    create_rag_systems_table()
    create_rag_evaluations_table()
    
    # LLM Models and Evaluations  
    create_llm_models_table()
    create_llm_evaluations_table()
    
    # Agent Evaluation Systems
    create_agent_evaluation_systems_table()
    create_agent_evaluation_results_table()
    
    # Model Statistics
    create_model_statistics_sessions_table()
    
    # What-If Analysis
    create_what_if_analysis_sessions_table()
    
    # Advanced Explainability
    create_advanced_explanation_sessions_table()
    
    # WebSocket Management
    create_websocket_connections_table()
    create_websocket_events_table()
    
    # Data Export
    create_data_export_jobs_table()
    
    # Cache Management
    create_cache_entries_table()
    
    # Audit Trails
    create_audit_logs_table()
    
    # Create indexes for performance
    create_advanced_services_indexes()

def downgrade() -> None:
    """Drop all advanced services tables"""
    
    # Drop tables in reverse order to handle foreign key constraints
    op.drop_table('audit_logs')
    op.drop_table('cache_entries')
    op.drop_table('data_export_jobs')
    op.drop_table('websocket_events')
    op.drop_table('websocket_connections')
    op.drop_table('advanced_explanation_sessions')
    op.drop_table('what_if_analysis_sessions')
    op.drop_table('model_statistics_sessions')
    op.drop_table('agent_evaluation_results')
    op.drop_table('agent_evaluation_systems')
    op.drop_table('llm_evaluations')
    op.drop_table('llm_models')
    op.drop_table('rag_evaluations')
    op.drop_table('rag_systems')

def create_rag_systems_table():
    """Create RAG systems table"""
    op.create_table(
        'rag_systems',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('system_id', sa.String(255), nullable=False, index=True),
        sa.Column('system_name', sa.String(255), nullable=False),
        sa.Column('system_type', sa.String(100), default='retrieval_augmented'),
        sa.Column('description', sa.Text),
        sa.Column('configuration', postgresql.JSON, default={}),
        sa.Column('metadata', postgresql.JSON, default={}),
        sa.Column('created_at', sa.DateTime, default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, default=sa.func.now(), onupdate=sa.func.now()),
        sa.Column('organization_id', postgresql.UUID(as_uuid=True), 
                 sa.ForeignKey('organizations.id'), nullable=False),
        sa.UniqueConstraint('system_id')
    )

def create_rag_evaluations_table():
    """Create RAG evaluations table"""
    
    # Create enum type
    rag_status_enum = postgresql.ENUM(
        'pending', 'running', 'completed', 'failed',
        name='ragevaluationstatusenum'
    )
    rag_status_enum.create(op.get_bind())
    
    op.create_table(
        'rag_evaluations',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('evaluation_name', sa.String(255), nullable=False),
        sa.Column('status', rag_status_enum, default='pending'),
        sa.Column('queries_count', sa.Integer, default=0),
        sa.Column('completed_queries', sa.Integer, default=0),
        sa.Column('retrieval_precision', sa.Float),
        sa.Column('retrieval_recall', sa.Float),
        sa.Column('retrieval_f1', sa.Float),
        sa.Column('generation_faithfulness', sa.Float),
        sa.Column('generation_relevancy', sa.Float),
        sa.Column('generation_coherence', sa.Float),
        sa.Column('overall_score', sa.Float),
        sa.Column('evaluation_results', postgresql.JSON, default={}),
        sa.Column('metrics_summary', postgresql.JSON, default={}),
        sa.Column('configuration', postgresql.JSON, default={}),
        sa.Column('started_at', sa.DateTime),
        sa.Column('completed_at', sa.DateTime),
        sa.Column('created_at', sa.DateTime, default=sa.func.now()),
        sa.Column('organization_id', postgresql.UUID(as_uuid=True), 
                 sa.ForeignKey('organizations.id'), nullable=False),
        sa.Column('rag_system_id', postgresql.UUID(as_uuid=True), 
                 sa.ForeignKey('rag_systems.id'), nullable=False),
        sa.Column('created_by', postgresql.UUID(as_uuid=True), 
                 sa.ForeignKey('users.id'), nullable=False)
    )

def create_llm_models_table():
    """Create LLM models table"""
    op.create_table(
        'llm_models',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('model_id', sa.String(255), nullable=False, index=True),
        sa.Column('model_name', sa.String(255), nullable=False),
        sa.Column('model_type', sa.String(100), default='text_generation'),
        sa.Column('provider', sa.String(100), nullable=False),
        sa.Column('description', sa.Text),
        sa.Column('configuration', postgresql.JSON, default={}),
        sa.Column('metadata', postgresql.JSON, default={}),
        sa.Column('created_at', sa.DateTime, default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, default=sa.func.now(), onupdate=sa.func.now()),
        sa.Column('organization_id', postgresql.UUID(as_uuid=True), 
                 sa.ForeignKey('organizations.id'), nullable=False),
        sa.UniqueConstraint('model_id')
    )

def create_llm_evaluations_table():
    """Create LLM evaluations table"""
    
    # Reuse existing evaluation status enum
    op.create_table(
        'llm_evaluations',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('evaluation_name', sa.String(255), nullable=False),
        sa.Column('status', postgresql.ENUM(name='evaluationstatusenum'), default='pending'),
        sa.Column('evaluation_framework', sa.String(100), nullable=False),
        sa.Column('task_type', sa.String(100), nullable=False),
        sa.Column('inputs_count', sa.Integer, default=0),
        sa.Column('completed_evaluations', sa.Integer, default=0),
        sa.Column('factual_accuracy', sa.Float),
        sa.Column('completeness', sa.Float),
        sa.Column('relevancy', sa.Float),
        sa.Column('logical_consistency', sa.Float),
        sa.Column('fluency', sa.Float),
        sa.Column('grammar', sa.Float),
        sa.Column('clarity', sa.Float),
        sa.Column('conciseness', sa.Float),
        sa.Column('bert_score', sa.Float),
        sa.Column('semantic_coherence', sa.Float),
        sa.Column('toxicity_score', sa.Float),
        sa.Column('bias_score', sa.Float),
        sa.Column('harmful_content_score', sa.Float),
        sa.Column('overall_score', sa.Float),
        sa.Column('evaluation_results', postgresql.JSON, default={}),
        sa.Column('metrics_breakdown', postgresql.JSON, default={}),
        sa.Column('configuration', postgresql.JSON, default={}),
        sa.Column('started_at', sa.DateTime),
        sa.Column('completed_at', sa.DateTime),
        sa.Column('created_at', sa.DateTime, default=sa.func.now()),
        sa.Column('organization_id', postgresql.UUID(as_uuid=True), 
                 sa.ForeignKey('organizations.id'), nullable=False),
        sa.Column('llm_model_id', postgresql.UUID(as_uuid=True), 
                 sa.ForeignKey('llm_models.id'), nullable=False),
        sa.Column('created_by', postgresql.UUID(as_uuid=True), 
                 sa.ForeignKey('users.id'), nullable=False)
    )

def create_agent_evaluation_systems_table():
    """Create agent evaluation systems table"""
    
    # Create agent type enum
    agent_type_enum = postgresql.ENUM(
        'conversational', 'rag', 'tool_using', 'code_generation', 'multimodal',
        name='agenttype'
    )
    agent_type_enum.create(op.get_bind())
    
    op.create_table(
        'agent_evaluation_systems',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('agent_id', sa.String(255), nullable=False, index=True),
        sa.Column('agent_name', sa.String(255), nullable=False),
        sa.Column('agent_type', agent_type_enum, nullable=False),
        sa.Column('description', sa.Text),
        sa.Column('configuration', postgresql.JSON, default={}),
        sa.Column('metadata', postgresql.JSON, default={}),
        sa.Column('created_at', sa.DateTime, default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, default=sa.func.now(), onupdate=sa.func.now()),
        sa.Column('organization_id', postgresql.UUID(as_uuid=True), 
                 sa.ForeignKey('organizations.id'), nullable=False),
        sa.UniqueConstraint('agent_id')
    )

def create_agent_evaluation_results_table():
    """Create agent evaluation results table"""
    op.create_table(
        'agent_evaluation_results',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('evaluation_name', sa.String(255), nullable=False),
        sa.Column('status', postgresql.ENUM(name='evaluationstatusenum'), default='pending'),
        sa.Column('task_completion_rate', sa.Float),
        sa.Column('response_accuracy', sa.Float),
        sa.Column('response_relevance', sa.Float),
        sa.Column('response_helpfulness', sa.Float),
        sa.Column('conversation_coherence', sa.Float),
        sa.Column('context_awareness', sa.Float),
        sa.Column('response_consistency', sa.Float),
        sa.Column('avg_response_time', sa.Float),
        sa.Column('error_rate', sa.Float),
        sa.Column('resource_efficiency', sa.Float),
        sa.Column('overall_performance_score', sa.Float),
        sa.Column('user_satisfaction_score', sa.Float),
        sa.Column('evaluation_results', postgresql.JSON, default={}),
        sa.Column('performance_breakdown', postgresql.JSON, default={}),
        sa.Column('configuration', postgresql.JSON, default={}),
        sa.Column('started_at', sa.DateTime),
        sa.Column('completed_at', sa.DateTime),
        sa.Column('created_at', sa.DateTime, default=sa.func.now()),
        sa.Column('organization_id', postgresql.UUID(as_uuid=True), 
                 sa.ForeignKey('organizations.id'), nullable=False),
        sa.Column('agent_system_id', postgresql.UUID(as_uuid=True), 
                 sa.ForeignKey('agent_evaluation_systems.id'), nullable=False),
        sa.Column('created_by', postgresql.UUID(as_uuid=True), 
                 sa.ForeignKey('users.id'), nullable=False)
    )

def create_model_statistics_sessions_table():
    """Create model statistics sessions table"""
    op.create_table(
        'model_statistics_sessions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('session_name', sa.String(255), nullable=False),
        sa.Column('model_type', sa.String(50), nullable=False),
        sa.Column('description', sa.Text),
        sa.Column('test_samples_count', sa.Integer),
        sa.Column('training_samples_count', sa.Integer),
        sa.Column('feature_count', sa.Integer),
        sa.Column('class_count', sa.Integer),
        
        # Classification metrics
        sa.Column('accuracy', sa.Float),
        sa.Column('precision_macro', sa.Float),
        sa.Column('precision_micro', sa.Float),
        sa.Column('precision_weighted', sa.Float),
        sa.Column('recall_macro', sa.Float),
        sa.Column('recall_micro', sa.Float),
        sa.Column('recall_weighted', sa.Float),
        sa.Column('f1_macro', sa.Float),
        sa.Column('f1_micro', sa.Float),
        sa.Column('f1_weighted', sa.Float),
        sa.Column('roc_auc_macro', sa.Float),
        sa.Column('matthews_corrcoef', sa.Float),
        sa.Column('cohen_kappa', sa.Float),
        
        # Regression metrics
        sa.Column('mae', sa.Float),
        sa.Column('mse', sa.Float),
        sa.Column('rmse', sa.Float),
        sa.Column('r2_score', sa.Float),
        sa.Column('adjusted_r2', sa.Float),
        sa.Column('explained_variance', sa.Float),
        
        sa.Column('cross_validation_results', postgresql.JSON, default={}),
        sa.Column('learning_curves_data', postgresql.JSON, default={}),
        sa.Column('statistical_tests', postgresql.JSON, default={}),
        sa.Column('detailed_results', postgresql.JSON, default={}),
        sa.Column('configuration', postgresql.JSON, default={}),
        sa.Column('created_at', sa.DateTime, default=sa.func.now()),
        sa.Column('completed_at', sa.DateTime),
        sa.Column('organization_id', postgresql.UUID(as_uuid=True), 
                 sa.ForeignKey('organizations.id'), nullable=False),
        sa.Column('model_id', postgresql.UUID(as_uuid=True), 
                 sa.ForeignKey('models.id'), nullable=False),
        sa.Column('created_by', postgresql.UUID(as_uuid=True), 
                 sa.ForeignKey('users.id'), nullable=False)
    )

def create_what_if_analysis_sessions_table():
    """Create what-if analysis sessions table"""
    op.create_table(
        'what_if_analysis_sessions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('session_name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text),
        sa.Column('analysis_type', sa.String(100), nullable=False),
        sa.Column('base_instance', postgresql.JSON, nullable=False),
        sa.Column('scenarios_data', postgresql.JSON, default=[]),
        sa.Column('scenario_results', postgresql.JSON, default=[]),
        sa.Column('feature_impact_analysis', postgresql.JSON, default={}),
        sa.Column('sensitivity_analysis', postgresql.JSON, default={}),
        sa.Column('optimization_results', postgresql.JSON, default={}),
        sa.Column('surrogate_tree_accuracy', sa.Float),
        sa.Column('decision_rules', postgresql.JSON, default=[]),
        sa.Column('tree_visualization_data', postgresql.JSON, default={}),
        sa.Column('configuration', postgresql.JSON, default={}),
        sa.Column('processing_time_ms', sa.Integer),
        sa.Column('created_at', sa.DateTime, default=sa.func.now()),
        sa.Column('completed_at', sa.DateTime),
        sa.Column('organization_id', postgresql.UUID(as_uuid=True), 
                 sa.ForeignKey('organizations.id'), nullable=False),
        sa.Column('model_id', postgresql.UUID(as_uuid=True), 
                 sa.ForeignKey('models.id'), nullable=False),
        sa.Column('created_by', postgresql.UUID(as_uuid=True), 
                 sa.ForeignKey('users.id'), nullable=False)
    )

def create_advanced_explanation_sessions_table():
    """Create advanced explanation sessions table"""
    op.create_table(
        'advanced_explanation_sessions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('session_name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text),
        sa.Column('explanation_method', sa.String(100), nullable=False),
        sa.Column('input_data', postgresql.JSON, nullable=False),
        sa.Column('feature_names', postgresql.JSON, default=[]),
        sa.Column('parameters', postgresql.JSON, default={}),
        sa.Column('explanation_results', postgresql.JSON, default={}),
        sa.Column('anchor_rules', postgresql.JSON, default=[]),
        sa.Column('anchor_precision', sa.Float),
        sa.Column('anchor_coverage', sa.Float),
        sa.Column('ale_plot_data', postgresql.JSON, default={}),
        sa.Column('prototype_data', postgresql.JSON, default=[]),
        sa.Column('similarity_scores', postgresql.JSON, default=[]),
        sa.Column('ice_lines', postgresql.JSON, default=[]),
        sa.Column('partial_dependence', postgresql.JSON, default=[]),
        sa.Column('counterfactuals', postgresql.JSON, default=[]),
        sa.Column('counterfactual_quality_metrics', postgresql.JSON, default={}),
        sa.Column('permutation_importance_scores', postgresql.JSON, default={}),
        sa.Column('eli5_analysis', postgresql.JSON, default={}),
        sa.Column('processing_time_ms', sa.Integer),
        sa.Column('configuration', postgresql.JSON, default={}),
        sa.Column('created_at', sa.DateTime, default=sa.func.now()),
        sa.Column('completed_at', sa.DateTime),
        sa.Column('organization_id', postgresql.UUID(as_uuid=True), 
                 sa.ForeignKey('organizations.id'), nullable=False),
        sa.Column('model_id', postgresql.UUID(as_uuid=True), 
                 sa.ForeignKey('models.id'), nullable=False),
        sa.Column('created_by', postgresql.UUID(as_uuid=True), 
                 sa.ForeignKey('users.id'), nullable=False)
    )

def create_websocket_connections_table():
    """Create WebSocket connections table"""
    op.create_table(
        'websocket_connections',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('connection_id', sa.String(255), nullable=False, index=True),
        sa.Column('user_agent', sa.String(500)),
        sa.Column('ip_address', sa.String(45)),
        sa.Column('subscriptions', postgresql.JSON, default=[]),
        sa.Column('is_active', sa.Boolean, default=True),
        sa.Column('last_activity_at', sa.DateTime, default=sa.func.now()),
        sa.Column('messages_sent', sa.Integer, default=0),
        sa.Column('messages_received', sa.Integer, default=0),
        sa.Column('connection_duration_seconds', sa.Integer, default=0),
        sa.Column('connected_at', sa.DateTime, default=sa.func.now()),
        sa.Column('disconnected_at', sa.DateTime),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), 
                 sa.ForeignKey('users.id'), nullable=False),
        sa.Column('organization_id', postgresql.UUID(as_uuid=True), 
                 sa.ForeignKey('organizations.id'), nullable=False),
        sa.UniqueConstraint('connection_id')
    )

def create_websocket_events_table():
    """Create WebSocket events table"""
    
    # Create WebSocket event type enum
    websocket_event_type_enum = postgresql.ENUM(
        'model_performance', 'system_health', 'evaluation_progress', 'alert_notification',
        'data_drift_detection', 'fairness_violation', 'anomaly_detection', 
        'batch_job_status', 'resource_usage', 'error_notification',
        name='websocketeventtypeenum'
    )
    websocket_event_type_enum.create(op.get_bind())
    
    op.create_table(
        'websocket_events',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('event_id', sa.String(255), nullable=False, index=True),
        sa.Column('event_type', websocket_event_type_enum, nullable=False),
        sa.Column('severity', sa.String(20), default='info'),
        sa.Column('event_data', postgresql.JSON, default={}),
        sa.Column('message', sa.Text),
        sa.Column('broadcast_to_all', sa.Boolean, default=False),
        sa.Column('target_user_ids', postgresql.JSON, default=[]),
        sa.Column('target_model_ids', postgresql.JSON, default=[]),
        sa.Column('total_recipients', sa.Integer, default=0),
        sa.Column('successful_deliveries', sa.Integer, default=0),
        sa.Column('failed_deliveries', sa.Integer, default=0),
        sa.Column('created_at', sa.DateTime, default=sa.func.now()),
        sa.Column('connection_id', postgresql.UUID(as_uuid=True), 
                 sa.ForeignKey('websocket_connections.id'), nullable=True),
        sa.Column('organization_id', postgresql.UUID(as_uuid=True), 
                 sa.ForeignKey('organizations.id'), nullable=False)
    )

def create_data_export_jobs_table():
    """Create data export jobs table"""
    
    # Create export job status enum
    export_status_enum = postgresql.ENUM(
        'pending', 'processing', 'completed', 'failed',
        name='exportjobstatusenum'
    )
    export_status_enum.create(op.get_bind())
    
    op.create_table(
        'data_export_jobs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('job_name', sa.String(255), nullable=False),
        sa.Column('export_type', sa.String(50), nullable=False),
        sa.Column('export_format', sa.String(20), nullable=False),
        sa.Column('status', export_status_enum, default='pending'),
        sa.Column('data_source', sa.String(100), nullable=False),
        sa.Column('source_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('export_filters', postgresql.JSON, default={}),
        sa.Column('export_options', postgresql.JSON, default={}),
        sa.Column('filename', sa.String(255)),
        sa.Column('file_path', sa.String(500)),
        sa.Column('file_size_bytes', sa.Integer),
        sa.Column('download_url', sa.String(500)),
        sa.Column('expires_at', sa.DateTime),
        sa.Column('total_records', sa.Integer, default=0),
        sa.Column('processed_records', sa.Integer, default=0),
        sa.Column('progress_percentage', sa.Float, default=0.0),
        sa.Column('processing_time_ms', sa.Integer),
        sa.Column('error_message', sa.Text),
        sa.Column('created_at', sa.DateTime, default=sa.func.now()),
        sa.Column('started_at', sa.DateTime),
        sa.Column('completed_at', sa.DateTime),
        sa.Column('organization_id', postgresql.UUID(as_uuid=True), 
                 sa.ForeignKey('organizations.id'), nullable=False),
        sa.Column('requested_by', postgresql.UUID(as_uuid=True), 
                 sa.ForeignKey('users.id'), nullable=False)
    )

def create_cache_entries_table():
    """Create cache entries table"""
    op.create_table(
        'cache_entries',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('cache_key', sa.String(255), nullable=False, index=True),
        sa.Column('cache_category', sa.String(100), nullable=False, index=True),
        sa.Column('cached_data', postgresql.JSON, nullable=False),
        sa.Column('data_size_bytes', sa.Integer),
        sa.Column('data_hash', sa.String(64), index=True),
        sa.Column('ttl_seconds', sa.Integer, default=3600),
        sa.Column('access_count', sa.Integer, default=0),
        sa.Column('last_accessed_at', sa.DateTime, default=sa.func.now()),
        sa.Column('created_at', sa.DateTime, default=sa.func.now()),
        sa.Column('expires_at', sa.DateTime, nullable=False),
        sa.Column('organization_id', postgresql.UUID(as_uuid=True), 
                 sa.ForeignKey('organizations.id'), nullable=False),
        sa.UniqueConstraint('cache_key')
    )

def create_audit_logs_table():
    """Create audit logs table"""
    
    # Create audit action enum
    audit_action_enum = postgresql.ENUM(
        'create', 'read', 'update', 'delete', 'execute', 'export', 'login', 'logout',
        name='auditactionenum'
    )
    audit_action_enum.create(op.get_bind())
    
    op.create_table(
        'audit_logs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('action', audit_action_enum, nullable=False),
        sa.Column('resource_type', sa.String(100), nullable=False),
        sa.Column('resource_id', sa.String(255), nullable=False, index=True),
        sa.Column('description', sa.Text, nullable=False),
        sa.Column('request_method', sa.String(10)),
        sa.Column('request_path', sa.String(500)),
        sa.Column('request_params', postgresql.JSON, default={}),
        sa.Column('request_body_hash', sa.String(64)),
        sa.Column('response_status', sa.Integer),
        sa.Column('processing_time_ms', sa.Integer),
        sa.Column('user_agent', sa.String(500)),
        sa.Column('ip_address', sa.String(45)),
        sa.Column('session_id', sa.String(255)),
        sa.Column('metadata', postgresql.JSON, default={}),
        sa.Column('tags', postgresql.JSON, default=[]),
        sa.Column('created_at', sa.DateTime, default=sa.func.now()),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), 
                 sa.ForeignKey('users.id'), nullable=True),
        sa.Column('organization_id', postgresql.UUID(as_uuid=True), 
                 sa.ForeignKey('organizations.id'), nullable=False)
    )

def create_advanced_services_indexes():
    """Create indexes for performance optimization"""
    
    # RAG Systems indexes
    op.create_index('idx_rag_systems_org_id', 'rag_systems', ['organization_id'])
    op.create_index('idx_rag_evaluations_system_id', 'rag_evaluations', ['rag_system_id'])
    op.create_index('idx_rag_evaluations_status', 'rag_evaluations', ['status'])
    
    # LLM Models indexes
    op.create_index('idx_llm_models_org_id', 'llm_models', ['organization_id'])
    op.create_index('idx_llm_evaluations_model_id', 'llm_evaluations', ['llm_model_id'])
    op.create_index('idx_llm_evaluations_status', 'llm_evaluations', ['status'])
    
    # Agent Evaluation indexes
    op.create_index('idx_agent_systems_org_id', 'agent_evaluation_systems', ['organization_id'])
    op.create_index('idx_agent_systems_type', 'agent_evaluation_systems', ['agent_type'])
    op.create_index('idx_agent_results_system_id', 'agent_evaluation_results', ['agent_system_id'])
    
    # Model Statistics indexes
    op.create_index('idx_model_stats_model_id', 'model_statistics_sessions', ['model_id'])
    op.create_index('idx_model_stats_type', 'model_statistics_sessions', ['model_type'])
    
    # What-If Analysis indexes
    op.create_index('idx_whatif_model_id', 'what_if_analysis_sessions', ['model_id'])
    op.create_index('idx_whatif_type', 'what_if_analysis_sessions', ['analysis_type'])
    
    # Advanced Explainability indexes
    op.create_index('idx_advanced_expl_model_id', 'advanced_explanation_sessions', ['model_id'])
    op.create_index('idx_advanced_expl_method', 'advanced_explanation_sessions', ['explanation_method'])
    
    # WebSocket indexes
    op.create_index('idx_websocket_user_id', 'websocket_connections', ['user_id'])
    op.create_index('idx_websocket_active', 'websocket_connections', ['is_active'])
    op.create_index('idx_websocket_events_type', 'websocket_events', ['event_type'])
    op.create_index('idx_websocket_events_created', 'websocket_events', ['created_at'])
    
    # Export Jobs indexes
    op.create_index('idx_export_jobs_status', 'data_export_jobs', ['status'])
    op.create_index('idx_export_jobs_source', 'data_export_jobs', ['data_source', 'source_id'])
    op.create_index('idx_export_jobs_user', 'data_export_jobs', ['requested_by'])
    
    # Cache indexes
    op.create_index('idx_cache_category', 'cache_entries', ['cache_category'])
    op.create_index('idx_cache_expires', 'cache_entries', ['expires_at'])
    
    # Audit Log indexes
    op.create_index('idx_audit_resource', 'audit_logs', ['resource_type', 'resource_id'])
    op.create_index('idx_audit_user', 'audit_logs', ['user_id'])
    op.create_index('idx_audit_created', 'audit_logs', ['created_at'])
    op.create_index('idx_audit_action', 'audit_logs', ['action'])
    op.create_index('idx_audit_org', 'audit_logs', ['organization_id'])