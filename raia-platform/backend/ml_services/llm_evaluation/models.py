# LLM Evaluation and RAG Testing - Database Models
from sqlalchemy import Column, String, DateTime, Text, Boolean, DECIMAL, Integer, JSON, UUID, ForeignKey, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, ARRAY
import uuid
from datetime import datetime

Base = declarative_base()

class LLMModel(Base):
    """LLM model registry for language models and RAG systems"""
    __tablename__ = "llm_models"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, unique=True)
    display_name = Column(String(255), nullable=False)
    description = Column(Text)
    
    # Model classification
    model_type = Column(String(50), nullable=False)  # 'language_model', 'rag_system', 'chat_bot', 'code_generator'
    model_architecture = Column(String(100))  # 'transformer', 'gpt', 'bert', 'llama', 'claude'
    model_size = Column(String(50))  # '7B', '13B', '70B', 'large', 'xl'
    
    # Model configuration
    provider = Column(String(100))  # 'openai', 'anthropic', 'huggingface', 'local'
    model_version = Column(String(50))  # 'gpt-4', 'claude-3', 'llama-2-70b'
    api_endpoint = Column(String(500))
    model_parameters = Column(JSON)  # temperature, max_tokens, top_p, etc.
    
    # RAG-specific configuration
    is_rag_system = Column(Boolean, default=False)
    knowledge_base_id = Column(PG_UUID(as_uuid=True), ForeignKey('knowledge_bases.id'))
    retrieval_strategy = Column(String(100))  # 'semantic_search', 'hybrid', 'keyword'
    embedding_model = Column(String(255))
    chunk_size = Column(Integer)
    chunk_overlap = Column(Integer)
    retrieval_k = Column(Integer)  # Number of documents to retrieve
    
    # Performance characteristics
    context_length = Column(Integer)  # Maximum context window
    avg_latency_ms = Column(Integer)
    cost_per_1k_tokens = Column(DECIMAL(10,6))
    throughput_tokens_per_second = Column(Integer)
    
    # Status and metadata
    status = Column(String(50), default='active')  # 'active', 'deprecated', 'experimental'
    supported_languages = Column(ARRAY(String))
    capabilities = Column(ARRAY(String))  # 'text_generation', 'question_answering', 'code_generation'
    tags = Column(ARRAY(String))
    metadata = Column(JSON)
    
    # Audit fields
    created_by = Column(PG_UUID(as_uuid=True), ForeignKey('users.id'))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    evaluations = relationship("LLMEvaluation", back_populates="llm_model")
    rag_evaluations = relationship("RAGEvaluation", back_populates="llm_model")

class KnowledgeBase(Base):
    """Knowledge base for RAG systems"""
    __tablename__ = "knowledge_bases"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    
    # Knowledge base configuration
    kb_type = Column(String(50))  # 'document_store', 'vector_db', 'graph_db'
    vector_store_type = Column(String(100))  # 'faiss', 'pinecone', 'chroma', 'weaviate'
    embedding_model = Column(String(255))
    embedding_dimension = Column(Integer)
    
    # Content information
    total_documents = Column(Integer, default=0)
    total_chunks = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    document_types = Column(ARRAY(String))  # 'pdf', 'txt', 'html', 'markdown'
    languages = Column(ARRAY(String))
    
    # Index configuration
    index_type = Column(String(50))  # 'flat', 'ivf', 'hnsw'
    similarity_metric = Column(String(50))  # 'cosine', 'euclidean', 'dot_product'
    
    # Performance metrics
    avg_retrieval_time_ms = Column(Integer)
    index_size_mb = Column(DECIMAL(10,2))
    last_updated = Column(DateTime)
    
    # Audit fields
    created_by = Column(PG_UUID(as_uuid=True), ForeignKey('users.id'))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class LLMEvaluation(Base):
    """LLM evaluation results and metrics"""
    __tablename__ = "llm_evaluations"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    llm_model_id = Column(PG_UUID(as_uuid=True), ForeignKey('llm_models.id'), nullable=False)
    
    # Evaluation configuration
    evaluation_name = Column(String(255), nullable=False)
    evaluation_type = Column(String(100))  # 'benchmark', 'custom', 'safety', 'performance'
    evaluation_suite = Column(String(100))  # 'hellaswag', 'mmlu', 'gsm8k', 'truthfulqa'
    dataset_name = Column(String(255))
    dataset_version = Column(String(50))
    
    # Evaluation scope
    task_type = Column(String(100))  # 'text_generation', 'question_answering', 'summarization', 'classification'
    evaluation_prompt_template = Column(Text)
    num_samples = Column(Integer)
    sample_selection_strategy = Column(String(100))  # 'random', 'stratified', 'balanced'
    
    # Execution details
    status = Column(String(50), default='pending')  # 'pending', 'running', 'completed', 'failed'
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    duration_minutes = Column(Integer)
    
    # Core performance metrics
    overall_score = Column(DECIMAL(8,6))  # 0-1 scale
    accuracy = Column(DECIMAL(8,6))
    bleu_score = Column(DECIMAL(8,6))
    rouge_1 = Column(DECIMAL(8,6))
    rouge_2 = Column(DECIMAL(8,6))
    rouge_l = Column(DECIMAL(8,6))
    bertscore_precision = Column(DECIMAL(8,6))
    bertscore_recall = Column(DECIMAL(8,6))
    bertscore_f1 = Column(DECIMAL(8,6))
    
    # Semantic similarity metrics
    semantic_similarity = Column(DECIMAL(8,6))
    embedding_similarity = Column(DECIMAL(8,6))
    
    # Reasoning and logic metrics
    logical_consistency = Column(DECIMAL(8,6))
    factual_accuracy = Column(DECIMAL(8,6))
    reasoning_quality = Column(DECIMAL(8,6))
    
    # Language quality metrics
    fluency_score = Column(DECIMAL(8,6))
    coherence_score = Column(DECIMAL(8,6))
    relevance_score = Column(DECIMAL(8,6))
    
    # Safety and bias metrics
    toxicity_score = Column(DECIMAL(8,6))
    bias_score = Column(DECIMAL(8,6))
    hallucination_rate = Column(DECIMAL(8,6))
    refusal_rate = Column(DECIMAL(8,6))  # Rate of refusing harmful requests
    
    # Performance metrics
    avg_response_time_ms = Column(Integer)
    total_tokens_generated = Column(Integer)
    avg_tokens_per_response = Column(Integer)
    token_efficiency = Column(DECIMAL(8,6))  # Quality per token
    
    # Cost metrics
    total_cost_usd = Column(DECIMAL(10,4))
    cost_per_sample = Column(DECIMAL(10,6))
    cost_per_token = Column(DECIMAL(12,8))
    
    # Detailed results
    sample_level_results = Column(JSON)  # Individual sample results
    error_analysis = Column(JSON)  # Common error patterns
    performance_breakdown = Column(JSON)  # Performance by category/topic
    
    # Statistical analysis
    confidence_intervals = Column(JSON)  # 95% confidence intervals for metrics
    statistical_significance = Column(Boolean)
    p_values = Column(JSON)
    
    # Human evaluation (if applicable)
    human_evaluation_included = Column(Boolean, default=False)
    human_evaluators_count = Column(Integer)
    inter_annotator_agreement = Column(DECIMAL(6,4))
    human_preference_rate = Column(DECIMAL(6,4))  # Rate of human preference over baseline
    
    # Comparison and benchmarking
    baseline_model = Column(String(255))
    comparison_results = Column(JSON)
    ranking_position = Column(Integer)  # Position in benchmark leaderboard
    
    # Configuration and reproducibility
    evaluation_config = Column(JSON)  # All evaluation parameters
    random_seed = Column(Integer)
    model_parameters_used = Column(JSON)
    
    # Audit fields
    created_by = Column(PG_UUID(as_uuid=True), ForeignKey('users.id'))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    llm_model = relationship("LLMModel", back_populates="evaluations")

class RAGEvaluation(Base):
    """RAG system evaluation results and metrics"""
    __tablename__ = "rag_evaluations"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    llm_model_id = Column(PG_UUID(as_uuid=True), ForeignKey('llm_models.id'), nullable=False)
    knowledge_base_id = Column(PG_UUID(as_uuid=True), ForeignKey('knowledge_bases.id'))
    
    # Evaluation configuration
    evaluation_name = Column(String(255), nullable=False)
    evaluation_type = Column(String(100))  # 'end_to_end', 'retrieval_only', 'generation_only'
    dataset_name = Column(String(255))
    evaluation_questions = Column(JSON)  # Questions used for evaluation
    
    # Execution details
    status = Column(String(50), default='pending')
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    duration_minutes = Column(Integer)
    num_questions = Column(Integer)
    
    # Retrieval metrics
    retrieval_precision_at_k = Column(JSON)  # P@1, P@3, P@5, P@10
    retrieval_recall_at_k = Column(JSON)  # R@1, R@3, R@5, R@10
    retrieval_map = Column(DECIMAL(8,6))  # Mean Average Precision
    retrieval_mrr = Column(DECIMAL(8,6))  # Mean Reciprocal Rank
    retrieval_ndcg = Column(DECIMAL(8,6))  # Normalized Discounted Cumulative Gain
    
    # Retrieval quality
    context_relevance = Column(DECIMAL(8,6))  # Relevance of retrieved context
    context_precision = Column(DECIMAL(8,6))  # Precision of retrieved context
    context_recall = Column(DECIMAL(8,6))  # Recall of retrieved context
    context_coverage = Column(DECIMAL(8,6))  # How well context covers the question
    
    # Generation metrics (conditioned on retrieved context)
    answer_accuracy = Column(DECIMAL(8,6))
    answer_completeness = Column(DECIMAL(8,6))
    answer_consistency = Column(DECIMAL(8,6))  # Consistency with retrieved context
    answer_conciseness = Column(DECIMAL(8,6))
    
    # Factual accuracy and groundedness
    factual_correctness = Column(DECIMAL(8,6))
    groundedness_score = Column(DECIMAL(8,6))  # How well grounded in retrieved docs
    citation_accuracy = Column(DECIMAL(8,6))  # Accuracy of citations/references
    hallucination_rate = Column(DECIMAL(8,6))  # Rate of hallucinated information
    
    # End-to-end RAG metrics
    overall_rag_score = Column(DECIMAL(8,6))
    faithfulness = Column(DECIMAL(8,6))  # Generated answer faithful to context
    answer_relevance = Column(DECIMAL(8,6))  # Answer relevance to question
    context_utilization = Column(DECIMAL(8,6))  # How well the context was used
    
    # Semantic evaluation
    semantic_answer_similarity = Column(DECIMAL(8,6))  # Similarity to reference answers
    embedding_similarity = Column(DECIMAL(8,6))
    
    # Robustness metrics
    consistency_across_rephrasing = Column(DECIMAL(8,6))  # Consistent answers for rephrased questions
    sensitivity_to_context_order = Column(DECIMAL(8,6))  # Sensitivity to document order
    noise_robustness = Column(DECIMAL(8,6))  # Robustness to irrelevant context
    
    # Performance metrics
    avg_retrieval_time_ms = Column(Integer)
    avg_generation_time_ms = Column(Integer)
    avg_end_to_end_time_ms = Column(Integer)
    
    # Cost metrics
    total_cost_usd = Column(DECIMAL(10,4))
    cost_per_question = Column(DECIMAL(10,6))
    retrieval_cost = Column(DECIMAL(10,4))
    generation_cost = Column(DECIMAL(10,4))
    
    # Detailed analysis
    question_level_results = Column(JSON)  # Results for each question
    retrieval_analysis = Column(JSON)  # Analysis of retrieved documents
    error_analysis = Column(JSON)  # Common error patterns
    failure_cases = Column(JSON)  # Detailed analysis of failure cases
    
    # Human evaluation
    human_evaluation_included = Column(Boolean, default=False)
    human_relevance_rating = Column(DECIMAL(6,4))
    human_accuracy_rating = Column(DECIMAL(6,4))
    human_preference_vs_baseline = Column(DECIMAL(6,4))
    
    # Configuration
    retrieval_config = Column(JSON)  # Retrieval parameters used
    generation_config = Column(JSON)  # Generation parameters used
    evaluation_config = Column(JSON)  # Full evaluation configuration
    
    # Audit fields
    created_by = Column(PG_UUID(as_uuid=True), ForeignKey('users.id'))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    llm_model = relationship("LLMModel", back_populates="rag_evaluations")
    knowledge_base = relationship("KnowledgeBase")

class LLMSafetyEvaluation(Base):
    """Safety and alignment evaluation for LLMs"""
    __tablename__ = "llm_safety_evaluations"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    llm_model_id = Column(PG_UUID(as_uuid=True), ForeignKey('llm_models.id'), nullable=False)
    
    # Evaluation configuration
    evaluation_name = Column(String(255), nullable=False)
    safety_categories = Column(ARRAY(String))  # 'toxicity', 'bias', 'harmful_content', 'privacy'
    evaluation_framework = Column(String(100))  # 'custom', 'anthropic_constitutional', 'openai_moderation'
    
    # Execution details
    status = Column(String(50), default='pending')
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    num_test_cases = Column(Integer)
    
    # Toxicity metrics
    toxicity_detection_rate = Column(DECIMAL(8,6))
    toxicity_generation_rate = Column(DECIMAL(8,6))
    severe_toxicity_rate = Column(DECIMAL(8,6))
    identity_attack_rate = Column(DECIMAL(8,6))
    insult_rate = Column(DECIMAL(8,6))
    profanity_rate = Column(DECIMAL(8,6))
    threat_rate = Column(DECIMAL(8,6))
    
    # Bias metrics
    gender_bias_score = Column(DECIMAL(8,6))
    racial_bias_score = Column(DECIMAL(8,6))
    religious_bias_score = Column(DECIMAL(8,6))
    political_bias_score = Column(DECIMAL(8,6))
    age_bias_score = Column(DECIMAL(8,6))
    overall_bias_score = Column(DECIMAL(8,6))
    
    # Harmful content metrics
    violence_content_rate = Column(DECIMAL(8,6))
    self_harm_content_rate = Column(DECIMAL(8,6))
    sexual_content_rate = Column(DECIMAL(8,6))
    illegal_activity_rate = Column(DECIMAL(8,6))
    misinformation_rate = Column(DECIMAL(8,6))
    
    # Privacy and security
    pii_leakage_rate = Column(DECIMAL(8,6))  # Personal Identifiable Information
    sensitive_info_exposure = Column(DECIMAL(8,6))
    prompt_injection_susceptibility = Column(DECIMAL(8,6))
    jailbreaking_success_rate = Column(DECIMAL(8,6))
    
    # Alignment metrics
    instruction_following_rate = Column(DECIMAL(8,6))
    refusal_appropriateness = Column(DECIMAL(8,6))  # Appropriate refusal of harmful requests
    value_alignment_score = Column(DECIMAL(8,6))
    truthfulness_score = Column(DECIMAL(8,6))
    
    # Robustness
    adversarial_robustness = Column(DECIMAL(8,6))
    prompt_robustness = Column(DECIMAL(8,6))
    consistency_under_perturbation = Column(DECIMAL(8,6))
    
    # Overall safety score
    overall_safety_score = Column(DECIMAL(8,6))
    safety_grade = Column(String(10))  # 'A+', 'A', 'B', 'C', 'D', 'F'
    
    # Detailed results
    category_breakdown = Column(JSON)  # Breakdown by safety category
    test_case_results = Column(JSON)  # Individual test case results
    violation_examples = Column(JSON)  # Examples of safety violations
    
    # Risk assessment
    deployment_risk_level = Column(String(20))  # 'low', 'medium', 'high', 'critical'
    recommended_mitigations = Column(ARRAY(String))
    usage_restrictions = Column(ARRAY(String))
    
    # Audit fields
    created_by = Column(PG_UUID(as_uuid=True), ForeignKey('users.id'))
    created_at = Column(DateTime, default=datetime.utcnow)

class LLMBenchmarkResult(Base):
    """Standardized benchmark results for LLM comparison"""
    __tablename__ = "llm_benchmark_results"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    llm_model_id = Column(PG_UUID(as_uuid=True), ForeignKey('llm_models.id'), nullable=False)
    
    # Benchmark identification
    benchmark_name = Column(String(100), nullable=False)  # 'MMLU', 'HellaSwag', 'GSM8K', 'TruthfulQA'
    benchmark_version = Column(String(50))
    benchmark_category = Column(String(100))  # 'knowledge', 'reasoning', 'safety', 'coding'
    
    # Results
    score = Column(DECIMAL(8,6), nullable=False)
    percentile_rank = Column(DECIMAL(5,2))  # Percentile among all models
    total_questions = Column(Integer)
    correct_answers = Column(Integer)
    
    # Performance breakdown
    category_scores = Column(JSON)  # Scores by subject/category
    difficulty_scores = Column(JSON)  # Scores by difficulty level
    
    # Execution details
    evaluation_date = Column(DateTime, default=datetime.utcnow)
    evaluation_config = Column(JSON)
    shots = Column(Integer)  # Few-shot examples used (0, 1, 5, etc.)
    
    # Metadata
    notes = Column(Text)
    verified = Column(Boolean, default=False)  # Whether result is officially verified
    
    # Audit fields
    created_by = Column(PG_UUID(as_uuid=True), ForeignKey('users.id'))
    created_at = Column(DateTime, default=datetime.utcnow)

class PromptTemplate(Base):
    """Prompt templates for consistent LLM evaluation"""
    __tablename__ = "prompt_templates"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    
    # Template configuration
    template_type = Column(String(100))  # 'evaluation', 'safety_test', 'benchmark', 'custom'
    task_type = Column(String(100))  # 'classification', 'generation', 'qa', 'summarization'
    
    # Template content
    system_prompt = Column(Text)
    user_prompt_template = Column(Text, nullable=False)
    few_shot_examples = Column(JSON)  # Example input/output pairs
    
    # Parameters
    recommended_parameters = Column(JSON)  # temperature, max_tokens, etc.
    supported_models = Column(ARRAY(String))  # Models this template works well with
    
    # Usage tracking
    usage_count = Column(Integer, default=0)
    success_rate = Column(DECIMAL(6,4))  # Success rate when used
    avg_score = Column(DECIMAL(8,6))  # Average score achieved
    
    # Validation
    validated = Column(Boolean, default=False)
    validation_results = Column(JSON)
    
    # Versioning
    version = Column(String(20), default='1.0')
    parent_template_id = Column(PG_UUID(as_uuid=True), ForeignKey('prompt_templates.id'))
    
    # Metadata
    tags = Column(ARRAY(String))
    language = Column(String(10), default='en')
    
    # Audit fields
    created_by = Column(PG_UUID(as_uuid=True), ForeignKey('users.id'))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    parent_template = relationship("PromptTemplate", remote_side=[id])

class EvaluationDataset(Base):
    """Datasets used for LLM and RAG evaluation"""
    __tablename__ = "evaluation_datasets"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    
    # Dataset configuration
    dataset_type = Column(String(100))  # 'benchmark', 'custom', 'synthetic', 'human_curated'
    task_type = Column(String(100))  # 'qa', 'summarization', 'classification', 'generation'
    domain = Column(String(100))  # 'general', 'medical', 'legal', 'technical', 'conversational'
    
    # Content information
    total_samples = Column(Integer, nullable=False)
    train_samples = Column(Integer)
    validation_samples = Column(Integer)
    test_samples = Column(Integer)
    
    # Data characteristics
    languages = Column(ARRAY(String))
    sample_format = Column(String(50))  # 'json', 'csv', 'jsonl', 'parquet'
    has_reference_answers = Column(Boolean, default=False)
    has_human_annotations = Column(Boolean, default=False)
    
    # Quality metrics
    annotation_quality_score = Column(DECIMAL(6,4))
    inter_annotator_agreement = Column(DECIMAL(6,4))
    data_freshness_date = Column(DateTime)  # When data was last updated
    
    # Storage and access
    storage_location = Column(String(500))  # Path or URL to dataset
    access_level = Column(String(50))  # 'public', 'private', 'restricted'
    license_type = Column(String(100))
    
    # Usage tracking
    usage_count = Column(Integer, default=0)
    last_used = Column(DateTime)
    
    # Validation
    validated = Column(Boolean, default=False)
    validation_results = Column(JSON)
    
    # Metadata
    tags = Column(ARRAY(String))
    metadata = Column(JSON)
    
    # Audit fields
    created_by = Column(PG_UUID(as_uuid=True), ForeignKey('users.id'))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)