"""
RAIA Platform - Agent Evaluation Service
Comprehensive evaluation system for RAG agents with multiple metrics and analytics
"""

import asyncio
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import statistics
from dataclasses import dataclass

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_database
from app.services.rag_agent import get_rag_agent_manager
from app.models.schemas import (
    AgentConfiguration,
    AgentEvaluation, 
    EvaluationResult,
    AgentEvaluationCreate,
    AgentEvaluationResponse,
    EvaluationDataItem,
    EvaluationMetrics,
    EvaluationResultResponse,
    EvaluationStatusEnum,
    TaskStatus,
    ModelComparisonRequest,
    ModelComparisonResult,
    ModelComparisonResponse
)

logger = structlog.get_logger(__name__)


@dataclass
class EvaluationTask:
    """Represents a running evaluation task"""
    task_id: str
    status: str
    progress: float
    total_items: int
    completed_items: int
    result: Optional[Dict[str, Any]]
    error: Optional[str]
    created_at: datetime
    updated_at: datetime
    estimated_completion: Optional[datetime]


class MetricsCalculator:
    """Advanced metrics calculation for agent responses"""
    
    def __init__(self):
        self.logger = logger.bind(component="metrics_calculator")
    
    async def calculate_relevance(
        self, 
        question: str, 
        answer: str, 
        expected_answer: Optional[str] = None
    ) -> float:
        """
        Calculate relevance of answer to question
        Score: 0.0 to 5.0
        """
        try:
            if not answer or not answer.strip():
                return 0.0
            
            question_words = set(question.lower().split())
            answer_words = set(answer.lower().split())
            
            if not question_words:
                return 2.5  # Default moderate score
            
            # Calculate word overlap
            overlap = len(question_words.intersection(answer_words))
            overlap_ratio = overlap / len(question_words)
            
            # Basic heuristic scoring
            if overlap_ratio >= 0.7:
                return 4.5 + (overlap_ratio - 0.7) * (0.5 / 0.3)
            elif overlap_ratio >= 0.5:
                return 3.5 + (overlap_ratio - 0.5) * (1.0 / 0.2)
            elif overlap_ratio >= 0.3:
                return 2.5 + (overlap_ratio - 0.3) * (1.0 / 0.2)
            elif overlap_ratio >= 0.1:
                return 1.0 + (overlap_ratio - 0.1) * (1.5 / 0.2)
            else:
                return overlap_ratio * 10  # Scale to 0-1 range
            
        except Exception as e:
            self.logger.error("Failed to calculate relevance", error=str(e))
            return 2.5  # Default moderate score
    
    async def calculate_groundedness(
        self, 
        answer: str, 
        context: Optional[str] = None,
        sources: Optional[List[Dict[str, Any]]] = None
    ) -> float:
        """
        Calculate how well the answer is grounded in provided context
        Score: 0.0 to 5.0
        """
        try:
            if not answer or not answer.strip():
                return 0.0
            
            # If no context provided, check for hedging language
            if not context and not sources:
                return self._calculate_confidence_score(answer)
            
            # Extract context text
            context_text = ""
            if context:
                context_text += context + " "
            
            if sources:
                for source in sources:
                    if isinstance(source, dict) and source.get('document_chunk'):
                        context_text += source['document_chunk'] + " "
            
            if not context_text.strip():
                return self._calculate_confidence_score(answer)
            
            # Calculate overlap with context
            context_words = set(context_text.lower().split())
            answer_words = set(answer.lower().split())
            
            if not context_words or not answer_words:
                return 2.5
            
            # Calculate contextual grounding
            overlap = len(context_words.intersection(answer_words))
            overlap_ratio = overlap / len(answer_words)
            
            # Check for direct quotes or specific facts
            quote_indicators = ['"', "'", "according to", "states that", "shows that"]
            has_quotes = any(indicator in answer.lower() for indicator in quote_indicators)
            
            base_score = min(overlap_ratio * 5, 4.0)
            
            if has_quotes:
                base_score = min(base_score + 0.5, 5.0)
            
            return base_score
            
        except Exception as e:
            self.logger.error("Failed to calculate groundedness", error=str(e))
            return 2.5
    
    def _calculate_confidence_score(self, answer: str) -> float:
        """Calculate confidence based on hedging language"""
        hedging_phrases = [
            "might", "could", "possibly", "maybe", "perhaps", 
            "i think", "i believe", "it seems", "appears to be",
            "likely", "probably", "uncertain", "not sure"
        ]
        
        answer_lower = answer.lower()
        hedge_count = sum(1 for phrase in hedging_phrases if phrase in answer_lower)
        
        # More hedging = less grounded
        if hedge_count == 0:
            return 4.5
        elif hedge_count <= 2:
            return 3.5
        elif hedge_count <= 4:
            return 2.5
        else:
            return 1.5
    
    async def calculate_coherence(self, answer: str) -> float:
        """
        Calculate coherence and logical flow of the answer
        Score: 0.0 to 5.0
        """
        try:
            if not answer or not answer.strip():
                return 0.0
            
            # Basic structure checks
            sentences = [s.strip() for s in answer.split('.') if s.strip()]
            paragraphs = [p.strip() for p in answer.split('\n\n') if p.strip()]
            words = answer.split()
            
            score = 2.5  # Base score
            
            # Length appropriateness
            if 20 <= len(words) <= 200:
                score += 0.5
            elif 10 <= len(words) <= 300:
                score += 0.3
            
            # Sentence structure
            if 2 <= len(sentences) <= 10:
                score += 0.5
            elif len(sentences) == 1 and len(words) >= 15:
                score += 0.3
            
            # Paragraph structure for longer texts
            if len(words) > 100 and len(paragraphs) > 1:
                score += 0.3
            
            # Check for connector words (indicates logical flow)
            connectors = [
                "however", "therefore", "furthermore", "moreover", "additionally",
                "consequently", "nevertheless", "meanwhile", "similarly", "likewise",
                "in contrast", "on the other hand", "as a result", "in addition"
            ]
            
            answer_lower = answer.lower()
            connector_count = sum(1 for connector in connectors if connector in answer_lower)
            
            if connector_count > 0:
                score += min(connector_count * 0.2, 0.5)
            
            # Check for repetition (reduces coherence)
            unique_words = set(words)
            if len(words) > 0:
                repetition_ratio = 1 - (len(unique_words) / len(words))
                if repetition_ratio > 0.5:
                    score -= 1.0
                elif repetition_ratio > 0.3:
                    score -= 0.5
            
            return min(max(score, 0.0), 5.0)
            
        except Exception as e:
            self.logger.error("Failed to calculate coherence", error=str(e))
            return 2.5
    
    async def calculate_similarity(
        self, 
        expected_answer: Optional[str], 
        actual_answer: str
    ) -> float:
        """
        Calculate similarity to expected answer
        Score: 0.0 to 1.0 (different scale for compatibility)
        """
        try:
            if not expected_answer or not actual_answer:
                return 0.5  # Default when comparison not possible
            
            expected_words = set(expected_answer.lower().split())
            actual_words = set(actual_answer.lower().split())
            
            if not expected_words and not actual_words:
                return 1.0
            elif not expected_words or not actual_words:
                return 0.0
            
            # Calculate Jaccard similarity
            intersection = expected_words.intersection(actual_words)
            union = expected_words.union(actual_words)
            
            jaccard = len(intersection) / len(union) if union else 0.0
            
            # Also calculate overlap ratio from both perspectives
            expected_overlap = len(intersection) / len(expected_words) if expected_words else 0.0
            actual_overlap = len(intersection) / len(actual_words) if actual_words else 0.0
            
            # Combine metrics
            similarity = (jaccard + expected_overlap + actual_overlap) / 3
            
            return min(max(similarity, 0.0), 1.0)
            
        except Exception as e:
            self.logger.error("Failed to calculate similarity", error=str(e))
            return 0.5
    
    async def calculate_fluency(self, answer: str) -> float:
        """
        Calculate fluency and language quality
        Score: 0.0 to 5.0
        """
        try:
            if not answer or not answer.strip():
                return 0.0
            
            words = answer.split()
            score = 2.5  # Base score
            
            # Length check
            if len(words) < 3:
                return 1.0
            
            # Grammar checks (basic)
            has_capital_start = answer[0].isupper() if answer else False
            has_ending_punctuation = answer.strip()[-1] in '.!?' if answer.strip() else False
            
            if has_capital_start:
                score += 0.5
            if has_ending_punctuation:
                score += 0.5
            
            # Word length variety (indicates natural language)
            word_lengths = [len(word) for word in words]
            if len(word_lengths) > 1:
                avg_length = statistics.mean(word_lengths)
                if 3 <= avg_length <= 6:  # Natural average
                    score += 0.5
            
            # Check for very long or very short sentences (may indicate issues)
            sentences = [s.strip() for s in answer.split('.') if s.strip()]
            if sentences:
                sentence_lengths = [len(s.split()) for s in sentences]
                avg_sentence_length = statistics.mean(sentence_lengths)
                
                if 8 <= avg_sentence_length <= 25:  # Natural range
                    score += 0.5
                elif avg_sentence_length < 3 or avg_sentence_length > 40:
                    score -= 0.5
            
            # Check for common fluency issues
            issues = 0
            
            # Repeated words in sequence
            for i in range(len(words) - 1):
                if words[i].lower() == words[i + 1].lower():
                    issues += 1
            
            # Very short or very long words (potential errors)
            extreme_words = [w for w in words if len(w) < 2 or len(w) > 15]
            issues += len(extreme_words) * 0.1
            
            # Subtract for issues
            score -= min(issues * 0.2, 1.5)
            
            return min(max(score, 0.0), 5.0)
            
        except Exception as e:
            self.logger.error("Failed to calculate fluency", error=str(e))
            return 2.5
    
    async def calculate_accuracy(
        self, 
        question: str, 
        expected_answer: Optional[str], 
        actual_answer: str,
        context: Optional[str] = None
    ) -> float:
        """
        Calculate factual accuracy of the answer
        Score: 0.0 to 5.0
        """
        try:
            if not actual_answer or not actual_answer.strip():
                return 0.0
            
            # If we have expected answer, compare directly
            if expected_answer and expected_answer.strip():
                similarity = await self.calculate_similarity(expected_answer, actual_answer)
                # Convert similarity (0-1) to accuracy score (0-5)
                return similarity * 5.0
            
            # If no expected answer, use other heuristics
            score = 2.5  # Base score for reasonable-looking answers
            
            # Check for specific factual patterns
            factual_indicators = [
                r'\d+%',  # Percentages
                r'\d+\s*(years?|months?|days?)',  # Time periods
                r'\$\d+',  # Money amounts
                r'\d+\s*(million|billion|thousand)',  # Large numbers
                r'in \d{4}',  # Years
                r'founded in',  # Founding dates
                r'established in',  # Establishment dates
            ]
            
            import re
            factual_count = 0
            for pattern in factual_indicators:
                if re.search(pattern, actual_answer, re.IGNORECASE):
                    factual_count += 1
            
            if factual_count > 0:
                score += min(factual_count * 0.3, 1.0)
            
            # Check for vague or evasive language (reduces accuracy)
            vague_phrases = [
                "it depends", "varies", "might be", "could be", "sometimes",
                "generally", "typically", "usually", "often", "many", "some"
            ]
            
            answer_lower = actual_answer.lower()
            vague_count = sum(1 for phrase in vague_phrases if phrase in answer_lower)
            
            if vague_count > 3:
                score -= 1.0
            elif vague_count > 1:
                score -= 0.5
            
            return min(max(score, 0.0), 5.0)
            
        except Exception as e:
            self.logger.error("Failed to calculate accuracy", error=str(e))
            return 2.5


class AgentEvaluationService:
    """Main service for agent evaluation with comprehensive features"""
    
    def __init__(self):
        self.logger = logger.bind(component="agent_evaluation_service")
        self.metrics_calculator = MetricsCalculator()
        self.agent_manager = get_rag_agent_manager()
        self.running_tasks: Dict[str, EvaluationTask] = {}
    
    async def create_evaluation(
        self,
        evaluation_request: AgentEvaluationCreate,
        organization_id: str,
        user_id: str
    ) -> AgentEvaluationResponse:
        """Create a new agent evaluation"""
        try:
            async with get_database() as db:
                # Get agent configuration
                config_query = select(AgentConfiguration).where(
                    AgentConfiguration.id == evaluation_request.agent_configuration_id,
                    AgentConfiguration.organization_id == uuid.UUID(organization_id)
                )
                config_result = await db.execute(config_query)
                config = config_result.scalar_one_or_none()
                
                if not config:
                    raise ValueError("Agent configuration not found")
                
                # Create evaluation record
                evaluation = AgentEvaluation(
                    id=uuid.uuid4(),
                    name=evaluation_request.name,
                    description=evaluation_request.description,
                    total_questions=len(evaluation_request.dataset),
                    dataset_metadata={
                        "dataset_size": len(evaluation_request.dataset),
                        "evaluation_config": evaluation_request.evaluation_config
                    },
                    organization_id=uuid.UUID(organization_id),
                    agent_configuration_id=evaluation_request.agent_configuration_id,
                    created_by=uuid.UUID(user_id)
                )
                
                db.add(evaluation)
                await db.commit()
                await db.refresh(evaluation)
                
                self.logger.info(
                    f"Created evaluation: {evaluation.name}",
                    evaluation_id=str(evaluation.id),
                    questions=len(evaluation_request.dataset)
                )
                
                # Convert to response
                return await self._convert_to_response(evaluation, db)
                
        except Exception as e:
            self.logger.error("Failed to create evaluation", error=str(e))
            raise
    
    async def start_evaluation(
        self,
        evaluation_id: str,
        dataset: List[EvaluationDataItem],
        organization_id: str
    ) -> str:
        """Start an evaluation task in the background"""
        task_id = str(uuid.uuid4())
        
        # Create task record
        task = EvaluationTask(
            task_id=task_id,
            status="pending",
            progress=0.0,
            total_items=len(dataset),
            completed_items=0,
            result=None,
            error=None,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            estimated_completion=None
        )
        
        self.running_tasks[task_id] = task
        
        # Start background task
        asyncio.create_task(
            self._run_evaluation_task(task_id, evaluation_id, dataset, organization_id)
        )
        
        self.logger.info(f"Started evaluation task: {task_id}")
        return task_id
    
    async def _run_evaluation_task(
        self,
        task_id: str,
        evaluation_id: str,
        dataset: List[EvaluationDataItem],
        organization_id: str
    ):
        """Run the evaluation task in the background"""
        task = self.running_tasks[task_id]
        
        try:
            task.status = "running"
            task.updated_at = datetime.utcnow()
            
            async with get_database() as db:
                # Get evaluation and config
                eval_query = select(AgentEvaluation).where(
                    AgentEvaluation.id == uuid.UUID(evaluation_id)
                )
                eval_result = await db.execute(eval_query)
                evaluation = eval_result.scalar_one_or_none()
                
                if not evaluation:
                    raise ValueError("Evaluation not found")
                
                # Get agent configuration
                config_query = select(AgentConfiguration).where(
                    AgentConfiguration.id == evaluation.agent_configuration_id
                )
                config_result = await db.execute(config_query)
                config = config_result.scalar_one_or_none()
                
                if not config:
                    raise ValueError("Agent configuration not found")
                
                # Update evaluation status
                evaluation.status = EvaluationStatusEnum.RUNNING
                evaluation.started_at = datetime.utcnow()
                await db.commit()
                
                # Get RAG agent
                agent = await self.agent_manager.get_agent(config, organization_id)
                
                # Process each question
                results = []
                total_tokens = 0
                total_time = 0
                
                for i, item in enumerate(dataset):
                    try:
                        # Generate response
                        start_time = time.time()
                        response = await agent.generate_response(
                            query=item.question,
                            chat_history=item.chat_history or [],
                            include_sources=True
                        )
                        end_time = time.time()
                        
                        response_time_ms = int((end_time - start_time) * 1000)
                        total_time += response_time_ms
                        total_tokens += response.tokens_used
                        
                        # Calculate metrics
                        metrics = await self._calculate_metrics(
                            question=item.question,
                            expected_answer=item.expected_answer,
                            actual_answer=response.answer,
                            context=response.context,
                            sources=response.sources
                        )
                        
                        # Create evaluation result
                        result = EvaluationResult(
                            id=uuid.uuid4(),
                            evaluation_id=evaluation.id,
                            question=item.question,
                            expected_answer=item.expected_answer,
                            actual_answer=response.answer,
                            context_used=response.context,
                            relevance_score=metrics.relevance,
                            groundedness_score=metrics.groundedness,
                            coherence_score=metrics.coherence,
                            similarity_score=metrics.similarity,
                            fluency_score=metrics.fluency,
                            accuracy_score=metrics.accuracy,
                            overall_score=metrics.average(),
                            tokens_used=response.tokens_used,
                            response_time_ms=response_time_ms,
                            sources_retrieved=len(response.sources),
                            sources=response.sources,
                            metadata=item.metadata
                        )
                        
                        db.add(result)
                        results.append(result)
                        
                        # Update progress
                        task.completed_items = i + 1
                        task.progress = (i + 1) / len(dataset)
                        task.updated_at = datetime.utcnow()
                        
                        # Commit periodically
                        if (i + 1) % 5 == 0:
                            await db.commit()
                        
                        self.logger.debug(f"Completed question {i + 1}/{len(dataset)}")
                        
                    except Exception as e:
                        self.logger.error(f"Error evaluating question {i + 1}", error=str(e))
                        # Continue with next question
                        continue
                
                # Calculate summary metrics
                if results:
                    avg_relevance = statistics.mean([r.relevance_score for r in results if r.relevance_score])
                    avg_groundedness = statistics.mean([r.groundedness_score for r in results if r.groundedness_score])
                    avg_coherence = statistics.mean([r.coherence_score for r in results if r.coherence_score])
                    avg_similarity = statistics.mean([r.similarity_score for r in results if r.similarity_score])
                    avg_fluency = statistics.mean([r.fluency_score for r in results if r.fluency_score])
                    avg_accuracy = statistics.mean([r.accuracy_score for r in results if r.accuracy_score])
                    overall_avg = statistics.mean([r.overall_score for r in results if r.overall_score])
                    
                    metrics_summary = {
                        "relevance": avg_relevance,
                        "groundedness": avg_groundedness,
                        "coherence": avg_coherence,
                        "similarity": avg_similarity,
                        "fluency": avg_fluency,
                        "accuracy": avg_accuracy,
                        "overall": overall_avg
                    }
                    
                    evaluation.overall_score = overall_avg
                    evaluation.metrics_summary = metrics_summary
                else:
                    evaluation.metrics_summary = {}
                
                # Update evaluation
                evaluation.status = EvaluationStatusEnum.COMPLETED
                evaluation.completed_at = datetime.utcnow()
                evaluation.completed_questions = len(results)
                evaluation.total_tokens_used = total_tokens
                evaluation.execution_time_seconds = int(total_time / 1000)
                evaluation.results = {
                    "summary": evaluation.metrics_summary,
                    "total_results": len(results),
                    "avg_response_time_ms": total_time / len(results) if results else 0
                }
                
                await db.commit()
                
                # Update task
                task.status = "completed"
                task.progress = 1.0
                task.result = {
                    "evaluation_id": str(evaluation.id),
                    "total_questions": len(dataset),
                    "completed_questions": len(results),
                    "overall_score": evaluation.overall_score,
                    "total_tokens": total_tokens,
                    "execution_time_seconds": evaluation.execution_time_seconds
                }
                task.updated_at = datetime.utcnow()
                
                self.logger.info(
                    f"Evaluation task completed: {task_id}",
                    questions_completed=len(results),
                    overall_score=evaluation.overall_score
                )
                
        except Exception as e:
            self.logger.error(f"Evaluation task failed: {task_id}", error=str(e))
            
            # Update task as failed
            task.status = "failed"
            task.error = str(e)
            task.updated_at = datetime.utcnow()
            
            # Update evaluation status
            try:
                async with get_database() as db:
                    eval_query = select(AgentEvaluation).where(
                        AgentEvaluation.id == uuid.UUID(evaluation_id)
                    )
                    eval_result = await db.execute(eval_query)
                    evaluation = eval_result.scalar_one_or_none()
                    
                    if evaluation:
                        evaluation.status = EvaluationStatusEnum.FAILED
                        evaluation.error_message = str(e)
                        await db.commit()
                        
            except Exception as db_error:
                self.logger.error("Failed to update evaluation status", error=str(db_error))
    
    async def _calculate_metrics(
        self,
        question: str,
        expected_answer: Optional[str],
        actual_answer: str,
        context: Optional[str],
        sources: List[Dict[str, Any]]
    ) -> EvaluationMetrics:
        """Calculate all evaluation metrics for a response"""
        
        # Run all metric calculations concurrently
        relevance_task = self.metrics_calculator.calculate_relevance(question, actual_answer, expected_answer)
        groundedness_task = self.metrics_calculator.calculate_groundedness(actual_answer, context, sources)
        coherence_task = self.metrics_calculator.calculate_coherence(actual_answer)
        similarity_task = self.metrics_calculator.calculate_similarity(expected_answer, actual_answer)
        fluency_task = self.metrics_calculator.calculate_fluency(actual_answer)
        accuracy_task = self.metrics_calculator.calculate_accuracy(question, expected_answer, actual_answer, context)
        
        # Wait for all calculations to complete
        relevance, groundedness, coherence, similarity, fluency, accuracy = await asyncio.gather(
            relevance_task, groundedness_task, coherence_task, 
            similarity_task, fluency_task, accuracy_task
        )
        
        return EvaluationMetrics(
            relevance=relevance,
            groundedness=groundedness,
            coherence=coherence,
            similarity=similarity,
            fluency=fluency,
            accuracy=accuracy,
            overall=None  # Will be calculated by average() method
        )
    
    async def get_task_status(self, task_id: str) -> TaskStatus:
        """Get the status of an evaluation task"""
        if task_id not in self.running_tasks:
            raise ValueError("Task not found")
        
        task = self.running_tasks[task_id]
        
        return TaskStatus(
            task_id=uuid.UUID(task_id),
            status=task.status,
            progress=task.progress,
            total_items=task.total_items,
            completed_items=task.completed_items,
            result=task.result,
            error=task.error,
            created_at=task.created_at,
            updated_at=task.updated_at,
            estimated_completion=task.estimated_completion
        )
    
    async def get_evaluations(
        self,
        organization_id: str,
        limit: int = 100,
        offset: int = 0,
        agent_configuration_id: Optional[str] = None
    ) -> List[AgentEvaluationResponse]:
        """Get evaluations for an organization"""
        try:
            async with get_database() as db:
                query = select(AgentEvaluation).where(
                    AgentEvaluation.organization_id == uuid.UUID(organization_id)
                )
                
                if agent_configuration_id:
                    query = query.where(
                        AgentEvaluation.agent_configuration_id == uuid.UUID(agent_configuration_id)
                    )
                
                query = query.limit(limit).offset(offset).order_by(
                    AgentEvaluation.created_at.desc()
                )
                
                result = await db.execute(query)
                evaluations = result.scalars().all()
                
                responses = []
                for evaluation in evaluations:
                    response = await self._convert_to_response(evaluation, db)
                    responses.append(response)
                
                return responses
                
        except Exception as e:
            self.logger.error("Failed to get evaluations", error=str(e))
            return []
    
    async def get_evaluation(
        self,
        evaluation_id: str,
        organization_id: str,
        include_results: bool = False
    ) -> Optional[AgentEvaluationResponse]:
        """Get a specific evaluation by ID"""
        try:
            async with get_database() as db:
                query = select(AgentEvaluation).where(
                    AgentEvaluation.id == uuid.UUID(evaluation_id),
                    AgentEvaluation.organization_id == uuid.UUID(organization_id)
                )
                
                result = await db.execute(query)
                evaluation = result.scalar_one_or_none()
                
                if not evaluation:
                    return None
                
                return await self._convert_to_response(evaluation, db, include_results)
                
        except Exception as e:
            self.logger.error(f"Failed to get evaluation: {evaluation_id}", error=str(e))
            return None
    
    async def _convert_to_response(
        self,
        evaluation: AgentEvaluation,
        db: AsyncSession,
        include_results: bool = False
    ) -> AgentEvaluationResponse:
        """Convert evaluation model to response schema"""
        
        # Get agent configuration
        config_query = select(AgentConfiguration).where(
            AgentConfiguration.id == evaluation.agent_configuration_id
        )
        config_result = await db.execute(config_query)
        config = config_result.scalar_one()
        
        # Convert config to response
        config_response = {
            "id": config.id,
            "name": config.name,
            "description": config.description,
            "version": config.version,
            "model_name": config.model_name,
            "model_provider": config.model_provider,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "top_p": config.top_p,
            "frequency_penalty": config.frequency_penalty,
            "presence_penalty": config.presence_penalty,
            "retrieval_strategy": config.retrieval_strategy,
            "retrieval_k": config.retrieval_k,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "embedding_model": config.embedding_model,
            "system_prompt": config.system_prompt,
            "evaluation_prompt": config.evaluation_prompt,
            "configuration": config.configuration,
            "metadata": config.metadata,
            "status": config.status,
            "created_at": config.created_at,
            "updated_at": config.updated_at
        }
        
        # Get results if requested
        evaluation_results = []
        if include_results:
            results_query = select(EvaluationResult).where(
                EvaluationResult.evaluation_id == evaluation.id
            ).order_by(EvaluationResult.created_at)
            
            results_result = await db.execute(results_query)
            results = results_result.scalars().all()
            
            for result in results:
                metrics = EvaluationMetrics(
                    relevance=result.relevance_score,
                    groundedness=result.groundedness_score,
                    coherence=result.coherence_score,
                    similarity=result.similarity_score,
                    fluency=result.fluency_score,
                    accuracy=result.accuracy_score,
                    overall=result.overall_score
                )
                
                evaluation_results.append(EvaluationResultResponse(
                    id=result.id,
                    question=result.question,
                    expected_answer=result.expected_answer,
                    actual_answer=result.actual_answer,
                    context_used=result.context_used,
                    metrics=metrics,
                    tokens_used=result.tokens_used,
                    response_time_ms=result.response_time_ms,
                    sources_retrieved=result.sources_retrieved,
                    sources=result.sources or [],
                    metadata=result.metadata or {},
                    created_at=result.created_at
                ))
        
        return AgentEvaluationResponse(
            id=evaluation.id,
            name=evaluation.name,
            description=evaluation.description,
            status=evaluation.status,
            total_questions=evaluation.total_questions,
            completed_questions=evaluation.completed_questions,
            overall_score=evaluation.overall_score,
            metrics_summary=evaluation.metrics_summary or {},
            started_at=evaluation.started_at,
            completed_at=evaluation.completed_at,
            execution_time_seconds=evaluation.execution_time_seconds,
            total_tokens_used=evaluation.total_tokens_used,
            results=evaluation.results or {},
            error_message=evaluation.error_message,
            created_at=evaluation.created_at,
            updated_at=evaluation.updated_at,
            agent_configuration=config_response,
            evaluation_results=evaluation_results
        )


# Global service instance
_evaluation_service: Optional[AgentEvaluationService] = None


def get_agent_evaluation_service() -> AgentEvaluationService:
    """Get or create the global agent evaluation service"""
    global _evaluation_service
    if _evaluation_service is None:
        _evaluation_service = AgentEvaluationService()
    return _evaluation_service