"""
Agent Evaluator Service
Comprehensive multi-agent evaluation and comparison service
"""

import asyncio
import json
import logging
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from pathlib import Path
import uuid
from collections import defaultdict, deque
import re
import numpy as np
from dataclasses import dataclass, asdict
from enum import Enum

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import get_database

logger = logging.getLogger(__name__)
settings = get_settings()

class AgentType(Enum):
    CONVERSATIONAL = "conversational"
    RAG = "rag"
    TOOL_USING = "tool_using"
    REASONING = "reasoning"
    CODING = "coding"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    MULTI_MODAL = "multi_modal"

class EvaluationDimension(Enum):
    ACCURACY = "accuracy"
    RELEVANCE = "relevance"
    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"
    EFFICIENCY = "efficiency"
    SAFETY = "safety"
    CREATIVITY = "creativity"
    REASONING = "reasoning"
    TOOL_USE = "tool_use"
    CONTEXT_AWARENESS = "context_awareness"

@dataclass
class AgentEvaluationResult:
    """Data class for agent evaluation results"""
    agent_id: str
    agent_name: str
    agent_type: str
    evaluation_id: str
    task_id: str
    
    # Input/Output
    input_prompt: str
    agent_response: str
    expected_output: Optional[str]
    context: Dict[str, Any]
    
    # Core Performance Metrics
    accuracy: float
    relevance: float
    completeness: float
    consistency: float
    efficiency: float
    
    # Agent-Specific Metrics
    reasoning_quality: float
    tool_usage_effectiveness: float
    context_utilization: float
    error_handling: float
    adaptability: float
    
    # Communication Metrics
    clarity: float
    conciseness: float
    tone_appropriateness: float
    information_density: float
    
    # Safety & Ethics
    safety_score: float
    bias_score: float
    toxicity_score: float
    privacy_compliance: float
    
    # Task-Specific Metrics
    task_success_rate: float
    instruction_following: float
    format_compliance: float
    creativity_score: float
    
    # Interaction Metrics
    response_time_ms: float
    conversation_turns: int
    user_satisfaction: float
    engagement_score: float
    
    # Technical Metrics
    token_efficiency: float
    resource_usage: float
    error_rate: float
    availability: float
    
    # Overall Metrics
    overall_score: float
    confidence_score: float
    
    # Metadata
    evaluation_time: datetime
    evaluator_version: str
    model_version: str
    
@dataclass
class AgentComparisonResult:
    """Data class for agent comparison results"""
    comparison_id: str
    agents: List[str]
    evaluation_tasks: int
    
    # Performance Rankings
    overall_ranking: List[Tuple[str, float]]
    dimension_rankings: Dict[str, List[Tuple[str, float]]]
    
    # Statistical Analysis
    statistical_significance: Dict[str, Dict[str, float]]
    effect_sizes: Dict[str, Dict[str, float]]
    
    # Strengths and Weaknesses
    agent_profiles: Dict[str, Dict[str, Any]]
    
    # Recommendations
    use_case_recommendations: Dict[str, List[str]]
    improvement_suggestions: Dict[str, List[str]]
    
    comparison_time: datetime

class AgentEvaluatorService:
    """
    Comprehensive agent evaluation and comparison service
    """
    
    def __init__(self):
        self.evaluation_cache = {}
        self.comparison_cache = {}
        self.agent_registry = {}
        self.evaluation_history = defaultdict(list)
        self.sentence_transformer = None
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Evaluation configuration
        self.eval_config = {
            'batch_size': 8,
            'max_concurrent': 3,
            'timeout_seconds': 120,
            'min_samples_for_comparison': 10,
            'statistical_significance_threshold': 0.05
        }
        
        # Dimension weights by agent type
        self.dimension_weights = {
            AgentType.CONVERSATIONAL: {
                'accuracy': 0.2, 'relevance': 0.25, 'clarity': 0.2,
                'engagement_score': 0.15, 'safety_score': 0.2
            },
            AgentType.RAG: {
                'accuracy': 0.3, 'relevance': 0.25, 'context_utilization': 0.2,
                'completeness': 0.15, 'efficiency': 0.1
            },
            AgentType.TOOL_USING: {
                'tool_usage_effectiveness': 0.3, 'task_success_rate': 0.25,
                'accuracy': 0.2, 'efficiency': 0.15, 'error_handling': 0.1
            },
            AgentType.REASONING: {
                'reasoning_quality': 0.35, 'accuracy': 0.25, 'consistency': 0.2,
                'completeness': 0.15, 'clarity': 0.05
            },
            AgentType.CODING: {
                'accuracy': 0.3, 'task_success_rate': 0.25, 'format_compliance': 0.2,
                'efficiency': 0.15, 'error_rate': 0.1
            },
            AgentType.CREATIVE: {
                'creativity_score': 0.3, 'relevance': 0.2, 'clarity': 0.2,
                'engagement_score': 0.15, 'tone_appropriateness': 0.15
            }
        }
        
    async def initialize_models(self):
        """
        Initialize evaluation models and resources
        """
        try:
            if self.sentence_transformer is None:
                self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            
            logger.info("Agent evaluation models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent evaluation models: {e}")
            raise
    
    async def register_agent(
        self,
        agent_id: str,
        agent_name: str,
        agent_type: AgentType,
        agent_instance: Any,
        capabilities: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Register an agent for evaluation
        
        Args:
            agent_id: Unique agent identifier
            agent_name: Human-readable agent name
            agent_type: Type of agent
            agent_instance: Agent instance or interface
            capabilities: List of agent capabilities
            metadata: Optional agent metadata
        
        Returns:
            Registration status
        """
        try:
            agent_info = {
                'agent_id': agent_id,
                'agent_name': agent_name,
                'agent_type': agent_type.value,
                'agent_instance': agent_instance,
                'capabilities': capabilities,
                'metadata': metadata or {},
                'registered_at': datetime.utcnow(),
                'evaluation_count': 0,
                'last_evaluation': None
            }
            
            self.agent_registry[agent_id] = agent_info
            
            return {
                'status': 'success',
                'agent_id': agent_id,
                'message': f'Agent {agent_name} registered successfully'
            }
            
        except Exception as e:
            logger.error(f"Failed to register agent {agent_id}: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'agent_id': agent_id
            }
    
    async def evaluate_agent(
        self,
        evaluation_id: str,
        agent_id: str,
        evaluation_tasks: List[Dict[str, Any]],
        evaluation_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single agent across multiple tasks
        
        Args:
            evaluation_id: Unique evaluation identifier
            agent_id: Agent to evaluate
            evaluation_tasks: List of evaluation tasks
            evaluation_config: Optional evaluation configuration
        
        Returns:
            Comprehensive evaluation results
        """
        try:
            if agent_id not in self.agent_registry:
                raise ValueError(f"Agent {agent_id} not registered")
            
            await self.initialize_models()
            
            agent_info = self.agent_registry[agent_id]
            config = {**self.eval_config, **(evaluation_config or {})}
            
            evaluation_results = {
                'evaluation_id': evaluation_id,
                'agent_id': agent_id,
                'agent_name': agent_info['agent_name'],
                'agent_type': agent_info['agent_type'],
                'start_time': datetime.utcnow(),
                'config': config,
                'total_tasks': len(evaluation_tasks),
                'task_results': [],
                'aggregate_metrics': {},
                'performance_profile': {},
                'status': 'running'
            }
            
            # Process tasks in batches
            batch_size = config['batch_size']
            task_batches = [evaluation_tasks[i:i + batch_size] 
                          for i in range(0, len(evaluation_tasks), batch_size)]
            
            all_task_results = []
            
            for batch_idx, task_batch in enumerate(task_batches):
                logger.info(f"Processing batch {batch_idx + 1}/{len(task_batches)} for agent {agent_id}")
                
                batch_results = await self._evaluate_task_batch(
                    agent_info, task_batch, config, batch_idx * batch_size
                )
                
                all_task_results.extend(batch_results)
                
                # Update progress
                evaluation_results['processed_tasks'] = len(all_task_results)
                evaluation_results['progress'] = len(all_task_results) / len(evaluation_tasks)
            
            evaluation_results['task_results'] = all_task_results
            
            # Calculate aggregate metrics
            aggregate_metrics = await self._calculate_aggregate_metrics(all_task_results)
            evaluation_results['aggregate_metrics'] = aggregate_metrics
            
            # Generate performance profile
            performance_profile = await self._generate_performance_profile(
                agent_info, all_task_results, aggregate_metrics
            )
            evaluation_results['performance_profile'] = performance_profile
            
            evaluation_results['end_time'] = datetime.utcnow()
            evaluation_results['duration_seconds'] = (
                evaluation_results['end_time'] - evaluation_results['start_time']
            ).total_seconds()
            evaluation_results['status'] = 'completed'
            
            # Update agent registry
            agent_info['evaluation_count'] += 1
            agent_info['last_evaluation'] = datetime.utcnow()
            
            # Store results
            self.evaluation_cache[evaluation_id] = evaluation_results
            self.evaluation_history[agent_id].append(evaluation_results)
            
            return {
                'status': 'success',
                **evaluation_results
            }
            
        except Exception as e:
            logger.error(f"Failed to evaluate agent {agent_id}: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'evaluation_id': evaluation_id,
                'agent_id': agent_id
            }
    
    async def compare_agents(
        self,
        comparison_id: str,
        agent_ids: List[str],
        evaluation_tasks: List[Dict[str, Any]],
        comparison_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple agents across the same set of tasks
        
        Args:
            comparison_id: Unique comparison identifier
            agent_ids: List of agent IDs to compare
            evaluation_tasks: Common set of evaluation tasks
            comparison_config: Optional comparison configuration
        
        Returns:
            Comprehensive comparison results
        """
        try:
            if len(agent_ids) < 2:
                raise ValueError("Need at least 2 agents for comparison")
            
            # Validate all agents are registered
            missing_agents = [aid for aid in agent_ids if aid not in self.agent_registry]
            if missing_agents:
                raise ValueError(f"Agents not registered: {missing_agents}")
            
            config = {**self.eval_config, **(comparison_config or {})}
            
            comparison_results = {
                'comparison_id': comparison_id,
                'agent_ids': agent_ids,
                'start_time': datetime.utcnow(),
                'total_tasks': len(evaluation_tasks),
                'agent_results': {},
                'comparison_analysis': {},
                'status': 'running'
            }
            
            # Evaluate each agent
            for agent_id in agent_ids:
                logger.info(f"Evaluating agent {agent_id} for comparison {comparison_id}")
                
                eval_id = f"{comparison_id}_{agent_id}"
                
                agent_eval = await self.evaluate_agent(
                    evaluation_id=eval_id,
                    agent_id=agent_id,
                    evaluation_tasks=evaluation_tasks,
                    evaluation_config=config
                )
                
                comparison_results['agent_results'][agent_id] = agent_eval
            
            # Perform comparison analysis
            comparison_analysis = await self._perform_comparison_analysis(
                comparison_results['agent_results'],
                config
            )
            comparison_results['comparison_analysis'] = comparison_analysis
            
            comparison_results['end_time'] = datetime.utcnow()
            comparison_results['duration_seconds'] = (
                comparison_results['end_time'] - comparison_results['start_time']
            ).total_seconds()
            comparison_results['status'] = 'completed'
            
            # Store comparison results
            self.comparison_cache[comparison_id] = comparison_results
            
            return {
                'status': 'success',
                **comparison_results
            }
            
        except Exception as e:
            logger.error(f"Failed to compare agents: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'comparison_id': comparison_id
            }
    
    async def _evaluate_task_batch(
        self,
        agent_info: Dict[str, Any],
        tasks: List[Dict[str, Any]],
        config: Dict[str, Any],
        start_idx: int
    ) -> List[AgentEvaluationResult]:
        """
        Evaluate a batch of tasks for an agent
        """
        semaphore = asyncio.Semaphore(config['max_concurrent'])
        
        async def process_single_task(i, task):
            async with semaphore:
                task_idx = start_idx + i
                try:
                    result = await self._evaluate_single_task(
                        agent_info=agent_info,
                        task=task,
                        task_id=f"task_{task_idx}",
                        config=config
                    )
                    return result
                except Exception as e:
                    logger.error(f"Error evaluating task {task_idx}: {e}")
                    return self._create_error_result(
                        agent_info, task, f"task_{task_idx}", str(e)
                    )
        
        # Process tasks concurrently
        tasks_coroutines = [process_single_task(i, task) for i, task in enumerate(tasks)]
        batch_results = await asyncio.gather(*tasks_coroutines, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = [r for r in batch_results if isinstance(r, AgentEvaluationResult)]
        
        return valid_results
    
    async def _evaluate_single_task(
        self,
        agent_info: Dict[str, Any],
        task: Dict[str, Any],
        task_id: str,
        config: Dict[str, Any]
    ) -> AgentEvaluationResult:
        """
        Evaluate a single task for an agent
        """
        try:
            agent_instance = agent_info['agent_instance']
            agent_type = AgentType(agent_info['agent_type'])
            
            input_prompt = task['prompt']
            expected_output = task.get('expected_output')
            context = task.get('context', {})
            task_type = task.get('task_type', 'general')
            
            # Execute the task with the agent
            start_time = datetime.utcnow()
            
            agent_response = await self._execute_agent_task(
                agent_instance, input_prompt, context, config
            )
            
            end_time = datetime.utcnow()
            response_time = (end_time - start_time).total_seconds() * 1000
            
            # Calculate all evaluation metrics
            metrics = await self._calculate_task_metrics(
                input_prompt=input_prompt,
                agent_response=agent_response,
                expected_output=expected_output,
                context=context,
                task_type=task_type,
                agent_type=agent_type
            )
            
            # Calculate overall score
            overall_score = await self._calculate_overall_score(
                metrics, agent_type
            )
            
            confidence_score = await self._calculate_confidence_score(
                metrics, agent_response
            )
            
            return AgentEvaluationResult(
                agent_id=agent_info['agent_id'],
                agent_name=agent_info['agent_name'],
                agent_type=agent_info['agent_type'],
                evaluation_id="",  # Will be set by caller
                task_id=task_id,
                input_prompt=input_prompt,
                agent_response=agent_response,
                expected_output=expected_output,
                context=context,
                **metrics,
                response_time_ms=response_time,
                conversation_turns=1,  # Default for single-turn tasks
                overall_score=overall_score,
                confidence_score=confidence_score,
                evaluation_time=datetime.utcnow(),
                evaluator_version="1.0",
                model_version=agent_info.get('metadata', {}).get('version', '1.0')
            )
            
        except Exception as e:
            logger.error(f"Error evaluating single task {task_id}: {e}")
            return self._create_error_result(agent_info, task, task_id, str(e))
    
    async def _execute_agent_task(
        self,
        agent_instance: Any,
        prompt: str,
        context: Dict[str, Any],
        config: Dict[str, Any]
    ) -> str:
        """
        Execute a task with the agent
        """
        try:
            timeout = config.get('timeout_seconds', 120)
            
            async def run_agent():
                if hasattr(agent_instance, 'execute_async'):
                    return await agent_instance.execute_async(prompt, context)
                elif hasattr(agent_instance, 'execute'):
                    return agent_instance.execute(prompt, context)
                elif hasattr(agent_instance, 'generate_response'):
                    return await agent_instance.generate_response(prompt, context)
                elif hasattr(agent_instance, 'respond'):
                    return agent_instance.respond(prompt)
                else:
                    # Fallback simulation
                    await asyncio.sleep(0.1)
                    return f"Agent response to: {prompt[:50]}..."
            
            response = await asyncio.wait_for(run_agent(), timeout=timeout)
            return str(response) if response else ""
            
        except asyncio.TimeoutError:
            return "[TIMEOUT ERROR]"
        except Exception as e:
            return f"[ERROR: {str(e)}]"
    
    async def _calculate_task_metrics(
        self,
        input_prompt: str,
        agent_response: str,
        expected_output: Optional[str],
        context: Dict[str, Any],
        task_type: str,
        agent_type: AgentType
    ) -> Dict[str, float]:
        """
        Calculate all evaluation metrics for a task
        """
        metrics = {
            # Core Performance Metrics
            'accuracy': 0.0,
            'relevance': 0.0,
            'completeness': 0.0,
            'consistency': 0.0,
            'efficiency': 0.0,
            
            # Agent-Specific Metrics
            'reasoning_quality': 0.0,
            'tool_usage_effectiveness': 0.0,
            'context_utilization': 0.0,
            'error_handling': 0.0,
            'adaptability': 0.0,
            
            # Communication Metrics
            'clarity': 0.0,
            'conciseness': 0.0,
            'tone_appropriateness': 0.0,
            'information_density': 0.0,
            
            # Safety & Ethics
            'safety_score': 1.0,
            'bias_score': 0.0,
            'toxicity_score': 0.0,
            'privacy_compliance': 1.0,
            
            # Task-Specific Metrics
            'task_success_rate': 0.0,
            'instruction_following': 0.0,
            'format_compliance': 0.0,
            'creativity_score': 0.0,
            
            # Interaction Metrics
            'user_satisfaction': 0.0,
            'engagement_score': 0.0,
            
            # Technical Metrics
            'token_efficiency': 0.0,
            'resource_usage': 0.0,
            'error_rate': 0.0,
            'availability': 1.0
        }
        
        if not agent_response or agent_response.startswith('[ERROR'):
            if agent_response.startswith('[TIMEOUT'):
                metrics['availability'] = 0.0
            else:
                metrics['error_rate'] = 1.0
            return metrics
        
        try:
            # 1. Core Performance Metrics
            core_metrics = await self._evaluate_core_performance(
                input_prompt, agent_response, expected_output
            )
            metrics.update(core_metrics)
            
            # 2. Agent-Specific Metrics
            agent_metrics = await self._evaluate_agent_specific_metrics(
                input_prompt, agent_response, context, agent_type
            )
            metrics.update(agent_metrics)
            
            # 3. Communication Metrics
            comm_metrics = await self._evaluate_communication_quality(
                agent_response, input_prompt
            )
            metrics.update(comm_metrics)
            
            # 4. Safety & Ethics
            safety_metrics = await self._evaluate_safety_and_ethics(
                agent_response
            )
            metrics.update(safety_metrics)
            
            # 5. Task-Specific Metrics
            task_metrics = await self._evaluate_task_specific_performance(
                input_prompt, agent_response, task_type, expected_output
            )
            metrics.update(task_metrics)
            
            # 6. Technical Metrics
            tech_metrics = await self._evaluate_technical_metrics(
                agent_response, input_prompt
            )
            metrics.update(tech_metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating task metrics: {e}")
            return metrics
    
    async def _evaluate_core_performance(
        self,
        prompt: str,
        response: str,
        expected: Optional[str]
    ) -> Dict[str, float]:
        """
        Evaluate core performance metrics
        """
        metrics = {}
        
        # Accuracy
        if expected:
            metrics['accuracy'] = await self._calculate_semantic_similarity(response, expected)
        else:
            metrics['accuracy'] = await self._estimate_accuracy(prompt, response)
        
        # Relevance
        metrics['relevance'] = await self._calculate_semantic_similarity(prompt, response)
        
        # Completeness
        metrics['completeness'] = await self._calculate_completeness(prompt, response, expected)
        
        # Consistency
        metrics['consistency'] = await self._calculate_consistency(response)
        
        # Efficiency (based on response length vs information density)
        metrics['efficiency'] = await self._calculate_efficiency(response)
        
        return metrics
    
    async def _evaluate_agent_specific_metrics(
        self,
        prompt: str,
        response: str,
        context: Dict[str, Any],
        agent_type: AgentType
    ) -> Dict[str, float]:
        """
        Evaluate agent-type-specific metrics
        """
        metrics = {}
        
        # Reasoning quality
        if agent_type in [AgentType.REASONING, AgentType.ANALYTICAL]:
            metrics['reasoning_quality'] = await self._evaluate_reasoning_quality(response)
        else:
            metrics['reasoning_quality'] = 0.5  # Neutral for non-reasoning tasks
        
        # Tool usage effectiveness
        if agent_type == AgentType.TOOL_USING:
            metrics['tool_usage_effectiveness'] = await self._evaluate_tool_usage(response, context)
        else:
            metrics['tool_usage_effectiveness'] = 0.0
        
        # Context utilization
        if context:
            metrics['context_utilization'] = await self._evaluate_context_utilization(response, context)
        else:
            metrics['context_utilization'] = 0.0
        
        # Error handling
        metrics['error_handling'] = await self._evaluate_error_handling(response)
        
        # Adaptability
        metrics['adaptability'] = await self._evaluate_adaptability(prompt, response)
        
        return metrics
    
    async def _evaluate_communication_quality(
        self,
        response: str,
        prompt: str
    ) -> Dict[str, float]:
        """
        Evaluate communication quality metrics
        """
        metrics = {}
        
        # Clarity
        metrics['clarity'] = await self._calculate_clarity(response)
        
        # Conciseness
        metrics['conciseness'] = await self._calculate_conciseness(response)
        
        # Tone appropriateness
        metrics['tone_appropriateness'] = await self._calculate_tone_appropriateness(prompt, response)
        
        # Information density
        metrics['information_density'] = await self._calculate_information_density(response)
        
        return metrics
    
    async def _evaluate_safety_and_ethics(self, response: str) -> Dict[str, float]:
        """
        Evaluate safety and ethical considerations
        """
        metrics = {}
        
        # Safety score (simplified)
        metrics['safety_score'] = await self._calculate_safety_score(response)
        
        # Bias score (simplified)
        metrics['bias_score'] = await self._calculate_bias_score(response)
        
        # Toxicity score (simplified)
        metrics['toxicity_score'] = await self._calculate_toxicity_score(response)
        
        # Privacy compliance (simplified)
        metrics['privacy_compliance'] = await self._calculate_privacy_compliance(response)
        
        return metrics
    
    async def _evaluate_task_specific_performance(
        self,
        prompt: str,
        response: str,
        task_type: str,
        expected: Optional[str]
    ) -> Dict[str, float]:
        """
        Evaluate task-specific performance metrics
        """
        metrics = {}
        
        # Task success rate
        metrics['task_success_rate'] = await self._calculate_task_success(response, task_type, expected)
        
        # Instruction following
        metrics['instruction_following'] = await self._calculate_instruction_following(prompt, response)
        
        # Format compliance
        metrics['format_compliance'] = await self._calculate_format_compliance(response, task_type)
        
        # Creativity score
        if task_type in ['creative', 'brainstorming', 'writing']:
            metrics['creativity_score'] = await self._calculate_creativity_score(response)
        else:
            metrics['creativity_score'] = 0.5
        
        return metrics
    
    async def _evaluate_technical_metrics(self, response: str, prompt: str) -> Dict[str, float]:
        """
        Evaluate technical performance metrics
        """
        metrics = {}
        
        # Token efficiency
        metrics['token_efficiency'] = await self._calculate_token_efficiency(prompt, response)
        
        # Resource usage (simplified)
        metrics['resource_usage'] = 0.5  # Placeholder
        
        # Error rate
        if '[ERROR' in response or '[TIMEOUT' in response:
            metrics['error_rate'] = 1.0
        else:
            metrics['error_rate'] = 0.0
        
        # Availability
        metrics['availability'] = 0.0 if '[TIMEOUT' in response else 1.0
        
        return metrics
    
    # Helper methods for metric calculations
    async def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between texts"""
        try:
            if not text1 or not text2:
                return 0.0
            
            if self.sentence_transformer is None:
                await self.initialize_models()
            
            embeddings = self.sentence_transformer.encode([text1, text2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            
            return float(max(0.0, similarity))
        except:
            return 0.0
    
    async def _estimate_accuracy(self, prompt: str, response: str) -> float:
        """Estimate accuracy without expected output"""
        if not response or len(response.strip()) < 5:
            return 0.0
        
        # Simple heuristics for accuracy estimation
        relevance = await self._calculate_semantic_similarity(prompt, response)
        
        # Penalize very short or very long responses
        length_score = 1.0
        words = len(response.split())
        if words < 5:
            length_score = 0.3
        elif words > 500:
            length_score = 0.7
        
        return (relevance + length_score) / 2
    
    async def _calculate_completeness(self, prompt: str, response: str, expected: Optional[str]) -> float:
        """Calculate completeness score"""
        if not response:
            return 0.0
        
        # If we have expected output, use it for comparison
        if expected:
            return await self._calculate_semantic_similarity(response, expected)
        
        # Otherwise, use response length as proxy
        words = len(response.split())
        if words < 5:
            return 0.2
        elif words < 20:
            return 0.6
        elif words < 100:
            return 0.9
        else:
            return 0.8
    
    async def _calculate_consistency(self, response: str) -> float:
        """Calculate internal consistency"""
        if not response:
            return 0.0
        
        # Check for contradictory statements
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip().lower() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return 1.0
        
        # Simple contradiction detection
        contradictory_pairs = [
            ('yes', 'no'), ('true', 'false'), ('good', 'bad'),
            ('possible', 'impossible'), ('always', 'never')
        ]
        
        contradiction_count = 0
        for pair in contradictory_pairs:
            if any(pair[0] in s for s in sentences) and any(pair[1] in s for s in sentences):
                contradiction_count += 1
        
        return max(0, 1 - (contradiction_count / len(contradictory_pairs)))
    
    async def _calculate_efficiency(self, response: str) -> float:
        """Calculate efficiency based on information density"""
        if not response:
            return 0.0
        
        words = response.split()
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0
        
        # Information density heuristic
        avg_sentence_length = len(words) / len(sentences)
        
        # Optimal range is 8-15 words per sentence
        if 8 <= avg_sentence_length <= 15:
            return 1.0
        elif avg_sentence_length < 5:
            return 0.5
        else:
            return max(0.3, 1.0 - (avg_sentence_length - 15) / 20)
    
    # Placeholder implementations for remaining metrics
    async def _evaluate_reasoning_quality(self, response: str) -> float:
        return 0.7  # Placeholder
    
    async def _evaluate_tool_usage(self, response: str, context: Dict[str, Any]) -> float:
        return 0.8  # Placeholder
    
    async def _evaluate_context_utilization(self, response: str, context: Dict[str, Any]) -> float:
        return 0.6  # Placeholder
    
    async def _evaluate_error_handling(self, response: str) -> float:
        return 0.9 if '[ERROR' not in response else 0.1
    
    async def _evaluate_adaptability(self, prompt: str, response: str) -> float:
        return 0.7  # Placeholder
    
    async def _calculate_clarity(self, response: str) -> float:
        return 0.8  # Placeholder
    
    async def _calculate_conciseness(self, response: str) -> float:
        words = len(response.split()) if response else 0
        return max(0, 1.0 - words / 200)  # Prefer shorter responses
    
    async def _calculate_tone_appropriateness(self, prompt: str, response: str) -> float:
        return 0.8  # Placeholder
    
    async def _calculate_information_density(self, response: str) -> float:
        return 0.7  # Placeholder
    
    async def _calculate_safety_score(self, response: str) -> float:
        return 0.9  # Placeholder - high safety
    
    async def _calculate_bias_score(self, response: str) -> float:
        return 0.1  # Placeholder - low bias
    
    async def _calculate_toxicity_score(self, response: str) -> float:
        return 0.1  # Placeholder - low toxicity
    
    async def _calculate_privacy_compliance(self, response: str) -> float:
        return 0.9  # Placeholder - high compliance
    
    async def _calculate_task_success(self, response: str, task_type: str, expected: Optional[str]) -> float:
        if expected:
            return await self._calculate_semantic_similarity(response, expected)
        return 0.7  # Placeholder
    
    async def _calculate_instruction_following(self, prompt: str, response: str) -> float:
        return await self._calculate_semantic_similarity(prompt, response)
    
    async def _calculate_format_compliance(self, response: str, task_type: str) -> float:
        return 0.9  # Placeholder
    
    async def _calculate_creativity_score(self, response: str) -> float:
        return 0.6  # Placeholder
    
    async def _calculate_token_efficiency(self, prompt: str, response: str) -> float:
        prompt_tokens = len(prompt.split())
        response_tokens = len(response.split()) if response else 1
        return min(1.0, prompt_tokens / response_tokens)
    
    async def _calculate_overall_score(self, metrics: Dict[str, float], agent_type: AgentType) -> float:
        """Calculate overall score based on agent type weights"""
        weights = self.dimension_weights.get(agent_type, {
            'accuracy': 0.25, 'relevance': 0.25, 'clarity': 0.2,
            'safety_score': 0.15, 'task_success_rate': 0.15
        })
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in metrics:
                weighted_score += metrics[metric] * weight
                total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    async def _calculate_confidence_score(self, metrics: Dict[str, float], response: str) -> float:
        """Calculate confidence score"""
        confidence_factors = []
        
        # Response quality factor
        if response and len(response.strip()) > 10:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.2)
        
        # Metric consistency factor
        metric_values = [v for v in metrics.values() if isinstance(v, (int, float))]
        if metric_values:
            variance = np.var(metric_values)
            consistency = max(0.0, 1.0 - variance)
            confidence_factors.append(consistency)
        
        # Safety factor
        safety = metrics.get('safety_score', 1.0)
        confidence_factors.append(safety)
        
        return float(np.mean(confidence_factors)) if confidence_factors else 0.0
    
    def _create_error_result(
        self,
        agent_info: Dict[str, Any],
        task: Dict[str, Any],
        task_id: str,
        error_msg: str
    ) -> AgentEvaluationResult:
        """Create error result for failed evaluations"""
        return AgentEvaluationResult(
            agent_id=agent_info['agent_id'],
            agent_name=agent_info['agent_name'],
            agent_type=agent_info['agent_type'],
            evaluation_id="",
            task_id=task_id,
            input_prompt=task.get('prompt', ''),
            agent_response=f"[ERROR: {error_msg}]",
            expected_output=task.get('expected_output'),
            context=task.get('context', {}),
            accuracy=0.0, relevance=0.0, completeness=0.0, consistency=0.0, efficiency=0.0,
            reasoning_quality=0.0, tool_usage_effectiveness=0.0, context_utilization=0.0,
            error_handling=0.0, adaptability=0.0, clarity=0.0, conciseness=0.0,
            tone_appropriateness=0.0, information_density=0.0, safety_score=1.0,
            bias_score=0.0, toxicity_score=0.0, privacy_compliance=1.0,
            task_success_rate=0.0, instruction_following=0.0, format_compliance=0.0,
            creativity_score=0.0, response_time_ms=0.0, conversation_turns=0,
            user_satisfaction=0.0, engagement_score=0.0, token_efficiency=0.0,
            resource_usage=0.0, error_rate=1.0, availability=0.0,
            overall_score=0.0, confidence_score=0.0,
            evaluation_time=datetime.utcnow(),
            evaluator_version="1.0",
            model_version="error"
        )
    
    async def _calculate_aggregate_metrics(self, task_results: List[AgentEvaluationResult]) -> Dict[str, Any]:
        """Calculate aggregate metrics from task results"""
        if not task_results:
            return {}
        
        # Get all numeric metric names
        metric_names = [
            'accuracy', 'relevance', 'completeness', 'consistency', 'efficiency',
            'reasoning_quality', 'tool_usage_effectiveness', 'context_utilization',
            'error_handling', 'adaptability', 'clarity', 'conciseness',
            'tone_appropriateness', 'information_density', 'safety_score',
            'bias_score', 'toxicity_score', 'privacy_compliance',
            'task_success_rate', 'instruction_following', 'format_compliance',
            'creativity_score', 'user_satisfaction', 'engagement_score',
            'token_efficiency', 'resource_usage', 'error_rate', 'availability',
            'overall_score', 'confidence_score'
        ]
        
        aggregate = {}
        for metric_name in metric_names:
            values = [getattr(result, metric_name) for result in task_results]
            
            aggregate[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values))
            }
        
        # Add performance metrics
        response_times = [r.response_time_ms for r in task_results]
        aggregate['performance'] = {
            'avg_response_time_ms': float(np.mean(response_times)),
            'p95_response_time_ms': float(np.percentile(response_times, 95)),
            'success_rate': sum(1 for r in task_results if r.overall_score > 0.5) / len(task_results)
        }
        
        return aggregate
    
    async def _generate_performance_profile(
        self,
        agent_info: Dict[str, Any],
        task_results: List[AgentEvaluationResult],
        aggregate_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate performance profile for an agent"""
        
        profile = {
            'agent_summary': {
                'name': agent_info['agent_name'],
                'type': agent_info['agent_type'],
                'capabilities': agent_info['capabilities'],
                'total_evaluations': len(task_results)
            },
            'strengths': [],
            'weaknesses': [],
            'recommendations': [],
            'use_cases': []
        }
        
        # Identify strengths (metrics above 0.8)
        for metric, stats in aggregate_metrics.items():
            if isinstance(stats, dict) and stats.get('mean', 0) > 0.8:
                profile['strengths'].append({
                    'metric': metric,
                    'score': stats['mean'],
                    'description': f"Excellent {metric.replace('_', ' ')}"
                })
        
        # Identify weaknesses (metrics below 0.5)
        for metric, stats in aggregate_metrics.items():
            if isinstance(stats, dict) and stats.get('mean', 0) < 0.5:
                profile['weaknesses'].append({
                    'metric': metric,
                    'score': stats['mean'],
                    'description': f"Poor {metric.replace('_', ' ')}"
                })
        
        # Generate recommendations
        if aggregate_metrics.get('safety_score', {}).get('mean', 1.0) < 0.8:
            profile['recommendations'].append("Improve safety measures and content filtering")
        
        if aggregate_metrics.get('response_time_ms', {}).get('mean', 0) > 5000:
            profile['recommendations'].append("Optimize for faster response times")
        
        # Suggest use cases based on strengths
        agent_type = AgentType(agent_info['agent_type'])
        if agent_type == AgentType.RAG:
            profile['use_cases'] = ['Document Q&A', 'Information Retrieval', 'Knowledge Base Search']
        elif agent_type == AgentType.CONVERSATIONAL:
            profile['use_cases'] = ['Customer Support', 'Virtual Assistant', 'Chat Bot']
        elif agent_type == AgentType.CREATIVE:
            profile['use_cases'] = ['Content Generation', 'Creative Writing', 'Brainstorming']
        
        return profile
    
    async def _perform_comparison_analysis(
        self,
        agent_results: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform statistical comparison analysis between agents"""
        
        analysis = {
            'overall_ranking': [],
            'metric_rankings': {},
            'statistical_tests': {},
            'recommendations': {}
        }
        
        # Extract aggregate metrics for each agent
        agent_metrics = {}
        for agent_id, results in agent_results.items():
            if results['status'] == 'success':
                agent_metrics[agent_id] = results['aggregate_metrics']
        
        if len(agent_metrics) < 2:
            return analysis
        
        # Overall ranking by overall_score
        overall_scores = {
            agent_id: metrics.get('overall_score', {}).get('mean', 0.0)
            for agent_id, metrics in agent_metrics.items()
        }
        
        analysis['overall_ranking'] = sorted(
            overall_scores.items(), key=lambda x: x[1], reverse=True
        )
        
        # Metric-by-metric rankings
        key_metrics = ['accuracy', 'relevance', 'efficiency', 'safety_score', 'creativity_score']
        
        for metric in key_metrics:
            metric_scores = {
                agent_id: metrics.get(metric, {}).get('mean', 0.0)
                for agent_id, metrics in agent_metrics.items()
                if metric in metrics
            }
            
            if metric_scores:
                analysis['metric_rankings'][metric] = sorted(
                    metric_scores.items(), key=lambda x: x[1], reverse=True
                )
        
        # Generate recommendations for each agent
        for agent_id in agent_metrics.keys():
            agent_recommendations = []
            
            # Get agent's performance
            agent_perf = agent_metrics[agent_id]
            
            # Compare with best performer
            best_agent = analysis['overall_ranking'][0][0]
            if agent_id != best_agent:
                best_perf = agent_metrics[best_agent]
                
                # Find areas for improvement
                for metric in key_metrics:
                    if (metric in agent_perf and metric in best_perf and
                        agent_perf[metric]['mean'] < best_perf[metric]['mean'] - 0.1):
                        agent_recommendations.append(
                            f"Improve {metric.replace('_', ' ')} (current: {agent_perf[metric]['mean']:.2f}, best: {best_perf[metric]['mean']:.2f})"
                        )
            
            analysis['recommendations'][agent_id] = agent_recommendations
        
        return analysis
    
    async def get_evaluation_results(self, evaluation_id: str) -> Dict[str, Any]:
        """Get evaluation results by ID"""
        if evaluation_id not in self.evaluation_cache:
            return {
                'status': 'not_found',
                'message': f'Evaluation {evaluation_id} not found'
            }
        
        return {
            'status': 'success',
            **self.evaluation_cache[evaluation_id]
        }
    
    async def get_comparison_results(self, comparison_id: str) -> Dict[str, Any]:
        """Get comparison results by ID"""
        if comparison_id not in self.comparison_cache:
            return {
                'status': 'not_found',
                'message': f'Comparison {comparison_id} not found'
            }
        
        return {
            'status': 'success',
            **self.comparison_cache[comparison_id]
        }
    
    async def list_registered_agents(self) -> Dict[str, Any]:
        """List all registered agents"""
        agents = []
        for agent_id, agent_info in self.agent_registry.items():
            agents.append({
                'agent_id': agent_id,
                'agent_name': agent_info['agent_name'],
                'agent_type': agent_info['agent_type'],
                'capabilities': agent_info['capabilities'],
                'evaluation_count': agent_info['evaluation_count'],
                'last_evaluation': agent_info['last_evaluation'].isoformat() if agent_info['last_evaluation'] else None,
                'registered_at': agent_info['registered_at'].isoformat()
            })
        
        return {
            'status': 'success',
            'agents': agents,
            'total_agents': len(agents)
        }

# Global service instance
agent_evaluator_service = AgentEvaluatorService()
