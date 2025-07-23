"""
LLM Evaluator Service
Comprehensive LLM evaluation with multiple metrics and benchmarks
"""

import asyncio
import json
import logging
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import uuid
from collections import defaultdict, deque
import re
import numpy as np
from dataclasses import dataclass
import statistics

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import get_database

logger = logging.getLogger(__name__)
settings = get_settings()

@dataclass
class LLMEvaluationResult:
    """Data class for LLM evaluation results"""
    task_id: str
    task_type: str
    prompt: str
    generated_response: str
    reference_response: Optional[str]
    
    # Content Quality Metrics
    relevance: float
    accuracy: float
    completeness: float
    coherence: float
    consistency: float
    
    # Language Quality Metrics
    fluency: float
    grammatical_correctness: float
    clarity: float
    conciseness: float
    
    # Semantic Metrics
    semantic_similarity: float
    factual_correctness: float
    logical_consistency: float
    
    # Automated Metrics
    bleu_score: float
    rouge_1: float
    rouge_2: float
    rouge_l: float
    bert_score_f1: float
    perplexity: Optional[float]
    
    # Task-Specific Metrics
    task_success_rate: float
    instruction_following: float
    format_compliance: float
    
    # Safety & Bias Metrics
    toxicity_score: float
    bias_score: float
    safety_score: float
    
    # Creativity & Style Metrics
    creativity_score: float
    style_consistency: float
    tone_appropriateness: float
    
    # Performance Metrics
    response_time_ms: float
    token_count: int
    cost_estimate: float
    
    # Overall Metrics
    overall_score: float
    confidence_score: float
    
    # Metadata
    evaluation_time: datetime
    model_name: str
    model_version: str
    temperature: Optional[float]
    max_tokens: Optional[int]
    
class LLMEvaluatorService:
    """
    Comprehensive LLM evaluation service with multiple evaluation frameworks
    """
    
    def __init__(self):
        self.evaluation_cache = {}
        self.evaluation_history = defaultdict(list)
        self.benchmark_datasets = {}
        self.sentence_transformer = None
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        
        # Evaluation configuration
        self.eval_config = {
            'batch_size': 16,
            'timeout_seconds': 60,
            'max_concurrent': 5,
            'temperature_range': [0.0, 1.0],
            'quality_thresholds': {
                'relevance': 0.7,
                'fluency': 0.8,
                'safety': 0.9,
                'overall': 0.75
            }
        }
        
        # Task type definitions
        self.task_types = {
            'generation': {
                'description': 'General text generation',
                'metrics': ['fluency', 'coherence', 'creativity'],
                'weight': {'content': 0.4, 'language': 0.4, 'creativity': 0.2}
            },
            'summarization': {
                'description': 'Text summarization',
                'metrics': ['relevance', 'completeness', 'conciseness'],
                'weight': {'content': 0.5, 'language': 0.3, 'semantic': 0.2}
            },
            'qa': {
                'description': 'Question answering',
                'metrics': ['accuracy', 'completeness', 'relevance'],
                'weight': {'content': 0.6, 'semantic': 0.3, 'language': 0.1}
            },
            'translation': {
                'description': 'Language translation',
                'metrics': ['accuracy', 'fluency', 'semantic_similarity'],
                'weight': {'semantic': 0.4, 'language': 0.4, 'automated': 0.2}
            },
            'classification': {
                'description': 'Text classification',
                'metrics': ['accuracy', 'consistency', 'format_compliance'],
                'weight': {'task_specific': 0.6, 'content': 0.3, 'language': 0.1}
            },
            'code_generation': {
                'description': 'Code generation',
                'metrics': ['accuracy', 'functionality', 'style_consistency'],
                'weight': {'task_specific': 0.7, 'content': 0.2, 'language': 0.1}
            },
            'creative_writing': {
                'description': 'Creative writing tasks',
                'metrics': ['creativity', 'coherence', 'style_consistency'],
                'weight': {'creativity': 0.5, 'language': 0.3, 'content': 0.2}
            }
        }
        
    async def initialize_models(self):
        """
        Initialize evaluation models and resources
        """
        try:
            # Load sentence transformer for semantic similarity
            if self.sentence_transformer is None:
                self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize benchmark datasets
            await self._load_benchmark_datasets()
            
            logger.info("LLM evaluation models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM evaluation models: {e}")
            raise
    
    async def evaluate_llm(
        self,
        evaluation_id: str,
        llm_model: Any,
        test_cases: List[Dict[str, Any]],
        evaluation_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate an LLM model comprehensively
        
        Args:
            evaluation_id: Unique evaluation identifier
            llm_model: LLM model to evaluate
            test_cases: List of test cases with prompts and expected outputs
            evaluation_config: Optional evaluation configuration
        
        Returns:
            Comprehensive evaluation results
        """
        try:
            await self.initialize_models()
            
            config = {**self.eval_config, **(evaluation_config or {})}
            
            evaluation_results = {
                'evaluation_id': evaluation_id,
                'start_time': datetime.utcnow(),
                'config': config,
                'total_test_cases': len(test_cases),
                'test_results': [],
                'aggregate_metrics': {},
                'benchmark_scores': {},
                'performance_stats': {},
                'status': 'running'
            }
            
            # Process test cases in batches
            batch_size = config['batch_size']
            test_batches = [test_cases[i:i + batch_size] for i in range(0, len(test_cases), batch_size)]
            
            all_test_results = []
            
            for batch_idx, test_batch in enumerate(test_batches):
                logger.info(f"Processing batch {batch_idx + 1}/{len(test_batches)}")
                
                batch_results = await self._evaluate_test_batch(
                    test_batch,
                    llm_model,
                    config,
                    batch_idx * batch_size
                )
                
                all_test_results.extend(batch_results)
                
                # Update progress
                evaluation_results['processed_cases'] = len(all_test_results)
                evaluation_results['progress'] = len(all_test_results) / len(test_cases)
            
            evaluation_results['test_results'] = all_test_results
            
            # Calculate aggregate metrics
            aggregate_metrics = await self._calculate_aggregate_metrics(all_test_results)
            evaluation_results['aggregate_metrics'] = aggregate_metrics
            
            # Run benchmark evaluations
            benchmark_scores = await self._run_benchmark_evaluations(llm_model, config)
            evaluation_results['benchmark_scores'] = benchmark_scores
            
            # Calculate performance statistics
            performance_stats = await self._calculate_performance_stats(all_test_results)
            evaluation_results['performance_stats'] = performance_stats
            
            evaluation_results['end_time'] = datetime.utcnow()
            evaluation_results['duration_seconds'] = (
                evaluation_results['end_time'] - evaluation_results['start_time']
            ).total_seconds()
            evaluation_results['status'] = 'completed'
            
            # Store results
            self.evaluation_cache[evaluation_id] = evaluation_results
            
            return {
                'status': 'success',
                **evaluation_results
            }
            
        except Exception as e:
            logger.error(f"Failed to evaluate LLM: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'evaluation_id': evaluation_id
            }
    
    async def _evaluate_test_batch(
        self,
        test_cases: List[Dict[str, Any]],
        llm_model: Any,
        config: Dict[str, Any],
        start_idx: int
    ) -> List[LLMEvaluationResult]:
        """
        Evaluate a batch of test cases
        """
        batch_results = []
        
        # Create semaphore for concurrent processing
        semaphore = asyncio.Semaphore(config['max_concurrent'])
        
        async def process_single_case(i, test_case):
            async with semaphore:
                case_idx = start_idx + i
                try:
                    result = await self._evaluate_single_case(
                        task_id=f"task_{case_idx}",
                        test_case=test_case,
                        llm_model=llm_model,
                        config=config
                    )
                    return result
                except Exception as e:
                    logger.error(f"Error evaluating test case {case_idx}: {e}")
                    return self._create_error_result(f"task_{case_idx}", test_case, str(e))
        
        # Process cases concurrently
        tasks = [process_single_case(i, case) for i, case in enumerate(test_cases)]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = [r for r in batch_results if isinstance(r, LLMEvaluationResult)]
        
        return valid_results
    
    async def _evaluate_single_case(
        self,
        task_id: str,
        test_case: Dict[str, Any],
        llm_model: Any,
        config: Dict[str, Any]
    ) -> LLMEvaluationResult:
        """
        Evaluate a single test case
        """
        try:
            prompt = test_case['prompt']
            reference_response = test_case.get('reference_response')
            task_type = test_case.get('task_type', 'generation')
            
            # Generate response from LLM
            start_time = datetime.utcnow()
            
            llm_response = await self._get_llm_response(
                llm_model, prompt, test_case.get('parameters', {}), config
            )
            
            end_time = datetime.utcnow()
            response_time = (end_time - start_time).total_seconds() * 1000
            
            generated_response = llm_response.get('response', '')
            
            # Initialize metrics
            metrics = await self._calculate_all_metrics(
                prompt=prompt,
                generated_response=generated_response,
                reference_response=reference_response,
                task_type=task_type,
                test_case=test_case
            )
            
            # Calculate overall score
            overall_score = await self._calculate_overall_score(metrics, task_type)
            confidence_score = await self._calculate_confidence_score(metrics, generated_response)
            
            return LLMEvaluationResult(
                task_id=task_id,
                task_type=task_type,
                prompt=prompt,
                generated_response=generated_response,
                reference_response=reference_response,
                **metrics,
                overall_score=overall_score,
                confidence_score=confidence_score,
                response_time_ms=response_time,
                token_count=len(generated_response.split()) if generated_response else 0,
                cost_estimate=await self._estimate_cost(prompt, generated_response),
                evaluation_time=datetime.utcnow(),
                model_name=getattr(llm_model, 'model_name', 'unknown'),
                model_version=getattr(llm_model, 'version', '1.0'),
                temperature=test_case.get('parameters', {}).get('temperature'),
                max_tokens=test_case.get('parameters', {}).get('max_tokens')
            )
            
        except Exception as e:
            logger.error(f"Error evaluating single case {task_id}: {e}")
            return self._create_error_result(task_id, test_case, str(e))
    
    async def _get_llm_response(
        self,
        llm_model: Any,
        prompt: str,
        parameters: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get response from LLM model
        """
        try:
            # Create timeout for response generation
            timeout = config.get('timeout_seconds', 60)
            
            async def generate_response():
                if hasattr(llm_model, 'generate_async'):
                    return await llm_model.generate_async(prompt, **parameters)
                elif hasattr(llm_model, 'generate'):
                    return llm_model.generate(prompt, **parameters)
                else:
                    # Fallback simulation
                    await asyncio.sleep(0.1)  # Simulate processing time
                    return f"Generated response for: {prompt[:50]}..."
            
            response = await asyncio.wait_for(generate_response(), timeout=timeout)
            
            return {
                'response': response,
                'success': True
            }
            
        except asyncio.TimeoutError:
            return {
                'response': '',
                'success': False,
                'error': 'Timeout'
            }
        except Exception as e:
            return {
                'response': '',
                'success': False,
                'error': str(e)
            }
    
    async def _calculate_all_metrics(
        self,
        prompt: str,
        generated_response: str,
        reference_response: Optional[str],
        task_type: str,
        test_case: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calculate all evaluation metrics
        """
        metrics = {
            # Content Quality Metrics
            'relevance': 0.0,
            'accuracy': 0.0,
            'completeness': 0.0,
            'coherence': 0.0,
            'consistency': 0.0,
            
            # Language Quality Metrics
            'fluency': 0.0,
            'grammatical_correctness': 0.0,
            'clarity': 0.0,
            'conciseness': 0.0,
            
            # Semantic Metrics
            'semantic_similarity': 0.0,
            'factual_correctness': 0.0,
            'logical_consistency': 0.0,
            
            # Automated Metrics
            'bleu_score': 0.0,
            'rouge_1': 0.0,
            'rouge_2': 0.0,
            'rouge_l': 0.0,
            'bert_score_f1': 0.0,
            'perplexity': None,
            
            # Task-Specific Metrics
            'task_success_rate': 0.0,
            'instruction_following': 0.0,
            'format_compliance': 0.0,
            
            # Safety & Bias Metrics
            'toxicity_score': 0.0,
            'bias_score': 0.0,
            'safety_score': 1.0,
            
            # Creativity & Style Metrics
            'creativity_score': 0.0,
            'style_consistency': 0.0,
            'tone_appropriateness': 0.0
        }
        
        if not generated_response:
            return metrics
        
        try:
            # 1. Content Quality Metrics
            content_metrics = await self._evaluate_content_quality(
                prompt, generated_response, reference_response, task_type
            )
            metrics.update(content_metrics)
            
            # 2. Language Quality Metrics
            language_metrics = await self._evaluate_language_quality(generated_response)
            metrics.update(language_metrics)
            
            # 3. Semantic Metrics
            if reference_response:
                semantic_metrics = await self._evaluate_semantic_quality(
                    generated_response, reference_response
                )
                metrics.update(semantic_metrics)
            
            # 4. Automated Metrics
            if reference_response:
                automated_metrics = await self._calculate_automated_metrics(
                    generated_response, reference_response
                )
                metrics.update(automated_metrics)
            
            # 5. Task-Specific Metrics
            task_metrics = await self._evaluate_task_specific_metrics(
                prompt, generated_response, task_type, test_case
            )
            metrics.update(task_metrics)
            
            # 6. Safety & Bias Metrics
            safety_metrics = await self._evaluate_safety_and_bias(generated_response)
            metrics.update(safety_metrics)
            
            # 7. Creativity & Style Metrics
            style_metrics = await self._evaluate_creativity_and_style(
                prompt, generated_response, task_type
            )
            metrics.update(style_metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return metrics
    
    async def _evaluate_content_quality(
        self,
        prompt: str,
        response: str,
        reference: Optional[str],
        task_type: str
    ) -> Dict[str, float]:
        """
        Evaluate content quality metrics
        """
        try:
            metrics = {}
            
            # Relevance: How relevant is the response to the prompt
            metrics['relevance'] = await self._calculate_relevance(prompt, response)
            
            # Accuracy: How accurate is the response (if reference available)
            if reference:
                metrics['accuracy'] = await self._calculate_accuracy(response, reference)
            else:
                metrics['accuracy'] = await self._estimate_accuracy(prompt, response, task_type)
            
            # Completeness: How complete is the response
            metrics['completeness'] = await self._calculate_completeness(prompt, response)
            
            # Coherence: How coherent is the response
            metrics['coherence'] = await self._calculate_coherence(response)
            
            # Consistency: Internal consistency of the response
            metrics['consistency'] = await self._calculate_consistency(response)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating content quality: {e}")
            return {
                'relevance': 0.0,
                'accuracy': 0.0,
                'completeness': 0.0,
                'coherence': 0.0,
                'consistency': 0.0
            }
    
    async def _evaluate_language_quality(self, response: str) -> Dict[str, float]:
        """
        Evaluate language quality metrics
        """
        try:
            metrics = {}
            
            # Fluency: Natural flow and readability
            metrics['fluency'] = await self._calculate_fluency(response)
            
            # Grammatical correctness: Grammar and syntax
            metrics['grammatical_correctness'] = await self._calculate_grammar_score(response)
            
            # Clarity: How clear and understandable is the response
            metrics['clarity'] = await self._calculate_clarity(response)
            
            # Conciseness: Appropriate length and brevity
            metrics['conciseness'] = await self._calculate_conciseness(response)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating language quality: {e}")
            return {
                'fluency': 0.0,
                'grammatical_correctness': 0.0,
                'clarity': 0.0,
                'conciseness': 0.0
            }
    
    async def _evaluate_semantic_quality(
        self,
        response: str,
        reference: str
    ) -> Dict[str, float]:
        """
        Evaluate semantic quality metrics
        """
        try:
            metrics = {}
            
            # Semantic similarity
            metrics['semantic_similarity'] = await self._calculate_semantic_similarity(response, reference)
            
            # Factual correctness (simplified)
            metrics['factual_correctness'] = await self._estimate_factual_correctness(response, reference)
            
            # Logical consistency
            metrics['logical_consistency'] = await self._calculate_logical_consistency(response)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating semantic quality: {e}")
            return {
                'semantic_similarity': 0.0,
                'factual_correctness': 0.0,
                'logical_consistency': 0.0
            }
    
    async def _calculate_automated_metrics(
        self,
        response: str,
        reference: str
    ) -> Dict[str, float]:
        """
        Calculate automated evaluation metrics
        """
        try:
            metrics = {}
            
            # BLEU Score
            try:
                response_tokens = response.split()
                reference_tokens = [reference.split()]
                metrics['bleu_score'] = sentence_bleu(reference_tokens, response_tokens)
            except:
                metrics['bleu_score'] = 0.0
            
            # ROUGE Scores
            try:
                rouge_scores = self.rouge_scorer.score(reference, response)
                metrics['rouge_1'] = rouge_scores['rouge1'].fmeasure
                metrics['rouge_2'] = rouge_scores['rouge2'].fmeasure
                metrics['rouge_l'] = rouge_scores['rougeL'].fmeasure
            except:
                metrics['rouge_1'] = metrics['rouge_2'] = metrics['rouge_l'] = 0.0
            
            # BERT Score
            try:
                P, R, F1 = bert_score([response], [reference], lang='en', verbose=False)
                metrics['bert_score_f1'] = F1.mean().item()
            except:
                metrics['bert_score_f1'] = 0.0
            
            # Perplexity (placeholder - would need specific model)
            metrics['perplexity'] = None
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating automated metrics: {e}")
            return {
                'bleu_score': 0.0,
                'rouge_1': 0.0,
                'rouge_2': 0.0,
                'rouge_l': 0.0,
                'bert_score_f1': 0.0,
                'perplexity': None
            }
    
    async def _evaluate_task_specific_metrics(
        self,
        prompt: str,
        response: str,
        task_type: str,
        test_case: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Evaluate task-specific metrics
        """
        try:
            metrics = {}
            
            # Task success rate
            metrics['task_success_rate'] = await self._calculate_task_success(response, task_type, test_case)
            
            # Instruction following
            metrics['instruction_following'] = await self._calculate_instruction_following(prompt, response)
            
            # Format compliance
            expected_format = test_case.get('expected_format')
            metrics['format_compliance'] = await self._calculate_format_compliance(response, expected_format)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating task-specific metrics: {e}")
            return {
                'task_success_rate': 0.0,
                'instruction_following': 0.0,
                'format_compliance': 0.0
            }
    
    async def _evaluate_safety_and_bias(self, response: str) -> Dict[str, float]:
        """
        Evaluate safety and bias metrics
        """
        try:
            metrics = {}
            
            # Toxicity score (simplified heuristics)
            metrics['toxicity_score'] = await self._calculate_toxicity_score(response)
            
            # Bias score (simplified heuristics)
            metrics['bias_score'] = await self._calculate_bias_score(response)
            
            # Overall safety score
            metrics['safety_score'] = 1.0 - max(metrics['toxicity_score'], metrics['bias_score'])
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating safety and bias: {e}")
            return {
                'toxicity_score': 0.0,
                'bias_score': 0.0,
                'safety_score': 1.0
            }
    
    async def _evaluate_creativity_and_style(
        self,
        prompt: str,
        response: str,
        task_type: str
    ) -> Dict[str, float]:
        """
        Evaluate creativity and style metrics
        """
        try:
            metrics = {}
            
            # Creativity score
            if task_type in ['creative_writing', 'generation']:
                metrics['creativity_score'] = await self._calculate_creativity_score(response)
            else:
                metrics['creativity_score'] = 0.5  # Neutral for non-creative tasks
            
            # Style consistency
            metrics['style_consistency'] = await self._calculate_style_consistency(response)
            
            # Tone appropriateness
            metrics['tone_appropriateness'] = await self._calculate_tone_appropriateness(prompt, response)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating creativity and style: {e}")
            return {
                'creativity_score': 0.0,
                'style_consistency': 0.0,
                'tone_appropriateness': 0.0
            }
    
    # Helper methods for metric calculations
    async def _calculate_relevance(self, prompt: str, response: str) -> float:
        """Calculate relevance score"""
        try:
            return await self._calculate_semantic_similarity(prompt, response)
        except:
            return 0.0
    
    async def _calculate_accuracy(self, response: str, reference: str) -> float:
        """Calculate accuracy against reference"""
        try:
            # Use semantic similarity as proxy for accuracy
            return await self._calculate_semantic_similarity(response, reference)
        except:
            return 0.0
    
    async def _estimate_accuracy(self, prompt: str, response: str, task_type: str) -> float:
        """Estimate accuracy without reference"""
        try:
            # Simple heuristics for accuracy estimation
            if not response or len(response.strip()) < 5:
                return 0.0
            
            # Task-specific accuracy estimation
            if task_type == 'qa':
                # For Q&A, check if response addresses the question
                return min(0.8, len(response.split()) / 20)  # Rough heuristic
            elif task_type == 'classification':
                # Check if response looks like a valid classification
                return 0.7 if len(response.split()) < 10 else 0.5
            else:
                return 0.6  # Default moderate score
        except:
            return 0.0
    
    async def _calculate_completeness(self, prompt: str, response: str) -> float:
        """Calculate completeness score"""
        try:
            if not response:
                return 0.0
            
            # Simple length-based heuristic
            response_length = len(response.split())
            
            if response_length < 5:
                return 0.2
            elif response_length < 20:
                return 0.6
            elif response_length < 100:
                return 0.9
            else:
                return 0.8  # Very long responses might be too verbose
        except:
            return 0.0
    
    async def _calculate_coherence(self, response: str) -> float:
        """Calculate coherence score"""
        try:
            if not response:
                return 0.0
            
            sentences = re.split(r'[.!?]+', response)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) < 2:
                return 1.0 if len(sentences) == 1 else 0.0
            
            # Calculate similarity between adjacent sentences
            similarities = []
            for i in range(len(sentences) - 1):
                sim = await self._calculate_semantic_similarity(sentences[i], sentences[i + 1])
                similarities.append(sim)
            
            return float(np.mean(similarities)) if similarities else 0.0
        except:
            return 0.0
    
    async def _calculate_consistency(self, response: str) -> float:
        """Calculate internal consistency"""
        try:
            if not response:
                return 0.0
            
            # Check for contradictory statements (simplified)
            sentences = re.split(r'[.!?]+', response)
            sentences = [s.strip().lower() for s in sentences if s.strip()]
            
            # Look for contradictory keywords
            contradictory_pairs = [
                ('yes', 'no'), ('always', 'never'), ('true', 'false'),
                ('good', 'bad'), ('possible', 'impossible')
            ]
            
            contradiction_score = 0
            for pair in contradictory_pairs:
                if any(pair[0] in s for s in sentences) and any(pair[1] in s for s in sentences):
                    contradiction_score += 1
            
            # Normalize (fewer contradictions = higher consistency)
            consistency = max(0, 1 - (contradiction_score / len(contradictory_pairs)))
            return consistency
        except:
            return 0.5
    
    async def _calculate_fluency(self, response: str) -> float:
        """Calculate fluency score"""
        try:
            if not response:
                return 0.0
            
            score = 1.0
            words = response.split()
            
            if len(words) == 0:
                return 0.0
            
            # Check average sentence length
            sentences = re.split(r'[.!?]+', response)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if sentences:
                avg_sentence_length = np.mean([len(s.split()) for s in sentences])
                if avg_sentence_length < 3 or avg_sentence_length > 40:
                    score *= 0.8
            
            # Check for repetition
            word_counts = {}
            for word in words:
                word_lower = word.lower()
                word_counts[word_lower] = word_counts.get(word_lower, 0) + 1
            
            max_repetition = max(word_counts.values()) if word_counts else 1
            repetition_ratio = max_repetition / len(words)
            
            if repetition_ratio > 0.3:
                score *= 0.6
            
            return float(max(0.0, score))
        except:
            return 0.0
    
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
    
    # Placeholder implementations for remaining methods
    async def _calculate_grammar_score(self, response: str) -> float:
        """Calculate grammar score (simplified)"""
        return 0.8  # Placeholder
    
    async def _calculate_clarity(self, response: str) -> float:
        """Calculate clarity score"""
        return 0.7  # Placeholder
    
    async def _calculate_conciseness(self, response: str) -> float:
        """Calculate conciseness score"""
        return 0.8  # Placeholder
    
    async def _estimate_factual_correctness(self, response: str, reference: str) -> float:
        """Estimate factual correctness"""
        return await self._calculate_semantic_similarity(response, reference)
    
    async def _calculate_logical_consistency(self, response: str) -> float:
        """Calculate logical consistency"""
        return 0.8  # Placeholder
    
    async def _calculate_task_success(self, response: str, task_type: str, test_case: Dict[str, Any]) -> float:
        """Calculate task success rate"""
        return 0.7  # Placeholder
    
    async def _calculate_instruction_following(self, prompt: str, response: str) -> float:
        """Calculate instruction following score"""
        return 0.8  # Placeholder
    
    async def _calculate_format_compliance(self, response: str, expected_format: Optional[str]) -> float:
        """Calculate format compliance score"""
        return 0.9 if not expected_format else 0.8  # Placeholder
    
    async def _calculate_toxicity_score(self, response: str) -> float:
        """Calculate toxicity score"""
        return 0.1  # Placeholder - low toxicity
    
    async def _calculate_bias_score(self, response: str) -> float:
        """Calculate bias score"""
        return 0.1  # Placeholder - low bias
    
    async def _calculate_creativity_score(self, response: str) -> float:
        """Calculate creativity score"""
        return 0.6  # Placeholder
    
    async def _calculate_style_consistency(self, response: str) -> float:
        """Calculate style consistency"""
        return 0.8  # Placeholder
    
    async def _calculate_tone_appropriateness(self, prompt: str, response: str) -> float:
        """Calculate tone appropriateness"""
        return 0.8  # Placeholder
    
    async def _calculate_overall_score(self, metrics: Dict[str, float], task_type: str) -> float:
        """Calculate overall score based on task type"""
        try:
            task_config = self.task_types.get(task_type, self.task_types['generation'])
            weights = task_config['weight']
            
            # Group metrics by category
            content_metrics = ['relevance', 'accuracy', 'completeness', 'coherence', 'consistency']
            language_metrics = ['fluency', 'grammatical_correctness', 'clarity', 'conciseness']
            semantic_metrics = ['semantic_similarity', 'factual_correctness', 'logical_consistency']
            automated_metrics = ['bleu_score', 'rouge_1', 'rouge_2', 'rouge_l', 'bert_score_f1']
            task_metrics = ['task_success_rate', 'instruction_following', 'format_compliance']
            creativity_metrics = ['creativity_score', 'style_consistency', 'tone_appropriateness']
            
            # Calculate category scores
            content_score = np.mean([metrics.get(m, 0.0) for m in content_metrics])
            language_score = np.mean([metrics.get(m, 0.0) for m in language_metrics])
            semantic_score = np.mean([metrics.get(m, 0.0) for m in semantic_metrics])
            automated_score = np.mean([metrics.get(m, 0.0) for m in automated_metrics if metrics.get(m, 0.0) > 0])
            task_score = np.mean([metrics.get(m, 0.0) for m in task_metrics])
            creativity_score = np.mean([metrics.get(m, 0.0) for m in creativity_metrics])
            
            # Weighted combination based on task type
            overall_score = (
                content_score * weights.get('content', 0.3) +
                language_score * weights.get('language', 0.3) +
                semantic_score * weights.get('semantic', 0.2) +
                automated_score * weights.get('automated', 0.1) +
                task_score * weights.get('task_specific', 0.3) +
                creativity_score * weights.get('creativity', 0.1)
            )
            
            return float(max(0.0, min(1.0, overall_score)))
        except:
            return 0.0
    
    async def _calculate_confidence_score(self, metrics: Dict[str, float], response: str) -> float:
        """Calculate confidence score"""
        try:
            confidence_factors = []
            
            # Response length factor
            if response:
                response_length = len(response.split())
                length_confidence = min(1.0, max(0.0, (response_length - 5) / 50))
                confidence_factors.append(length_confidence)
            
            # Metric consistency factor
            metric_values = [v for v in metrics.values() if isinstance(v, (int, float)) and v > 0]
            if metric_values:
                variance = np.var(metric_values)
                consistency_confidence = max(0.0, 1.0 - variance)
                confidence_factors.append(consistency_confidence)
            
            # Safety factor
            safety_confidence = metrics.get('safety_score', 1.0)
            confidence_factors.append(safety_confidence)
            
            return float(np.mean(confidence_factors)) if confidence_factors else 0.0
        except:
            return 0.0
    
    async def _estimate_cost(self, prompt: str, response: str) -> float:
        """Estimate API cost"""
        # Simple token-based cost estimation
        input_tokens = len(prompt.split())
        output_tokens = len(response.split()) if response else 0
        
        # Rough cost estimate (per 1000 tokens)
        input_cost_per_1k = 0.001  # $0.001 per 1k input tokens
        output_cost_per_1k = 0.002  # $0.002 per 1k output tokens
        
        cost = (input_tokens / 1000) * input_cost_per_1k + (output_tokens / 1000) * output_cost_per_1k
        return round(cost, 6)
    
    def _create_error_result(self, task_id: str, test_case: Dict[str, Any], error_msg: str) -> LLMEvaluationResult:
        """Create error result for failed evaluations"""
        return LLMEvaluationResult(
            task_id=task_id,
            task_type=test_case.get('task_type', 'unknown'),
            prompt=test_case.get('prompt', ''),
            generated_response='',
            reference_response=test_case.get('reference_response'),
            relevance=0.0, accuracy=0.0, completeness=0.0, coherence=0.0, consistency=0.0,
            fluency=0.0, grammatical_correctness=0.0, clarity=0.0, conciseness=0.0,
            semantic_similarity=0.0, factual_correctness=0.0, logical_consistency=0.0,
            bleu_score=0.0, rouge_1=0.0, rouge_2=0.0, rouge_l=0.0, bert_score_f1=0.0, perplexity=None,
            task_success_rate=0.0, instruction_following=0.0, format_compliance=0.0,
            toxicity_score=0.0, bias_score=0.0, safety_score=1.0,
            creativity_score=0.0, style_consistency=0.0, tone_appropriateness=0.0,
            response_time_ms=0.0, token_count=0, cost_estimate=0.0,
            overall_score=0.0, confidence_score=0.0,
            evaluation_time=datetime.utcnow(),
            model_name='error', model_version='1.0',
            temperature=None, max_tokens=None
        )
    
    async def _calculate_aggregate_metrics(self, test_results: List[LLMEvaluationResult]) -> Dict[str, Any]:
        """Calculate aggregate metrics from test results"""
        if not test_results:
            return {}
        
        # Get all metric names
        metric_names = [
            'relevance', 'accuracy', 'completeness', 'coherence', 'consistency',
            'fluency', 'grammatical_correctness', 'clarity', 'conciseness',
            'semantic_similarity', 'factual_correctness', 'logical_consistency',
            'bleu_score', 'rouge_1', 'rouge_2', 'rouge_l', 'bert_score_f1',
            'task_success_rate', 'instruction_following', 'format_compliance',
            'toxicity_score', 'bias_score', 'safety_score',
            'creativity_score', 'style_consistency', 'tone_appropriateness',
            'overall_score', 'confidence_score'
        ]
        
        aggregate = {}
        for metric_name in metric_names:
            values = [getattr(result, metric_name) for result in test_results 
                     if getattr(result, metric_name) is not None]
            
            if values:
                aggregate[metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values))
                }
        
        return aggregate
    
    async def _calculate_performance_stats(self, test_results: List[LLMEvaluationResult]) -> Dict[str, Any]:
        """Calculate performance statistics"""
        if not test_results:
            return {}
        
        response_times = [r.response_time_ms for r in test_results]
        token_counts = [r.token_count for r in test_results]
        costs = [r.cost_estimate for r in test_results]
        
        return {
            'response_time_ms': {
                'mean': float(np.mean(response_times)),
                'p50': float(np.percentile(response_times, 50)),
                'p95': float(np.percentile(response_times, 95))
            },
            'token_statistics': {
                'mean_tokens': float(np.mean(token_counts)),
                'total_tokens': sum(token_counts)
            },
            'cost_statistics': {
                'total_cost': sum(costs),
                'cost_per_test': float(np.mean(costs))
            },
            'success_rate': sum(1 for r in test_results if r.overall_score > 0.5) / len(test_results)
        }
    
    async def _load_benchmark_datasets(self):
        """Load benchmark datasets for evaluation"""
        # Placeholder for loading benchmark datasets
        self.benchmark_datasets = {
            'general_qa': [],
            'summarization': [],
            'translation': [],
            'code_generation': []
        }
    
    async def _run_benchmark_evaluations(self, llm_model: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run benchmark evaluations"""
        # Placeholder for benchmark evaluations
        return {
            'general_qa_score': 0.75,
            'summarization_score': 0.68,
            'translation_score': 0.82,
            'code_generation_score': 0.71
        }

# Global service instance
llm_evaluator_service = LLMEvaluatorService()
