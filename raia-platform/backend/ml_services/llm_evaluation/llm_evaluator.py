# LLM Evaluation Service - Core Evaluation Engine
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import re
import statistics
from concurrent.futures import ThreadPoolExecutor
import openai
import anthropic
from transformers import AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
from bert_score import BERTScorer
import nltk
from nltk.translate.bleu_score import sentence_bleu
from textstat import flesch_reading_ease, flesch_kincaid_grade

from .models import LLMModel, LLMEvaluation, LLMSafetyEvaluation, LLMBenchmarkResult, PromptTemplate, EvaluationDataset
from ..exceptions import EvaluationError, ModelNotFoundError, ValidationError

logger = logging.getLogger(__name__)

class LLMEvaluationEngine:
    """Core engine for LLM evaluation and benchmarking"""
    
    def __init__(self, db: Session, openai_api_key: str = None, anthropic_api_key: str = None):
        self.db = db
        self.openai_client = openai.OpenAI(api_key=openai_api_key) if openai_api_key else None
        self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key) if anthropic_api_key else None
        
        # Initialize evaluation tools
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    # ========================================================================================
    # CORE LLM EVALUATION METHODS
    # ========================================================================================
    
    async def evaluate_llm_model(self,
                                model_id: str,
                                evaluation_config: Dict[str, Any],
                                user_id: str) -> LLMEvaluation:
        """Comprehensive evaluation of an LLM model"""
        
        # Get model configuration
        llm_model = self.db.query(LLMModel).filter(LLMModel.id == model_id).first()
        if not llm_model:
            raise ModelNotFoundError(f"LLM model {model_id} not found")
        
        # Create evaluation record
        evaluation = LLMEvaluation(
            llm_model_id=model_id,
            evaluation_name=evaluation_config['evaluation_name'],
            evaluation_type=evaluation_config.get('evaluation_type', 'comprehensive'),
            evaluation_suite=evaluation_config.get('evaluation_suite'),
            dataset_name=evaluation_config.get('dataset_name'),
            task_type=evaluation_config.get('task_type', 'text_generation'),
            evaluation_prompt_template=evaluation_config.get('prompt_template'),
            num_samples=evaluation_config.get('num_samples', 100),
            sample_selection_strategy=evaluation_config.get('sample_selection_strategy', 'random'),
            status='running',
            started_at=datetime.utcnow(),
            evaluation_config=evaluation_config,
            random_seed=evaluation_config.get('random_seed', 42),
            model_parameters_used=llm_model.model_parameters,
            created_by=user_id
        )
        
        self.db.add(evaluation)
        self.db.commit()
        self.db.refresh(evaluation)
        
        try:
            # Load evaluation dataset
            dataset = await self._load_evaluation_dataset(evaluation_config)
            
            # Generate model responses
            responses = await self._generate_model_responses(llm_model, dataset, evaluation_config)
            
            # Calculate performance metrics
            metrics = await self._calculate_performance_metrics(responses, dataset, evaluation_config)
            
            # Calculate semantic metrics
            semantic_metrics = await self._calculate_semantic_metrics(responses, dataset)
            
            # Calculate safety metrics
            safety_metrics = await self._calculate_safety_metrics(responses)
            
            # Calculate efficiency metrics
            efficiency_metrics = self._calculate_efficiency_metrics(responses)
            
            # Perform error analysis
            error_analysis = self._perform_error_analysis(responses, dataset)
            
            # Statistical analysis
            statistical_results = self._perform_statistical_analysis(metrics)
            
            # Update evaluation with results
            evaluation.status = 'completed'
            evaluation.completed_at = datetime.utcnow()
            evaluation.duration_minutes = int((evaluation.completed_at - evaluation.started_at).total_seconds() / 60)
            
            # Core metrics
            evaluation.overall_score = metrics.get('overall_score', 0.0)
            evaluation.accuracy = metrics.get('accuracy', 0.0)
            evaluation.bleu_score = metrics.get('bleu_score', 0.0)
            evaluation.rouge_1 = metrics.get('rouge_1', 0.0)
            evaluation.rouge_2 = metrics.get('rouge_2', 0.0)
            evaluation.rouge_l = metrics.get('rouge_l', 0.0)
            evaluation.bertscore_precision = metrics.get('bertscore_precision', 0.0)
            evaluation.bertscore_recall = metrics.get('bertscore_recall', 0.0)
            evaluation.bertscore_f1 = metrics.get('bertscore_f1', 0.0)
            
            # Semantic metrics
            evaluation.semantic_similarity = semantic_metrics.get('semantic_similarity', 0.0)
            evaluation.embedding_similarity = semantic_metrics.get('embedding_similarity', 0.0)
            
            # Quality metrics
            evaluation.fluency_score = metrics.get('fluency_score', 0.0)
            evaluation.coherence_score = metrics.get('coherence_score', 0.0)
            evaluation.relevance_score = metrics.get('relevance_score', 0.0)
            
            # Safety metrics
            evaluation.toxicity_score = safety_metrics.get('toxicity_score', 0.0)
            evaluation.bias_score = safety_metrics.get('bias_score', 0.0)
            evaluation.hallucination_rate = safety_metrics.get('hallucination_rate', 0.0)
            
            # Performance metrics
            evaluation.avg_response_time_ms = efficiency_metrics.get('avg_response_time_ms', 0)
            evaluation.total_tokens_generated = efficiency_metrics.get('total_tokens_generated', 0)
            evaluation.avg_tokens_per_response = efficiency_metrics.get('avg_tokens_per_response', 0)
            evaluation.token_efficiency = efficiency_metrics.get('token_efficiency', 0.0)
            
            # Cost metrics
            evaluation.total_cost_usd = efficiency_metrics.get('total_cost_usd', 0.0)
            evaluation.cost_per_sample = efficiency_metrics.get('cost_per_sample', 0.0)
            evaluation.cost_per_token = efficiency_metrics.get('cost_per_token', 0.0)
            
            # Detailed results
            evaluation.sample_level_results = [
                {
                    'sample_id': i,
                    'input': resp['input'],
                    'expected_output': resp.get('expected_output'),
                    'actual_output': resp['output'],
                    'metrics': resp.get('metrics', {}),
                    'response_time_ms': resp.get('response_time_ms', 0),
                    'tokens_generated': resp.get('tokens_generated', 0),
                    'cost': resp.get('cost', 0.0)
                }
                for i, resp in enumerate(responses)
            ]
            
            evaluation.error_analysis = error_analysis
            evaluation.performance_breakdown = metrics.get('breakdown', {})
            evaluation.confidence_intervals = statistical_results.get('confidence_intervals', {})
            evaluation.statistical_significance = statistical_results.get('statistical_significance', False)
            evaluation.p_values = statistical_results.get('p_values', {})
            
            self.db.commit()
            self.db.refresh(evaluation)
            
            logger.info(f"Completed LLM evaluation {evaluation.id} for model {model_id}")
            
            return evaluation
            
        except Exception as e:
            evaluation.status = 'failed'
            evaluation.completed_at = datetime.utcnow()
            self.db.commit()
            
            logger.error(f"LLM evaluation failed: {str(e)}")
            raise EvaluationError(f"LLM evaluation failed: {str(e)}")
    
    async def _load_evaluation_dataset(self, evaluation_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Load and prepare evaluation dataset"""
        
        dataset_name = evaluation_config.get('dataset_name')
        if not dataset_name:
            raise ValidationError("Dataset name is required for evaluation")
        
        # Query dataset from database
        dataset_record = self.db.query(EvaluationDataset).filter(
            EvaluationDataset.name == dataset_name
        ).first()
        
        if not dataset_record:
            raise ValidationError(f"Dataset {dataset_name} not found")
        
        # Load dataset samples
        # This is a simplified implementation - in practice, you'd load from the actual storage location
        dataset_samples = self._load_dataset_from_storage(dataset_record.storage_location)
        
        # Sample selection
        num_samples = evaluation_config.get('num_samples', len(dataset_samples))
        strategy = evaluation_config.get('sample_selection_strategy', 'random')
        
        if strategy == 'random':
            np.random.seed(evaluation_config.get('random_seed', 42))
            selected_indices = np.random.choice(len(dataset_samples), min(num_samples, len(dataset_samples)), replace=False)
            selected_samples = [dataset_samples[i] for i in selected_indices]
        elif strategy == 'stratified':
            # Implement stratified sampling based on categories/difficulty
            selected_samples = self._stratified_sample(dataset_samples, num_samples)
        else:
            selected_samples = dataset_samples[:num_samples]
        
        return selected_samples
    
    async def _generate_model_responses(self,
                                      llm_model: LLMModel,
                                      dataset: List[Dict[str, Any]],
                                      evaluation_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate model responses for evaluation dataset"""
        
        responses = []
        start_time = datetime.utcnow()
        total_tokens = 0
        total_cost = 0.0
        
        # Set up model client
        client = self._get_model_client(llm_model)
        
        # Generate responses with concurrent processing
        semaphore = asyncio.Semaphore(evaluation_config.get('max_concurrent_requests', 5))
        
        async def generate_single_response(sample: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                try:
                    response_start = datetime.utcnow()
                    
                    # Prepare prompt
                    prompt = self._prepare_prompt(sample, evaluation_config.get('prompt_template'))
                    
                    # Generate response
                    output, tokens_used, cost = await self._call_model_api(
                        client, llm_model, prompt, llm_model.model_parameters or {}
                    )
                    
                    response_time = (datetime.utcnow() - response_start).total_seconds() * 1000
                    
                    return {
                        'input': sample.get('input', sample.get('question', '')),
                        'expected_output': sample.get('output', sample.get('answer')),
                        'output': output,
                        'tokens_generated': tokens_used,
                        'response_time_ms': int(response_time),
                        'cost': cost,
                        'metadata': sample.get('metadata', {})
                    }
                    
                except Exception as e:
                    logger.error(f"Error generating response for sample: {str(e)}")
                    return {
                        'input': sample.get('input', ''),
                        'expected_output': sample.get('output'),
                        'output': '',
                        'error': str(e),
                        'tokens_generated': 0,
                        'response_time_ms': 0,
                        'cost': 0.0
                    }
        
        # Generate all responses concurrently
        response_tasks = [generate_single_response(sample) for sample in dataset]
        responses = await asyncio.gather(*response_tasks)
        
        return responses
    
    async def _calculate_performance_metrics(self,
                                           responses: List[Dict[str, Any]],
                                           dataset: List[Dict[str, Any]],
                                           evaluation_config: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        
        valid_responses = [r for r in responses if 'error' not in r and r.get('output')]
        
        if not valid_responses:
            return {'overall_score': 0.0}
        
        metrics = {}
        
        # BLEU score calculation
        bleu_scores = []
        for response in valid_responses:
            if response.get('expected_output'):
                reference = [response['expected_output'].split()]
                candidate = response['output'].split()
                try:
                    bleu = sentence_bleu(reference, candidate)
                    bleu_scores.append(bleu)
                except:
                    bleu_scores.append(0.0)
        
        metrics['bleu_score'] = np.mean(bleu_scores) if bleu_scores else 0.0
        
        # ROUGE score calculation
        rouge_1_scores, rouge_2_scores, rouge_l_scores = [], [], []
        for response in valid_responses:
            if response.get('expected_output'):
                try:
                    scores = self.rouge_scorer.score(response['expected_output'], response['output'])
                    rouge_1_scores.append(scores['rouge1'].fmeasure)
                    rouge_2_scores.append(scores['rouge2'].fmeasure)
                    rouge_l_scores.append(scores['rougeL'].fmeasure)
                except:
                    rouge_1_scores.append(0.0)
                    rouge_2_scores.append(0.0)
                    rouge_l_scores.append(0.0)
        
        metrics['rouge_1'] = np.mean(rouge_1_scores) if rouge_1_scores else 0.0
        metrics['rouge_2'] = np.mean(rouge_2_scores) if rouge_2_scores else 0.0
        metrics['rouge_l'] = np.mean(rouge_l_scores) if rouge_l_scores else 0.0
        
        # BERTScore calculation
        if valid_responses:
            try:
                candidates = [r['output'] for r in valid_responses if r.get('expected_output')]
                references = [r['expected_output'] for r in valid_responses if r.get('expected_output')]
                
                if candidates and references:
                    P, R, F1 = self.bert_scorer.score(candidates, references)
                    metrics['bertscore_precision'] = float(P.mean())
                    metrics['bertscore_recall'] = float(R.mean())
                    metrics['bertscore_f1'] = float(F1.mean())
                else:
                    metrics['bertscore_precision'] = 0.0
                    metrics['bertscore_recall'] = 0.0
                    metrics['bertscore_f1'] = 0.0
            except Exception as e:
                logger.warning(f"BERTScore calculation failed: {str(e)}")
                metrics['bertscore_precision'] = 0.0
                metrics['bertscore_recall'] = 0.0
                metrics['bertscore_f1'] = 0.0
        
        # Fluency assessment using readability metrics
        fluency_scores = []
        for response in valid_responses:
            if response.get('output'):
                try:
                    # Simple fluency proxy using readability
                    readability = flesch_reading_ease(response['output'])
                    # Normalize to 0-1 scale (higher is better)
                    fluency = max(0.0, min(1.0, (readability + 100) / 200))
                    fluency_scores.append(fluency)
                except:
                    fluency_scores.append(0.5)  # Neutral score
        
        metrics['fluency_score'] = np.mean(fluency_scores) if fluency_scores else 0.0
        
        # Coherence and relevance (simplified heuristics)
        coherence_scores = []
        relevance_scores = []
        
        for response in valid_responses:
            output = response.get('output', '')
            
            # Coherence: measure sentence-to-sentence similarity
            sentences = re.split(r'[.!?]+', output)
            if len(sentences) > 1:
                sentence_embeddings = self.sentence_transformer.encode(sentences)
                if len(sentence_embeddings) > 1:
                    similarities = []
                    for i in range(len(sentence_embeddings) - 1):
                        sim = np.dot(sentence_embeddings[i], sentence_embeddings[i+1])
                        similarities.append(max(0, sim))  # Ensure non-negative
                    coherence_scores.append(np.mean(similarities))
                else:
                    coherence_scores.append(0.5)
            else:
                coherence_scores.append(0.5)
            
            # Relevance: similarity between input and output
            if response.get('input'):
                try:
                    input_emb = self.sentence_transformer.encode([response['input']])
                    output_emb = self.sentence_transformer.encode([output])
                    relevance = max(0, np.dot(input_emb[0], output_emb[0]))
                    relevance_scores.append(relevance)
                except:
                    relevance_scores.append(0.5)
        
        metrics['coherence_score'] = np.mean(coherence_scores) if coherence_scores else 0.0
        metrics['relevance_score'] = np.mean(relevance_scores) if relevance_scores else 0.0
        
        # Calculate overall score as weighted average
        weights = {
            'bleu_score': 0.15,
            'rouge_l': 0.15,
            'bertscore_f1': 0.25,
            'fluency_score': 0.15,
            'coherence_score': 0.15,
            'relevance_score': 0.15
        }
        
        overall_score = sum(metrics.get(metric, 0.0) * weight for metric, weight in weights.items())
        metrics['overall_score'] = overall_score
        
        # Performance breakdown by categories (if metadata available)
        breakdown = self._calculate_performance_breakdown(valid_responses, metrics)
        metrics['breakdown'] = breakdown
        
        return metrics
    
    async def _calculate_semantic_metrics(self,
                                        responses: List[Dict[str, Any]],
                                        dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate semantic similarity metrics"""
        
        semantic_similarities = []
        embedding_similarities = []
        
        for response in responses:
            if response.get('expected_output') and response.get('output'):
                try:
                    # Semantic similarity using sentence transformers
                    emb1 = self.sentence_transformer.encode([response['expected_output']])
                    emb2 = self.sentence_transformer.encode([response['output']])
                    
                    similarity = np.dot(emb1[0], emb2[0]) / (np.linalg.norm(emb1[0]) * np.linalg.norm(emb2[0]))
                    semantic_similarities.append(max(0, similarity))  # Ensure non-negative
                    
                    # Cosine similarity for embeddings
                    cosine_sim = np.dot(emb1[0], emb2[0])
                    embedding_similarities.append(max(0, cosine_sim))
                    
                except Exception as e:
                    logger.warning(f"Semantic similarity calculation failed: {str(e)}")
                    semantic_similarities.append(0.0)
                    embedding_similarities.append(0.0)
        
        return {
            'semantic_similarity': np.mean(semantic_similarities) if semantic_similarities else 0.0,
            'embedding_similarity': np.mean(embedding_similarities) if embedding_similarities else 0.0
        }
    
    async def _calculate_safety_metrics(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate safety and bias metrics"""
        
        # This is a simplified implementation
        # In practice, you'd use specialized models like Perspective API or custom safety classifiers
        
        toxicity_scores = []
        bias_scores = []
        hallucination_rates = []
        
        # Simple heuristics for demonstration
        toxic_keywords = ['hate', 'violence', 'harmful', 'dangerous', 'illegal']
        bias_keywords = ['always', 'never', 'all', 'stereotype']
        
        for response in responses:
            output = response.get('output', '').lower()
            
            # Simple toxicity detection
            toxicity_score = sum(1 for keyword in toxic_keywords if keyword in output) / len(toxic_keywords)
            toxicity_scores.append(min(1.0, toxicity_score))
            
            # Simple bias detection
            bias_score = sum(1 for keyword in bias_keywords if keyword in output) / len(bias_keywords)
            bias_scores.append(min(1.0, bias_score))
            
            # Hallucination detection (simplified - check for made-up facts)
            # This would require more sophisticated fact-checking in practice
            hallucination_indicators = ['according to my knowledge', 'i believe', 'it seems like']
            hallucination_rate = sum(1 for indicator in hallucination_indicators if indicator in output)
            hallucination_rates.append(min(1.0, hallucination_rate / 3))
        
        return {
            'toxicity_score': np.mean(toxicity_scores) if toxicity_scores else 0.0,
            'bias_score': np.mean(bias_scores) if bias_scores else 0.0,
            'hallucination_rate': np.mean(hallucination_rates) if hallucination_rates else 0.0
        }
    
    def _calculate_efficiency_metrics(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate efficiency and cost metrics"""
        
        valid_responses = [r for r in responses if 'error' not in r]
        
        if not valid_responses:
            return {}
        
        response_times = [r.get('response_time_ms', 0) for r in valid_responses]
        tokens_generated = [r.get('tokens_generated', 0) for r in valid_responses]
        costs = [r.get('cost', 0.0) for r in valid_responses]
        
        total_tokens = sum(tokens_generated)
        total_cost = sum(costs)
        
        metrics = {
            'avg_response_time_ms': int(np.mean(response_times)) if response_times else 0,
            'total_tokens_generated': total_tokens,
            'avg_tokens_per_response': int(np.mean(tokens_generated)) if tokens_generated else 0,
            'total_cost_usd': total_cost,
            'cost_per_sample': total_cost / len(valid_responses) if valid_responses else 0.0,
            'cost_per_token': total_cost / total_tokens if total_tokens > 0 else 0.0
        }
        
        # Token efficiency (quality per token)
        if total_tokens > 0:
            # This is a simplified metric - would need quality scores in practice
            avg_quality = 0.7  # Placeholder
            metrics['token_efficiency'] = avg_quality / (total_tokens / len(valid_responses))
        else:
            metrics['token_efficiency'] = 0.0
        
        return metrics
    
    def _perform_error_analysis(self,
                               responses: List[Dict[str, Any]],
                               dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform error analysis on model responses"""
        
        error_patterns = {
            'empty_responses': 0,
            'api_errors': 0,
            'too_short': 0,
            'too_long': 0,
            'off_topic': 0,
            'repetitive': 0
        }
        
        total_responses = len(responses)
        
        for response in responses:
            output = response.get('output', '')
            
            if 'error' in response:
                error_patterns['api_errors'] += 1
            elif not output.strip():
                error_patterns['empty_responses'] += 1
            elif len(output.split()) < 5:
                error_patterns['too_short'] += 1
            elif len(output.split()) > 500:
                error_patterns['too_long'] += 1
            
            # Check for repetitive content
            sentences = output.split('.')
            if len(sentences) > 2:
                unique_sentences = set(s.strip().lower() for s in sentences if s.strip())
                if len(unique_sentences) < len(sentences) * 0.8:
                    error_patterns['repetitive'] += 1
        
        # Convert to percentages
        error_analysis = {
            pattern: (count / total_responses) * 100 if total_responses > 0 else 0
            for pattern, count in error_patterns.items()
        }
        
        # Add common failure cases
        error_analysis['common_failure_cases'] = self._identify_common_failures(responses)
        
        return error_analysis
    
    def _perform_statistical_analysis(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical analysis on evaluation results"""
        
        # Calculate confidence intervals for key metrics
        key_metrics = ['overall_score', 'bleu_score', 'rouge_l', 'bertscore_f1']
        confidence_intervals = {}
        
        for metric in key_metrics:
            if metric in metrics:
                # Simplified confidence interval calculation
                # In practice, you'd use bootstrap or other methods with actual sample data
                value = metrics[metric]
                margin_error = 0.02  # 2% margin of error (simplified)
                confidence_intervals[metric] = {
                    'lower': max(0.0, value - margin_error),
                    'upper': min(1.0, value + margin_error),
                    'confidence_level': 0.95
                }
        
        return {
            'confidence_intervals': confidence_intervals,
            'statistical_significance': metrics.get('overall_score', 0) > 0.5,
            'p_values': {'overall_significance': 0.01}  # Placeholder
        }
    
    # ========================================================================================
    # UTILITY METHODS
    # ========================================================================================
    
    def _get_model_client(self, llm_model: LLMModel):
        """Get appropriate API client for the model"""
        
        provider = llm_model.provider.lower()
        
        if provider == 'openai' and self.openai_client:
            return self.openai_client
        elif provider == 'anthropic' and self.anthropic_client:
            return self.anthropic_client
        else:
            raise EvaluationError(f"No client configured for provider: {provider}")
    
    async def _call_model_api(self,
                            client: Any,
                            llm_model: LLMModel,
                            prompt: str,
                            parameters: Dict[str, Any]) -> Tuple[str, int, float]:
        """Call model API and return response, tokens, and cost"""
        
        try:
            if llm_model.provider.lower() == 'openai':
                response = await self._call_openai_api(client, llm_model, prompt, parameters)
            elif llm_model.provider.lower() == 'anthropic':
                response = await self._call_anthropic_api(client, llm_model, prompt, parameters)
            else:
                raise EvaluationError(f"Unsupported provider: {llm_model.provider}")
            
            return response
            
        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            return "", 0, 0.0
    
    async def _call_openai_api(self,
                             client,
                             llm_model: LLMModel,
                             prompt: str,
                             parameters: Dict[str, Any]) -> Tuple[str, int, float]:
        """Call OpenAI API"""
        
        try:
            response = await client.chat.completions.create(
                model=llm_model.model_version,
                messages=[{"role": "user", "content": prompt}],
                temperature=parameters.get('temperature', 0.7),
                max_tokens=parameters.get('max_tokens', 1000),
                top_p=parameters.get('top_p', 1.0)
            )
            
            output = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            
            # Calculate cost (simplified - would use actual pricing)
            cost = tokens_used * llm_model.cost_per_1k_tokens / 1000 if llm_model.cost_per_1k_tokens else 0.0
            
            return output, tokens_used, float(cost)
            
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise
    
    async def _call_anthropic_api(self,
                                client,
                                llm_model: LLMModel,
                                prompt: str,
                                parameters: Dict[str, Any]) -> Tuple[str, int, float]:
        """Call Anthropic API"""
        
        try:
            response = await client.messages.create(
                model=llm_model.model_version,
                messages=[{"role": "user", "content": prompt}],
                temperature=parameters.get('temperature', 0.7),
                max_tokens=parameters.get('max_tokens', 1000)
            )
            
            output = response.content[0].text
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            
            # Calculate cost
            cost = tokens_used * llm_model.cost_per_1k_tokens / 1000 if llm_model.cost_per_1k_tokens else 0.0
            
            return output, tokens_used, float(cost)
            
        except Exception as e:
            logger.error(f"Anthropic API error: {str(e)}")
            raise
    
    def _prepare_prompt(self, sample: Dict[str, Any], template: str = None) -> str:
        """Prepare prompt for model evaluation"""
        
        if template:
            # Use provided template
            return template.format(**sample)
        else:
            # Default prompt format
            input_text = sample.get('input', sample.get('question', ''))
            return f"Please respond to the following: {input_text}"
    
    def _load_dataset_from_storage(self, storage_location: str) -> List[Dict[str, Any]]:
        """Load dataset from storage location"""
        
        # This is a simplified implementation
        # In practice, you'd load from various sources (S3, local files, databases, etc.)
        
        try:
            if storage_location.endswith('.json'):
                with open(storage_location, 'r') as f:
                    return json.load(f)
            elif storage_location.endswith('.jsonl'):
                samples = []
                with open(storage_location, 'r') as f:
                    for line in f:
                        samples.append(json.loads(line))
                return samples
            else:
                # Return sample data for demonstration
                return [
                    {"input": "What is machine learning?", "output": "Machine learning is a subset of artificial intelligence..."},
                    {"input": "Explain neural networks", "output": "Neural networks are computing systems inspired by biological neural networks..."},
                    # Add more samples...
                ]
        except Exception as e:
            logger.warning(f"Failed to load dataset from {storage_location}: {str(e)}")
            return []
    
    def _stratified_sample(self, dataset: List[Dict[str, Any]], num_samples: int) -> List[Dict[str, Any]]:
        """Perform stratified sampling on dataset"""
        
        # Simple stratified sampling by category if available
        categories = {}
        for sample in dataset:
            category = sample.get('category', 'default')
            if category not in categories:
                categories[category] = []
            categories[category].append(sample)
        
        # Sample proportionally from each category
        samples_per_category = num_samples // len(categories)
        selected_samples = []
        
        for category, category_samples in categories.items():
            n_samples = min(samples_per_category, len(category_samples))
            selected = np.random.choice(len(category_samples), n_samples, replace=False)
            selected_samples.extend([category_samples[i] for i in selected])
        
        return selected_samples
    
    def _calculate_performance_breakdown(self,
                                       responses: List[Dict[str, Any]],
                                       overall_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate performance breakdown by categories"""
        
        breakdown = {
            'by_length': {},
            'by_complexity': {},
            'by_category': {}
        }
        
        # Group responses by characteristics
        short_responses = [r for r in responses if len(r.get('output', '').split()) < 50]
        medium_responses = [r for r in responses if 50 <= len(r.get('output', '').split()) < 150]
        long_responses = [r for r in responses if len(r.get('output', '').split()) >= 150]
        
        breakdown['by_length'] = {
            'short': len(short_responses),
            'medium': len(medium_responses),
            'long': len(long_responses)
        }
        
        return breakdown
    
    def _identify_common_failures(self, responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify common failure patterns"""
        
        failures = []
        
        # Identify responses with errors
        error_responses = [r for r in responses if 'error' in r]
        if error_responses:
            failures.append({
                'type': 'api_errors',
                'count': len(error_responses),
                'examples': [r.get('error', '') for r in error_responses[:3]]
            })
        
        # Identify very short responses
        short_responses = [r for r in responses if len(r.get('output', '').split()) < 3]
        if short_responses:
            failures.append({
                'type': 'insufficient_response_length',
                'count': len(short_responses),
                'examples': [r.get('output', '') for r in short_responses[:3]]
            })
        
        return failures


class EvaluationError(Exception):
    """Custom exception for evaluation errors"""
    pass