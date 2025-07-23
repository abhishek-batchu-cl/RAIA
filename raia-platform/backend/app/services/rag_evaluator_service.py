"""
RAG Evaluator Service
Comprehensive RAG (Retrieval-Augmented Generation) evaluation with multiple metrics
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

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import get_database

logger = logging.getLogger(__name__)
settings = get_settings()

@dataclass
class RAGEvaluationResult:
    """Data class for RAG evaluation results"""
    query_id: str
    query: str
    retrieved_docs: List[str]
    generated_response: str
    ground_truth: Optional[str]
    
    # Retrieval Metrics
    retrieval_precision: float
    retrieval_recall: float
    retrieval_f1: float
    retrieval_mrr: float  # Mean Reciprocal Rank
    retrieval_ndcg: float  # Normalized Discounted Cumulative Gain
    
    # Generation Metrics
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    context_relevancy: float
    
    # Language Quality Metrics
    bleu_score: float
    rouge_1: float
    rouge_2: float
    rouge_l: float
    bert_score_f1: float
    
    # Semantic Metrics
    semantic_similarity: float
    coherence: float
    fluency: float
    groundedness: float
    
    # Overall Metrics
    overall_score: float
    confidence_score: float
    
    # Additional Metadata
    evaluation_time: datetime
    model_version: str
    retrieval_time_ms: float
    generation_time_ms: float
    
class RAGEvaluatorService:
    """
    Comprehensive RAG evaluation service with multiple evaluation metrics
    """
    
    def __init__(self):
        self.evaluation_cache = {}
        self.evaluation_history = defaultdict(list)
        self.models = {}
        self.sentence_transformer = None
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        
        # Evaluation configuration
        self.eval_config = {
            'retrieval_top_k': 5,
            'similarity_threshold': 0.7,
            'min_context_length': 50,
            'max_context_length': 2000,
            'batch_size': 32,
            'timeout_seconds': 30
        }
        
    async def initialize_models(self):
        """
        Initialize evaluation models
        """
        try:
            # Load sentence transformer for semantic similarity
            if self.sentence_transformer is None:
                self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            
            logger.info("RAG evaluation models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG evaluation models: {e}")
            raise
    
    async def evaluate_rag_system(
        self,
        evaluation_id: str,
        queries: List[str],
        rag_system: Any,  # RAG system to evaluate
        ground_truth: Optional[List[str]] = None,
        relevant_docs: Optional[List[List[str]]] = None,
        evaluation_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a RAG system comprehensively
        
        Args:
            evaluation_id: Unique evaluation identifier
            queries: List of test queries
            rag_system: RAG system to evaluate
            ground_truth: Optional ground truth answers
            relevant_docs: Optional relevant documents for each query
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
                'total_queries': len(queries),
                'query_results': [],
                'aggregate_metrics': {},
                'performance_stats': {},
                'status': 'running'
            }
            
            # Process queries in batches
            batch_size = config['batch_size']
            query_batches = [queries[i:i + batch_size] for i in range(0, len(queries), batch_size)]
            
            all_query_results = []
            
            for batch_idx, query_batch in enumerate(query_batches):
                logger.info(f"Processing batch {batch_idx + 1}/{len(query_batches)}")
                
                batch_results = await self._evaluate_query_batch(
                    query_batch,
                    rag_system,
                    ground_truth,
                    relevant_docs,
                    config,
                    batch_idx * batch_size
                )
                
                all_query_results.extend(batch_results)
                
                # Update progress
                evaluation_results['processed_queries'] = len(all_query_results)
                evaluation_results['progress'] = len(all_query_results) / len(queries)
            
            evaluation_results['query_results'] = all_query_results
            
            # Calculate aggregate metrics
            aggregate_metrics = await self._calculate_aggregate_metrics(all_query_results)
            evaluation_results['aggregate_metrics'] = aggregate_metrics
            
            # Calculate performance statistics
            performance_stats = await self._calculate_performance_stats(all_query_results)
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
            logger.error(f"Failed to evaluate RAG system: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'evaluation_id': evaluation_id
            }
    
    async def _evaluate_query_batch(
        self,
        queries: List[str],
        rag_system: Any,
        ground_truth: Optional[List[str]],
        relevant_docs: Optional[List[List[str]]],
        config: Dict[str, Any],
        start_idx: int
    ) -> List[RAGEvaluationResult]:
        """
        Evaluate a batch of queries
        """
        batch_results = []
        
        for i, query in enumerate(queries):
            query_idx = start_idx + i
            
            try:
                # Generate response using RAG system
                start_time = datetime.utcnow()
                
                rag_response = await self._get_rag_response(rag_system, query, config)
                
                end_time = datetime.utcnow()
                total_time = (end_time - start_time).total_seconds() * 1000
                
                # Extract components
                retrieved_docs = rag_response.get('retrieved_docs', [])
                generated_response = rag_response.get('response', '')
                retrieval_time = rag_response.get('retrieval_time_ms', 0)
                generation_time = rag_response.get('generation_time_ms', 0)
                
                # Get ground truth and relevant docs for this query
                query_ground_truth = ground_truth[query_idx] if ground_truth and query_idx < len(ground_truth) else None
                query_relevant_docs = relevant_docs[query_idx] if relevant_docs and query_idx < len(relevant_docs) else None
                
                # Evaluate this query
                result = await self._evaluate_single_query(
                    query_id=f"query_{query_idx}",
                    query=query,
                    retrieved_docs=retrieved_docs,
                    generated_response=generated_response,
                    ground_truth=query_ground_truth,
                    relevant_docs=query_relevant_docs,
                    retrieval_time_ms=retrieval_time,
                    generation_time_ms=generation_time
                )
                
                batch_results.append(result)
                
            except Exception as e:
                logger.error(f"Error evaluating query '{query}': {e}")
                # Create error result
                error_result = RAGEvaluationResult(
                    query_id=f"query_{query_idx}",
                    query=query,
                    retrieved_docs=[],
                    generated_response="",
                    ground_truth=query_ground_truth,
                    retrieval_precision=0.0,
                    retrieval_recall=0.0,
                    retrieval_f1=0.0,
                    retrieval_mrr=0.0,
                    retrieval_ndcg=0.0,
                    faithfulness=0.0,
                    answer_relevancy=0.0,
                    context_precision=0.0,
                    context_recall=0.0,
                    context_relevancy=0.0,
                    bleu_score=0.0,
                    rouge_1=0.0,
                    rouge_2=0.0,
                    rouge_l=0.0,
                    bert_score_f1=0.0,
                    semantic_similarity=0.0,
                    coherence=0.0,
                    fluency=0.0,
                    groundedness=0.0,
                    overall_score=0.0,
                    confidence_score=0.0,
                    evaluation_time=datetime.utcnow(),
                    model_version="unknown",
                    retrieval_time_ms=0.0,
                    generation_time_ms=0.0
                )
                batch_results.append(error_result)
        
        return batch_results
    
    async def _get_rag_response(self, rag_system: Any, query: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get response from RAG system
        """
        try:
            # This would interface with the actual RAG system
            # For now, simulate the interface
            
            retrieval_start = datetime.utcnow()
            
            # Simulate retrieval
            if hasattr(rag_system, 'retrieve'):
                retrieved_docs = await rag_system.retrieve(query, k=config['retrieval_top_k'])
            else:
                retrieved_docs = []
            
            retrieval_time = (datetime.utcnow() - retrieval_start).total_seconds() * 1000
            
            generation_start = datetime.utcnow()
            
            # Simulate generation
            if hasattr(rag_system, 'generate'):
                response = await rag_system.generate(query, retrieved_docs)
            else:
                response = "Sample response"
            
            generation_time = (datetime.utcnow() - generation_start).total_seconds() * 1000
            
            return {
                'retrieved_docs': retrieved_docs,
                'response': response,
                'retrieval_time_ms': retrieval_time,
                'generation_time_ms': generation_time
            }
            
        except Exception as e:
            logger.error(f"Error getting RAG response: {e}")
            return {
                'retrieved_docs': [],
                'response': '',
                'retrieval_time_ms': 0.0,
                'generation_time_ms': 0.0
            }
    
    async def _evaluate_single_query(
        self,
        query_id: str,
        query: str,
        retrieved_docs: List[str],
        generated_response: str,
        ground_truth: Optional[str],
        relevant_docs: Optional[List[str]],
        retrieval_time_ms: float,
        generation_time_ms: float
    ) -> RAGEvaluationResult:
        """
        Evaluate a single query comprehensively
        """
        try:
            # Initialize all metrics
            metrics = {
                'retrieval_precision': 0.0,
                'retrieval_recall': 0.0,
                'retrieval_f1': 0.0,
                'retrieval_mrr': 0.0,
                'retrieval_ndcg': 0.0,
                'faithfulness': 0.0,
                'answer_relevancy': 0.0,
                'context_precision': 0.0,
                'context_recall': 0.0,
                'context_relevancy': 0.0,
                'bleu_score': 0.0,
                'rouge_1': 0.0,
                'rouge_2': 0.0,
                'rouge_l': 0.0,
                'bert_score_f1': 0.0,
                'semantic_similarity': 0.0,
                'coherence': 0.0,
                'fluency': 0.0,
                'groundedness': 0.0
            }
            
            # 1. Retrieval Evaluation
            if relevant_docs and retrieved_docs:
                retrieval_metrics = await self._evaluate_retrieval(
                    retrieved_docs, relevant_docs
                )
                metrics.update(retrieval_metrics)
            
            # 2. Generation Evaluation
            if generated_response:
                # Context-based evaluation
                if retrieved_docs:
                    context_metrics = await self._evaluate_context(
                        query, generated_response, retrieved_docs
                    )
                    metrics.update(context_metrics)
                
                # Ground truth comparison
                if ground_truth:
                    truth_metrics = await self._evaluate_against_ground_truth(
                        generated_response, ground_truth
                    )
                    metrics.update(truth_metrics)
                
                # Language quality evaluation
                quality_metrics = await self._evaluate_language_quality(
                    generated_response, ground_truth or query
                )
                metrics.update(quality_metrics)
                
                # Semantic evaluation
                semantic_metrics = await self._evaluate_semantic_quality(
                    query, generated_response, retrieved_docs
                )
                metrics.update(semantic_metrics)
            
            # Calculate overall score
            overall_score = await self._calculate_overall_score(metrics)
            confidence_score = await self._calculate_confidence_score(metrics, generated_response, retrieved_docs)
            
            return RAGEvaluationResult(
                query_id=query_id,
                query=query,
                retrieved_docs=retrieved_docs,
                generated_response=generated_response,
                ground_truth=ground_truth,
                **metrics,
                overall_score=overall_score,
                confidence_score=confidence_score,
                evaluation_time=datetime.utcnow(),
                model_version="1.0",
                retrieval_time_ms=retrieval_time_ms,
                generation_time_ms=generation_time_ms
            )
            
        except Exception as e:
            logger.error(f"Error evaluating single query: {e}")
            # Return default result with error
            return RAGEvaluationResult(
                query_id=query_id,
                query=query,
                retrieved_docs=retrieved_docs,
                generated_response=generated_response,
                ground_truth=ground_truth,
                retrieval_precision=0.0,
                retrieval_recall=0.0,
                retrieval_f1=0.0,
                retrieval_mrr=0.0,
                retrieval_ndcg=0.0,
                faithfulness=0.0,
                answer_relevancy=0.0,
                context_precision=0.0,
                context_recall=0.0,
                context_relevancy=0.0,
                bleu_score=0.0,
                rouge_1=0.0,
                rouge_2=0.0,
                rouge_l=0.0,
                bert_score_f1=0.0,
                semantic_similarity=0.0,
                coherence=0.0,
                fluency=0.0,
                groundedness=0.0,
                overall_score=0.0,
                confidence_score=0.0,
                evaluation_time=datetime.utcnow(),
                model_version="1.0",
                retrieval_time_ms=retrieval_time_ms,
                generation_time_ms=generation_time_ms
            )
    
    async def _evaluate_retrieval(self, retrieved_docs: List[str], relevant_docs: List[str]) -> Dict[str, float]:
        """
        Evaluate retrieval performance
        """
        try:
            if not retrieved_docs or not relevant_docs:
                return {
                    'retrieval_precision': 0.0,
                    'retrieval_recall': 0.0,
                    'retrieval_f1': 0.0,
                    'retrieval_mrr': 0.0,
                    'retrieval_ndcg': 0.0
                }
            
            # Convert to sets for easier comparison
            retrieved_set = set(retrieved_docs)
            relevant_set = set(relevant_docs)
            
            # Calculate precision, recall, F1
            if len(retrieved_set) > 0:
                precision = len(retrieved_set.intersection(relevant_set)) / len(retrieved_set)
            else:
                precision = 0.0
            
            if len(relevant_set) > 0:
                recall = len(retrieved_set.intersection(relevant_set)) / len(relevant_set)
            else:
                recall = 0.0
            
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0
            
            # Calculate MRR (Mean Reciprocal Rank)
            mrr = 0.0
            for i, doc in enumerate(retrieved_docs):
                if doc in relevant_set:
                    mrr = 1.0 / (i + 1)
                    break
            
            # Calculate NDCG (simplified version)
            dcg = 0.0
            idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(relevant_docs), len(retrieved_docs)))])
            
            for i, doc in enumerate(retrieved_docs):
                if doc in relevant_set:
                    dcg += 1.0 / np.log2(i + 2)
            
            ndcg = dcg / idcg if idcg > 0 else 0.0
            
            return {
                'retrieval_precision': precision,
                'retrieval_recall': recall,
                'retrieval_f1': f1,
                'retrieval_mrr': mrr,
                'retrieval_ndcg': ndcg
            }
            
        except Exception as e:
            logger.error(f"Error evaluating retrieval: {e}")
            return {
                'retrieval_precision': 0.0,
                'retrieval_recall': 0.0,
                'retrieval_f1': 0.0,
                'retrieval_mrr': 0.0,
                'retrieval_ndcg': 0.0
            }
    
    async def _evaluate_context(
        self,
        query: str,
        response: str,
        context_docs: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate context usage and relevancy
        """
        try:
            if not context_docs or not response:
                return {
                    'faithfulness': 0.0,
                    'answer_relevancy': 0.0,
                    'context_precision': 0.0,
                    'context_recall': 0.0,
                    'context_relevancy': 0.0
                }
            
            context_text = " ".join(context_docs)
            
            # 1. Faithfulness: How well grounded is the response in the context
            faithfulness = await self._calculate_faithfulness(response, context_text)
            
            # 2. Answer Relevancy: How relevant is the response to the query
            answer_relevancy = await self._calculate_relevancy(query, response)
            
            # 3. Context Precision: How precise is the retrieved context
            context_precision = await self._calculate_context_precision(query, context_docs)
            
            # 4. Context Recall: How complete is the retrieved context
            context_recall = await self._calculate_context_recall(query, response, context_docs)
            
            # 5. Context Relevancy: How relevant is the context to the query
            context_relevancy = await self._calculate_relevancy(query, context_text)
            
            return {
                'faithfulness': faithfulness,
                'answer_relevancy': answer_relevancy,
                'context_precision': context_precision,
                'context_recall': context_recall,
                'context_relevancy': context_relevancy
            }
            
        except Exception as e:
            logger.error(f"Error evaluating context: {e}")
            return {
                'faithfulness': 0.0,
                'answer_relevancy': 0.0,
                'context_precision': 0.0,
                'context_recall': 0.0,
                'context_relevancy': 0.0
            }
    
    async def _evaluate_against_ground_truth(
        self,
        response: str,
        ground_truth: str
    ) -> Dict[str, float]:
        """
        Evaluate response against ground truth
        """
        try:
            if not response or not ground_truth:
                return {
                    'bleu_score': 0.0,
                    'rouge_1': 0.0,
                    'rouge_2': 0.0,
                    'rouge_l': 0.0,
                    'bert_score_f1': 0.0,
                    'semantic_similarity': 0.0
                }
            
            # 1. BLEU Score
            try:
                response_tokens = response.split()
                ground_truth_tokens = [ground_truth.split()]
                bleu = sentence_bleu(ground_truth_tokens, response_tokens)
            except:
                bleu = 0.0
            
            # 2. ROUGE Scores
            try:
                rouge_scores = self.rouge_scorer.score(ground_truth, response)
                rouge_1 = rouge_scores['rouge1'].fmeasure
                rouge_2 = rouge_scores['rouge2'].fmeasure
                rouge_l = rouge_scores['rougeL'].fmeasure
            except:
                rouge_1 = rouge_2 = rouge_l = 0.0
            
            # 3. BERT Score
            try:
                P, R, F1 = bert_score([response], [ground_truth], lang='en', verbose=False)
                bert_f1 = F1.mean().item()
            except:
                bert_f1 = 0.0
            
            # 4. Semantic Similarity
            semantic_sim = await self._calculate_semantic_similarity(response, ground_truth)
            
            return {
                'bleu_score': bleu,
                'rouge_1': rouge_1,
                'rouge_2': rouge_2,
                'rouge_l': rouge_l,
                'bert_score_f1': bert_f1,
                'semantic_similarity': semantic_sim
            }
            
        except Exception as e:
            logger.error(f"Error evaluating against ground truth: {e}")
            return {
                'bleu_score': 0.0,
                'rouge_1': 0.0,
                'rouge_2': 0.0,
                'rouge_l': 0.0,
                'bert_score_f1': 0.0,
                'semantic_similarity': 0.0
            }
    
    async def _evaluate_language_quality(self, response: str, reference: str) -> Dict[str, float]:
        """
        Evaluate language quality of the response
        """
        try:
            if not response:
                return {
                    'coherence': 0.0,
                    'fluency': 0.0
                }
            
            # 1. Coherence - sentence connectivity and logical flow
            coherence = await self._calculate_coherence(response)
            
            # 2. Fluency - language naturalness and grammatical correctness
            fluency = await self._calculate_fluency(response)
            
            return {
                'coherence': coherence,
                'fluency': fluency
            }
            
        except Exception as e:
            logger.error(f"Error evaluating language quality: {e}")
            return {
                'coherence': 0.0,
                'fluency': 0.0
            }
    
    async def _evaluate_semantic_quality(
        self,
        query: str,
        response: str,
        context_docs: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate semantic quality of the response
        """
        try:
            if not response:
                return {
                    'groundedness': 0.0
                }
            
            # Groundedness - how well the response is grounded in the provided context
            if context_docs:
                context_text = " ".join(context_docs)
                groundedness = await self._calculate_faithfulness(response, context_text)
            else:
                groundedness = 0.0
            
            return {
                'groundedness': groundedness
            }
            
        except Exception as e:
            logger.error(f"Error evaluating semantic quality: {e}")
            return {
                'groundedness': 0.0
            }
    
    async def _calculate_faithfulness(self, response: str, context: str) -> float:
        """
        Calculate faithfulness score using semantic similarity
        """
        try:
            if not response or not context:
                return 0.0
            
            # Split response into sentences
            response_sentences = re.split(r'[.!?]+', response)
            response_sentences = [s.strip() for s in response_sentences if s.strip()]
            
            if not response_sentences:
                return 0.0
            
            # Calculate similarity of each sentence with context
            similarities = []
            for sentence in response_sentences:
                sim = await self._calculate_semantic_similarity(sentence, context)
                similarities.append(sim)
            
            # Return average similarity
            return float(np.mean(similarities)) if similarities else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating faithfulness: {e}")
            return 0.0
    
    async def _calculate_relevancy(self, query: str, text: str) -> float:
        """
        Calculate relevancy score using semantic similarity
        """
        return await self._calculate_semantic_similarity(query, text)
    
    async def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts
        """
        try:
            if not text1 or not text2:
                return 0.0
            
            if self.sentence_transformer is None:
                await self.initialize_models()
            
            # Get embeddings
            embeddings = self.sentence_transformer.encode([text1, text2])
            
            # Calculate cosine similarity
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            
            return float(max(0.0, similarity))
            
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            # Fallback to TF-IDF similarity
            try:
                tfidf_matrix = self.tfidf_vectorizer.fit_transform([text1, text2])
                similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                return float(max(0.0, similarity))
            except:
                return 0.0
    
    async def _calculate_context_precision(self, query: str, context_docs: List[str]) -> float:
        """
        Calculate context precision - how many retrieved docs are relevant
        """
        try:
            if not context_docs:
                return 0.0
            
            relevant_count = 0
            threshold = 0.5  # Similarity threshold for relevance
            
            for doc in context_docs:
                similarity = await self._calculate_semantic_similarity(query, doc)
                if similarity > threshold:
                    relevant_count += 1
            
            return relevant_count / len(context_docs)
            
        except Exception as e:
            logger.error(f"Error calculating context precision: {e}")
            return 0.0
    
    async def _calculate_context_recall(self, query: str, response: str, context_docs: List[str]) -> float:
        """
        Calculate context recall - how much relevant context was used
        """
        try:
            if not context_docs or not response:
                return 0.0
            
            used_context_count = 0
            threshold = 0.3  # Lower threshold for context usage
            
            for doc in context_docs:
                similarity = await self._calculate_semantic_similarity(response, doc)
                if similarity > threshold:
                    used_context_count += 1
            
            return used_context_count / len(context_docs) if context_docs else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating context recall: {e}")
            return 0.0
    
    async def _calculate_coherence(self, text: str) -> float:
        """
        Calculate coherence score based on sentence connectivity
        """
        try:
            if not text:
                return 0.0
            
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) < 2:
                return 1.0 if len(sentences) == 1 else 0.0
            
            # Calculate average similarity between adjacent sentences
            similarities = []
            for i in range(len(sentences) - 1):
                sim = await self._calculate_semantic_similarity(sentences[i], sentences[i + 1])
                similarities.append(sim)
            
            return float(np.mean(similarities)) if similarities else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating coherence: {e}")
            return 0.0
    
    async def _calculate_fluency(self, text: str) -> float:
        """
        Calculate fluency score based on language patterns
        """
        try:
            if not text:
                return 0.0
            
            # Simple heuristics for fluency
            score = 1.0
            
            # Check for basic grammar patterns
            words = text.split()
            if len(words) == 0:
                return 0.0
            
            # Penalize very short or very long sentences
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            avg_sentence_length = np.mean([len(s.split()) for s in sentences]) if sentences else 0
            
            if avg_sentence_length < 5 or avg_sentence_length > 30:
                score *= 0.8
            
            # Check for repeated words (potential generation issues)
            word_counts = {}
            for word in words:
                word_lower = word.lower()
                word_counts[word_lower] = word_counts.get(word_lower, 0) + 1
            
            max_repetition = max(word_counts.values()) if word_counts else 1
            repetition_ratio = max_repetition / len(words)
            
            if repetition_ratio > 0.3:
                score *= 0.7
            
            return float(max(0.0, score))
            
        except Exception as e:
            logger.error(f"Error calculating fluency: {e}")
            return 0.0
    
    async def _calculate_overall_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate overall score from individual metrics
        """
        try:
            # Define weights for different metric categories
            weights = {
                'retrieval': 0.25,
                'generation': 0.35,
                'language_quality': 0.2,
                'semantic': 0.2
            }
            
            # Group metrics by category
            retrieval_metrics = [
                'retrieval_precision', 'retrieval_recall', 'retrieval_f1',
                'retrieval_mrr', 'retrieval_ndcg'
            ]
            
            generation_metrics = [
                'faithfulness', 'answer_relevancy', 'context_precision',
                'context_recall', 'context_relevancy'
            ]
            
            language_metrics = [
                'bleu_score', 'rouge_1', 'rouge_2', 'rouge_l',
                'bert_score_f1', 'coherence', 'fluency'
            ]
            
            semantic_metrics = [
                'semantic_similarity', 'groundedness'
            ]
            
            # Calculate category scores
            retrieval_score = np.mean([metrics.get(m, 0.0) for m in retrieval_metrics])
            generation_score = np.mean([metrics.get(m, 0.0) for m in generation_metrics])
            language_score = np.mean([metrics.get(m, 0.0) for m in language_metrics])
            semantic_score = np.mean([metrics.get(m, 0.0) for m in semantic_metrics])
            
            # Calculate weighted overall score
            overall_score = (
                retrieval_score * weights['retrieval'] +
                generation_score * weights['generation'] +
                language_score * weights['language_quality'] +
                semantic_score * weights['semantic']
            )
            
            return float(max(0.0, min(1.0, overall_score)))
            
        except Exception as e:
            logger.error(f"Error calculating overall score: {e}")
            return 0.0
    
    async def _calculate_confidence_score(self, metrics: Dict[str, float], response: str, context_docs: List[str]) -> float:
        """
        Calculate confidence score based on various factors
        """
        try:
            confidence_factors = []
            
            # Factor 1: Response length (too short or too long reduces confidence)
            if response:
                response_length = len(response.split())
                if 10 <= response_length <= 200:
                    length_confidence = 1.0
                elif response_length < 5:
                    length_confidence = 0.3
                else:
                    length_confidence = max(0.5, 1.0 - (response_length - 200) / 500)
                confidence_factors.append(length_confidence)
            else:
                confidence_factors.append(0.0)
            
            # Factor 2: Context availability
            context_confidence = min(1.0, len(context_docs) / 3) if context_docs else 0.0
            confidence_factors.append(context_confidence)
            
            # Factor 3: Metric consistency (high variance reduces confidence)
            metric_values = [v for v in metrics.values() if v > 0]
            if metric_values:
                metric_variance = np.var(metric_values)
                variance_confidence = max(0.0, 1.0 - metric_variance)
                confidence_factors.append(variance_confidence)
            else:
                confidence_factors.append(0.0)
            
            # Factor 4: Groundedness (if available)
            groundedness = metrics.get('groundedness', 0.0)
            confidence_factors.append(groundedness)
            
            return float(np.mean(confidence_factors)) if confidence_factors else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating confidence score: {e}")
            return 0.0
    
    async def _calculate_aggregate_metrics(self, query_results: List[RAGEvaluationResult]) -> Dict[str, Any]:
        """
        Calculate aggregate metrics from query results
        """
        try:
            if not query_results:
                return {}
            
            # Collect all metric values
            metrics = {}
            metric_names = [
                'retrieval_precision', 'retrieval_recall', 'retrieval_f1',
                'retrieval_mrr', 'retrieval_ndcg', 'faithfulness',
                'answer_relevancy', 'context_precision', 'context_recall',
                'context_relevancy', 'bleu_score', 'rouge_1', 'rouge_2',
                'rouge_l', 'bert_score_f1', 'semantic_similarity',
                'coherence', 'fluency', 'groundedness', 'overall_score',
                'confidence_score'
            ]
            
            for metric_name in metric_names:
                values = [getattr(result, metric_name) for result in query_results]
                
                metrics[metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values)),
                    'p25': float(np.percentile(values, 25)),
                    'p75': float(np.percentile(values, 75))
                }
            
            # Calculate pass rates (scores above threshold)
            pass_rates = {}
            thresholds = {'overall_score': 0.7, 'confidence_score': 0.6}
            
            for metric_name, threshold in thresholds.items():
                values = [getattr(result, metric_name) for result in query_results]
                pass_count = sum(1 for v in values if v >= threshold)
                pass_rates[f"{metric_name}_pass_rate"] = pass_count / len(values)
            
            return {
                'metrics': metrics,
                'pass_rates': pass_rates,
                'total_queries': len(query_results)
            }
            
        except Exception as e:
            logger.error(f"Error calculating aggregate metrics: {e}")
            return {}
    
    async def _calculate_performance_stats(self, query_results: List[RAGEvaluationResult]) -> Dict[str, Any]:
        """
        Calculate performance statistics
        """
        try:
            if not query_results:
                return {}
            
            retrieval_times = [result.retrieval_time_ms for result in query_results]
            generation_times = [result.generation_time_ms for result in query_results]
            
            return {
                'retrieval_time_ms': {
                    'mean': float(np.mean(retrieval_times)),
                    'p50': float(np.percentile(retrieval_times, 50)),
                    'p95': float(np.percentile(retrieval_times, 95)),
                    'max': float(np.max(retrieval_times))
                },
                'generation_time_ms': {
                    'mean': float(np.mean(generation_times)),
                    'p50': float(np.percentile(generation_times, 50)),
                    'p95': float(np.percentile(generation_times, 95)),
                    'max': float(np.max(generation_times))
                },
                'queries_per_second': len(query_results) / max(1, sum(retrieval_times + generation_times) / 1000),
                'success_rate': sum(1 for r in query_results if r.overall_score > 0) / len(query_results)
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance stats: {e}")
            return {}
    
    async def get_evaluation_results(self, evaluation_id: str) -> Dict[str, Any]:
        """
        Get evaluation results by ID
        
        Args:
            evaluation_id: Evaluation identifier
        
        Returns:
            Evaluation results
        """
        try:
            if evaluation_id not in self.evaluation_cache:
                return {
                    'status': 'not_found',
                    'message': f'Evaluation {evaluation_id} not found',
                    'evaluation_id': evaluation_id
                }
            
            return {
                'status': 'success',
                **self.evaluation_cache[evaluation_id]
            }
            
        except Exception as e:
            logger.error(f"Error getting evaluation results: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'evaluation_id': evaluation_id
            }
    
    async def compare_rag_systems(
        self,
        comparison_id: str,
        systems: List[Any],
        system_names: List[str],
        test_queries: List[str],
        ground_truth: Optional[List[str]] = None,
        relevant_docs: Optional[List[List[str]]] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple RAG systems
        
        Args:
            comparison_id: Unique comparison identifier
            systems: List of RAG systems to compare
            system_names: Names for each system
            test_queries: Test queries
            ground_truth: Optional ground truth answers
            relevant_docs: Optional relevant documents
        
        Returns:
            Comparison results
        """
        try:
            if len(systems) != len(system_names):
                raise ValueError("Number of systems must match number of system names")
            
            comparison_results = {
                'comparison_id': comparison_id,
                'start_time': datetime.utcnow(),
                'system_names': system_names,
                'total_queries': len(test_queries),
                'system_results': {},
                'comparison_metrics': {},
                'status': 'running'
            }
            
            # Evaluate each system
            for i, (system, system_name) in enumerate(zip(systems, system_names)):
                logger.info(f"Evaluating system {i+1}/{len(systems)}: {system_name}")
                
                eval_id = f"{comparison_id}_system_{i}"
                
                system_eval = await self.evaluate_rag_system(
                    evaluation_id=eval_id,
                    queries=test_queries,
                    rag_system=system,
                    ground_truth=ground_truth,
                    relevant_docs=relevant_docs
                )
                
                comparison_results['system_results'][system_name] = system_eval
            
            # Calculate comparison metrics
            comparison_metrics = await self._calculate_comparison_metrics(
                comparison_results['system_results']
            )
            comparison_results['comparison_metrics'] = comparison_metrics
            
            comparison_results['end_time'] = datetime.utcnow()
            comparison_results['duration_seconds'] = (
                comparison_results['end_time'] - comparison_results['start_time']
            ).total_seconds()
            comparison_results['status'] = 'completed'
            
            # Store comparison results
            self.evaluation_cache[comparison_id] = comparison_results
            
            return {
                'status': 'success',
                **comparison_results
            }
            
        except Exception as e:
            logger.error(f"Error comparing RAG systems: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'comparison_id': comparison_id
            }
    
    async def _calculate_comparison_metrics(self, system_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate comparison metrics between systems
        """
        try:
            comparison = {
                'ranking': {},
                'statistical_significance': {},
                'metric_comparison': {}
            }
            
            # Extract aggregate metrics for each system
            system_metrics = {}
            for system_name, results in system_results.items():
                if results['status'] == 'success':
                    system_metrics[system_name] = results['aggregate_metrics']['metrics']
            
            if len(system_metrics) < 2:
                return comparison
            
            # Rank systems by overall score
            overall_scores = {
                name: metrics['overall_score']['mean'] 
                for name, metrics in system_metrics.items()
            }
            
            ranked_systems = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
            comparison['ranking']['overall_score'] = {
                'ranked_systems': ranked_systems,
                'best_system': ranked_systems[0][0],
                'worst_system': ranked_systems[-1][0]
            }
            
            # Metric-by-metric comparison
            metric_names = ['overall_score', 'faithfulness', 'answer_relevancy', 'retrieval_f1']
            
            for metric_name in metric_names:
                if all(metric_name in metrics for metrics in system_metrics.values()):
                    metric_values = {
                        name: metrics[metric_name]['mean'] 
                        for name, metrics in system_metrics.items()
                    }
                    
                    ranked_metric = sorted(metric_values.items(), key=lambda x: x[1], reverse=True)
                    comparison['metric_comparison'][metric_name] = {
                        'ranking': ranked_metric,
                        'best_system': ranked_metric[0][0],
                        'best_score': ranked_metric[0][1],
                        'score_range': ranked_metric[0][1] - ranked_metric[-1][1]
                    }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error calculating comparison metrics: {e}")
            return {}

# Global service instance
rag_evaluator_service = RAGEvaluatorService()
