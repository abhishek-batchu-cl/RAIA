# RAG System Evaluation Service - Retrieval-Augmented Generation Testing
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
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
import nltk
from nltk.corpus import stopwords
from collections import Counter
import spacy

from .models import LLMModel, RAGEvaluation, KnowledgeBase, EvaluationDataset
from ..exceptions import EvaluationError, ModelNotFoundError, ValidationError

logger = logging.getLogger(__name__)

class RAGEvaluationEngine:
    """Comprehensive RAG system evaluation engine"""
    
    def __init__(self, db: Session, openai_api_key: str = None, anthropic_api_key: str = None):
        self.db = db
        self.openai_client = openai.OpenAI(api_key=openai_api_key) if openai_api_key else None
        self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key) if anthropic_api_key else None
        
        # Initialize evaluation tools
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Load language processing tools
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Some features may be limited.")
            self.nlp = None
        
        try:
            self.stopwords = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords')
            self.stopwords = set(stopwords.words('english'))
    
    # ========================================================================================
    # CORE RAG EVALUATION METHODS
    # ========================================================================================
    
    async def evaluate_rag_system(self,
                                 model_id: str,
                                 evaluation_config: Dict[str, Any],
                                 user_id: str) -> RAGEvaluation:
        """Comprehensive evaluation of a RAG system"""
        
        # Get RAG model configuration
        rag_model = self.db.query(LLMModel).filter(
            LLMModel.id == model_id,
            LLMModel.is_rag_system == True
        ).first()
        
        if not rag_model:
            raise ModelNotFoundError(f"RAG model {model_id} not found")
        
        # Get knowledge base
        knowledge_base = self.db.query(KnowledgeBase).filter(
            KnowledgeBase.id == rag_model.knowledge_base_id
        ).first()
        
        if not knowledge_base:
            raise ValidationError(f"Knowledge base not found for RAG model {model_id}")
        
        # Create evaluation record
        evaluation = RAGEvaluation(
            llm_model_id=model_id,
            knowledge_base_id=knowledge_base.id,
            evaluation_name=evaluation_config['evaluation_name'],
            evaluation_type=evaluation_config.get('evaluation_type', 'end_to_end'),
            dataset_name=evaluation_config.get('dataset_name'),
            evaluation_questions=evaluation_config.get('questions', []),
            status='running',
            started_at=datetime.utcnow(),
            num_questions=len(evaluation_config.get('questions', [])),
            retrieval_config=rag_model.model_parameters,
            generation_config=evaluation_config.get('generation_config', {}),
            evaluation_config=evaluation_config,
            created_by=user_id
        )
        
        self.db.add(evaluation)
        self.db.commit()
        self.db.refresh(evaluation)
        
        try:
            # Load evaluation questions
            questions = await self._load_rag_evaluation_questions(evaluation_config)
            
            # Load knowledge base documents
            documents = await self._load_knowledge_base_documents(knowledge_base)
            
            # Perform RAG evaluation
            if evaluation_config.get('evaluation_type') == 'retrieval_only':
                results = await self._evaluate_retrieval_only(rag_model, questions, documents)
            elif evaluation_config.get('evaluation_type') == 'generation_only':
                results = await self._evaluate_generation_only(rag_model, questions, evaluation_config)
            else:
                # End-to-end evaluation
                results = await self._evaluate_end_to_end(rag_model, questions, documents, evaluation_config)
            
            # Calculate comprehensive metrics
            metrics = await self._calculate_rag_metrics(results, questions, documents)
            
            # Perform detailed analysis
            detailed_analysis = await self._perform_rag_analysis(results, questions, documents)
            
            # Update evaluation with results
            evaluation.status = 'completed'
            evaluation.completed_at = datetime.utcnow()
            evaluation.duration_minutes = int((evaluation.completed_at - evaluation.started_at).total_seconds() / 60)
            
            # Retrieval metrics
            evaluation.retrieval_precision_at_k = metrics.get('retrieval_precision_at_k', {})
            evaluation.retrieval_recall_at_k = metrics.get('retrieval_recall_at_k', {})
            evaluation.retrieval_map = metrics.get('retrieval_map', 0.0)
            evaluation.retrieval_mrr = metrics.get('retrieval_mrr', 0.0)
            evaluation.retrieval_ndcg = metrics.get('retrieval_ndcg', 0.0)
            
            # Context quality metrics
            evaluation.context_relevance = metrics.get('context_relevance', 0.0)
            evaluation.context_precision = metrics.get('context_precision', 0.0)
            evaluation.context_recall = metrics.get('context_recall', 0.0)
            evaluation.context_coverage = metrics.get('context_coverage', 0.0)
            
            # Generation metrics
            evaluation.answer_accuracy = metrics.get('answer_accuracy', 0.0)
            evaluation.answer_completeness = metrics.get('answer_completeness', 0.0)
            evaluation.answer_consistency = metrics.get('answer_consistency', 0.0)
            evaluation.answer_conciseness = metrics.get('answer_conciseness', 0.0)
            
            # Factual accuracy
            evaluation.factual_correctness = metrics.get('factual_correctness', 0.0)
            evaluation.groundedness_score = metrics.get('groundedness_score', 0.0)
            evaluation.citation_accuracy = metrics.get('citation_accuracy', 0.0)
            evaluation.hallucination_rate = metrics.get('hallucination_rate', 0.0)
            
            # End-to-end RAG metrics
            evaluation.overall_rag_score = metrics.get('overall_rag_score', 0.0)
            evaluation.faithfulness = metrics.get('faithfulness', 0.0)
            evaluation.answer_relevance = metrics.get('answer_relevance', 0.0)
            evaluation.context_utilization = metrics.get('context_utilization', 0.0)
            
            # Semantic evaluation
            evaluation.semantic_answer_similarity = metrics.get('semantic_answer_similarity', 0.0)
            evaluation.embedding_similarity = metrics.get('embedding_similarity', 0.0)
            
            # Robustness metrics
            evaluation.consistency_across_rephrasing = metrics.get('consistency_across_rephrasing', 0.0)
            evaluation.sensitivity_to_context_order = metrics.get('sensitivity_to_context_order', 0.0)
            evaluation.noise_robustness = metrics.get('noise_robustness', 0.0)
            
            # Performance metrics
            evaluation.avg_retrieval_time_ms = int(metrics.get('avg_retrieval_time_ms', 0))
            evaluation.avg_generation_time_ms = int(metrics.get('avg_generation_time_ms', 0))
            evaluation.avg_end_to_end_time_ms = int(metrics.get('avg_end_to_end_time_ms', 0))
            
            # Cost metrics
            evaluation.total_cost_usd = metrics.get('total_cost_usd', 0.0)
            evaluation.cost_per_question = metrics.get('cost_per_question', 0.0)
            evaluation.retrieval_cost = metrics.get('retrieval_cost', 0.0)
            evaluation.generation_cost = metrics.get('generation_cost', 0.0)
            
            # Detailed analysis
            evaluation.question_level_results = detailed_analysis.get('question_level_results', [])
            evaluation.retrieval_analysis = detailed_analysis.get('retrieval_analysis', {})
            evaluation.error_analysis = detailed_analysis.get('error_analysis', {})
            evaluation.failure_cases = detailed_analysis.get('failure_cases', {})
            
            self.db.commit()
            self.db.refresh(evaluation)
            
            logger.info(f"Completed RAG evaluation {evaluation.id} for model {model_id}")
            
            return evaluation
            
        except Exception as e:
            evaluation.status = 'failed'
            evaluation.completed_at = datetime.utcnow()
            self.db.commit()
            
            logger.error(f"RAG evaluation failed: {str(e)}")
            raise EvaluationError(f"RAG evaluation failed: {str(e)}")
    
    async def _load_rag_evaluation_questions(self, evaluation_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Load RAG evaluation questions"""
        
        questions = evaluation_config.get('questions', [])
        if not questions and evaluation_config.get('dataset_name'):
            # Load from dataset
            dataset_record = self.db.query(EvaluationDataset).filter(
                EvaluationDataset.name == evaluation_config['dataset_name']
            ).first()
            
            if dataset_record:
                questions = self._load_dataset_questions(dataset_record.storage_location)
        
        # Ensure required format
        formatted_questions = []
        for q in questions:
            if isinstance(q, str):
                formatted_questions.append({
                    'question': q,
                    'expected_answer': None,
                    'relevant_documents': []
                })
            else:
                formatted_questions.append(q)
        
        return formatted_questions
    
    async def _load_knowledge_base_documents(self, knowledge_base: KnowledgeBase) -> List[Dict[str, Any]]:
        """Load documents from knowledge base"""
        
        # This is a simplified implementation
        # In practice, you'd load from the actual vector store/document store
        
        documents = []
        try:
            # Simulate loading documents
            for i in range(min(1000, knowledge_base.total_documents or 100)):
                documents.append({
                    'id': f'doc_{i}',
                    'content': f'Sample document content {i}...',
                    'metadata': {'source': f'source_{i}', 'type': 'text'},
                    'embedding': np.random.rand(384).tolist()  # Simulate embedding
                })
        except Exception as e:
            logger.warning(f"Failed to load knowledge base documents: {str(e)}")
        
        return documents
    
    async def _evaluate_end_to_end(self,
                                 rag_model: LLMModel,
                                 questions: List[Dict[str, Any]],
                                 documents: List[Dict[str, Any]],
                                 evaluation_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform end-to-end RAG evaluation"""
        
        results = []
        
        # Set up concurrent processing
        semaphore = asyncio.Semaphore(evaluation_config.get('max_concurrent_requests', 3))
        
        async def evaluate_single_question(question_data: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                try:
                    start_time = datetime.utcnow()
                    
                    # Step 1: Retrieval
                    retrieval_start = datetime.utcnow()
                    retrieved_docs = await self._retrieve_documents(
                        question_data['question'], documents, rag_model
                    )
                    retrieval_time = (datetime.utcnow() - retrieval_start).total_seconds() * 1000
                    
                    # Step 2: Generation
                    generation_start = datetime.utcnow()
                    generated_answer, tokens_used, cost = await self._generate_answer(
                        rag_model, question_data['question'], retrieved_docs
                    )
                    generation_time = (datetime.utcnow() - generation_start).total_seconds() * 1000
                    
                    total_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                    
                    return {
                        'question': question_data['question'],
                        'expected_answer': question_data.get('expected_answer'),
                        'relevant_documents': question_data.get('relevant_documents', []),
                        'retrieved_documents': retrieved_docs,
                        'generated_answer': generated_answer,
                        'retrieval_time_ms': retrieval_time,
                        'generation_time_ms': generation_time,
                        'total_time_ms': total_time,
                        'tokens_used': tokens_used,
                        'cost': cost
                    }
                    
                except Exception as e:
                    logger.error(f"Error evaluating question: {str(e)}")
                    return {
                        'question': question_data['question'],
                        'error': str(e),
                        'generated_answer': '',
                        'retrieved_documents': [],
                        'retrieval_time_ms': 0,
                        'generation_time_ms': 0,
                        'total_time_ms': 0,
                        'tokens_used': 0,
                        'cost': 0.0
                    }
        
        # Process all questions concurrently
        tasks = [evaluate_single_question(q) for q in questions]
        results = await asyncio.gather(*tasks)
        
        return results
    
    async def _retrieve_documents(self,
                                question: str,
                                documents: List[Dict[str, Any]],
                                rag_model: LLMModel) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a question"""
        
        # Encode question
        question_embedding = self.sentence_transformer.encode([question])[0]
        
        # Calculate similarities with all documents
        doc_similarities = []
        for doc in documents:
            if 'embedding' in doc:
                doc_embedding = np.array(doc['embedding'])
                similarity = cosine_similarity([question_embedding], [doc_embedding])[0][0]
                doc_similarities.append((doc, similarity))
        
        # Sort by similarity and get top-k
        k = rag_model.retrieval_k or 5
        doc_similarities.sort(key=lambda x: x[1], reverse=True)
        
        retrieved_docs = []
        for doc, similarity in doc_similarities[:k]:
            retrieved_docs.append({
                'id': doc['id'],
                'content': doc['content'],
                'metadata': doc.get('metadata', {}),
                'similarity_score': similarity
            })
        
        return retrieved_docs
    
    async def _generate_answer(self,
                             rag_model: LLMModel,
                             question: str,
                             retrieved_docs: List[Dict[str, Any]]) -> Tuple[str, int, float]:
        """Generate answer using retrieved context"""
        
        # Prepare context from retrieved documents
        context = "\n\n".join([
            f"Document {i+1}: {doc['content'][:500]}..."
            for i, doc in enumerate(retrieved_docs)
        ])
        
        # Prepare RAG prompt
        prompt = f"""Based on the following context, please answer the question.
        
Context:
{context}

Question: {question}

Answer:"""
        
        # Get model client
        client = self._get_model_client(rag_model)
        
        # Generate response
        try:
            if rag_model.provider.lower() == 'openai':
                response = await client.chat.completions.create(
                    model=rag_model.model_version,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=rag_model.model_parameters.get('temperature', 0.3),
                    max_tokens=rag_model.model_parameters.get('max_tokens', 500)
                )
                
                answer = response.choices[0].message.content
                tokens_used = response.usage.total_tokens
                
            elif rag_model.provider.lower() == 'anthropic':
                response = await client.messages.create(
                    model=rag_model.model_version,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=rag_model.model_parameters.get('temperature', 0.3),
                    max_tokens=rag_model.model_parameters.get('max_tokens', 500)
                )
                
                answer = response.content[0].text
                tokens_used = response.usage.input_tokens + response.usage.output_tokens
            
            else:
                raise EvaluationError(f"Unsupported provider: {rag_model.provider}")
            
            # Calculate cost
            cost = tokens_used * (rag_model.cost_per_1k_tokens or 0.002) / 1000
            
            return answer, tokens_used, cost
            
        except Exception as e:
            logger.error(f"Answer generation failed: {str(e)}")
            return "", 0, 0.0
    
    async def _calculate_rag_metrics(self,
                                   results: List[Dict[str, Any]],
                                   questions: List[Dict[str, Any]],
                                   documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive RAG metrics"""
        
        valid_results = [r for r in results if 'error' not in r]
        
        if not valid_results:
            return {'overall_rag_score': 0.0}
        
        metrics = {}
        
        # Calculate retrieval metrics
        retrieval_metrics = self._calculate_retrieval_metrics(valid_results, questions)
        metrics.update(retrieval_metrics)
        
        # Calculate generation metrics
        generation_metrics = self._calculate_generation_metrics(valid_results, questions)
        metrics.update(generation_metrics)
        
        # Calculate end-to-end metrics
        end_to_end_metrics = self._calculate_end_to_end_metrics(valid_results, questions)
        metrics.update(end_to_end_metrics)
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(valid_results)
        metrics.update(performance_metrics)
        
        # Calculate cost metrics
        cost_metrics = self._calculate_cost_metrics(valid_results)
        metrics.update(cost_metrics)
        
        # Calculate overall RAG score
        rag_score = self._calculate_overall_rag_score(metrics)
        metrics['overall_rag_score'] = rag_score
        
        return metrics
    
    def _calculate_retrieval_metrics(self,
                                   results: List[Dict[str, Any]],
                                   questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate retrieval-specific metrics"""
        
        precision_at_k = {k: [] for k in [1, 3, 5, 10]}
        recall_at_k = {k: [] for k in [1, 3, 5, 10]}
        reciprocal_ranks = []
        context_relevance_scores = []
        
        for result, question in zip(results, questions):
            retrieved_docs = result.get('retrieved_documents', [])
            relevant_docs = set(question.get('relevant_documents', []))
            
            if not relevant_docs:
                # If no ground truth, use similarity scores as proxy
                if retrieved_docs:
                    avg_similarity = np.mean([doc.get('similarity_score', 0) for doc in retrieved_docs])
                    context_relevance_scores.append(avg_similarity)
                continue
            
            retrieved_ids = [doc['id'] for doc in retrieved_docs]
            
            # Calculate precision@k and recall@k
            for k in precision_at_k.keys():
                retrieved_at_k = set(retrieved_ids[:k])
                relevant_retrieved = retrieved_at_k.intersection(relevant_docs)
                
                precision = len(relevant_retrieved) / k if k > 0 else 0
                recall = len(relevant_retrieved) / len(relevant_docs) if relevant_docs else 0
                
                precision_at_k[k].append(precision)
                recall_at_k[k].append(recall)
            
            # Calculate Mean Reciprocal Rank
            for i, doc_id in enumerate(retrieved_ids):
                if doc_id in relevant_docs:
                    reciprocal_ranks.append(1.0 / (i + 1))
                    break
            else:
                reciprocal_ranks.append(0.0)
            
            # Context relevance (if we have retrieved docs)
            if retrieved_docs:
                relevance_scores = []
                for doc in retrieved_docs:
                    if doc['id'] in relevant_docs:
                        relevance_scores.append(1.0)
                    else:
                        # Use similarity as proxy for relevance
                        relevance_scores.append(doc.get('similarity_score', 0.0))
                context_relevance_scores.append(np.mean(relevance_scores))
        
        return {
            'retrieval_precision_at_k': {
                f'p_at_{k}': np.mean(scores) if scores else 0.0
                for k, scores in precision_at_k.items()
            },
            'retrieval_recall_at_k': {
                f'r_at_{k}': np.mean(scores) if scores else 0.0
                for k, scores in recall_at_k.items()
            },
            'retrieval_mrr': np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0,
            'context_relevance': np.mean(context_relevance_scores) if context_relevance_scores else 0.0,
            'context_precision': np.mean(precision_at_k[5]) if precision_at_k[5] else 0.0,
            'context_recall': np.mean(recall_at_k[5]) if recall_at_k[5] else 0.0
        }
    
    def _calculate_generation_metrics(self,
                                    results: List[Dict[str, Any]],
                                    questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate generation-specific metrics"""
        
        accuracy_scores = []
        completeness_scores = []
        consistency_scores = []
        conciseness_scores = []
        groundedness_scores = []
        
        for result, question in zip(results, questions):
            generated_answer = result.get('generated_answer', '')
            expected_answer = question.get('expected_answer')
            retrieved_docs = result.get('retrieved_documents', [])
            
            if not generated_answer:
                continue
            
            # Answer accuracy (if we have expected answer)
            if expected_answer:
                # Use semantic similarity as accuracy measure
                try:
                    gen_emb = self.sentence_transformer.encode([generated_answer])
                    exp_emb = self.sentence_transformer.encode([expected_answer])
                    accuracy = cosine_similarity(gen_emb, exp_emb)[0][0]
                    accuracy_scores.append(max(0, accuracy))
                except:
                    accuracy_scores.append(0.0)
            
            # Completeness (coverage of question aspects)
            completeness = self._calculate_completeness(generated_answer, question['question'])
            completeness_scores.append(completeness)
            
            # Consistency with retrieved context
            if retrieved_docs:
                consistency = self._calculate_consistency(generated_answer, retrieved_docs)
                consistency_scores.append(consistency)
            
            # Conciseness (information density)
            conciseness = self._calculate_conciseness(generated_answer)
            conciseness_scores.append(conciseness)
            
            # Groundedness in retrieved documents
            if retrieved_docs:
                groundedness = self._calculate_groundedness(generated_answer, retrieved_docs)
                groundedness_scores.append(groundedness)
        
        return {
            'answer_accuracy': np.mean(accuracy_scores) if accuracy_scores else 0.0,
            'answer_completeness': np.mean(completeness_scores) if completeness_scores else 0.0,
            'answer_consistency': np.mean(consistency_scores) if consistency_scores else 0.0,
            'answer_conciseness': np.mean(conciseness_scores) if conciseness_scores else 0.0,
            'groundedness_score': np.mean(groundedness_scores) if groundedness_scores else 0.0
        }
    
    def _calculate_end_to_end_metrics(self,
                                    results: List[Dict[str, Any]],
                                    questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate end-to-end RAG metrics"""
        
        faithfulness_scores = []
        answer_relevance_scores = []
        context_utilization_scores = []
        semantic_similarity_scores = []
        
        for result, question in zip(results, questions):
            generated_answer = result.get('generated_answer', '')
            retrieved_docs = result.get('retrieved_documents', [])
            
            if not generated_answer:
                continue
            
            # Faithfulness: how faithful the answer is to the retrieved context
            if retrieved_docs:
                faithfulness = self._calculate_faithfulness(generated_answer, retrieved_docs)
                faithfulness_scores.append(faithfulness)
            
            # Answer relevance: how relevant the answer is to the question
            relevance = self._calculate_answer_relevance(generated_answer, question['question'])
            answer_relevance_scores.append(relevance)
            
            # Context utilization: how well the retrieved context was used
            if retrieved_docs:
                utilization = self._calculate_context_utilization(generated_answer, retrieved_docs)
                context_utilization_scores.append(utilization)
            
            # Semantic similarity to expected answer (if available)
            if question.get('expected_answer'):
                try:
                    gen_emb = self.sentence_transformer.encode([generated_answer])
                    exp_emb = self.sentence_transformer.encode([question['expected_answer']])
                    similarity = cosine_similarity(gen_emb, exp_emb)[0][0]
                    semantic_similarity_scores.append(max(0, similarity))
                except:
                    semantic_similarity_scores.append(0.0)
        
        return {
            'faithfulness': np.mean(faithfulness_scores) if faithfulness_scores else 0.0,
            'answer_relevance': np.mean(answer_relevance_scores) if answer_relevance_scores else 0.0,
            'context_utilization': np.mean(context_utilization_scores) if context_utilization_scores else 0.0,
            'semantic_answer_similarity': np.mean(semantic_similarity_scores) if semantic_similarity_scores else 0.0,
            'embedding_similarity': np.mean(semantic_similarity_scores) if semantic_similarity_scores else 0.0
        }
    
    def _calculate_performance_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate performance metrics"""
        
        retrieval_times = [r.get('retrieval_time_ms', 0) for r in results if 'error' not in r]
        generation_times = [r.get('generation_time_ms', 0) for r in results if 'error' not in r]
        total_times = [r.get('total_time_ms', 0) for r in results if 'error' not in r]
        
        return {
            'avg_retrieval_time_ms': np.mean(retrieval_times) if retrieval_times else 0,
            'avg_generation_time_ms': np.mean(generation_times) if generation_times else 0,
            'avg_end_to_end_time_ms': np.mean(total_times) if total_times else 0
        }
    
    def _calculate_cost_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate cost metrics"""
        
        costs = [r.get('cost', 0.0) for r in results if 'error' not in r]
        total_cost = sum(costs)
        
        # Estimate retrieval cost (usually much lower than generation)
        retrieval_cost = total_cost * 0.1  # Rough estimate
        generation_cost = total_cost * 0.9
        
        return {
            'total_cost_usd': total_cost,
            'cost_per_question': total_cost / len(results) if results else 0.0,
            'retrieval_cost': retrieval_cost,
            'generation_cost': generation_cost
        }
    
    def _calculate_overall_rag_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall RAG performance score"""
        
        # Weighted combination of key metrics
        weights = {
            'context_relevance': 0.15,
            'answer_accuracy': 0.20,
            'faithfulness': 0.20,
            'answer_relevance': 0.15,
            'groundedness_score': 0.15,
            'context_utilization': 0.10,
            'answer_completeness': 0.05
        }
        
        score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in metrics:
                score += metrics[metric] * weight
                total_weight += weight
        
        return score / total_weight if total_weight > 0 else 0.0
    
    # ========================================================================================
    # HELPER METHODS FOR SPECIFIC METRIC CALCULATIONS
    # ========================================================================================
    
    def _calculate_completeness(self, answer: str, question: str) -> float:
        """Calculate how complete the answer is"""
        
        # Extract key concepts from question
        question_words = set(question.lower().split()) - self.stopwords
        answer_words = set(answer.lower().split()) - self.stopwords
        
        # Calculate concept coverage
        if not question_words:
            return 1.0
        
        coverage = len(question_words.intersection(answer_words)) / len(question_words)
        
        # Factor in answer length (completeness also depends on sufficient detail)
        length_factor = min(1.0, len(answer.split()) / 50)  # Assume 50 words is sufficient
        
        return (coverage + length_factor) / 2
    
    def _calculate_consistency(self, answer: str, retrieved_docs: List[Dict[str, Any]]) -> float:
        """Calculate consistency between answer and retrieved context"""
        
        if not retrieved_docs:
            return 0.0
        
        # Combine all retrieved context
        context = " ".join([doc.get('content', '') for doc in retrieved_docs])
        
        try:
            # Use semantic similarity to measure consistency
            answer_emb = self.sentence_transformer.encode([answer])
            context_emb = self.sentence_transformer.encode([context])
            
            consistency = cosine_similarity(answer_emb, context_emb)[0][0]
            return max(0, consistency)
        except:
            return 0.0
    
    def _calculate_conciseness(self, answer: str) -> float:
        """Calculate conciseness of the answer"""
        
        words = answer.split()
        if not words:
            return 0.0
        
        # Measure information density (inverse of redundancy)
        unique_words = set(word.lower() for word in words)
        word_diversity = len(unique_words) / len(words)
        
        # Penalize extremely long or short answers
        length_penalty = 1.0
        if len(words) > 200:
            length_penalty = 200 / len(words)
        elif len(words) < 10:
            length_penalty = len(words) / 10
        
        return word_diversity * length_penalty
    
    def _calculate_groundedness(self, answer: str, retrieved_docs: List[Dict[str, Any]]) -> float:
        """Calculate how well the answer is grounded in retrieved documents"""
        
        if not retrieved_docs:
            return 0.0
        
        # Extract key claims/facts from answer
        answer_sentences = [s.strip() for s in answer.split('.') if s.strip()]
        
        grounded_sentences = 0
        total_sentences = len(answer_sentences)
        
        if total_sentences == 0:
            return 0.0
        
        for sentence in answer_sentences:
            # Check if sentence can be supported by any retrieved document
            sentence_emb = self.sentence_transformer.encode([sentence])
            
            max_support = 0.0
            for doc in retrieved_docs:
                doc_content = doc.get('content', '')
                if doc_content:
                    try:
                        doc_emb = self.sentence_transformer.encode([doc_content])
                        support = cosine_similarity(sentence_emb, doc_emb)[0][0]
                        max_support = max(max_support, support)
                    except:
                        continue
            
            # Consider sentence grounded if similarity > threshold
            if max_support > 0.5:
                grounded_sentences += 1
        
        return grounded_sentences / total_sentences
    
    def _calculate_faithfulness(self, answer: str, retrieved_docs: List[Dict[str, Any]]) -> float:
        """Calculate faithfulness of answer to retrieved context"""
        
        # This is similar to groundedness but focuses on factual consistency
        return self._calculate_groundedness(answer, retrieved_docs)
    
    def _calculate_answer_relevance(self, answer: str, question: str) -> float:
        """Calculate how relevant the answer is to the question"""
        
        try:
            answer_emb = self.sentence_transformer.encode([answer])
            question_emb = self.sentence_transformer.encode([question])
            
            relevance = cosine_similarity(answer_emb, question_emb)[0][0]
            return max(0, relevance)
        except:
            return 0.0
    
    def _calculate_context_utilization(self, answer: str, retrieved_docs: List[Dict[str, Any]]) -> float:
        """Calculate how well the retrieved context was utilized"""
        
        if not retrieved_docs:
            return 0.0
        
        utilized_docs = 0
        total_docs = len(retrieved_docs)
        
        for doc in retrieved_docs:
            doc_content = doc.get('content', '')
            if doc_content:
                try:
                    # Check if any part of the document content appears in answer
                    doc_emb = self.sentence_transformer.encode([doc_content])
                    answer_emb = self.sentence_transformer.encode([answer])
                    
                    utilization = cosine_similarity(doc_emb, answer_emb)[0][0]
                    
                    # Consider utilized if similarity > threshold
                    if utilization > 0.3:
                        utilized_docs += 1
                except:
                    continue
        
        return utilized_docs / total_docs if total_docs > 0 else 0.0
    
    async def _perform_rag_analysis(self,
                                  results: List[Dict[str, Any]],
                                  questions: List[Dict[str, Any]],
                                  documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform detailed RAG analysis"""
        
        return {
            'question_level_results': [
                {
                    'question_id': i,
                    'question': result.get('question', ''),
                    'answer': result.get('generated_answer', ''),
                    'retrieved_doc_count': len(result.get('retrieved_documents', [])),
                    'retrieval_time_ms': result.get('retrieval_time_ms', 0),
                    'generation_time_ms': result.get('generation_time_ms', 0),
                    'total_time_ms': result.get('total_time_ms', 0),
                    'cost': result.get('cost', 0.0)
                }
                for i, result in enumerate(results)
            ],
            'retrieval_analysis': {
                'avg_docs_retrieved': np.mean([
                    len(r.get('retrieved_documents', [])) for r in results
                ]),
                'avg_similarity_score': np.mean([
                    np.mean([doc.get('similarity_score', 0) for doc in r.get('retrieved_documents', [])])
                    for r in results if r.get('retrieved_documents')
                ])
            },
            'error_analysis': self._analyze_rag_errors(results),
            'failure_cases': self._identify_rag_failures(results, questions)
        }
    
    def _analyze_rag_errors(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze common error patterns in RAG evaluation"""
        
        error_patterns = {
            'no_relevant_docs_retrieved': 0,
            'generation_failures': 0,
            'empty_answers': 0,
            'hallucinated_answers': 0
        }
        
        for result in results:
            if 'error' in result:
                error_patterns['generation_failures'] += 1
                continue
            
            retrieved_docs = result.get('retrieved_documents', [])
            answer = result.get('generated_answer', '')
            
            # No relevant documents
            if not retrieved_docs or all(doc.get('similarity_score', 0) < 0.3 for doc in retrieved_docs):
                error_patterns['no_relevant_docs_retrieved'] += 1
            
            # Empty or very short answers
            if not answer or len(answer.split()) < 3:
                error_patterns['empty_answers'] += 1
            
            # Potential hallucination (answer mentions facts not in retrieved docs)
            if retrieved_docs and answer:
                # Simple heuristic: if answer is very different from all retrieved docs
                try:
                    answer_emb = self.sentence_transformer.encode([answer])
                    max_similarity = 0.0
                    
                    for doc in retrieved_docs:
                        if doc.get('content'):
                            doc_emb = self.sentence_transformer.encode([doc['content']])
                            sim = cosine_similarity(answer_emb, doc_emb)[0][0]
                            max_similarity = max(max_similarity, sim)
                    
                    if max_similarity < 0.2:  # Very low similarity suggests hallucination
                        error_patterns['hallucinated_answers'] += 1
                except:
                    pass
        
        return error_patterns
    
    def _identify_rag_failures(self,
                              results: List[Dict[str, Any]],
                              questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify specific failure cases for analysis"""
        
        failures = []
        
        for i, (result, question) in enumerate(zip(results, questions)):
            if 'error' in result or not result.get('generated_answer'):
                failures.append({
                    'question_id': i,
                    'question': question['question'],
                    'failure_type': 'generation_error',
                    'details': result.get('error', 'Empty answer generated')
                })
            
            elif len(result.get('retrieved_documents', [])) == 0:
                failures.append({
                    'question_id': i,
                    'question': question['question'],
                    'failure_type': 'retrieval_failure',
                    'details': 'No documents retrieved'
                })
        
        return {
            'total_failures': len(failures),
            'failure_rate': len(failures) / len(results) if results else 0,
            'failure_details': failures[:10]  # Show first 10 failures
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
    
    def _load_dataset_questions(self, storage_location: str) -> List[Dict[str, Any]]:
        """Load questions from dataset storage location"""
        
        try:
            if storage_location.endswith('.json'):
                with open(storage_location, 'r') as f:
                    return json.load(f)
            elif storage_location.endswith('.jsonl'):
                questions = []
                with open(storage_location, 'r') as f:
                    for line in f:
                        questions.append(json.loads(line))
                return questions
            else:
                # Return sample questions for demonstration
                return [
                    {
                        "question": "What is machine learning?",
                        "expected_answer": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
                        "relevant_documents": ["doc_1", "doc_5", "doc_12"]
                    },
                    {
                        "question": "How do neural networks work?",
                        "expected_answer": "Neural networks work by processing data through interconnected layers of artificial neurons, each applying weights and activation functions to transform inputs into outputs.",
                        "relevant_documents": ["doc_3", "doc_7", "doc_15"]
                    }
                ]
        except Exception as e:
            logger.warning(f"Failed to load questions from {storage_location}: {str(e)}")
            return []
    
    async def _evaluate_retrieval_only(self,
                                     rag_model: LLMModel,
                                     questions: List[Dict[str, Any]],
                                     documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate retrieval component only"""
        
        results = []
        
        for question_data in questions:
            start_time = datetime.utcnow()
            
            retrieved_docs = await self._retrieve_documents(
                question_data['question'], documents, rag_model
            )
            
            retrieval_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            results.append({
                'question': question_data['question'],
                'relevant_documents': question_data.get('relevant_documents', []),
                'retrieved_documents': retrieved_docs,
                'retrieval_time_ms': retrieval_time,
                'generation_time_ms': 0,
                'total_time_ms': retrieval_time,
                'generated_answer': None,
                'tokens_used': 0,
                'cost': 0.0
            })
        
        return results
    
    async def _evaluate_generation_only(self,
                                      rag_model: LLMModel,
                                      questions: List[Dict[str, Any]],
                                      evaluation_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Evaluate generation component with provided context"""
        
        results = []
        
        for question_data in questions:
            # Use provided context or simulate perfect retrieval
            context_docs = question_data.get('context_documents', [
                {'id': 'perfect_doc', 'content': 'Perfect context for the question...'}
            ])
            
            start_time = datetime.utcnow()
            
            generated_answer, tokens_used, cost = await self._generate_answer(
                rag_model, question_data['question'], context_docs
            )
            
            generation_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            results.append({
                'question': question_data['question'],
                'expected_answer': question_data.get('expected_answer'),
                'retrieved_documents': context_docs,
                'generated_answer': generated_answer,
                'retrieval_time_ms': 0,
                'generation_time_ms': generation_time,
                'total_time_ms': generation_time,
                'tokens_used': tokens_used,
                'cost': cost
            })
        
        return results


class RAGEvaluationError(Exception):
    """Custom exception for RAG evaluation errors"""
    pass