"""
LLM Evaluation API Endpoints
Comprehensive LLM evaluation with content quality, language quality, and safety metrics
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from typing import Dict, List, Optional, Any
import json
import pandas as pd
from io import StringIO

from app.core.auth import get_current_active_user
from app.models.schemas import User
from app.services.llm_evaluator_service import llm_evaluator_service

router = APIRouter()

@router.post("/register")
async def register_llm_model(
    model_id: str = Form(...),
    model_name: str = Form(...),
    model_type: str = Form("text_generation"),
    provider: str = Form("custom"),
    description: Optional[str] = Form(None),
    current_user: User = Depends(get_current_active_user)
):
    """Register a new LLM model for evaluation"""
    try:
        result = await llm_evaluator_service.register_model(
            model_id=model_id,
            model_name=model_name,
            model_type=model_type,
            provider=provider,
            description=description
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/evaluate")
async def evaluate_llm_responses(
    model_id: str = Form(...),
    inputs: str = Form(...),  # JSON string of input prompts
    outputs: str = Form(...),  # JSON string of model outputs
    ground_truth: Optional[str] = Form(None),  # JSON string of expected outputs
    evaluation_framework: str = Form("comprehensive"),
    task_type: str = Form("general"),
    current_user: User = Depends(get_current_active_user)
):
    """Evaluate LLM responses with comprehensive metrics"""
    try:
        # Parse JSON strings
        inputs_list = json.loads(inputs)
        outputs_list = json.loads(outputs)
        ground_truth_list = json.loads(ground_truth) if ground_truth else None
        
        result = await llm_evaluator_service.evaluate_responses(
            model_id=model_id,
            inputs=inputs_list,
            outputs=outputs_list,
            ground_truth=ground_truth_list,
            evaluation_framework=evaluation_framework,
            task_type=task_type
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/evaluate-batch")
async def batch_evaluate_llm(
    model_id: str = Form(...),
    evaluation_data: UploadFile = File(...),
    evaluation_framework: str = Form("comprehensive"),
    task_type: str = Form("general"),
    current_user: User = Depends(get_current_active_user)
):
    """Batch evaluate LLM from uploaded file"""
    try:
        # Read file content
        content = await evaluation_data.read()
        
        if evaluation_data.filename.endswith('.csv'):
            df = pd.read_csv(StringIO(content.decode('utf-8')))
            batch_data = df.to_dict('records')
        else:
            batch_data = json.loads(content.decode('utf-8'))
        
        result = await llm_evaluator_service.batch_evaluate(
            model_id=model_id,
            evaluation_data=batch_data,
            evaluation_framework=evaluation_framework,
            task_type=task_type
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/compare")
async def compare_llm_models(
    model_ids: str = Form(...),  # JSON string of model IDs
    comparison_data: str = Form(...),  # JSON string of evaluation data
    comparison_metrics: str = Form("content_quality,language_quality,safety_score"),
    current_user: User = Depends(get_current_active_user)
):
    """Compare multiple LLM models performance"""
    try:
        models_list = json.loads(model_ids)
        eval_data = json.loads(comparison_data)
        metrics_list = comparison_metrics.split(",")
        
        result = await llm_evaluator_service.compare_models(
            model_ids=models_list,
            evaluation_data=eval_data,
            comparison_metrics=metrics_list
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/models/{model_id}/metrics")
async def get_model_metrics(
    model_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get detailed metrics for an LLM model"""
    try:
        result = await llm_evaluator_service.get_model_metrics(model_id)
        if result["status"] == "error":
            raise HTTPException(status_code=404, detail=result["message"])
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/evaluate-safety")
async def evaluate_safety(
    model_id: str = Form(...),
    texts: str = Form(...),  # JSON string of texts to evaluate
    safety_categories: str = Form("toxicity,bias,harmful_content"),
    current_user: User = Depends(get_current_active_user)
):
    """Evaluate text safety across multiple categories"""
    try:
        texts_list = json.loads(texts)
        categories = safety_categories.split(",")
        
        result = await llm_evaluator_service.evaluate_safety(
            model_id=model_id,
            texts=texts_list,
            safety_categories=categories
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/evaluate-task-specific")
async def evaluate_task_specific(
    model_id: str = Form(...),
    task_type: str = Form(...),  # summarization, qa, translation, etc.
    inputs: str = Form(...),
    outputs: str = Form(...),
    references: Optional[str] = Form(None),
    current_user: User = Depends(get_current_active_user)
):
    """Evaluate LLM for specific tasks with task-appropriate metrics"""
    try:
        inputs_list = json.loads(inputs)
        outputs_list = json.loads(outputs)
        references_list = json.loads(references) if references else None
        
        result = await llm_evaluator_service.evaluate_task_specific(
            model_id=model_id,
            task_type=task_type,
            inputs=inputs_list,
            outputs=outputs_list,
            references=references_list
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/models/{model_id}/analysis")
async def get_detailed_analysis(
    model_id: str,
    analysis_type: str = "comprehensive",
    current_user: User = Depends(get_current_active_user)
):
    """Get detailed analysis for LLM model performance"""
    try:
        result = await llm_evaluator_service.get_detailed_analysis(
            model_id=model_id,
            analysis_type=analysis_type
        )
        if result["status"] == "error":
            raise HTTPException(status_code=404, detail=result["message"])
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/models")
async def list_llm_models(
    current_user: User = Depends(get_current_active_user)
):
    """List all registered LLM models"""
    try:
        models = list(llm_evaluator_service.models.keys())
        return {
            "status": "success",
            "models": models,
            "count": len(models)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/models/{model_id}")
async def delete_llm_model(
    model_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Delete an LLM model and its evaluation data"""
    try:
        result = await llm_evaluator_service.cleanup_model(model_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/frameworks/available")
async def get_available_frameworks(
    current_user: User = Depends(get_current_active_user)
):
    """Get list of available evaluation frameworks"""
    return {
        "status": "success",
        "frameworks": {
            "comprehensive": "Full evaluation across all dimensions",
            "content_quality": "Focus on factual accuracy and completeness", 
            "language_quality": "Focus on fluency and grammar",
            "safety": "Focus on toxicity and bias detection",
            "task_specific": "Task-optimized evaluation metrics"
        },
        "task_types": [
            "general", "summarization", "question_answering", "translation",
            "code_generation", "creative_writing", "dialogue", "instruction_following"
        ]
    }

@router.get("/metrics/available")
async def get_available_metrics(
    current_user: User = Depends(get_current_active_user)
):
    """Get list of all available LLM evaluation metrics"""
    return {
        "status": "success",
        "content_quality_metrics": [
            "factual_accuracy", "completeness", "relevancy", "logical_consistency"
        ],
        "language_quality_metrics": [
            "fluency", "grammar", "clarity", "conciseness", "coherence"
        ],
        "semantic_metrics": [
            "bert_score", "sentence_similarity", "semantic_coherence"
        ],
        "safety_metrics": [
            "toxicity_score", "bias_detection", "harmful_content", "ethical_alignment"
        ],
        "task_specific_metrics": {
            "summarization": ["rouge_score", "bleu_score", "extractiveness"],
            "qa": ["answer_accuracy", "answer_completeness", "context_relevance"],
            "translation": ["bleu_score", "meteor_score", "chrf_score"],
            "code": ["code_correctness", "code_efficiency", "code_style"]
        }
    }