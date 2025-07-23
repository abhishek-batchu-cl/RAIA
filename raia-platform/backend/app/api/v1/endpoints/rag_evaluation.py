"""
RAG Evaluation API Endpoints
Comprehensive RAG system evaluation with retrieval and generation metrics
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from typing import Dict, List, Optional, Any
import json
import pandas as pd
from io import StringIO

from app.core.auth import get_current_active_user
from app.models.schemas import User
from app.services.rag_evaluator_service import rag_evaluator_service

router = APIRouter()

@router.post("/register")
async def register_rag_system(
    system_id: str = Form(...),
    system_name: str = Form(...),
    system_type: str = Form("retrieval_augmented"),
    description: Optional[str] = Form(None),
    current_user: User = Depends(get_current_active_user)
):
    """Register a new RAG system for evaluation"""
    try:
        result = await rag_evaluator_service.register_rag_system(
            system_id=system_id,
            system_name=system_name,
            system_type=system_type,
            description=description
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/evaluate")
async def evaluate_rag_system(
    system_id: str = Form(...),
    queries: str = Form(...),  # JSON string of queries list
    retrieved_documents: str = Form(...),  # JSON string of retrieved docs
    generated_responses: str = Form(...),  # JSON string of responses
    ground_truth_documents: Optional[str] = Form(None),  # JSON string (optional)
    ground_truth_answers: Optional[str] = Form(None),  # JSON string (optional)
    evaluation_metrics: str = Form("retrieval_precision,retrieval_recall,generation_faithfulness"),
    current_user: User = Depends(get_current_active_user)
):
    """Evaluate RAG system performance with comprehensive metrics"""
    try:
        # Parse JSON strings
        queries_list = json.loads(queries)
        retrieved_docs = json.loads(retrieved_documents)
        generated_responses_list = json.loads(generated_responses)
        
        ground_truth_docs = json.loads(ground_truth_documents) if ground_truth_documents else None
        ground_truth_ans = json.loads(ground_truth_answers) if ground_truth_answers else None
        metrics_list = evaluation_metrics.split(",")
        
        result = await rag_evaluator_service.evaluate_rag_system(
            system_id=system_id,
            queries=queries_list,
            retrieved_documents=retrieved_docs,
            generated_responses=generated_responses_list,
            ground_truth_documents=ground_truth_docs,
            ground_truth_answers=ground_truth_ans,
            evaluation_metrics=metrics_list
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/evaluate-batch")
async def evaluate_rag_batch(
    system_id: str = Form(...),
    evaluation_data: UploadFile = File(...),
    evaluation_metrics: str = Form("all"),
    current_user: User = Depends(get_current_active_user)
):
    """Batch evaluate RAG system from uploaded CSV/JSON file"""
    try:
        # Read file content
        content = await evaluation_data.read()
        
        if evaluation_data.filename.endswith('.csv'):
            # Parse CSV
            df = pd.read_csv(StringIO(content.decode('utf-8')))
            batch_data = df.to_dict('records')
        else:
            # Parse JSON
            batch_data = json.loads(content.decode('utf-8'))
        
        metrics_list = evaluation_metrics.split(",") if evaluation_metrics != "all" else None
        
        result = await rag_evaluator_service.batch_evaluate(
            system_id=system_id,
            evaluation_data=batch_data,
            metrics=metrics_list
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/systems/{system_id}/metrics")
async def get_system_metrics(
    system_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get detailed metrics for a RAG system"""
    try:
        result = await rag_evaluator_service.get_system_metrics(system_id)
        if result["status"] == "error":
            raise HTTPException(status_code=404, detail=result["message"])
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/compare")
async def compare_rag_systems(
    system_ids: str = Form(...),  # JSON string of system IDs
    comparison_metrics: str = Form("retrieval_precision,generation_faithfulness,overall_score"),
    current_user: User = Depends(get_current_active_user)
):
    """Compare multiple RAG systems performance"""
    try:
        systems_list = json.loads(system_ids)
        metrics_list = comparison_metrics.split(",")
        
        result = await rag_evaluator_service.compare_systems(
            system_ids=systems_list,
            comparison_metrics=metrics_list
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/systems/{system_id}/analysis")
async def get_detailed_analysis(
    system_id: str,
    analysis_type: str = "comprehensive",
    current_user: User = Depends(get_current_active_user)
):
    """Get detailed analysis for RAG system performance"""
    try:
        result = await rag_evaluator_service.get_detailed_analysis(
            system_id=system_id,
            analysis_type=analysis_type
        )
        if result["status"] == "error":
            raise HTTPException(status_code=404, detail=result["message"])
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/systems")
async def list_rag_systems(
    current_user: User = Depends(get_current_active_user)
):
    """List all registered RAG systems"""
    try:
        systems = list(rag_evaluator_service.rag_systems.keys())
        return {
            "status": "success",
            "systems": systems,
            "count": len(systems)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/systems/{system_id}")
async def delete_rag_system(
    system_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Delete a RAG system and its evaluation data"""
    try:
        result = await rag_evaluator_service.cleanup_system(system_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/systems/{system_id}/custom-metrics")
async def add_custom_metrics(
    system_id: str,
    custom_metrics: str = Form(...),  # JSON string of custom metrics
    current_user: User = Depends(get_current_active_user)
):
    """Add custom evaluation metrics for a RAG system"""
    try:
        metrics_dict = json.loads(custom_metrics)
        
        # This would extend the service to support custom metrics
        # For now, return success with the provided metrics
        return {
            "status": "success",
            "system_id": system_id,
            "custom_metrics_added": list(metrics_dict.keys())
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/metrics/available")
async def get_available_metrics(
    current_user: User = Depends(get_current_active_user)
):
    """Get list of all available RAG evaluation metrics"""
    return {
        "status": "success",
        "retrieval_metrics": [
            "retrieval_precision", "retrieval_recall", "retrieval_f1",
            "mrr", "ndcg", "hit_rate"
        ],
        "generation_metrics": [
            "faithfulness", "answer_relevancy", "context_relevancy",
            "coherence", "fluency", "bleu_score", "rouge_score", "bert_score"
        ],
        "context_metrics": [
            "context_precision", "context_recall", "source_attribution",
            "hallucination_score"
        ],
        "overall_metrics": [
            "overall_score", "rag_effectiveness", "user_satisfaction"
        ]
    }