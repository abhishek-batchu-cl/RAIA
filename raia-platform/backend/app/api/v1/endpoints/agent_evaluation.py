"""
RAIA Platform - Agent Evaluation API Endpoints
Complete REST API for agent evaluation, configuration, and testing
"""

import asyncio
from typing import Dict, List, Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import get_current_active_user, require_agent_evaluation
from app.core.database import get_database
from app.models.schemas import User, AgentConfigurationCreate, AgentConfigurationResponse, AgentEvaluationCreate, AgentEvaluationResponse, ChatRequest, ChatResponse
from app.services.agent_evaluation_service import AgentEvaluationService
from app.services.configuration_service import ConfigurationService
from app.services.document_service import DocumentService
from app.services.rag_agent import RAGAgentService

logger = structlog.get_logger(__name__)
router = APIRouter()

# Initialize services
eval_service = AgentEvaluationService()
config_service = ConfigurationService()
doc_service = DocumentService()
rag_service = RAGAgentService()


# Configuration Management Endpoints
@router.get("/configurations", response_model=List[AgentConfigurationResponse])
async def list_configurations(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    current_user: User = Depends(require_agent_evaluation),
    db: AsyncSession = Depends(get_database)
) -> List[AgentConfigurationResponse]:
    """
    List all agent configurations for the user's organization.
    
    - **skip**: Number of configurations to skip (pagination)
    - **limit**: Maximum number of configurations to return
    
    Returns list of agent configurations with metadata.
    """
    configurations = await config_service.list_configurations(
        db, 
        organization_id=current_user.organization_id,
        skip=skip,
        limit=limit
    )
    
    logger.info(
        "Listed agent configurations",
        user_id=current_user.id,
        org_id=current_user.organization_id,
        count=len(configurations)
    )
    
    return [AgentConfigurationResponse.from_orm(config) for config in configurations]


@router.post("/configurations", response_model=AgentConfigurationResponse)
async def create_configuration(
    config_data: AgentConfigurationCreate,
    current_user: User = Depends(require_agent_evaluation),
    db: AsyncSession = Depends(get_database)
) -> AgentConfigurationResponse:
    """
    Create a new agent configuration.
    
    - **name**: Configuration name (must be unique within organization)
    - **description**: Optional description
    - **model_provider**: LLM provider (openai, anthropic, local)
    - **model_name**: Specific model name
    - **system_prompt**: System prompt for the agent
    - **temperature**: Model temperature (0.0-1.0)
    - **max_tokens**: Maximum tokens in response
    - **rag_enabled**: Enable RAG functionality
    - **rag_settings**: RAG configuration settings
    """
    try:
        # Validate configuration
        await config_service.validate_configuration(config_data.dict())
        
        configuration = await config_service.create_configuration(
            db,
            config_data=config_data,
            organization_id=current_user.organization_id,
            created_by=current_user.id
        )
        
        logger.info(
            "Agent configuration created",
            user_id=current_user.id,
            config_id=configuration.id,
            name=configuration.name
        )
        
        return AgentConfigurationResponse.from_orm(configuration)
        
    except ValueError as e:
        logger.warning("Invalid configuration data", error=str(e), user_id=current_user.id)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Failed to create configuration", error=str(e), user_id=current_user.id)
        raise HTTPException(status_code=500, detail="Failed to create configuration")


@router.get("/configurations/{config_id}", response_model=AgentConfigurationResponse)
async def get_configuration(
    config_id: UUID,
    current_user: User = Depends(require_agent_evaluation),
    db: AsyncSession = Depends(get_database)
) -> AgentConfigurationResponse:
    """
    Get a specific agent configuration by ID.
    """
    configuration = await config_service.get_configuration(
        db, 
        config_id=config_id,
        organization_id=current_user.organization_id
    )
    
    if not configuration:
        raise HTTPException(status_code=404, detail="Configuration not found")
    
    return AgentConfigurationResponse.from_orm(configuration)


@router.put("/configurations/{config_id}", response_model=AgentConfigurationResponse)
async def update_configuration(
    config_id: UUID,
    config_update: dict,
    current_user: User = Depends(require_agent_evaluation),
    db: AsyncSession = Depends(get_database)
) -> AgentConfigurationResponse:
    """
    Update an existing agent configuration.
    """
    try:
        # Validate update data
        if config_update:
            await config_service.validate_configuration(config_update)
        
        configuration = await config_service.update_configuration(
            db,
            config_id=config_id,
            organization_id=current_user.organization_id,
            update_data=config_update
        )
        
        if not configuration:
            raise HTTPException(status_code=404, detail="Configuration not found")
        
        logger.info(
            "Agent configuration updated",
            user_id=current_user.id,
            config_id=config_id,
            fields=list(config_update.keys())
        )
        
        return AgentConfigurationResponse.from_orm(configuration)
        
    except ValueError as e:
        logger.warning("Invalid configuration update", error=str(e), user_id=current_user.id)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Failed to update configuration", error=str(e), user_id=current_user.id)
        raise HTTPException(status_code=500, detail="Failed to update configuration")


@router.delete("/configurations/{config_id}")
async def delete_configuration(
    config_id: UUID,
    current_user: User = Depends(require_agent_evaluation),
    db: AsyncSession = Depends(get_database)
) -> dict:
    """
    Delete an agent configuration.
    """
    success = await config_service.delete_configuration(
        db,
        config_id=config_id,
        organization_id=current_user.organization_id
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="Configuration not found")
    
    logger.info(
        "Agent configuration deleted",
        user_id=current_user.id,
        config_id=config_id
    )
    
    return {"message": "Configuration deleted successfully"}


# Chat Interface Endpoints
@router.post("/chat", response_model=ChatResponse)
async def chat_with_agent(
    chat_request: ChatRequest,
    current_user: User = Depends(require_agent_evaluation),
    db: AsyncSession = Depends(get_database)
) -> ChatResponse:
    """
    Chat with an agent using a specific configuration.
    
    - **message**: User message
    - **session_id**: Optional chat session ID (creates new session if not provided)
    - **configuration**: Chat configuration including config_id
    """
    try:
        config_id = chat_request.configuration.get("config_id")
        if not config_id:
            raise HTTPException(status_code=400, detail="config_id is required in configuration")
        
        # Get agent configuration
        configuration = await config_service.get_configuration(
            db,
            config_id=UUID(config_id),
            organization_id=current_user.organization_id
        )
        
        if not configuration:
            raise HTTPException(status_code=404, detail="Configuration not found")
        
        # Use RAG agent service to generate response
        response = await rag_service.chat(
            db,
            user_message=chat_request.message,
            configuration=configuration,
            session_id=chat_request.session_id,
            user_id=current_user.id,
            organization_id=current_user.organization_id
        )
        
        logger.info(
            "Chat response generated",
            user_id=current_user.id,
            config_id=config_id,
            session_id=response.session_id
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Chat request failed", error=str(e), user_id=current_user.id)
        raise HTTPException(status_code=500, detail="Failed to process chat request")


@router.get("/chat/sessions")
async def list_chat_sessions(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    current_user: User = Depends(require_agent_evaluation),
    db: AsyncSession = Depends(get_database)
) -> List[dict]:
    """
    List chat sessions for the current user.
    """
    sessions = await rag_service.list_chat_sessions(
        db,
        user_id=current_user.id,
        organization_id=current_user.organization_id,
        skip=skip,
        limit=limit
    )
    
    return [
        {
            "id": session.id,
            "name": session.name,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "message_count": len(session.messages),
            "configuration": session.configuration
        }
        for session in sessions
    ]


@router.get("/chat/sessions/{session_id}/history")
async def get_chat_history(
    session_id: UUID,
    current_user: User = Depends(require_agent_evaluation),
    db: AsyncSession = Depends(get_database)
) -> dict:
    """
    Get chat history for a specific session.
    """
    session = await rag_service.get_chat_session(
        db,
        session_id=session_id,
        user_id=current_user.id,
        organization_id=current_user.organization_id
    )
    
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    
    return {
        "session_id": session.id,
        "name": session.name,
        "messages": session.messages,
        "configuration": session.configuration,
        "created_at": session.created_at,
        "updated_at": session.updated_at
    }


# Document Management Endpoints
@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(require_agent_evaluation),
    db: AsyncSession = Depends(get_database)
) -> dict:
    """
    Upload and process a document for RAG.
    
    Supported formats: PDF, TXT, DOCX
    Maximum file size: 100MB
    """
    try:
        # Validate file type
        allowed_types = {".pdf", ".txt", ".docx"}
        file_extension = f".{file.filename.split('.')[-1].lower()}"
        
        if file_extension not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_types)}"
            )
        
        # Read file content
        content = await file.read()
        
        # Process document
        document = await doc_service.process_document(
            db,
            file_name=file.filename,
            file_content=content,
            file_type=file_extension,
            organization_id=current_user.organization_id,
            uploaded_by=current_user.id
        )
        
        logger.info(
            "Document uploaded and processed",
            user_id=current_user.id,
            document_id=document.id,
            filename=file.filename
        )
        
        return {
            "document_id": document.id,
            "filename": document.name,
            "status": "processed",
            "chunks_created": document.processing_metadata.get("chunks_created", 0),
            "message": "Document processed successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Document upload failed", error=str(e), user_id=current_user.id)
        raise HTTPException(status_code=500, detail="Failed to process document")


@router.get("/documents")
async def list_documents(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    current_user: User = Depends(require_agent_evaluation),
    db: AsyncSession = Depends(get_database)
) -> List[dict]:
    """
    List uploaded documents.
    """
    documents = await doc_service.list_documents(
        db,
        organization_id=current_user.organization_id,
        skip=skip,
        limit=limit
    )
    
    return [
        {
            "id": doc.id,
            "name": doc.name,
            "original_filename": doc.original_filename,
            "file_type": doc.file_type,
            "file_size": doc.file_size,
            "processed": doc.processed,
            "created_at": doc.created_at,
            "processing_metadata": doc.processing_metadata
        }
        for doc in documents
    ]


@router.post("/documents/search")
async def search_documents(
    query: dict,
    current_user: User = Depends(require_agent_evaluation),
    db: AsyncSession = Depends(get_database)
) -> dict:
    """
    Search documents using semantic similarity.
    
    - **query**: Search query string
    - **k**: Number of results to return (default: 5)
    - **min_score**: Minimum similarity score (default: 0.0)
    """
    try:
        search_query = query.get("query", "")
        k = query.get("k", 5)
        min_score = query.get("min_score", 0.0)
        
        if not search_query:
            raise HTTPException(status_code=400, detail="Search query is required")
        
        results = await doc_service.search_documents(
            db,
            query=search_query,
            organization_id=current_user.organization_id,
            k=k,
            min_score=min_score
        )
        
        logger.info(
            "Document search performed",
            user_id=current_user.id,
            query=search_query,
            results_count=len(results)
        )
        
        return {
            "query": search_query,
            "results": results,
            "total": len(results)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Document search failed", error=str(e), user_id=current_user.id)
        raise HTTPException(status_code=500, detail="Document search failed")


# Evaluation Endpoints
@router.post("/evaluations", response_model=AgentEvaluationResponse)
async def create_evaluation(
    evaluation_data: AgentEvaluationCreate,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_agent_evaluation),
    db: AsyncSession = Depends(get_database)
) -> AgentEvaluationResponse:
    """
    Create and start a new agent evaluation.
    
    - **name**: Evaluation name
    - **description**: Optional description
    - **configuration_id**: Agent configuration to evaluate
    - **dataset_id**: Dataset to use for evaluation (optional)
    - **evaluation_config**: Evaluation configuration and parameters
    """
    try:
        # Create evaluation record
        evaluation = await eval_service.create_evaluation(
            db,
            evaluation_data=evaluation_data,
            organization_id=current_user.organization_id,
            created_by=current_user.id
        )
        
        # Start evaluation in background
        background_tasks.add_task(
            eval_service.run_evaluation_background,
            str(evaluation.id),
            current_user.organization_id
        )
        
        logger.info(
            "Agent evaluation created and started",
            user_id=current_user.id,
            evaluation_id=evaluation.id,
            config_id=evaluation_data.configuration_id
        )
        
        return AgentEvaluationResponse.from_orm(evaluation)
        
    except Exception as e:
        logger.error("Failed to create evaluation", error=str(e), user_id=current_user.id)
        raise HTTPException(status_code=500, detail="Failed to create evaluation")


@router.get("/evaluations", response_model=List[AgentEvaluationResponse])
async def list_evaluations(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    status: Optional[str] = Query(None),
    current_user: User = Depends(require_agent_evaluation),
    db: AsyncSession = Depends(get_database)
) -> List[AgentEvaluationResponse]:
    """
    List agent evaluations for the organization.
    
    - **skip**: Number of evaluations to skip
    - **limit**: Maximum number to return
    - **status**: Filter by status (pending, running, completed, failed)
    """
    evaluations = await eval_service.list_evaluations(
        db,
        organization_id=current_user.organization_id,
        skip=skip,
        limit=limit,
        status=status
    )
    
    return [AgentEvaluationResponse.from_orm(eval) for eval in evaluations]


@router.get("/evaluations/{evaluation_id}", response_model=AgentEvaluationResponse)
async def get_evaluation(
    evaluation_id: UUID,
    current_user: User = Depends(require_agent_evaluation),
    db: AsyncSession = Depends(get_database)
) -> AgentEvaluationResponse:
    """
    Get a specific evaluation by ID.
    """
    evaluation = await eval_service.get_evaluation(
        db,
        evaluation_id=evaluation_id,
        organization_id=current_user.organization_id
    )
    
    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    
    return AgentEvaluationResponse.from_orm(evaluation)


@router.get("/evaluations/{evaluation_id}/results")
async def get_evaluation_results(
    evaluation_id: UUID,
    current_user: User = Depends(require_agent_evaluation),
    db: AsyncSession = Depends(get_database)
) -> dict:
    """
    Get detailed results for a specific evaluation.
    """
    results = await eval_service.get_evaluation_results(
        db,
        evaluation_id=evaluation_id,
        organization_id=current_user.organization_id
    )
    
    if not results:
        raise HTTPException(status_code=404, detail="Evaluation results not found")
    
    return results


@router.post("/evaluations/compare")
async def compare_evaluations(
    evaluation_ids: List[UUID],
    current_user: User = Depends(require_agent_evaluation),
    db: AsyncSession = Depends(get_database)
) -> dict:
    """
    Compare multiple evaluations and generate comparison metrics.
    
    - **evaluation_ids**: List of evaluation IDs to compare (2-10 evaluations)
    """
    if len(evaluation_ids) < 2:
        raise HTTPException(status_code=400, detail="At least 2 evaluations required for comparison")
    
    if len(evaluation_ids) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 evaluations allowed for comparison")
    
    try:
        comparison = await eval_service.compare_evaluations(
            db,
            evaluation_ids=evaluation_ids,
            organization_id=current_user.organization_id
        )
        
        logger.info(
            "Evaluations compared",
            user_id=current_user.id,
            evaluation_ids=[str(id) for id in evaluation_ids]
        )
        
        return comparison
        
    except Exception as e:
        logger.error("Evaluation comparison failed", error=str(e), user_id=current_user.id)
        raise HTTPException(status_code=500, detail="Failed to compare evaluations")


# Analytics Endpoints
@router.get("/analytics/summary")
async def get_analytics_summary(
    days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(require_agent_evaluation),
    db: AsyncSession = Depends(get_database)
) -> dict:
    """
    Get analytics summary for agent evaluations.
    
    - **days**: Number of days to include in summary (1-365)
    """
    try:
        summary = await eval_service.get_analytics_summary(
            db,
            organization_id=current_user.organization_id,
            days=days
        )
        
        return summary
        
    except Exception as e:
        logger.error("Failed to get analytics summary", error=str(e), user_id=current_user.id)
        raise HTTPException(status_code=500, detail="Failed to get analytics summary")


@router.get("/analytics/usage")
async def get_usage_analytics(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    granularity: str = Query("day", regex="^(hour|day|week|month)$"),
    current_user: User = Depends(require_agent_evaluation),
    db: AsyncSession = Depends(get_database)
) -> dict:
    """
    Get detailed usage analytics.
    
    - **start_date**: Start date (ISO format, default: 30 days ago)
    - **end_date**: End date (ISO format, default: now)
    - **granularity**: Data granularity (hour, day, week, month)
    """
    try:
        analytics = await eval_service.get_usage_analytics(
            db,
            organization_id=current_user.organization_id,
            start_date=start_date,
            end_date=end_date,
            granularity=granularity
        )
        
        return analytics
        
    except Exception as e:
        logger.error("Failed to get usage analytics", error=str(e), user_id=current_user.id)
        raise HTTPException(status_code=500, detail="Failed to get usage analytics")