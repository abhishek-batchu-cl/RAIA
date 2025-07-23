"""
RAIA Platform - RAG Agent Implementation
Advanced Retrieval-Augmented Generation agent with multi-model support
"""

import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import structlog

from app.core.llm_providers import get_llm_provider_manager, LLMResponse
from app.services.document_service import get_document_service
from app.models.schemas import (
    AgentConfiguration, 
    AgentChatMessage, 
    AgentChatResponse, 
    DocumentSearchResult
)

logger = structlog.get_logger(__name__)


class RAGAgent:
    """
    Advanced RAG Agent with enterprise features:
    - Multi-model LLM support (OpenAI, Anthropic, local models)
    - Sophisticated retrieval strategies
    - Context management and optimization
    - Usage tracking and analytics
    - Error handling and fallbacks
    """
    
    def __init__(self, config: AgentConfiguration, organization_id: str):
        self.config = config
        self.organization_id = organization_id
        self.logger = logger.bind(
            agent_id=str(config.id),
            agent_name=config.name,
            model=config.model_name
        )
        
        # Initialize services
        self.llm_manager = get_llm_provider_manager()
        self.document_service = get_document_service()
        
        self.logger.info("RAG Agent initialized", model_provider=config.model_provider.value)
    
    async def generate_response(
        self,
        query: str,
        chat_history: List[Dict[str, Any]] = None,
        include_sources: bool = True,
        session_id: Optional[str] = None
    ) -> AgentChatResponse:
        """
        Generate a response using the RAG pipeline
        
        Args:
            query: User's question or input
            chat_history: Previous messages for context
            include_sources: Whether to include source information
            session_id: Optional session ID for tracking
            
        Returns:
            AgentChatResponse with answer, sources, and metadata
        """
        start_time = time.time()
        session_id = session_id or str(uuid.uuid4())
        
        try:
            self.logger.info("Generating RAG response", query=query[:100], session_id=session_id)
            
            # Step 1: Retrieve relevant context
            retrieval_start = time.time()
            context_results = await self._retrieve_context(query)
            retrieval_time = int((time.time() - retrieval_start) * 1000)
            
            self.logger.debug(
                "Context retrieval completed",
                results_count=len(context_results),
                retrieval_time_ms=retrieval_time
            )
            
            # Step 2: Prepare messages with context
            messages = await self._prepare_messages(query, context_results, chat_history)
            
            # Step 3: Generate LLM response
            llm_response = await self._generate_llm_response(messages)
            
            # Step 4: Prepare response
            total_time = int((time.time() - start_time) * 1000)
            
            # Create chat message
            chat_message = AgentChatMessage(
                role="assistant",
                content=llm_response.content,
                timestamp=datetime.utcnow(),
                metadata={
                    "model_name": self.config.model_name,
                    "provider": self.config.model_provider.value,
                    "retrieval_time_ms": retrieval_time,
                    "sources_count": len(context_results)
                },
                sources=[result.dict() for result in context_results] if include_sources else [],
                tokens_used=llm_response.usage_stats.total_tokens,
                response_time_ms=total_time
            )
            
            # Prepare sources for response
            sources = []
            if include_sources:
                sources = [
                    {
                        "document": result.source_document,
                        "chunk": result.document_chunk[:200] + "..." if len(result.document_chunk) > 200 else result.document_chunk,
                        "similarity_score": result.similarity_score,
                        "chunk_index": result.chunk_index,
                        "metadata": result.metadata
                    }
                    for result in context_results
                ]
            
            response = AgentChatResponse(
                answer=llm_response.content,
                session_id=uuid.UUID(session_id),
                agent_configuration_id=self.config.id,
                context=self._format_context(context_results) if context_results else None,
                sources=sources,
                tokens_used=llm_response.usage_stats.total_tokens,
                response_time_ms=total_time,
                model_name=self.config.model_name,
                message=chat_message
            )
            
            self.logger.info(
                "RAG response generated successfully",
                response_time_ms=total_time,
                tokens_used=llm_response.usage_stats.total_tokens,
                sources_count=len(context_results)
            )
            
            return response
            
        except Exception as e:
            self.logger.error("Failed to generate RAG response", error=str(e))
            
            # Return fallback response
            error_time = int((time.time() - start_time) * 1000)
            
            error_message = AgentChatMessage(
                role="assistant",
                content="I'm sorry, I encountered an error while processing your request. Please try again or rephrase your question.",
                timestamp=datetime.utcnow(),
                metadata={
                    "error": str(e),
                    "model_name": self.config.model_name,
                    "provider": self.config.model_provider.value
                },
                response_time_ms=error_time
            )
            
            return AgentChatResponse(
                answer=error_message.content,
                session_id=uuid.UUID(session_id),
                agent_configuration_id=self.config.id,
                context=None,
                sources=[],
                tokens_used=0,
                response_time_ms=error_time,
                model_name=self.config.model_name,
                message=error_message
            )
    
    async def _retrieve_context(self, query: str) -> List[DocumentSearchResult]:
        """Retrieve relevant context using configured strategy"""
        try:
            if self.config.retrieval_strategy == "similarity":
                return await self._similarity_retrieval(query)
            elif self.config.retrieval_strategy == "hybrid":
                return await self._hybrid_retrieval(query)
            else:
                # Default to similarity
                return await self._similarity_retrieval(query)
                
        except Exception as e:
            self.logger.error("Context retrieval failed", error=str(e))
            return []
    
    async def _similarity_retrieval(self, query: str) -> List[DocumentSearchResult]:
        """Basic similarity-based retrieval"""
        search_response = await self.document_service.search_documents(
            query=query,
            organization_id=self.organization_id,
            n_results=self.config.retrieval_k,
            embedding_model=self.config.embedding_model
        )
        
        return search_response.results
    
    async def _hybrid_retrieval(self, query: str) -> List[DocumentSearchResult]:
        """Hybrid retrieval combining multiple strategies"""
        # For now, implement as enhanced similarity retrieval
        # In the future, this could combine:
        # - Semantic similarity
        # - Keyword matching
        # - Recency weighting
        # - User preference learning
        
        search_response = await self.document_service.search_documents(
            query=query,
            organization_id=self.organization_id,
            n_results=min(self.config.retrieval_k * 2, 20),  # Get more candidates
            embedding_model=self.config.embedding_model
        )
        
        # Apply additional filtering/ranking
        results = search_response.results
        
        # Filter by minimum similarity threshold
        min_similarity = 0.3
        filtered_results = [r for r in results if r.similarity_score >= min_similarity]
        
        # Return top K results
        return filtered_results[:self.config.retrieval_k]
    
    async def _prepare_messages(
        self,
        query: str,
        context_results: List[DocumentSearchResult],
        chat_history: List[Dict[str, Any]] = None
    ) -> List[Dict[str, str]]:
        """Prepare messages for LLM with context and history"""
        messages = []
        
        # Add system prompt
        messages.append({
            "role": "system",
            "content": self.config.system_prompt
        })
        
        # Add chat history if provided
        if chat_history:
            for msg in chat_history[-10:]:  # Limit history to last 10 messages
                if msg.get("role") in ["user", "assistant"]:
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
        
        # Prepare user message with context
        if context_results:
            context_text = self._format_context(context_results)
            user_content = f"""Context information:
{context_text}

Question: {query}

Please answer the question based on the provided context. If the context doesn't contain relevant information to fully answer the question, please say so and provide what information you can."""
        else:
            user_content = f"""Question: {query}

Please answer this question. Note that I don't have specific context documents available, so provide a helpful general response."""
        
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        return messages
    
    def _format_context(self, context_results: List[DocumentSearchResult]) -> str:
        """Format context results into a readable string"""
        if not context_results:
            return ""
        
        context_parts = []
        for i, result in enumerate(context_results):
            context_parts.append(
                f"[Source {i+1} - {result.source_document}]:\n{result.document_chunk}\n"
            )
        
        return "\n".join(context_parts)
    
    async def _generate_llm_response(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """Generate response using configured LLM"""
        try:
            return await self.llm_manager.auto_generate(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                frequency_penalty=self.config.frequency_penalty,
                presence_penalty=self.config.presence_penalty
            )
            
        except Exception as e:
            self.logger.error("LLM generation failed", error=str(e))
            
            # Try with fallback parameters if the original request failed
            try:
                self.logger.info("Retrying with fallback parameters")
                return await self.llm_manager.auto_generate(
                    model=self.config.model_name,
                    messages=messages,
                    temperature=0.7,  # Safe default
                    max_tokens=1000   # Conservative limit
                )
            except Exception as fallback_error:
                self.logger.error("Fallback LLM generation also failed", error=str(fallback_error))
                raise fallback_error
    
    def get_configuration_info(self) -> Dict[str, Any]:
        """Get current agent configuration information"""
        return {
            "id": str(self.config.id),
            "name": self.config.name,
            "description": self.config.description,
            "version": self.config.version,
            "model_name": self.config.model_name,
            "model_provider": self.config.model_provider.value,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "retrieval_strategy": self.config.retrieval_strategy,
            "retrieval_k": self.config.retrieval_k,
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "embedding_model": self.config.embedding_model,
            "status": self.config.status.value,
            "created_at": self.config.created_at.isoformat() if self.config.created_at else None,
            "updated_at": self.config.updated_at.isoformat() if self.config.updated_at else None
        }
    
    async def validate_configuration(self) -> Dict[str, Any]:
        """Validate the agent configuration"""
        validation_results = {
            "is_valid": True,
            "issues": [],
            "provider_status": "unknown",
            "model_available": False
        }
        
        try:
            # Check if provider is available
            provider = self.llm_manager.get_provider_for_model(self.config.model_name)
            if not provider:
                validation_results["is_valid"] = False
                validation_results["issues"].append(f"No provider available for model: {self.config.model_name}")
            else:
                # Validate provider connection
                is_connected = await provider.validate_connection()
                validation_results["provider_status"] = "connected" if is_connected else "disconnected"
                
                if not is_connected:
                    validation_results["is_valid"] = False
                    validation_results["issues"].append(f"Provider {provider.provider_name} is not accessible")
                
                # Check if model is available
                available_models = provider.get_available_models()
                validation_results["model_available"] = self.config.model_name in available_models
                
                if not validation_results["model_available"]:
                    validation_results["is_valid"] = False
                    validation_results["issues"].append(f"Model {self.config.model_name} not available in provider")
            
            # Validate configuration parameters
            if self.config.temperature < 0 or self.config.temperature > 2:
                validation_results["is_valid"] = False
                validation_results["issues"].append("Temperature must be between 0 and 2")
            
            if self.config.max_tokens <= 0:
                validation_results["is_valid"] = False
                validation_results["issues"].append("Max tokens must be greater than 0")
            
            if self.config.retrieval_k <= 0:
                validation_results["is_valid"] = False
                validation_results["issues"].append("Retrieval K must be greater than 0")
            
            # Check system prompt
            if not self.config.system_prompt or not self.config.system_prompt.strip():
                validation_results["is_valid"] = False
                validation_results["issues"].append("System prompt is required")
            
            return validation_results
            
        except Exception as e:
            self.logger.error("Configuration validation failed", error=str(e))
            return {
                "is_valid": False,
                "issues": [f"Validation error: {str(e)}"],
                "provider_status": "error",
                "model_available": False
            }


class RAGAgentManager:
    """Manager for RAG agents with caching and lifecycle management"""
    
    def __init__(self):
        self.agents: Dict[str, RAGAgent] = {}
        self.logger = logger.bind(component="rag_agent_manager")
    
    async def get_agent(self, config: AgentConfiguration, organization_id: str) -> RAGAgent:
        """Get or create a RAG agent for the given configuration"""
        agent_key = f"{config.id}_{organization_id}"
        
        if agent_key not in self.agents:
            self.logger.info(f"Creating new RAG agent: {config.name}")
            self.agents[agent_key] = RAGAgent(config, organization_id)
        
        return self.agents[agent_key]
    
    def remove_agent(self, config_id: str, organization_id: str):
        """Remove agent from cache"""
        agent_key = f"{config_id}_{organization_id}"
        if agent_key in self.agents:
            del self.agents[agent_key]
            self.logger.info(f"Removed agent from cache: {agent_key}")
    
    def clear_cache(self):
        """Clear all cached agents"""
        self.agents.clear()
        self.logger.info("Cleared all cached agents")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "cached_agents": len(self.agents),
            "agent_keys": list(self.agents.keys())
        }


# Global agent manager instance
_agent_manager: Optional[RAGAgentManager] = None


def get_rag_agent_manager() -> RAGAgentManager:
    """Get or create the global RAG agent manager"""
    global _agent_manager
    if _agent_manager is None:
        _agent_manager = RAGAgentManager()
    return _agent_manager