"""
RAIA Platform - Unified LLM Provider Interface
Supports OpenAI, Anthropic, and local models with async patterns and enterprise features
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from dataclasses import dataclass
from enum import Enum

import structlog
import openai
import anthropic
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

from app.core.config import get_settings
from app.models.schemas import ModelProviderEnum

logger = structlog.get_logger(__name__)
settings = get_settings()


@dataclass
class LLMUsageStats:
    """Statistics about LLM usage for a single request"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    response_time_ms: int = 0
    model_name: str = ""
    provider: str = ""


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider"""
    content: str
    usage_stats: LLMUsageStats
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class LLMProvider(ABC):
    """Abstract base class for all LLM providers"""
    
    def __init__(self, provider_name: str, config: Dict[str, Any]):
        self.provider_name = provider_name
        self.config = config
        self.logger = logger.bind(provider=provider_name)
    
    @abstractmethod
    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> LLMResponse:
        """Generate a response from the LLM"""
        pass
    
    @abstractmethod
    async def stream_generate(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream generate a response from the LLM"""
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Get list of available models for this provider"""
        pass
    
    @abstractmethod
    async def validate_connection(self) -> bool:
        """Validate connection to the provider"""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI LLM Provider with enterprise features"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("openai", config)
        
        self.api_key = config.get("api_key") or settings.openai_api_key
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.organization = config.get("organization")
        self.base_url = config.get("base_url")
        
        # Initialize async client
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            organization=self.organization,
            base_url=self.base_url,
            timeout=config.get("timeout", 120.0),
            max_retries=config.get("max_retries", 3)
        )
        
        self.available_models = [
            "gpt-4o",
            "gpt-4o-mini", 
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k"
        ]
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> LLMResponse:
        """Generate response using OpenAI API"""
        try:
            start_time = time.time()
            
            # Prepare request parameters
            params = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                **kwargs
            }
            
            self.logger.debug(
                "Generating OpenAI response",
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                message_count=len(messages)
            )
            
            # Make API call
            response = await self.client.chat.completions.create(**params)
            
            end_time = time.time()
            response_time_ms = int((end_time - start_time) * 1000)
            
            # Extract content and usage
            content = response.choices[0].message.content or ""
            usage = response.usage
            
            usage_stats = LLMUsageStats(
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
                response_time_ms=response_time_ms,
                model_name=model,
                provider="openai"
            )
            
            self.logger.info(
                "OpenAI response generated successfully",
                model=model,
                response_time_ms=response_time_ms,
                total_tokens=usage_stats.total_tokens
            )
            
            return LLMResponse(
                content=content,
                usage_stats=usage_stats,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "response_id": response.id,
                    "created": response.created
                }
            )
            
        except openai.RateLimitError as e:
            self.logger.warning("OpenAI rate limit exceeded", error=str(e))
            raise
        except openai.AuthenticationError as e:
            self.logger.error("OpenAI authentication failed", error=str(e))
            raise
        except Exception as e:
            self.logger.error("OpenAI API error", error=str(e))
            raise
    
    async def stream_generate(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream generate response using OpenAI API"""
        try:
            params = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
                **kwargs
            }
            
            self.logger.debug("Starting OpenAI stream generation", model=model)
            
            async for chunk in await self.client.chat.completions.create(**params):
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            self.logger.error("OpenAI streaming error", error=str(e))
            raise
    
    def get_available_models(self) -> List[str]:
        """Get available OpenAI models"""
        return self.available_models.copy()
    
    async def validate_connection(self) -> bool:
        """Validate OpenAI connection"""
        try:
            models = await self.client.models.list()
            return len(models.data) > 0
        except Exception as e:
            self.logger.error("OpenAI connection validation failed", error=str(e))
            return False


class AnthropicProvider(LLMProvider):
    """Anthropic Claude LLM Provider"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("anthropic", config)
        
        self.api_key = config.get("api_key") or settings.anthropic_api_key
        if not self.api_key:
            raise ValueError("Anthropic API key is required")
        
        # Initialize async client
        self.client = AsyncAnthropic(
            api_key=self.api_key,
            timeout=config.get("timeout", 120.0),
            max_retries=config.get("max_retries", 3)
        )
        
        self.available_models = [
            "claude-3-5-sonnet-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307"
        ]
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> LLMResponse:
        """Generate response using Anthropic API"""
        try:
            start_time = time.time()
            
            # Convert messages to Anthropic format
            system_message = None
            anthropic_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    anthropic_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            # Prepare request parameters
            params = {
                "model": model,
                "messages": anthropic_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            
            if system_message:
                params["system"] = system_message
                
            params.update(kwargs)
            
            self.logger.debug(
                "Generating Anthropic response",
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                message_count=len(anthropic_messages)
            )
            
            # Make API call
            response = await self.client.messages.create(**params)
            
            end_time = time.time()
            response_time_ms = int((end_time - start_time) * 1000)
            
            # Extract content and usage
            content = ""
            if response.content:
                content = "\n".join([block.text for block in response.content if hasattr(block, 'text')])
            
            usage = response.usage
            usage_stats = LLMUsageStats(
                prompt_tokens=usage.input_tokens if usage else 0,
                completion_tokens=usage.output_tokens if usage else 0,
                total_tokens=(usage.input_tokens + usage.output_tokens) if usage else 0,
                response_time_ms=response_time_ms,
                model_name=model,
                provider="anthropic"
            )
            
            self.logger.info(
                "Anthropic response generated successfully",
                model=model,
                response_time_ms=response_time_ms,
                total_tokens=usage_stats.total_tokens
            )
            
            return LLMResponse(
                content=content,
                usage_stats=usage_stats,
                metadata={
                    "stop_reason": response.stop_reason,
                    "response_id": response.id,
                    "type": response.type
                }
            )
            
        except anthropic.RateLimitError as e:
            self.logger.warning("Anthropic rate limit exceeded", error=str(e))
            raise
        except anthropic.AuthenticationError as e:
            self.logger.error("Anthropic authentication failed", error=str(e))
            raise
        except Exception as e:
            self.logger.error("Anthropic API error", error=str(e))
            raise
    
    async def stream_generate(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream generate response using Anthropic API"""
        try:
            # Convert messages to Anthropic format
            system_message = None
            anthropic_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    anthropic_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            params = {
                "model": model,
                "messages": anthropic_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
            }
            
            if system_message:
                params["system"] = system_message
                
            params.update(kwargs)
            
            self.logger.debug("Starting Anthropic stream generation", model=model)
            
            async with self.client.messages.stream(**params) as stream:
                async for chunk in stream:
                    if chunk.type == "content_block_delta" and hasattr(chunk.delta, 'text'):
                        yield chunk.delta.text
                        
        except Exception as e:
            self.logger.error("Anthropic streaming error", error=str(e))
            raise
    
    def get_available_models(self) -> List[str]:
        """Get available Anthropic models"""
        return self.available_models.copy()
    
    async def validate_connection(self) -> bool:
        """Validate Anthropic connection"""
        try:
            # Simple validation request
            response = await self.client.messages.create(
                model=self.available_models[0],
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=1
            )
            return response is not None
        except Exception as e:
            self.logger.error("Anthropic connection validation failed", error=str(e))
            return False


class LocalModelProvider(LLMProvider):
    """Local model provider for self-hosted models"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("local", config)
        
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.timeout = config.get("timeout", 120.0)
        self.available_models = config.get("models", ["llama2", "codellama", "mistral"])
        
    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> LLMResponse:
        """Generate response using local model (Ollama format)"""
        try:
            import aiohttp
            
            start_time = time.time()
            
            # Convert messages to prompt
            prompt = self._messages_to_prompt(messages)
            
            # Prepare request
            payload = {
                "model": model,
                "prompt": prompt,
                "temperature": temperature,
                "num_predict": max_tokens,
                "stream": False
            }
            
            self.logger.debug(
                "Generating local model response",
                model=model,
                base_url=self.base_url,
                temperature=temperature
            )
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(self.timeout)) as session:
                async with session.post(f"{self.base_url}/api/generate", json=payload) as response:
                    if response.status != 200:
                        raise Exception(f"Local model API error: {response.status}")
                    
                    result = await response.json()
                    
                    end_time = time.time()
                    response_time_ms = int((end_time - start_time) * 1000)
                    
                    content = result.get("response", "")
                    
                    # Estimate token usage (approximate)
                    prompt_tokens = len(prompt.split())
                    completion_tokens = len(content.split())
                    
                    usage_stats = LLMUsageStats(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens,
                        response_time_ms=response_time_ms,
                        model_name=model,
                        provider="local"
                    )
                    
                    self.logger.info(
                        "Local model response generated successfully",
                        model=model,
                        response_time_ms=response_time_ms,
                        estimated_tokens=usage_stats.total_tokens
                    )
                    
                    return LLMResponse(
                        content=content,
                        usage_stats=usage_stats,
                        metadata={
                            "done": result.get("done", False),
                            "context": result.get("context", [])
                        }
                    )
                    
        except Exception as e:
            self.logger.error("Local model API error", error=str(e))
            raise
    
    async def stream_generate(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream generate response using local model"""
        try:
            import aiohttp
            import json
            
            # Convert messages to prompt
            prompt = self._messages_to_prompt(messages)
            
            payload = {
                "model": model,
                "prompt": prompt,
                "temperature": temperature,
                "num_predict": max_tokens,
                "stream": True
            }
            
            self.logger.debug("Starting local model stream generation", model=model)
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(self.timeout)) as session:
                async with session.post(f"{self.base_url}/api/generate", json=payload) as response:
                    if response.status != 200:
                        raise Exception(f"Local model API error: {response.status}")
                    
                    async for line in response.content:
                        if line:
                            try:
                                chunk = json.loads(line.decode('utf-8'))
                                if chunk.get("response"):
                                    yield chunk["response"]
                            except json.JSONDecodeError:
                                continue
                                
        except Exception as e:
            self.logger.error("Local model streaming error", error=str(e))
            raise
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages to a simple prompt format"""
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        return "\n".join(prompt_parts) + "\nAssistant: "
    
    def get_available_models(self) -> List[str]:
        """Get available local models"""
        return self.available_models.copy()
    
    async def validate_connection(self) -> bool:
        """Validate local model connection"""
        try:
            import aiohttp
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(10.0)) as session:
                async with session.get(f"{self.base_url}/api/tags") as response:
                    return response.status == 200
                    
        except Exception as e:
            self.logger.error("Local model connection validation failed", error=str(e))
            return False


class LLMProviderManager:
    """Manager for all LLM providers with caching and error handling"""
    
    def __init__(self):
        self.providers: Dict[str, LLMProvider] = {}
        self.logger = logger.bind(component="llm_provider_manager")
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize all configured providers"""
        try:
            # OpenAI Provider
            if settings.openai_api_key:
                openai_config = {
                    "api_key": settings.openai_api_key,
                    "timeout": 120.0,
                    "max_retries": 3
                }
                self.providers["openai"] = OpenAIProvider(openai_config)
                self.logger.info("OpenAI provider initialized")
            
            # Anthropic Provider
            if settings.anthropic_api_key:
                anthropic_config = {
                    "api_key": settings.anthropic_api_key,
                    "timeout": 120.0,
                    "max_retries": 3
                }
                self.providers["anthropic"] = AnthropicProvider(anthropic_config)
                self.logger.info("Anthropic provider initialized")
            
            # Local Provider (always available if configured)
            if hasattr(settings, 'local_model_base_url'):
                local_config = {
                    "base_url": settings.local_model_base_url,
                    "timeout": 120.0,
                    "models": getattr(settings, 'local_models', ["llama2", "mistral"])
                }
                self.providers["local"] = LocalModelProvider(local_config)
                self.logger.info("Local model provider initialized")
                
        except Exception as e:
            self.logger.error("Failed to initialize LLM providers", error=str(e))
    
    def get_provider(self, provider_name: str) -> Optional[LLMProvider]:
        """Get provider by name"""
        return self.providers.get(provider_name)
    
    def get_provider_for_model(self, model_name: str) -> Optional[LLMProvider]:
        """Get appropriate provider for a model name"""
        # OpenAI models
        if model_name.startswith("gpt-"):
            return self.get_provider("openai")
        
        # Anthropic models
        if model_name.startswith("claude-"):
            return self.get_provider("anthropic")
        
        # Check if available in any provider
        for provider in self.providers.values():
            if model_name in provider.get_available_models():
                return provider
        
        return None
    
    async def generate(
        self,
        provider_name: str,
        messages: List[Dict[str, str]],
        model: str,
        **kwargs
    ) -> LLMResponse:
        """Generate response using specified provider"""
        provider = self.get_provider(provider_name)
        if not provider:
            raise ValueError(f"Provider '{provider_name}' not found or not initialized")
        
        return await provider.generate(messages, model, **kwargs)
    
    async def auto_generate(
        self,
        model: str,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> LLMResponse:
        """Automatically select provider and generate response"""
        provider = self.get_provider_for_model(model)
        if not provider:
            raise ValueError(f"No provider found for model '{model}'")
        
        return await provider.generate(messages, model, **kwargs)
    
    def get_all_available_models(self) -> Dict[str, List[str]]:
        """Get all available models from all providers"""
        all_models = {}
        for name, provider in self.providers.items():
            all_models[name] = provider.get_available_models()
        return all_models
    
    async def validate_all_connections(self) -> Dict[str, bool]:
        """Validate connections to all providers"""
        results = {}
        for name, provider in self.providers.items():
            try:
                results[name] = await provider.validate_connection()
            except Exception as e:
                self.logger.error(f"Connection validation failed for {name}", error=str(e))
                results[name] = False
        return results


# Global provider manager instance
_provider_manager: Optional[LLMProviderManager] = None


def get_llm_provider_manager() -> LLMProviderManager:
    """Get or create the global LLM provider manager"""
    global _provider_manager
    if _provider_manager is None:
        _provider_manager = LLMProviderManager()
    return _provider_manager


async def generate_llm_response(
    model: str,
    messages: List[Dict[str, str]],
    provider: Optional[str] = None,
    **kwargs
) -> LLMResponse:
    """
    Convenience function to generate LLM response
    
    Args:
        model: Model name to use
        messages: List of messages in OpenAI format
        provider: Optional provider name, auto-detected if None
        **kwargs: Additional parameters for the LLM
    
    Returns:
        LLMResponse with content and usage statistics
    """
    manager = get_llm_provider_manager()
    
    if provider:
        return await manager.generate(provider, messages, model, **kwargs)
    else:
        return await manager.auto_generate(model, messages, **kwargs)