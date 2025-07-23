"""
RAIA Platform - Configuration Management Service
Manages agent configurations with version control, templates, and deployment tracking
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import structlog
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_database
from app.models.schemas import (
    AgentConfiguration,
    AgentConfigurationCreate,
    AgentConfigurationUpdate,
    AgentConfigurationResponse,
    AgentConfigurationStatusEnum,
    ModelProviderEnum
)

logger = structlog.get_logger(__name__)


class ConfigurationTemplate:
    """Predefined configuration templates for common use cases"""
    
    @staticmethod
    def get_templates() -> Dict[str, Dict[str, Any]]:
        """Get all available configuration templates"""
        return {
            "customer_support": {
                "name": "Customer Support Agent",
                "description": "Optimized for customer support and FAQ responses",
                "system_prompt": """You are a helpful customer support agent. Your role is to:

1. Provide accurate and helpful information based on the available documentation
2. Be polite, professional, and empathetic
3. If you cannot find specific information, clearly state what you don't know
4. Guide users to appropriate resources or next steps when needed
5. Always maintain a helpful and solution-oriented tone

When answering questions:
- Use clear, simple language
- Provide step-by-step instructions when applicable
- Include relevant links or references from the documentation
- Ask clarifying questions if the user's request is unclear""",
                "model_name": "gpt-4o-mini",
                "model_provider": ModelProviderEnum.OPENAI,
                "temperature": 0.3,
                "max_tokens": 1500,
                "retrieval_k": 7,
                "retrieval_strategy": "similarity"
            },
            "technical_documentation": {
                "name": "Technical Documentation Assistant",
                "description": "Specialized for technical documentation and API references",
                "system_prompt": """You are a technical documentation assistant. Your expertise includes:

1. Explaining complex technical concepts clearly
2. Providing accurate code examples and API usage
3. Helping users understand documentation and implementation details
4. Troubleshooting technical issues based on available information

Guidelines:
- Use precise technical language when appropriate
- Provide code examples when relevant
- Reference specific documentation sections or API endpoints
- Explain both the "what" and "why" of technical solutions
- Suggest best practices and common patterns""",
                "model_name": "gpt-4o",
                "model_provider": ModelProviderEnum.OPENAI,
                "temperature": 0.2,
                "max_tokens": 2000,
                "retrieval_k": 10,
                "retrieval_strategy": "similarity"
            },
            "research_assistant": {
                "name": "Research Assistant",
                "description": "Designed for research tasks and information synthesis",
                "system_prompt": """You are a research assistant specialized in analyzing and synthesizing information. Your approach:

1. Thoroughly analyze provided sources and context
2. Synthesize information from multiple sources when available
3. Clearly distinguish between facts from sources and general knowledge
4. Provide comprehensive, well-structured responses
5. Identify potential gaps or limitations in available information

Response structure:
- Start with a direct answer when possible
- Provide supporting evidence and details
- Reference specific sources when applicable
- Highlight any uncertainties or conflicting information
- Suggest areas for further investigation if relevant""",
                "model_name": "claude-3-5-sonnet-20241022",
                "model_provider": ModelProviderEnum.ANTHROPIC,
                "temperature": 0.4,
                "max_tokens": 3000,
                "retrieval_k": 12,
                "retrieval_strategy": "hybrid"
            },
            "educational_tutor": {
                "name": "Educational Tutor",
                "description": "Optimized for educational content and student assistance",
                "system_prompt": """You are an educational tutor focused on helping students learn effectively. Your teaching approach:

1. Break down complex topics into understandable segments
2. Use examples and analogies to clarify difficult concepts
3. Encourage active learning through questions and exercises
4. Adapt explanations to different learning levels
5. Provide encouragement and positive reinforcement

Teaching methods:
- Start with foundational concepts before advanced topics
- Use the Socratic method when appropriate
- Provide multiple perspectives on topics
- Include practical applications and real-world examples
- Offer additional resources for deeper learning""",
                "model_name": "gpt-4o",
                "model_provider": ModelProviderEnum.OPENAI,
                "temperature": 0.6,
                "max_tokens": 2500,
                "retrieval_k": 8,
                "retrieval_strategy": "similarity"
            }
        }


class ConfigurationVersionManager:
    """Manages configuration versioning and history"""
    
    def __init__(self):
        self.logger = logger.bind(component="config_version_manager")
    
    def generate_version(self, existing_versions: List[str]) -> str:
        """Generate next version number"""
        if not existing_versions:
            return "1.0"
        
        try:
            # Parse semantic versions
            versions = []
            for v in existing_versions:
                try:
                    major, minor = map(int, v.split('.'))
                    versions.append((major, minor))
                except ValueError:
                    continue
            
            if not versions:
                return "1.0"
            
            # Get highest version
            latest = max(versions)
            
            # Increment minor version
            return f"{latest[0]}.{latest[1] + 1}"
            
        except Exception as e:
            self.logger.warning("Failed to generate version", error=str(e))
            return "1.0"
    
    async def create_version_snapshot(
        self,
        config: AgentConfiguration,
        db: AsyncSession,
        change_description: Optional[str] = None
    ) -> str:
        """Create a version snapshot of the configuration"""
        try:
            # Get existing versions for this config name
            existing_query = select(AgentConfiguration).where(
                AgentConfiguration.name == config.name,
                AgentConfiguration.organization_id == config.organization_id
            )
            
            result = await db.execute(existing_query)
            existing_configs = result.scalars().all()
            
            existing_versions = [c.version for c in existing_configs]
            new_version = self.generate_version(existing_versions)
            
            # Create new version
            config.version = new_version
            if change_description:
                if not config.metadata:
                    config.metadata = {}
                config.metadata['version_notes'] = change_description
            
            return new_version
            
        except Exception as e:
            self.logger.error("Failed to create version snapshot", error=str(e))
            raise


class ConfigurationValidator:
    """Validates configuration parameters and constraints"""
    
    def __init__(self):
        self.logger = logger.bind(component="config_validator")
    
    def validate_model_parameters(self, config: AgentConfigurationCreate) -> List[str]:
        """Validate model parameters"""
        issues = []
        
        # Temperature validation
        if not (0.0 <= config.temperature <= 2.0):
            issues.append("Temperature must be between 0.0 and 2.0")
        
        # Max tokens validation
        if config.max_tokens <= 0:
            issues.append("Max tokens must be greater than 0")
        elif config.max_tokens > 100000:
            issues.append("Max tokens should not exceed 100,000 for performance reasons")
        
        # Top-p validation
        if not (0.0 <= config.top_p <= 1.0):
            issues.append("Top-p must be between 0.0 and 1.0")
        
        # Frequency penalty validation
        if not (-2.0 <= config.frequency_penalty <= 2.0):
            issues.append("Frequency penalty must be between -2.0 and 2.0")
        
        # Presence penalty validation
        if not (-2.0 <= config.presence_penalty <= 2.0):
            issues.append("Presence penalty must be between -2.0 and 2.0")
        
        return issues
    
    def validate_retrieval_parameters(self, config: AgentConfigurationCreate) -> List[str]:
        """Validate retrieval parameters"""
        issues = []
        
        # Retrieval K validation
        if config.retrieval_k <= 0:
            issues.append("Retrieval K must be greater than 0")
        elif config.retrieval_k > 50:
            issues.append("Retrieval K should not exceed 50 for performance reasons")
        
        # Chunk size validation
        if config.chunk_size <= 0:
            issues.append("Chunk size must be greater than 0")
        elif config.chunk_size > 10000:
            issues.append("Chunk size should not exceed 10,000 characters")
        
        # Chunk overlap validation
        if config.chunk_overlap < 0:
            issues.append("Chunk overlap cannot be negative")
        elif config.chunk_overlap >= config.chunk_size:
            issues.append("Chunk overlap must be less than chunk size")
        
        return issues
    
    def validate_prompts(self, config: AgentConfigurationCreate) -> List[str]:
        """Validate prompt content"""
        issues = []
        
        # System prompt validation
        if not config.system_prompt or not config.system_prompt.strip():
            issues.append("System prompt is required and cannot be empty")
        elif len(config.system_prompt) > 10000:
            issues.append("System prompt should not exceed 10,000 characters")
        
        # Check for potential prompt injection patterns
        suspicious_patterns = [
            "ignore previous instructions",
            "disregard the above",
            "forget everything",
            "new instructions:",
            "override system prompt"
        ]
        
        prompt_lower = config.system_prompt.lower()
        for pattern in suspicious_patterns:
            if pattern in prompt_lower:
                issues.append(f"System prompt contains potentially problematic content: '{pattern}'")
        
        return issues
    
    def validate_full_configuration(self, config: AgentConfigurationCreate) -> Dict[str, List[str]]:
        """Perform full configuration validation"""
        validation_results = {
            "model_parameters": self.validate_model_parameters(config),
            "retrieval_parameters": self.validate_retrieval_parameters(config),
            "prompts": self.validate_prompts(config),
            "general": []
        }
        
        # Name validation
        if not config.name or not config.name.strip():
            validation_results["general"].append("Configuration name is required")
        elif len(config.name) > 255:
            validation_results["general"].append("Configuration name must be 255 characters or less")
        
        return validation_results


class ConfigurationDeploymentManager:
    """Manages configuration deployments and status tracking"""
    
    def __init__(self):
        self.logger = logger.bind(component="config_deployment_manager")
    
    async def deploy_configuration(
        self,
        config_id: str,
        organization_id: str,
        db: AsyncSession
    ) -> bool:
        """Deploy a configuration (mark as active)"""
        try:
            query = select(AgentConfiguration).where(
                AgentConfiguration.id == uuid.UUID(config_id),
                AgentConfiguration.organization_id == uuid.UUID(organization_id)
            )
            
            result = await db.execute(query)
            config = result.scalar_one_or_none()
            
            if not config:
                return False
            
            config.status = AgentConfigurationStatusEnum.ACTIVE
            await db.commit()
            
            self.logger.info(f"Deployed configuration: {config.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to deploy configuration: {config_id}", error=str(e))
            return False
    
    async def retire_configuration(
        self,
        config_id: str,
        organization_id: str,
        db: AsyncSession
    ) -> bool:
        """Retire a configuration (mark as archived)"""
        try:
            query = select(AgentConfiguration).where(
                AgentConfiguration.id == uuid.UUID(config_id),
                AgentConfiguration.organization_id == uuid.UUID(organization_id)
            )
            
            result = await db.execute(query)
            config = result.scalar_one_or_none()
            
            if not config:
                return False
            
            config.status = AgentConfigurationStatusEnum.ARCHIVED
            await db.commit()
            
            self.logger.info(f"Retired configuration: {config.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to retire configuration: {config_id}", error=str(e))
            return False


class ConfigurationService:
    """Main configuration management service"""
    
    def __init__(self):
        self.logger = logger.bind(component="configuration_service")
        self.validator = ConfigurationValidator()
        self.version_manager = ConfigurationVersionManager()
        self.deployment_manager = ConfigurationDeploymentManager()
        self.templates = ConfigurationTemplate()
    
    async def create_configuration(
        self,
        config_data: AgentConfigurationCreate,
        organization_id: str,
        user_id: str,
        validate: bool = True
    ) -> Tuple[AgentConfigurationResponse, Dict[str, List[str]]]:
        """
        Create a new agent configuration with validation
        
        Returns:
            Tuple of (created_config, validation_issues)
        """
        validation_issues = {}
        
        try:
            # Validate configuration if requested
            if validate:
                validation_issues = self.validator.validate_full_configuration(config_data)
                
                # Check if there are any critical issues
                all_issues = []
                for issue_list in validation_issues.values():
                    all_issues.extend(issue_list)
                
                if all_issues:
                    self.logger.warning(
                        "Configuration validation issues found",
                        config_name=config_data.name,
                        issues_count=len(all_issues)
                    )
            
            async with get_database() as db:
                # Check for name conflicts
                existing_query = select(AgentConfiguration).where(
                    and_(
                        AgentConfiguration.name == config_data.name,
                        AgentConfiguration.organization_id == uuid.UUID(organization_id),
                        AgentConfiguration.status != AgentConfigurationStatusEnum.ARCHIVED
                    )
                )
                
                result = await db.execute(existing_query)
                existing_config = result.scalar_one_or_none()
                
                if existing_config:
                    if "general" not in validation_issues:
                        validation_issues["general"] = []
                    validation_issues["general"].append(
                        f"Configuration with name '{config_data.name}' already exists"
                    )
                
                # Create configuration
                config = AgentConfiguration(
                    id=uuid.uuid4(),
                    name=config_data.name,
                    description=config_data.description,
                    version=config_data.version,
                    model_name=config_data.model_name,
                    model_provider=config_data.model_provider,
                    temperature=config_data.temperature,
                    max_tokens=config_data.max_tokens,
                    top_p=config_data.top_p,
                    frequency_penalty=config_data.frequency_penalty,
                    presence_penalty=config_data.presence_penalty,
                    retrieval_strategy=config_data.retrieval_strategy,
                    retrieval_k=config_data.retrieval_k,
                    chunk_size=config_data.chunk_size,
                    chunk_overlap=config_data.chunk_overlap,
                    embedding_model=config_data.embedding_model,
                    system_prompt=config_data.system_prompt,
                    evaluation_prompt=config_data.evaluation_prompt,
                    configuration=config_data.configuration,
                    metadata=config_data.metadata,
                    organization_id=uuid.UUID(organization_id),
                    created_by=uuid.UUID(user_id)
                )
                
                # Generate version if not provided
                if not config.version:
                    await self.version_manager.create_version_snapshot(config, db)
                
                db.add(config)
                await db.commit()
                await db.refresh(config)
                
                self.logger.info(
                    f"Created configuration: {config.name}",
                    config_id=str(config.id),
                    version=config.version
                )
                
                # Convert to response
                response = AgentConfigurationResponse(
                    id=config.id,
                    name=config.name,
                    description=config.description,
                    version=config.version,
                    model_name=config.model_name,
                    model_provider=config.model_provider,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    top_p=config.top_p,
                    frequency_penalty=config.frequency_penalty,
                    presence_penalty=config.presence_penalty,
                    retrieval_strategy=config.retrieval_strategy,
                    retrieval_k=config.retrieval_k,
                    chunk_size=config.chunk_size,
                    chunk_overlap=config.chunk_overlap,
                    embedding_model=config.embedding_model,
                    system_prompt=config.system_prompt,
                    evaluation_prompt=config.evaluation_prompt,
                    configuration=config.configuration,
                    metadata=config.metadata,
                    status=config.status,
                    created_at=config.created_at,
                    updated_at=config.updated_at
                )
                
                return response, validation_issues
                
        except Exception as e:
            self.logger.error("Failed to create configuration", error=str(e))
            raise
    
    async def create_from_template(
        self,
        template_name: str,
        config_name: str,
        organization_id: str,
        user_id: str,
        customizations: Optional[Dict[str, Any]] = None
    ) -> AgentConfigurationResponse:
        """Create configuration from a predefined template"""
        try:
            templates = self.templates.get_templates()
            
            if template_name not in templates:
                raise ValueError(f"Template '{template_name}' not found")
            
            template = templates[template_name]
            
            # Apply customizations
            if customizations:
                template.update(customizations)
            
            # Create configuration data
            config_data = AgentConfigurationCreate(
                name=config_name,
                **template
            )
            
            response, validation_issues = await self.create_configuration(
                config_data=config_data,
                organization_id=organization_id,
                user_id=user_id,
                validate=True
            )
            
            self.logger.info(
                f"Created configuration from template: {template_name}",
                config_name=config_name
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Failed to create configuration from template: {template_name}", error=str(e))
            raise
    
    async def clone_configuration(
        self,
        source_config_id: str,
        new_name: str,
        organization_id: str,
        user_id: str,
        customizations: Optional[Dict[str, Any]] = None
    ) -> AgentConfigurationResponse:
        """Clone an existing configuration"""
        try:
            async with get_database() as db:
                # Get source configuration
                query = select(AgentConfiguration).where(
                    AgentConfiguration.id == uuid.UUID(source_config_id),
                    AgentConfiguration.organization_id == uuid.UUID(organization_id)
                )
                
                result = await db.execute(query)
                source_config = result.scalar_one_or_none()
                
                if not source_config:
                    raise ValueError("Source configuration not found")
                
                # Create configuration data from source
                config_data = AgentConfigurationCreate(
                    name=new_name,
                    description=f"Cloned from {source_config.name}",
                    version="1.0",  # Reset version for clone
                    model_name=source_config.model_name,
                    model_provider=source_config.model_provider,
                    temperature=source_config.temperature,
                    max_tokens=source_config.max_tokens,
                    top_p=source_config.top_p,
                    frequency_penalty=source_config.frequency_penalty,
                    presence_penalty=source_config.presence_penalty,
                    retrieval_strategy=source_config.retrieval_strategy,
                    retrieval_k=source_config.retrieval_k,
                    chunk_size=source_config.chunk_size,
                    chunk_overlap=source_config.chunk_overlap,
                    embedding_model=source_config.embedding_model,
                    system_prompt=source_config.system_prompt,
                    evaluation_prompt=source_config.evaluation_prompt,
                    configuration=source_config.configuration,
                    metadata={
                        **source_config.metadata,
                        "cloned_from": str(source_config.id),
                        "cloned_at": datetime.utcnow().isoformat()
                    }
                )
                
                # Apply customizations
                if customizations:
                    for field, value in customizations.items():
                        if hasattr(config_data, field):
                            setattr(config_data, field, value)
                
                response, _ = await self.create_configuration(
                    config_data=config_data,
                    organization_id=organization_id,
                    user_id=user_id,
                    validate=True
                )
                
                self.logger.info(
                    f"Cloned configuration: {source_config.name} -> {new_name}",
                    source_id=source_config_id,
                    new_id=str(response.id)
                )
                
                return response
                
        except Exception as e:
            self.logger.error(f"Failed to clone configuration: {source_config_id}", error=str(e))
            raise
    
    async def get_configuration_history(
        self,
        config_name: str,
        organization_id: str
    ) -> List[AgentConfigurationResponse]:
        """Get version history for a configuration"""
        try:
            async with get_database() as db:
                query = select(AgentConfiguration).where(
                    AgentConfiguration.name == config_name,
                    AgentConfiguration.organization_id == uuid.UUID(organization_id)
                ).order_by(AgentConfiguration.created_at.desc())
                
                result = await db.execute(query)
                configs = result.scalars().all()
                
                return [
                    AgentConfigurationResponse(
                        id=config.id,
                        name=config.name,
                        description=config.description,
                        version=config.version,
                        model_name=config.model_name,
                        model_provider=config.model_provider,
                        temperature=config.temperature,
                        max_tokens=config.max_tokens,
                        top_p=config.top_p,
                        frequency_penalty=config.frequency_penalty,
                        presence_penalty=config.presence_penalty,
                        retrieval_strategy=config.retrieval_strategy,
                        retrieval_k=config.retrieval_k,
                        chunk_size=config.chunk_size,
                        chunk_overlap=config.chunk_overlap,
                        embedding_model=config.embedding_model,
                        system_prompt=config.system_prompt,
                        evaluation_prompt=config.evaluation_prompt,
                        configuration=config.configuration,
                        metadata=config.metadata,
                        status=config.status,
                        created_at=config.created_at,
                        updated_at=config.updated_at
                    )
                    for config in configs
                ]
                
        except Exception as e:
            self.logger.error(f"Failed to get configuration history: {config_name}", error=str(e))
            return []
    
    def get_available_templates(self) -> Dict[str, Dict[str, Any]]:
        """Get all available configuration templates"""
        return self.templates.get_templates()
    
    async def export_configuration(
        self,
        config_id: str,
        organization_id: str,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """Export configuration for backup or sharing"""
        try:
            async with get_database() as db:
                query = select(AgentConfiguration).where(
                    AgentConfiguration.id == uuid.UUID(config_id),
                    AgentConfiguration.organization_id == uuid.UUID(organization_id)
                )
                
                result = await db.execute(query)
                config = result.scalar_one_or_none()
                
                if not config:
                    raise ValueError("Configuration not found")
                
                export_data = {
                    "name": config.name,
                    "description": config.description,
                    "version": config.version,
                    "model_name": config.model_name,
                    "model_provider": config.model_provider.value,
                    "temperature": config.temperature,
                    "max_tokens": config.max_tokens,
                    "top_p": config.top_p,
                    "frequency_penalty": config.frequency_penalty,
                    "presence_penalty": config.presence_penalty,
                    "retrieval_strategy": config.retrieval_strategy,
                    "retrieval_k": config.retrieval_k,
                    "chunk_size": config.chunk_size,
                    "chunk_overlap": config.chunk_overlap,
                    "embedding_model": config.embedding_model,
                    "system_prompt": config.system_prompt,
                    "evaluation_prompt": config.evaluation_prompt,
                    "configuration": config.configuration
                }
                
                if include_metadata:
                    export_data["metadata"] = {
                        "exported_at": datetime.utcnow().isoformat(),
                        "original_id": str(config.id),
                        "created_at": config.created_at.isoformat(),
                        "status": config.status.value,
                        **config.metadata
                    }
                
                return export_data
                
        except Exception as e:
            self.logger.error(f"Failed to export configuration: {config_id}", error=str(e))
            raise


# Global service instance
_configuration_service: Optional[ConfigurationService] = None


def get_configuration_service() -> ConfigurationService:
    """Get or create the global configuration service"""
    global _configuration_service
    if _configuration_service is None:
        _configuration_service = ConfigurationService()
    return _configuration_service