# Workflow/Pipeline Orchestration API
import os
import uuid
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from enum import Enum

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, WebSocket
from pydantic import BaseModel, validator
from sqlalchemy.orm import Session
from sqlalchemy import Column, String, DateTime, Text, Boolean, JSON, UUID as PG_UUID, Integer, Float, ForeignKey, func, desc, and_
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

# Workflow orchestration libraries
import airflow
from airflow.models import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from celery import Celery
import prefect
from prefect import flow, task, get_run_logger

# Task execution
import docker
import kubernetes
from kubernetes import client as k8s_client, config as k8s_config

Base = declarative_base()
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/workflows", tags=["workflows"])

# Enums
class WorkflowStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"

class ExecutionStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"

class TaskType(str, Enum):
    DATA_LOAD = "data_load"
    DATA_TRANSFORM = "data_transform"
    FEATURE_ENGINEER = "feature_engineer"
    MODEL_TRAIN = "model_train"
    MODEL_EVALUATE = "model_evaluate"
    MODEL_DEPLOY = "model_deploy"
    CUSTOM_SCRIPT = "custom_script"
    HTTP_REQUEST = "http_request"
    EMAIL_NOTIFICATION = "email_notification"

# Database Models
class Workflow(Base):
    """Workflow/Pipeline definition"""
    __tablename__ = "workflows"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    display_name = Column(String(255))
    description = Column(Text)
    
    # Workflow configuration
    workflow_type = Column(String(100))  # ml_pipeline, data_pipeline, deployment_pipeline
    version = Column(String(50), default="1.0.0")
    
    # DAG definition
    dag_definition = Column(JSON)  # Workflow graph structure
    task_definitions = Column(JSON)  # Individual task configurations
    
    # Scheduling
    schedule_type = Column(String(50), default="manual")  # manual, cron, event, continuous
    schedule_config = Column(JSON)  # Cron expression, event triggers, etc.
    is_scheduled = Column(Boolean, default=False)
    next_run_time = Column(DateTime)
    
    # Configuration and parameters
    default_parameters = Column(JSON)  # Default parameter values
    environment_variables = Column(JSON)  # Environment variables
    resource_requirements = Column(JSON)  # CPU, memory, GPU requirements
    
    # Retry and timeout configuration
    max_retries = Column(Integer, default=3)
    retry_delay_seconds = Column(Integer, default=60)
    timeout_seconds = Column(Integer, default=3600)
    
    # Status and lifecycle
    status = Column(String(50), default=WorkflowStatus.DRAFT)
    is_active = Column(Boolean, default=True)
    
    # Statistics
    total_executions = Column(Integer, default=0)
    successful_executions = Column(Integer, default=0)
    failed_executions = Column(Integer, default=0)
    avg_execution_time_seconds = Column(Float, default=0.0)
    last_execution_time = Column(DateTime)
    
    # Metadata
    tags = Column(JSON)
    metadata = Column(JSON)
    
    # Ownership
    created_by = Column(String(255), nullable=False)
    organization_id = Column(String(255))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    executions = relationship("WorkflowExecution", back_populates="workflow", cascade="all, delete-orphan")

class WorkflowExecution(Base):
    """Workflow execution instance"""
    __tablename__ = "workflow_executions"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workflow_id = Column(PG_UUID(as_uuid=True), ForeignKey('workflows.id'), nullable=False)
    
    # Execution identification
    execution_name = Column(String(255))
    run_id = Column(String(255), unique=True)
    
    # Execution configuration
    parameters = Column(JSON)  # Runtime parameters
    triggered_by = Column(String(100))  # manual, schedule, event, api
    trigger_info = Column(JSON)  # Additional trigger information
    
    # Status and timing
    status = Column(String(50), default=ExecutionStatus.QUEUED)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    duration_seconds = Column(Integer)
    
    # Resource usage
    resource_usage = Column(JSON)  # CPU, memory, storage usage
    
    # Logs and outputs
    execution_log = Column(Text)
    error_message = Column(Text)
    outputs = Column(JSON)  # Execution outputs/artifacts
    
    # Metadata
    environment = Column(JSON)  # Execution environment info
    
    # Ownership
    created_by = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    workflow = relationship("Workflow", back_populates="executions")
    task_executions = relationship("TaskExecution", back_populates="workflow_execution", cascade="all, delete-orphan")

class TaskExecution(Base):
    """Individual task execution within a workflow"""
    __tablename__ = "task_executions"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workflow_execution_id = Column(PG_UUID(as_uuid=True), ForeignKey('workflow_executions.id'), nullable=False)
    
    # Task identification
    task_id = Column(String(255), nullable=False)  # Task ID from workflow definition
    task_name = Column(String(255))
    task_type = Column(String(100))
    
    # Status and timing
    status = Column(String(50), default=TaskStatus.PENDING)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    duration_seconds = Column(Integer)
    
    # Execution details
    retry_count = Column(Integer, default=0)
    error_message = Column(Text)
    task_log = Column(Text)
    
    # Input/Output
    inputs = Column(JSON)
    outputs = Column(JSON)
    artifacts = Column(JSON)  # Generated artifacts
    
    # Resource usage
    resource_usage = Column(JSON)
    
    # Relationships
    workflow_execution = relationship("WorkflowExecution", back_populates="task_executions")

# Pydantic Models
class TaskDefinition(BaseModel):
    id: str
    name: str
    task_type: TaskType
    config: Dict[str, Any]
    dependencies: List[str] = []  # List of task IDs this task depends on
    retry_count: int = 0
    timeout_seconds: Optional[int] = None
    resource_requirements: Optional[Dict[str, Any]] = {}

class WorkflowCreate(BaseModel):
    name: str
    display_name: Optional[str] = None
    description: Optional[str] = None
    workflow_type: str = "ml_pipeline"
    task_definitions: List[TaskDefinition]
    default_parameters: Optional[Dict[str, Any]] = {}
    schedule_type: str = "manual"
    schedule_config: Optional[Dict[str, Any]] = {}
    max_retries: int = 3
    timeout_seconds: int = 3600
    tags: Optional[List[str]] = []

class WorkflowUpdate(BaseModel):
    display_name: Optional[str] = None
    description: Optional[str] = None
    task_definitions: Optional[List[TaskDefinition]] = None
    default_parameters: Optional[Dict[str, Any]] = None
    schedule_type: Optional[str] = None
    schedule_config: Optional[Dict[str, Any]] = None
    status: Optional[WorkflowStatus] = None
    tags: Optional[List[str]] = None

class WorkflowExecutionCreate(BaseModel):
    execution_name: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = {}
    triggered_by: str = "manual"

class WorkflowResponse(BaseModel):
    id: str
    name: str
    display_name: Optional[str]
    description: Optional[str]
    workflow_type: str
    status: str
    total_executions: int
    successful_executions: int
    failed_executions: int
    created_by: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True

class WorkflowExecutionResponse(BaseModel):
    id: str
    workflow_id: str
    execution_name: Optional[str]
    status: str
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    duration_seconds: Optional[int]
    triggered_by: str
    created_at: datetime
    
    class Config:
        orm_mode = True

# Dependency injection
def get_db():
    pass

def get_current_user():
    return "current_user_id"

# Workflow Orchestration Service
class WorkflowOrchestrationService:
    """Service for workflow orchestration and execution"""
    
    def __init__(self, db: Session, executor_type: str = "local"):
        self.db = db
        self.executor_type = executor_type  # local, celery, airflow, prefect, kubernetes
        
        # Initialize executors
        self._init_executors()
        
        # Task registry
        self.task_handlers = {
            TaskType.DATA_LOAD: self._execute_data_load,
            TaskType.DATA_TRANSFORM: self._execute_data_transform,
            TaskType.FEATURE_ENGINEER: self._execute_feature_engineer,
            TaskType.MODEL_TRAIN: self._execute_model_train,
            TaskType.MODEL_EVALUATE: self._execute_model_evaluate,
            TaskType.MODEL_DEPLOY: self._execute_model_deploy,
            TaskType.CUSTOM_SCRIPT: self._execute_custom_script,
            TaskType.HTTP_REQUEST: self._execute_http_request,
            TaskType.EMAIL_NOTIFICATION: self._execute_email_notification
        }
    
    def _init_executors(self):
        """Initialize workflow executors"""
        
        if self.executor_type == "celery":
            self.celery_app = Celery('workflow_executor')
            self.celery_app.config_from_object({
                'broker_url': 'redis://localhost:6379/0',
                'result_backend': 'redis://localhost:6379/0'
            })
        elif self.executor_type == "kubernetes":
            try:
                k8s_config.load_incluster_config()
            except:
                k8s_config.load_kube_config()
            self.k8s_v1 = k8s_client.CoreV1Api()
            self.k8s_batch = k8s_client.BatchV1Api()
    
    async def create_workflow(self, workflow_data: WorkflowCreate, user_id: str) -> Workflow:
        """Create a new workflow"""
        
        # Build DAG structure from task definitions
        dag_definition = self._build_dag_definition(workflow_data.task_definitions)
        
        workflow = Workflow(
            name=workflow_data.name,
            display_name=workflow_data.display_name or workflow_data.name,
            description=workflow_data.description,
            workflow_type=workflow_data.workflow_type,
            dag_definition=dag_definition,
            task_definitions=[task.dict() for task in workflow_data.task_definitions],
            default_parameters=workflow_data.default_parameters,
            schedule_type=workflow_data.schedule_type,
            schedule_config=workflow_data.schedule_config,
            max_retries=workflow_data.max_retries,
            timeout_seconds=workflow_data.timeout_seconds,
            tags=workflow_data.tags,
            created_by=user_id,
            organization_id=self._get_user_org(user_id)
        )
        
        self.db.add(workflow)
        self.db.commit()
        self.db.refresh(workflow)
        
        logger.info(f"Created workflow {workflow.name} (ID: {workflow.id})")
        return workflow
    
    async def execute_workflow(self, workflow_id: str, execution_data: WorkflowExecutionCreate, user_id: str) -> WorkflowExecution:
        """Start workflow execution"""
        
        workflow = self._get_workflow_by_id(workflow_id, user_id)
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        if workflow.status != WorkflowStatus.ACTIVE:
            raise HTTPException(status_code=400, detail="Workflow is not active")
        
        # Create execution record
        run_id = f"{workflow.name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        execution = WorkflowExecution(
            workflow_id=workflow.id,
            execution_name=execution_data.execution_name or run_id,
            run_id=run_id,
            parameters=execution_data.parameters or workflow.default_parameters,
            triggered_by=execution_data.triggered_by,
            status=ExecutionStatus.QUEUED,
            created_by=user_id
        )
        
        self.db.add(execution)
        
        # Update workflow statistics
        workflow.total_executions += 1
        workflow.last_execution_time = datetime.utcnow()
        
        self.db.commit()
        self.db.refresh(execution)
        
        # Start execution asynchronously
        await self._start_workflow_execution(execution)
        
        logger.info(f"Started workflow execution {run_id}")
        return execution
    
    async def _start_workflow_execution(self, execution: WorkflowExecution):
        """Start the actual workflow execution"""
        
        workflow = execution.workflow
        
        try:
            # Update execution status
            execution.status = ExecutionStatus.RUNNING
            execution.start_time = datetime.utcnow()
            self.db.commit()
            
            # Execute workflow based on executor type
            if self.executor_type == "local":
                await self._execute_local_workflow(execution)
            elif self.executor_type == "celery":
                await self._execute_celery_workflow(execution)
            elif self.executor_type == "airflow":
                await self._execute_airflow_workflow(execution)
            elif self.executor_type == "prefect":
                await self._execute_prefect_workflow(execution)
            elif self.executor_type == "kubernetes":
                await self._execute_kubernetes_workflow(execution)
            else:
                raise ValueError(f"Unsupported executor type: {self.executor_type}")
            
            # Mark execution as completed
            execution.status = ExecutionStatus.COMPLETED
            execution.end_time = datetime.utcnow()
            execution.duration_seconds = int((execution.end_time - execution.start_time).total_seconds())
            
            # Update workflow statistics
            workflow.successful_executions += 1
            self._update_avg_execution_time(workflow)
            
            self.db.commit()
            
        except Exception as e:
            # Mark execution as failed
            execution.status = ExecutionStatus.FAILED
            execution.end_time = datetime.utcnow()
            execution.error_message = str(e)
            if execution.start_time:
                execution.duration_seconds = int((execution.end_time - execution.start_time).total_seconds())
            
            # Update workflow statistics
            workflow.failed_executions += 1
            
            self.db.commit()
            
            logger.error(f"Workflow execution {execution.run_id} failed: {str(e)}")
    
    async def _execute_local_workflow(self, execution: WorkflowExecution):
        """Execute workflow locally using threading/asyncio"""
        
        workflow = execution.workflow
        task_definitions = {task['id']: task for task in workflow.task_definitions}
        dag = workflow.dag_definition
        
        # Create task execution records
        task_executions = {}
        for task_id in task_definitions.keys():
            task_exec = TaskExecution(
                workflow_execution_id=execution.id,
                task_id=task_id,
                task_name=task_definitions[task_id]['name'],
                task_type=task_definitions[task_id]['task_type'],
                status=TaskStatus.PENDING
            )
            self.db.add(task_exec)
            task_executions[task_id] = task_exec
        
        self.db.commit()
        
        # Execute tasks in topological order
        completed_tasks = set()
        
        while len(completed_tasks) < len(task_definitions):
            # Find tasks ready to execute
            ready_tasks = []
            for task_id, task_def in task_definitions.items():
                if (task_id not in completed_tasks and 
                    all(dep in completed_tasks for dep in task_def.get('dependencies', []))):
                    ready_tasks.append(task_id)
            
            if not ready_tasks:
                raise RuntimeError("Circular dependency detected or no tasks ready")
            
            # Execute ready tasks
            for task_id in ready_tasks:
                await self._execute_task(task_executions[task_id], task_definitions[task_id], execution.parameters)
                completed_tasks.add(task_id)
    
    async def _execute_task(self, task_execution: TaskExecution, task_definition: Dict[str, Any], parameters: Dict[str, Any]):
        """Execute a single task"""
        
        try:
            # Update task status
            task_execution.status = TaskStatus.RUNNING
            task_execution.start_time = datetime.utcnow()
            self.db.commit()
            
            # Get task handler
            task_type = TaskType(task_definition['task_type'])
            handler = self.task_handlers.get(task_type)
            
            if not handler:
                raise ValueError(f"No handler for task type: {task_type}")
            
            # Execute task
            result = await handler(task_execution, task_definition, parameters)
            
            # Update task with results
            task_execution.status = TaskStatus.COMPLETED
            task_execution.end_time = datetime.utcnow()
            task_execution.duration_seconds = int((task_execution.end_time - task_execution.start_time).total_seconds())
            task_execution.outputs = result.get('outputs', {})
            task_execution.artifacts = result.get('artifacts', {})
            
            self.db.commit()
            
        except Exception as e:
            # Mark task as failed
            task_execution.status = TaskStatus.FAILED
            task_execution.end_time = datetime.utcnow()
            task_execution.error_message = str(e)
            if task_execution.start_time:
                task_execution.duration_seconds = int((task_execution.end_time - task_execution.start_time).total_seconds())
            
            self.db.commit()
            
            # Re-raise to fail the workflow
            raise e
    
    async def get_workflows(self, user_id: str, skip: int = 0, limit: int = 100, filters: Optional[Dict[str, Any]] = None) -> List[Workflow]:
        """Get workflows with filtering"""
        
        query = self.db.query(Workflow).filter(
            (Workflow.created_by == user_id) |
            (Workflow.organization_id == self._get_user_org(user_id))
        )
        
        if filters:
            if filters.get('status'):
                query = query.filter(Workflow.status == filters['status'])
            if filters.get('workflow_type'):
                query = query.filter(Workflow.workflow_type == filters['workflow_type'])
        
        return query.order_by(desc(Workflow.created_at)).offset(skip).limit(limit).all()
    
    async def get_workflow_executions(self, workflow_id: str, user_id: str, skip: int = 0, limit: int = 50) -> List[WorkflowExecution]:
        """Get workflow executions"""
        
        workflow = self._get_workflow_by_id(workflow_id, user_id)
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        return self.db.query(WorkflowExecution).filter(
            WorkflowExecution.workflow_id == workflow.id
        ).order_by(desc(WorkflowExecution.created_at)).offset(skip).limit(limit).all()
    
    async def cancel_execution(self, execution_id: str, user_id: str) -> Dict[str, Any]:
        """Cancel a running workflow execution"""
        
        execution = self.db.query(WorkflowExecution).filter(
            WorkflowExecution.id == execution_id
        ).first()
        
        if not execution:
            raise HTTPException(status_code=404, detail="Execution not found")
        
        workflow = execution.workflow
        if workflow.created_by != user_id:
            raise HTTPException(status_code=403, detail="Permission denied")
        
        if execution.status not in [ExecutionStatus.QUEUED, ExecutionStatus.RUNNING]:
            raise HTTPException(status_code=400, detail="Execution cannot be cancelled")
        
        # Update execution status
        execution.status = ExecutionStatus.CANCELLED
        execution.end_time = datetime.utcnow()
        if execution.start_time:
            execution.duration_seconds = int((execution.end_time - execution.start_time).total_seconds())
        
        # Cancel running tasks
        self.db.query(TaskExecution).filter(
            TaskExecution.workflow_execution_id == execution.id,
            TaskExecution.status == TaskStatus.RUNNING
        ).update({'status': TaskStatus.FAILED, 'error_message': 'Cancelled by user'})
        
        self.db.commit()
        
        return {'success': True, 'message': 'Execution cancelled'}
    
    # Task handlers
    async def _execute_data_load(self, task_execution: TaskExecution, task_definition: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data loading task"""
        
        config = task_definition['config']
        source_type = config.get('source_type', 'file')
        
        if source_type == 'file':
            file_path = config['file_path']
            # Load data from file
            # This would integrate with the dataset management service
            return {
                'outputs': {'data_path': file_path, 'row_count': 1000},
                'artifacts': {'data_sample': 'sample_data.csv'}
            }
        elif source_type == 'database':
            # Load data from database
            connection_string = config['connection_string']
            query = config['query']
            # Execute database query
            return {
                'outputs': {'data_path': 'database_extract.csv', 'row_count': 5000},
                'artifacts': {}
            }
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
    
    async def _execute_data_transform(self, task_execution: TaskExecution, task_definition: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data transformation task"""
        
        config = task_definition['config']
        transformation_type = config.get('transformation_type', 'clean')
        
        # This would integrate with the dataset transformation service
        return {
            'outputs': {'transformed_data_path': 'transformed_data.csv', 'transformation_log': 'Applied cleaning'},
            'artifacts': {'transformation_report': 'transform_report.html'}
        }
    
    async def _execute_feature_engineer(self, task_execution: TaskExecution, task_definition: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute feature engineering task"""
        
        config = task_definition['config']
        
        return {
            'outputs': {'feature_data_path': 'features.csv', 'feature_count': 25},
            'artifacts': {'feature_importance': 'feature_importance.json'}
        }
    
    async def _execute_model_train(self, task_execution: TaskExecution, task_definition: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute model training task"""
        
        config = task_definition['config']
        model_type = config.get('model_type', 'random_forest')
        
        # This would integrate with the model management service
        return {
            'outputs': {'model_path': 'trained_model.pkl', 'accuracy': 0.95},
            'artifacts': {'training_log': 'training.log', 'metrics': 'metrics.json'}
        }
    
    async def _execute_model_evaluate(self, task_execution: TaskExecution, task_definition: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute model evaluation task"""
        
        config = task_definition['config']
        
        return {
            'outputs': {'evaluation_score': 0.92, 'evaluation_report_path': 'evaluation.html'},
            'artifacts': {'confusion_matrix': 'confusion_matrix.png', 'roc_curve': 'roc_curve.png'}
        }
    
    async def _execute_model_deploy(self, task_execution: TaskExecution, task_definition: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute model deployment task"""
        
        config = task_definition['config']
        deployment_target = config.get('target', 'staging')
        
        # This would integrate with the deployment service
        return {
            'outputs': {'endpoint_url': f'https://api.example.com/models/{uuid.uuid4()}', 'deployment_id': str(uuid.uuid4())},
            'artifacts': {'deployment_config': 'deployment.yaml'}
        }
    
    async def _execute_custom_script(self, task_execution: TaskExecution, task_definition: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute custom script task"""
        
        config = task_definition['config']
        script_path = config['script_path']
        script_args = config.get('args', [])
        
        # Execute script using subprocess or container
        import subprocess
        result = subprocess.run(['python', script_path] + script_args, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Script failed: {result.stderr}")
        
        return {
            'outputs': {'exit_code': result.returncode, 'stdout': result.stdout},
            'artifacts': {}
        }
    
    async def _execute_http_request(self, task_execution: TaskExecution, task_definition: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute HTTP request task"""
        
        config = task_definition['config']
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            async with session.request(
                method=config.get('method', 'GET'),
                url=config['url'],
                json=config.get('json'),
                headers=config.get('headers', {})
            ) as response:
                response_text = await response.text()
                
                return {
                    'outputs': {
                        'status_code': response.status,
                        'response': response_text,
                        'headers': dict(response.headers)
                    },
                    'artifacts': {}
                }
    
    async def _execute_email_notification(self, task_execution: TaskExecution, task_definition: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute email notification task"""
        
        config = task_definition['config']
        
        # This would integrate with an email service
        return {
            'outputs': {'email_sent': True, 'recipients': config.get('recipients', [])},
            'artifacts': {}
        }
    
    # Helper methods
    def _build_dag_definition(self, task_definitions: List[TaskDefinition]) -> Dict[str, Any]:
        """Build DAG structure from task definitions"""
        
        dag = {
            'nodes': [],
            'edges': []
        }
        
        # Add nodes
        for task in task_definitions:
            dag['nodes'].append({
                'id': task.id,
                'name': task.name,
                'type': task.task_type
            })
        
        # Add edges (dependencies)
        for task in task_definitions:
            for dependency in task.dependencies:
                dag['edges'].append({
                    'source': dependency,
                    'target': task.id
                })
        
        return dag
    
    def _get_workflow_by_id(self, workflow_id: str, user_id: str) -> Optional[Workflow]:
        """Get workflow by ID with access control"""
        
        return self.db.query(Workflow).filter(
            Workflow.id == workflow_id,
            (
                (Workflow.created_by == user_id) |
                (Workflow.organization_id == self._get_user_org(user_id))
            )
        ).first()
    
    def _get_user_org(self, user_id: str) -> Optional[str]:
        """Get user's organization ID"""
        return "default_org"
    
    def _update_avg_execution_time(self, workflow: Workflow):
        """Update average execution time for workflow"""
        
        avg_time = self.db.query(func.avg(WorkflowExecution.duration_seconds)).filter(
            WorkflowExecution.workflow_id == workflow.id,
            WorkflowExecution.status == ExecutionStatus.COMPLETED
        ).scalar()
        
        if avg_time:
            workflow.avg_execution_time_seconds = float(avg_time)
    
    # Placeholder methods for other executors
    async def _execute_celery_workflow(self, execution: WorkflowExecution):
        """Execute workflow using Celery"""
        # Implementation for Celery execution
        pass
    
    async def _execute_airflow_workflow(self, execution: WorkflowExecution):
        """Execute workflow using Airflow"""
        # Implementation for Airflow execution
        pass
    
    async def _execute_prefect_workflow(self, execution: WorkflowExecution):
        """Execute workflow using Prefect"""
        # Implementation for Prefect execution
        pass
    
    async def _execute_kubernetes_workflow(self, execution: WorkflowExecution):
        """Execute workflow using Kubernetes Jobs"""
        # Implementation for Kubernetes execution
        pass

# API Endpoints
@router.post("/", response_model=WorkflowResponse)
async def create_workflow(
    workflow_data: WorkflowCreate,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Create a new workflow"""
    
    service = WorkflowOrchestrationService(db)
    workflow = await service.create_workflow(workflow_data, current_user)
    return workflow

@router.get("/", response_model=List[WorkflowResponse])
async def get_workflows(
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None,
    workflow_type: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Get workflows with filtering"""
    
    filters = {}
    if status:
        filters['status'] = status
    if workflow_type:
        filters['workflow_type'] = workflow_type
    
    service = WorkflowOrchestrationService(db)
    workflows = await service.get_workflows(current_user, skip, limit, filters)
    return workflows

@router.get("/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(
    workflow_id: str,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Get a specific workflow"""
    
    service = WorkflowOrchestrationService(db)
    workflow = service._get_workflow_by_id(workflow_id, current_user)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return workflow

@router.put("/{workflow_id}", response_model=WorkflowResponse)
async def update_workflow(
    workflow_id: str,
    workflow_data: WorkflowUpdate,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Update workflow"""
    
    service = WorkflowOrchestrationService(db)
    workflow = service._get_workflow_by_id(workflow_id, current_user)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    if workflow.created_by != current_user:
        raise HTTPException(status_code=403, detail="Permission denied")
    
    # Update fields
    for field, value in workflow_data.dict(exclude_unset=True).items():
        if field == 'task_definitions' and value:
            # Rebuild DAG when task definitions change
            setattr(workflow, 'dag_definition', service._build_dag_definition(value))
            setattr(workflow, 'task_definitions', [task.dict() for task in value])
        else:
            setattr(workflow, field, value)
    
    workflow.updated_at = datetime.utcnow()
    service.db.commit()
    
    return workflow

@router.post("/{workflow_id}/execute", response_model=WorkflowExecutionResponse)
async def execute_workflow(
    workflow_id: str,
    execution_data: WorkflowExecutionCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Execute workflow"""
    
    service = WorkflowOrchestrationService(db)
    execution = await service.execute_workflow(workflow_id, execution_data, current_user)
    return execution

@router.get("/{workflow_id}/executions", response_model=List[WorkflowExecutionResponse])
async def get_workflow_executions(
    workflow_id: str,
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Get workflow executions"""
    
    service = WorkflowOrchestrationService(db)
    executions = await service.get_workflow_executions(workflow_id, current_user, skip, limit)
    return executions

@router.get("/executions/{execution_id}")
async def get_execution_details(
    execution_id: str,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Get detailed execution information"""
    
    service = WorkflowOrchestrationService(db)
    execution = service.db.query(WorkflowExecution).filter(
        WorkflowExecution.id == execution_id
    ).first()
    
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")
    
    workflow = execution.workflow
    if workflow.created_by != current_user:
        raise HTTPException(status_code=403, detail="Permission denied")
    
    # Get task executions
    task_executions = service.db.query(TaskExecution).filter(
        TaskExecution.workflow_execution_id == execution.id
    ).all()
    
    return {
        'execution': {
            'id': str(execution.id),
            'workflow_id': str(execution.workflow_id),
            'execution_name': execution.execution_name,
            'status': execution.status,
            'start_time': execution.start_time,
            'end_time': execution.end_time,
            'duration_seconds': execution.duration_seconds,
            'parameters': execution.parameters,
            'error_message': execution.error_message
        },
        'task_executions': [
            {
                'id': str(task.id),
                'task_id': task.task_id,
                'task_name': task.task_name,
                'task_type': task.task_type,
                'status': task.status,
                'start_time': task.start_time,
                'end_time': task.end_time,
                'duration_seconds': task.duration_seconds,
                'outputs': task.outputs,
                'error_message': task.error_message
            }
            for task in task_executions
        ]
    }

@router.post("/executions/{execution_id}/cancel")
async def cancel_execution(
    execution_id: str,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Cancel workflow execution"""
    
    service = WorkflowOrchestrationService(db)
    result = await service.cancel_execution(execution_id, current_user)
    return result

@router.delete("/{workflow_id}")
async def delete_workflow(
    workflow_id: str,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Delete workflow"""
    
    service = WorkflowOrchestrationService(db)
    workflow = service._get_workflow_by_id(workflow_id, current_user)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    if workflow.created_by != current_user:
        raise HTTPException(status_code=403, detail="Permission denied")
    
    # Check for running executions
    running_executions = service.db.query(WorkflowExecution).filter(
        WorkflowExecution.workflow_id == workflow.id,
        WorkflowExecution.status.in_([ExecutionStatus.QUEUED, ExecutionStatus.RUNNING])
    ).count()
    
    if running_executions > 0:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete workflow with running executions"
        )
    
    # Delete workflow (cascading)
    service.db.delete(workflow)
    service.db.commit()
    
    logger.info(f"Deleted workflow {workflow.name} (ID: {workflow.id})")
    
    return {'success': True, 'message': 'Workflow deleted successfully'}

@router.websocket("/{workflow_id}/executions/{execution_id}/logs")
async def get_execution_logs(
    websocket: WebSocket,
    workflow_id: str,
    execution_id: str
):
    """Stream execution logs via WebSocket"""
    
    await websocket.accept()
    
    try:
        # This would stream real-time execution logs
        # For now, just send a sample message
        await websocket.send_text(json.dumps({
            'type': 'log',
            'timestamp': datetime.utcnow().isoformat(),
            'message': 'Execution started',
            'level': 'INFO'
        }))
        
        # Keep connection open and stream logs
        while True:
            await asyncio.sleep(1)
            # In real implementation, this would stream actual logs
            
    except Exception as e:
        await websocket.send_text(json.dumps({
            'type': 'error',
            'message': str(e)
        }))
    finally:
        await websocket.close()