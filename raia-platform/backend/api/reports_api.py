# Report Generation API
import os
import uuid
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from enum import Enum

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, validator
from sqlalchemy.orm import Session
from sqlalchemy import Column, String, DateTime, Text, Boolean, JSON, UUID as PG_UUID, Integer, Float, ForeignKey, func, desc
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

# Report generation libraries
import jinja2
from jinja2 import Environment, FileSystemLoader, Template
import pdfkit
import weasyprint
from weasyprint import HTML, CSS

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.utils import PlotlyJSONEncoder

# Data processing
import pandas as pd
import numpy as np
from io import BytesIO, StringIO
import base64

# Email and file handling
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import zipfile
import shutil

Base = declarative_base()
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/reports", tags=["reports"])

# Enums
class ReportType(str, Enum):
    EXPERIMENT_SUMMARY = "experiment_summary"
    MODEL_PERFORMANCE = "model_performance"
    DATA_QUALITY = "data_quality"
    USAGE_ANALYTICS = "usage_analytics"
    COMPLIANCE_AUDIT = "compliance_audit"
    EXECUTIVE_SUMMARY = "executive_summary"
    CUSTOM = "custom"

class ReportFormat(str, Enum):
    PDF = "pdf"
    HTML = "html"
    EXCEL = "excel"
    JSON = "json"
    CSV = "csv"

class ReportStatus(str, Enum):
    DRAFT = "draft"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"

class ScheduleFrequency(str, Enum):
    ONCE = "once"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"

# Database Models
class ReportTemplate(Base):
    """Report template definitions"""
    __tablename__ = "report_templates"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    display_name = Column(String(255))
    description = Column(Text)
    
    # Template configuration
    report_type = Column(String(100), nullable=False)
    template_content = Column(Text, nullable=False)  # Jinja2 template
    css_content = Column(Text)  # CSS styles
    js_content = Column(Text)  # JavaScript for interactive elements
    
    # Template structure
    sections = Column(JSON)  # List of report sections
    parameters = Column(JSON)  # Template parameters schema
    data_requirements = Column(JSON)  # Required data sources
    
    # Configuration
    default_format = Column(String(50), default=ReportFormat.PDF)
    supported_formats = Column(JSON)  # List of supported formats
    
    # Metadata
    tags = Column(JSON)
    category = Column(String(100))
    
    # Version control
    version = Column(String(50), default="1.0.0")
    is_active = Column(Boolean, default=True)
    
    # Ownership
    created_by = Column(String(255), nullable=False)
    organization_id = Column(String(255))
    is_public = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    reports = relationship("Report", back_populates="template")

class Report(Base):
    """Generated reports"""
    __tablename__ = "reports"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    template_id = Column(PG_UUID(as_uuid=True), ForeignKey('report_templates.id'))
    
    # Report identification
    name = Column(String(255), nullable=False)
    title = Column(String(255))
    description = Column(Text)
    
    # Report configuration
    report_type = Column(String(100))
    format = Column(String(50), default=ReportFormat.PDF)
    parameters = Column(JSON)  # Report generation parameters
    
    # Data sources
    data_sources = Column(JSON)  # List of data sources used
    filters = Column(JSON)  # Applied filters
    date_range = Column(JSON)  # Report date range
    
    # Generation details
    status = Column(String(50), default=ReportStatus.DRAFT)
    generation_started = Column(DateTime)
    generation_completed = Column(DateTime)
    generation_duration_seconds = Column(Integer)
    error_message = Column(Text)
    
    # File information
    file_path = Column(String(1000))
    file_size_bytes = Column(Integer)
    file_url = Column(String(1000))  # Public URL if shared
    
    # Content metadata
    page_count = Column(Integer)
    charts_count = Column(Integer)
    tables_count = Column(Integer)
    
    # Sharing and access
    is_public = Column(Boolean, default=False)
    share_token = Column(String(255))  # Token for public sharing
    expires_at = Column(DateTime)
    download_count = Column(Integer, default=0)
    
    # Scheduling
    is_scheduled = Column(Boolean, default=False)
    schedule_config = Column(JSON)  # Schedule configuration
    next_generation = Column(DateTime)
    
    # Email distribution
    email_recipients = Column(JSON)  # List of email recipients
    is_emailed = Column(Boolean, default=False)
    emailed_at = Column(DateTime)
    
    # Ownership
    created_by = Column(String(255), nullable=False)
    organization_id = Column(String(255))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    template = relationship("ReportTemplate", back_populates="reports")

class ReportSnapshot(Base):
    """Report data snapshots for historical tracking"""
    __tablename__ = "report_snapshots"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    report_id = Column(PG_UUID(as_uuid=True), ForeignKey('reports.id'))
    
    # Snapshot metadata
    snapshot_date = Column(DateTime, default=datetime.utcnow)
    data_hash = Column(String(255))  # Hash of the data for comparison
    
    # Snapshot data
    raw_data = Column(JSON)  # Raw data used for generation
    computed_metrics = Column(JSON)  # Computed metrics/aggregations
    charts_data = Column(JSON)  # Chart data and configurations
    
    # File references
    snapshot_file_path = Column(String(1000))
    
    created_at = Column(DateTime, default=datetime.utcnow)

# Pydantic Models
class ReportTemplateCreate(BaseModel):
    name: str
    display_name: Optional[str] = None
    description: Optional[str] = None
    report_type: ReportType
    template_content: str
    css_content: Optional[str] = None
    sections: Optional[List[Dict[str, Any]]] = []
    parameters: Optional[Dict[str, Any]] = {}
    data_requirements: Optional[List[str]] = []
    supported_formats: Optional[List[ReportFormat]] = [ReportFormat.PDF]
    tags: Optional[List[str]] = []
    category: Optional[str] = None
    is_public: bool = False

class ReportTemplateUpdate(BaseModel):
    display_name: Optional[str] = None
    description: Optional[str] = None
    template_content: Optional[str] = None
    css_content: Optional[str] = None
    sections: Optional[List[Dict[str, Any]]] = None
    parameters: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None
    tags: Optional[List[str]] = None

class ReportCreate(BaseModel):
    name: str
    title: Optional[str] = None
    description: Optional[str] = None
    template_id: Optional[str] = None
    report_type: Optional[ReportType] = None
    format: ReportFormat = ReportFormat.PDF
    parameters: Optional[Dict[str, Any]] = {}
    data_sources: Optional[List[str]] = []
    filters: Optional[Dict[str, Any]] = {}
    date_range: Optional[Dict[str, Any]] = {}
    email_recipients: Optional[List[str]] = []
    expires_at: Optional[datetime] = None

class ReportScheduleCreate(BaseModel):
    frequency: ScheduleFrequency
    start_date: datetime
    end_date: Optional[datetime] = None
    time_of_day: str = "09:00"  # HH:MM format
    email_recipients: List[str]
    report_config: Dict[str, Any]

class ReportResponse(BaseModel):
    id: str
    name: str
    title: Optional[str]
    report_type: Optional[str]
    format: str
    status: str
    file_path: Optional[str]
    file_size_bytes: Optional[int]
    page_count: Optional[int]
    download_count: int
    created_by: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True

class ReportTemplateResponse(BaseModel):
    id: str
    name: str
    display_name: Optional[str]
    description: Optional[str]
    report_type: str
    category: Optional[str]
    version: str
    is_active: bool
    created_by: str
    created_at: datetime
    
    class Config:
        orm_mode = True

# Dependency injection
def get_db():
    pass

def get_current_user():
    return "current_user_id"

# Report Generation Service
class ReportGenerationService:
    """Service for report generation and management"""
    
    def __init__(self, db: Session, storage_path: str = "/tmp/raia_reports"):
        self.db = db
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize Jinja2 environment
        template_dir = self.storage_path / "templates"
        template_dir.mkdir(exist_ok=True)
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=True
        )
        
        # Chart configurations
        plt.style.use('seaborn-v0_8')
        pio.templates.default = "plotly_white"
        
        # Built-in report generators
        self.report_generators = {
            ReportType.EXPERIMENT_SUMMARY: self._generate_experiment_summary,
            ReportType.MODEL_PERFORMANCE: self._generate_model_performance,
            ReportType.DATA_QUALITY: self._generate_data_quality,
            ReportType.USAGE_ANALYTICS: self._generate_usage_analytics,
            ReportType.COMPLIANCE_AUDIT: self._generate_compliance_audit,
            ReportType.EXECUTIVE_SUMMARY: self._generate_executive_summary
        }
    
    async def create_template(self, template_data: ReportTemplateCreate, user_id: str) -> ReportTemplate:
        """Create a new report template"""
        
        template = ReportTemplate(
            name=template_data.name,
            display_name=template_data.display_name or template_data.name,
            description=template_data.description,
            report_type=template_data.report_type,
            template_content=template_data.template_content,
            css_content=template_data.css_content,
            sections=template_data.sections,
            parameters=template_data.parameters,
            data_requirements=template_data.data_requirements,
            supported_formats=template_data.supported_formats,
            tags=template_data.tags,
            category=template_data.category,
            is_public=template_data.is_public,
            created_by=user_id,
            organization_id=self._get_user_org(user_id)
        )
        
        self.db.add(template)
        self.db.commit()
        self.db.refresh(template)
        
        # Save template file
        template_file = self.storage_path / "templates" / f"{template.id}.html"
        with open(template_file, 'w', encoding='utf-8') as f:
            f.write(template_data.template_content)
        
        logger.info(f"Created report template {template.name} (ID: {template.id})")
        return template
    
    async def generate_report(self, report_data: ReportCreate, user_id: str, background_tasks: BackgroundTasks) -> Report:
        """Generate a new report"""
        
        # Create report record
        report = Report(
            name=report_data.name,
            title=report_data.title or report_data.name,
            description=report_data.description,
            template_id=report_data.template_id,
            report_type=report_data.report_type,
            format=report_data.format,
            parameters=report_data.parameters,
            data_sources=report_data.data_sources,
            filters=report_data.filters,
            date_range=report_data.date_range,
            email_recipients=report_data.email_recipients,
            expires_at=report_data.expires_at,
            created_by=user_id,
            organization_id=self._get_user_org(user_id)
        )
        
        self.db.add(report)
        self.db.commit()
        self.db.refresh(report)
        
        # Start generation in background
        background_tasks.add_task(self._generate_report_async, report.id)
        
        logger.info(f"Started report generation {report.name} (ID: {report.id})")
        return report
    
    async def _generate_report_async(self, report_id: str):
        """Generate report asynchronously"""
        
        report = self.db.query(Report).filter(Report.id == report_id).first()
        if not report:
            logger.error(f"Report {report_id} not found")
            return
        
        try:
            # Update status
            report.status = ReportStatus.GENERATING
            report.generation_started = datetime.utcnow()
            self.db.commit()
            
            # Generate report content
            if report.template_id:
                content = await self._generate_from_template(report)
            elif report.report_type:
                generator = self.report_generators.get(ReportType(report.report_type))
                if generator:
                    content = await generator(report)
                else:
                    raise ValueError(f"No generator for report type: {report.report_type}")
            else:
                raise ValueError("No template or report type specified")
            
            # Convert to specified format
            file_path = await self._convert_to_format(content, report)
            
            # Update report record
            report.status = ReportStatus.COMPLETED
            report.generation_completed = datetime.utcnow()
            report.generation_duration_seconds = int(
                (report.generation_completed - report.generation_started).total_seconds()
            )
            report.file_path = str(file_path)
            report.file_size_bytes = file_path.stat().st_size
            
            # Generate sharing token if needed
            if report.expires_at or report.email_recipients:
                report.share_token = self._generate_share_token()
            
            self.db.commit()
            
            # Send email if recipients specified
            if report.email_recipients:
                await self._send_report_email(report)
            
            logger.info(f"Completed report generation {report.name}")
            
        except Exception as e:
            # Mark as failed
            report.status = ReportStatus.FAILED
            report.error_message = str(e)
            report.generation_completed = datetime.utcnow()
            if report.generation_started:
                report.generation_duration_seconds = int(
                    (report.generation_completed - report.generation_started).total_seconds()
                )
            
            self.db.commit()
            
            logger.error(f"Report generation failed for {report.name}: {str(e)}")
    
    async def _generate_from_template(self, report: Report) -> str:
        """Generate report from template"""
        
        template = report.template
        if not template:
            raise ValueError("Template not found")
        
        # Get template
        template_obj = self.jinja_env.get_template(f"{template.id}.html")
        
        # Gather data
        context = await self._gather_report_data(report)
        
        # Add common context
        context.update({
            'report': {
                'name': report.name,
                'title': report.title,
                'generated_at': datetime.utcnow(),
                'generated_by': report.created_by
            },
            'parameters': report.parameters or {}
        })
        
        # Render template
        html_content = template_obj.render(**context)
        
        return html_content
    
    async def _gather_report_data(self, report: Report) -> Dict[str, Any]:
        """Gather data for report generation"""
        
        context = {}
        
        # This would integrate with other services to fetch data
        # For now, using mock data
        
        if 'experiments' in (report.data_sources or []):
            context['experiments'] = await self._get_experiment_data(report.filters)
        
        if 'models' in (report.data_sources or []):
            context['models'] = await self._get_model_data(report.filters)
        
        if 'datasets' in (report.data_sources or []):
            context['datasets'] = await self._get_dataset_data(report.filters)
        
        if 'usage' in (report.data_sources or []):
            context['usage'] = await self._get_usage_data(report.filters)
        
        # Generate charts
        context['charts'] = await self._generate_charts(context, report)
        
        return context
    
    async def _convert_to_format(self, html_content: str, report: Report) -> Path:
        """Convert HTML content to specified format"""
        
        report_dir = self.storage_path / str(report.id)
        report_dir.mkdir(exist_ok=True)
        
        if report.format == ReportFormat.HTML:
            file_path = report_dir / f"{report.name}.html"
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
        elif report.format == ReportFormat.PDF:
            file_path = report_dir / f"{report.name}.pdf"
            
            # Use WeasyPrint for better CSS support
            html_doc = HTML(string=html_content)
            css_styles = CSS(string="""
                @page {
                    size: A4;
                    margin: 1in;
                }
                body {
                    font-family: 'Arial', sans-serif;
                    font-size: 12px;
                    line-height: 1.4;
                }
                .page-break {
                    page-break-after: always;
                }
                .chart-container {
                    text-align: center;
                    margin: 20px 0;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }
                th, td {
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }
                th {
                    background-color: #f2f2f2;
                    font-weight: bold;
                }
            """)
            
            html_doc.write_pdf(str(file_path), stylesheets=[css_styles])
            
        elif report.format == ReportFormat.EXCEL:
            # Convert data to Excel format
            file_path = report_dir / f"{report.name}.xlsx"
            await self._export_to_excel(report, file_path)
            
        elif report.format == ReportFormat.JSON:
            # Export as structured JSON
            file_path = report_dir / f"{report.name}.json"
            data = await self._gather_report_data(report)
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        else:
            raise ValueError(f"Unsupported format: {report.format}")
        
        return file_path
    
    async def get_reports(self, user_id: str, skip: int = 0, limit: int = 100, filters: Optional[Dict[str, Any]] = None) -> List[Report]:
        """Get reports with filtering"""
        
        query = self.db.query(Report).filter(
            (Report.created_by == user_id) |
            (Report.organization_id == self._get_user_org(user_id))
        )
        
        if filters:
            if filters.get('status'):
                query = query.filter(Report.status == filters['status'])
            if filters.get('report_type'):
                query = query.filter(Report.report_type == filters['report_type'])
            if filters.get('format'):
                query = query.filter(Report.format == filters['format'])
        
        return query.order_by(desc(Report.created_at)).offset(skip).limit(limit).all()
    
    async def get_templates(self, user_id: str, skip: int = 0, limit: int = 100) -> List[ReportTemplate]:
        """Get report templates"""
        
        query = self.db.query(ReportTemplate).filter(
            (ReportTemplate.created_by == user_id) |
            (ReportTemplate.is_public == True) |
            (ReportTemplate.organization_id == self._get_user_org(user_id))
        ).filter(ReportTemplate.is_active == True)
        
        return query.order_by(desc(ReportTemplate.created_at)).offset(skip).limit(limit).all()
    
    # Built-in report generators
    async def _generate_experiment_summary(self, report: Report) -> str:
        """Generate experiment summary report"""
        
        experiments_data = await self._get_experiment_data(report.filters)
        
        template_content = """
        <html>
        <head>
            <title>{{ report.title }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { text-align: center; border-bottom: 2px solid #333; padding-bottom: 20px; }
                .section { margin: 30px 0; }
                .metrics { display: flex; justify-content: space-around; margin: 20px 0; }
                .metric-box { text-align: center; padding: 20px; background: #f5f5f5; border-radius: 8px; }
                table { width: 100%; border-collapse: collapse; margin-top: 20px; }
                th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{{ report.title }}</h1>
                <p>Generated on {{ report.generated_at.strftime('%Y-%m-%d %H:%M:%S') }}</p>
            </div>
            
            <div class="section">
                <h2>Summary</h2>
                <div class="metrics">
                    <div class="metric-box">
                        <h3>{{ experiments|length }}</h3>
                        <p>Total Experiments</p>
                    </div>
                    <div class="metric-box">
                        <h3>{{ experiments|selectattr("status", "equalto", "completed")|list|length }}</h3>
                        <p>Completed</p>
                    </div>
                    <div class="metric-box">
                        <h3>{{ experiments|map(attribute="total_runs")|sum }}</h3>
                        <p>Total Runs</p>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Experiment Details</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Name</th>
                            <th>Type</th>
                            <th>Status</th>
                            <th>Total Runs</th>
                            <th>Success Rate</th>
                            <th>Best Score</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for exp in experiments %}
                        <tr>
                            <td>{{ exp.name }}</td>
                            <td>{{ exp.experiment_type }}</td>
                            <td>{{ exp.status }}</td>
                            <td>{{ exp.total_runs }}</td>
                            <td>{{ "%.1f%%"|format(exp.successful_runs / exp.total_runs * 100) if exp.total_runs > 0 else "N/A" }}</td>
                            <td>{{ "%.3f"|format(exp.best_metric_value) if exp.best_metric_value else "N/A" }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
            
            {% if charts.experiment_timeline %}
            <div class="section">
                <h2>Timeline</h2>
                <div class="chart-container">
                    {{ charts.experiment_timeline | safe }}
                </div>
            </div>
            {% endif %}
        </body>
        </html>
        """
        
        template = Template(template_content)
        context = {
            'report': {
                'title': report.title,
                'generated_at': datetime.utcnow()
            },
            'experiments': experiments_data,
            'charts': await self._generate_charts({'experiments': experiments_data}, report)
        }
        
        return template.render(**context)
    
    async def _generate_model_performance(self, report: Report) -> str:
        """Generate model performance report"""
        
        models_data = await self._get_model_data(report.filters)
        
        template_content = """
        <html>
        <head>
            <title>Model Performance Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { text-align: center; border-bottom: 2px solid #333; padding-bottom: 20px; }
                .section { margin: 30px 0; }
                .performance-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
                .model-card { border: 1px solid #ddd; border-radius: 8px; padding: 20px; background: #f9f9f9; }
                .metric { display: flex; justify-content: space-between; margin: 10px 0; }
                .metric-value { font-weight: bold; color: #2c3e50; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Model Performance Report</h1>
                <p>Generated on {{ report.generated_at.strftime('%Y-%m-%d %H:%M:%S') }}</p>
            </div>
            
            <div class="section">
                <h2>Performance Overview</h2>
                <div class="performance-grid">
                    {% for model in models %}
                    <div class="model-card">
                        <h3>{{ model.name }}</h3>
                        <div class="metric">
                            <span>Status:</span>
                            <span class="metric-value">{{ model.status }}</span>
                        </div>
                        <div class="metric">
                            <span>Algorithm:</span>
                            <span class="metric-value">{{ model.algorithm }}</span>
                        </div>
                        {% if model.metrics %}
                        {% for key, value in model.metrics.items() %}
                        <div class="metric">
                            <span>{{ key.title() }}:</span>
                            <span class="metric-value">{{ "%.3f"|format(value) if value is number else value }}</span>
                        </div>
                        {% endfor %}
                        {% endif %}
                    </div>
                    {% endfor %}
                </div>
            </div>
            
            {% if charts.performance_comparison %}
            <div class="section">
                <h2>Performance Comparison</h2>
                <div class="chart-container">
                    {{ charts.performance_comparison | safe }}
                </div>
            </div>
            {% endif %}
        </body>
        </html>
        """
        
        template = Template(template_content)
        context = {
            'report': {
                'title': report.title,
                'generated_at': datetime.utcnow()
            },
            'models': models_data,
            'charts': await self._generate_charts({'models': models_data}, report)
        }
        
        return template.render(**context)
    
    async def _generate_data_quality(self, report: Report) -> str:
        """Generate data quality report"""
        
        datasets_data = await self._get_dataset_data(report.filters)
        
        # Mock data quality template
        return "<html><body><h1>Data Quality Report</h1></body></html>"
    
    async def _generate_usage_analytics(self, report: Report) -> str:
        """Generate usage analytics report"""
        
        usage_data = await self._get_usage_data(report.filters)
        
        # Mock usage analytics template
        return "<html><body><h1>Usage Analytics Report</h1></body></html>"
    
    async def _generate_compliance_audit(self, report: Report) -> str:
        """Generate compliance audit report"""
        
        # Mock compliance audit template
        return "<html><body><h1>Compliance Audit Report</h1></body></html>"
    
    async def _generate_executive_summary(self, report: Report) -> str:
        """Generate executive summary report"""
        
        # Gather all relevant data
        experiments_data = await self._get_experiment_data(report.filters)
        models_data = await self._get_model_data(report.filters)
        usage_data = await self._get_usage_data(report.filters)
        
        # Mock executive summary template
        return "<html><body><h1>Executive Summary</h1></body></html>"
    
    # Data gathering methods (mock implementations)
    async def _get_experiment_data(self, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Get experiment data (mock)"""
        
        # In real implementation, this would query the experiments API
        return [
            {
                'id': str(uuid.uuid4()),
                'name': 'ML Model Training v1',
                'experiment_type': 'classification',
                'status': 'completed',
                'total_runs': 15,
                'successful_runs': 13,
                'best_metric_value': 0.952,
                'created_at': datetime.utcnow() - timedelta(days=5)
            },
            {
                'id': str(uuid.uuid4()),
                'name': 'Hyperparameter Tuning',
                'experiment_type': 'optimization',
                'status': 'completed',
                'total_runs': 50,
                'successful_runs': 48,
                'best_metric_value': 0.967,
                'created_at': datetime.utcnow() - timedelta(days=3)
            }
        ]
    
    async def _get_model_data(self, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Get model data (mock)"""
        
        return [
            {
                'id': str(uuid.uuid4()),
                'name': 'RandomForest Classifier',
                'algorithm': 'random_forest',
                'status': 'deployed',
                'metrics': {
                    'accuracy': 0.952,
                    'precision': 0.948,
                    'recall': 0.945,
                    'f1_score': 0.946
                }
            },
            {
                'id': str(uuid.uuid4()),
                'name': 'XGBoost Regressor',
                'algorithm': 'xgboost',
                'status': 'trained',
                'metrics': {
                    'rmse': 0.123,
                    'mae': 0.089,
                    'r2_score': 0.876
                }
            }
        ]
    
    async def _get_dataset_data(self, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Get dataset data (mock)"""
        
        return [
            {
                'id': str(uuid.uuid4()),
                'name': 'Customer Data',
                'row_count': 10000,
                'column_count': 25,
                'missing_values': 150,
                'quality_score': 0.85
            }
        ]
    
    async def _get_usage_data(self, filters: Optional[Dict] = None) -> Dict[str, Any]:
        """Get usage analytics data (mock)"""
        
        return {
            'total_users': 45,
            'active_users': 32,
            'experiments_created': 127,
            'models_deployed': 23,
            'reports_generated': 89
        }
    
    async def _generate_charts(self, context: Dict[str, Any], report: Report) -> Dict[str, str]:
        """Generate charts for the report"""
        
        charts = {}
        
        # Example: Experiment timeline chart
        if 'experiments' in context:
            experiments = context['experiments']
            
            # Create timeline chart
            fig = go.Figure()
            
            dates = [exp['created_at'] for exp in experiments]
            names = [exp['name'] for exp in experiments]
            scores = [exp.get('best_metric_value', 0) for exp in experiments]
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=scores,
                mode='markers+lines',
                text=names,
                hovertemplate='<b>%{text}</b><br>Score: %{y:.3f}<br>Date: %{x}<extra></extra>'
            ))
            
            fig.update_layout(
                title='Experiment Performance Timeline',
                xaxis_title='Date',
                yaxis_title='Best Score',
                height=400
            )
            
            charts['experiment_timeline'] = fig.to_html(include_plotlyjs='inline', div_id="timeline")
        
        # Example: Model performance comparison
        if 'models' in context:
            models = context['models']
            
            # Create performance comparison chart
            model_names = [model['name'] for model in models]
            accuracies = [model['metrics'].get('accuracy', model['metrics'].get('r2_score', 0)) for model in models]
            
            fig = go.Figure(data=[
                go.Bar(x=model_names, y=accuracies, text=[f'{acc:.3f}' for acc in accuracies], textposition='auto')
            ])
            
            fig.update_layout(
                title='Model Performance Comparison',
                xaxis_title='Model',
                yaxis_title='Score',
                height=400
            )
            
            charts['performance_comparison'] = fig.to_html(include_plotlyjs='inline', div_id="performance")
        
        return charts
    
    async def _export_to_excel(self, report: Report, file_path: Path):
        """Export report data to Excel"""
        
        # Gather data
        data = await self._gather_report_data(report)
        
        with pd.ExcelWriter(str(file_path), engine='openpyxl') as writer:
            
            # Export experiments data
            if 'experiments' in data:
                df_exp = pd.DataFrame(data['experiments'])
                df_exp.to_excel(writer, sheet_name='Experiments', index=False)
            
            # Export models data
            if 'models' in data:
                df_models = pd.DataFrame(data['models'])
                df_models.to_excel(writer, sheet_name='Models', index=False)
            
            # Export datasets data
            if 'datasets' in data:
                df_datasets = pd.DataFrame(data['datasets'])
                df_datasets.to_excel(writer, sheet_name='Datasets', index=False)
    
    async def _send_report_email(self, report: Report):
        """Send report via email"""
        
        # Mock email sending
        logger.info(f"Sending report {report.name} to {report.email_recipients}")
        
        report.is_emailed = True
        report.emailed_at = datetime.utcnow()
        self.db.commit()
    
    # Helper methods
    def _generate_share_token(self) -> str:
        """Generate sharing token"""
        return secrets.token_urlsafe(32)
    
    def _get_user_org(self, user_id: str) -> Optional[str]:
        """Get user's organization ID"""
        return "default_org"

# API Endpoints
@router.post("/templates/", response_model=ReportTemplateResponse)
async def create_template(
    template_data: ReportTemplateCreate,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Create a new report template"""
    
    service = ReportGenerationService(db)
    template = await service.create_template(template_data, current_user)
    return template

@router.get("/templates/", response_model=List[ReportTemplateResponse])
async def get_templates(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Get report templates"""
    
    service = ReportGenerationService(db)
    templates = await service.get_templates(current_user, skip, limit)
    return templates

@router.post("/", response_model=ReportResponse)
async def generate_report(
    report_data: ReportCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Generate a new report"""
    
    service = ReportGenerationService(db)
    report = await service.generate_report(report_data, current_user, background_tasks)
    return report

@router.get("/", response_model=List[ReportResponse])
async def get_reports(
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None,
    report_type: Optional[str] = None,
    format: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Get reports with filtering"""
    
    filters = {}
    if status:
        filters['status'] = status
    if report_type:
        filters['report_type'] = report_type
    if format:
        filters['format'] = format
    
    service = ReportGenerationService(db)
    reports = await service.get_reports(current_user, skip, limit, filters)
    return reports

@router.get("/{report_id}", response_model=ReportResponse)
async def get_report(
    report_id: str,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Get a specific report"""
    
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    # Check access permissions
    if report.created_by != current_user and not report.is_public:
        raise HTTPException(status_code=403, detail="Permission denied")
    
    return report

@router.get("/{report_id}/download")
async def download_report(
    report_id: str,
    token: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: Optional[str] = Depends(get_current_user)
):
    """Download report file"""
    
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    # Check access permissions
    if not current_user and not (token and report.share_token == token):
        raise HTTPException(status_code=401, detail="Authentication required")
    
    if current_user and report.created_by != current_user and not report.is_public:
        raise HTTPException(status_code=403, detail="Permission denied")
    
    # Check if expired
    if report.expires_at and report.expires_at < datetime.utcnow():
        raise HTTPException(status_code=410, detail="Report has expired")
    
    if not report.file_path or not Path(report.file_path).exists():
        raise HTTPException(status_code=404, detail="Report file not found")
    
    # Update download count
    report.download_count += 1
    db.commit()
    
    return FileResponse(
        path=report.file_path,
        filename=f"{report.name}.{report.format}",
        media_type='application/octet-stream'
    )

@router.delete("/{report_id}")
async def delete_report(
    report_id: str,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Delete a report"""
    
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    if report.created_by != current_user:
        raise HTTPException(status_code=403, detail="Permission denied")
    
    # Delete file
    if report.file_path and Path(report.file_path).exists():
        report_dir = Path(report.file_path).parent
        if report_dir.exists():
            shutil.rmtree(report_dir)
    
    # Delete database record
    db.delete(report)
    db.commit()
    
    return {"message": "Report deleted successfully"}

@router.post("/{report_id}/share")
async def share_report(
    report_id: str,
    expires_in_hours: int = Query(default=24),
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Create a public sharing link for the report"""
    
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    if report.created_by != current_user:
        raise HTTPException(status_code=403, detail="Permission denied")
    
    # Generate sharing token and set expiration
    service = ReportGenerationService(db)
    report.share_token = service._generate_share_token()
    report.expires_at = datetime.utcnow() + timedelta(hours=expires_in_hours)
    report.is_public = True
    
    db.commit()
    
    share_url = f"/api/v1/reports/{report_id}/download?token={report.share_token}"
    
    return {
        "share_url": share_url,
        "expires_at": report.expires_at,
        "token": report.share_token
    }