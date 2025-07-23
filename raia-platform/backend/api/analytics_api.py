# Analytics API for Usage Metrics and Insights
import os
import uuid
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from enum import Enum

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, validator
from sqlalchemy.orm import Session
from sqlalchemy import Column, String, DateTime, Text, Boolean, JSON, UUID as PG_UUID, Integer, Float, ForeignKey, func, desc, and_
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

# Analytics and data processing
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder

# Time series analysis
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

Base = declarative_base()
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])

# Enums
class MetricType(str, Enum):
    USER_ACTIVITY = "user_activity"
    MODEL_PERFORMANCE = "model_performance"
    SYSTEM_HEALTH = "system_health"
    RESOURCE_USAGE = "resource_usage"
    BUSINESS_KPI = "business_kpi"
    CUSTOM = "custom"

class AggregationType(str, Enum):
    COUNT = "count"
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    STDDEV = "stddev"
    PERCENTILE = "percentile"

class TimeGranularity(str, Enum):
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"

class AnalyticsType(str, Enum):
    DESCRIPTIVE = "descriptive"
    DIAGNOSTIC = "diagnostic"
    PREDICTIVE = "predictive"
    PRESCRIPTIVE = "prescriptive"

# Database Models
class AnalyticsEvent(Base):
    """Analytics event data"""
    __tablename__ = "analytics_events"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Event identification
    event_name = Column(String(255), nullable=False, index=True)
    event_category = Column(String(100), index=True)
    event_type = Column(String(100), index=True)
    
    # Event data
    properties = Column(JSON)  # Event properties
    numerical_values = Column(JSON)  # Numerical metrics for aggregation
    
    # Context
    user_id = Column(String(255), index=True)
    session_id = Column(String(255), index=True)
    organization_id = Column(String(255), index=True)
    
    # Resource context
    resource_type = Column(String(100))  # model, experiment, dataset, etc.
    resource_id = Column(String(255))
    
    # Technical context
    ip_address = Column(String(45))
    user_agent = Column(String(1000))
    platform = Column(String(100))
    device_type = Column(String(100))
    
    # Geographical context
    country = Column(String(100))
    region = Column(String(100))
    city = Column(String(100))
    
    # Temporal context
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    date = Column(DateTime, index=True)  # Date without time for daily aggregations
    hour = Column(Integer, index=True)  # Hour of day (0-23)
    day_of_week = Column(Integer, index=True)  # Day of week (0-6)
    
    # Metadata
    version = Column(String(50))  # Application version
    experiment_variant = Column(String(100))  # A/B testing variant

class AnalyticsDashboard(Base):
    """Custom analytics dashboards"""
    __tablename__ = "analytics_dashboards"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    
    # Dashboard configuration
    config = Column(JSON, nullable=False)  # Dashboard layout and widgets
    filters = Column(JSON)  # Default filters
    refresh_interval = Column(Integer, default=300)  # Auto-refresh interval in seconds
    
    # Access control
    is_public = Column(Boolean, default=False)
    shared_with = Column(JSON)  # List of user IDs with access
    
    # Usage tracking
    view_count = Column(Integer, default=0)
    last_viewed = Column(DateTime)
    
    # Ownership
    created_by = Column(String(255), nullable=False)
    organization_id = Column(String(255))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class AnalyticsInsight(Base):
    """AI-generated insights from analytics data"""
    __tablename__ = "analytics_insights"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Insight metadata
    insight_type = Column(String(100), nullable=False)  # anomaly, trend, correlation, etc.
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    
    # Insight details
    confidence_score = Column(Float)  # 0-1 confidence in the insight
    importance_score = Column(Float)  # Business importance (0-1)
    category = Column(String(100))  # performance, usage, health, etc.
    
    # Data context
    data_source = Column(String(255))  # What data was analyzed
    time_period = Column(JSON)  # Time range of analysis
    filters_applied = Column(JSON)  # Any filters used
    
    # Supporting data
    metrics = Column(JSON)  # Key metrics supporting the insight
    visualizations = Column(JSON)  # Chart configurations
    recommendations = Column(JSON)  # Action recommendations
    
    # Status
    is_active = Column(Boolean, default=True)
    is_resolved = Column(Boolean, default=False)
    acknowledged_by = Column(String(255))
    acknowledged_at = Column(DateTime)
    
    # Ownership
    created_by = Column(String(255))  # AI or user ID
    organization_id = Column(String(255))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)

# Pydantic Models
class AnalyticsEventCreate(BaseModel):
    event_name: str
    event_category: Optional[str] = None
    event_type: Optional[str] = None
    properties: Optional[Dict[str, Any]] = {}
    numerical_values: Optional[Dict[str, float]] = {}
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None

class DashboardCreate(BaseModel):
    name: str
    description: Optional[str] = None
    config: Dict[str, Any]
    filters: Optional[Dict[str, Any]] = {}
    refresh_interval: int = 300
    is_public: bool = False
    shared_with: Optional[List[str]] = []

class DashboardResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    config: Dict[str, Any]
    view_count: int
    last_viewed: Optional[datetime]
    created_by: str
    created_at: datetime
    
    class Config:
        orm_mode = True

class AnalyticsQuery(BaseModel):
    metric_type: Optional[MetricType] = None
    event_names: Optional[List[str]] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    granularity: TimeGranularity = TimeGranularity.DAY
    aggregation: AggregationType = AggregationType.COUNT
    filters: Optional[Dict[str, Any]] = {}
    group_by: Optional[List[str]] = []
    limit: Optional[int] = 1000

class InsightResponse(BaseModel):
    id: str
    insight_type: str
    title: str
    description: str
    confidence_score: Optional[float]
    importance_score: Optional[float]
    category: Optional[str]
    metrics: Optional[Dict[str, Any]]
    recommendations: Optional[Dict[str, Any]]
    created_at: datetime
    
    class Config:
        orm_mode = True

# Analytics Service
class AnalyticsService:
    """Service for analytics data collection and analysis"""
    
    def __init__(self, db: Session):
        self.db = db
        
        # Cache for frequently accessed data
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Insight generators
        self.insight_generators = {
            'anomaly_detection': self._generate_anomaly_insights,
            'trend_analysis': self._generate_trend_insights,
            'correlation_analysis': self._generate_correlation_insights,
            'usage_patterns': self._generate_usage_pattern_insights,
            'performance_insights': self._generate_performance_insights
        }
    
    async def track_event(self, event_data: AnalyticsEventCreate, user_id: Optional[str] = None, request_context: Optional[Dict] = None) -> Dict[str, Any]:
        """Track an analytics event"""
        
        # Enrich event data with context
        enriched_event = self._enrich_event_data(event_data, user_id, request_context)
        
        # Create analytics event
        event = AnalyticsEvent(**enriched_event)
        
        self.db.add(event)
        self.db.commit()
        self.db.refresh(event)
        
        # Trigger real-time insights if needed
        await self._check_real_time_insights(event)
        
        return {'success': True, 'event_id': str(event.id)}
    
    async def query_analytics(self, query: AnalyticsQuery, user_id: str) -> Dict[str, Any]:
        """Query analytics data"""
        
        # Build base query
        base_query = self.db.query(AnalyticsEvent).filter(
            AnalyticsEvent.organization_id == self._get_user_org(user_id)
        )
        
        # Apply filters
        if query.event_names:
            base_query = base_query.filter(AnalyticsEvent.event_name.in_(query.event_names))
        
        if query.start_date:
            base_query = base_query.filter(AnalyticsEvent.timestamp >= query.start_date)
        
        if query.end_date:
            base_query = base_query.filter(AnalyticsEvent.timestamp <= query.end_date)
        
        # Apply custom filters
        for key, value in (query.filters or {}).items():
            if key == 'user_id':
                base_query = base_query.filter(AnalyticsEvent.user_id == value)
            elif key == 'resource_type':
                base_query = base_query.filter(AnalyticsEvent.resource_type == value)
            elif key == 'event_category':
                base_query = base_query.filter(AnalyticsEvent.event_category == value)
        
        # Execute query and process results
        events = base_query.limit(query.limit).all()
        
        # Aggregate data based on query parameters
        aggregated_data = self._aggregate_events(events, query)
        
        return {
            'data': aggregated_data,
            'total_events': len(events),
            'query_params': query.dict()
        }
    
    async def get_dashboard_data(self, dashboard_id: str, user_id: str, time_range: Optional[Dict[str, datetime]] = None) -> Dict[str, Any]:
        """Get data for a dashboard"""
        
        dashboard = self.db.query(AnalyticsDashboard).filter(
            AnalyticsDashboard.id == dashboard_id
        ).first()
        
        if not dashboard:
            raise HTTPException(status_code=404, detail="Dashboard not found")
        
        # Check access permissions
        if not self._can_access_dashboard(dashboard, user_id):
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Update view count
        dashboard.view_count += 1
        dashboard.last_viewed = datetime.utcnow()
        self.db.commit()
        
        # Process dashboard configuration and get data for each widget
        dashboard_data = await self._process_dashboard_config(dashboard.config, user_id, time_range)
        
        return {
            'dashboard': {
                'id': str(dashboard.id),
                'name': dashboard.name,
                'description': dashboard.description,
                'config': dashboard.config
            },
            'data': dashboard_data,
            'generated_at': datetime.utcnow().isoformat()
        }
    
    async def generate_insights(self, user_id: str, insight_types: Optional[List[str]] = None, time_range: Optional[Dict[str, datetime]] = None) -> List[AnalyticsInsight]:
        """Generate analytics insights"""
        
        insights = []
        
        # Default time range: last 30 days
        if not time_range:
            time_range = {
                'start': datetime.utcnow() - timedelta(days=30),
                'end': datetime.utcnow()
            }
        
        # Get base data for analysis
        base_data = await self._get_base_analytics_data(user_id, time_range)
        
        # Generate insights based on requested types
        generators_to_run = insight_types or list(self.insight_generators.keys())
        
        for generator_name in generators_to_run:
            generator = self.insight_generators.get(generator_name)
            if generator:
                try:
                    generator_insights = await generator(base_data, user_id, time_range)
                    insights.extend(generator_insights)
                except Exception as e:
                    logger.error(f"Error generating {generator_name} insights: {str(e)}")
        
        # Sort insights by importance and confidence
        insights.sort(key=lambda x: (x.importance_score or 0) * (x.confidence_score or 0), reverse=True)
        
        # Save insights to database
        for insight in insights[:10]:  # Limit to top 10 insights
            self.db.add(insight)
        
        self.db.commit()
        
        return insights
    
    async def get_usage_summary(self, user_id: str, time_range: Optional[Dict[str, datetime]] = None) -> Dict[str, Any]:
        """Get usage summary statistics"""
        
        if not time_range:
            time_range = {
                'start': datetime.utcnow() - timedelta(days=30),
                'end': datetime.utcnow()
            }
        
        org_id = self._get_user_org(user_id)
        
        # Base query for the time range
        base_query = self.db.query(AnalyticsEvent).filter(
            AnalyticsEvent.organization_id == org_id,
            AnalyticsEvent.timestamp >= time_range['start'],
            AnalyticsEvent.timestamp <= time_range['end']
        )
        
        # Total events
        total_events = base_query.count()
        
        # Unique users
        unique_users = base_query.filter(AnalyticsEvent.user_id.isnot(None)).distinct(AnalyticsEvent.user_id).count()
        
        # Active sessions
        unique_sessions = base_query.filter(AnalyticsEvent.session_id.isnot(None)).distinct(AnalyticsEvent.session_id).count()
        
        # Top events
        top_events = base_query.with_entities(
            AnalyticsEvent.event_name,
            func.count(AnalyticsEvent.id).label('count')
        ).group_by(AnalyticsEvent.event_name).order_by(desc('count')).limit(10).all()
        
        # Daily trend
        daily_trend = base_query.with_entities(
            func.date(AnalyticsEvent.timestamp).label('date'),
            func.count(AnalyticsEvent.id).label('count')
        ).group_by(func.date(AnalyticsEvent.timestamp)).order_by('date').all()
        
        # Resource usage
        resource_usage = base_query.filter(AnalyticsEvent.resource_type.isnot(None)).with_entities(
            AnalyticsEvent.resource_type,
            func.count(AnalyticsEvent.id).label('count')
        ).group_by(AnalyticsEvent.resource_type).order_by(desc('count')).all()
        
        return {
            'summary': {
                'total_events': total_events,
                'unique_users': unique_users,
                'unique_sessions': unique_sessions,
                'time_range': time_range
            },
            'top_events': [{'event_name': name, 'count': count} for name, count in top_events],
            'daily_trend': [{'date': date.isoformat(), 'count': count} for date, count in daily_trend],
            'resource_usage': [{'resource_type': rt, 'count': count} for rt, count in resource_usage]
        }
    
    async def get_user_analytics(self, target_user_id: str, requesting_user_id: str, time_range: Optional[Dict[str, datetime]] = None) -> Dict[str, Any]:
        """Get analytics for a specific user"""
        
        # Permission check - users can view their own analytics or admins can view any
        if target_user_id != requesting_user_id and not self._is_admin(requesting_user_id):
            raise HTTPException(status_code=403, detail="Permission denied")
        
        if not time_range:
            time_range = {
                'start': datetime.utcnow() - timedelta(days=30),
                'end': datetime.utcnow()
            }
        
        # User activity query
        user_query = self.db.query(AnalyticsEvent).filter(
            AnalyticsEvent.user_id == target_user_id,
            AnalyticsEvent.timestamp >= time_range['start'],
            AnalyticsEvent.timestamp <= time_range['end']
        )
        
        # Activity summary
        total_actions = user_query.count()
        unique_sessions = user_query.distinct(AnalyticsEvent.session_id).count()
        
        # Most active days
        daily_activity = user_query.with_entities(
            func.date(AnalyticsEvent.timestamp).label('date'),
            func.count(AnalyticsEvent.id).label('count')
        ).group_by(func.date(AnalyticsEvent.timestamp)).order_by(desc('count')).limit(10).all()
        
        # Activity by hour of day
        hourly_activity = user_query.with_entities(
            AnalyticsEvent.hour,
            func.count(AnalyticsEvent.id).label('count')
        ).group_by(AnalyticsEvent.hour).order_by(AnalyticsEvent.hour).all()
        
        # Most used features
        feature_usage = user_query.with_entities(
            AnalyticsEvent.event_name,
            func.count(AnalyticsEvent.id).label('count')
        ).group_by(AnalyticsEvent.event_name).order_by(desc('count')).limit(10).all()
        
        # Device and platform info
        device_usage = user_query.filter(AnalyticsEvent.device_type.isnot(None)).with_entities(
            AnalyticsEvent.device_type,
            func.count(AnalyticsEvent.id).label('count')
        ).group_by(AnalyticsEvent.device_type).all()
        
        return {
            'user_id': target_user_id,
            'time_range': time_range,
            'summary': {
                'total_actions': total_actions,
                'unique_sessions': unique_sessions,
                'avg_actions_per_session': total_actions / unique_sessions if unique_sessions > 0 else 0
            },
            'daily_activity': [{'date': date.isoformat(), 'count': count} for date, count in daily_activity],
            'hourly_pattern': [{'hour': hour, 'count': count} for hour, count in hourly_activity],
            'feature_usage': [{'feature': name, 'count': count} for name, count in feature_usage],
            'device_usage': [{'device': device, 'count': count} for device, count in device_usage]
        }
    
    # Private helper methods
    def _enrich_event_data(self, event_data: AnalyticsEventCreate, user_id: Optional[str], request_context: Optional[Dict]) -> Dict[str, Any]:
        """Enrich event data with additional context"""
        
        now = datetime.utcnow()
        
        enriched = {
            'event_name': event_data.event_name,
            'event_category': event_data.event_category,
            'event_type': event_data.event_type,
            'properties': event_data.properties,
            'numerical_values': event_data.numerical_values,
            'resource_type': event_data.resource_type,
            'resource_id': event_data.resource_id,
            'user_id': user_id or event_data.user_id,
            'session_id': event_data.session_id,
            'organization_id': self._get_user_org(user_id) if user_id else None,
            'timestamp': now,
            'date': now.date(),
            'hour': now.hour,
            'day_of_week': now.weekday()
        }
        
        # Add request context if available
        if request_context:
            enriched.update({
                'ip_address': request_context.get('ip_address'),
                'user_agent': request_context.get('user_agent'),
                'platform': self._extract_platform(request_context.get('user_agent', '')),
                'device_type': self._extract_device_type(request_context.get('user_agent', ''))
            })
        
        return enriched
    
    def _aggregate_events(self, events: List[AnalyticsEvent], query: AnalyticsQuery) -> Dict[str, Any]:
        """Aggregate events based on query parameters"""
        
        if not events:
            return {'series': [], 'totals': {}}
        
        # Convert to DataFrame for easier processing
        event_data = []
        for event in events:
            event_dict = {
                'timestamp': event.timestamp,
                'event_name': event.event_name,
                'user_id': event.user_id,
                'resource_type': event.resource_type,
                'date': event.date,
                'hour': event.hour
            }
            
            # Add numerical values
            if event.numerical_values:
                event_dict.update(event.numerical_values)
            
            event_data.append(event_dict)
        
        df = pd.DataFrame(event_data)
        
        # Group by specified columns and time granularity
        group_cols = []
        
        # Add time grouping
        if query.granularity == TimeGranularity.HOUR:
            df['time_group'] = df['timestamp'].dt.floor('H')
            group_cols.append('time_group')
        elif query.granularity == TimeGranularity.DAY:
            df['time_group'] = df['date']
            group_cols.append('time_group')
        elif query.granularity == TimeGranularity.WEEK:
            df['time_group'] = df['timestamp'].dt.to_period('W').dt.start_time
            group_cols.append('time_group')
        elif query.granularity == TimeGranularity.MONTH:
            df['time_group'] = df['timestamp'].dt.to_period('M').dt.start_time
            group_cols.append('time_group')
        
        # Add custom group by columns
        if query.group_by:
            for col in query.group_by:
                if col in df.columns:
                    group_cols.append(col)
        
        if not group_cols:
            group_cols = ['event_name']
        
        # Perform aggregation
        if query.aggregation == AggregationType.COUNT:
            aggregated = df.groupby(group_cols).size().reset_index(name='value')
        else:
            # For other aggregations, we need a numerical column
            numerical_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
            if numerical_cols:
                agg_col = numerical_cols[0]  # Use first numerical column
                
                if query.aggregation == AggregationType.SUM:
                    aggregated = df.groupby(group_cols)[agg_col].sum().reset_index(name='value')
                elif query.aggregation == AggregationType.AVG:
                    aggregated = df.groupby(group_cols)[agg_col].mean().reset_index(name='value')
                elif query.aggregation == AggregationType.MIN:
                    aggregated = df.groupby(group_cols)[agg_col].min().reset_index(name='value')
                elif query.aggregation == AggregationType.MAX:
                    aggregated = df.groupby(group_cols)[agg_col].max().reset_index(name='value')
                else:
                    aggregated = df.groupby(group_cols).size().reset_index(name='value')
            else:
                aggregated = df.groupby(group_cols).size().reset_index(name='value')
        
        # Convert back to list of dictionaries
        result_data = aggregated.to_dict('records')
        
        # Calculate totals
        totals = {
            'total_events': len(events),
            'unique_users': df['user_id'].nunique() if 'user_id' in df.columns else 0,
            'date_range': {
                'start': df['timestamp'].min().isoformat() if len(df) > 0 else None,
                'end': df['timestamp'].max().isoformat() if len(df) > 0 else None
            }
        }
        
        return {
            'series': result_data,
            'totals': totals
        }
    
    async def _process_dashboard_config(self, config: Dict[str, Any], user_id: str, time_range: Optional[Dict[str, datetime]]) -> Dict[str, Any]:
        """Process dashboard configuration and get data for widgets"""
        
        dashboard_data = {}
        
        widgets = config.get('widgets', [])
        
        for widget in widgets:
            widget_id = widget.get('id')
            widget_type = widget.get('type')
            widget_config = widget.get('config', {})
            
            try:
                if widget_type == 'metric':
                    widget_data = await self._get_metric_widget_data(widget_config, user_id, time_range)
                elif widget_type == 'chart':
                    widget_data = await self._get_chart_widget_data(widget_config, user_id, time_range)
                elif widget_type == 'table':
                    widget_data = await self._get_table_widget_data(widget_config, user_id, time_range)
                else:
                    widget_data = {'error': f'Unknown widget type: {widget_type}'}
                
                dashboard_data[widget_id] = widget_data
                
            except Exception as e:
                logger.error(f"Error processing widget {widget_id}: {str(e)}")
                dashboard_data[widget_id] = {'error': str(e)}
        
        return dashboard_data
    
    async def _get_metric_widget_data(self, config: Dict[str, Any], user_id: str, time_range: Optional[Dict[str, datetime]]) -> Dict[str, Any]:
        """Get data for metric widget"""
        
        # Build query based on widget configuration
        query = AnalyticsQuery(
            event_names=config.get('event_names'),
            start_date=time_range['start'] if time_range else None,
            end_date=time_range['end'] if time_range else None,
            aggregation=AggregationType(config.get('aggregation', 'count')),
            filters=config.get('filters', {})
        )
        
        result = await self.query_analytics(query, user_id)
        
        # Calculate the metric value
        series_data = result['data']['series']
        if series_data:
            if query.aggregation == AggregationType.COUNT:
                metric_value = sum(item['value'] for item in series_data)
            else:
                metric_value = series_data[0]['value'] if len(series_data) == 1 else sum(item['value'] for item in series_data)
        else:
            metric_value = 0
        
        return {
            'value': metric_value,
            'format': config.get('format', 'number'),
            'label': config.get('label', 'Metric'),
            'trend': self._calculate_trend(series_data) if len(series_data) > 1 else None
        }
    
    async def _get_chart_widget_data(self, config: Dict[str, Any], user_id: str, time_range: Optional[Dict[str, datetime]]) -> Dict[str, Any]:
        """Get data for chart widget"""
        
        query = AnalyticsQuery(
            event_names=config.get('event_names'),
            start_date=time_range['start'] if time_range else None,
            end_date=time_range['end'] if time_range else None,
            granularity=TimeGranularity(config.get('granularity', 'day')),
            aggregation=AggregationType(config.get('aggregation', 'count')),
            filters=config.get('filters', {}),
            group_by=config.get('group_by', [])
        )
        
        result = await self.query_analytics(query, user_id)
        
        return {
            'data': result['data']['series'],
            'chart_type': config.get('chart_type', 'line'),
            'x_axis': config.get('x_axis', 'time_group'),
            'y_axis': config.get('y_axis', 'value')
        }
    
    async def _get_table_widget_data(self, config: Dict[str, Any], user_id: str, time_range: Optional[Dict[str, datetime]]) -> Dict[str, Any]:
        """Get data for table widget"""
        
        query = AnalyticsQuery(
            event_names=config.get('event_names'),
            start_date=time_range['start'] if time_range else None,
            end_date=time_range['end'] if time_range else None,
            filters=config.get('filters', {}),
            group_by=config.get('group_by', []),
            limit=config.get('limit', 100)
        )
        
        result = await self.query_analytics(query, user_id)
        
        return {
            'rows': result['data']['series'],
            'columns': config.get('columns', []),
            'total_rows': len(result['data']['series'])
        }
    
    # Insight generators
    async def _generate_anomaly_insights(self, base_data: Dict[str, Any], user_id: str, time_range: Dict[str, datetime]) -> List[AnalyticsInsight]:
        """Generate anomaly detection insights"""
        
        insights = []
        
        # Analyze daily event counts for anomalies
        daily_counts = base_data.get('daily_counts', [])
        
        if len(daily_counts) >= 7:  # Need at least a week of data
            values = [item['count'] for item in daily_counts]
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            # Find outliers (values beyond 2 standard deviations)
            outliers = []
            for i, val in enumerate(values):
                if abs(val - mean_val) > 2 * std_val:
                    outliers.append((daily_counts[i]['date'], val, mean_val))
            
            if outliers:
                for date, value, expected in outliers[-3:]:  # Last 3 outliers
                    deviation = ((value - expected) / expected) * 100
                    
                    insight = AnalyticsInsight(
                        insight_type='anomaly',
                        title=f'Unusual Activity on {date}',
                        description=f'Activity was {abs(deviation):.1f}% {"higher" if deviation > 0 else "lower"} than expected ({value} vs {expected:.0f})',
                        confidence_score=min(abs(deviation) / 100, 1.0),
                        importance_score=0.8 if abs(deviation) > 50 else 0.5,
                        category='usage',
                        metrics={'date': date, 'actual': value, 'expected': expected, 'deviation_percent': deviation},
                        organization_id=self._get_user_org(user_id)
                    )
                    insights.append(insight)
        
        return insights
    
    async def _generate_trend_insights(self, base_data: Dict[str, Any], user_id: str, time_range: Dict[str, datetime]) -> List[AnalyticsInsight]:
        """Generate trend analysis insights"""
        
        insights = []
        
        daily_counts = base_data.get('daily_counts', [])
        
        if len(daily_counts) >= 14:  # Need at least 2 weeks
            values = [item['count'] for item in daily_counts]
            
            # Calculate trend using linear regression
            x = np.arange(len(values))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
            
            # Determine trend significance
            if abs(r_value) > 0.5 and p_value < 0.05:  # Significant trend
                trend_direction = 'increasing' if slope > 0 else 'decreasing'
                weekly_change = slope * 7
                
                insight = AnalyticsInsight(
                    insight_type='trend',
                    title=f'Activity is {trend_direction}',
                    description=f'Usage has been {trend_direction} by approximately {abs(weekly_change):.1f} events per week',
                    confidence_score=abs(r_value),
                    importance_score=min(abs(weekly_change) / np.mean(values), 1.0),
                    category='growth',
                    metrics={
                        'slope': slope,
                        'r_squared': r_value ** 2,
                        'weekly_change': weekly_change,
                        'trend_direction': trend_direction
                    },
                    recommendations={
                        'action': 'Monitor trend closely' if trend_direction == 'decreasing' else 'Leverage positive momentum',
                        'priority': 'high' if abs(weekly_change) > np.mean(values) * 0.1 else 'medium'
                    },
                    organization_id=self._get_user_org(user_id)
                )
                insights.append(insight)
        
        return insights
    
    async def _generate_correlation_insights(self, base_data: Dict[str, Any], user_id: str, time_range: Dict[str, datetime]) -> List[AnalyticsInsight]:
        """Generate correlation analysis insights"""
        
        insights = []
        
        # Analyze correlation between different event types
        event_counts = base_data.get('event_type_counts', {})
        
        if len(event_counts) >= 2:
            # Convert to time series for correlation analysis
            # This would require more detailed time series data
            
            # Placeholder insight
            insight = AnalyticsInsight(
                insight_type='correlation',
                title='Feature Usage Correlation',
                description='Some features show strong usage correlation patterns',
                confidence_score=0.6,
                importance_score=0.4,
                category='behavior',
                metrics={'correlations': 'Available in detailed analysis'},
                organization_id=self._get_user_org(user_id)
            )
            insights.append(insight)
        
        return insights
    
    async def _generate_usage_pattern_insights(self, base_data: Dict[str, Any], user_id: str, time_range: Dict[str, datetime]) -> List[AnalyticsInsight]:
        """Generate usage pattern insights"""
        
        insights = []
        
        # Analyze hourly usage patterns
        hourly_counts = base_data.get('hourly_counts', [])
        
        if hourly_counts:
            peak_hours = sorted(hourly_counts, key=lambda x: x['count'], reverse=True)[:3]
            
            insight = AnalyticsInsight(
                insight_type='usage_pattern',
                title='Peak Usage Hours Identified',
                description=f'Highest activity occurs at {peak_hours[0]["hour"]}:00, {peak_hours[1]["hour"]}:00, and {peak_hours[2]["hour"]}:00',
                confidence_score=0.9,
                importance_score=0.6,
                category='behavior',
                metrics={
                    'peak_hours': [(item['hour'], item['count']) for item in peak_hours],
                    'total_peak_usage': sum(item['count'] for item in peak_hours)
                },
                recommendations={
                    'action': 'Schedule maintenance outside peak hours',
                    'optimal_times': [f"{item['hour']}:00" for item in peak_hours]
                },
                organization_id=self._get_user_org(user_id)
            )
            insights.append(insight)
        
        return insights
    
    async def _generate_performance_insights(self, base_data: Dict[str, Any], user_id: str, time_range: Dict[str, datetime]) -> List[AnalyticsInsight]:
        """Generate performance insights"""
        
        insights = []
        
        # Analyze error rates and performance metrics
        error_rate = base_data.get('error_rate', 0)
        
        if error_rate > 0.05:  # More than 5% error rate
            insight = AnalyticsInsight(
                insight_type='performance',
                title='Elevated Error Rate Detected',
                description=f'Error rate is {error_rate:.1%}, which is above the recommended threshold of 5%',
                confidence_score=0.95,
                importance_score=0.9,
                category='health',
                metrics={'error_rate': error_rate, 'threshold': 0.05},
                recommendations={
                    'action': 'Investigate error sources and implement fixes',
                    'priority': 'critical' if error_rate > 0.1 else 'high'
                },
                organization_id=self._get_user_org(user_id)
            )
            insights.append(insight)
        
        return insights
    
    async def _get_base_analytics_data(self, user_id: str, time_range: Dict[str, datetime]) -> Dict[str, Any]:
        """Get base analytics data for insight generation"""
        
        org_id = self._get_user_org(user_id)
        
        base_query = self.db.query(AnalyticsEvent).filter(
            AnalyticsEvent.organization_id == org_id,
            AnalyticsEvent.timestamp >= time_range['start'],
            AnalyticsEvent.timestamp <= time_range['end']
        )
        
        # Daily counts
        daily_counts = base_query.with_entities(
            func.date(AnalyticsEvent.timestamp).label('date'),
            func.count(AnalyticsEvent.id).label('count')
        ).group_by(func.date(AnalyticsEvent.timestamp)).order_by('date').all()
        
        daily_counts_list = [{'date': date.isoformat(), 'count': count} for date, count in daily_counts]
        
        # Hourly counts
        hourly_counts = base_query.with_entities(
            AnalyticsEvent.hour,
            func.count(AnalyticsEvent.id).label('count')
        ).group_by(AnalyticsEvent.hour).order_by(AnalyticsEvent.hour).all()
        
        hourly_counts_list = [{'hour': hour, 'count': count} for hour, count in hourly_counts]
        
        # Event type counts
        event_type_counts = base_query.with_entities(
            AnalyticsEvent.event_name,
            func.count(AnalyticsEvent.id).label('count')
        ).group_by(AnalyticsEvent.event_name).all()
        
        event_type_dict = {name: count for name, count in event_type_counts}
        
        # Error rate calculation (simplified)
        total_events = base_query.count()
        error_events = base_query.filter(
            AnalyticsEvent.event_name.like('%error%') | 
            AnalyticsEvent.event_category == 'error'
        ).count()
        
        error_rate = error_events / total_events if total_events > 0 else 0
        
        return {
            'daily_counts': daily_counts_list,
            'hourly_counts': hourly_counts_list,
            'event_type_counts': event_type_dict,
            'error_rate': error_rate,
            'total_events': total_events
        }
    
    async def _check_real_time_insights(self, event: AnalyticsEvent):
        """Check for real-time insights based on new events"""
        
        # Real-time anomaly detection
        if event.event_name == 'error' or event.event_category == 'error':
            # Check if error rate is spiking
            recent_events = self.db.query(AnalyticsEvent).filter(
                AnalyticsEvent.organization_id == event.organization_id,
                AnalyticsEvent.timestamp >= datetime.utcnow() - timedelta(minutes=5)
            ).count()
            
            error_events = self.db.query(AnalyticsEvent).filter(
                AnalyticsEvent.organization_id == event.organization_id,
                AnalyticsEvent.timestamp >= datetime.utcnow() - timedelta(minutes=5),
                (AnalyticsEvent.event_name.like('%error%') | (AnalyticsEvent.event_category == 'error'))
            ).count()
            
            if recent_events > 0 and (error_events / recent_events) > 0.2:  # 20% error rate
                # Create real-time alert insight
                insight = AnalyticsInsight(
                    insight_type='real_time_alert',
                    title='Error Rate Spike Detected',
                    description=f'Error rate spiked to {(error_events/recent_events):.1%} in the last 5 minutes',
                    confidence_score=0.95,
                    importance_score=1.0,
                    category='critical',
                    metrics={'error_rate': error_events/recent_events, 'time_window': '5_minutes'},
                    organization_id=event.organization_id,
                    expires_at=datetime.utcnow() + timedelta(hours=1)
                )
                
                self.db.add(insight)
                self.db.commit()
    
    # Utility methods
    def _extract_platform(self, user_agent: str) -> Optional[str]:
        """Extract platform from user agent"""
        if 'Windows' in user_agent:
            return 'Windows'
        elif 'Mac' in user_agent:
            return 'Mac'
        elif 'Linux' in user_agent:
            return 'Linux'
        elif 'Android' in user_agent:
            return 'Android'
        elif 'iOS' in user_agent:
            return 'iOS'
        return 'Unknown'
    
    def _extract_device_type(self, user_agent: str) -> Optional[str]:
        """Extract device type from user agent"""
        if 'Mobile' in user_agent:
            return 'Mobile'
        elif 'Tablet' in user_agent:
            return 'Tablet'
        else:
            return 'Desktop'
    
    def _calculate_trend(self, series_data: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Calculate trend from series data"""
        if len(series_data) < 2:
            return None
        
        values = [item['value'] for item in series_data]
        current = values[-1]
        previous = values[-2]
        
        if previous == 0:
            return None
        
        change_percent = ((current - previous) / previous) * 100
        
        return {
            'direction': 'up' if change_percent > 0 else 'down',
            'change_percent': abs(change_percent),
            'current_value': current,
            'previous_value': previous
        }
    
    def _can_access_dashboard(self, dashboard: AnalyticsDashboard, user_id: str) -> bool:
        """Check if user can access dashboard"""
        return (
            dashboard.created_by == user_id or
            dashboard.is_public or
            (dashboard.shared_with and user_id in dashboard.shared_with) or
            dashboard.organization_id == self._get_user_org(user_id)
        )
    
    def _get_user_org(self, user_id: str) -> Optional[str]:
        """Get user's organization ID"""
        return "default_org"
    
    def _is_admin(self, user_id: str) -> bool:
        """Check if user is admin"""
        return False  # Simplified for demo

# Dependency injection
def get_db():
    pass

def get_current_user():
    return "current_user_id"

# API Endpoints
@router.post("/events")
async def track_event(
    event_data: AnalyticsEventCreate,
    request: Request,
    db: Session = Depends(get_db),
    current_user: Optional[str] = Depends(get_current_user)
):
    """Track an analytics event"""
    
    service = AnalyticsService(db)
    
    # Extract request context
    request_context = {
        'ip_address': request.client.host,
        'user_agent': request.headers.get('user-agent', '')
    }
    
    result = await service.track_event(event_data, current_user, request_context)
    return result

@router.post("/query")
async def query_analytics(
    query: AnalyticsQuery,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Query analytics data"""
    
    service = AnalyticsService(db)
    result = await service.query_analytics(query, current_user)
    return result

@router.get("/summary")
async def get_usage_summary(
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Get usage summary statistics"""
    
    time_range = None
    if start_date or end_date:
        time_range = {
            'start': start_date or (datetime.utcnow() - timedelta(days=30)),
            'end': end_date or datetime.utcnow()
        }
    
    service = AnalyticsService(db)
    summary = await service.get_usage_summary(current_user, time_range)
    return summary

@router.get("/users/{user_id}")
async def get_user_analytics(
    user_id: str,
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Get analytics for a specific user"""
    
    time_range = None
    if start_date or end_date:
        time_range = {
            'start': start_date or (datetime.utcnow() - timedelta(days=30)),
            'end': end_date or datetime.utcnow()
        }
    
    service = AnalyticsService(db)
    analytics = await service.get_user_analytics(user_id, current_user, time_range)
    return analytics

@router.post("/dashboards", response_model=DashboardResponse)
async def create_dashboard(
    dashboard_data: DashboardCreate,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Create a custom dashboard"""
    
    dashboard = AnalyticsDashboard(
        name=dashboard_data.name,
        description=dashboard_data.description,
        config=dashboard_data.config,
        filters=dashboard_data.filters,
        refresh_interval=dashboard_data.refresh_interval,
        is_public=dashboard_data.is_public,
        shared_with=dashboard_data.shared_with,
        created_by=current_user,
        organization_id=AnalyticsService(db)._get_user_org(current_user)
    )
    
    db.add(dashboard)
    db.commit()
    db.refresh(dashboard)
    
    return dashboard

@router.get("/dashboards", response_model=List[DashboardResponse])
async def get_dashboards(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Get accessible dashboards"""
    
    service = AnalyticsService(db)
    
    query = db.query(AnalyticsDashboard).filter(
        (AnalyticsDashboard.created_by == current_user) |
        (AnalyticsDashboard.is_public == True) |
        (AnalyticsDashboard.organization_id == service._get_user_org(current_user))
    )
    
    dashboards = query.order_by(desc(AnalyticsDashboard.created_at)).offset(skip).limit(limit).all()
    return dashboards

@router.get("/dashboards/{dashboard_id}")
async def get_dashboard_data(
    dashboard_id: str,
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Get dashboard data"""
    
    time_range = None
    if start_date or end_date:
        time_range = {
            'start': start_date or (datetime.utcnow() - timedelta(days=30)),
            'end': end_date or datetime.utcnow()
        }
    
    service = AnalyticsService(db)
    data = await service.get_dashboard_data(dashboard_id, current_user, time_range)
    return data

@router.get("/insights", response_model=List[InsightResponse])
async def get_insights(
    insight_types: Optional[List[str]] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Get analytics insights"""
    
    time_range = None
    if start_date or end_date:
        time_range = {
            'start': start_date or (datetime.utcnow() - timedelta(days=30)),
            'end': end_date or datetime.utcnow()
        }
    
    service = AnalyticsService(db)
    insights = await service.generate_insights(current_user, insight_types, time_range)
    return insights

@router.post("/insights/{insight_id}/acknowledge")
async def acknowledge_insight(
    insight_id: str,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Acknowledge an insight"""
    
    insight = db.query(AnalyticsInsight).filter(AnalyticsInsight.id == insight_id).first()
    if not insight:
        raise HTTPException(status_code=404, detail="Insight not found")
    
    insight.acknowledged_by = current_user
    insight.acknowledged_at = datetime.utcnow()
    
    db.commit()
    
    return {"message": "Insight acknowledged"}

@router.delete("/dashboards/{dashboard_id}")
async def delete_dashboard(
    dashboard_id: str,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Delete a dashboard"""
    
    dashboard = db.query(AnalyticsDashboard).filter(AnalyticsDashboard.id == dashboard_id).first()
    if not dashboard:
        raise HTTPException(status_code=404, detail="Dashboard not found")
    
    if dashboard.created_by != current_user:
        raise HTTPException(status_code=403, detail="Permission denied")
    
    db.delete(dashboard)
    db.commit()
    
    return {"message": "Dashboard deleted successfully"}