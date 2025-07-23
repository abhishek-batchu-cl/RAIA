# Smart Insights Generation Service
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
import uuid
from dataclasses import dataclass, asdict
from sqlalchemy.orm import Session
from sqlalchemy import Column, String, DateTime, Text, Boolean, JSON, UUID as PG_UUID, Float, Integer
from sqlalchemy.ext.declarative import declarative_base
import numpy as np
import pandas as pd

Base = declarative_base()

@dataclass
class SmartInsight:
    id: str
    type: str  # 'alert', 'recommendation', 'achievement', 'trend'
    priority: str  # 'critical', 'high', 'medium', 'low'
    title: str
    description: str
    action: Optional[Dict[str, Any]] = None
    impact: str = 'medium'  # 'high', 'medium', 'low'
    category: str = 'performance'  # 'performance', 'drift', 'bias', 'quality', 'business'
    timestamp: datetime = None
    dismissed: bool = False
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}

class UserInsightPreferences(Base):
    """Store user preferences for insight generation"""
    __tablename__ = "user_insight_preferences"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(255), nullable=False, unique=True)
    role = Column(String(50), default='data_scientist')  # data_scientist, business_user, ml_engineer, executive
    focus_areas = Column(JSON, default=['performance', 'drift'])  # List of areas to focus on
    notification_level = Column(String(20), default='important')  # all, important, critical
    favorite_models = Column(JSON, default=list)
    dismissed_insights = Column(JSON, default=list)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

class GeneratedInsight(Base):
    """Store generated insights for tracking and analytics"""
    __tablename__ = "generated_insights"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    insight_id = Column(String(255), unique=True, nullable=False)
    user_id = Column(String(255))
    type = Column(String(50), nullable=False)
    priority = Column(String(20), nullable=False)
    category = Column(String(50), nullable=False)
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=False)
    action_data = Column(JSON)
    impact = Column(String(20), default='medium')
    metadata = Column(JSON, default=dict)
    dismissed = Column(Boolean, default=False)
    clicked = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class SmartInsightsGenerator:
    """AI-powered service for generating personalized insights for ML models and data"""
    
    def __init__(self, db_session: Session = None):
        self.db = db_session
        
        # Insight generation rules and templates
        self.insight_generators = {
            'performance_monitoring': self._generate_performance_insights,
            'data_drift_detection': self._generate_drift_insights,
            'bias_detection': self._generate_bias_insights,
            'data_quality': self._generate_quality_insights,
            'business_impact': self._generate_business_insights,
            'optimization_opportunities': self._generate_optimization_insights
        }

    async def generate_insights_for_user(self, user_id: str, 
                                       user_preferences: Optional[Dict[str, Any]] = None) -> List[SmartInsight]:
        """Generate personalized insights for a specific user"""
        
        # Get user preferences
        if user_preferences is None:
            user_preferences = await self._get_user_preferences(user_id)
        
        # Generate insights based on preferences and current system state
        insights = []
        
        for generator_name, generator_func in self.insight_generators.items():
            try:
                generated = await generator_func(user_id, user_preferences)
                insights.extend(generated)
            except Exception as e:
                print(f"Error in {generator_name}: {e}")
                continue
        
        # Filter and prioritize insights based on user preferences
        filtered_insights = self._filter_insights_by_preferences(insights, user_preferences)
        
        # Store insights in database
        if self.db:
            await self._store_insights(user_id, filtered_insights)
        
        return filtered_insights

    async def _generate_performance_insights(self, user_id: str, 
                                           preferences: Dict[str, Any]) -> List[SmartInsight]:
        """Generate insights about model performance"""
        insights = []
        
        # Simulate model performance analysis
        models_data = await self._get_models_performance_data(user_id)
        
        for model_data in models_data:
            # Performance drop detection
            if model_data.get('accuracy_drop', 0) > 0.05:  # 5% drop
                insights.append(SmartInsight(
                    id=f"perf_drop_{model_data['model_id']}",
                    type='alert',
                    priority='critical' if model_data['accuracy_drop'] > 0.15 else 'high',
                    title=f"{model_data['model_name']} Performance Drop Detected",
                    description=f"Model accuracy decreased by {model_data['accuracy_drop']:.1%} over the past {model_data['time_period']}. Current accuracy: {model_data['current_accuracy']:.1%}",
                    action={
                        'label': 'Investigate Model',
                        'route': f'/model-monitoring/{model_data["model_id"]}'
                    },
                    impact='high',
                    category='performance',
                    metadata={
                        'model_id': model_data['model_id'],
                        'accuracy_drop': model_data['accuracy_drop'],
                        'current_accuracy': model_data['current_accuracy']
                    }
                ))
            
            # Performance improvement celebration
            elif model_data.get('accuracy_improvement', 0) > 0.03:
                insights.append(SmartInsight(
                    id=f"perf_improve_{model_data['model_id']}",
                    type='achievement',
                    priority='medium',
                    title=f"Great Progress on {model_data['model_name']}!",
                    description=f"Model performance improved by {model_data['accuracy_improvement']:.1%}. Your recent optimizations are working!",
                    action={
                        'label': 'View Improvements',
                        'route': f'/model-monitoring/{model_data["model_id"]}'
                    },
                    impact='medium',
                    category='performance',
                    metadata=model_data
                ))
        
        return insights

    async def _generate_drift_insights(self, user_id: str, 
                                     preferences: Dict[str, Any]) -> List[SmartInsight]:
        """Generate insights about data drift"""
        insights = []
        
        # Simulate drift detection data
        drift_data = await self._get_drift_detection_data(user_id)
        
        for drift_info in drift_data:
            if drift_info['drift_detected']:
                severity = drift_info['drift_magnitude']
                priority = 'critical' if severity > 0.8 else 'high' if severity > 0.5 else 'medium'
                
                insights.append(SmartInsight(
                    id=f"drift_{drift_info['feature']}_{drift_info['model_id']}",
                    type='alert',
                    priority=priority,
                    title=f"Data Drift in {drift_info['feature'].replace('_', ' ').title()}",
                    description=f"Significant drift detected in {drift_info['feature']} feature with magnitude {severity:.2f}. This may impact model predictions.",
                    action={
                        'label': 'Analyze Drift',
                        'route': '/data-drift',
                        'params': {'feature': drift_info['feature'], 'model': drift_info['model_id']}
                    },
                    impact='high' if severity > 0.7 else 'medium',
                    category='drift',
                    metadata=drift_info
                ))
        
        return insights

    async def _generate_bias_insights(self, user_id: str, 
                                    preferences: Dict[str, Any]) -> List[SmartInsight]:
        """Generate insights about model bias and fairness"""
        insights = []
        
        bias_data = await self._get_bias_detection_data(user_id)
        
        for bias_info in bias_data:
            if bias_info['bias_detected']:
                insights.append(SmartInsight(
                    id=f"bias_{bias_info['model_id']}_{bias_info['protected_attribute']}",
                    type='alert',
                    priority='high',
                    title=f"Potential Bias Detected in {bias_info['model_name']}",
                    description=f"Bias detected for {bias_info['protected_attribute']} with {bias_info['bias_metric']}: {bias_info['bias_value']:.3f} (threshold: {bias_info['threshold']:.3f})",
                    action={
                        'label': 'Review Bias Analysis',
                        'route': '/bias-detection',
                        'params': {'model': bias_info['model_id']}
                    },
                    impact='high',
                    category='bias',
                    metadata=bias_info
                ))
            elif bias_info.get('bias_improvement'):
                insights.append(SmartInsight(
                    id=f"bias_improve_{bias_info['model_id']}",
                    type='achievement',
                    priority='medium',
                    title=f"Fairness Improved in {bias_info['model_name']}",
                    description=f"Bias mitigation successful! {bias_info['bias_metric']} improved by {bias_info['improvement']:.1%}",
                    action={
                        'label': 'View Fairness Report',
                        'route': '/bias-detection'
                    },
                    impact='high',
                    category='bias',
                    metadata=bias_info
                ))
        
        return insights

    async def _generate_quality_insights(self, user_id: str, 
                                       preferences: Dict[str, Any]) -> List[SmartInsight]:
        """Generate insights about data quality issues"""
        insights = []
        
        quality_data = await self._get_data_quality_metrics(user_id)
        
        for quality_info in quality_data:
            if quality_info['missing_data_percentage'] > 0.15:  # 15% threshold
                insights.append(SmartInsight(
                    id=f"quality_missing_{quality_info['dataset']}",
                    type='alert',
                    priority='medium',
                    title=f"High Missing Data in {quality_info['dataset']}",
                    description=f"{quality_info['missing_data_percentage']:.1%} of data is missing. This may impact model performance.",
                    action={
                        'label': 'View Data Quality Report',
                        'route': '/data-quality'
                    },
                    impact='medium',
                    category='quality',
                    metadata=quality_info
                ))
        
        return insights

    async def _generate_business_insights(self, user_id: str, 
                                        preferences: Dict[str, Any]) -> List[SmartInsight]:
        """Generate business-focused insights"""
        insights = []
        
        # Simulate business metrics
        business_data = await self._get_business_metrics_data(user_id)
        
        for metric in business_data:
            if metric['roi_improvement'] > 0.1:  # 10% ROI improvement
                insights.append(SmartInsight(
                    id=f"business_roi_{metric['model_id']}",
                    type='achievement',
                    priority='high',
                    title=f"Strong ROI from {metric['model_name']}",
                    description=f"Model generated ${metric['revenue_impact']:,.0f} in additional revenue with {metric['roi_improvement']:.1%} ROI improvement.",
                    action={
                        'label': 'View Business Impact',
                        'route': '/business-metrics'
                    },
                    impact='high',
                    category='business',
                    metadata=metric
                ))
        
        return insights

    async def _generate_optimization_insights(self, user_id: str, 
                                            preferences: Dict[str, Any]) -> List[SmartInsight]:
        """Generate optimization recommendations"""
        insights = []
        
        # Feature engineering opportunities
        feature_data = await self._analyze_feature_opportunities(user_id)
        
        for opportunity in feature_data:
            if opportunity['potential_improvement'] > 0.05:  # 5% potential improvement
                insights.append(SmartInsight(
                    id=f"opt_feature_{opportunity['model_id']}",
                    type='recommendation',
                    priority='high' if opportunity['potential_improvement'] > 0.1 else 'medium',
                    title=f"Feature Engineering Opportunity for {opportunity['model_name']}",
                    description=f"Combining {', '.join(opportunity['suggested_features'])} could improve performance by {opportunity['potential_improvement']:.1%}",
                    action={
                        'label': 'Explore Feature Engineering',
                        'route': '/feature-engineering',
                        'params': {'suggested': opportunity['suggested_features']}
                    },
                    impact='medium',
                    category='performance',
                    metadata=opportunity
                ))
        
        return insights

    def _filter_insights_by_preferences(self, insights: List[SmartInsight], 
                                      preferences: Dict[str, Any]) -> List[SmartInsight]:
        """Filter insights based on user preferences"""
        
        notification_level = preferences.get('notification_level', 'important')
        focus_areas = preferences.get('focus_areas', ['performance', 'drift'])
        
        filtered = []
        for insight in insights:
            # Filter by notification level
            if notification_level == 'critical' and insight.priority != 'critical':
                continue
            elif notification_level == 'important' and insight.priority not in ['critical', 'high']:
                continue
            
            # Filter by focus areas
            if insight.category not in focus_areas:
                continue
            
            filtered.append(insight)
        
        # Sort by priority and timestamp
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        filtered.sort(key=lambda x: (priority_order[x.priority], x.timestamp), reverse=True)
        
        return filtered[:20]  # Limit to 20 insights

    async def _get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user preferences from database"""
        if not self.db:
            # Return default preferences
            return {
                'role': 'data_scientist',
                'focus_areas': ['performance', 'drift', 'bias'],
                'notification_level': 'important',
                'favorite_models': []
            }
        
        try:
            prefs = self.db.query(UserInsightPreferences).filter(
                UserInsightPreferences.user_id == user_id
            ).first()
            
            if prefs:
                return {
                    'role': prefs.role,
                    'focus_areas': prefs.focus_areas,
                    'notification_level': prefs.notification_level,
                    'favorite_models': prefs.favorite_models
                }
            else:
                # Create default preferences
                default_prefs = UserInsightPreferences(
                    user_id=user_id,
                    role='data_scientist',
                    focus_areas=['performance', 'drift', 'bias'],
                    notification_level='important'
                )
                self.db.add(default_prefs)
                self.db.commit()
                
                return {
                    'role': 'data_scientist',
                    'focus_areas': ['performance', 'drift', 'bias'],
                    'notification_level': 'important',
                    'favorite_models': []
                }
        except Exception as e:
            print(f"Error getting user preferences: {e}")
            return {
                'role': 'data_scientist',
                'focus_areas': ['performance', 'drift', 'bias'],
                'notification_level': 'important',
                'favorite_models': []
            }

    async def _store_insights(self, user_id: str, insights: List[SmartInsight]):
        """Store generated insights in database"""
        if not self.db:
            return
        
        try:
            for insight in insights:
                db_insight = GeneratedInsight(
                    insight_id=insight.id,
                    user_id=user_id,
                    type=insight.type,
                    priority=insight.priority,
                    category=insight.category,
                    title=insight.title,
                    description=insight.description,
                    action_data=insight.action,
                    impact=insight.impact,
                    metadata=insight.metadata or {}
                )
                self.db.add(db_insight)
            
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            print(f"Error storing insights: {e}")

    # Mock data generation methods for demonstration
    async def _get_models_performance_data(self, user_id: str) -> List[Dict[str, Any]]:
        """Mock method to get model performance data"""
        return [
            {
                'model_id': 'credit-scoring-v2',
                'model_name': 'Credit Scoring Model',
                'current_accuracy': 0.847,
                'accuracy_drop': 0.12,
                'time_period': 'week',
                'baseline_accuracy': 0.967
            },
            {
                'model_id': 'fraud-detection-v1',
                'model_name': 'Fraud Detection Model',
                'current_accuracy': 0.934,
                'accuracy_improvement': 0.045,
                'time_period': 'month'
            }
        ]

    async def _get_drift_detection_data(self, user_id: str) -> List[Dict[str, Any]]:
        """Mock method to get drift detection data"""
        return [
            {
                'model_id': 'credit-scoring-v2',
                'feature': 'customer_age',
                'drift_detected': True,
                'drift_magnitude': 0.73,
                'drift_type': 'distribution_shift',
                'baseline_mean': 35.2,
                'current_mean': 42.8
            },
            {
                'model_id': 'credit-scoring-v2',
                'feature': 'annual_income',
                'drift_detected': True,
                'drift_magnitude': 0.45,
                'drift_type': 'statistical_shift'
            }
        ]

    async def _get_bias_detection_data(self, user_id: str) -> List[Dict[str, Any]]:
        """Mock method to get bias detection data"""
        return [
            {
                'model_id': 'credit-scoring-v2',
                'model_name': 'Credit Scoring Model',
                'protected_attribute': 'gender',
                'bias_detected': False,
                'bias_improvement': True,
                'bias_metric': 'demographic_parity',
                'bias_value': 0.023,
                'threshold': 0.05,
                'improvement': 0.23
            }
        ]

    async def _get_data_quality_metrics(self, user_id: str) -> List[Dict[str, Any]]:
        """Mock method to get data quality metrics"""
        return [
            {
                'dataset': 'customer_applications',
                'missing_data_percentage': 0.18,
                'duplicate_records': 45,
                'outlier_percentage': 0.05
            }
        ]

    async def _get_business_metrics_data(self, user_id: str) -> List[Dict[str, Any]]:
        """Mock method to get business metrics data"""
        return [
            {
                'model_id': 'credit-scoring-v2',
                'model_name': 'Credit Scoring Model',
                'revenue_impact': 2450000,
                'roi_improvement': 0.15,
                'cost_savings': 450000
            }
        ]

    async def _analyze_feature_opportunities(self, user_id: str) -> List[Dict[str, Any]]:
        """Mock method to analyze feature engineering opportunities"""
        return [
            {
                'model_id': 'credit-scoring-v2',
                'model_name': 'Credit Scoring Model',
                'suggested_features': ['annual_income', 'debt_ratio'],
                'potential_improvement': 0.085,
                'confidence': 0.78
            }
        ]

# Factory function
def create_insights_generator(db_session: Session = None) -> SmartInsightsGenerator:
    """Create and return a SmartInsightsGenerator instance"""
    return SmartInsightsGenerator(db_session)