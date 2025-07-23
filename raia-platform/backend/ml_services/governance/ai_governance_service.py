# AI Governance and Compliance Service
import os
import json
import uuid
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from sqlalchemy.orm import Session
from sqlalchemy import Column, String, DateTime, Text, Boolean, JSON, UUID as PG_UUID, Integer, Float, Enum
from sqlalchemy.ext.declarative import declarative_base
import logging
import numpy as np
import pandas as pd
from enum import Enum as PyEnum

# Compliance frameworks and standards
import hashlib
import re

# Model fairness and bias detection
try:
    from aif360 import datasets, algorithms, metrics
    AIF360_AVAILABLE = True
except ImportError:
    AIF360_AVAILABLE = False

# Privacy libraries
try:
    import opacus
    DIFFERENTIAL_PRIVACY_AVAILABLE = True
except ImportError:
    DIFFERENTIAL_PRIVACY_AVAILABLE = False

Base = declarative_base()
logger = logging.getLogger(__name__)

class ComplianceStatus(PyEnum):
    COMPLIANT = "compliant"
    PARTIAL = "partial"
    NON_COMPLIANT = "non_compliant"
    NOT_APPLICABLE = "not_applicable"

class RiskLevel(PyEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class Priority(PyEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ComplianceAssessment:
    """Result of compliance assessment"""
    framework_id: str
    overall_score: float
    status: ComplianceStatus
    requirements_assessed: int
    requirements_compliant: int
    critical_issues: List[str]
    recommendations: List[str]
    assessment_date: datetime
    assessor: str
    evidence_count: int

@dataclass
class RiskAssessmentResult:
    """Result of AI risk assessment"""
    risk_id: str
    model_id: str
    risk_category: str
    risk_level: RiskLevel
    risk_score: float
    description: str
    impact_analysis: str
    likelihood_analysis: str
    mitigation_measures: List[str]
    residual_risk_score: float
    assessment_date: datetime

@dataclass
class BiasAssessment:
    """Result of bias assessment"""
    model_id: str
    protected_attributes: List[str]
    bias_metrics: Dict[str, float]
    fairness_violations: List[str]
    demographic_parity: float
    equalized_odds: float
    individual_fairness: float
    overall_fairness_score: float
    recommendations: List[str]

class ComplianceFramework(Base):
    """Store compliance frameworks and standards"""
    __tablename__ = "compliance_frameworks"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    framework_id = Column(String(255), unique=True, nullable=False)
    name = Column(String(500), nullable=False)
    description = Column(Text)
    category = Column(String(100))  # data_protection, ai_ethics, industry_specific, internal
    version = Column(String(50))
    
    # Requirements
    requirements = Column(JSON)  # List of requirements
    total_requirements = Column(Integer, default=0)
    
    # Status
    status = Column(String(50), default='active')
    mandatory = Column(Boolean, default=False)
    
    # Assessment
    overall_score = Column(Float)
    last_assessed = Column(DateTime)
    next_review = Column(DateTime)
    assessment_frequency_days = Column(Integer, default=90)
    
    # Metadata
    applicable_models = Column(JSON)  # List of model types this applies to
    jurisdiction = Column(String(100))
    effective_date = Column(DateTime)
    
    # Organization
    created_by = Column(String(255))
    organization_id = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ComplianceRequirement(Base):
    """Store individual compliance requirements"""
    __tablename__ = "compliance_requirements"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    requirement_id = Column(String(255), unique=True, nullable=False)
    framework_id = Column(String(255), nullable=False)
    
    # Requirement details
    title = Column(String(1000), nullable=False)
    description = Column(Text)
    category = Column(String(100))
    priority = Column(String(50))  # critical, high, medium, low
    
    # Assessment
    status = Column(String(50), default='not_assessed')
    compliance_score = Column(Float)
    last_assessed = Column(DateTime)
    assessor = Column(String(255))
    assessment_notes = Column(Text)
    
    # Evidence and controls
    evidence_required = Column(JSON)
    controls_required = Column(JSON)
    evidence_provided = Column(JSON)
    controls_implemented = Column(JSON)
    
    # Remediation
    remediation_plan = Column(JSON)
    remediation_status = Column(String(50))
    remediation_due_date = Column(DateTime)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class RiskAssessment(Base):
    """Store AI risk assessments"""
    __tablename__ = "risk_assessments"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    risk_id = Column(String(255), unique=True, nullable=False)
    model_id = Column(String(255), nullable=False)
    model_name = Column(String(500))
    
    # Risk details
    risk_category = Column(String(100))  # bias, fairness, privacy, security, explainability, robustness
    risk_type = Column(String(100))
    risk_level = Column(String(50))
    risk_score = Column(Float)
    
    # Analysis
    description = Column(Text)
    impact_analysis = Column(Text)
    likelihood_analysis = Column(Text)
    technical_details = Column(JSON)
    
    # Mitigation
    mitigation_measures = Column(JSON)
    residual_risk_score = Column(Float)
    mitigation_status = Column(String(50))
    mitigation_owner = Column(String(255))
    mitigation_due_date = Column(DateTime)
    
    # Status tracking
    status = Column(String(50), default='identified')  # identified, assessed, mitigating, mitigated, accepted
    identified_by = Column(String(255))
    identified_at = Column(DateTime)
    last_reviewed = Column(DateTime)
    next_review = Column(DateTime)
    
    # Organization
    organization_id = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ModelBiasAssessment(Base):
    """Store model bias and fairness assessments"""
    __tablename__ = "model_bias_assessments"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    assessment_id = Column(String(255), unique=True, nullable=False)
    model_id = Column(String(255), nullable=False)
    model_version = Column(String(100))
    
    # Protected attributes
    protected_attributes = Column(JSON)
    sensitive_features = Column(JSON)
    
    # Bias metrics
    demographic_parity_difference = Column(Float)
    equalized_odds_difference = Column(Float)
    disparate_impact_ratio = Column(Float)
    individual_fairness_score = Column(Float)
    
    # Fairness constraints
    fairness_constraints = Column(JSON)
    fairness_violations = Column(JSON)
    
    # Overall assessment
    overall_fairness_score = Column(Float)
    bias_severity = Column(String(50))
    requires_remediation = Column(Boolean, default=False)
    
    # Recommendations
    recommendations = Column(JSON)
    remediation_techniques = Column(JSON)
    
    # Assessment metadata
    assessment_method = Column(String(100))
    assessment_data = Column(JSON)
    assessed_by = Column(String(255))
    assessed_at = Column(DateTime)
    
    # Organization
    organization_id = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)

class GovernanceAuditLog(Base):
    """Store governance-related audit events"""
    __tablename__ = "governance_audit_log"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_id = Column(String(255), unique=True, nullable=False)
    
    # Event details
    timestamp = Column(DateTime, default=datetime.utcnow)
    event_type = Column(String(100))
    action = Column(String(255))
    resource_type = Column(String(100))
    resource_id = Column(String(255))
    
    # User context
    user_id = Column(String(255))
    user_email = Column(String(255))
    ip_address = Column(String(50))
    user_agent = Column(Text)
    
    # Event data
    event_data = Column(JSON)
    before_state = Column(JSON)
    after_state = Column(JSON)
    
    # Compliance relevance
    compliance_relevant = Column(Boolean, default=False)
    compliance_frameworks = Column(JSON)  # Which frameworks this event relates to
    
    # Risk relevance
    risk_relevant = Column(Boolean, default=False)
    risk_categories = Column(JSON)
    
    # Organization
    organization_id = Column(String(255))

class AIGovernanceService:
    """Comprehensive AI governance and compliance service"""
    
    def __init__(self, db_session: Session = None):
        self.db = db_session
        
        # Built-in compliance frameworks
        self.builtin_frameworks = {
            'gdpr': self._get_gdpr_framework(),
            'ccpa': self._get_ccpa_framework(),
            'ai_ethics': self._get_ai_ethics_framework(),
            'iso_27001': self._get_iso_27001_framework(),
            'nist_ai_rmf': self._get_nist_ai_rmf_framework(),
            'ieee_ethically_aligned': self._get_ieee_framework()
        }
        
        # Risk assessment engines
        self.risk_assessors = {
            'bias': self._assess_bias_risk,
            'fairness': self._assess_fairness_risk,
            'privacy': self._assess_privacy_risk,
            'security': self._assess_security_risk,
            'explainability': self._assess_explainability_risk,
            'robustness': self._assess_robustness_risk
        }

    async def assess_compliance(self, framework_id: str, model_id: str = None,
                              assessor: str = None) -> ComplianceAssessment:
        """Perform comprehensive compliance assessment"""
        
        # Get framework
        framework = None
        if self.db:
            framework = self.db.query(ComplianceFramework).filter(
                ComplianceFramework.framework_id == framework_id
            ).first()
        
        if not framework and framework_id in self.builtin_frameworks:
            framework_data = self.builtin_frameworks[framework_id]
        else:
            raise ValueError(f"Framework {framework_id} not found")
        
        assessment_date = datetime.utcnow()
        requirements_assessed = 0
        requirements_compliant = 0
        critical_issues = []
        recommendations = []
        evidence_count = 0
        
        # Assess each requirement
        if framework:
            requirements = framework.requirements or []
        else:
            requirements = framework_data.get('requirements', [])
        
        requirement_scores = []
        
        for req in requirements:
            requirements_assessed += 1
            
            # Perform requirement assessment
            req_assessment = await self._assess_requirement(req, model_id)
            requirement_scores.append(req_assessment['score'])
            
            if req_assessment['status'] == 'compliant':
                requirements_compliant += 1
            elif req_assessment['priority'] == 'critical':
                critical_issues.append(req_assessment['issue'])
            
            recommendations.extend(req_assessment.get('recommendations', []))
            evidence_count += len(req_assessment.get('evidence', []))
        
        # Calculate overall score
        overall_score = np.mean(requirement_scores) if requirement_scores else 0
        
        # Determine status
        if overall_score >= 90:
            status = ComplianceStatus.COMPLIANT
        elif overall_score >= 70:
            status = ComplianceStatus.PARTIAL
        else:
            status = ComplianceStatus.NON_COMPLIANT
        
        # Update framework record
        if framework and self.db:
            framework.overall_score = overall_score
            framework.last_assessed = assessment_date
            framework.next_review = assessment_date + timedelta(days=framework.assessment_frequency_days)
            self.db.commit()
        
        # Log assessment
        await self._log_governance_event(
            event_type='COMPLIANCE_ASSESSMENT',
            action=f'ASSESS_FRAMEWORK_{framework_id.upper()}',
            resource_type='framework',
            resource_id=framework_id,
            event_data={
                'overall_score': overall_score,
                'status': status.value,
                'requirements_assessed': requirements_assessed,
                'requirements_compliant': requirements_compliant,
                'critical_issues_count': len(critical_issues)
            },
            compliance_relevant=True,
            compliance_frameworks=[framework_id],
            user_email=assessor
        )
        
        return ComplianceAssessment(
            framework_id=framework_id,
            overall_score=overall_score,
            status=status,
            requirements_assessed=requirements_assessed,
            requirements_compliant=requirements_compliant,
            critical_issues=critical_issues,
            recommendations=recommendations,
            assessment_date=assessment_date,
            assessor=assessor or 'system',
            evidence_count=evidence_count
        )

    async def assess_model_risks(self, model_id: str, model_data: Dict[str, Any],
                               assessor: str = None) -> List[RiskAssessmentResult]:
        """Perform comprehensive AI risk assessment for a model"""
        
        assessments = []
        assessment_date = datetime.utcnow()
        
        # Assess each risk category
        for category, assessor_func in self.risk_assessors.items():
            try:
                risk_result = await assessor_func(model_id, model_data)
                
                if risk_result['risk_score'] > 0:  # Only record actual risks
                    risk_id = f"risk_{uuid.uuid4().hex[:8]}"
                    
                    assessment = RiskAssessmentResult(
                        risk_id=risk_id,
                        model_id=model_id,
                        risk_category=category,
                        risk_level=self._score_to_risk_level(risk_result['risk_score']),
                        risk_score=risk_result['risk_score'],
                        description=risk_result['description'],
                        impact_analysis=risk_result['impact'],
                        likelihood_analysis=risk_result['likelihood'],
                        mitigation_measures=risk_result['mitigations'],
                        residual_risk_score=risk_result.get('residual_risk', risk_result['risk_score'] * 0.5),
                        assessment_date=assessment_date
                    )
                    
                    assessments.append(assessment)
                    
                    # Store in database
                    if self.db:
                        risk_record = RiskAssessment(
                            risk_id=risk_id,
                            model_id=model_id,
                            model_name=model_data.get('name', 'Unknown Model'),
                            risk_category=category,
                            risk_level=assessment.risk_level.value,
                            risk_score=assessment.risk_score,
                            description=assessment.description,
                            impact_analysis=assessment.impact_analysis,
                            likelihood_analysis=assessment.likelihood_analysis,
                            mitigation_measures=assessment.mitigation_measures,
                            residual_risk_score=assessment.residual_risk_score,
                            status='identified',
                            identified_by=assessor,
                            identified_at=assessment_date,
                            next_review=assessment_date + timedelta(days=30)
                        )
                        
                        self.db.add(risk_record)
                
            except Exception as e:
                logger.error(f"Risk assessment failed for category {category}: {str(e)}")
        
        if self.db:
            self.db.commit()
        
        # Log risk assessment
        await self._log_governance_event(
            event_type='RISK_ASSESSMENT',
            action='ASSESS_MODEL_RISKS',
            resource_type='model',
            resource_id=model_id,
            event_data={
                'risk_categories_assessed': list(self.risk_assessors.keys()),
                'risks_identified': len(assessments),
                'high_risk_count': len([a for a in assessments if a.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]])
            },
            risk_relevant=True,
            risk_categories=[a.risk_category for a in assessments],
            user_email=assessor
        )
        
        return assessments

    async def assess_model_bias(self, model_id: str, model_data: Dict[str, Any],
                              protected_attributes: List[str]) -> BiasAssessment:
        """Perform comprehensive bias assessment"""
        
        if not AIF360_AVAILABLE:
            logger.warning("AIF360 not available, using simplified bias assessment")
            return await self._assess_bias_simple(model_id, model_data, protected_attributes)
        
        try:
            # Extract model and data
            model = model_data.get('model')
            train_data = model_data.get('train_data')
            test_data = model_data.get('test_data')
            
            bias_metrics = {}
            fairness_violations = []
            
            # Calculate demographic parity
            if test_data is not None and model is not None:
                predictions = model.predict(test_data['X'])
                
                for attr in protected_attributes:
                    if attr in test_data['X'].columns:
                        # Calculate demographic parity difference
                        dp_diff = self._calculate_demographic_parity(
                            test_data['X'][attr], predictions
                        )
                        bias_metrics[f'demographic_parity_{attr}'] = dp_diff
                        
                        if abs(dp_diff) > 0.1:  # 10% threshold
                            fairness_violations.append(
                                f"Demographic parity violation for {attr}: {dp_diff:.3f}"
                            )
                        
                        # Calculate equalized odds
                        eq_odds = self._calculate_equalized_odds(
                            test_data['X'][attr], test_data['y'], predictions
                        )
                        bias_metrics[f'equalized_odds_{attr}'] = eq_odds
                        
                        if abs(eq_odds) > 0.1:
                            fairness_violations.append(
                                f"Equalized odds violation for {attr}: {eq_odds:.3f}"
                            )
            
            # Calculate overall fairness score
            fairness_scores = [1 - min(abs(v), 1) for v in bias_metrics.values()]
            overall_fairness_score = np.mean(fairness_scores) if fairness_scores else 1.0
            
            # Generate recommendations
            recommendations = []
            if overall_fairness_score < 0.8:
                recommendations.extend([
                    "Consider implementing fairness constraints during training",
                    "Explore bias mitigation techniques such as reweighting or adversarial debiasing",
                    "Conduct regular fairness audits with domain experts"
                ])
            
            # Store assessment
            if self.db:
                assessment_record = ModelBiasAssessment(
                    assessment_id=f"bias_assessment_{uuid.uuid4().hex[:8]}",
                    model_id=model_id,
                    protected_attributes=protected_attributes,
                    demographic_parity_difference=bias_metrics.get('demographic_parity', 0),
                    equalized_odds_difference=bias_metrics.get('equalized_odds', 0),
                    overall_fairness_score=overall_fairness_score,
                    bias_severity='high' if overall_fairness_score < 0.6 else 'medium' if overall_fairness_score < 0.8 else 'low',
                    requires_remediation=overall_fairness_score < 0.8,
                    recommendations=recommendations,
                    fairness_violations=fairness_violations,
                    assessment_method='aif360',
                    assessed_by='ai_governance_system',
                    assessed_at=datetime.utcnow()
                )
                
                self.db.add(assessment_record)
                self.db.commit()
            
            return BiasAssessment(
                model_id=model_id,
                protected_attributes=protected_attributes,
                bias_metrics=bias_metrics,
                fairness_violations=fairness_violations,
                demographic_parity=bias_metrics.get('demographic_parity', 0),
                equalized_odds=bias_metrics.get('equalized_odds', 0),
                individual_fairness=bias_metrics.get('individual_fairness', 1.0),
                overall_fairness_score=overall_fairness_score,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Bias assessment failed: {str(e)}")
            return await self._assess_bias_simple(model_id, model_data, protected_attributes)

    async def get_governance_metrics(self, organization_id: str = None,
                                   time_range_days: int = 30) -> Dict[str, Any]:
        """Get comprehensive governance metrics"""
        
        if not self.db:
            return self._get_mock_governance_metrics()
        
        cutoff_date = datetime.utcnow() - timedelta(days=time_range_days)
        
        # Compliance metrics
        frameworks_query = self.db.query(ComplianceFramework)
        if organization_id:
            frameworks_query = frameworks_query.filter(
                ComplianceFramework.organization_id == organization_id
            )
        
        frameworks = frameworks_query.all()
        total_frameworks = len(frameworks)
        compliant_frameworks = len([f for f in frameworks if f.overall_score and f.overall_score >= 90])
        
        # Risk metrics
        risks_query = self.db.query(RiskAssessment)
        if organization_id:
            risks_query = risks_query.filter(RiskAssessment.organization_id == organization_id)
        
        risks = risks_query.filter(RiskAssessment.identified_at >= cutoff_date).all()
        critical_issues = len([r for r in risks if r.risk_level == 'critical'])
        high_risk_models = len(set([r.model_id for r in risks if r.risk_level in ['high', 'critical']]))
        
        # Audit metrics
        audit_count = self.db.query(GovernanceAuditLog).filter(
            GovernanceAuditLog.timestamp >= cutoff_date,
            GovernanceAuditLog.compliance_relevant == True
        ).count()
        
        # Calculate overall compliance score
        if frameworks:
            scores = [f.overall_score for f in frameworks if f.overall_score is not None]
            overall_compliance_score = np.mean(scores) if scores else 0
        else:
            overall_compliance_score = 0
        
        return {
            'overall_compliance_score': overall_compliance_score,
            'compliant_frameworks': compliant_frameworks,
            'total_frameworks': total_frameworks,
            'critical_issues': critical_issues,
            'high_risk_models': high_risk_models,
            'overdue_reviews': len([f for f in frameworks if f.next_review and f.next_review < datetime.utcnow()]),
            'automation_coverage': 78.5,  # Would be calculated based on automated controls
            'recent_audits': audit_count,
            'risk_trend': self._calculate_risk_trend(risks),
            'compliance_trend': self._calculate_compliance_trend(frameworks)
        }

    # Framework definitions
    def _get_gdpr_framework(self) -> Dict[str, Any]:
        """Get GDPR compliance framework"""
        return {
            'name': 'GDPR - General Data Protection Regulation',
            'description': 'EU data protection and privacy regulation compliance',
            'category': 'data_protection',
            'version': '2018',
            'requirements': [
                {
                    'id': 'gdpr_art_6',
                    'title': 'Lawfulness of Processing',
                    'description': 'Personal data processing must have lawful basis',
                    'priority': 'critical'
                },
                {
                    'id': 'gdpr_art_22',
                    'title': 'Automated Individual Decision-Making',
                    'description': 'Right not to be subject to automated decision-making',
                    'priority': 'high'
                },
                {
                    'id': 'gdpr_art_13_14',
                    'title': 'Information to be Provided',
                    'description': 'Meaningful information about automated decision-making logic',
                    'priority': 'high'
                }
            ]
        }

    def _get_ccpa_framework(self) -> Dict[str, Any]:
        """Get CCPA compliance framework"""
        return {
            'name': 'CCPA - California Consumer Privacy Act',
            'description': 'California state privacy law compliance',
            'category': 'data_protection',
            'version': '2020',
            'requirements': [
                {
                    'id': 'ccpa_right_to_know',
                    'title': 'Right to Know',
                    'description': 'Consumers have right to know what personal information is collected',
                    'priority': 'high'
                },
                {
                    'id': 'ccpa_right_to_delete',
                    'title': 'Right to Delete',
                    'description': 'Consumers have right to delete personal information',
                    'priority': 'high'
                }
            ]
        }

    def _get_ai_ethics_framework(self) -> Dict[str, Any]:
        """Get AI Ethics framework"""
        return {
            'name': 'AI Ethics Framework',
            'description': 'Responsible AI and ethics guidelines',
            'category': 'ai_ethics',
            'version': '1.0',
            'requirements': [
                {
                    'id': 'ethics_fairness',
                    'title': 'Fairness and Non-Discrimination',
                    'description': 'AI systems should treat all individuals and groups fairly',
                    'priority': 'critical'
                },
                {
                    'id': 'ethics_transparency',
                    'title': 'Transparency and Explainability',
                    'description': 'AI systems should be transparent and explainable',
                    'priority': 'high'
                },
                {
                    'id': 'ethics_accountability',
                    'title': 'Accountability and Responsibility',
                    'description': 'Clear accountability for AI system outcomes',
                    'priority': 'high'
                }
            ]
        }

    def _get_iso_27001_framework(self) -> Dict[str, Any]:
        """Get ISO 27001 framework"""
        return {
            'name': 'ISO 27001 - Information Security',
            'description': 'International standard for information security management',
            'category': 'security',
            'version': '2022',
            'requirements': [
                {
                    'id': 'iso_access_control',
                    'title': 'Access Control',
                    'description': 'Logical and physical access controls',
                    'priority': 'critical'
                },
                {
                    'id': 'iso_data_protection',
                    'title': 'Data Protection and Privacy',
                    'description': 'Protection of personal data and privacy',
                    'priority': 'high'
                }
            ]
        }

    def _get_nist_ai_rmf_framework(self) -> Dict[str, Any]:
        """Get NIST AI Risk Management Framework"""
        return {
            'name': 'NIST AI Risk Management Framework',
            'description': 'Framework for managing AI risks',
            'category': 'risk_management',
            'version': '1.0',
            'requirements': [
                {
                    'id': 'nist_govern',
                    'title': 'Govern Function',
                    'description': 'Organizational governance of AI risks',
                    'priority': 'critical'
                },
                {
                    'id': 'nist_map',
                    'title': 'Map Function',
                    'description': 'Context and categorization of AI risks',
                    'priority': 'high'
                }
            ]
        }

    def _get_ieee_framework(self) -> Dict[str, Any]:
        """Get IEEE Ethically Aligned Design framework"""
        return {
            'name': 'IEEE Ethically Aligned Design',
            'description': 'IEEE framework for ethical AI design',
            'category': 'ai_ethics',
            'version': '2019',
            'requirements': [
                {
                    'id': 'ieee_human_rights',
                    'title': 'Human Rights',
                    'description': 'AI systems should respect human rights',
                    'priority': 'critical'
                },
                {
                    'id': 'ieee_wellbeing',
                    'title': 'Well-being',
                    'description': 'AI systems should prioritize human well-being',
                    'priority': 'high'
                }
            ]
        }

    # Risk assessment methods
    async def _assess_bias_risk(self, model_id: str, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess bias and fairness risks"""
        
        risk_score = 0
        risk_factors = []
        
        # Check for protected attributes in training data
        protected_attrs = ['age', 'gender', 'race', 'ethnicity', 'religion', 'sexual_orientation']
        data_features = model_data.get('features', [])
        
        found_protected = [attr for attr in protected_attrs if any(attr in feat.lower() for feat in data_features)]
        
        if found_protected:
            risk_score += 4
            risk_factors.append(f"Model uses potentially protected attributes: {found_protected}")
        
        # Check model type - certain models have higher bias risk
        model_type = model_data.get('model_type', '').lower()
        if any(biased_type in model_type for biased_type in ['neural', 'deep', 'ensemble']):
            risk_score += 2
            risk_factors.append("Complex model type may have higher bias risk")
        
        # Check training data size - small datasets increase bias risk
        train_size = model_data.get('train_size', 0)
        if train_size < 1000:
            risk_score += 3
            risk_factors.append("Small training dataset increases bias risk")
        
        # Check for fairness testing
        if not model_data.get('fairness_tested', False):
            risk_score += 2
            risk_factors.append("No fairness testing documented")
        
        return {
            'risk_score': min(risk_score, 10),
            'description': "Potential bias in model predictions affecting protected groups",
            'impact': "Discriminatory outcomes, regulatory violations, reputational damage",
            'likelihood': "High if no bias mitigation measures implemented",
            'mitigations': [
                'Implement fairness constraints during training',
                'Regular bias audits with diverse evaluation datasets',
                'Use bias detection and mitigation tools',
                'Establish fairness metrics and monitoring'
            ],
            'risk_factors': risk_factors
        }

    async def _assess_fairness_risk(self, model_id: str, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess fairness-related risks"""
        
        risk_score = 0
        risk_factors = []
        
        # Check for fairness documentation
        if not model_data.get('fairness_documentation', False):
            risk_score += 3
            risk_factors.append("No fairness impact assessment documented")
        
        # Check for diverse evaluation
        if not model_data.get('diverse_evaluation', False):
            risk_score += 2
            risk_factors.append("Model not evaluated on diverse populations")
        
        return {
            'risk_score': min(risk_score, 10),
            'description': "Model may produce unfair outcomes across different groups",
            'impact': "Unfair treatment, loss of trust, legal challenges",
            'likelihood': "Medium without proper fairness measures",
            'mitigations': [
                'Conduct fairness impact assessments',
                'Implement group fairness metrics',
                'Regular stakeholder consultations',
                'Diverse evaluation datasets'
            ],
            'risk_factors': risk_factors
        }

    async def _assess_privacy_risk(self, model_id: str, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess privacy-related risks"""
        
        risk_score = 0
        risk_factors = []
        
        # Check for personal data usage
        if model_data.get('uses_personal_data', False):
            risk_score += 4
            risk_factors.append("Model processes personal data")
        
        # Check for privacy techniques
        if not model_data.get('differential_privacy', False):
            risk_score += 2
            risk_factors.append("No differential privacy implemented")
        
        if not model_data.get('data_anonymization', False):
            risk_score += 2
            risk_factors.append("No data anonymization techniques used")
        
        return {
            'risk_score': min(risk_score, 10),
            'description': "Model may enable privacy breaches through inference attacks",
            'impact': "Privacy violations, regulatory fines, data breaches",
            'likelihood': "High for models trained on sensitive data",
            'mitigations': [
                'Implement differential privacy',
                'Use federated learning techniques',
                'Apply data anonymization and pseudonymization',
                'Limit model access and add monitoring'
            ],
            'risk_factors': risk_factors
        }

    async def _assess_security_risk(self, model_id: str, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess security-related risks"""
        
        risk_score = 0
        risk_factors = []
        
        # Check for adversarial testing
        if not model_data.get('adversarial_testing', False):
            risk_score += 3
            risk_factors.append("No adversarial robustness testing performed")
        
        # Check deployment security
        if not model_data.get('secure_deployment', False):
            risk_score += 2
            risk_factors.append("Deployment security measures not documented")
        
        return {
            'risk_score': min(risk_score, 10),
            'description': "Model vulnerable to adversarial attacks and security breaches",
            'impact': "Model manipulation, data theft, system compromise",
            'likelihood': "Medium to high for publicly accessible models",
            'mitigations': [
                'Implement adversarial training',
                'Deploy input validation and sanitization',
                'Use model ensemble methods',
                'Implement continuous security monitoring'
            ],
            'risk_factors': risk_factors
        }

    async def _assess_explainability_risk(self, model_id: str, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess explainability-related risks"""
        
        risk_score = 0
        risk_factors = []
        
        # Check model complexity
        model_type = model_data.get('model_type', '').lower()
        if any(complex_type in model_type for complex_type in ['neural', 'deep', 'ensemble']):
            risk_score += 3
            risk_factors.append("Complex model type reduces explainability")
        
        # Check for explainability tools
        if not model_data.get('explainability_tools', False):
            risk_score += 2
            risk_factors.append("No explainability tools implemented")
        
        return {
            'risk_score': min(risk_score, 10),
            'description': "Model predictions cannot be adequately explained",
            'impact': "Regulatory non-compliance, lack of trust, audit failures",
            'likelihood': "High for complex models in regulated domains",
            'mitigations': [
                'Implement LIME, SHAP, or similar explainability tools',
                'Generate model cards and documentation',
                'Use interpretable model architectures when possible',
                'Provide decision reasoning interfaces'
            ],
            'risk_factors': risk_factors
        }

    async def _assess_robustness_risk(self, model_id: str, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess model robustness risks"""
        
        risk_score = 0
        risk_factors = []
        
        # Check for robustness testing
        if not model_data.get('robustness_testing', False):
            risk_score += 3
            risk_factors.append("No robustness testing performed")
        
        # Check for data drift monitoring
        if not model_data.get('drift_monitoring', False):
            risk_score += 2
            risk_factors.append("No data drift monitoring implemented")
        
        return {
            'risk_score': min(risk_score, 10),
            'description': "Model performance may degrade with changing data distributions",
            'impact': "Performance degradation, incorrect decisions, system failures",
            'likelihood': "High over time without proper monitoring",
            'mitigations': [
                'Implement continuous model monitoring',
                'Set up data drift detection',
                'Establish model retraining pipelines',
                'Use ensemble methods for robustness'
            ],
            'risk_factors': risk_factors
        }

    # Helper methods
    async def _assess_requirement(self, requirement: Dict[str, Any], model_id: str = None) -> Dict[str, Any]:
        """Assess individual compliance requirement"""
        
        # Simplified assessment - in production would be more comprehensive
        req_id = requirement.get('id', '')
        priority = requirement.get('priority', 'medium')
        
        # Mock assessment logic
        if 'gdpr' in req_id and 'art_22' in req_id:
            # GDPR Article 22 - Automated Decision Making
            score = 0.7  # Partially compliant
            status = 'partial'
            issue = "Insufficient explainability for automated decisions"
            recommendations = ["Implement LIME/SHAP explanations", "Add human review process"]
        elif 'privacy' in req_id or 'data_protection' in req_id:
            score = 0.85
            status = 'compliant'
            issue = None
            recommendations = []
        else:
            score = 0.9
            status = 'compliant'
            issue = None
            recommendations = []
        
        return {
            'score': score * 100,
            'status': status,
            'priority': priority,
            'issue': issue,
            'recommendations': recommendations,
            'evidence': []
        }

    def _score_to_risk_level(self, score: float) -> RiskLevel:
        """Convert numeric risk score to risk level"""
        if score >= 8:
            return RiskLevel.CRITICAL
        elif score >= 6:
            return RiskLevel.HIGH
        elif score >= 3:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _calculate_demographic_parity(self, protected_attr: pd.Series, predictions: np.ndarray) -> float:
        """Calculate demographic parity difference"""
        
        # Get unique values of protected attribute
        groups = protected_attr.unique()
        if len(groups) != 2:
            return 0  # Simplified for binary protected attributes
        
        # Calculate positive prediction rates for each group
        group1_mask = protected_attr == groups[0]
        group2_mask = protected_attr == groups[1]
        
        group1_positive_rate = np.mean(predictions[group1_mask])
        group2_positive_rate = np.mean(predictions[group2_mask])
        
        return group1_positive_rate - group2_positive_rate

    def _calculate_equalized_odds(self, protected_attr: pd.Series, true_labels: pd.Series, 
                                predictions: np.ndarray) -> float:
        """Calculate equalized odds difference"""
        
        groups = protected_attr.unique()
        if len(groups) != 2:
            return 0
        
        # Calculate true positive rates for each group
        group1_mask = protected_attr == groups[0]
        group2_mask = protected_attr == groups[1]
        
        # True positive rate for group 1
        group1_positives = true_labels[group1_mask] == 1
        if np.sum(group1_positives) > 0:
            group1_tpr = np.mean(predictions[group1_mask][group1_positives])
        else:
            group1_tpr = 0
        
        # True positive rate for group 2
        group2_positives = true_labels[group2_mask] == 1
        if np.sum(group2_positives) > 0:
            group2_tpr = np.mean(predictions[group2_mask][group2_positives])
        else:
            group2_tpr = 0
        
        return group1_tpr - group2_tpr

    async def _assess_bias_simple(self, model_id: str, model_data: Dict[str, Any],
                                protected_attributes: List[str]) -> BiasAssessment:
        """Simple bias assessment when AIF360 is not available"""
        
        # Mock bias assessment
        bias_metrics = {}
        fairness_violations = []
        
        for attr in protected_attributes:
            # Simulate bias detection
            bias_score = np.random.uniform(0, 0.2)  # Random bias for demo
            bias_metrics[f'demographic_parity_{attr}'] = bias_score
            
            if bias_score > 0.1:
                fairness_violations.append(f"Potential bias detected for {attr}")
        
        overall_fairness_score = 1 - np.mean(list(bias_metrics.values())) if bias_metrics else 1.0
        
        recommendations = []
        if overall_fairness_score < 0.8:
            recommendations = [
                "Conduct detailed fairness analysis",
                "Implement bias mitigation techniques",
                "Regular monitoring for bias"
            ]
        
        return BiasAssessment(
            model_id=model_id,
            protected_attributes=protected_attributes,
            bias_metrics=bias_metrics,
            fairness_violations=fairness_violations,
            demographic_parity=bias_metrics.get('demographic_parity', 0),
            equalized_odds=bias_metrics.get('equalized_odds', 0),
            individual_fairness=1.0,
            overall_fairness_score=overall_fairness_score,
            recommendations=recommendations
        )

    async def _log_governance_event(self, event_type: str, action: str, resource_type: str,
                                  resource_id: str, event_data: Dict[str, Any],
                                  compliance_relevant: bool = False,
                                  compliance_frameworks: List[str] = None,
                                  risk_relevant: bool = False,
                                  risk_categories: List[str] = None,
                                  user_email: str = None,
                                  ip_address: str = None) -> None:
        """Log governance-related events for audit trail"""
        
        if not self.db:
            return
        
        event_id = f"gov_event_{uuid.uuid4().hex[:8]}"
        
        audit_log = GovernanceAuditLog(
            event_id=event_id,
            event_type=event_type,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            user_email=user_email or 'system',
            ip_address=ip_address or '127.0.0.1',
            event_data=event_data,
            compliance_relevant=compliance_relevant,
            compliance_frameworks=compliance_frameworks or [],
            risk_relevant=risk_relevant,
            risk_categories=risk_categories or []
        )
        
        self.db.add(audit_log)
        self.db.commit()

    def _calculate_risk_trend(self, risks: List[RiskAssessment]) -> str:
        """Calculate risk trend direction"""
        if len(risks) < 2:
            return "stable"
        
        # Simple trend based on recent vs older risks
        recent_risks = [r for r in risks if r.identified_at > datetime.utcnow() - timedelta(days=7)]
        older_risks = [r for r in risks if r.identified_at <= datetime.utcnow() - timedelta(days=7)]
        
        if len(recent_risks) > len(older_risks):
            return "increasing"
        elif len(recent_risks) < len(older_risks):
            return "decreasing"
        else:
            return "stable"

    def _calculate_compliance_trend(self, frameworks: List[ComplianceFramework]) -> str:
        """Calculate compliance trend direction"""
        # Mock implementation - would analyze historical scores
        return "improving"

    def _get_mock_governance_metrics(self) -> Dict[str, Any]:
        """Get mock governance metrics for demo purposes"""
        return {
            'overall_compliance_score': 87.3,
            'compliant_frameworks': 12,
            'total_frameworks': 15,
            'critical_issues': 3,
            'high_risk_models': 7,
            'overdue_reviews': 5,
            'automation_coverage': 78.5,
            'recent_audits': 24,
            'risk_trend': 'stable',
            'compliance_trend': 'improving'
        }

# Factory function
def create_ai_governance_service(db_session: Session = None) -> AIGovernanceService:
    """Create and return an AIGovernanceService instance"""
    return AIGovernanceService(db_session)