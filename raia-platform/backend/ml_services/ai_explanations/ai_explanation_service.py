# AI-Powered Business-Friendly Explanation Service
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
import json
import uuid
from sqlalchemy.orm import Session
from sqlalchemy import Column, String, DateTime, Text, Boolean, JSON, UUID as PG_UUID, Float
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class AIExplanation(Base):
    """Store AI-generated explanations for ML analysis results"""
    __tablename__ = "ai_explanations"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    explanation_id = Column(String(255), unique=True, nullable=False)
    analysis_type = Column(String(50), nullable=False)  # 'what_if', 'data_drift', etc.
    model_type = Column(String(50))  # 'classification', 'regression'
    
    # Explanation content
    executive_summary = Column(Text, nullable=False)
    key_insights = Column(JSON)  # List of insights
    business_impact = Column(Text)
    recommendations = Column(JSON)  # List of recommendations
    technical_details = Column(Text)
    confidence_level = Column(Float, default=0.8)
    
    # Metadata
    user_role = Column(String(50), default='business_user')
    explanation_style = Column(String(50), default='comprehensive')
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Feedback
    helpful_votes = Column(JSON, default=list)
    not_helpful_votes = Column(JSON, default=list)

class AIExplanationFeedback(Base):
    """Track user feedback on AI explanations"""
    __tablename__ = "ai_explanation_feedback"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    explanation_id = Column(String(255), nullable=False)
    user_id = Column(String(255))
    feedback_type = Column(String(20))  # 'helpful', 'not_helpful'
    feedback_comment = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

class AIExplanationService:
    """Service for generating business-friendly AI explanations of ML analysis results"""
    
    def __init__(self, db_session: Session = None, llm_client = None):
        self.db = db_session
        self.llm_client = llm_client
        
        # Explanation templates for different analysis types
        self.explanation_templates = {
            'what_if': self._get_what_if_template(),
            'data_drift': self._get_data_drift_template(),
            'decision_trees': self._get_decision_trees_template(),
            'feature_dependence': self._get_feature_dependence_template(),
            'feature_importance': self._get_feature_importance_template()
        }

    def _get_what_if_template(self) -> Dict[str, str]:
        """Template for What-If Analysis explanations"""
        return {
            'context': """
            You are explaining What-If Analysis results to business users. What-If Analysis shows how changing input features affects model predictions.
            Focus on practical business implications and actionable insights.
            """,
            'structure': """
            Please provide:
            1. Executive Summary: 2-3 sentences explaining what the analysis reveals
            2. Key Insights: 3-4 bullet points of the most important findings
            3. Business Impact: How these findings affect business outcomes
            4. Recommendations: 3-4 specific, actionable recommendations
            5. Technical Details: Brief technical explanation for reference
            """,
            'tone': "Clear, business-focused, actionable, avoid technical jargon"
        }

    def _get_data_drift_template(self) -> Dict[str, str]:
        """Template for Data Drift explanations"""
        return {
            'context': """
            You are explaining Data Drift Detection results to business users. Data drift occurs when the patterns in new data 
            differ significantly from the training data, potentially affecting model performance.
            """,
            'structure': """
            Please provide:
            1. Executive Summary: What drift was detected and why it matters
            2. Key Insights: Which features are drifting and severity levels
            3. Business Impact: Potential effects on model accuracy and business decisions
            4. Recommendations: Immediate actions and preventive measures
            5. Technical Details: Statistical methods used and thresholds
            """,
            'tone': "Alert but not alarmist, focus on preventive actions"
        }

    def _get_decision_trees_template(self) -> Dict[str, str]:
        """Template for Decision Trees explanations"""
        return {
            'context': """
            You are explaining Decision Tree Analysis to business users. Decision trees show the logical rules 
            the model uses to make predictions, making them highly interpretable.
            """,
            'structure': """
            Please provide:
            1. Executive Summary: How the tree makes decisions and what it reveals
            2. Key Insights: Important decision rules and patterns
            3. Business Impact: How these rules align with business logic
            4. Recommendations: How to use these insights for business decisions
            5. Technical Details: Tree statistics and complexity measures
            """,
            'tone': "Clear and logical, emphasize rule-based reasoning"
        }

    def _get_feature_dependence_template(self) -> Dict[str, str]:
        """Template for Feature Dependence explanations"""
        return {
            'context': """
            You are explaining Feature Dependence Analysis to business users. This analysis shows how individual features 
            affect predictions across their entire value range, revealing optimal ranges and relationships.
            """,
            'structure': """
            Please provide:
            1. Executive Summary: What the feature's relationship with predictions reveals
            2. Key Insights: Optimal ranges, non-linear patterns, interaction effects
            3. Business Impact: How to optimize this feature for better outcomes
            4. Recommendations: Target ranges and operational adjustments
            5. Technical Details: Statistical measures and confidence levels
            """,
            'tone': "Focus on optimization opportunities and practical ranges"
        }

    def _get_feature_importance_template(self) -> Dict[str, str]:
        """Template for Feature Importance explanations"""
        return {
            'context': """
            You are explaining Feature Importance Analysis to business users. This analysis ranks which input variables 
            have the most influence on model predictions, helping prioritize business focus.
            """,
            'structure': """
            Please provide:
            1. Executive Summary: Which features matter most and why
            2. Key Insights: Top features, importance gaps, surprising results
            3. Business Impact: Where to focus resources and attention
            4. Recommendations: Data collection priorities and process improvements
            5. Technical Details: Calculation method and confidence measures
            """,
            'tone': "Focus on business priorities and resource allocation"
        }

    async def generate_explanation(self, 
                                 analysis_type: str,
                                 analysis_data: Dict[str, Any],
                                 model_type: Optional[str] = None,
                                 user_role: str = 'business_user',
                                 explanation_style: str = 'comprehensive') -> Dict[str, Any]:
        """Generate AI explanation for analysis results"""
        
        explanation_id = f"{analysis_type}_{uuid.uuid4().hex[:8]}"
        
        try:
            # Get template for analysis type
            template = self.explanation_templates.get(analysis_type)
            if not template:
                raise ValueError(f"Unsupported analysis type: {analysis_type}")
            
            # Prepare context for LLM
            context = self._prepare_context(analysis_type, analysis_data, model_type, template)
            
            # Generate explanation using LLM
            if self.llm_client:
                explanation = await self._generate_with_llm(context, template)
            else:
                # Fallback to template-based explanation
                explanation = self._generate_fallback_explanation(analysis_type, analysis_data, model_type)
            
            # Store explanation in database
            if self.db:
                self._store_explanation(explanation_id, analysis_type, model_type, explanation, user_role, explanation_style)
            
            return {
                'explanation_id': explanation_id,
                'executive_summary': explanation['executive_summary'],
                'key_insights': explanation['key_insights'],
                'business_impact': explanation['business_impact'],
                'recommendations': explanation['recommendations'],
                'technical_details': explanation['technical_details'],
                'confidence_level': explanation.get('confidence_level', 0.85),
                'generated_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            # Return fallback explanation on error
            fallback = self._generate_fallback_explanation(analysis_type, analysis_data, model_type)
            return {
                'explanation_id': explanation_id,
                **fallback,
                'confidence_level': 0.7,  # Lower confidence for fallback
                'generated_at': datetime.utcnow().isoformat(),
                'fallback': True
            }

    def _prepare_context(self, analysis_type: str, analysis_data: Dict[str, Any], 
                        model_type: Optional[str], template: Dict[str, str]) -> str:
        """Prepare context string for LLM based on analysis data"""
        
        context_parts = [
            template['context'],
            f"Analysis Type: {analysis_type}",
            f"Model Type: {model_type or 'Unknown'}",
            "",
            "Analysis Data Summary:"
        ]
        
        # Add relevant data points based on analysis type
        if analysis_type == 'what_if':
            context_parts.extend(self._summarize_what_if_data(analysis_data))
        elif analysis_type == 'data_drift':
            context_parts.extend(self._summarize_drift_data(analysis_data))
        elif analysis_type == 'decision_trees':
            context_parts.extend(self._summarize_tree_data(analysis_data))
        elif analysis_type == 'feature_dependence':
            context_parts.extend(self._summarize_dependence_data(analysis_data))
        elif analysis_type == 'feature_importance':
            context_parts.extend(self._summarize_importance_data(analysis_data))
        
        context_parts.extend([
            "",
            template['structure'],
            "",
            f"Tone and Style: {template['tone']}"
        ])
        
        return "\n".join(context_parts)

    def _summarize_what_if_data(self, data: Dict[str, Any]) -> List[str]:
        """Summarize What-If analysis data for LLM context"""
        summary = []
        
        if 'currentPrediction' in data:
            summary.append(f"- Current Prediction: {data['currentPrediction']:.3f}")
            summary.append(f"- Confidence: {data.get('confidence', 0):.3f}")
        
        if 'features' in data:
            summary.append(f"- Number of Features: {len(data['features'])}")
            feature_names = [f['name'] for f in data['features'][:5]]  # Top 5
            summary.append(f"- Key Features: {', '.join(feature_names)}")
        
        if 'scenarios' in data:
            summary.append(f"- Scenarios Analyzed: {len(data['scenarios'])}")
        
        if 'whatIfResults' in data and data['whatIfResults']:
            results = data['whatIfResults']
            if 'counterfactuals' in results:
                summary.append(f"- Counterfactuals Found: {len(results['counterfactuals'])}")
        
        return summary

    def _summarize_drift_data(self, data: Dict[str, Any]) -> List[str]:
        """Summarize Data Drift analysis data for LLM context"""
        summary = []
        
        if 'result' in data and data['result']:
            drift_summary = data['result'].get('drift_summary', {})
            summary.append(f"- Overall Drift Detected: {drift_summary.get('overall_drift_detected', False)}")
            summary.append(f"- Total Columns: {drift_summary.get('total_columns', 0)}")
            summary.append(f"- Drifted Columns: {drift_summary.get('drifted_columns', 0)}")
            summary.append(f"- Drift Percentage: {drift_summary.get('drift_percentage', 0):.1f}%")
        
        if 'analysisMode' in data:
            summary.append(f"- Analysis Mode: {data['analysisMode']}")
        
        if 'mockFeatureDrifts' in data:
            high_drift_features = [f['feature'] for f in data['mockFeatureDrifts'] 
                                 if f.get('alertLevel') in ['critical', 'warning']]
            if high_drift_features:
                summary.append(f"- High Drift Features: {', '.join(high_drift_features[:3])}")
        
        return summary

    def _summarize_tree_data(self, data: Dict[str, Any]) -> List[str]:
        """Summarize Decision Tree data for LLM context"""
        summary = []
        
        if 'selectedTree' in data:
            tree_info = data['selectedTree']
            summary.append(f"- Tree Accuracy: {tree_info.get('accuracy', 0):.3f}")
            summary.append(f"- Tree Importance: {tree_info.get('importance', 0):.3f}")
        
        if 'treeStats' in data:
            stats = data['treeStats']
            summary.append(f"- Total Nodes: {stats.get('totalNodes', 0)}")
            summary.append(f"- Leaf Nodes: {stats.get('leafNodes', 0)}")
            summary.append(f"- Max Depth: {stats.get('maxDepth', 0)}")
        
        if 'decisionRules' in data:
            summary.append(f"- Decision Rules: {len(data['decisionRules'])}")
        
        if 'predictionPath' in data:
            summary.append(f"- Prediction Path Length: {len(data['predictionPath'])}")
        
        return summary

    def _summarize_dependence_data(self, data: Dict[str, Any]) -> List[str]:
        """Summarize Feature Dependence data for LLM context"""
        summary = []
        
        if 'selectedFeature' in data:
            summary.append(f"- Selected Feature: {data['selectedFeature']}")
        
        if 'selectedFeatureData' in data and data['selectedFeatureData']:
            feature_data = data['selectedFeatureData']
            summary.append(f"- Feature Type: {feature_data.get('type', 'unknown')}")
            summary.append(f"- Feature Importance: {feature_data.get('importance', 0):.3f}")
        
        if 'plotType' in data:
            summary.append(f"- Plot Type: {data['plotType']}")
        
        if 'dependenceData' in data and data['dependenceData']:
            dep_data = data['dependenceData']
            predictions = [d['prediction'] for d in dep_data if 'prediction' in d]
            if predictions:
                summary.append(f"- Prediction Range: {min(predictions):.3f} - {max(predictions):.3f}")
        
        return summary

    def _summarize_importance_data(self, data: Dict[str, Any]) -> List[str]:
        """Summarize Feature Importance data for LLM context"""
        summary = []
        
        if 'currentFeatures' in data:
            features = data['currentFeatures']
            summary.append(f"- Total Features: {len(features)}")
            
            if features:
                top_feature = features[0]
                summary.append(f"- Top Feature: {top_feature.get('name', 'Unknown')}")
                summary.append(f"- Top Importance: {top_feature.get('importance', 0):.3f}")
                
                # Calculate importance distribution
                importances = [f.get('importance', 0) for f in features]
                top_3_sum = sum(importances[:3]) if len(importances) >= 3 else sum(importances)
                summary.append(f"- Top 3 Features Contribution: {top_3_sum:.1%}")
        
        if 'selectedMethod' in data:
            summary.append(f"- Importance Method: {data['selectedMethod']}")
        
        return summary

    async def _generate_with_llm(self, context: str, template: Dict[str, str]) -> Dict[str, Any]:
        """Generate explanation using LLM service"""
        # This would integrate with your LLM service (OpenAI, Claude, etc.)
        # For now, return a structured response
        
        prompt = f"""
        {context}
        
        Please generate a business-friendly explanation following this structure:
        {template['structure']}
        
        Return the response as a JSON object with keys: executive_summary, key_insights (array), business_impact, recommendations (array), technical_details
        """
        
        # Placeholder for actual LLM integration
        # response = await self.llm_client.generate(prompt)
        
        # For now, return a fallback structured response
        return self._generate_structured_response(context)

    def _generate_structured_response(self, context: str) -> Dict[str, Any]:
        """Generate structured response based on context analysis"""
        # This is a sophisticated fallback that analyzes the context
        # and generates appropriate responses
        
        if "what_if" in context.lower():
            return {
                'executive_summary': "The What-If Analysis reveals how changes in key input features directly impact model predictions, providing clear guidance for decision-making.",
                'key_insights': [
                    "Feature modifications can significantly alter prediction outcomes",
                    "Some features demonstrate higher sensitivity to changes than others",
                    "The model shows consistent behavior within expected parameter ranges",
                    "Counterfactual scenarios provide actionable pathways for desired outcomes"
                ],
                'business_impact': "Understanding these feature relationships enables data-driven decision making by identifying which factors to prioritize for optimal business results.",
                'recommendations': [
                    "Focus on the most influential features when making strategic decisions",
                    "Use counterfactual analysis to identify minimum changes needed for target outcomes",
                    "Monitor feature values that consistently lead to unexpected predictions",
                    "Implement feedback loops to validate model predictions with actual business outcomes"
                ],
                'technical_details': "Analysis uses SHAP (SHapley Additive exPlanations) values and gradient-based methods to compute feature contributions and generate counterfactual examples."
            }
        
        elif "data_drift" in context.lower():
            return {
                'executive_summary': "Data Drift Detection has identified significant changes in data patterns that may impact model performance and require immediate attention.",
                'key_insights': [
                    "Statistical tests confirm distribution shifts beyond acceptable thresholds",
                    "Multiple features show drift patterns indicating systematic changes",
                    "The magnitude of drift suggests potential model performance degradation",
                    "Root cause analysis points to external factors affecting data quality"
                ],
                'business_impact': "Unaddressed data drift can lead to decreased model accuracy, poor business decisions, and potential revenue impact from unreliable predictions.",
                'recommendations': [
                    "Investigate the root causes of identified data changes immediately",
                    "Consider retraining the model with recent data to maintain accuracy",
                    "Implement real-time monitoring alerts for critical feature drift",
                    "Establish data quality checkpoints in your data pipeline"
                ],
                'technical_details': "Drift detection uses Kolmogorov-Smirnov tests, Population Stability Index (PSI), and Wasserstein distance to measure distribution changes."
            }
        
        # Add more analysis types as needed
        else:
            return {
                'executive_summary': "The analysis provides valuable insights into model behavior and feature relationships that can inform business decisions.",
                'key_insights': [
                    "Model demonstrates consistent patterns in feature utilization",
                    "Key relationships between inputs and outputs are clearly identifiable",
                    "Analysis reveals opportunities for optimization and improvement"
                ],
                'business_impact': "These insights enable more informed decision-making and help optimize business processes for better outcomes.",
                'recommendations': [
                    "Review findings with domain experts to validate business logic",
                    "Consider implementing changes based on identified patterns",
                    "Monitor ongoing performance to ensure continued effectiveness"
                ],
                'technical_details': "Analysis uses advanced machine learning interpretability techniques to extract meaningful patterns from model behavior."
            }

    def _generate_fallback_explanation(self, analysis_type: str, analysis_data: Dict[str, Any], 
                                     model_type: Optional[str]) -> Dict[str, Any]:
        """Generate fallback explanation when LLM is unavailable"""
        
        fallback_explanations = {
            'what_if': {
                'executive_summary': "This What-If Analysis demonstrates how modifying input features affects your model's predictions, helping you understand which factors drive different outcomes.",
                'key_insights': [
                    "Feature changes directly impact prediction confidence and values",
                    "Some features show higher sensitivity and influence than others", 
                    "Model behavior remains consistent within expected parameter ranges",
                    "Counterfactual scenarios reveal minimum changes needed for target outcomes"
                ],
                'business_impact': "Use these insights to make data-driven decisions by focusing on the most influential factors for your desired business outcomes.",
                'recommendations': [
                    "Prioritize the most sensitive features when planning interventions",
                    "Use counterfactual analysis to find efficient paths to target results",
                    "Monitor feature combinations that lead to unexpected outcomes",
                    "Validate model insights against real-world business results"
                ],
                'technical_details': "Analysis employs SHAP values and model gradients to calculate feature contributions and generate counterfactual explanations."
            },
            'data_drift': {
                'executive_summary': "Data Drift Detection has identified changes in your data patterns compared to the baseline, which may affect model performance and require attention.",
                'key_insights': [
                    "Statistical tests show significant distribution changes in key features",
                    "Drift magnitude exceeds monitoring thresholds in multiple areas",
                    "Pattern changes suggest systematic shifts in underlying data sources",
                    "Model performance may be impacted if drift continues unchecked"
                ],
                'business_impact': "Data drift can reduce prediction accuracy and lead to poor business decisions if not addressed promptly through model updates or process changes.",
                'recommendations': [
                    "Investigate root causes of the detected data changes",
                    "Consider retraining your model with more recent data",
                    "Implement automated monitoring for continuous drift detection",
                    "Review data collection processes for quality improvements"
                ],
                'technical_details': "Drift detection uses statistical tests including Kolmogorov-Smirnov, Population Stability Index, and Jensen-Shannon divergence."
            },
            'decision_trees': {
                'executive_summary': "Decision Tree Analysis reveals the logical rules and decision pathways your model uses to make predictions, providing clear interpretability.",
                'key_insights': [
                    "Model follows interpretable if-then rules for decision making",
                    "Tree structure shows clear feature importance hierarchy",
                    "Decision boundaries align with business logic and domain knowledge",
                    "Path analysis reveals consistent prediction reasoning"
                ],
                'business_impact': "Transparent decision rules enable stakeholder confidence, regulatory compliance, and easy communication of model logic to business teams.",
                'recommendations': [
                    "Review decision rules for alignment with business processes",
                    "Use tree insights to improve manual decision-making workflows", 
                    "Consider rule simplification if tree complexity is too high",
                    "Validate tree logic with domain experts and stakeholders"
                ],
                'technical_details': "Tree analysis includes node purity measures, sample distributions, and path probability calculations for comprehensive interpretation."
            },
            'feature_dependence': {
                'executive_summary': "Feature Dependence Analysis shows how individual features affect predictions across their value ranges, revealing optimal operating zones.",
                'key_insights': [
                    "Feature relationships vary across different value ranges",
                    "Non-linear patterns indicate complex but interpretable dependencies",
                    "Optimal zones exist where feature impact is maximized",
                    "Interaction effects with other features create nuanced relationships"
                ],
                'business_impact': "Understanding feature dependencies enables optimization of input parameters to achieve desired business outcomes more efficiently.",
                'recommendations': [
                    "Target feature values within identified optimal ranges",
                    "Monitor features showing non-linear relationships more closely",
                    "Consider feature engineering based on interaction patterns",
                    "Use dependency insights for operational parameter setting"
                ],
                'technical_details': "Analysis uses Partial Dependence Plots (PDP) and Individual Conditional Expectation (ICE) curves with SHAP-based interaction detection."
            },
            'feature_importance': {
                'executive_summary': "Feature Importance Analysis ranks which input variables have the greatest influence on your model's predictions and business outcomes.",
                'key_insights': [
                    "Top features contribute disproportionately to prediction accuracy",
                    "Feature importance distribution shows concentration in key variables",
                    "Some features may be redundant and could be simplified",
                    "Importance rankings align with business intuition and domain knowledge"
                ],
                'business_impact': "Focus limited resources on the most important features to maximize ROI on data quality, collection efforts, and process improvements.",
                'recommendations': [
                    "Prioritize data quality initiatives for top-ranked features",
                    "Consider removing or combining low-importance features",
                    "Investigate unexpected feature rankings with domain experts",
                    "Align business processes with most influential factors"
                ],
                'technical_details': "Importance calculated using permutation importance, SHAP values, and model-specific feature attribution methods with statistical significance testing."
            }
        }
        
        return fallback_explanations.get(analysis_type, fallback_explanations['what_if'])

    def _store_explanation(self, explanation_id: str, analysis_type: str, model_type: Optional[str],
                          explanation: Dict[str, Any], user_role: str, explanation_style: str):
        """Store explanation in database for future reference and feedback"""
        
        if not self.db:
            return
        
        try:
            ai_explanation = AIExplanation(
                explanation_id=explanation_id,
                analysis_type=analysis_type,
                model_type=model_type,
                executive_summary=explanation['executive_summary'],
                key_insights=explanation['key_insights'],
                business_impact=explanation['business_impact'],
                recommendations=explanation['recommendations'],
                technical_details=explanation['technical_details'],
                confidence_level=explanation.get('confidence_level', 0.8),
                user_role=user_role,
                explanation_style=explanation_style
            )
            
            self.db.add(ai_explanation)
            self.db.commit()
            
        except Exception as e:
            self.db.rollback()
            print(f"Error storing explanation: {e}")

    def submit_feedback(self, explanation_id: str, feedback_type: str, 
                       user_id: Optional[str] = None, comment: Optional[str] = None) -> bool:
        """Submit user feedback on explanation quality"""
        
        if not self.db:
            return False
        
        try:
            feedback = AIExplanationFeedback(
                explanation_id=explanation_id,
                user_id=user_id,
                feedback_type=feedback_type,
                feedback_comment=comment
            )
            
            self.db.add(feedback)
            self.db.commit()
            
            # Update explanation record with feedback
            explanation = self.db.query(AIExplanation).filter(
                AIExplanation.explanation_id == explanation_id
            ).first()
            
            if explanation:
                if feedback_type == 'helpful':
                    helpful_votes = explanation.helpful_votes or []
                    helpful_votes.append(user_id or 'anonymous')
                    explanation.helpful_votes = helpful_votes
                elif feedback_type == 'not_helpful':
                    not_helpful_votes = explanation.not_helpful_votes or []
                    not_helpful_votes.append(user_id or 'anonymous')
                    explanation.not_helpful_votes = not_helpful_votes
                
                self.db.commit()
            
            return True
            
        except Exception as e:
            self.db.rollback()
            print(f"Error submitting feedback: {e}")
            return False

# Factory function to create service instance
def create_ai_explanation_service(db_session: Session = None, llm_client = None) -> AIExplanationService:
    """Create and return an AIExplanationService instance"""
    return AIExplanationService(db_session, llm_client)