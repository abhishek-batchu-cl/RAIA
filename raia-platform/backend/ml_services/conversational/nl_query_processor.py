# Natural Language Query Processing Service
import re
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from sqlalchemy.orm import Session
from sqlalchemy import Column, String, DateTime, Text, JSON, UUID as PG_UUID
from sqlalchemy.ext.declarative import declarative_base
import uuid

Base = declarative_base()

@dataclass
class QueryIntent:
    intent: str  # 'performance', 'drift', 'bias', 'feature_importance', 'explanation', 'general'
    entities: Dict[str, Any]  # Extracted entities like model names, time periods, metrics
    confidence: float
    query_type: str  # 'question', 'command', 'request'
    suggested_action: Optional[str] = None

@dataclass 
class QueryResponse:
    content: str
    metadata: Dict[str, Any]
    confidence: float
    query_type: str  # 'chart', 'insight', 'explanation', 'recommendation'
    sources: List[str]
    follow_up_suggestions: List[str] = None

class ConversationHistory(Base):
    """Store conversation history for context and personalization"""
    __tablename__ = "conversation_history"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(String(255), nullable=False)
    user_id = Column(String(255))
    user_query = Column(Text, nullable=False)
    intent = Column(String(100))
    entities = Column(JSON)
    response = Column(Text, nullable=False)
    response_metadata = Column(JSON)
    confidence = Column(String(20))
    timestamp = Column(DateTime, default=datetime.utcnow)

class NaturalLanguageQueryProcessor:
    """Process natural language queries and generate contextual responses about ML models and data"""
    
    def __init__(self, db_session: Session = None):
        self.db = db_session
        
        # Intent patterns for query classification
        self.intent_patterns = {
            'performance': [
                r'how.*perform|performance|accuracy|precision|recall|f1|score|metric',
                r'model.*doing|model.*good|model.*bad|working.*well',
                r'success.*rate|error.*rate|prediction.*quality'
            ],
            'drift': [
                r'data.*drift|distribution.*change|feature.*drift|drift.*detect',
                r'data.*shift|pattern.*change|statistical.*change',
                r'baseline.*compare|historical.*data'
            ],
            'bias': [
                r'bias|fair|discrimination|equit|demographic.*parity',
                r'protected.*group|unfair|prejudice|equal.*treatment',
                r'disparate.*impact|calibration'
            ],
            'feature_importance': [
                r'feature.*important|important.*feature|top.*feature',
                r'which.*feature|most.*influence|key.*factor',
                r'contribution|impact.*prediction|driver'
            ],
            'explanation': [
                r'explain|why.*predict|how.*decide|reason.*decision',
                r'understand.*model|interpret|make.*sense',
                r'customer.*explanation|business.*explanation'
            ],
            'comparison': [
                r'compare.*model|model.*comparison|which.*better',
                r'versus|vs|against|benchmark',
                r'best.*model|top.*perform'
            ],
            'trends': [
                r'trend|over.*time|historical|past.*week|last.*month',
                r'getting.*better|getting.*worse|improving|declining',
                r'time.*series|temporal'
            ]
        }
        
        # Entity extraction patterns
        self.entity_patterns = {
            'model_name': r'(?:model|algorithm)\s+([a-zA-Z0-9_-]+)|([a-zA-Z0-9_-]+)(?:\s+model)',
            'time_period': r'(?:past|last|over)\s+(week|month|day|year|quarter)',
            'metric': r'(accuracy|precision|recall|f1|auc|mse|mae|rmse)',
            'threshold': r'(?:above|below|over|under)\s+(\d+(?:\.\d+)?)',
            'feature': r'feature\s+([a-zA-Z0-9_]+)|([a-zA-Z0-9_]+)\s+feature'
        }
        
        # Response templates
        self.response_templates = {
            'performance': self._get_performance_template(),
            'drift': self._get_drift_template(),
            'bias': self._get_bias_template(),
            'feature_importance': self._get_feature_importance_template(),
            'explanation': self._get_explanation_template(),
            'general': self._get_general_template()
        }

    async def process_query(self, query: str, user_id: str = None, 
                          session_id: str = None) -> QueryResponse:
        """Process a natural language query and return a contextual response"""
        
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Classify intent and extract entities
        intent = self._classify_intent(query)
        
        # Generate response based on intent
        response = await self._generate_response(query, intent, user_id)
        
        # Store conversation in database
        if self.db:
            await self._store_conversation(session_id, user_id, query, intent, response)
        
        return response

    def _classify_intent(self, query: str) -> QueryIntent:
        """Classify the query intent and extract entities"""
        
        query_lower = query.lower()
        intent_scores = {}
        
        # Calculate scores for each intent
        for intent_name, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, query_lower))
                score += matches
            intent_scores[intent_name] = score
        
        # Get the highest scoring intent
        if not any(intent_scores.values()):
            best_intent = 'general'
            confidence = 0.5
        else:
            best_intent = max(intent_scores.items(), key=lambda x: x[1])[0]
            confidence = min(intent_scores[best_intent] / 3.0, 1.0)  # Normalize
        
        # Extract entities
        entities = self._extract_entities(query)
        
        # Determine query type
        query_type = 'question'
        if any(word in query_lower for word in ['show', 'display', 'generate', 'create']):
            query_type = 'command'
        elif any(word in query_lower for word in ['help', 'how to', 'can you']):
            query_type = 'request'
        
        return QueryIntent(
            intent=best_intent,
            entities=entities,
            confidence=confidence,
            query_type=query_type
        )

    def _extract_entities(self, query: str) -> Dict[str, Any]:
        """Extract entities from the query"""
        entities = {}
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                if entity_type == 'model_name':
                    # Handle tuple matches from alternation
                    entities[entity_type] = [match[0] or match[1] for match in matches if match[0] or match[1]]
                elif entity_type == 'threshold':
                    entities[entity_type] = float(matches[0])
                else:
                    entities[entity_type] = matches[0] if len(matches) == 1 else matches
        
        return entities

    async def _generate_response(self, query: str, intent: QueryIntent, 
                               user_id: str = None) -> QueryResponse:
        """Generate a contextual response based on the query intent"""
        
        template = self.response_templates.get(intent.intent, self.response_templates['general'])
        
        # Get relevant data based on intent
        data = await self._get_relevant_data(intent, user_id)
        
        # Generate response using template and data
        response_content = template['generator'](query, intent, data)
        
        # Determine response metadata
        metadata = {
            'intent': intent.intent,
            'entities': intent.entities,
            'confidence': intent.confidence,
            'data_sources': data.get('sources', []),
            'query_type': template['response_type']
        }
        
        # Generate follow-up suggestions
        follow_ups = template.get('follow_ups', [])
        
        return QueryResponse(
            content=response_content,
            metadata=metadata,
            confidence=intent.confidence,
            query_type=template['response_type'],
            sources=data.get('sources', []),
            follow_up_suggestions=follow_ups
        )

    async def _get_relevant_data(self, intent: QueryIntent, user_id: str = None) -> Dict[str, Any]:
        """Retrieve relevant data based on the query intent"""
        
        # This would integrate with your actual data sources
        # For now, returning mock data
        
        if intent.intent == 'performance':
            return {
                'models': [
                    {
                        'name': 'credit_scoring_v2',
                        'accuracy': 0.847,
                        'precision': 0.823,
                        'recall': 0.871,
                        'f1': 0.846,
                        'trend': 'decreasing',
                        'last_updated': datetime.now() - timedelta(hours=2)
                    }
                ],
                'sources': ['model_monitoring', 'performance_metrics']
            }
        elif intent.intent == 'drift':
            return {
                'drift_results': [
                    {
                        'feature': 'customer_age',
                        'drift_magnitude': 0.73,
                        'baseline_mean': 35.2,
                        'current_mean': 42.8,
                        'impact': 'high'
                    },
                    {
                        'feature': 'annual_income',
                        'drift_magnitude': 0.61,
                        'impact': 'medium'
                    }
                ],
                'sources': ['drift_detection', 'feature_monitoring']
            }
        elif intent.intent == 'bias':
            return {
                'bias_results': {
                    'gender_bias': 0.023,
                    'age_bias': 0.018,
                    'geographic_bias': 0.062,
                    'improvement': 0.23,
                    'status': 'improved'
                },
                'sources': ['bias_detection', 'fairness_metrics']
            }
        elif intent.intent == 'feature_importance':
            return {
                'features': [
                    {'name': 'customer_age', 'importance': 0.342, 'rank': 1},
                    {'name': 'annual_income', 'importance': 0.289, 'rank': 2},
                    {'name': 'credit_score', 'importance': 0.156, 'rank': 3},
                    {'name': 'account_balance', 'importance': 0.134, 'rank': 4},
                    {'name': 'employment_type', 'importance': 0.098, 'rank': 5}
                ],
                'sources': ['feature_importance', 'shap_analysis']
            }
        
        return {'sources': []}

    def _get_performance_template(self) -> Dict[str, Any]:
        """Template for performance-related queries"""
        def generate_response(query: str, intent: QueryIntent, data: Dict[str, Any]) -> str:
            if not data.get('models'):
                return "I don't have current performance data available. Please check your model monitoring dashboard."
            
            model = data['models'][0]
            trend_indicator = "↓" if model['trend'] == 'decreasing' else "↑" if model['trend'] == 'increasing' else "→"
            
            return f"""Based on your model monitoring data, here's the performance summary:

**{model['name'].replace('_', ' ').title()} Performance:**
• Current Accuracy: {model['accuracy']:.1%} {trend_indicator}
• Precision: {model['precision']:.1%}
• Recall: {model['recall']:.1%}
• F1 Score: {model['f1']:.1%}

**Key Findings:**
• Performance {'drop' if model['trend'] == 'decreasing' else 'improvement'} detected in recent data
• Last updated: {model['last_updated'].strftime('%H:%M:%S')} ago
• {'Recommend investigating feature drift and data quality' if model['trend'] == 'decreasing' else 'Model is performing within expected ranges'}

Would you like me to show you the detailed performance trends or investigate specific metrics?"""
        
        return {
            'generator': generate_response,
            'response_type': 'insight',
            'follow_ups': [
                "Show me performance trends over time",
                "Compare with other models",
                "Investigate performance drop causes"
            ]
        }

    def _get_drift_template(self) -> Dict[str, Any]:
        """Template for data drift queries"""
        def generate_response(query: str, intent: QueryIntent, data: Dict[str, Any]) -> str:
            drift_results = data.get('drift_results', [])
            if not drift_results:
                return "No significant data drift detected in your current models. All features are within acceptable ranges."
            
            high_drift_features = [r for r in drift_results if r.get('impact') == 'high']
            
            response = "I've detected significant data drift in several features:\n\n"
            
            for result in drift_results[:3]:  # Show top 3
                response += f"**{result['feature'].replace('_', ' ').title()}**: {result['drift_magnitude']:.2f} drift magnitude\n"
                if 'baseline_mean' in result:
                    response += f"  - Historical mean: {result['baseline_mean']}\n"
                    response += f"  - Current mean: {result['current_mean']}\n"
                response += f"  - Impact: {result['impact'].title()}\n\n"
            
            response += "**Recommendations:**\n"
            response += "1. Investigate external factors causing distribution shifts\n"
            response += "2. Consider retraining with recent data\n"
            response += "3. Implement drift monitoring alerts\n\n"
            response += "Would you like me to generate a detailed drift report or show distribution comparisons?"
            
            return response
        
        return {
            'generator': generate_response,
            'response_type': 'chart',
            'follow_ups': [
                "Show drift visualization",
                "Investigate drift causes",
                "Set up drift monitoring"
            ]
        }

    def _get_bias_template(self) -> Dict[str, Any]:
        """Template for bias and fairness queries"""
        def generate_response(query: str, intent: QueryIntent, data: Dict[str, Any]) -> str:
            bias_data = data.get('bias_results', {})
            if not bias_data:
                return "Bias assessment data is not currently available. Please run a fairness analysis first."
            
            status = "improved" if bias_data.get('status') == 'improved' else "concerning"
            
            response = f"{'Great news!' if status == 'improved' else 'Attention needed:'} Your model fairness analysis:\n\n"
            response += "**Bias Assessment Results:**\n"
            
            gender_status = "✅ Within acceptable range" if bias_data.get('gender_bias', 0) < 0.05 else "⚠️ Above threshold"
            age_status = "✅ Within acceptable range" if bias_data.get('age_bias', 0) < 0.05 else "⚠️ Above threshold"
            geo_status = "✅ Within acceptable range" if bias_data.get('geographic_bias', 0) < 0.05 else "⚠️ Above threshold"
            
            response += f"• **Gender Bias**: {gender_status} ({bias_data.get('gender_bias', 0):.1%})\n"
            response += f"• **Age Bias**: {age_status} ({bias_data.get('age_bias', 0):.1%})\n"
            response += f"• **Geographic Bias**: {geo_status} ({bias_data.get('geographic_bias', 0):.1%})\n\n"
            
            if bias_data.get('improvement'):
                response += "**Recent Improvements:**\n"
                response += f"• Overall bias reduced by {bias_data['improvement']:.1%} after mitigation\n"
                response += "• Demographic parity achieved across protected groups\n\n"
            
            response += "Would you like me to generate a comprehensive fairness report for compliance?"
            
            return response
        
        return {
            'generator': generate_response,
            'response_type': 'insight',
            'follow_ups': [
                "Generate fairness report",
                "Show bias mitigation strategies",
                "Check compliance requirements"
            ]
        }

    def _get_feature_importance_template(self) -> Dict[str, Any]:
        """Template for feature importance queries"""
        def generate_response(query: str, intent: QueryIntent, data: Dict[str, Any]) -> str:
            features = data.get('features', [])
            if not features:
                return "Feature importance data is not available. Please run feature analysis first."
            
            response = "Here are the most important features for your model:\n\n"
            response += "**Top Features by Importance:**\n"
            
            for i, feature in enumerate(features[:5], 1):
                response += f"{i}. **{feature['name'].replace('_', ' ').title()}** ({feature['importance']:.1%})\n"
            
            top_3_contribution = sum(f['importance'] for f in features[:3])
            response += f"\n**Key Insights:**\n"
            response += f"• Top 3 features contribute {top_3_contribution:.1%} of predictions\n"
            response += f"• Total of {len(features)} features analyzed\n"
            response += "• Strong feature hierarchy indicates model interpretability\n\n"
            response += "Would you like to explore feature dependencies or see the complete ranking?"
            
            return response
        
        return {
            'generator': generate_response,
            'response_type': 'chart',
            'follow_ups': [
                "Show feature dependencies",
                "Analyze feature interactions",
                "Compare feature importance across models"
            ]
        }

    def _get_explanation_template(self) -> Dict[str, Any]:
        """Template for explanation queries"""
        def generate_response(query: str, intent: QueryIntent, data: Dict[str, Any]) -> str:
            return """I can help explain model predictions in business-friendly terms:

**For Customer-Facing Explanations:**
"Your application was carefully reviewed using our automated decision system. The decision considered several key factors:

• **Credit History** (High Impact): Your payment record influenced the decision positively
• **Income Stability** (Medium Impact): Your employment history was considered favorably  
• **Financial Position** (Medium Impact): Your current financial status is within acceptable ranges
• **Application Completeness** (Low Impact): All required information was provided"

**For Internal Teams:**
• SHAP values available for detailed feature analysis
• Confidence score: 87.3%
• Counterfactual explanations generated
• Regulatory compliance factors included

Would you like me to customize this explanation for a specific case or create a template?"""
        
        return {
            'generator': generate_response,
            'response_type': 'explanation',
            'follow_ups': [
                "Customize explanation for specific case",
                "Generate customer-friendly template",
                "Show technical SHAP analysis"
            ]
        }

    def _get_general_template(self) -> Dict[str, Any]:
        """Template for general queries"""
        def generate_response(query: str, intent: QueryIntent, data: Dict[str, Any]) -> str:
            return """I understand you're asking about your ML models and data. I can help you with:

**Model Performance**
• Accuracy, precision, recall metrics
• Performance trends over time
• Model comparison and benchmarking

**Data Analysis**
• Feature importance and relationships
• Data quality and drift detection
• Statistical insights and patterns

**Explainability**
• Prediction explanations (SHAP, LIME)
• Feature impact analysis
• Business-friendly interpretations

**Fairness & Bias**
• Bias detection across protected groups
• Fairness metrics and compliance
• Mitigation strategies

Could you rephrase your question or try one of the suggested queries? I'm here to help make your data more understandable!"""
        
        return {
            'generator': generate_response,
            'response_type': 'recommendation',
            'follow_ups': [
                "How is my model performing?",
                "Check for data drift",
                "Explain a specific prediction"
            ]
        }

    async def _store_conversation(self, session_id: str, user_id: str, query: str, 
                                intent: QueryIntent, response: QueryResponse):
        """Store conversation in database for learning and personalization"""
        if not self.db:
            return
        
        try:
            conversation = ConversationHistory(
                session_id=session_id,
                user_id=user_id,
                user_query=query,
                intent=intent.intent,
                entities=intent.entities,
                response=response.content,
                response_metadata=response.metadata,
                confidence=str(intent.confidence)
            )
            
            self.db.add(conversation)
            self.db.commit()
            
        except Exception as e:
            self.db.rollback()
            print(f"Error storing conversation: {e}")

# Factory function
def create_nl_query_processor(db_session: Session = None) -> NaturalLanguageQueryProcessor:
    """Create and return a NaturalLanguageQueryProcessor instance"""
    return NaturalLanguageQueryProcessor(db_session)