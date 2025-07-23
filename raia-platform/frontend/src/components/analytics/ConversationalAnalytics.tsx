import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  MessageCircle, Send, Mic, MicOff, Brain, BarChart3, 
  TrendingUp, AlertCircle, Download, Copy, ThumbsUp, 
  ThumbsDown, Sparkles, User, Bot, Loader2
} from 'lucide-react';
import Card from '@/components/common/Card';
import Button from '@/components/common/Button';
import { apiClient } from '@/services/api';

interface ChatMessage {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  metadata?: {
    query_type?: 'chart' | 'insight' | 'explanation' | 'recommendation';
    chart_data?: any;
    confidence?: number;
    sources?: string[];
  };
  loading?: boolean;
}

interface SuggestedQuery {
  id: string;
  text: string;
  category: 'performance' | 'drift' | 'bias' | 'business';
  icon: React.ReactNode;
}

const ConversationalAnalytics: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: '1',
      type: 'assistant',
      content: "Hello! I'm your AI Analytics Assistant. I can help you understand your ML models, analyze data patterns, and provide insights. Try asking me something like:\n\n• \"How is my credit scoring model performing?\"\n• \"Show me data drift in customer features\"\n• \"Which features are most important?\"\n• \"Explain why this prediction was made\"",
      timestamp: new Date(),
      metadata: { confidence: 1.0 }
    }
  ]);
  
  const [currentMessage, setCurrentMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [showSuggestions, setShowSuggestions] = useState(true);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  
  const suggestedQueries: SuggestedQuery[] = [
    {
      id: '1',
      text: "How is my credit scoring model performing this week?",
      category: 'performance',
      icon: <TrendingUp className="w-4 h-4" />
    },
    {
      id: '2',
      text: "Show me features with data drift above 0.5",
      category: 'drift',
      icon: <AlertCircle className="w-4 h-4" />
    },
    {
      id: '3',
      text: "Which features are most important for loan approvals?",
      category: 'performance',
      icon: <BarChart3 className="w-4 h-4" />
    },
    {
      id: '4',
      text: "Check for bias in my model predictions",
      category: 'bias',
      icon: <Brain className="w-4 h-4" />
    },
    {
      id: '5',
      text: "What's the ROI impact of my fraud detection model?",
      category: 'business',
      icon: <TrendingUp className="w-4 h-4" />
    },
    {
      id: '6',
      text: "Explain this loan rejection to the customer",
      category: 'performance',
      icon: <MessageCircle className="w-4 h-4" />
    }
  ];

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSendMessage = async (messageText?: string) => {
    const text = messageText || currentMessage.trim();
    if (!text || isLoading) return;

    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      type: 'user',
      content: text,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setCurrentMessage('');
    setIsLoading(true);
    setShowSuggestions(false);

    // Add loading message
    const loadingMessage: ChatMessage = {
      id: `loading-${Date.now()}`,
      type: 'assistant',
      content: 'Analyzing your query and generating insights...',
      timestamp: new Date(),
      loading: true
    };

    setMessages(prev => [...prev, loadingMessage]);

    try {
      // Process the query
      const response = await processNaturalLanguageQuery(text);
      
      // Remove loading message and add response
      setMessages(prev => {
        const withoutLoading = prev.filter(m => !m.loading);
        return [...withoutLoading, response];
      });
      
    } catch (error) {
      console.error('Error processing query:', error);
      const errorMessage: ChatMessage = {
        id: `error-${Date.now()}`,
        type: 'assistant',
        content: "I apologize, but I encountered an issue processing your request. Please try rephrasing your question or ask something else.",
        timestamp: new Date(),
        metadata: { confidence: 0.1 }
      };
      
      setMessages(prev => {
        const withoutLoading = prev.filter(m => !m.loading);
        return [...withoutLoading, errorMessage];
      });
    } finally {
      setIsLoading(false);
    }
  };

  const processNaturalLanguageQuery = async (query: string): Promise<ChatMessage> => {
    // Simulate AI processing delay
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    const queryLower = query.toLowerCase();
    let response: ChatMessage;
    
    if (queryLower.includes('performance') || queryLower.includes('accuracy')) {
      response = {
        id: `response-${Date.now()}`,
        type: 'assistant',
        content: `Based on your model monitoring data, here's the performance summary:

**Credit Scoring Model v2:**
• Current Accuracy: 84.7% (↓ 12% from last week)
• Precision: 82.3%
• Recall: 87.1%
• F1 Score: 84.6%

**Key Findings:**
• Performance drop detected in the past 7 days
• Likely caused by data distribution changes in customer demographics
• Recommend investigating feature drift and retraining

Would you like me to show you the detailed drift analysis or create a performance report?`,
        timestamp: new Date(),
        metadata: {
          query_type: 'insight',
          confidence: 0.92,
          chart_data: {
            accuracy_trend: [0.967, 0.963, 0.954, 0.947, 0.847],
            dates: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
          },
          sources: ['model_monitoring', 'performance_metrics']
        }
      };
    } else if (queryLower.includes('drift') || queryLower.includes('distribution')) {
      response = {
        id: `response-${Date.now()}`,
        type: 'assistant',
        content: `I've detected significant data drift in several features:

**High Drift Features (>0.5):**
• **Customer Age**: 0.73 drift magnitude
  - Historical mean: 35.2 years
  - Current mean: 42.8 years
  - Impact: High (affects 67% of predictions)

• **Annual Income**: 0.61 drift magnitude
  - Distribution shifted towards higher income brackets
  - Impact: Medium (affects 34% of predictions)

**Recommendations:**
1. Investigate external factors causing age demographic shift
2. Consider retraining with recent data
3. Implement drift monitoring alerts

Would you like me to generate a detailed drift report or show you the distribution comparisons?`,
        timestamp: new Date(),
        metadata: {
          query_type: 'chart',
          confidence: 0.89,
          sources: ['drift_detection', 'feature_monitoring']
        }
      };
    } else if (queryLower.includes('feature') && queryLower.includes('important')) {
      response = {
        id: `response-${Date.now()}`,
        type: 'assistant',
        content: `Here are the most important features for your loan approval model:

**Top 5 Features by Importance:**
1. **Customer Age** (34.2%) - Primary risk factor
2. **Annual Income** (28.9%) - Strong predictor of repayment ability
3. **Credit Score** (15.6%) - Historical creditworthiness
4. **Account Balance** (13.4%) - Financial stability indicator
5. **Employment Type** (9.8%) - Income stability factor

**Insights:**
• Top 3 features contribute to 78.7% of predictions
• Age and income show strong interaction effects
• Consider feature engineering: Income-to-Age ratio could improve performance by ~8%

Would you like to explore feature dependencies or see the complete importance ranking?`,
        timestamp: new Date(),
        metadata: {
          query_type: 'chart',
          confidence: 0.95,
          sources: ['feature_importance', 'shap_analysis']
        }
      };
    } else if (queryLower.includes('bias') || queryLower.includes('fair')) {
      response = {
        id: `response-${Date.now()}`,
        type: 'assistant',
        content: `Great news! Your model fairness has improved significantly:

**Bias Assessment Results:**
• **Gender Bias**: ✅ Within acceptable range (2.3% vs 5% threshold)
• **Age Bias**: ✅ Demographic parity achieved (1.8% difference)
• **Geographic Bias**: ⚠️ Minor concern (6.2% - just above 5% threshold)

**Recent Improvements:**
• Gender bias reduced by 23% after bias mitigation
• Equalized odds improved across all demographics
• Calibration maintained while improving fairness

**Recommendations:**
• Monitor geographic bias in rural vs urban predictions
• Consider additional data collection in underrepresented regions

Would you like me to generate a comprehensive fairness report for compliance?`,
        timestamp: new Date(),
        metadata: {
          query_type: 'insight',
          confidence: 0.91,
          sources: ['bias_detection', 'fairness_metrics']
        }
      };
    } else if (queryLower.includes('roi') || queryLower.includes('business') || queryLower.includes('revenue')) {
      response = {
        id: `response-${Date.now()}`,
        type: 'assistant',
        content: `Your ML models are delivering strong business value:

**Financial Impact Summary:**
• **Total Revenue Generated**: $2.45M (this quarter)
• **Cost Savings**: $450K (reduced manual reviews)
• **ROI**: 15% improvement over baseline
• **False Positive Reduction**: 34% (saving $180K in unnecessary investigations)

**Model-Specific Performance:**
• **Credit Scoring**: $1.8M revenue, 89% accuracy
• **Fraud Detection**: $650K savings, 94% precision
• **Risk Assessment**: $200K efficiency gains

**Growth Opportunities:**
• A/B testing new model variant could add $320K annually
• Feature optimization could improve ROI by additional 8-12%

Would you like a detailed business case report or ROI projection analysis?`,
        timestamp: new Date(),
        metadata: {
          query_type: 'insight',
          confidence: 0.87,
          sources: ['business_metrics', 'roi_analysis']
        }
      };
    } else if (queryLower.includes('explain') || queryLower.includes('why')) {
      response = {
        id: `response-${Date.now()}`,
        type: 'assistant',
        content: `I can help explain model predictions in business-friendly terms:

**For Customer-Facing Explanations:**
"Your loan application was carefully reviewed using our automated system. The decision was based on several key factors:

• **Credit History** (High Impact): Your excellent payment record positively influenced the decision
• **Income Stability** (Medium Impact): Your steady employment history was considered favorably  
• **Debt-to-Income Ratio** (Medium Impact): Your current debt level is within acceptable ranges
• **Application Completeness** (Low Impact): All required information was provided

The system also considered industry standards and regulatory requirements to ensure fair and consistent decisions."

**For Internal Teams:**
• SHAP values show top contributing features
• Confidence score: 87.3%
• Decision boundary analysis available
• Counterfactual explanations generated

Would you like me to customize this explanation for a specific prediction or create a template?`,
        timestamp: new Date(),
        metadata: {
          query_type: 'explanation',
          confidence: 0.94,
          sources: ['shap_explanations', 'lime_analysis']
        }
      };
    } else {
      response = {
        id: `response-${Date.now()}`,
        type: 'assistant',
        content: `I understand you're asking about your ML models and data. I can help you with:

**Model Performance**
• Accuracy, precision, recall metrics
• Performance trends over time
• Model comparison and A/B testing

**Data Analysis**
• Feature importance and relationships
• Data quality and drift detection
• Statistical insights and patterns

**Explainability**
• Prediction explanations (SHAP, LIME)
• Feature impact analysis
• Business-friendly interpretations

**Business Impact**
• ROI and revenue analysis
• Cost savings calculations
• Risk assessment reports

Could you rephrase your question or try one of the suggested queries below? I'm here to help make your data more understandable!`,
        timestamp: new Date(),
        metadata: {
          query_type: 'recommendation',
          confidence: 0.75
        }
      };
    }
    
    return response;
  };

  const handleVoiceToggle = () => {
    setIsRecording(!isRecording);
    // Voice recording logic would go here
    if (!isRecording) {
      // Start recording
      console.log('Starting voice recording...');
    } else {
      // Stop recording and process
      console.log('Stopping voice recording...');
    }
  };

  const copyMessage = (content: string) => {
    navigator.clipboard.writeText(content);
  };

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'performance': return 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200';
      case 'drift': return 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200';
      case 'bias': return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200';
      case 'business': return 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200';
      default: return 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200';
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Header */}
      <div className="text-center">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center justify-center space-x-3 mb-4"
        >
          <div className="p-3 bg-gradient-to-r from-purple-500 to-blue-500 rounded-full">
            <Brain className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-3xl font-bold text-neutral-900 dark:text-neutral-100">
            Conversational Analytics
          </h1>
        </motion.div>
        <p className="text-neutral-600 dark:text-neutral-400">
          Ask questions about your models and data in natural language
        </p>
      </div>

      {/* Suggested Queries */}
      {showSuggestions && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          className="space-y-3"
        >
          <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
            Try asking me:
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {suggestedQueries.map((query) => (
              <motion.button
                key={query.id}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => handleSendMessage(query.text)}
                className="p-4 bg-white dark:bg-neutral-800 border border-neutral-200 dark:border-neutral-700 rounded-lg text-left hover:shadow-md transition-all duration-200 group"
                disabled={isLoading}
              >
                <div className="flex items-start space-x-3">
                  <div className={`p-2 rounded-full ${getCategoryColor(query.category)}`}>
                    {query.icon}
                  </div>
                  <div className="flex-1">
                    <p className="text-sm font-medium text-neutral-900 dark:text-neutral-100 group-hover:text-primary-600 dark:group-hover:text-primary-400">
                      {query.text}
                    </p>
                  </div>
                </div>
              </motion.button>
            ))}
          </div>
        </motion.div>
      )}

      {/* Chat Interface */}
      <Card className="h-96 flex flex-col">
        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          <AnimatePresence>
            {messages.map((message) => (
              <motion.div
                key={message.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div className={`max-w-[80%] ${message.type === 'user' ? 'order-2' : 'order-1'}`}>
                  <div className={`flex items-start space-x-3 ${message.type === 'user' ? 'flex-row-reverse space-x-reverse' : ''}`}>
                    {/* Avatar */}
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                      message.type === 'user' 
                        ? 'bg-primary-500' 
                        : message.loading
                          ? 'bg-gradient-to-r from-purple-500 to-blue-500 animate-pulse'
                          : 'bg-gradient-to-r from-purple-500 to-blue-500'
                    }`}>
                      {message.type === 'user' ? (
                        <User className="w-4 h-4 text-white" />
                      ) : message.loading ? (
                        <Loader2 className="w-4 h-4 text-white animate-spin" />
                      ) : (
                        <Bot className="w-4 h-4 text-white" />
                      )}
                    </div>

                    {/* Message Bubble */}
                    <div className={`rounded-2xl px-4 py-3 ${
                      message.type === 'user' 
                        ? 'bg-primary-500 text-white' 
                        : 'bg-neutral-100 dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100'
                    }`}>
                      <div className="whitespace-pre-wrap">{message.content}</div>
                      
                      {/* Message Metadata */}
                      {message.metadata && !message.loading && (
                        <div className="mt-3 pt-3 border-t border-neutral-200 dark:border-neutral-600">
                          <div className="flex items-center justify-between text-xs">
                            <div className="flex items-center space-x-2">
                              {message.metadata.confidence && (
                                <span className="text-neutral-500 dark:text-neutral-400">
                                  Confidence: {(message.metadata.confidence * 100).toFixed(0)}%
                                </span>
                              )}
                              {message.metadata.query_type && (
                                <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                                  message.metadata.query_type === 'chart' ? 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200' :
                                  message.metadata.query_type === 'insight' ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' :
                                  message.metadata.query_type === 'explanation' ? 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200' :
                                  'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200'
                                }`}>
                                  {message.metadata.query_type}
                                </span>
                              )}
                            </div>
                            
                            {/* Message Actions */}
                            <div className="flex items-center space-x-1">
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => copyMessage(message.content)}
                                className="text-neutral-400 hover:text-neutral-600 dark:hover:text-neutral-300"
                              >
                                <Copy className="w-3 h-3" />
                              </Button>
                              <Button
                                variant="ghost"
                                size="sm"
                                className="text-neutral-400 hover:text-green-600"
                              >
                                <ThumbsUp className="w-3 h-3" />
                              </Button>
                              <Button
                                variant="ghost"
                                size="sm"
                                className="text-neutral-400 hover:text-red-600"
                              >
                                <ThumbsDown className="w-3 h-3" />
                              </Button>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                  
                  <div className={`text-xs text-neutral-500 dark:text-neutral-400 mt-1 ${
                    message.type === 'user' ? 'text-right' : 'text-left'
                  }`}>
                    {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </div>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="border-t border-neutral-200 dark:border-neutral-700 p-4">
          <div className="flex items-center space-x-3">
            <div className="flex-1 relative">
              <input
                ref={inputRef}
                type="text"
                value={currentMessage}
                onChange={(e) => setCurrentMessage(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                placeholder="Ask about your models, data, or get explanations..."
                disabled={isLoading}
                className="w-full px-4 py-3 pr-12 border border-neutral-200 dark:border-neutral-700 rounded-full bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100 focus:outline-none focus:ring-2 focus:ring-primary-500 disabled:opacity-50"
              />
              <Button
                variant="ghost"
                size="sm"
                onClick={handleVoiceToggle}
                className={`absolute right-3 top-1/2 transform -translate-y-1/2 ${
                  isRecording ? 'text-red-500' : 'text-neutral-400 hover:text-primary-500'
                }`}
              >
                {isRecording ? <MicOff className="w-4 h-4" /> : <Mic className="w-4 h-4" />}
              </Button>
            </div>
            
            <Button
              variant="primary"
              onClick={() => handleSendMessage()}
              disabled={!currentMessage.trim() || isLoading}
              className="rounded-full p-3"
            >
              {isLoading ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <Send className="w-5 h-5" />
              )}
            </Button>
          </div>
          
          <div className="flex items-center justify-between mt-2 text-xs text-neutral-500 dark:text-neutral-400">
            <div className="flex items-center space-x-2">
              <Sparkles className="w-3 h-3" />
              <span>Powered by AI Analytics Engine</span>
            </div>
            <span>{currentMessage.length}/500</span>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default ConversationalAnalytics;