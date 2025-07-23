import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  TrendingUp, TrendingDown, AlertTriangle, CheckCircle, Brain, 
  Target, Activity, BarChart3, Users, Clock, Star, ArrowRight,
  Eye, Zap, Shield, Lightbulb, Maximize2, X, RefreshCw
} from 'lucide-react';
import Card from '@/components/common/Card';
import Button from '@/components/common/Button';
import { apiClient } from '@/services/api';

interface SmartInsight {
  id: string;
  type: 'alert' | 'recommendation' | 'achievement' | 'trend';
  priority: 'critical' | 'high' | 'medium' | 'low';
  title: string;
  description: string;
  action?: {
    label: string;
    route: string;
    params?: any;
  };
  impact: 'high' | 'medium' | 'low';
  category: 'performance' | 'drift' | 'bias' | 'quality' | 'business';
  timestamp: Date;
  dismissed?: boolean;
  metadata?: any;
}

interface UserPreferences {
  role: 'data_scientist' | 'business_user' | 'ml_engineer' | 'executive';
  focus_areas: string[];
  notification_level: 'all' | 'important' | 'critical';
  favorite_models: string[];
}

const SmartInsightsDashboard: React.FC = () => {
  const [insights, setInsights] = useState<SmartInsight[]>([]);
  const [userPreferences, setUserPreferences] = useState<UserPreferences>({
    role: 'data_scientist',
    focus_areas: ['performance', 'drift', 'bias'],
    notification_level: 'important',
    favorite_models: []
  });
  const [loading, setLoading] = useState(true);
  const [expandedInsight, setExpandedInsight] = useState<string | null>(null);
  const [showSettings, setShowSettings] = useState(false);

  // Generate smart insights based on user role and preferences
  const generateSmartInsights = (): SmartInsight[] => {
    const baseInsights: SmartInsight[] = [
      {
        id: '1',
        type: 'alert',
        priority: 'critical',
        title: 'Credit Scoring Model Performance Drop',
        description: 'Your primary credit scoring model has shown a 12% decrease in accuracy over the past week. This could impact loan approval decisions.',
        action: {
          label: 'Investigate Model',
          route: '/model-monitoring/credit-scoring-v2'
        },
        impact: 'high',
        category: 'performance',
        timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000) // 2 hours ago
      },
      {
        id: '2',
        type: 'recommendation',
        priority: 'high',
        title: 'Feature Engineering Opportunity Detected',
        description: 'Analysis shows that combining "Annual_Income" and "Debt_Ratio" could improve model performance by 8-15%.',
        action: {
          label: 'Explore Feature Engineering',
          route: '/feature-engineering',
          params: { suggested: ['Annual_Income', 'Debt_Ratio'] }
        },
        impact: 'medium',
        category: 'performance',
        timestamp: new Date(Date.now() - 4 * 60 * 60 * 1000) // 4 hours ago
      },
      {
        id: '3',
        type: 'alert',
        priority: 'high',
        title: 'Data Drift Detected in Customer Age Distribution',
        description: 'Significant drift detected in customer age feature. The mean age has shifted from 35.2 to 42.8 years, potentially affecting model predictions.',
        action: {
          label: 'View Drift Analysis',
          route: '/data-drift'
        },
        impact: 'medium',
        category: 'drift',
        timestamp: new Date(Date.now() - 6 * 60 * 60 * 1000) // 6 hours ago
      },
      {
        id: '4',
        type: 'achievement',
        priority: 'medium',
        title: 'Model Fairness Improved by 23%',
        description: 'Your recent bias mitigation strategies have successfully improved demographic parity across all protected groups.',
        action: {
          label: 'View Bias Report',
          route: '/bias-detection'
        },
        impact: 'high',
        category: 'bias',
        timestamp: new Date(Date.now() - 24 * 60 * 60 * 1000) // 1 day ago
      },
      {
        id: '5',
        type: 'recommendation',
        priority: 'medium',
        title: 'A/B Test Opportunity',
        description: 'Current model shows 89% accuracy. A newly trained variant achieves 92% on test data. Consider running an A/B test.',
        action: {
          label: 'Setup A/B Test',
          route: '/ab-testing'
        },
        impact: 'medium',
        category: 'business',
        timestamp: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000) // 2 days ago
      }
    ];

    // Filter insights based on user preferences
    return baseInsights.filter(insight => {
      if (userPreferences.notification_level === 'critical' && insight.priority !== 'critical') {
        return false;
      }
      if (userPreferences.notification_level === 'important' && 
          !['critical', 'high'].includes(insight.priority)) {
        return false;
      }
      return userPreferences.focus_areas.includes(insight.category);
    });
  };

  useEffect(() => {
    const loadInsights = async () => {
      setLoading(true);
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      setInsights(generateSmartInsights());
      setLoading(false);
    };
    
    loadInsights();
  }, [userPreferences]);

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'critical': return 'bg-red-500';
      case 'high': return 'bg-orange-500';
      case 'medium': return 'bg-yellow-500';
      case 'low': return 'bg-green-500';
      default: return 'bg-gray-500';
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'alert': return <AlertTriangle className="w-5 h-5" />;
      case 'recommendation': return <Lightbulb className="w-5 h-5" />;
      case 'achievement': return <Star className="w-5 h-5" />;
      case 'trend': return <TrendingUp className="w-5 h-5" />;
      default: return <Brain className="w-5 h-5" />;
    }
  };

  const dismissInsight = (insightId: string) => {
    setInsights(prev => prev.filter(insight => insight.id !== insightId));
  };

  const formatTimeAgo = (timestamp: Date) => {
    const now = new Date();
    const diffInHours = Math.floor((now.getTime() - timestamp.getTime()) / (1000 * 60 * 60));
    
    if (diffInHours < 1) return 'Just now';
    if (diffInHours < 24) return `${diffInHours}h ago`;
    return `${Math.floor(diffInHours / 24)}d ago`;
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-center">
          <Brain className="w-12 h-12 text-primary-500 animate-pulse mx-auto mb-4" />
          <p className="text-lg text-neutral-600 dark:text-neutral-400">
            Analyzing your models and generating insights...
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-neutral-900 dark:text-neutral-100 flex items-center">
            <Brain className="w-8 h-8 text-primary-500 mr-3" />
            Smart Insights
          </h2>
          <p className="text-neutral-600 dark:text-neutral-400 mt-1">
            AI-powered recommendations and alerts tailored for {userPreferences.role.replace('_', ' ')}
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <Button
            variant="outline"
            size="sm"
            leftIcon={<RefreshCw className="w-4 h-4" />}
            onClick={() => setInsights(generateSmartInsights())}
          >
            Refresh
          </Button>
          <Button
            variant="outline"
            size="sm"
            leftIcon={<Target className="w-4 h-4" />}
            onClick={() => setShowSettings(true)}
          >
            Preferences
          </Button>
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="bg-gradient-to-r from-red-50 to-orange-50 dark:from-red-900/20 dark:to-orange-900/20 p-4 rounded-lg border border-red-200 dark:border-red-800"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-red-700 dark:text-red-300">Critical Issues</p>
              <p className="text-2xl font-bold text-red-900 dark:text-red-100">
                {insights.filter(i => i.priority === 'critical').length}
              </p>
            </div>
            <AlertTriangle className="w-8 h-8 text-red-500" />
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-gradient-to-r from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 p-4 rounded-lg border border-blue-200 dark:border-blue-800"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-blue-700 dark:text-blue-300">Recommendations</p>
              <p className="text-2xl font-bold text-blue-900 dark:text-blue-100">
                {insights.filter(i => i.type === 'recommendation').length}
              </p>
            </div>
            <Lightbulb className="w-8 h-8 text-blue-500" />
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 p-4 rounded-lg border border-green-200 dark:border-green-800"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-green-700 dark:text-green-300">Achievements</p>
              <p className="text-2xl font-bold text-green-900 dark:text-green-100">
                {insights.filter(i => i.type === 'achievement').length}
              </p>
            </div>
            <Star className="w-8 h-8 text-green-500" />
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 p-4 rounded-lg border border-purple-200 dark:border-purple-800"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-purple-700 dark:text-purple-300">High Impact</p>
              <p className="text-2xl font-bold text-purple-900 dark:text-purple-100">
                {insights.filter(i => i.impact === 'high').length}
              </p>
            </div>
            <Zap className="w-8 h-8 text-purple-500" />
          </div>
        </motion.div>
      </div>

      {/* Insights List */}
      <div className="space-y-4">
        <AnimatePresence>
          {insights.map((insight, index) => (
            <motion.div
              key={insight.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              transition={{ delay: index * 0.1 }}
              className="group"
            >
              <Card className="hover:shadow-lg transition-all duration-200 border-l-4" 
                    style={{ borderLeftColor: getPriorityColor(insight.priority).replace('bg-', '') }}>
                <div className="flex items-start space-x-4">
                  <div className={`p-2 rounded-full ${getPriorityColor(insight.priority)} text-white flex-shrink-0`}>
                    {getTypeIcon(insight.type)}
                  </div>
                  
                  <div className="flex-1 min-w-0">
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 group-hover:text-primary-600 dark:group-hover:text-primary-400 transition-colors">
                          {insight.title}
                        </h3>
                        <p className="text-neutral-600 dark:text-neutral-400 mt-1 text-sm">
                          {insight.description}
                        </p>
                        
                        <div className="flex items-center space-x-4 mt-3">
                          <span className={`px-2 py-1 text-xs rounded-full font-medium ${
                            insight.impact === 'high' ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200' :
                            insight.impact === 'medium' ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200' :
                            'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                          }`}>
                            {insight.impact.toUpperCase()} IMPACT
                          </span>
                          <span className="text-xs text-neutral-500 dark:text-neutral-400">
                            {formatTimeAgo(insight.timestamp)}
                          </span>
                        </div>
                      </div>
                      
                      <div className="flex items-center space-x-2 ml-4">
                        {insight.action && (
                          <Button
                            variant="primary"
                            size="sm"
                            rightIcon={<ArrowRight className="w-4 h-4" />}
                            className="opacity-0 group-hover:opacity-100 transition-opacity"
                          >
                            {insight.action.label}
                          </Button>
                        )}
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => setExpandedInsight(expandedInsight === insight.id ? null : insight.id)}
                          className="opacity-0 group-hover:opacity-100 transition-opacity"
                        >
                          <Eye className="w-4 h-4" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => dismissInsight(insight.id)}
                          className="opacity-0 group-hover:opacity-100 transition-opacity text-neutral-400 hover:text-red-500"
                        >
                          <X className="w-4 h-4" />
                        </Button>
                      </div>
                    </div>
                    
                    {/* Expanded Details */}
                    <AnimatePresence>
                      {expandedInsight === insight.id && (
                        <motion.div
                          initial={{ opacity: 0, height: 0 }}
                          animate={{ opacity: 1, height: 'auto' }}
                          exit={{ opacity: 0, height: 0 }}
                          className="mt-4 pt-4 border-t border-neutral-200 dark:border-neutral-700"
                        >
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                            <div>
                              <h4 className="font-medium text-neutral-900 dark:text-neutral-100 mb-2">
                                Details
                              </h4>
                              <p className="text-neutral-600 dark:text-neutral-400">
                                Category: <span className="font-medium">{insight.category}</span>
                              </p>
                              <p className="text-neutral-600 dark:text-neutral-400">
                                Priority: <span className="font-medium capitalize">{insight.priority}</span>
                              </p>
                            </div>
                            <div>
                              <h4 className="font-medium text-neutral-900 dark:text-neutral-100 mb-2">
                                Next Steps
                              </h4>
                              <p className="text-neutral-600 dark:text-neutral-400">
                                {insight.action ? 
                                  `Click "${insight.action.label}" to investigate further` :
                                  'Monitor situation and check back later'
                                }
                              </p>
                            </div>
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                </div>
              </Card>
            </motion.div>
          ))}
        </AnimatePresence>
        
        {insights.length === 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-center py-12"
          >
            <CheckCircle className="w-16 h-16 text-green-500 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-neutral-900 dark:text-neutral-100 mb-2">
              All Good! 🎉
            </h3>
            <p className="text-neutral-600 dark:text-neutral-400">
              No critical insights at the moment. Your models are performing well.
            </p>
          </motion.div>
        )}
      </div>

      {/* Settings Modal */}
      <AnimatePresence>
        {showSettings && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
            onClick={() => setShowSettings(false)}
          >
            <motion.div
              initial={{ scale: 0.9 }}
              animate={{ scale: 1 }}
              exit={{ scale: 0.9 }}
              className="bg-white dark:bg-neutral-800 rounded-lg p-6 w-full max-w-md mx-4"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
                  Insight Preferences
                </h3>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowSettings(false)}
                >
                  <X className="w-4 h-4" />
                </Button>
              </div>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                    Role
                  </label>
                  <select
                    value={userPreferences.role}
                    onChange={(e) => setUserPreferences(prev => ({ 
                      ...prev, 
                      role: e.target.value as UserPreferences['role']
                    }))}
                    className="w-full p-2 border border-neutral-200 dark:border-neutral-700 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
                  >
                    <option value="data_scientist">Data Scientist</option>
                    <option value="business_user">Business User</option>
                    <option value="ml_engineer">ML Engineer</option>
                    <option value="executive">Executive</option>
                  </select>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                    Notification Level
                  </label>
                  <select
                    value={userPreferences.notification_level}
                    onChange={(e) => setUserPreferences(prev => ({ 
                      ...prev, 
                      notification_level: e.target.value as UserPreferences['notification_level']
                    }))}
                    className="w-full p-2 border border-neutral-200 dark:border-neutral-700 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
                  >
                    <option value="all">All Insights</option>
                    <option value="important">Important Only</option>
                    <option value="critical">Critical Only</option>
                  </select>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                    Focus Areas
                  </label>
                  <div className="space-y-2">
                    {['performance', 'drift', 'bias', 'quality', 'business'].map(area => (
                      <label key={area} className="flex items-center">
                        <input
                          type="checkbox"
                          checked={userPreferences.focus_areas.includes(area)}
                          onChange={(e) => {
                            const newFocusAreas = e.target.checked
                              ? [...userPreferences.focus_areas, area]
                              : userPreferences.focus_areas.filter(a => a !== area);
                            setUserPreferences(prev => ({ ...prev, focus_areas: newFocusAreas }));
                          }}
                          className="mr-2"
                        />
                        <span className="text-sm capitalize text-neutral-700 dark:text-neutral-300">
                          {area}
                        </span>
                      </label>
                    ))}
                  </div>
                </div>
              </div>
              
              <div className="flex justify-end mt-6">
                <Button
                  variant="primary"
                  onClick={() => setShowSettings(false)}
                >
                  Save Preferences
                </Button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default SmartInsightsDashboard;