import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Brain, MessageCircle, BarChart3, Activity, TrendingUp, 
  AlertTriangle, Users, Zap, Target, Globe, Maximize2, 
  Minimize2, RefreshCw, Settings, Plus, ChevronDown
} from 'lucide-react';
import Card from '@/components/common/Card';
import Button from '@/components/common/Button';
import SmartInsightsDashboard from '@/components/dashboard/SmartInsightsDashboard';
import ConversationalAnalytics from '@/components/analytics/ConversationalAnalytics';

interface DashboardWidget {
  id: string;
  title: string;
  component: React.ComponentType<any>;
  size: 'small' | 'medium' | 'large' | 'full';
  position: { x: number; y: number };
  minimized?: boolean;
  props?: any;
}

interface SystemMetrics {
  total_models: number;
  active_models: number;
  total_predictions: number;
  avg_accuracy: number;
  data_quality_score: number;
  drift_alerts: number;
  bias_alerts: number;
  performance_trend: 'up' | 'down' | 'stable';
}

const SmartDashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'overview' | 'insights' | 'chat' | 'analytics'>('overview');
  const [isCustomizing, setIsCustomizing] = useState(false);
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics>({
    total_models: 12,
    active_models: 8,
    total_predictions: 1453920,
    avg_accuracy: 0.892,
    data_quality_score: 0.94,
    drift_alerts: 3,
    bias_alerts: 1,
    performance_trend: 'up'
  });

  const containerVariants = {
    initial: { opacity: 0 },
    animate: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
      },
    },
  };

  const itemVariants = {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0 },
  };

  const MetricCard: React.FC<{
    title: string;
    value: string | number;
    icon: React.ReactNode;
    trend?: 'up' | 'down' | 'stable';
    color: string;
    description?: string;
  }> = ({ title, value, icon, trend, color, description }) => (
    <motion.div variants={itemVariants}>
      <Card className="hover:shadow-lg transition-all duration-200">
        <div className="flex items-center justify-between">
          <div className="flex-1">
            <div className="flex items-center space-x-2 mb-2">
              <div className={`p-2 rounded-lg ${color}`}>
                {icon}
              </div>
              {trend && (
                <div className={`flex items-center text-sm ${
                  trend === 'up' ? 'text-green-500' : 
                  trend === 'down' ? 'text-red-500' : 
                  'text-gray-500'
                }`}>
                  <TrendingUp className={`w-3 h-3 ${trend === 'down' ? 'rotate-180' : ''}`} />
                </div>
              )}
            </div>
            <div className="space-y-1">
              <p className="text-2xl font-bold text-neutral-900 dark:text-neutral-100">
                {typeof value === 'number' && value > 1000 ? 
                  value.toLocaleString() : value
                }
              </p>
              <p className="text-sm font-medium text-neutral-600 dark:text-neutral-400">
                {title}
              </p>
              {description && (
                <p className="text-xs text-neutral-500 dark:text-neutral-500">
                  {description}
                </p>
              )}
            </div>
          </div>
        </div>
      </Card>
    </motion.div>
  );

  const QuickActionCard: React.FC<{
    title: string;
    description: string;
    icon: React.ReactNode;
    onClick: () => void;
    color: string;
  }> = ({ title, description, icon, onClick, color }) => (
    <motion.div
      variants={itemVariants}
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
    >
      <Card 
        className="cursor-pointer hover:shadow-lg transition-all duration-200 group"
        onClick={onClick}
      >
        <div className="flex items-start space-x-4">
          <div className={`p-3 rounded-lg ${color} group-hover:scale-110 transition-transform duration-200`}>
            {icon}
          </div>
          <div className="flex-1">
            <h3 className="font-semibold text-neutral-900 dark:text-neutral-100 group-hover:text-primary-600 dark:group-hover:text-primary-400 transition-colors">
              {title}
            </h3>
            <p className="text-sm text-neutral-600 dark:text-neutral-400 mt-1">
              {description}
            </p>
          </div>
          <ChevronDown className="w-4 h-4 text-neutral-400 group-hover:text-primary-500 transform group-hover:rotate-180 transition-all duration-200" />
        </div>
      </Card>
    </motion.div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-neutral-50 to-blue-50 dark:from-neutral-900 dark:to-blue-900/20">
      {/* Header */}
      <div className="bg-white/80 dark:bg-neutral-800/80 backdrop-blur-sm border-b border-neutral-200 dark:border-neutral-700 sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-neutral-900 dark:text-neutral-100 flex items-center">
                <div className="w-8 h-8 bg-gradient-to-r from-purple-500 to-blue-500 rounded-lg flex items-center justify-center mr-3">
                  <Brain className="w-5 h-5 text-white" />
                </div>
                RAIA Smart Dashboard
              </h1>
              <p className="text-neutral-600 dark:text-neutral-400 mt-1">
                AI-Powered Responsible Analytics Platform
              </p>
            </div>
            
            <div className="flex items-center space-x-3">
              {/* Tab Navigation */}
              <div className="flex items-center bg-neutral-100 dark:bg-neutral-700 rounded-lg p-1">
                {[
                  { id: 'overview', label: 'Overview', icon: <BarChart3 className="w-4 h-4" /> },
                  { id: 'insights', label: 'Insights', icon: <Brain className="w-4 h-4" /> },
                  { id: 'chat', label: 'Chat', icon: <MessageCircle className="w-4 h-4" /> },
                  { id: 'analytics', label: 'Analytics', icon: <Activity className="w-4 h-4" /> },
                ].map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id as any)}
                    className={`flex items-center space-x-2 px-4 py-2 rounded-md text-sm font-medium transition-all duration-200 ${
                      activeTab === tab.id
                        ? 'bg-white dark:bg-neutral-800 text-primary-600 dark:text-primary-400 shadow-sm'
                        : 'text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-neutral-100'
                    }`}
                  >
                    {tab.icon}
                    <span className="hidden sm:inline">{tab.label}</span>
                  </button>
                ))}
              </div>
              
              <Button variant="outline" size="sm" leftIcon={<RefreshCw className="w-4 h-4" />}>
                Refresh
              </Button>
              <Button variant="outline" size="sm" leftIcon={<Settings className="w-4 h-4" />}>
                Settings
              </Button>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-6 py-6">
        <AnimatePresence mode="wait">
          {activeTab === 'overview' && (
            <motion.div
              key="overview"
              variants={containerVariants}
              initial="initial"
              animate="animate"
              exit={{ opacity: 0 }}
              className="space-y-6"
            >
              {/* System Metrics */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <MetricCard
                  title="Total Models"
                  value={systemMetrics.total_models}
                  icon={<Brain className="w-5 h-5 text-white" />}
                  color="bg-gradient-to-r from-purple-500 to-blue-500"
                  description={`${systemMetrics.active_models} currently active`}
                />
                <MetricCard
                  title="Predictions Made"
                  value={systemMetrics.total_predictions}
                  icon={<Target className="w-5 h-5 text-white" />}
                  trend="up"
                  color="bg-gradient-to-r from-green-500 to-emerald-500"
                  description="This month"
                />
                <MetricCard
                  title="Average Accuracy"
                  value={`${(systemMetrics.avg_accuracy * 100).toFixed(1)}%`}
                  icon={<TrendingUp className="w-5 h-5 text-white" />}
                  trend={systemMetrics.performance_trend}
                  color="bg-gradient-to-r from-blue-500 to-cyan-500"
                  description="Across all models"
                />
                <MetricCard
                  title="Data Quality Score"
                  value={`${(systemMetrics.data_quality_score * 100).toFixed(0)}%`}
                  icon={<Activity className="w-5 h-5 text-white" />}
                  trend="stable"
                  color="bg-gradient-to-r from-orange-500 to-red-500"
                  description="Quality assessment"
                />
              </div>

              {/* Alerts Summary */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <motion.div variants={itemVariants}>
                  <Card title="System Alerts" icon={<AlertTriangle className="w-5 h-5 text-orange-500" />}>
                    <div className="space-y-3">
                      <div className="flex items-center justify-between p-3 bg-red-50 dark:bg-red-900/20 rounded-lg border border-red-200 dark:border-red-800">
                        <div className="flex items-center space-x-3">
                          <AlertTriangle className="w-5 h-5 text-red-500" />
                          <div>
                            <p className="font-medium text-red-900 dark:text-red-100">Data Drift Alerts</p>
                            <p className="text-sm text-red-700 dark:text-red-300">{systemMetrics.drift_alerts} models affected</p>
                          </div>
                        </div>
                        <Button variant="outline" size="sm" onClick={() => setActiveTab('insights')}>
                          View
                        </Button>
                      </div>
                      
                      <div className="flex items-center justify-between p-3 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg border border-yellow-200 dark:border-yellow-800">
                        <div className="flex items-center space-x-3">
                          <Users className="w-5 h-5 text-yellow-500" />
                          <div>
                            <p className="font-medium text-yellow-900 dark:text-yellow-100">Bias Alerts</p>
                            <p className="text-sm text-yellow-700 dark:text-yellow-300">{systemMetrics.bias_alerts} model requires attention</p>
                          </div>
                        </div>
                        <Button variant="outline" size="sm" onClick={() => setActiveTab('insights')}>
                          Review
                        </Button>
                      </div>
                    </div>
                  </Card>
                </motion.div>

                <motion.div variants={itemVariants}>
                  <Card title="Quick Actions" icon={<Zap className="w-5 h-5 text-primary-500" />}>
                    <div className="space-y-3">
                      <QuickActionCard
                        title="Train New Model"
                        description="Start a new model training session with your latest data"
                        icon={<Brain className="w-5 h-5 text-white" />}
                        color="bg-gradient-to-r from-purple-500 to-blue-500"
                        onClick={() => console.log('Train new model')}
                      />
                      <QuickActionCard
                        title="Analyze Data Drift"
                        description="Check for distribution changes in your features"
                        icon={<Activity className="w-5 h-5 text-white" />}
                        color="bg-gradient-to-r from-orange-500 to-red-500"
                        onClick={() => setActiveTab('analytics')}
                      />
                      <QuickActionCard
                        title="Chat with AI"
                        description="Ask questions about your models and get instant insights"
                        icon={<MessageCircle className="w-5 h-5 text-white" />}
                        color="bg-gradient-to-r from-green-500 to-emerald-500"
                        onClick={() => setActiveTab('chat')}
                      />
                    </div>
                  </Card>
                </motion.div>
              </div>

              {/* Recent Activity & Performance Charts would go here */}
              <motion.div variants={itemVariants}>
                <Card title="System Performance Overview">
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <div className="space-y-4">
                      <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
                        Model Performance Trends
                      </h3>
                      <div className="h-48 bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-lg flex items-center justify-center">
                        <div className="text-center">
                          <BarChart3 className="w-12 h-12 text-primary-500 mx-auto mb-2" />
                          <p className="text-neutral-600 dark:text-neutral-400">Performance chart placeholder</p>
                          <p className="text-sm text-neutral-500">Real-time model metrics would appear here</p>
                        </div>
                      </div>
                    </div>
                    
                    <div className="space-y-4">
                      <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
                        Data Quality Metrics
                      </h3>
                      <div className="h-48 bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-lg flex items-center justify-center">
                        <div className="text-center">
                          <Activity className="w-12 h-12 text-green-500 mx-auto mb-2" />
                          <p className="text-neutral-600 dark:text-neutral-400">Data quality dashboard</p>
                          <p className="text-sm text-neutral-500">Quality metrics and drift detection</p>
                        </div>
                      </div>
                    </div>
                  </div>
                </Card>
              </motion.div>
            </motion.div>
          )}

          {activeTab === 'insights' && (
            <motion.div
              key="insights"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.3 }}
            >
              <SmartInsightsDashboard />
            </motion.div>
          )}

          {activeTab === 'chat' && (
            <motion.div
              key="chat"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.3 }}
            >
              <ConversationalAnalytics />
            </motion.div>
          )}

          {activeTab === 'analytics' && (
            <motion.div
              key="analytics"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.3 }}
              className="space-y-6"
            >
              <Card title="Advanced Analytics" icon={<Activity className="w-5 h-5 text-primary-500" />}>
                <div className="text-center py-12">
                  <Activity className="w-16 h-16 text-primary-500 mx-auto mb-4" />
                  <h3 className="text-xl font-semibold text-neutral-900 dark:text-neutral-100 mb-2">
                    Advanced Analytics Coming Soon
                  </h3>
                  <p className="text-neutral-600 dark:text-neutral-400 mb-6">
                    Interactive model comparison, A/B testing, and predictive health monitoring features are in development.
                  </p>
                  <div className="flex justify-center space-x-4">
                    <Button variant="outline" onClick={() => setActiveTab('insights')}>
                      View Smart Insights
                    </Button>
                    <Button variant="primary" onClick={() => setActiveTab('chat')}>
                      Ask AI Questions
                    </Button>
                  </div>
                </div>
              </Card>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};

export default SmartDashboard;