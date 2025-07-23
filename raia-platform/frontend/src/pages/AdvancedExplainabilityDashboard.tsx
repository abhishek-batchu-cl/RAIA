import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Eye,
  Layers,
  Search,
  Zap,
  Target,
  TrendingUp,
  BarChart3,
  Settings,
  Download,
  RefreshCw,
  Lightbulb,
  GitBranch
} from 'lucide-react';
import Card from '../components/common/Card';
import MetricCard from '../components/common/MetricCard';
import Button from '../components/common/Button';

interface ExplainabilitySession {
  id: string;
  sessionName: string;
  modelName: string;
  method: 'anchor' | 'ale' | 'prototype' | 'ice' | 'counterfactual' | 'permutation';
  status: 'pending' | 'running' | 'completed' | 'failed';
  processingTime: number;
  createdAt: string;
  results?: any;
}

interface ExplanationMethod {
  key: string;
  name: string;
  description: string;
  type: 'local' | 'global' | 'prototype';
  icon: React.ComponentType<any>;
  complexity: 'low' | 'medium' | 'high';
  supported: boolean;
}

const AdvancedExplainabilityDashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'overview' | 'methods' | 'sessions' | 'insights'>('overview');
  const [selectedMethod, setSelectedMethod] = useState<string>('anchor');
  const [isRunning, setIsRunning] = useState(false);

  const explanationMethods: ExplanationMethod[] = [
    {
      key: 'anchor',
      name: 'Anchor Explanations',
      description: 'Find minimal sufficient conditions for predictions using rule-based anchors',
      type: 'local',
      icon: Target,
      complexity: 'medium',
      supported: true
    },
    {
      key: 'ale',
      name: 'ALE Plots',
      description: 'Accumulated Local Effects plots for feature importance analysis',
      type: 'global',
      icon: TrendingUp,
      complexity: 'medium',
      supported: true
    },
    {
      key: 'prototype',
      name: 'Prototype Analysis',
      description: 'Find representative prototypes and criticisms in the data space',
      type: 'prototype',
      icon: Search,
      complexity: 'high',
      supported: true
    },
    {
      key: 'ice',
      name: 'ICE Plots',
      description: 'Individual Conditional Expectation plots for instance-level insights',
      type: 'local',
      icon: Layers,
      complexity: 'medium',
      supported: true
    },
    {
      key: 'counterfactual',
      name: 'Counterfactuals',
      description: 'Generate what-if scenarios with minimal feature changes',
      type: 'local',
      icon: GitBranch,
      complexity: 'high',
      supported: true
    },
    {
      key: 'permutation',
      name: 'Permutation Importance',
      description: 'Measure feature importance through permutation-based analysis',
      type: 'global',
      icon: RefreshCw,
      complexity: 'low',
      supported: true
    }
  ];

  const [sessions] = useState<ExplainabilitySession[]>([
    {
      id: 'session_001',
      sessionName: 'Customer Churn Analysis',
      modelName: 'RandomForest_v2.1',
      method: 'anchor',
      status: 'completed',
      processingTime: 1240,
      createdAt: '2024-01-30T10:30:00Z',
      results: {
        anchorPrecision: 0.95,
        anchorCoverage: 0.23,
        rulesGenerated: 8
      }
    },
    {
      id: 'session_002',
      sessionName: 'Feature Importance Study',
      modelName: 'XGBoost_v1.3',
      method: 'ale',
      status: 'running',
      processingTime: 0,
      createdAt: '2024-01-30T11:15:00Z'
    },
    {
      id: 'session_003',
      sessionName: 'Prototype Discovery',
      modelName: 'NeuralNet_v3.0',
      method: 'prototype',
      status: 'completed',
      processingTime: 3420,
      createdAt: '2024-01-30T09:00:00Z',
      results: {
        prototypesFound: 15,
        criticismsFound: 8,
        avgSimilarity: 0.87
      }
    }
  ]);

  const overviewMetrics = [
    {
      title: 'Active Methods',
      value: explanationMethods.filter(m => m.supported).length.toString(),
      change: '+2',
      changeType: 'positive' as const,
      icon: Eye,
      description: 'Advanced explainability methods available'
    },
    {
      title: 'Sessions Completed',
      value: sessions.filter(s => s.status === 'completed').length.toString(),
      change: '+5',
      changeType: 'positive' as const,
      icon: BarChart3,
      description: 'Explanation sessions this week'
    },
    {
      title: 'Avg Processing Time',
      value: '24s',
      change: '-12%',
      changeType: 'positive' as const,
      icon: Zap,
      description: 'Time to generate explanations'
    },
    {
      title: 'Explanation Quality',
      value: '91%',
      change: '+3.2%',
      changeType: 'positive' as const,
      icon: Target,
      description: 'Average explanation accuracy score'
    }
  ];

  const handleRunExplanation = async (method: string, modelId: string) => {
    setIsRunning(true);
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 3000));
    setIsRunning(false);
  };

  const renderOverview = () => (
    <div className="space-y-6">
      {/* Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {overviewMetrics.map((metric, index) => (
          <motion.div
            key={metric.title}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
          >
            <MetricCard
              title={metric.title}
              value={metric.value}
              change={metric.change}
              changeType={metric.changeType}
              icon={React.createElement(metric.icon, { className: "w-5 h-5" })}
              description={metric.description}
            />
          </motion.div>
        ))}
      </div>

      {/* Methods Overview */}
      <Card>
        <div className="p-6">
          <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-6">
            Available Explanation Methods
          </h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {explanationMethods.map((method, index) => (
              <motion.div
                key={method.key}
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: index * 0.1 }}
                className={`p-4 rounded-lg border-2 transition-all cursor-pointer ${
                  selectedMethod === method.key
                    ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                    : 'border-neutral-200 dark:border-neutral-700 hover:border-neutral-300 dark:hover:border-neutral-600'
                }`}
                onClick={() => setSelectedMethod(method.key)}
              >
                <div className="flex items-center justify-between mb-2">
                  <method.icon className={`w-5 h-5 ${
                    selectedMethod === method.key ? 'text-blue-600 dark:text-blue-400' : 'text-neutral-500'
                  }`} />
                  <div className="flex items-center space-x-2">
                    <span className={`px-2 py-1 rounded text-xs font-medium ${
                      method.type === 'local' ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300' :
                      method.type === 'global' ? 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300' :
                      'bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300'
                    }`}>
                      {method.type}
                    </span>
                    <span className={`px-2 py-1 rounded text-xs font-medium ${
                      method.complexity === 'low' ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300' :
                      method.complexity === 'medium' ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300' :
                      'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300'
                    }`}>
                      {method.complexity}
                    </span>
                  </div>
                </div>
                <h4 className="font-medium text-neutral-900 dark:text-neutral-100 mb-2">
                  {method.name}
                </h4>
                <p className="text-sm text-neutral-600 dark:text-neutral-400">
                  {method.description}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </Card>

      {/* Recent Sessions */}
      <Card>
        <div className="p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
              Recent Explanation Sessions
            </h3>
            <Button variant="primary" size="sm" onClick={() => setActiveTab('sessions')}>
              View All
            </Button>
          </div>
          
          <div className="space-y-4">
            {sessions.slice(0, 3).map((session, index) => (
              <motion.div
                key={session.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
                className="flex items-center justify-between p-4 border border-neutral-200 dark:border-neutral-700 rounded-lg"
              >
                <div className="flex items-center space-x-4">
                  <div className={`w-3 h-3 rounded-full ${
                    session.status === 'completed' ? 'bg-green-500' :
                    session.status === 'running' ? 'bg-blue-500 animate-pulse' :
                    session.status === 'failed' ? 'bg-red-500' : 'bg-yellow-500'
                  }`} />
                  <div>
                    <h4 className="font-medium text-neutral-900 dark:text-neutral-100">
                      {session.sessionName}
                    </h4>
                    <p className="text-sm text-neutral-600 dark:text-neutral-400">
                      {session.modelName} • {explanationMethods.find(m => m.key === session.method)?.name}
                    </p>
                  </div>
                </div>
                
                <div className="flex items-center space-x-4">
                  <div className="text-right">
                    <div className="text-sm font-medium text-neutral-900 dark:text-neutral-100">
                      {session.status === 'completed' ? `${session.processingTime}ms` : session.status}
                    </div>
                    <div className="text-xs text-neutral-500 dark:text-neutral-400">
                      {new Date(session.createdAt).toLocaleDateString()}
                    </div>
                  </div>
                  
                  {session.status === 'completed' && (
                    <Button variant="outline" size="sm">
                      View Results
                    </Button>
                  )}
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </Card>
    </div>
  );

  const renderMethods = () => (
    <div className="space-y-6">
      <Card>
        <div className="p-6">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
              Explanation Method Configuration
            </h3>
            <Button 
              variant="primary"
              onClick={() => handleRunExplanation(selectedMethod, 'model_001')}
              disabled={isRunning}
            >
              {isRunning ? 'Generating...' : 'Run Explanation'}
            </Button>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Method Selection */}
            <div className="space-y-4">
              <h4 className="font-medium text-neutral-900 dark:text-neutral-100">
                Select Method
              </h4>
              {explanationMethods.map((method) => (
                <div
                  key={method.key}
                  className={`p-3 rounded-lg border cursor-pointer transition-all ${
                    selectedMethod === method.key
                      ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                      : 'border-neutral-200 dark:border-neutral-700 hover:border-neutral-300 dark:hover:border-neutral-600'
                  }`}
                  onClick={() => setSelectedMethod(method.key)}
                >
                  <div className="flex items-center space-x-3">
                    <method.icon className={`w-4 h-4 ${
                      selectedMethod === method.key ? 'text-blue-600 dark:text-blue-400' : 'text-neutral-500'
                    }`} />
                    <div className="flex-1">
                      <div className="font-medium text-neutral-900 dark:text-neutral-100 text-sm">
                        {method.name}
                      </div>
                      <div className="text-xs text-neutral-600 dark:text-neutral-400">
                        {method.type} • {method.complexity} complexity
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* Configuration Panel */}
            <div className="lg:col-span-2 space-y-4">
              <h4 className="font-medium text-neutral-900 dark:text-neutral-100">
                Configuration
              </h4>
              
              <div className="bg-neutral-50 dark:bg-neutral-800 rounded-lg p-4">
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                      Model Selection
                    </label>
                    <select className="w-full px-3 py-2 border border-neutral-300 dark:border-neutral-600 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100">
                      <option>RandomForest_v2.1</option>
                      <option>XGBoost_v1.3</option>
                      <option>NeuralNet_v3.0</option>
                    </select>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                      Sample Size
                    </label>
                    <input
                      type="number"
                      className="w-full px-3 py-2 border border-neutral-300 dark:border-neutral-600 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
                      defaultValue={1000}
                    />
                  </div>

                  {selectedMethod === 'anchor' && (
                    <>
                      <div>
                        <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                          Precision Threshold
                        </label>
                        <input
                          type="range"
                          min="0.8"
                          max="1.0"
                          step="0.01"
                          defaultValue={0.95}
                          className="w-full"
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                          Max Rules
                        </label>
                        <input
                          type="number"
                          className="w-full px-3 py-2 border border-neutral-300 dark:border-neutral-600 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
                          defaultValue={10}
                        />
                      </div>
                    </>
                  )}

                  {selectedMethod === 'ale' && (
                    <div>
                      <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                        Grid Resolution
                      </label>
                      <select className="w-full px-3 py-2 border border-neutral-300 dark:border-neutral-600 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100">
                        <option>50</option>
                        <option>100</option>
                        <option>200</option>
                      </select>
                    </div>
                  )}

                  {selectedMethod === 'prototype' && (
                    <>
                      <div>
                        <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                          Number of Prototypes
                        </label>
                        <input
                          type="number"
                          className="w-full px-3 py-2 border border-neutral-300 dark:border-neutral-600 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
                          defaultValue={10}
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                          Number of Criticisms
                        </label>
                        <input
                          type="number"
                          className="w-full px-3 py-2 border border-neutral-300 dark:border-neutral-600 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
                          defaultValue={5}
                        />
                      </div>
                    </>
                  )}
                </div>
              </div>

              {/* Method Description */}
              <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-700 rounded-lg p-4">
                <div className="flex items-start space-x-3">
                  <Lightbulb className="w-5 h-5 text-blue-600 dark:text-blue-400 mt-0.5" />
                  <div>
                    <h5 className="font-medium text-blue-900 dark:text-blue-100 mb-2">
                      About {explanationMethods.find(m => m.key === selectedMethod)?.name}
                    </h5>
                    <p className="text-sm text-blue-800 dark:text-blue-200">
                      {explanationMethods.find(m => m.key === selectedMethod)?.description}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </Card>
    </div>
  );

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-neutral-900 dark:text-neutral-100">
            Advanced Explainability
          </h1>
          <p className="text-neutral-600 dark:text-neutral-400 mt-1">
            Deep insights into model behavior with advanced explanation methods
          </p>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="flex space-x-1 bg-neutral-100 dark:bg-neutral-800 p-1 rounded-lg w-fit">
        {[
          { key: 'overview', label: 'Overview', icon: BarChart3 },
          { key: 'methods', label: 'Methods', icon: Settings },
          { key: 'sessions', label: 'Sessions', icon: Eye },
          { key: 'insights', label: 'Insights', icon: Lightbulb }
        ].map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key as typeof activeTab)}
            className={`flex items-center space-x-2 px-4 py-2 rounded-md text-sm font-medium transition-all ${
              activeTab === tab.key
                ? 'bg-white dark:bg-neutral-700 text-neutral-900 dark:text-neutral-100 shadow-sm'
                : 'text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-neutral-100'
            }`}
          >
            <tab.icon className="w-4 h-4" />
            <span>{tab.label}</span>
          </button>
        ))}
      </div>

      {/* Tab Content */}
      {activeTab === 'overview' && renderOverview()}
      {activeTab === 'methods' && renderMethods()}
      {activeTab === 'sessions' && (
        <Card>
          <div className="p-6">
            <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4">
              All Explanation Sessions
            </h3>
            <p className="text-neutral-600 dark:text-neutral-400">
              Detailed view of all explanation sessions coming soon.
            </p>
          </div>
        </Card>
      )}
      {activeTab === 'insights' && (
        <Card>
          <div className="p-6">
            <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4">
              Generated Insights (Coming Soon)
            </h3>
            <p className="text-neutral-600 dark:text-neutral-400">
              AI-powered insights and recommendations based on explanations.
            </p>
          </div>
        </Card>
      )}
    </div>
  );
};

export default AdvancedExplainabilityDashboard;