import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Brain,
  MessageSquare,
  Shield,
  Zap,
  Target,
  AlertTriangle,
  CheckCircle,
  TrendingUp,
  BarChart3,
  Settings,
  Play,
  Pause
} from 'lucide-react';
import Card from '../components/common/Card';
import MetricCard from '../components/common/MetricCard';
import Button from '../components/common/Button';

interface LLMModel {
  id: string;
  name: string;
  provider: string;
  type: string;
  status: 'active' | 'inactive';
  lastEvaluated: string;
  overallScore: number;
}

interface LLMEvaluation {
  id: string;
  modelName: string;
  framework: string;
  taskType: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  factualAccuracy: number;
  completeness: number;
  relevancy: number;
  logicalConsistency: number;
  fluency: number;
  grammar: number;
  clarity: number;
  conciseness: number;
  bertScore: number;
  semanticCoherence: number;
  toxicityScore: number;
  biasScore: number;
  harmfulContentScore: number;
  overallScore: number;
  inputsCount: number;
  completedEvaluations: number;
  createdAt: string;
}

const LLMEvaluationDashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'overview' | 'evaluations' | 'models' | 'safety'>('overview');
  const [selectedModel, setSelectedModel] = useState<string>('all');
  const [isRunningEvaluation, setIsRunningEvaluation] = useState(false);

  // Mock data - would come from API
  const [llmModels] = useState<LLMModel[]>([
    {
      id: 'llm_001',
      name: 'GPT-4 Turbo',
      provider: 'OpenAI',
      type: 'text_generation',
      status: 'active',
      lastEvaluated: '2024-01-30T10:30:00Z',
      overallScore: 0.91
    },
    {
      id: 'llm_002',
      name: 'Claude-3 Sonnet',
      provider: 'Anthropic',
      type: 'text_generation',
      status: 'active',
      lastEvaluated: '2024-01-30T09:15:00Z',
      overallScore: 0.89
    },
    {
      id: 'llm_003',
      name: 'Llama 2 70B',
      provider: 'Meta',
      type: 'text_generation',
      status: 'inactive',
      lastEvaluated: '2024-01-29T16:45:00Z',
      overallScore: 0.85
    }
  ]);

  const [evaluations] = useState<LLMEvaluation[]>([
    {
      id: 'eval_001',
      modelName: 'GPT-4 Turbo',
      framework: 'comprehensive',
      taskType: 'question_answering',
      status: 'completed',
      factualAccuracy: 0.93,
      completeness: 0.89,
      relevancy: 0.91,
      logicalConsistency: 0.94,
      fluency: 0.96,
      grammar: 0.97,
      clarity: 0.92,
      conciseness: 0.88,
      bertScore: 0.90,
      semanticCoherence: 0.93,
      toxicityScore: 0.02,
      biasScore: 0.08,
      harmfulContentScore: 0.01,
      overallScore: 0.91,
      inputsCount: 1000,
      completedEvaluations: 1000,
      createdAt: '2024-01-30T10:30:00Z'
    },
    {
      id: 'eval_002',
      modelName: 'Claude-3 Sonnet',
      framework: 'safety_focused',
      taskType: 'text_generation',
      status: 'running',
      factualAccuracy: 0.91,
      completeness: 0.87,
      relevancy: 0.89,
      logicalConsistency: 0.92,
      fluency: 0.94,
      grammar: 0.95,
      clarity: 0.90,
      conciseness: 0.86,
      bertScore: 0.88,
      semanticCoherence: 0.91,
      toxicityScore: 0.01,
      biasScore: 0.06,
      harmfulContentScore: 0.01,
      overallScore: 0.89,
      inputsCount: 800,
      completedEvaluations: 480,
      createdAt: '2024-01-30T11:00:00Z'
    }
  ]);

  const overviewMetrics = [
    {
      title: 'Active Models',
      value: llmModels.filter(m => m.status === 'active').length.toString(),
      change: '+1',
      changeType: 'positive' as const,
      icon: Brain,
      description: 'LLM models currently being evaluated'
    },
    {
      title: 'Content Quality',
      value: '91.2%',
      change: '+2.3%',
      changeType: 'positive' as const,
      icon: Target,
      description: 'Average factual accuracy and relevancy'
    },
    {
      title: 'Language Quality',
      value: '93.5%',
      change: '+1.1%',
      changeType: 'positive' as const,
      icon: MessageSquare,
      description: 'Fluency, grammar, and clarity scores'
    },
    {
      title: 'Safety Score',
      value: '98.2%',
      change: '+0.8%',
      changeType: 'positive' as const,
      icon: Shield,
      description: 'Low toxicity and bias detection'
    }
  ];

  const handleStartEvaluation = async (modelId: string, framework: string) => {
    setIsRunningEvaluation(true);
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 2000));
    setIsRunningEvaluation(false);
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

      {/* Model Performance Comparison */}
      <Card>
        <div className="p-6">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
              Model Performance Comparison
            </h3>
            <Button variant="outline" size="sm">
              Export Report
            </Button>
          </div>
          
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-neutral-200 dark:border-neutral-700">
                  <th className="text-left p-3 font-medium text-neutral-900 dark:text-neutral-100">Model</th>
                  <th className="text-left p-3 font-medium text-neutral-900 dark:text-neutral-100">Provider</th>
                  <th className="text-left p-3 font-medium text-neutral-900 dark:text-neutral-100">Content Quality</th>
                  <th className="text-left p-3 font-medium text-neutral-900 dark:text-neutral-100">Language Quality</th>
                  <th className="text-left p-3 font-medium text-neutral-900 dark:text-neutral-100">Safety Score</th>
                  <th className="text-left p-3 font-medium text-neutral-900 dark:text-neutral-100">Overall</th>
                  <th className="text-left p-3 font-medium text-neutral-900 dark:text-neutral-100">Status</th>
                </tr>
              </thead>
              <tbody>
                {llmModels.map((model) => (
                  <tr key={model.id} className="border-b border-neutral-100 dark:border-neutral-800 hover:bg-neutral-50 dark:hover:bg-neutral-800/50">
                    <td className="p-3">
                      <div className="font-medium text-neutral-900 dark:text-neutral-100">
                        {model.name}
                      </div>
                    </td>
                    <td className="p-3 text-neutral-600 dark:text-neutral-400">
                      {model.provider}
                    </td>
                    <td className="p-3">
                      <div className="flex items-center space-x-2">
                        <div className="w-12 bg-neutral-200 dark:bg-neutral-700 rounded-full h-2">
                          <div className="bg-blue-500 h-2 rounded-full" style={{ width: '91%' }} />
                        </div>
                        <span className="text-sm font-medium text-neutral-900 dark:text-neutral-100">91%</span>
                      </div>
                    </td>
                    <td className="p-3">
                      <div className="flex items-center space-x-2">
                        <div className="w-12 bg-neutral-200 dark:bg-neutral-700 rounded-full h-2">
                          <div className="bg-green-500 h-2 rounded-full" style={{ width: '94%' }} />
                        </div>
                        <span className="text-sm font-medium text-neutral-900 dark:text-neutral-100">94%</span>
                      </div>
                    </td>
                    <td className="p-3">
                      <div className="flex items-center space-x-2">
                        <div className="w-12 bg-neutral-200 dark:bg-neutral-700 rounded-full h-2">
                          <div className="bg-purple-500 h-2 rounded-full" style={{ width: '98%' }} />
                        </div>
                        <span className="text-sm font-medium text-neutral-900 dark:text-neutral-100">98%</span>
                      </div>
                    </td>
                    <td className="p-3">
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                        model.overallScore >= 0.9 ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300' :
                        model.overallScore >= 0.8 ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300' :
                        'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300'
                      }`}>
                        {(model.overallScore * 100).toFixed(1)}%
                      </span>
                    </td>
                    <td className="p-3">
                      <div className="flex items-center space-x-2">
                        <div className={`w-2 h-2 rounded-full ${
                          model.status === 'active' ? 'bg-green-500' : 'bg-gray-400'
                        }`} />
                        <span className="text-sm text-neutral-600 dark:text-neutral-400 capitalize">
                          {model.status}
                        </span>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </Card>

      {/* Recent Evaluations Summary */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <div className="p-6">
            <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4">
              Content Quality Metrics
            </h3>
            <div className="space-y-4">
              {[
                { metric: 'Factual Accuracy', value: 0.93, color: 'blue' },
                { metric: 'Relevancy', value: 0.91, color: 'green' },
                { metric: 'Completeness', value: 0.89, color: 'purple' },
                { metric: 'Logical Consistency', value: 0.94, color: 'indigo' }
              ].map((item, index) => (
                <div key={item.metric} className="flex items-center justify-between">
                  <span className="text-sm text-neutral-600 dark:text-neutral-400">
                    {item.metric}
                  </span>
                  <div className="flex items-center space-x-2">
                    <div className="w-20 bg-neutral-200 dark:bg-neutral-700 rounded-full h-2">
                      <div
                        className={`h-2 rounded-full bg-${item.color}-500`}
                        style={{ width: `${item.value * 100}%` }}
                      />
                    </div>
                    <span className="text-sm font-medium text-neutral-900 dark:text-neutral-100 min-w-[3rem]">
                      {(item.value * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </Card>

        <Card>
          <div className="p-6">
            <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4">
              Safety & Ethics Metrics
            </h3>
            <div className="space-y-4">
              {[
                { metric: 'Toxicity Score', value: 0.02, inverted: true, color: 'red' },
                { metric: 'Bias Score', value: 0.08, inverted: true, color: 'yellow' },
                { metric: 'Harmful Content', value: 0.01, inverted: true, color: 'red' },
                { metric: 'Safety Compliance', value: 0.98, inverted: false, color: 'green' }
              ].map((item, index) => (
                <div key={item.metric} className="flex items-center justify-between">
                  <span className="text-sm text-neutral-600 dark:text-neutral-400">
                    {item.metric}
                  </span>
                  <div className="flex items-center space-x-2">
                    <div className="w-20 bg-neutral-200 dark:bg-neutral-700 rounded-full h-2">
                      <div
                        className={`h-2 rounded-full ${
                          item.inverted
                            ? item.value <= 0.05 ? 'bg-green-500' : item.value <= 0.15 ? 'bg-yellow-500' : 'bg-red-500'
                            : 'bg-green-500'
                        }`}
                        style={{ width: `${item.inverted ? (1 - item.value) * 100 : item.value * 100}%` }}
                      />
                    </div>
                    <span className="text-sm font-medium text-neutral-900 dark:text-neutral-100 min-w-[3rem]">
                      {item.inverted ? (item.value * 100).toFixed(1) : (item.value * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </Card>
      </div>
    </div>
  );

  const renderEvaluations = () => (
    <Card>
      <div className="p-6">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
            LLM Evaluation Results
          </h3>
          <div className="flex items-center space-x-4">
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="px-3 py-2 border border-neutral-300 dark:border-neutral-600 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
            >
              <option value="all">All Models</option>
              {llmModels.map(model => (
                <option key={model.id} value={model.id}>{model.name}</option>
              ))}
            </select>
            <Button 
              variant="primary"
              onClick={() => handleStartEvaluation('new', 'comprehensive')}
              disabled={isRunningEvaluation}
            >
              {isRunningEvaluation ? 'Starting...' : 'New Evaluation'}
            </Button>
          </div>
        </div>

        <div className="space-y-4">
          {evaluations.map((eval, index) => (
            <motion.div
              key={eval.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className="border border-neutral-200 dark:border-neutral-700 rounded-lg p-4"
            >
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h4 className="font-medium text-neutral-900 dark:text-neutral-100">
                    {eval.modelName}
                  </h4>
                  <p className="text-sm text-neutral-600 dark:text-neutral-400">
                    {eval.framework} • {eval.taskType.replace('_', ' ')} • {eval.inputsCount} inputs
                  </p>
                </div>
                <div className="flex items-center space-x-4">
                  <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                    eval.status === 'completed' ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300' :
                    eval.status === 'running' ? 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300' :
                    eval.status === 'failed' ? 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300' :
                    'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300'
                  }`}>
                    {eval.status}
                  </span>
                  {eval.status === 'running' && (
                    <div className="text-sm text-neutral-600 dark:text-neutral-400">
                      {Math.round((eval.completedEvaluations / eval.inputsCount) * 100)}% complete
                    </div>
                  )}
                </div>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                    {(eval.factualAccuracy * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-neutral-600 dark:text-neutral-400">
                    Factual Accuracy
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                    {(eval.fluency * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-neutral-600 dark:text-neutral-400">
                    Fluency
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">
                    {(eval.toxicityScore * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-neutral-600 dark:text-neutral-400">
                    Toxicity (lower is better)
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-neutral-900 dark:text-neutral-100">
                    {(eval.overallScore * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-neutral-600 dark:text-neutral-400">
                    Overall Score
                  </div>
                </div>
              </div>

              {eval.status === 'running' && (
                <div className="mt-4">
                  <div className="w-full bg-neutral-200 dark:bg-neutral-700 rounded-full h-2">
                    <div
                      className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${(eval.completedEvaluations / eval.inputsCount) * 100}%` }}
                    />
                  </div>
                </div>
              )}
            </motion.div>
          ))}
        </div>
      </div>
    </Card>
  );

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-neutral-900 dark:text-neutral-100">
            LLM Evaluation Dashboard
          </h1>
          <p className="text-neutral-600 dark:text-neutral-400 mt-1">
            Comprehensive Large Language Model evaluation and safety analysis
          </p>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="flex space-x-1 bg-neutral-100 dark:bg-neutral-800 p-1 rounded-lg w-fit">
        {[
          { key: 'overview', label: 'Overview', icon: BarChart3 },
          { key: 'evaluations', label: 'Evaluations', icon: CheckCircle },
          { key: 'models', label: 'Models', icon: Brain },
          { key: 'safety', label: 'Safety', icon: Shield }
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
      {activeTab === 'evaluations' && renderEvaluations()}
      {activeTab === 'models' && (
        <Card>
          <div className="p-6">
            <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4">
              Model Management (Coming Soon)
            </h3>
            <p className="text-neutral-600 dark:text-neutral-400">
              Register and manage your LLM models for comprehensive evaluation.
            </p>
          </div>
        </Card>
      )}
      {activeTab === 'safety' && (
        <Card>
          <div className="p-6">
            <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4">
              Safety Analysis (Coming Soon)
            </h3>
            <p className="text-neutral-600 dark:text-neutral-400">
              Deep dive into safety metrics, bias detection, and content moderation.
            </p>
          </div>
        </Card>
      )}
    </div>
  );
};

export default LLMEvaluationDashboard;