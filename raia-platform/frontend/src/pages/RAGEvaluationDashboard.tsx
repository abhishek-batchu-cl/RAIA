import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Search,
  BookOpen,
  Target,
  CheckCircle,
  AlertCircle,
  TrendingUp,
  Activity,
  FileText,
  BarChart3,
  Zap,
  Clock,
  Database
} from 'lucide-react';
import Card from '../components/common/Card';
import MetricCard from '../components/common/MetricCard';
import Button from '../components/common/Button';

interface RAGSystem {
  id: string;
  name: string;
  type: string;
  status: 'active' | 'inactive' | 'error';
  lastEvaluated: string;
  queryCount: number;
  avgScore: number;
}

interface RAGEvaluation {
  id: string;
  systemName: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  retrievalPrecision: number;
  retrievalRecall: number;
  retrievalF1: number;
  generationFaithfulness: number;
  generationRelevancy: number;
  generationCoherence: number;
  overallScore: number;
  queriesCount: number;
  completedQueries: number;
  createdAt: string;
}

const RAGEvaluationDashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'overview' | 'evaluations' | 'systems'>('overview');
  const [selectedSystem, setSelectedSystem] = useState<string>('all');
  const [isLoading, setIsLoading] = useState(false);

  // Mock data - would come from API
  const [ragSystems] = useState<RAGSystem[]>([
    {
      id: 'rag_001',
      name: 'Customer Support RAG',
      type: 'retrieval_augmented',
      status: 'active',
      lastEvaluated: '2024-01-30T10:30:00Z',
      queryCount: 15420,
      avgScore: 0.89
    },
    {
      id: 'rag_002', 
      name: 'Documentation Assistant',
      type: 'retrieval_augmented',
      status: 'active',
      lastEvaluated: '2024-01-30T09:15:00Z',
      queryCount: 8930,
      avgScore: 0.92
    },
    {
      id: 'rag_003',
      name: 'Legal Research RAG',
      type: 'retrieval_augmented',
      status: 'inactive',
      lastEvaluated: '2024-01-29T16:45:00Z',
      queryCount: 3240,
      avgScore: 0.85
    }
  ]);

  const [evaluations] = useState<RAGEvaluation[]>([
    {
      id: 'eval_001',
      systemName: 'Customer Support RAG',
      status: 'completed',
      retrievalPrecision: 0.87,
      retrievalRecall: 0.84,
      retrievalF1: 0.855,
      generationFaithfulness: 0.91,
      generationRelevancy: 0.89,
      generationCoherence: 0.93,
      overallScore: 0.89,
      queriesCount: 500,
      completedQueries: 500,
      createdAt: '2024-01-30T10:30:00Z'
    },
    {
      id: 'eval_002',
      systemName: 'Documentation Assistant',
      status: 'running',
      retrievalPrecision: 0.92,
      retrievalRecall: 0.88,
      retrievalF1: 0.90,
      generationFaithfulness: 0.94,
      generationRelevancy: 0.91,
      generationCoherence: 0.95,
      overallScore: 0.92,
      queriesCount: 300,
      completedQueries: 180,
      createdAt: '2024-01-30T11:00:00Z'
    }
  ]);

  const overviewMetrics = [
    {
      title: 'Active RAG Systems',
      value: ragSystems.filter(s => s.status === 'active').length.toString(),
      change: '+2',
      changeType: 'positive' as const,
      icon: Database,
      description: 'Currently deployed and serving queries'
    },
    {
      title: 'Total Evaluations',
      value: evaluations.length.toString(),
      change: '+3',
      changeType: 'positive' as const,
      icon: Activity,
      description: 'Completed evaluation runs this month'
    },
    {
      title: 'Avg Retrieval Score',
      value: '0.89',
      change: '+2.1%',
      changeType: 'positive' as const,
      icon: Search,
      description: 'Average precision across all systems'
    },
    {
      title: 'Avg Generation Score',
      value: '0.91',
      change: '+1.8%',
      changeType: 'positive' as const,
      icon: FileText,
      description: 'Average faithfulness and coherence'
    }
  ];

  const handleStartEvaluation = async (systemId: string) => {
    setIsLoading(true);
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 2000));
    setIsLoading(false);
    // Would trigger real evaluation via API
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

      {/* Recent Evaluations */}
      <Card>
        <div className="p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
              Recent Evaluations
            </h3>
            <Button 
              variant="primary" 
              size="sm"
              onClick={() => setActiveTab('evaluations')}
            >
              View All
            </Button>
          </div>
          
          <div className="space-y-4">
            {evaluations.slice(0, 3).map((eval, index) => (
              <motion.div
                key={eval.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
                className="flex items-center justify-between p-4 border border-neutral-200 dark:border-neutral-700 rounded-lg"
              >
                <div className="flex items-center space-x-4">
                  <div className={`w-3 h-3 rounded-full ${
                    eval.status === 'completed' ? 'bg-green-500' :
                    eval.status === 'running' ? 'bg-blue-500 animate-pulse' :
                    eval.status === 'failed' ? 'bg-red-500' : 'bg-yellow-500'
                  }`} />
                  <div>
                    <h4 className="font-medium text-neutral-900 dark:text-neutral-100">
                      {eval.systemName}
                    </h4>
                    <p className="text-sm text-neutral-600 dark:text-neutral-400">
                      {eval.queriesCount} queries • Overall: {(eval.overallScore * 100).toFixed(1)}%
                    </p>
                  </div>
                </div>
                
                <div className="flex items-center space-x-4">
                  <div className="text-right">
                    <div className="text-sm font-medium text-neutral-900 dark:text-neutral-100">
                      {eval.status === 'running' 
                        ? `${Math.round((eval.completedQueries / eval.queriesCount) * 100)}%` 
                        : eval.status}
                    </div>
                    <div className="text-xs text-neutral-500 dark:text-neutral-400">
                      {new Date(eval.createdAt).toLocaleDateString()}
                    </div>
                  </div>
                  
                  {eval.status === 'running' && (
                    <div className="w-16 bg-neutral-200 dark:bg-neutral-700 rounded-full h-2">
                      <div
                        className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${(eval.completedQueries / eval.queriesCount) * 100}%` }}
                      />
                    </div>
                  )}
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </Card>
    </div>
  );

  const renderEvaluations = () => (
    <div className="space-y-6">
      <Card>
        <div className="p-6">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
              Evaluation Results
            </h3>
            <Button 
              variant="primary"
              onClick={() => handleStartEvaluation('new')}
              disabled={isLoading}
            >
              {isLoading ? 'Starting...' : 'Start New Evaluation'}
            </Button>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-neutral-200 dark:border-neutral-700">
                  <th className="text-left p-3 font-medium text-neutral-900 dark:text-neutral-100">System</th>
                  <th className="text-left p-3 font-medium text-neutral-900 dark:text-neutral-100">Status</th>
                  <th className="text-left p-3 font-medium text-neutral-900 dark:text-neutral-100">Retrieval F1</th>
                  <th className="text-left p-3 font-medium text-neutral-900 dark:text-neutral-100">Faithfulness</th>
                  <th className="text-left p-3 font-medium text-neutral-900 dark:text-neutral-100">Relevancy</th>
                  <th className="text-left p-3 font-medium text-neutral-900 dark:text-neutral-100">Overall</th>
                  <th className="text-left p-3 font-medium text-neutral-900 dark:text-neutral-100">Date</th>
                </tr>
              </thead>
              <tbody>
                {evaluations.map((eval) => (
                  <tr key={eval.id} className="border-b border-neutral-100 dark:border-neutral-800 hover:bg-neutral-50 dark:hover:bg-neutral-800/50">
                    <td className="p-3">
                      <div className="font-medium text-neutral-900 dark:text-neutral-100">
                        {eval.systemName}
                      </div>
                      <div className="text-xs text-neutral-500 dark:text-neutral-400">
                        {eval.queriesCount} queries
                      </div>
                    </td>
                    <td className="p-3">
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                        eval.status === 'completed' ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300' :
                        eval.status === 'running' ? 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300' :
                        eval.status === 'failed' ? 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300' :
                        'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300'
                      }`}>
                        {eval.status}
                      </span>
                    </td>
                    <td className="p-3 font-medium text-neutral-900 dark:text-neutral-100">
                      {(eval.retrievalF1 * 100).toFixed(1)}%
                    </td>
                    <td className="p-3 font-medium text-neutral-900 dark:text-neutral-100">
                      {(eval.generationFaithfulness * 100).toFixed(1)}%
                    </td>
                    <td className="p-3 font-medium text-neutral-900 dark:text-neutral-100">
                      {(eval.generationRelevancy * 100).toFixed(1)}%
                    </td>
                    <td className="p-3">
                      <div className="flex items-center space-x-2">
                        <div className={`w-12 bg-neutral-200 dark:bg-neutral-700 rounded-full h-2`}>
                          <div
                            className={`h-2 rounded-full ${
                              eval.overallScore >= 0.9 ? 'bg-green-500' :
                              eval.overallScore >= 0.8 ? 'bg-yellow-500' : 'bg-red-500'
                            }`}
                            style={{ width: `${eval.overallScore * 100}%` }}
                          />
                        </div>
                        <span className="text-sm font-medium text-neutral-900 dark:text-neutral-100">
                          {(eval.overallScore * 100).toFixed(1)}%
                        </span>
                      </div>
                    </td>
                    <td className="p-3 text-neutral-600 dark:text-neutral-400">
                      {new Date(eval.createdAt).toLocaleDateString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </Card>
    </div>
  );

  const renderSystems = () => (
    <div className="space-y-6">
      <Card>
        <div className="p-6">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
              RAG Systems
            </h3>
            <Button variant="primary">
              Register New System
            </Button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {ragSystems.map((system, index) => (
              <motion.div
                key={system.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
              >
                <Card className="h-full">
                  <div className="p-4">
                    <div className="flex items-center justify-between mb-4">
                      <h4 className="font-semibold text-neutral-900 dark:text-neutral-100">
                        {system.name}
                      </h4>
                      <span className={`w-3 h-3 rounded-full ${
                        system.status === 'active' ? 'bg-green-500' :
                        system.status === 'inactive' ? 'bg-yellow-500' : 'bg-red-500'
                      }`} />
                    </div>
                    
                    <div className="space-y-3">
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-neutral-600 dark:text-neutral-400">Type</span>
                        <span className="font-medium text-neutral-900 dark:text-neutral-100">
                          {system.type.replace('_', ' ')}
                        </span>
                      </div>
                      
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-neutral-600 dark:text-neutral-400">Queries</span>
                        <span className="font-medium text-neutral-900 dark:text-neutral-100">
                          {system.queryCount.toLocaleString()}
                        </span>
                      </div>
                      
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-neutral-600 dark:text-neutral-400">Avg Score</span>
                        <span className="font-medium text-neutral-900 dark:text-neutral-100">
                          {(system.avgScore * 100).toFixed(1)}%
                        </span>
                      </div>
                      
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-neutral-600 dark:text-neutral-400">Last Evaluated</span>
                        <span className="text-xs text-neutral-500 dark:text-neutral-400">
                          {new Date(system.lastEvaluated).toLocaleDateString()}
                        </span>
                      </div>
                    </div>
                    
                    <div className="mt-4 pt-4 border-t border-neutral-200 dark:border-neutral-700">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleStartEvaluation(system.id)}
                        disabled={isLoading || system.status !== 'active'}
                        className="w-full"
                      >
                        {system.status === 'active' ? 'Evaluate' : 'Inactive'}
                      </Button>
                    </div>
                  </div>
                </Card>
              </motion.div>
            ))}
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
            RAG Evaluation Dashboard
          </h1>
          <p className="text-neutral-600 dark:text-neutral-400 mt-1">
            Evaluate and monitor Retrieval-Augmented Generation systems
          </p>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="flex space-x-1 bg-neutral-100 dark:bg-neutral-800 p-1 rounded-lg w-fit">
        {[
          { key: 'overview', label: 'Overview', icon: BarChart3 },
          { key: 'evaluations', label: 'Evaluations', icon: CheckCircle },
          { key: 'systems', label: 'Systems', icon: Database }
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
      {activeTab === 'systems' && renderSystems()}
    </div>
  );
};

export default RAGEvaluationDashboard;