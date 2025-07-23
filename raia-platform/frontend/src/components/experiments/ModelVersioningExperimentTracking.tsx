import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  GitBranch, History, Clock, BarChart3, Brain, 
  CheckCircle, X, AlertTriangle, TrendingUp, TrendingDown,
  Tag, Calendar, User, Filter, Search, Download,
  Upload, Eye, Play, Pause, ArrowRight, ArrowLeft,
  Copy, Share2, Lock, Unlock, Star, GitCommit,
  Package, Layers, Activity, Target, Zap,
  ChevronDown, ChevronRight, MoreVertical, Settings
} from 'lucide-react';
import Card from '@/components/common/Card';
import Button from '@/components/common/Button';
import { apiClient } from '@/services/api';

interface ModelVersion {
  version_id: string;
  model_id: string;
  version_number: string;
  version_tag: string;
  status: 'active' | 'archived' | 'deprecated' | 'production';
  created_at: Date;
  created_by: string;
  parent_version?: string;
  commit_message: string;
  model_artifacts: {
    model_file: string;
    file_size: string;
    framework: string;
    framework_version: string;
  };
  training_config: {
    algorithm: string;
    hyperparameters: Record<string, any>;
    features: string[];
    target: string;
  };
  performance_metrics: {
    accuracy?: number;
    precision?: number;
    recall?: number;
    f1_score?: number;
    auc?: number;
    mse?: number;
    mae?: number;
    r2?: number;
    custom_metrics?: Record<string, number>;
  };
  dataset_info: {
    dataset_id: string;
    dataset_version: string;
    train_size: number;
    test_size: number;
    validation_size?: number;
  };
  deployment_info?: {
    endpoint: string;
    deployed_at: Date;
    deployment_status: string;
  };
}

interface Experiment {
  experiment_id: string;
  name: string;
  description: string;
  status: 'running' | 'completed' | 'failed' | 'paused';
  created_at: Date;
  updated_at: Date;
  created_by: string;
  tags: string[];
  runs: ExperimentRun[];
  best_run?: ExperimentRun;
  baseline_run?: ExperimentRun;
  objective: {
    metric: string;
    direction: 'maximize' | 'minimize';
  };
}

interface ExperimentRun {
  run_id: string;
  experiment_id: string;
  run_number: number;
  status: 'running' | 'completed' | 'failed';
  started_at: Date;
  completed_at?: Date;
  duration_seconds?: number;
  hyperparameters: Record<string, any>;
  metrics: Record<string, number>;
  artifacts: {
    model_path?: string;
    logs_path?: string;
    plots?: string[];
  };
  system_metrics?: {
    cpu_usage: number;
    memory_usage: number;
    gpu_usage?: number;
  };
  git_info?: {
    commit_hash: string;
    branch: string;
    dirty: boolean;
  };
  notes?: string;
}

interface Comparison {
  versions: ModelVersion[];
  metrics_comparison: Record<string, number[]>;
  feature_differences: string[][];
  hyperparameter_differences: Record<string, any[]>;
}

const ModelVersioningExperimentTracking: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'versions' | 'experiments' | 'compare'>('versions');
  const [modelVersions, setModelVersions] = useState<ModelVersion[]>([]);
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [selectedVersions, setSelectedVersions] = useState<string[]>([]);
  const [selectedExperiment, setSelectedExperiment] = useState<Experiment | null>(null);
  const [showNewVersionModal, setShowNewVersionModal] = useState(false);
  const [showNewExperimentModal, setShowNewExperimentModal] = useState(false);
  const [filterStatus, setFilterStatus] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState<'date' | 'performance' | 'name'>('date');

  // Mock data
  const mockVersions: ModelVersion[] = [
    {
      version_id: 'v-1',
      model_id: 'model-001',
      version_number: 'v2.1.0',
      version_tag: 'production',
      status: 'production',
      created_at: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000),
      created_by: 'sarah.chen@company.com',
      parent_version: 'v-2',
      commit_message: 'Improved feature engineering and hyperparameter tuning',
      model_artifacts: {
        model_file: 'credit_risk_v2.1.0.pkl',
        file_size: '125 MB',
        framework: 'scikit-learn',
        framework_version: '1.3.0'
      },
      training_config: {
        algorithm: 'RandomForestClassifier',
        hyperparameters: {
          n_estimators: 200,
          max_depth: 20,
          min_samples_split: 5,
          class_weight: 'balanced'
        },
        features: ['age', 'income', 'credit_score', 'debt_ratio', 'employment_length'],
        target: 'default_risk'
      },
      performance_metrics: {
        accuracy: 0.924,
        precision: 0.891,
        recall: 0.876,
        f1_score: 0.883,
        auc: 0.952
      },
      dataset_info: {
        dataset_id: 'ds-credit-2024',
        dataset_version: 'v3.2',
        train_size: 80000,
        test_size: 20000,
        validation_size: 10000
      },
      deployment_info: {
        endpoint: 'https://api.company.com/models/credit-risk/v2.1.0',
        deployed_at: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000),
        deployment_status: 'active'
      }
    },
    {
      version_id: 'v-2',
      model_id: 'model-001',
      version_number: 'v2.0.0',
      version_tag: 'stable',
      status: 'active',
      created_at: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
      created_by: 'michael.torres@company.com',
      parent_version: 'v-3',
      commit_message: 'Major update: Added new features and retrained on larger dataset',
      model_artifacts: {
        model_file: 'credit_risk_v2.0.0.pkl',
        file_size: '118 MB',
        framework: 'scikit-learn',
        framework_version: '1.3.0'
      },
      training_config: {
        algorithm: 'RandomForestClassifier',
        hyperparameters: {
          n_estimators: 150,
          max_depth: 15,
          min_samples_split: 10
        },
        features: ['age', 'income', 'credit_score', 'debt_ratio'],
        target: 'default_risk'
      },
      performance_metrics: {
        accuracy: 0.912,
        precision: 0.878,
        recall: 0.865,
        f1_score: 0.871,
        auc: 0.941
      },
      dataset_info: {
        dataset_id: 'ds-credit-2024',
        dataset_version: 'v3.0',
        train_size: 70000,
        test_size: 17500,
        validation_size: 8750
      }
    },
    {
      version_id: 'v-3',
      model_id: 'model-001',
      version_number: 'v1.9.2',
      version_tag: 'archived',
      status: 'archived',
      created_at: new Date(Date.now() - 14 * 24 * 60 * 60 * 1000),
      created_by: 'emily.wang@company.com',
      commit_message: 'Bug fix: Corrected feature scaling issue',
      model_artifacts: {
        model_file: 'credit_risk_v1.9.2.pkl',
        file_size: '95 MB',
        framework: 'scikit-learn',
        framework_version: '1.2.0'
      },
      training_config: {
        algorithm: 'GradientBoostingClassifier',
        hyperparameters: {
          n_estimators: 100,
          learning_rate: 0.1,
          max_depth: 10
        },
        features: ['age', 'income', 'credit_score'],
        target: 'default_risk'
      },
      performance_metrics: {
        accuracy: 0.898,
        precision: 0.862,
        recall: 0.851,
        f1_score: 0.856,
        auc: 0.928
      },
      dataset_info: {
        dataset_id: 'ds-credit-2023',
        dataset_version: 'v2.8',
        train_size: 50000,
        test_size: 12500
      }
    }
  ];

  const mockExperiments: Experiment[] = [
    {
      experiment_id: 'exp-1',
      name: 'Credit Risk Model Hyperparameter Search',
      description: 'Grid search for optimal Random Forest hyperparameters',
      status: 'completed',
      created_at: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000),
      updated_at: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000),
      created_by: 'sarah.chen@company.com',
      tags: ['hyperparameter-tuning', 'random-forest', 'credit-risk'],
      objective: {
        metric: 'f1_score',
        direction: 'maximize'
      },
      runs: [
        {
          run_id: 'run-1',
          experiment_id: 'exp-1',
          run_number: 1,
          status: 'completed',
          started_at: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000),
          completed_at: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000 + 2 * 60 * 60 * 1000),
          duration_seconds: 7200,
          hyperparameters: {
            n_estimators: 100,
            max_depth: 10,
            min_samples_split: 10
          },
          metrics: {
            accuracy: 0.905,
            precision: 0.871,
            recall: 0.859,
            f1_score: 0.865,
            auc: 0.935
          },
          artifacts: {
            model_path: '/experiments/exp-1/run-1/model.pkl',
            logs_path: '/experiments/exp-1/run-1/logs.txt'
          },
          system_metrics: {
            cpu_usage: 78.5,
            memory_usage: 4.2,
            gpu_usage: 0
          }
        },
        {
          run_id: 'run-2',
          experiment_id: 'exp-1',
          run_number: 2,
          status: 'completed',
          started_at: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000 + 2.5 * 60 * 60 * 1000),
          completed_at: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000 + 4.5 * 60 * 60 * 1000),
          duration_seconds: 7200,
          hyperparameters: {
            n_estimators: 200,
            max_depth: 20,
            min_samples_split: 5
          },
          metrics: {
            accuracy: 0.924,
            precision: 0.891,
            recall: 0.876,
            f1_score: 0.883,
            auc: 0.952
          },
          artifacts: {
            model_path: '/experiments/exp-1/run-2/model.pkl',
            logs_path: '/experiments/exp-1/run-2/logs.txt'
          },
          system_metrics: {
            cpu_usage: 85.2,
            memory_usage: 5.8,
            gpu_usage: 0
          },
          notes: 'Best performing configuration'
        }
      ],
      best_run: undefined // Will be set below
    },
    {
      experiment_id: 'exp-2',
      name: 'Feature Engineering Impact Study',
      description: 'Testing impact of new engineered features on model performance',
      status: 'running',
      created_at: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000),
      updated_at: new Date(Date.now() - 2 * 60 * 60 * 1000),
      created_by: 'michael.torres@company.com',
      tags: ['feature-engineering', 'ablation-study'],
      objective: {
        metric: 'auc',
        direction: 'maximize'
      },
      runs: [
        {
          run_id: 'run-3',
          experiment_id: 'exp-2',
          run_number: 1,
          status: 'completed',
          started_at: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000),
          completed_at: new Date(Date.now() - 20 * 60 * 60 * 1000),
          duration_seconds: 14400,
          hyperparameters: {
            features: ['age', 'income', 'credit_score', 'debt_ratio'],
            feature_engineering: 'baseline'
          },
          metrics: {
            accuracy: 0.912,
            precision: 0.878,
            recall: 0.865,
            f1_score: 0.871,
            auc: 0.941
          },
          artifacts: {
            model_path: '/experiments/exp-2/run-1/model.pkl'
          }
        },
        {
          run_id: 'run-4',
          experiment_id: 'exp-2',
          run_number: 2,
          status: 'running',
          started_at: new Date(Date.now() - 2 * 60 * 60 * 1000),
          hyperparameters: {
            features: ['age', 'income', 'credit_score', 'debt_ratio', 'income_age_ratio', 'debt_income_interaction'],
            feature_engineering: 'advanced'
          },
          metrics: {
            accuracy: 0.918,
            precision: 0.885,
            recall: 0.871,
            f1_score: 0.878,
            auc: 0.948
          },
          artifacts: {}
        }
      ]
    }
  ];

  // Set best runs
  mockExperiments[0].best_run = mockExperiments[0].runs[1];
  mockExperiments[0].baseline_run = mockExperiments[0].runs[0];

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      // In production, this would call the API
      setModelVersions(mockVersions);
      setExperiments(mockExperiments);
    } catch (error) {
      console.error('Error loading data:', error);
      setModelVersions(mockVersions);
      setExperiments(mockExperiments);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'production':
      case 'active':
      case 'completed':
        return 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400';
      case 'running':
        return 'bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-400';
      case 'archived':
      case 'paused':
        return 'bg-gray-100 text-gray-800 dark:bg-gray-900/20 dark:text-gray-400';
      case 'deprecated':
      case 'failed':
        return 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400';
      default:
        return 'bg-gray-100 text-gray-800 dark:bg-gray-900/20 dark:text-gray-400';
    }
  };

  const getMetricChange = (current: number, previous: number) => {
    const change = ((current - previous) / previous) * 100;
    return {
      value: change,
      direction: change > 0 ? 'up' : 'down',
      color: change > 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
    };
  };

  const formatDuration = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    if (hours > 0) {
      return `${hours}h ${minutes}m`;
    }
    return `${minutes}m`;
  };

  const toggleVersionSelection = (versionId: string) => {
    setSelectedVersions(prev => {
      if (prev.includes(versionId)) {
        return prev.filter(id => id !== versionId);
      } else if (prev.length < 3) {
        return [...prev, versionId];
      }
      return prev;
    });
  };

  const promoteToProduction = (versionId: string) => {
    setModelVersions(prev => prev.map(v => ({
      ...v,
      status: v.version_id === versionId ? 'production' : (v.status === 'production' ? 'active' : v.status)
    })));
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-neutral-900 dark:text-neutral-100 flex items-center">
            <div className="w-8 h-8 bg-gradient-to-r from-purple-500 to-blue-500 rounded-lg flex items-center justify-center mr-3">
              <GitBranch className="w-5 h-5 text-white" />
            </div>
            Model Versioning & Experiment Tracking
          </h1>
          <p className="text-neutral-600 dark:text-neutral-400 mt-1">
            Track model versions, experiments, and compare performance across iterations
          </p>
        </div>
        
        <div className="flex items-center space-x-2">
          <Button
            variant="outline"
            size="sm"
            leftIcon={<Download className="w-4 h-4" />}
          >
            Export
          </Button>
          <Button
            variant="outline"
            size="sm"
            leftIcon={<Settings className="w-4 h-4" />}
          >
            Settings
          </Button>
          <Button
            variant="primary"
            size="sm"
            leftIcon={<Plus className="w-4 h-4" />}
            onClick={() => activeTab === 'versions' ? setShowNewVersionModal(true) : setShowNewExperimentModal(true)}
          >
            New {activeTab === 'versions' ? 'Version' : 'Experiment'}
          </Button>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex space-x-1 bg-neutral-100 dark:bg-neutral-800 rounded-lg p-1">
        {[
          { id: 'versions', label: 'Model Versions', icon: <GitCommit className="w-4 h-4" /> },
          { id: 'experiments', label: 'Experiments', icon: <Activity className="w-4 h-4" /> },
          { id: 'compare', label: 'Compare', icon: <BarChart3 className="w-4 h-4" /> }
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as any)}
            className={`flex-1 flex items-center justify-center space-x-2 px-4 py-2 rounded-md text-sm font-medium transition-all duration-200 ${
              activeTab === tab.id
                ? 'bg-white dark:bg-neutral-700 text-primary-600 dark:text-primary-400 shadow-sm'
                : 'text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-neutral-100'
            }`}
          >
            {tab.icon}
            <span>{tab.label}</span>
          </button>
        ))}
      </div>

      {/* Filters */}
      <Card>
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <Search className="w-4 h-4 text-neutral-400" />
              <input
                type="text"
                placeholder="Search versions or experiments..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="border-0 bg-transparent text-neutral-900 dark:text-neutral-100 placeholder-neutral-400 focus:outline-none"
              />
            </div>
            
            <div className="flex items-center space-x-1 bg-neutral-100 dark:bg-neutral-800 rounded-lg p-1">
              {['all', 'active', 'archived'].map((status) => (
                <button
                  key={status}
                  onClick={() => setFilterStatus(status)}
                  className={`px-3 py-1 rounded-md text-sm font-medium capitalize transition-colors ${
                    filterStatus === status
                      ? 'bg-white dark:bg-neutral-700 text-primary-600 dark:text-primary-400 shadow-sm'
                      : 'text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-neutral-100'
                  }`}
                >
                  {status}
                </button>
              ))}
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            <span className="text-sm text-neutral-500 dark:text-neutral-400">Sort by:</span>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as any)}
              className="text-sm border-0 bg-transparent text-neutral-900 dark:text-neutral-100 focus:outline-none"
            >
              <option value="date">Date</option>
              <option value="performance">Performance</option>
              <option value="name">Name</option>
            </select>
          </div>
        </div>
      </Card>

      {/* Content */}
      <AnimatePresence mode="wait">
        {activeTab === 'versions' && (
          <motion.div
            key="versions"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="space-y-4"
          >
            {modelVersions.map((version, index) => {
              const previousVersion = modelVersions.find(v => v.version_id === version.parent_version);
              
              return (
                <motion.div
                  key={version.version_id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                >
                  <Card className="hover:shadow-lg transition-all duration-200">
                    <div className="space-y-4">
                      {/* Version Header */}
                      <div className="flex items-start justify-between">
                        <div className="flex items-start space-x-4">
                          <div className="p-2 bg-purple-100 dark:bg-purple-900/20 rounded-lg">
                            <Package className="w-6 h-6 text-purple-600 dark:text-purple-400" />
                          </div>
                          
                          <div className="flex-1">
                            <div className="flex items-center space-x-3 mb-1">
                              <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
                                {version.version_number}
                              </h3>
                              <span className={`px-2 py-1 text-xs rounded-full font-medium ${getStatusColor(version.status)}`}>
                                {version.status}
                              </span>
                              {version.version_tag && (
                                <span className="flex items-center space-x-1 text-xs text-neutral-500 dark:text-neutral-400">
                                  <Tag className="w-3 h-3" />
                                  <span>{version.version_tag}</span>
                                </span>
                              )}
                            </div>
                            
                            <p className="text-sm text-neutral-600 dark:text-neutral-400 mb-2">
                              {version.commit_message}
                            </p>
                            
                            <div className="flex items-center space-x-4 text-xs text-neutral-500 dark:text-neutral-400">
                              <span className="flex items-center space-x-1">
                                <User className="w-3 h-3" />
                                <span>{version.created_by.split('@')[0]}</span>
                              </span>
                              <span className="flex items-center space-x-1">
                                <Clock className="w-3 h-3" />
                                <span>{new Date(version.created_at).toLocaleDateString()}</span>
                              </span>
                              <span className="flex items-center space-x-1">
                                <HardDrive className="w-3 h-3" />
                                <span>{version.model_artifacts.file_size}</span>
                              </span>
                            </div>
                          </div>
                        </div>
                        
                        <div className="flex items-center space-x-2">
                          <input
                            type="checkbox"
                            checked={selectedVersions.includes(version.version_id)}
                            onChange={() => toggleVersionSelection(version.version_id)}
                            className="w-4 h-4 text-primary-600 border-neutral-300 rounded focus:ring-primary-500"
                          />
                          <Button variant="ghost" size="sm">
                            <MoreVertical className="w-4 h-4" />
                          </Button>
                        </div>
                      </div>
                      
                      {/* Performance Metrics */}
                      <div className="grid grid-cols-5 gap-4">
                        {Object.entries(version.performance_metrics).slice(0, 5).map(([metric, value]) => (
                          <div key={metric} className="text-center">
                            <div className="text-2xl font-bold text-neutral-900 dark:text-neutral-100">
                              {typeof value === 'number' ? value.toFixed(3) : value}
                            </div>
                            <div className="text-xs text-neutral-500 dark:text-neutral-400 capitalize">
                              {metric.replace('_', ' ')}
                            </div>
                            {previousVersion && previousVersion.performance_metrics[metric as keyof typeof previousVersion.performance_metrics] && (
                              <div className={`text-xs flex items-center justify-center space-x-1 mt-1 ${
                                getMetricChange(
                                  value as number, 
                                  previousVersion.performance_metrics[metric as keyof typeof previousVersion.performance_metrics] as number
                                ).color
                              }`}>
                                {getMetricChange(
                                  value as number,
                                  previousVersion.performance_metrics[metric as keyof typeof previousVersion.performance_metrics] as number
                                ).direction === 'up' ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
                                <span>
                                  {Math.abs(getMetricChange(
                                    value as number,
                                    previousVersion.performance_metrics[metric as keyof typeof previousVersion.performance_metrics] as number
                                  ).value).toFixed(1)}%
                                </span>
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                      
                      {/* Additional Info */}
                      <div className="grid grid-cols-3 gap-4 pt-4 border-t border-neutral-200 dark:border-neutral-700">
                        <div>
                          <h4 className="text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-1">
                            Training Configuration
                          </h4>
                          <div className="space-y-1 text-xs text-neutral-600 dark:text-neutral-400">
                            <div>Algorithm: {version.training_config.algorithm}</div>
                            <div>Features: {version.training_config.features.length}</div>
                            <div>Target: {version.training_config.target}</div>
                          </div>
                        </div>
                        
                        <div>
                          <h4 className="text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-1">
                            Dataset Info
                          </h4>
                          <div className="space-y-1 text-xs text-neutral-600 dark:text-neutral-400">
                            <div>Version: {version.dataset_info.dataset_version}</div>
                            <div>Train: {version.dataset_info.train_size.toLocaleString()} samples</div>
                            <div>Test: {version.dataset_info.test_size.toLocaleString()} samples</div>
                          </div>
                        </div>
                        
                        <div>
                          <h4 className="text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-1">
                            Deployment Status
                          </h4>
                          {version.deployment_info ? (
                            <div className="space-y-1 text-xs text-neutral-600 dark:text-neutral-400">
                              <div className="flex items-center space-x-1">
                                <CheckCircle className="w-3 h-3 text-green-500" />
                                <span>Deployed</span>
                              </div>
                              <div className="truncate">{version.deployment_info.endpoint}</div>
                            </div>
                          ) : (
                            <div className="text-xs text-neutral-500 dark:text-neutral-400">
                              Not deployed
                            </div>
                          )}
                        </div>
                      </div>
                      
                      {/* Actions */}
                      <div className="flex items-center justify-between pt-4 border-t border-neutral-200 dark:border-neutral-700">
                        <div className="flex items-center space-x-2">
                          {version.parent_version && (
                            <Button variant="outline" size="sm" leftIcon={<GitBranch className="w-4 h-4" />}>
                              View Diff
                            </Button>
                          )}
                          <Button variant="outline" size="sm" leftIcon={<Eye className="w-4 h-4" />}>
                            Details
                          </Button>
                          <Button variant="outline" size="sm" leftIcon={<Download className="w-4 h-4" />}>
                            Download
                          </Button>
                        </div>
                        
                        <div className="flex items-center space-x-2">
                          {version.status !== 'production' && (
                            <Button
                              variant="outline"
                              size="sm"
                              leftIcon={<ArrowRight className="w-4 h-4" />}
                              onClick={() => promoteToProduction(version.version_id)}
                            >
                              Promote to Production
                            </Button>
                          )}
                          <Button
                            variant="primary"
                            size="sm"
                            leftIcon={<Layers className="w-4 h-4" />}
                          >
                            Create New Version
                          </Button>
                        </div>
                      </div>
                    </div>
                  </Card>
                </motion.div>
              );
            })}
          </motion.div>
        )}

        {activeTab === 'experiments' && (
          <motion.div
            key="experiments"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="grid grid-cols-1 lg:grid-cols-3 gap-6"
          >
            {/* Experiments List */}
            <div className="lg:col-span-2 space-y-4">
              {experiments.map((experiment, index) => (
                <motion.div
                  key={experiment.experiment_id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  onClick={() => setSelectedExperiment(experiment)}
                  className="cursor-pointer"
                >
                  <Card className={`hover:shadow-lg transition-all duration-200 ${
                    selectedExperiment?.experiment_id === experiment.experiment_id ? 'ring-2 ring-primary-500' : ''
                  }`}>
                    <div className="space-y-3">
                      {/* Experiment Header */}
                      <div className="flex items-start justify-between">
                        <div>
                          <h3 className="font-semibold text-neutral-900 dark:text-neutral-100 mb-1">
                            {experiment.name}
                          </h3>
                          <p className="text-sm text-neutral-600 dark:text-neutral-400">
                            {experiment.description}
                          </p>
                        </div>
                        
                        <span className={`px-2 py-1 text-xs rounded-full font-medium ${getStatusColor(experiment.status)}`}>
                          {experiment.status}
                        </span>
                      </div>
                      
                      {/* Progress & Stats */}
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-4 text-sm">
                          <span className="text-neutral-600 dark:text-neutral-400">
                            {experiment.runs.length} runs
                          </span>
                          {experiment.best_run && (
                            <span className="flex items-center space-x-1 text-green-600 dark:text-green-400">
                              <Star className="w-4 h-4" />
                              <span>
                                Best {experiment.objective.metric}: {experiment.best_run.metrics[experiment.objective.metric]?.toFixed(3)}
                              </span>
                            </span>
                          )}
                        </div>
                        
                        {experiment.status === 'running' && (
                          <div className="flex items-center space-x-2">
                            <div className="w-24 bg-neutral-200 dark:bg-neutral-700 rounded-full h-2">
                              <motion.div
                                className="bg-primary-500 h-2 rounded-full"
                                initial={{ width: 0 }}
                                animate={{ width: '60%' }}
                                transition={{ duration: 2, repeat: Infinity }}
                              />
                            </div>
                            <Play className="w-4 h-4 text-primary-500 animate-pulse" />
                          </div>
                        )}
                      </div>
                      
                      {/* Tags */}
                      <div className="flex flex-wrap gap-1">
                        {experiment.tags.map((tag) => (
                          <span
                            key={tag}
                            className="px-2 py-1 text-xs bg-neutral-100 dark:bg-neutral-800 text-neutral-600 dark:text-neutral-400 rounded"
                          >
                            {tag}
                          </span>
                        ))}
                      </div>
                    </div>
                  </Card>
                </motion.div>
              ))}
            </div>

            {/* Experiment Details */}
            {selectedExperiment && (
              <Card
                title={
                  <div className="flex items-center justify-between">
                    <span>Experiment Runs</span>
                    <Button variant="ghost" size="sm" onClick={() => setSelectedExperiment(null)}>
                      <X className="w-4 h-4" />
                    </Button>
                  </div>
                }
              >
                <div className="space-y-4">
                  <div className="text-sm text-neutral-600 dark:text-neutral-400">
                    <div>Objective: {selectedExperiment.objective.direction} {selectedExperiment.objective.metric}</div>
                    <div>Created by: {selectedExperiment.created_by.split('@')[0]}</div>
                  </div>
                  
                  <div className="space-y-3">
                    {selectedExperiment.runs.map((run) => (
                      <div
                        key={run.run_id}
                        className={`p-3 rounded-lg border ${
                          run.run_id === selectedExperiment.best_run?.run_id
                            ? 'border-green-500 bg-green-50 dark:bg-green-900/20'
                            : 'border-neutral-200 dark:border-neutral-700'
                        }`}
                      >
                        <div className="flex items-start justify-between mb-2">
                          <div className="flex items-center space-x-2">
                            <span className="font-medium text-sm text-neutral-900 dark:text-neutral-100">
                              Run #{run.run_number}
                            </span>
                            {run.run_id === selectedExperiment.best_run?.run_id && (
                              <Star className="w-4 h-4 text-green-500" />
                            )}
                          </div>
                          <span className={`text-xs ${getStatusColor(run.status)}`}>
                            {run.status}
                          </span>
                        </div>
                        
                        <div className="grid grid-cols-2 gap-2 text-xs">
                          {Object.entries(run.metrics).slice(0, 4).map(([metric, value]) => (
                            <div key={metric}>
                              <span className="text-neutral-500 dark:text-neutral-400">
                                {metric}:
                              </span>
                              <span className="ml-1 font-medium text-neutral-900 dark:text-neutral-100">
                                {value.toFixed(3)}
                              </span>
                            </div>
                          ))}
                        </div>
                        
                        {run.duration_seconds && (
                          <div className="mt-2 text-xs text-neutral-500 dark:text-neutral-400">
                            Duration: {formatDuration(run.duration_seconds)}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                  
                  <div className="pt-4 border-t border-neutral-200 dark:border-neutral-700">
                    <Button variant="primary" size="sm" className="w-full">
                      View Full Results
                    </Button>
                  </div>
                </div>
              </Card>
            )}
          </motion.div>
        )}

        {activeTab === 'compare' && (
          <motion.div
            key="compare"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
          >
            {selectedVersions.length >= 2 ? (
              <Card title="Model Version Comparison">
                <div className="space-y-6">
                  {/* Selected Versions */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {selectedVersions.map((versionId) => {
                      const version = modelVersions.find(v => v.version_id === versionId);
                      if (!version) return null;
                      
                      return (
                        <div
                          key={versionId}
                          className="p-4 bg-neutral-50 dark:bg-neutral-800 rounded-lg"
                        >
                          <div className="flex items-center justify-between mb-2">
                            <h4 className="font-semibold text-neutral-900 dark:text-neutral-100">
                              {version.version_number}
                            </h4>
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => toggleVersionSelection(versionId)}
                            >
                              <X className="w-4 h-4" />
                            </Button>
                          </div>
                          <div className="text-sm text-neutral-600 dark:text-neutral-400">
                            {version.training_config.algorithm}
                          </div>
                          <div className="text-xs text-neutral-500 dark:text-neutral-400 mt-1">
                            {new Date(version.created_at).toLocaleDateString()}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                  
                  {/* Metrics Comparison */}
                  <div>
                    <h4 className="font-medium text-neutral-900 dark:text-neutral-100 mb-3">
                      Performance Metrics
                    </h4>
                    <div className="space-y-3">
                      {['accuracy', 'precision', 'recall', 'f1_score', 'auc'].map((metric) => {
                        const values = selectedVersions.map(vId => {
                          const version = modelVersions.find(v => v.version_id === vId);
                          return version?.performance_metrics[metric as keyof typeof version.performance_metrics] as number || 0;
                        });
                        const maxValue = Math.max(...values);
                        
                        return (
                          <div key={metric} className="space-y-2">
                            <div className="flex items-center justify-between text-sm">
                              <span className="font-medium capitalize text-neutral-700 dark:text-neutral-300">
                                {metric.replace('_', ' ')}
                              </span>
                              <span className="text-neutral-500 dark:text-neutral-400">
                                Best: {maxValue.toFixed(3)}
                              </span>
                            </div>
                            <div className="space-y-1">
                              {selectedVersions.map((vId, idx) => {
                                const version = modelVersions.find(v => v.version_id === vId);
                                const value = version?.performance_metrics[metric as keyof typeof version.performance_metrics] as number || 0;
                                const percentage = (value / maxValue) * 100;
                                const isMax = value === maxValue;
                                
                                return (
                                  <div key={vId} className="flex items-center space-x-2">
                                    <span className="text-xs w-16 text-neutral-600 dark:text-neutral-400">
                                      v{idx + 1}
                                    </span>
                                    <div className="flex-1 bg-neutral-200 dark:bg-neutral-700 rounded-full h-6 relative">
                                      <motion.div
                                        className={`h-6 rounded-full flex items-center justify-end px-2 ${
                                          isMax ? 'bg-green-500' : 'bg-primary-500'
                                        }`}
                                        initial={{ width: 0 }}
                                        animate={{ width: `${percentage}%` }}
                                        transition={{ duration: 0.5, delay: idx * 0.1 }}
                                      >
                                        <span className="text-xs text-white font-medium">
                                          {value.toFixed(3)}
                                        </span>
                                      </motion.div>
                                    </div>
                                  </div>
                                );
                              })}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                  
                  {/* Feature Comparison */}
                  <div>
                    <h4 className="font-medium text-neutral-900 dark:text-neutral-100 mb-3">
                      Feature Sets
                    </h4>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      {selectedVersions.map((vId, idx) => {
                        const version = modelVersions.find(v => v.version_id === vId);
                        if (!version) return null;
                        
                        return (
                          <div key={vId} className="space-y-2">
                            <h5 className="text-sm font-medium text-neutral-700 dark:text-neutral-300">
                              Version {idx + 1} Features ({version.training_config.features.length})
                            </h5>
                            <div className="space-y-1">
                              {version.training_config.features.map((feature) => (
                                <div
                                  key={feature}
                                  className="text-xs px-2 py-1 bg-neutral-100 dark:bg-neutral-800 rounded"
                                >
                                  {feature}
                                </div>
                              ))}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>
              </Card>
            ) : (
              <Card>
                <div className="text-center py-12">
                  <BarChart3 className="w-12 h-12 text-neutral-400 mx-auto mb-4" />
                  <p className="text-neutral-600 dark:text-neutral-400">
                    Select at least 2 model versions to compare
                  </p>
                  <p className="text-sm text-neutral-500 dark:text-neutral-400 mt-2">
                    You have selected {selectedVersions.length} version{selectedVersions.length !== 1 ? 's' : ''}
                  </p>
                </div>
              </Card>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* New Version Modal */}
      <AnimatePresence>
        {showNewVersionModal && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
            onClick={() => setShowNewVersionModal(false)}
          >
            <motion.div
              initial={{ scale: 0.9 }}
              animate={{ scale: 1 }}
              exit={{ scale: 0.9 }}
              className="bg-white dark:bg-neutral-800 rounded-lg p-6 w-full max-w-lg mx-4"
              onClick={(e) => e.stopPropagation()}
            >
              <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4">
                Create New Model Version
              </h3>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                    Version Number
                  </label>
                  <input
                    type="text"
                    placeholder="e.g., v2.2.0"
                    className="w-full p-3 border border-neutral-200 dark:border-neutral-700 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                    Commit Message
                  </label>
                  <textarea
                    rows={3}
                    placeholder="Describe the changes in this version..."
                    className="w-full p-3 border border-neutral-200 dark:border-neutral-700 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100 resize-none"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                    Model File
                  </label>
                  <div className="border-2 border-dashed border-neutral-300 dark:border-neutral-600 rounded-lg p-4 text-center">
                    <Upload className="w-8 h-8 text-neutral-400 mx-auto mb-2" />
                    <p className="text-sm text-neutral-600 dark:text-neutral-400">
                      Drop your model file here or click to browse
                    </p>
                  </div>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                    Version Tag
                  </label>
                  <select className="w-full p-3 border border-neutral-200 dark:border-neutral-700 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100">
                    <option>stable</option>
                    <option>experimental</option>
                    <option>production-candidate</option>
                  </select>
                </div>
              </div>
              
              <div className="flex justify-end space-x-3 mt-6">
                <Button variant="outline" onClick={() => setShowNewVersionModal(false)}>
                  Cancel
                </Button>
                <Button variant="primary">
                  Create Version
                </Button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default ModelVersioningExperimentTracking;