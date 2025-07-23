import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Rocket, Play, Pause, Square, Settings, GitBranch, Package,
  CheckCircle, AlertTriangle, Clock, Eye, Download, Upload,
  Server, Globe, Lock, Unlock, Activity, TrendingUp,
  Database, Cloud, Shield, Zap, Target, Users, Bell,
  Monitor, Code, FileText, BarChart3, RefreshCw, ArrowRight,
  Copy, Edit, Trash2, MoreVertical, ExternalLink, Terminal
} from 'lucide-react';

interface Environment {
  id: string;
  name: string;
  type: 'development' | 'staging' | 'production';
  status: 'healthy' | 'degraded' | 'unhealthy' | 'offline';
  url: string;
  region: string;
  compute_config: {
    cpu_cores: number;
    memory_gb: number;
    gpu_count?: number;
    auto_scaling: boolean;
    min_replicas: number;
    max_replicas: number;
  };
  security_config: {
    auth_enabled: boolean;
    ssl_enabled: boolean;
    rate_limiting: boolean;
    ip_whitelist?: string[];
  };
  monitoring: {
    uptime_percentage: number;
    avg_response_time: number;
    error_rate: number;
    requests_per_minute: number;
  };
}

interface DeploymentPipeline {
  id: string;
  name: string;
  description: string;
  model_id: string;
  model_name: string;
  model_version: string;
  
  // Pipeline configuration
  strategy: 'blue_green' | 'canary' | 'rolling' | 'recreate';
  auto_rollback: boolean;
  health_checks: {
    endpoint: string;
    timeout_seconds: number;
    failure_threshold: number;
    success_threshold: number;
  };
  
  // Environments
  environments: Environment[];
  current_environment: string;
  
  // Status
  status: 'created' | 'deploying' | 'deployed' | 'failed' | 'rolling_back' | 'paused';
  deployment_progress: number;
  
  // History
  deployment_history: DeploymentExecution[];
  
  // Notifications
  notification_config: {
    slack_webhook?: string;
    email_addresses?: string[];
    on_success: boolean;
    on_failure: boolean;
  };
  
  created_at: string;
  updated_at: string;
  created_by: string;
}

interface DeploymentExecution {
  id: string;
  pipeline_id: string;
  version: string;
  environment: string;
  status: 'running' | 'succeeded' | 'failed' | 'cancelled' | 'rolled_back';
  strategy: string;
  
  // Execution details
  started_at: string;
  completed_at?: string;
  duration_minutes?: number;
  triggered_by: string;
  trigger_type: 'manual' | 'automated' | 'scheduled' | 'webhook';
  
  // Progress tracking
  steps: DeploymentStep[];
  current_step: number;
  
  // Results
  artifacts: {
    logs_url: string;
    metrics_url: string;
    health_check_results: any[];
  };
  
  // Rollback info
  rollback_version?: string;
  rollback_reason?: string;
  
  error_message?: string;
}

interface DeploymentStep {
  id: string;
  name: string;
  description: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped';
  started_at?: string;
  completed_at?: string;
  duration_seconds?: number;
  logs?: string[];
  error_message?: string;
}

interface DeploymentTemplate {
  id: string;
  name: string;
  description: string;
  type: 'api_service' | 'batch_job' | 'streaming' | 'edge_deployment';
  framework_support: string[];
  default_config: any;
  environments: Partial<Environment>[];
}

const ModelDeploymentPipeline: React.FC = () => {
  const [pipelines, setPipelines] = useState<DeploymentPipeline[]>([]);
  const [selectedPipeline, setSelectedPipeline] = useState<DeploymentPipeline | null>(null);
  const [activeExecutions, setActiveExecutions] = useState<DeploymentExecution[]>([]);
  const [templates, setTemplates] = useState<DeploymentTemplate[]>([]);
  
  const [activeTab, setActiveTab] = useState<'overview' | 'pipelines' | 'executions' | 'environments' | 'templates'>('overview');
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [selectedExecution, setSelectedExecution] = useState<DeploymentExecution | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Mock data
  useEffect(() => {
    const mockEnvironments: Environment[] = [
      {
        id: 'env_dev',
        name: 'Development',
        type: 'development',
        status: 'healthy',
        url: 'https://dev-api.raia.com',
        region: 'us-east-1',
        compute_config: {
          cpu_cores: 2,
          memory_gb: 4,
          auto_scaling: false,
          min_replicas: 1,
          max_replicas: 1
        },
        security_config: {
          auth_enabled: false,
          ssl_enabled: true,
          rate_limiting: false
        },
        monitoring: {
          uptime_percentage: 99.2,
          avg_response_time: 45,
          error_rate: 0.8,
          requests_per_minute: 12
        }
      },
      {
        id: 'env_staging',
        name: 'Staging',
        type: 'staging',
        status: 'healthy',
        url: 'https://staging-api.raia.com',
        region: 'us-east-1',
        compute_config: {
          cpu_cores: 4,
          memory_gb: 8,
          auto_scaling: true,
          min_replicas: 2,
          max_replicas: 4
        },
        security_config: {
          auth_enabled: true,
          ssl_enabled: true,
          rate_limiting: true
        },
        monitoring: {
          uptime_percentage: 99.8,
          avg_response_time: 32,
          error_rate: 0.3,
          requests_per_minute: 45
        }
      },
      {
        id: 'env_prod',
        name: 'Production',
        type: 'production',
        status: 'healthy',
        url: 'https://api.raia.com',
        region: 'us-east-1',
        compute_config: {
          cpu_cores: 8,
          memory_gb: 16,
          gpu_count: 1,
          auto_scaling: true,
          min_replicas: 3,
          max_replicas: 10
        },
        security_config: {
          auth_enabled: true,
          ssl_enabled: true,
          rate_limiting: true,
          ip_whitelist: ['10.0.0.0/8', '172.16.0.0/12']
        },
        monitoring: {
          uptime_percentage: 99.95,
          avg_response_time: 28,
          error_rate: 0.1,
          requests_per_minute: 1250
        }
      }
    ];

    const mockSteps: DeploymentStep[] = [
      {
        id: 'step_1',
        name: 'Validate Model',
        description: 'Validate model artifacts and dependencies',
        status: 'completed',
        started_at: '2024-01-16T10:00:00Z',
        completed_at: '2024-01-16T10:01:30Z',
        duration_seconds: 90
      },
      {
        id: 'step_2',
        name: 'Build Container',
        description: 'Build Docker container with model and dependencies',
        status: 'completed',
        started_at: '2024-01-16T10:01:30Z',
        completed_at: '2024-01-16T10:05:45Z',
        duration_seconds: 255
      },
      {
        id: 'step_3',
        name: 'Deploy to Staging',
        description: 'Deploy container to staging environment',
        status: 'running',
        started_at: '2024-01-16T10:05:45Z',
        duration_seconds: 120
      },
      {
        id: 'step_4',
        name: 'Health Checks',
        description: 'Run health checks and validation tests',
        status: 'pending'
      },
      {
        id: 'step_5',
        name: 'Deploy to Production',
        description: 'Deploy to production with blue-green strategy',
        status: 'pending'
      }
    ];

    const mockExecutions: DeploymentExecution[] = [
      {
        id: 'exec_001',
        pipeline_id: 'pipeline_001',
        version: 'v2.1.0',
        environment: 'production',
        status: 'running',
        strategy: 'blue_green',
        started_at: '2024-01-16T10:00:00Z',
        triggered_by: 'ml_engineer',
        trigger_type: 'manual',
        steps: mockSteps,
        current_step: 2,
        artifacts: {
          logs_url: 'https://logs.raia.com/exec_001',
          metrics_url: 'https://metrics.raia.com/exec_001',
          health_check_results: []
        }
      },
      {
        id: 'exec_002',
        pipeline_id: 'pipeline_002',
        version: 'v1.8.3',
        environment: 'staging',
        status: 'succeeded',
        strategy: 'rolling',
        started_at: '2024-01-16T08:30:00Z',
        completed_at: '2024-01-16T08:45:00Z',
        duration_minutes: 15,
        triggered_by: 'auto_deploy_bot',
        trigger_type: 'automated',
        steps: mockSteps.map(s => ({ ...s, status: 'completed' as const })),
        current_step: 4,
        artifacts: {
          logs_url: 'https://logs.raia.com/exec_002',
          metrics_url: 'https://metrics.raia.com/exec_002',
          health_check_results: [
            { endpoint: '/health', status: 'ok', response_time: 45 },
            { endpoint: '/predict', status: 'ok', response_time: 67 }
          ]
        }
      }
    ];

    const mockPipelines: DeploymentPipeline[] = [
      {
        id: 'pipeline_001',
        name: 'Customer Churn Model Pipeline',
        description: 'Production deployment pipeline for customer churn prediction model',
        model_id: 'model_001',
        model_name: 'Customer Churn Predictor',
        model_version: 'v2.1.0',
        strategy: 'blue_green',
        auto_rollback: true,
        health_checks: {
          endpoint: '/health',
          timeout_seconds: 30,
          failure_threshold: 3,
          success_threshold: 2
        },
        environments: mockEnvironments,
        current_environment: 'staging',
        status: 'deploying',
        deployment_progress: 65,
        deployment_history: mockExecutions,
        notification_config: {
          slack_webhook: 'https://hooks.slack.com/services/...',
          email_addresses: ['team@company.com'],
          on_success: true,
          on_failure: true
        },
        created_at: '2024-01-10T14:20:00Z',
        updated_at: '2024-01-16T10:00:00Z',
        created_by: 'ml_engineer'
      },
      {
        id: 'pipeline_002',
        name: 'NLP Sentiment Analysis Pipeline',
        description: 'High-throughput deployment for sentiment analysis API',
        model_id: 'model_002',
        model_name: 'Sentiment Analysis Pro',
        model_version: 'v1.8.3',
        strategy: 'canary',
        auto_rollback: true,
        health_checks: {
          endpoint: '/predict/health',
          timeout_seconds: 15,
          failure_threshold: 2,
          success_threshold: 3
        },
        environments: mockEnvironments,
        current_environment: 'production',
        status: 'deployed',
        deployment_progress: 100,
        deployment_history: [mockExecutions[1]],
        notification_config: {
          email_addresses: ['nlp-team@company.com'],
          on_success: false,
          on_failure: true
        },
        created_at: '2024-01-08T09:15:00Z',
        updated_at: '2024-01-16T08:45:00Z',
        created_by: 'nlp_engineer'
      }
    ];

    const mockTemplates: DeploymentTemplate[] = [
      {
        id: 'template_api',
        name: 'REST API Service',
        description: 'Standard REST API deployment for real-time inference',
        type: 'api_service',
        framework_support: ['scikit-learn', 'tensorflow', 'pytorch', 'onnx'],
        default_config: {
          container_port: 8080,
          health_check_path: '/health',
          auto_scaling: true
        },
        environments: [
          { type: 'development', compute_config: { cpu_cores: 2, memory_gb: 4 } },
          { type: 'staging', compute_config: { cpu_cores: 4, memory_gb: 8 } },
          { type: 'production', compute_config: { cpu_cores: 8, memory_gb: 16 } }
        ]
      },
      {
        id: 'template_batch',
        name: 'Batch Processing Job',
        description: 'Scheduled batch inference for large datasets',
        type: 'batch_job',
        framework_support: ['scikit-learn', 'tensorflow', 'pytorch', 'spark'],
        default_config: {
          schedule: '0 2 * * *',
          timeout_hours: 6,
          retry_count: 3
        },
        environments: [
          { type: 'development', compute_config: { cpu_cores: 4, memory_gb: 8 } },
          { type: 'production', compute_config: { cpu_cores: 16, memory_gb: 32 } }
        ]
      }
    ];

    setPipelines(mockPipelines);
    setSelectedPipeline(mockPipelines[0]);
    setActiveExecutions(mockExecutions.filter(e => e.status === 'running'));
    setTemplates(mockTemplates);
    setIsLoading(false);
  }, []);

  const handleDeployPipeline = useCallback(async (pipelineId: string, environment: string) => {
    // Implement deployment
    console.log('Deploying pipeline:', pipelineId, 'to', environment);
  }, []);

  const handleRollbackDeployment = useCallback(async (pipelineId: string, version: string) => {
    // Implement rollback
    console.log('Rolling back pipeline:', pipelineId, 'to version', version);
  }, []);

  const handlePausePipeline = useCallback(async (pipelineId: string) => {
    // Implement pipeline pause
    setPipelines(prev => prev.map(p =>
      p.id === pipelineId
        ? { ...p, status: 'paused' as const }
        : p
    ));
  }, []);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': case 'deployed': case 'succeeded': case 'completed': return 'text-green-600 bg-green-50';
      case 'deploying': case 'running': case 'rolling_back': return 'text-blue-600 bg-blue-50';
      case 'degraded': case 'paused': case 'pending': return 'text-yellow-600 bg-yellow-50';
      case 'unhealthy': case 'failed': case 'offline': return 'text-red-600 bg-red-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  const getEnvironmentIcon = (type: string) => {
    switch (type) {
      case 'development': return <Code className="w-4 h-4" />;
      case 'staging': return <Eye className="w-4 h-4" />;
      case 'production': return <Globe className="w-4 h-4" />;
      default: return <Server className="w-4 h-4" />;
    }
  };

  const renderOverview = () => (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* Summary Cards */}
      <div className="lg:col-span-3 grid grid-cols-2 md:grid-cols-4 gap-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white p-6 rounded-lg shadow-sm border border-gray-200"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Active Pipelines</p>
              <p className="text-3xl font-bold text-blue-600">
                {pipelines.filter(p => p.status !== 'paused').length}
              </p>
            </div>
            <Rocket className="w-8 h-8 text-blue-500" />
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="bg-white p-6 rounded-lg shadow-sm border border-gray-200"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Running Deployments</p>
              <p className="text-3xl font-bold text-green-600">{activeExecutions.length}</p>
            </div>
            <Activity className="w-8 h-8 text-green-500" />
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-white p-6 rounded-lg shadow-sm border border-gray-200"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Success Rate</p>
              <p className="text-3xl font-bold text-purple-600">94.2%</p>
            </div>
            <Target className="w-8 h-8 text-purple-500" />
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="bg-white p-6 rounded-lg shadow-sm border border-gray-200"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Avg Deploy Time</p>
              <p className="text-3xl font-bold text-orange-600">12m</p>
            </div>
            <Clock className="w-8 h-8 text-orange-500" />
          </div>
        </motion.div>
      </div>

      {/* Pipeline Status */}
      <div className="lg:col-span-2 bg-white rounded-lg shadow-sm border border-gray-200">
        <div className="p-6 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
            <GitBranch className="w-5 h-5" />
            Deployment Pipelines
          </h3>
        </div>
        <div className="p-6">
          <div className="space-y-4">
            {pipelines.map((pipeline) => (
              <motion.div
                key={pipeline.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                className="border border-gray-200 rounded-lg p-4 hover:border-blue-300 cursor-pointer transition-colors"
                onClick={() => setSelectedPipeline(pipeline)}
              >
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-3">
                    <Package className="w-5 h-5 text-blue-500" />
                    <div>
                      <h4 className="font-medium text-gray-900">{pipeline.name}</h4>
                      <p className="text-sm text-gray-600">{pipeline.model_name} v{pipeline.model_version}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className={`text-sm font-medium px-3 py-1 rounded-full ${getStatusColor(pipeline.status)}`}>
                      {pipeline.status.replace('_', ' ')}
                    </span>
                    <span className="text-xs text-gray-500 capitalize">{pipeline.strategy.replace('_', ' ')}</span>
                  </div>
                </div>

                {pipeline.status === 'deploying' && (
                  <div className="mb-3">
                    <div className="flex justify-between text-sm text-gray-600 mb-1">
                      <span>Deployment Progress</span>
                      <span>{pipeline.deployment_progress}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-blue-600 h-2 rounded-full transition-all duration-500"
                        style={{ width: `${pipeline.deployment_progress}%` }}
                      />
                    </div>
                  </div>
                )}

                <div className="grid grid-cols-3 gap-4 text-sm">
                  <div>
                    <span className="text-gray-600">Current Env:</span>
                    <p className="font-medium capitalize">{pipeline.current_environment}</p>
                  </div>
                  <div>
                    <span className="text-gray-600">Strategy:</span>
                    <p className="font-medium capitalize">{pipeline.strategy.replace('_', ' ')}</p>
                  </div>
                  <div>
                    <span className="text-gray-600">Auto Rollback:</span>
                    <p className="font-medium">{pipeline.auto_rollback ? 'Enabled' : 'Disabled'}</p>
                  </div>
                </div>

                <div className="mt-3 flex items-center justify-between">
                  <span className="text-xs text-gray-500">
                    Updated {new Date(pipeline.updated_at).toLocaleString()}
                  </span>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDeployPipeline(pipeline.id, 'production');
                      }}
                      className="text-blue-600 hover:text-blue-800 text-sm"
                    >
                      Deploy
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handlePausePipeline(pipeline.id);
                      }}
                      className="text-gray-600 hover:text-gray-800"
                    >
                      <Pause className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </div>

      {/* Environment Health */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200">
        <div className="p-6 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
            <Monitor className="w-5 h-5" />
            Environment Health
          </h3>
        </div>
        <div className="p-6">
          <div className="space-y-4">
            {selectedPipeline?.environments.map((env) => (
              <motion.div
                key={env.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="border border-gray-200 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    {getEnvironmentIcon(env.type)}
                    <h4 className="font-medium text-gray-900">{env.name}</h4>
                  </div>
                  <span className={`text-sm font-medium px-2 py-1 rounded-full ${getStatusColor(env.status)}`}>
                    {env.status}
                  </span>
                </div>

                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Uptime:</span>
                    <span className="font-medium">{env.monitoring.uptime_percentage}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Response Time:</span>
                    <span className="font-medium">{env.monitoring.avg_response_time}ms</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Error Rate:</span>
                    <span className="font-medium">{env.monitoring.error_rate}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Requests/min:</span>
                    <span className="font-medium">{env.monitoring.requests_per_minute}</span>
                  </div>
                </div>

                <div className="mt-3 flex items-center gap-2">
                  <button className="text-blue-600 hover:text-blue-800 text-sm flex items-center gap-1">
                    <ExternalLink className="w-3 h-3" />
                    View Logs
                  </button>
                  <button className="text-gray-600 hover:text-gray-800 text-sm flex items-center gap-1">
                    <BarChart3 className="w-3 h-3" />
                    Metrics
                  </button>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </div>

      {/* Recent Executions */}
      <div className="lg:col-span-3 bg-white rounded-lg shadow-sm border border-gray-200">
        <div className="p-6 border-b border-gray-200">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
              <Activity className="w-5 h-5" />
              Recent Deployments
            </h3>
            <button className="text-blue-600 hover:text-blue-800 text-sm">
              View All
            </button>
          </div>
        </div>
        <div className="p-6">
          <div className="space-y-4">
            {pipelines.flatMap(p => p.deployment_history).slice(0, 5).map((execution) => (
              <motion.div
                key={execution.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex items-center justify-between p-4 border border-gray-200 rounded-lg hover:border-blue-300 cursor-pointer transition-colors"
                onClick={() => setSelectedExecution(execution)}
              >
                <div className="flex items-center gap-3">
                  <div className={`w-3 h-3 rounded-full ${getStatusColor(execution.status)}`} />
                  <div>
                    <h4 className="font-medium text-gray-900">
                      {pipelines.find(p => p.id === execution.pipeline_id)?.name}
                    </h4>
                    <p className="text-sm text-gray-600">
                      {execution.version} → {execution.environment}
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-4 text-sm text-gray-600">
                  <span className="capitalize">{execution.strategy.replace('_', ' ')}</span>
                  <span>{execution.triggered_by}</span>
                  <span>{new Date(execution.started_at).toLocaleString()}</span>
                  {execution.duration_minutes && (
                    <span>{execution.duration_minutes}m</span>
                  )}
                </div>

                <div className="flex items-center gap-2">
                  <span className={`text-sm font-medium px-3 py-1 rounded-full ${getStatusColor(execution.status)}`}>
                    {execution.status}
                  </span>
                  <ArrowRight className="w-4 h-4 text-gray-400" />
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );

  const renderPipelines = () => (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-xl font-semibold text-gray-900">Deployment Pipelines</h2>
        <button
          onClick={() => setShowCreateModal(true)}
          className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors flex items-center gap-2"
        >
          <Rocket className="w-4 h-4" />
          Create Pipeline
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {pipelines.map((pipeline) => (
          <motion.div
            key={pipeline.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white rounded-lg shadow-sm border border-gray-200"
          >
            <div className="p-6 border-b border-gray-200">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-lg font-medium text-gray-900">{pipeline.name}</h3>
                <div className="flex items-center gap-2">
                  <span className={`text-sm font-medium px-3 py-1 rounded-full ${getStatusColor(pipeline.status)}`}>
                    {pipeline.status.replace('_', ' ')}
                  </span>
                  <button className="text-gray-400 hover:text-gray-600">
                    <MoreVertical className="w-4 h-4" />
                  </button>
                </div>
              </div>
              <p className="text-gray-600 text-sm">{pipeline.description}</p>
            </div>

            <div className="p-6">
              {/* Pipeline Configuration */}
              <div className="grid grid-cols-2 gap-4 mb-6">
                <div>
                  <span className="text-sm text-gray-600">Model:</span>
                  <p className="font-medium">{pipeline.model_name} v{pipeline.model_version}</p>
                </div>
                <div>
                  <span className="text-sm text-gray-600">Strategy:</span>
                  <p className="font-medium capitalize">{pipeline.strategy.replace('_', ' ')}</p>
                </div>
                <div>
                  <span className="text-sm text-gray-600">Auto Rollback:</span>
                  <p className="font-medium">{pipeline.auto_rollback ? 'Enabled' : 'Disabled'}</p>
                </div>
                <div>
                  <span className="text-sm text-gray-600">Environments:</span>
                  <p className="font-medium">{pipeline.environments.length}</p>
                </div>
              </div>

              {/* Current Deployment Progress */}
              {pipeline.status === 'deploying' && (
                <div className="mb-6">
                  <div className="flex justify-between text-sm text-gray-600 mb-2">
                    <span>Deployment Progress</span>
                    <span>{pipeline.deployment_progress}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-3">
                    <div
                      className="bg-blue-600 h-3 rounded-full transition-all duration-500"
                      style={{ width: `${pipeline.deployment_progress}%` }}
                    />
                  </div>
                </div>
              )}

              {/* Environment Status */}
              <div className="mb-6">
                <h4 className="text-sm font-medium text-gray-900 mb-2">Environments</h4>
                <div className="grid grid-cols-3 gap-2">
                  {pipeline.environments.map((env) => (
                    <div
                      key={env.id}
                      className={`text-center py-2 px-3 rounded-lg border ${
                        pipeline.current_environment === env.type 
                          ? 'border-blue-500 bg-blue-50'
                          : 'border-gray-200 bg-gray-50'
                      }`}
                    >
                      <div className="flex items-center justify-center gap-1 mb-1">
                        {getEnvironmentIcon(env.type)}
                        <span className="text-xs font-medium capitalize">{env.type}</span>
                      </div>
                      <div className={`w-2 h-2 rounded-full mx-auto ${
                        env.status === 'healthy' ? 'bg-green-500' :
                        env.status === 'degraded' ? 'bg-yellow-500' : 'bg-red-500'
                      }`} />
                    </div>
                  ))}
                </div>
              </div>

              {/* Actions */}
              <div className="flex gap-2">
                <button
                  onClick={() => handleDeployPipeline(pipeline.id, 'production')}
                  className="flex-1 bg-blue-600 text-white py-2 px-3 rounded-lg hover:bg-blue-700 transition-colors text-sm flex items-center justify-center gap-2"
                  disabled={pipeline.status === 'deploying'}
                >
                  <Rocket className="w-4 h-4" />
                  Deploy
                </button>
                <button
                  onClick={() => handleRollbackDeployment(pipeline.id, 'previous')}
                  className="bg-gray-100 text-gray-700 py-2 px-3 rounded-lg hover:bg-gray-200 transition-colors text-sm flex items-center gap-2"
                >
                  <RefreshCw className="w-4 h-4" />
                  Rollback
                </button>
                <button
                  onClick={() => setSelectedPipeline(pipeline)}
                  className="bg-gray-100 text-gray-700 py-2 px-3 rounded-lg hover:bg-gray-200 transition-colors"
                >
                  <Eye className="w-4 h-4" />
                </button>
              </div>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
        >
          <Rocket className="w-8 h-8 text-blue-500" />
        </motion.div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto p-6">
      <div className="mb-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 mb-2">
              Model Deployment Pipeline
            </h1>
            <p className="text-gray-600">
              Automated deployment, monitoring, and management of ML models in production environments
            </p>
          </div>
          <div className="flex items-center gap-3">
            <button className="bg-gray-100 text-gray-700 px-4 py-2 rounded-lg hover:bg-gray-200 transition-colors flex items-center gap-2">
              <FileText className="w-4 h-4" />
              Documentation
            </button>
            <button
              onClick={() => setShowCreateModal(true)}
              className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors flex items-center gap-2"
            >
              <Rocket className="w-4 h-4" />
              Create Pipeline
            </button>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="flex space-x-1 mb-6 bg-gray-100 p-1 rounded-lg">
        {[
          { key: 'overview', label: 'Overview', icon: BarChart3 },
          { key: 'pipelines', label: 'Pipelines', icon: GitBranch },
          { key: 'executions', label: 'Executions', icon: Activity },
          { key: 'environments', label: 'Environments', icon: Server },
          { key: 'templates', label: 'Templates', icon: Package }
        ].map(({ key, label, icon: Icon }) => (
          <button
            key={key}
            onClick={() => setActiveTab(key as any)}
            className={`flex items-center gap-2 px-4 py-2 rounded-md transition-colors ${
              activeTab === key
                ? 'bg-white text-blue-600 shadow-sm'
                : 'text-gray-600 hover:text-gray-800'
            }`}
          >
            <Icon className="w-4 h-4" />
            {label}
          </button>
        ))}
      </div>

      {/* Content */}
      <AnimatePresence mode="wait">
        <motion.div
          key={activeTab}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          transition={{ duration: 0.2 }}
        >
          {activeTab === 'overview' && renderOverview()}
          {activeTab === 'pipelines' && renderPipelines()}
          {activeTab === 'executions' && (
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-6">Deployment Executions</h2>
              {/* Execution history content would go here */}
            </div>
          )}
          {activeTab === 'environments' && (
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-6">Environment Management</h2>
              {/* Environment management content would go here */}
            </div>
          )}
          {activeTab === 'templates' && (
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-6">Deployment Templates</h2>
              {/* Template management content would go here */}
            </div>
          )}
        </motion.div>
      </AnimatePresence>

      {/* Running Jobs Indicator */}
      {activeExecutions.length > 0 && (
        <div className="fixed bottom-4 right-4 bg-white rounded-lg shadow-lg border border-gray-200 p-4 max-w-sm">
          <h4 className="font-medium text-gray-900 mb-2 flex items-center gap-2">
            <Activity className="w-4 h-4 text-blue-500 animate-pulse" />
            Active Deployments ({activeExecutions.length})
          </h4>
          <div className="space-y-2">
            {activeExecutions.map((execution) => (
              <div key={execution.id} className="text-sm">
                <div className="flex justify-between items-center">
                  <span className="text-gray-700">
                    {pipelines.find(p => p.id === execution.pipeline_id)?.name}
                  </span>
                  <span className="text-xs text-gray-500">{execution.environment}</span>
                </div>
                <div className="text-xs text-gray-500">
                  Step {execution.current_step + 1} of {execution.steps.length}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelDeploymentPipeline;