import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  RefreshCw, AlertTriangle, CheckCircle, Clock, Settings,
  TrendingUp, TrendingDown, Database, Brain, GitBranch,
  Play, Pause, BarChart3, Calendar, Target, Activity,
  Zap, AlertCircle, Info, Download, Upload
} from 'lucide-react';

interface RetrainingSchedule {
  id: string;
  model_id: string;
  model_name: string;
  schedule_type: 'time_based' | 'performance_based' | 'drift_based' | 'manual';
  schedule_config: {
    interval_hours?: number;
    performance_threshold?: number;
    drift_threshold?: number;
    min_samples?: number;
  };
  status: 'active' | 'paused' | 'disabled';
  last_retrain: string | null;
  next_retrain: string | null;
}

interface RetrainingJob {
  job_id: string;
  model_id: string;
  model_name: string;
  trigger_type: 'scheduled' | 'drift_detected' | 'performance_drop' | 'manual';
  trigger_reason: string;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress_percentage: number;
  started_at: string | null;
  completed_at: string | null;
  duration_minutes: number | null;
  metrics: {
    old_accuracy?: number;
    new_accuracy?: number;
    improvement?: number;
    drift_score?: number;
    samples_processed?: number;
  };
  error_message?: string;
}

interface DriftAlert {
  id: string;
  model_id: string;
  model_name: string;
  drift_type: 'feature_drift' | 'target_drift' | 'prediction_drift';
  drift_score: number;
  threshold: number;
  severity: 'low' | 'medium' | 'high' | 'critical';
  detected_at: string;
  features_affected: string[];
  auto_retrain_triggered: boolean;
}

interface PerformanceMetrics {
  model_id: string;
  current_accuracy: number;
  baseline_accuracy: number;
  performance_drop: number;
  trend: 'improving' | 'stable' | 'declining';
  samples_since_last_retrain: number;
  days_since_last_retrain: number;
}

const AutomatedModelRetraining: React.FC = () => {
  const [retrainingSchedules, setRetrainingSchedules] = useState<RetrainingSchedule[]>([]);
  const [activeJobs, setActiveJobs] = useState<RetrainingJob[]>([]);
  const [completedJobs, setCompletedJobs] = useState<RetrainingJob[]>([]);
  const [driftAlerts, setDriftAlerts] = useState<DriftAlert[]>([]);
  const [performanceMetrics, setPerformanceMetrics] = useState<PerformanceMetrics[]>([]);
  
  const [selectedModel, setSelectedModel] = useState<string>('all');
  const [activeTab, setActiveTab] = useState<'overview' | 'schedules' | 'jobs' | 'alerts' | 'config'>('overview');
  const [showCreateSchedule, setShowCreateSchedule] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  // Mock data
  useEffect(() => {
    const mockSchedules: RetrainingSchedule[] = [
      {
        id: 'sched_001',
        model_id: 'model_001',
        model_name: 'Customer Churn Predictor',
        schedule_type: 'drift_based',
        schedule_config: {
          drift_threshold: 0.1,
          min_samples: 1000
        },
        status: 'active',
        last_retrain: '2024-01-15T10:30:00Z',
        next_retrain: null
      },
      {
        id: 'sched_002',
        model_id: 'model_002',
        model_name: 'Fraud Detection System',
        schedule_type: 'time_based',
        schedule_config: {
          interval_hours: 168 // Weekly
        },
        status: 'active',
        last_retrain: '2024-01-14T02:00:00Z',
        next_retrain: '2024-01-21T02:00:00Z'
      },
      {
        id: 'sched_003',
        model_id: 'model_003',
        model_name: 'Price Optimization Model',
        schedule_type: 'performance_based',
        schedule_config: {
          performance_threshold: 0.85,
          min_samples: 500
        },
        status: 'paused',
        last_retrain: '2024-01-10T14:20:00Z',
        next_retrain: null
      }
    ];

    const mockActiveJobs: RetrainingJob[] = [
      {
        job_id: 'job_001',
        model_id: 'model_001',
        model_name: 'Customer Churn Predictor',
        trigger_type: 'drift_detected',
        trigger_reason: 'Feature drift detected in customer_age and purchase_frequency features',
        status: 'running',
        progress_percentage: 67,
        started_at: '2024-01-16T09:15:00Z',
        completed_at: null,
        duration_minutes: 45,
        metrics: {
          drift_score: 0.15,
          samples_processed: 15420
        }
      },
      {
        job_id: 'job_002',
        model_id: 'model_004',
        model_name: 'Inventory Forecaster',
        trigger_type: 'manual',
        trigger_reason: 'Manual retraining triggered by user',
        status: 'queued',
        progress_percentage: 0,
        started_at: null,
        completed_at: null,
        duration_minutes: null,
        metrics: {}
      }
    ];

    const mockCompletedJobs: RetrainingJob[] = [
      {
        job_id: 'job_completed_001',
        model_id: 'model_002',
        model_name: 'Fraud Detection System',
        trigger_type: 'scheduled',
        trigger_reason: 'Weekly scheduled retraining',
        status: 'completed',
        progress_percentage: 100,
        started_at: '2024-01-15T02:00:00Z',
        completed_at: '2024-01-15T03:42:00Z',
        duration_minutes: 102,
        metrics: {
          old_accuracy: 0.932,
          new_accuracy: 0.941,
          improvement: 0.009,
          samples_processed: 24891
        }
      },
      {
        job_id: 'job_completed_002',
        model_id: 'model_003',
        model_name: 'Price Optimization Model',
        trigger_type: 'performance_drop',
        trigger_reason: 'Performance dropped below 85% threshold',
        status: 'failed',
        progress_percentage: 23,
        started_at: '2024-01-14T16:30:00Z',
        completed_at: '2024-01-14T16:51:00Z',
        duration_minutes: 21,
        metrics: {
          old_accuracy: 0.834,
          samples_processed: 1247
        },
        error_message: 'Insufficient training data quality - data validation failed'
      }
    ];

    const mockDriftAlerts: DriftAlert[] = [
      {
        id: 'alert_001',
        model_id: 'model_001',
        model_name: 'Customer Churn Predictor',
        drift_type: 'feature_drift',
        drift_score: 0.15,
        threshold: 0.1,
        severity: 'high',
        detected_at: '2024-01-16T09:10:00Z',
        features_affected: ['customer_age', 'purchase_frequency', 'account_balance'],
        auto_retrain_triggered: true
      },
      {
        id: 'alert_002',
        model_id: 'model_005',
        model_name: 'Recommendation Engine',
        drift_type: 'prediction_drift',
        drift_score: 0.08,
        threshold: 0.05,
        severity: 'medium',
        detected_at: '2024-01-16T07:22:00Z',
        features_affected: ['user_engagement', 'content_preference'],
        auto_retrain_triggered: false
      }
    ];

    const mockPerformanceMetrics: PerformanceMetrics[] = [
      {
        model_id: 'model_001',
        current_accuracy: 0.892,
        baseline_accuracy: 0.915,
        performance_drop: -0.023,
        trend: 'declining',
        samples_since_last_retrain: 15420,
        days_since_last_retrain: 2
      },
      {
        model_id: 'model_002',
        current_accuracy: 0.941,
        baseline_accuracy: 0.932,
        performance_drop: 0.009,
        trend: 'improving',
        samples_since_last_retrain: 2341,
        days_since_last_retrain: 1
      },
      {
        model_id: 'model_003',
        current_accuracy: 0.834,
        baseline_accuracy: 0.887,
        performance_drop: -0.053,
        trend: 'declining',
        samples_since_last_retrain: 8934,
        days_since_last_retrain: 7
      }
    ];

    setRetrainingSchedules(mockSchedules);
    setActiveJobs(mockActiveJobs);
    setCompletedJobs(mockCompletedJobs);
    setDriftAlerts(mockDriftAlerts);
    setPerformanceMetrics(mockPerformanceMetrics);
    setIsLoading(false);
  }, []);

  const handleTriggerRetraining = useCallback(async (modelId: string) => {
    // Implement manual retraining trigger
    console.log('Triggering manual retraining for model:', modelId);
  }, []);

  const handlePauseResumeSchedule = useCallback(async (scheduleId: string, action: 'pause' | 'resume') => {
    setRetrainingSchedules(prev => prev.map(schedule =>
      schedule.id === scheduleId
        ? { ...schedule, status: action === 'pause' ? 'paused' : 'active' }
        : schedule
    ));
  }, []);

  const handleCancelJob = useCallback(async (jobId: string) => {
    setActiveJobs(prev => prev.map(job =>
      job.job_id === jobId
        ? { ...job, status: 'cancelled' as const }
        : job
    ));
  }, []);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': case 'running': case 'completed': return 'text-green-600';
      case 'paused': case 'queued': return 'text-yellow-600';
      case 'failed': case 'cancelled': case 'disabled': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'low': return 'text-blue-600 bg-blue-50';
      case 'medium': return 'text-yellow-600 bg-yellow-50';
      case 'high': return 'text-orange-600 bg-orange-50';
      case 'critical': return 'text-red-600 bg-red-50';
      default: return 'text-gray-600 bg-gray-50';
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
              <p className="text-sm text-gray-600">Active Schedules</p>
              <p className="text-3xl font-bold text-blue-600">
                {retrainingSchedules.filter(s => s.status === 'active').length}
              </p>
            </div>
            <RefreshCw className="w-8 h-8 text-blue-500" />
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
              <p className="text-sm text-gray-600">Running Jobs</p>
              <p className="text-3xl font-bold text-green-600">
                {activeJobs.filter(j => j.status === 'running').length}
              </p>
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
              <p className="text-sm text-gray-600">Drift Alerts</p>
              <p className="text-3xl font-bold text-orange-600">
                {driftAlerts.filter(a => a.severity !== 'low').length}
              </p>
            </div>
            <AlertTriangle className="w-8 h-8 text-orange-500" />
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
              <p className="text-sm text-gray-600">Success Rate</p>
              <p className="text-3xl font-bold text-purple-600">
                {Math.round((completedJobs.filter(j => j.status === 'completed').length / completedJobs.length) * 100)}%
              </p>
            </div>
            <Target className="w-8 h-8 text-purple-500" />
          </div>
        </motion.div>
      </div>

      {/* Active Jobs */}
      <div className="lg:col-span-2 bg-white rounded-lg shadow-sm border border-gray-200">
        <div className="p-6 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
            <Activity className="w-5 h-5" />
            Active Retraining Jobs
          </h3>
        </div>
        <div className="p-6">
          <div className="space-y-4">
            {activeJobs.map((job) => (
              <motion.div
                key={job.job_id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                className="border border-gray-200 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-3">
                    <Brain className="w-5 h-5 text-blue-500" />
                    <div>
                      <h4 className="font-medium text-gray-900">{job.model_name}</h4>
                      <p className="text-sm text-gray-600">{job.trigger_reason}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className={`text-sm font-medium ${getStatusColor(job.status)}`}>
                      {job.status.toUpperCase()}
                    </span>
                    {job.status === 'running' && (
                      <button
                        onClick={() => handleCancelJob(job.job_id)}
                        className="text-red-600 hover:text-red-800 text-sm"
                      >
                        Cancel
                      </button>
                    )}
                  </div>
                </div>
                
                {job.status === 'running' && (
                  <div className="mb-3">
                    <div className="flex justify-between text-sm text-gray-600 mb-1">
                      <span>Progress</span>
                      <span>{job.progress_percentage}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-blue-600 h-2 rounded-full transition-all duration-500"
                        style={{ width: `${job.progress_percentage}%` }}
                      />
                    </div>
                  </div>
                )}
                
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
                  <div>
                    <span className="text-gray-600">Trigger:</span>
                    <p className="font-medium">{job.trigger_type.replace('_', ' ')}</p>
                  </div>
                  {job.metrics.samples_processed && (
                    <div>
                      <span className="text-gray-600">Samples:</span>
                      <p className="font-medium">{job.metrics.samples_processed.toLocaleString()}</p>
                    </div>
                  )}
                  {job.duration_minutes && (
                    <div>
                      <span className="text-gray-600">Duration:</span>
                      <p className="font-medium">{job.duration_minutes}m</p>
                    </div>
                  )}
                </div>
              </motion.div>
            ))}
            
            {activeJobs.length === 0 && (
              <div className="text-center py-8">
                <Clock className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-500">No active retraining jobs</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Recent Drift Alerts */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200">
        <div className="p-6 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
            <AlertTriangle className="w-5 h-5" />
            Recent Drift Alerts
          </h3>
        </div>
        <div className="p-6">
          <div className="space-y-4">
            {driftAlerts.slice(0, 5).map((alert) => (
              <motion.div
                key={alert.id}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                className="border border-gray-200 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <span className={`text-xs font-medium px-2 py-1 rounded-full ${getSeverityColor(alert.severity)}`}>
                    {alert.severity.toUpperCase()}
                  </span>
                  <span className="text-xs text-gray-500">
                    {new Date(alert.detected_at).toLocaleString()}
                  </span>
                </div>
                
                <h4 className="font-medium text-gray-900 mb-1">{alert.model_name}</h4>
                <p className="text-sm text-gray-600 mb-2">
                  {alert.drift_type.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())} detected
                </p>
                
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">Drift Score: {alert.drift_score.toFixed(3)}</span>
                  {alert.auto_retrain_triggered && (
                    <span className="text-green-600 flex items-center gap-1">
                      <Zap className="w-3 h-3" />
                      Auto-retrain triggered
                    </span>
                  )}
                </div>
              </motion.div>
            ))}
            
            {driftAlerts.length === 0 && (
              <div className="text-center py-8">
                <CheckCircle className="w-12 h-12 text-green-400 mx-auto mb-4" />
                <p className="text-gray-500">No recent drift alerts</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Performance Metrics */}
      <div className="lg:col-span-3 bg-white rounded-lg shadow-sm border border-gray-200">
        <div className="p-6 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
            <BarChart3 className="w-5 h-5" />
            Model Performance Overview
          </h3>
        </div>
        <div className="p-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {performanceMetrics.map((metrics) => {
              const schedule = retrainingSchedules.find(s => s.model_id === metrics.model_id);
              return (
                <motion.div
                  key={metrics.model_id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="border border-gray-200 rounded-lg p-4"
                >
                  <div className="flex items-center justify-between mb-3">
                    <h4 className="font-medium text-gray-900">{schedule?.model_name || `Model ${metrics.model_id}`}</h4>
                    <div className="flex items-center gap-1">
                      {metrics.trend === 'improving' && <TrendingUp className="w-4 h-4 text-green-500" />}
                      {metrics.trend === 'declining' && <TrendingDown className="w-4 h-4 text-red-500" />}
                      {metrics.trend === 'stable' && <div className="w-4 h-1 bg-gray-400 rounded" />}
                    </div>
                  </div>
                  
                  <div className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Current Accuracy:</span>
                      <span className="text-sm font-medium">{(metrics.current_accuracy * 100).toFixed(1)}%</span>
                    </div>
                    
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Performance Change:</span>
                      <span className={`text-sm font-medium ${metrics.performance_drop >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {metrics.performance_drop >= 0 ? '+' : ''}{(metrics.performance_drop * 100).toFixed(1)}%
                      </span>
                    </div>
                    
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Days Since Retrain:</span>
                      <span className="text-sm font-medium">{metrics.days_since_last_retrain}</span>
                    </div>
                    
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">New Samples:</span>
                      <span className="text-sm font-medium">{metrics.samples_since_last_retrain.toLocaleString()}</span>
                    </div>
                  </div>
                  
                  <div className="mt-4 pt-3 border-t border-gray-200">
                    <button
                      onClick={() => handleTriggerRetraining(metrics.model_id)}
                      className="w-full bg-blue-600 text-white text-sm py-2 px-3 rounded-lg hover:bg-blue-700 transition-colors flex items-center justify-center gap-2"
                    >
                      <RefreshCw className="w-4 h-4" />
                      Trigger Retraining
                    </button>
                  </div>
                </motion.div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );

  const renderSchedules = () => (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-xl font-semibold text-gray-900">Retraining Schedules</h2>
        <button
          onClick={() => setShowCreateSchedule(true)}
          className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors flex items-center gap-2"
        >
          <Calendar className="w-4 h-4" />
          Create Schedule
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {retrainingSchedules.map((schedule) => (
          <motion.div
            key={schedule.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white rounded-lg shadow-sm border border-gray-200"
          >
            <div className="p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-medium text-gray-900">{schedule.model_name}</h3>
                <div className="flex items-center gap-2">
                  <span className={`text-sm font-medium ${getStatusColor(schedule.status)}`}>
                    {schedule.status.toUpperCase()}
                  </span>
                  <button
                    onClick={() => handlePauseResumeSchedule(
                      schedule.id,
                      schedule.status === 'active' ? 'pause' : 'resume'
                    )}
                    className="text-blue-600 hover:text-blue-800"
                  >
                    {schedule.status === 'active' ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                  </button>
                  <button className="text-gray-600 hover:text-gray-800">
                    <Settings className="w-4 h-4" />
                  </button>
                </div>
              </div>

              <div className="space-y-3">
                <div>
                  <span className="text-sm text-gray-600">Schedule Type:</span>
                  <p className="font-medium capitalize">{schedule.schedule_type.replace('_', ' ')}</p>
                </div>

                {schedule.schedule_type === 'time_based' && schedule.schedule_config.interval_hours && (
                  <div>
                    <span className="text-sm text-gray-600">Interval:</span>
                    <p className="font-medium">Every {schedule.schedule_config.interval_hours / 24} days</p>
                  </div>
                )}

                {schedule.schedule_type === 'performance_based' && schedule.schedule_config.performance_threshold && (
                  <div>
                    <span className="text-sm text-gray-600">Performance Threshold:</span>
                    <p className="font-medium">{(schedule.schedule_config.performance_threshold * 100).toFixed(1)}%</p>
                  </div>
                )}

                {schedule.schedule_type === 'drift_based' && schedule.schedule_config.drift_threshold && (
                  <div>
                    <span className="text-sm text-gray-600">Drift Threshold:</span>
                    <p className="font-medium">{schedule.schedule_config.drift_threshold.toFixed(3)}</p>
                  </div>
                )}

                <div>
                  <span className="text-sm text-gray-600">Last Retrain:</span>
                  <p className="font-medium">
                    {schedule.last_retrain ? new Date(schedule.last_retrain).toLocaleDateString() : 'Never'}
                  </p>
                </div>

                {schedule.next_retrain && (
                  <div>
                    <span className="text-sm text-gray-600">Next Retrain:</span>
                    <p className="font-medium">{new Date(schedule.next_retrain).toLocaleDateString()}</p>
                  </div>
                )}
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
          <RefreshCw className="w-8 h-8 text-blue-500" />
        </motion.div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto p-6">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          Automated Model Retraining
        </h1>
        <p className="text-gray-600">
          Monitor model performance, detect drift, and automatically retrain models to maintain optimal performance
        </p>
      </div>

      {/* Navigation Tabs */}
      <div className="flex space-x-1 mb-6 bg-gray-100 p-1 rounded-lg">
        {[
          { key: 'overview', label: 'Overview', icon: BarChart3 },
          { key: 'schedules', label: 'Schedules', icon: Calendar },
          { key: 'jobs', label: 'Jobs', icon: Activity },
          { key: 'alerts', label: 'Alerts', icon: AlertTriangle }
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
          {activeTab === 'schedules' && renderSchedules()}
          {activeTab === 'jobs' && (
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-6">Retraining Jobs History</h2>
              {/* Jobs content would go here */}
            </div>
          )}
          {activeTab === 'alerts' && (
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-6">Drift Detection Alerts</h2>
              {/* Alerts content would go here */}
            </div>
          )}
        </motion.div>
      </AnimatePresence>
    </div>
  );
};

export default AutomatedModelRetraining;