import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Activity, TrendingUp, TrendingDown, AlertTriangle, CheckCircle,
  Clock, Target, Zap, Bell, Settings, Eye, RefreshCw,
  BarChart3, LineChart, PieChart, Monitor, Server, Database,
  Cpu, MemoryStick, HardDrive, Wifi, Globe, Users,
  ArrowUp, ArrowDown, Minus, Play, Pause, Filter,
  Download, Share2, AlertCircle, Shield, Brain, Code
} from 'lucide-react';
import Card from '@/components/common/Card';
import Button from '@/components/common/Button';
import { apiClient } from '@/services/api';

interface ModelMetrics {
  model_id: string;
  model_name: string;
  version: string;
  status: 'healthy' | 'warning' | 'critical' | 'unknown';
  deployment_status: 'active' | 'inactive' | 'deploying' | 'failed';
  last_updated: Date;
  
  // Performance metrics
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  auc_score?: number;
  
  // Operational metrics
  request_rate: number; // requests per second
  response_time: number; // milliseconds
  error_rate: number; // percentage
  throughput: number; // predictions per second
  
  // System metrics
  cpu_usage: number;
  memory_usage: number;
  gpu_usage?: number;
  disk_usage: number;
  
  // Data quality metrics
  data_drift_score: number;
  feature_drift_count: number;
  outlier_rate: number;
  missing_data_rate: number;
  
  // Business metrics
  prediction_volume: number;
  user_feedback_score: number;
  business_impact_score: number;
}

interface Alert {
  id: string;
  model_id: string;
  model_name: string;
  type: 'performance' | 'system' | 'data_quality' | 'business';
  severity: 'low' | 'medium' | 'high' | 'critical';
  title: string;
  description: string;
  metric: string;
  threshold: number;
  current_value: number;
  triggered_at: Date;
  acknowledged: boolean;
  acknowledged_by?: string;
  resolved: boolean;
  resolved_at?: Date;
}

interface ThresholdConfig {
  metric: string;
  warning_threshold: number;
  critical_threshold: number;
  comparison: 'greater_than' | 'less_than' | 'deviation';
  enabled: boolean;
}

interface PerformanceTimeSeries {
  timestamp: Date;
  model_id: string;
  metrics: Record<string, number>;
}

const RealTimePerformanceMonitoring: React.FC = () => {
  const [models, setModels] = useState<ModelMetrics[]>([]);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [selectedModel, setSelectedModel] = useState<ModelMetrics | null>(null);
  const [timeSeriesData, setTimeSeriesData] = useState<PerformanceTimeSeries[]>([]);
  const [isMonitoring, setIsMonitoring] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState(30); // seconds
  const [showAlertsPanel, setShowAlertsPanel] = useState(false);
  const [showThresholds, setShowThresholds] = useState(false);
  const [filterStatus, setFilterStatus] = useState<string>('all');
  const [timeRange, setTimeRange] = useState<string>('1h');
  
  const monitoringIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // Mock data
  const mockModels: ModelMetrics[] = [
    {
      model_id: 'model-001',
      model_name: 'Credit Risk Assessment',
      version: 'v2.1.0',
      status: 'healthy',
      deployment_status: 'active',
      last_updated: new Date(),
      accuracy: 0.924,
      precision: 0.891,
      recall: 0.876,
      f1_score: 0.883,
      auc_score: 0.952,
      request_rate: 145.7,
      response_time: 23.4,
      error_rate: 0.12,
      throughput: 142.3,
      cpu_usage: 34.2,
      memory_usage: 67.8,
      gpu_usage: 0,
      disk_usage: 23.1,
      data_drift_score: 0.03,
      feature_drift_count: 1,
      outlier_rate: 0.02,
      missing_data_rate: 0.001,
      prediction_volume: 12450,
      user_feedback_score: 4.7,
      business_impact_score: 8.9
    },
    {
      model_id: 'model-002',
      model_name: 'Fraud Detection',
      version: 'v1.8.3',
      status: 'warning',
      deployment_status: 'active',
      last_updated: new Date(),
      accuracy: 0.889,
      precision: 0.856,
      recall: 0.901,
      f1_score: 0.878,
      auc_score: 0.923,
      request_rate: 67.3,
      response_time: 156.7,
      error_rate: 0.45,
      throughput: 65.8,
      cpu_usage: 78.9,
      memory_usage: 89.4,
      gpu_usage: 45.6,
      disk_usage: 56.7,
      data_drift_score: 0.12,
      feature_drift_count: 4,
      outlier_rate: 0.08,
      missing_data_rate: 0.003,
      prediction_volume: 8900,
      user_feedback_score: 4.2,
      business_impact_score: 7.8
    },
    {
      model_id: 'model-003',
      model_name: 'Customer Segmentation',
      version: 'v3.0.1',
      status: 'critical',
      deployment_status: 'active',
      last_updated: new Date(),
      accuracy: 0.712,
      precision: 0.684,
      recall: 0.748,
      f1_score: 0.714,
      request_rate: 23.1,
      response_time: 340.2,
      error_rate: 2.8,
      throughput: 19.7,
      cpu_usage: 92.3,
      memory_usage: 94.1,
      disk_usage: 78.9,
      data_drift_score: 0.28,
      feature_drift_count: 12,
      outlier_rate: 0.15,
      missing_data_rate: 0.012,
      prediction_volume: 3200,
      user_feedback_score: 2.8,
      business_impact_score: 4.1
    }
  ];

  const mockAlerts: Alert[] = [
    {
      id: 'alert-1',
      model_id: 'model-003',
      model_name: 'Customer Segmentation',
      type: 'performance',
      severity: 'critical',
      title: 'Model Accuracy Degradation',
      description: 'Model accuracy has dropped below critical threshold of 75%',
      metric: 'accuracy',
      threshold: 0.75,
      current_value: 0.712,
      triggered_at: new Date(Date.now() - 15 * 60 * 1000),
      acknowledged: false,
      resolved: false
    },
    {
      id: 'alert-2',
      model_id: 'model-003',
      model_name: 'Customer Segmentation',
      type: 'data_quality',
      severity: 'high',
      title: 'Significant Data Drift Detected',
      description: 'Data drift score exceeds warning threshold, indicating distribution changes',
      metric: 'data_drift_score',
      threshold: 0.2,
      current_value: 0.28,
      triggered_at: new Date(Date.now() - 45 * 60 * 1000),
      acknowledged: false,
      resolved: false
    },
    {
      id: 'alert-3',
      model_id: 'model-002',
      model_name: 'Fraud Detection',
      type: 'system',
      severity: 'medium',
      title: 'High Memory Usage',
      description: 'Memory usage is approaching capacity limits',
      metric: 'memory_usage',
      threshold: 85,
      current_value: 89.4,
      triggered_at: new Date(Date.now() - 2 * 60 * 60 * 1000),
      acknowledged: true,
      acknowledged_by: 'ops@company.com',
      resolved: false
    }
  ];

  useEffect(() => {
    loadInitialData();
    startMonitoring();
    
    return () => {
      stopMonitoring();
    };
  }, []);

  useEffect(() => {
    if (isMonitoring) {
      startMonitoring();
    } else {
      stopMonitoring();
    }
  }, [isMonitoring, refreshInterval]);

  const loadInitialData = async () => {
    try {
      // In production, this would call the API
      setModels(mockModels);
      setAlerts(mockAlerts);
      generateTimeSeriesData();
    } catch (error) {
      console.error('Error loading monitoring data:', error);
      setModels(mockModels);
      setAlerts(mockAlerts);
      generateTimeSeriesData();
    }
  };

  const generateTimeSeriesData = () => {
    const now = new Date();
    const data: PerformanceTimeSeries[] = [];
    
    // Generate last 1 hour of data at 1-minute intervals
    for (let i = 60; i >= 0; i--) {
      const timestamp = new Date(now.getTime() - i * 60 * 1000);
      
      mockModels.forEach(model => {
        // Add some realistic variation
        const variation = (Math.random() - 0.5) * 0.1;
        
        data.push({
          timestamp,
          model_id: model.model_id,
          metrics: {
            accuracy: Math.max(0, Math.min(1, model.accuracy + variation * 0.5)),
            response_time: Math.max(0, model.response_time + variation * 50),
            error_rate: Math.max(0, model.error_rate + variation * 0.5),
            cpu_usage: Math.max(0, Math.min(100, model.cpu_usage + variation * 20)),
            memory_usage: Math.max(0, Math.min(100, model.memory_usage + variation * 15)),
            data_drift_score: Math.max(0, model.data_drift_score + variation * 0.1)
          }
        });
      });
    }
    
    setTimeSeriesData(data);
  };

  const startMonitoring = () => {
    stopMonitoring(); // Clear existing interval
    
    monitoringIntervalRef.current = setInterval(() => {
      updateMetrics();
    }, refreshInterval * 1000);
    
    // In production, would establish WebSocket connection
    // connectWebSocket();
  };

  const stopMonitoring = () => {
    if (monitoringIntervalRef.current) {
      clearInterval(monitoringIntervalRef.current);
      monitoringIntervalRef.current = null;
    }
    
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  };

  const updateMetrics = async () => {
    try {
      // Simulate real-time updates with slight variations
      setModels(prevModels => 
        prevModels.map(model => ({
          ...model,
          last_updated: new Date(),
          // Add realistic variations
          accuracy: Math.max(0, Math.min(1, model.accuracy + (Math.random() - 0.5) * 0.01)),
          response_time: Math.max(0, model.response_time + (Math.random() - 0.5) * 5),
          error_rate: Math.max(0, model.error_rate + (Math.random() - 0.5) * 0.1),
          request_rate: Math.max(0, model.request_rate + (Math.random() - 0.5) * 10),
          cpu_usage: Math.max(0, Math.min(100, model.cpu_usage + (Math.random() - 0.5) * 5)),
          memory_usage: Math.max(0, Math.min(100, model.memory_usage + (Math.random() - 0.5) * 3))
        }))
      );
      
      // Update time series data
      const now = new Date();
      setTimeSeriesData(prevData => {
        const newData = [...prevData];
        
        models.forEach(model => {
          newData.push({
            timestamp: now,
            model_id: model.model_id,
            metrics: {
              accuracy: model.accuracy,
              response_time: model.response_time,
              error_rate: model.error_rate,
              cpu_usage: model.cpu_usage,
              memory_usage: model.memory_usage,
              data_drift_score: model.data_drift_score
            }
          });
        });
        
        // Keep only last hour of data
        const oneHourAgo = now.getTime() - 60 * 60 * 1000;
        return newData.filter(d => d.timestamp.getTime() > oneHourAgo);
      });
      
    } catch (error) {
      console.error('Error updating metrics:', error);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
        return 'text-green-600 dark:text-green-400';
      case 'warning':
        return 'text-yellow-600 dark:text-yellow-400';
      case 'critical':
        return 'text-red-600 dark:text-red-400';
      case 'unknown':
        return 'text-gray-600 dark:text-gray-400';
      default:
        return 'text-gray-600 dark:text-gray-400';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'warning':
        return <AlertTriangle className="w-5 h-5 text-yellow-500" />;
      case 'critical':
        return <AlertCircle className="w-5 h-5 text-red-500" />;
      case 'unknown':
        return <Monitor className="w-5 h-5 text-gray-500" />;
      default:
        return <Monitor className="w-5 h-5 text-gray-500" />;
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return 'bg-red-500 text-white';
      case 'high':
        return 'bg-orange-500 text-white';
      case 'medium':
        return 'bg-yellow-500 text-white';
      case 'low':
        return 'bg-blue-500 text-white';
      default:
        return 'bg-gray-500 text-white';
    }
  };

  const formatMetric = (value: number, type: 'percentage' | 'decimal' | 'number' | 'time') => {
    switch (type) {
      case 'percentage':
        return `${(value * 100).toFixed(1)}%`;
      case 'decimal':
        return value.toFixed(3);
      case 'time':
        return `${value.toFixed(1)}ms`;
      default:
        return value.toFixed(1);
    }
  };

  const getMetricTrend = (modelId: string, metric: string): 'up' | 'down' | 'stable' => {
    const modelData = timeSeriesData.filter(d => d.model_id === modelId);
    if (modelData.length < 2) return 'stable';
    
    const recent = modelData.slice(-5);
    const older = modelData.slice(-10, -5);
    
    if (recent.length === 0 || older.length === 0) return 'stable';
    
    const recentAvg = recent.reduce((sum, d) => sum + (d.metrics[metric] || 0), 0) / recent.length;
    const olderAvg = older.reduce((sum, d) => sum + (d.metrics[metric] || 0), 0) / older.length;
    
    const change = (recentAvg - olderAvg) / olderAvg;
    
    if (Math.abs(change) < 0.05) return 'stable';
    return change > 0 ? 'up' : 'down';
  };

  const acknowledgeAlert = async (alertId: string) => {
    setAlerts(prev => 
      prev.map(alert => 
        alert.id === alertId 
          ? { ...alert, acknowledged: true, acknowledged_by: 'current-user@company.com' }
          : alert
      )
    );
  };

  const resolveAlert = async (alertId: string) => {
    setAlerts(prev => 
      prev.map(alert => 
        alert.id === alertId 
          ? { ...alert, resolved: true, resolved_at: new Date() }
          : alert
      )
    );
  };

  const filteredModels = models.filter(model => {
    if (filterStatus === 'all') return true;
    return model.status === filterStatus;
  });

  const activeAlerts = alerts.filter(alert => !alert.resolved);
  const criticalAlerts = activeAlerts.filter(alert => alert.severity === 'critical');

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-neutral-900 dark:text-neutral-100 flex items-center">
            <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-lg flex items-center justify-center mr-3">
              <Activity className="w-5 h-5 text-white" />
            </div>
            Real-time Performance Monitoring
          </h1>
          <p className="text-neutral-600 dark:text-neutral-400 mt-1">
            Live monitoring of model performance, system health, and data quality
          </p>
        </div>
        
        <div className="flex items-center space-x-2">
          <div className="flex items-center space-x-2 px-3 py-1 bg-neutral-100 dark:bg-neutral-800 rounded-lg">
            <div className={`w-2 h-2 rounded-full ${isMonitoring ? 'bg-green-500 animate-pulse' : 'bg-gray-400'}`} />
            <span className="text-sm text-neutral-600 dark:text-neutral-400">
              {isMonitoring ? 'Live' : 'Paused'}
            </span>
          </div>
          
          <Button
            variant="outline"
            size="sm"
            leftIcon={activeAlerts.length > 0 ? <Bell className="w-4 h-4" /> : <Bell className="w-4 h-4" />}
            onClick={() => setShowAlertsPanel(true)}
            className={activeAlerts.length > 0 ? 'text-red-600 border-red-300' : ''}
          >
            Alerts ({activeAlerts.length})
          </Button>
          
          <Button
            variant="outline"
            size="sm"
            leftIcon={<Settings className="w-4 h-4" />}
            onClick={() => setShowThresholds(true)}
          >
            Thresholds
          </Button>
          
          <Button
            variant={isMonitoring ? "outline" : "primary"}
            size="sm"
            leftIcon={isMonitoring ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            onClick={() => setIsMonitoring(!isMonitoring)}
          >
            {isMonitoring ? 'Pause' : 'Resume'}
          </Button>
        </div>
      </div>

      {/* Critical Alerts Banner */}
      {criticalAlerts.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4"
        >
          <div className="flex items-start space-x-3">
            <AlertCircle className="w-6 h-6 text-red-500 flex-shrink-0 mt-0.5" />
            <div className="flex-1">
              <h3 className="text-red-800 dark:text-red-200 font-semibold">
                {criticalAlerts.length} Critical Alert{criticalAlerts.length !== 1 ? 's' : ''} Require Immediate Attention
              </h3>
              <div className="mt-2 space-y-1">
                {criticalAlerts.slice(0, 2).map(alert => (
                  <div key={alert.id} className="text-sm text-red-700 dark:text-red-300">
                    <span className="font-medium">{alert.model_name}:</span> {alert.title}
                  </div>
                ))}
                {criticalAlerts.length > 2 && (
                  <div className="text-sm text-red-600 dark:text-red-400">
                    And {criticalAlerts.length - 2} more critical alerts...
                  </div>
                )}
              </div>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowAlertsPanel(true)}
              className="border-red-300 text-red-700 hover:bg-red-100"
            >
              View All
            </Button>
          </div>
        </motion.div>
      )}

      {/* Filters */}
      <Card>
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-1 bg-neutral-100 dark:bg-neutral-800 rounded-lg p-1">
              {['all', 'healthy', 'warning', 'critical'].map((status) => (
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
            
            <div className="text-sm text-neutral-600 dark:text-neutral-400">
              Showing {filteredModels.length} of {models.length} models
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            <span className="text-sm text-neutral-600 dark:text-neutral-400">Refresh:</span>
            <select
              value={refreshInterval}
              onChange={(e) => setRefreshInterval(Number(e.target.value))}
              className="text-sm border-0 bg-transparent text-neutral-900 dark:text-neutral-100 focus:outline-none"
            >
              <option value={10}>10s</option>
              <option value={30}>30s</option>
              <option value={60}>1m</option>
              <option value={300}>5m</option>
            </select>
          </div>
        </div>
      </Card>

      {/* Model Performance Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
        <AnimatePresence>
          {filteredModels.map((model, index) => (
            <motion.div
              key={model.model_id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ delay: index * 0.1 }}
              onClick={() => setSelectedModel(model)}
              className="cursor-pointer"
            >
              <Card className={`hover:shadow-lg transition-all duration-200 ${
                selectedModel?.model_id === model.model_id ? 'ring-2 ring-primary-500' : ''
              }`}>
                <div className="space-y-4">
                  {/* Model Header */}
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center space-x-2 mb-1">
                        <h3 className="font-semibold text-neutral-900 dark:text-neutral-100">
                          {model.model_name}
                        </h3>
                        {getStatusIcon(model.status)}
                      </div>
                      <div className="flex items-center space-x-3 text-sm text-neutral-600 dark:text-neutral-400">
                        <span>{model.version}</span>
                        <span>•</span>
                        <span className={getStatusColor(model.status)}>
                          {model.status}
                        </span>
                      </div>
                    </div>
                    
                    <div className="text-right">
                      <div className="text-xs text-neutral-500 dark:text-neutral-400">
                        Updated {new Date(model.last_updated).toLocaleTimeString()}
                      </div>
                    </div>
                  </div>
                  
                  {/* Key Metrics */}
                  <div className="grid grid-cols-2 gap-3">
                    <div className="p-3 bg-neutral-50 dark:bg-neutral-800 rounded-lg">
                      <div className="flex items-center justify-between">
                        <div>
                          <div className="text-sm font-medium text-neutral-700 dark:text-neutral-300">Accuracy</div>
                          <div className="text-lg font-bold text-neutral-900 dark:text-neutral-100">
                            {formatMetric(model.accuracy, 'percentage')}
                          </div>
                        </div>
                        <div className="flex items-center">
                          {getMetricTrend(model.model_id, 'accuracy') === 'up' && <TrendingUp className="w-4 h-4 text-green-500" />}
                          {getMetricTrend(model.model_id, 'accuracy') === 'down' && <TrendingDown className="w-4 h-4 text-red-500" />}
                          {getMetricTrend(model.model_id, 'accuracy') === 'stable' && <Minus className="w-4 h-4 text-gray-500" />}
                        </div>
                      </div>
                    </div>
                    
                    <div className="p-3 bg-neutral-50 dark:bg-neutral-800 rounded-lg">
                      <div className="flex items-center justify-between">
                        <div>
                          <div className="text-sm font-medium text-neutral-700 dark:text-neutral-300">Response Time</div>
                          <div className="text-lg font-bold text-neutral-900 dark:text-neutral-100">
                            {formatMetric(model.response_time, 'time')}
                          </div>
                        </div>
                        <div className="flex items-center">
                          {getMetricTrend(model.model_id, 'response_time') === 'up' && <TrendingUp className="w-4 h-4 text-red-500" />}
                          {getMetricTrend(model.model_id, 'response_time') === 'down' && <TrendingDown className="w-4 h-4 text-green-500" />}
                          {getMetricTrend(model.model_id, 'response_time') === 'stable' && <Minus className="w-4 h-4 text-gray-500" />}
                        </div>
                      </div>
                    </div>
                    
                    <div className="p-3 bg-neutral-50 dark:bg-neutral-800 rounded-lg">
                      <div className="flex items-center justify-between">
                        <div>
                          <div className="text-sm font-medium text-neutral-700 dark:text-neutral-300">Error Rate</div>
                          <div className="text-lg font-bold text-neutral-900 dark:text-neutral-100">
                            {formatMetric(model.error_rate, 'percentage')}
                          </div>
                        </div>
                        <div className="flex items-center">
                          {getMetricTrend(model.model_id, 'error_rate') === 'up' && <TrendingUp className="w-4 h-4 text-red-500" />}
                          {getMetricTrend(model.model_id, 'error_rate') === 'down' && <TrendingDown className="w-4 h-4 text-green-500" />}
                          {getMetricTrend(model.model_id, 'error_rate') === 'stable' && <Minus className="w-4 h-4 text-gray-500" />}
                        </div>
                      </div>
                    </div>
                    
                    <div className="p-3 bg-neutral-50 dark:bg-neutral-800 rounded-lg">
                      <div className="flex items-center justify-between">
                        <div>
                          <div className="text-sm font-medium text-neutral-700 dark:text-neutral-300">Data Drift</div>
                          <div className="text-lg font-bold text-neutral-900 dark:text-neutral-100">
                            {formatMetric(model.data_drift_score, 'decimal')}
                          </div>
                        </div>
                        <div className="flex items-center">
                          {model.data_drift_score > 0.2 ? <AlertTriangle className="w-4 h-4 text-red-500" /> :
                           model.data_drift_score > 0.1 ? <AlertTriangle className="w-4 h-4 text-yellow-500" /> :
                           <CheckCircle className="w-4 h-4 text-green-500" />}
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  {/* System Resources */}
                  <div className="space-y-2">
                    <h4 className="text-sm font-medium text-neutral-700 dark:text-neutral-300">System Resources</h4>
                    <div className="space-y-1">
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-neutral-600 dark:text-neutral-400">CPU</span>
                        <span className="font-medium text-neutral-900 dark:text-neutral-100">{model.cpu_usage.toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-neutral-200 dark:bg-neutral-700 rounded-full h-1.5">
                        <motion.div
                          className={`h-1.5 rounded-full ${
                            model.cpu_usage > 80 ? 'bg-red-500' :
                            model.cpu_usage > 60 ? 'bg-yellow-500' : 'bg-green-500'
                          }`}
                          initial={{ width: 0 }}
                          animate={{ width: `${model.cpu_usage}%` }}
                          transition={{ duration: 0.5 }}
                        />
                      </div>
                      
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-neutral-600 dark:text-neutral-400">Memory</span>
                        <span className="font-medium text-neutral-900 dark:text-neutral-100">{model.memory_usage.toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-neutral-200 dark:bg-neutral-700 rounded-full h-1.5">
                        <motion.div
                          className={`h-1.5 rounded-full ${
                            model.memory_usage > 80 ? 'bg-red-500' :
                            model.memory_usage > 60 ? 'bg-yellow-500' : 'bg-green-500'
                          }`}
                          initial={{ width: 0 }}
                          animate={{ width: `${model.memory_usage}%` }}
                          transition={{ duration: 0.5 }}
                        />
                      </div>
                    </div>
                  </div>
                  
                  {/* Quick Actions */}
                  <div className="flex items-center space-x-2 pt-3 border-t border-neutral-200 dark:border-neutral-700">
                    <Button variant="outline" size="sm" leftIcon={<Eye className="w-4 h-4" />}>
                      Details
                    </Button>
                    <Button variant="outline" size="sm" leftIcon={<BarChart3 className="w-4 h-4" />}>
                      Metrics
                    </Button>
                    {model.status === 'critical' && (
                      <Button variant="primary" size="sm" leftIcon={<RefreshCw className="w-4 h-4" />}>
                        Restart
                      </Button>
                    )}
                  </div>
                </div>
              </Card>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>

      {/* Detailed Model View */}
      {selectedModel && (
        <Card title={`${selectedModel.model_name} - Detailed Metrics`}>
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
            {/* Performance Metrics */}
            <div className="space-y-4">
              <h4 className="font-medium text-neutral-900 dark:text-neutral-100">Performance</h4>
              <div className="space-y-3">
                {[
                  { label: 'Accuracy', value: selectedModel.accuracy, type: 'percentage' as const },
                  { label: 'Precision', value: selectedModel.precision, type: 'percentage' as const },
                  { label: 'Recall', value: selectedModel.recall, type: 'percentage' as const },
                  { label: 'F1 Score', value: selectedModel.f1_score, type: 'percentage' as const }
                ].map(metric => (
                  <div key={metric.label} className="flex items-center justify-between">
                    <span className="text-sm text-neutral-600 dark:text-neutral-400">{metric.label}</span>
                    <span className="font-medium text-neutral-900 dark:text-neutral-100">
                      {formatMetric(metric.value, metric.type)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
            
            {/* Operational Metrics */}
            <div className="space-y-4">
              <h4 className="font-medium text-neutral-900 dark:text-neutral-100">Operations</h4>
              <div className="space-y-3">
                {[
                  { label: 'Request Rate', value: selectedModel.request_rate, type: 'number' as const, unit: '/sec' },
                  { label: 'Response Time', value: selectedModel.response_time, type: 'time' as const },
                  { label: 'Error Rate', value: selectedModel.error_rate, type: 'percentage' as const },
                  { label: 'Throughput', value: selectedModel.throughput, type: 'number' as const, unit: '/sec' }
                ].map(metric => (
                  <div key={metric.label} className="flex items-center justify-between">
                    <span className="text-sm text-neutral-600 dark:text-neutral-400">{metric.label}</span>
                    <span className="font-medium text-neutral-900 dark:text-neutral-100">
                      {formatMetric(metric.value, metric.type)}{metric.unit || ''}
                    </span>
                  </div>
                ))}
              </div>
            </div>
            
            {/* System Metrics */}
            <div className="space-y-4">
              <h4 className="font-medium text-neutral-900 dark:text-neutral-100">System</h4>
              <div className="space-y-3">
                {[
                  { label: 'CPU Usage', value: selectedModel.cpu_usage, type: 'percentage' as const },
                  { label: 'Memory Usage', value: selectedModel.memory_usage, type: 'percentage' as const },
                  { label: 'Disk Usage', value: selectedModel.disk_usage, type: 'percentage' as const },
                  ...(selectedModel.gpu_usage ? [{ label: 'GPU Usage', value: selectedModel.gpu_usage, type: 'percentage' as const }] : [])
                ].map(metric => (
                  <div key={metric.label} className="flex items-center justify-between">
                    <span className="text-sm text-neutral-600 dark:text-neutral-400">{metric.label}</span>
                    <span className="font-medium text-neutral-900 dark:text-neutral-100">
                      {formatMetric(metric.value / 100, metric.type)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
            
            {/* Data Quality Metrics */}
            <div className="space-y-4">
              <h4 className="font-medium text-neutral-900 dark:text-neutral-100">Data Quality</h4>
              <div className="space-y-3">
                {[
                  { label: 'Data Drift Score', value: selectedModel.data_drift_score, type: 'decimal' as const },
                  { label: 'Feature Drift Count', value: selectedModel.feature_drift_count, type: 'number' as const },
                  { label: 'Outlier Rate', value: selectedModel.outlier_rate, type: 'percentage' as const },
                  { label: 'Missing Data Rate', value: selectedModel.missing_data_rate, type: 'percentage' as const }
                ].map(metric => (
                  <div key={metric.label} className="flex items-center justify-between">
                    <span className="text-sm text-neutral-600 dark:text-neutral-400">{metric.label}</span>
                    <span className="font-medium text-neutral-900 dark:text-neutral-100">
                      {formatMetric(metric.value, metric.type)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </Card>
      )}

      {/* Alerts Panel */}
      <AnimatePresence>
        {showAlertsPanel && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
            onClick={() => setShowAlertsPanel(false)}
          >
            <motion.div
              initial={{ scale: 0.9 }}
              animate={{ scale: 1 }}
              exit={{ scale: 0.9 }}
              className="bg-white dark:bg-neutral-800 rounded-lg p-6 w-full max-w-4xl mx-4 max-h-[80vh] overflow-y-auto"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
                  Active Alerts ({activeAlerts.length})
                </h3>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowAlertsPanel(false)}
                >
                  ×
                </Button>
              </div>
              
              <div className="space-y-4">
                {activeAlerts.map((alert) => (
                  <div
                    key={alert.id}
                    className={`p-4 border rounded-lg ${
                      alert.severity === 'critical' ? 'border-red-300 bg-red-50 dark:bg-red-900/20' :
                      alert.severity === 'high' ? 'border-orange-300 bg-orange-50 dark:bg-orange-900/20' :
                      alert.severity === 'medium' ? 'border-yellow-300 bg-yellow-50 dark:bg-yellow-900/20' :
                      'border-blue-300 bg-blue-50 dark:bg-blue-900/20'
                    }`}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center space-x-3 mb-2">
                          <span className={`px-2 py-1 text-xs rounded-full font-medium ${getSeverityColor(alert.severity)}`}>
                            {alert.severity}
                          </span>
                          <span className="text-sm font-medium text-neutral-900 dark:text-neutral-100">
                            {alert.model_name}
                          </span>
                        </div>
                        
                        <h4 className="font-semibold text-neutral-900 dark:text-neutral-100 mb-1">
                          {alert.title}
                        </h4>
                        <p className="text-sm text-neutral-600 dark:text-neutral-400 mb-2">
                          {alert.description}
                        </p>
                        
                        <div className="flex items-center space-x-4 text-xs text-neutral-500 dark:text-neutral-400">
                          <span>Threshold: {alert.threshold}</span>
                          <span>Current: {alert.current_value}</span>
                          <span>{new Date(alert.triggered_at).toLocaleString()}</span>
                        </div>
                      </div>
                      
                      <div className="flex items-center space-x-2">
                        {!alert.acknowledged && (
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => acknowledgeAlert(alert.id)}
                          >
                            Acknowledge
                          </Button>
                        )}
                        <Button
                          variant="primary"
                          size="sm"
                          onClick={() => resolveAlert(alert.id)}
                        >
                          Resolve
                        </Button>
                      </div>
                    </div>
                  </div>
                ))}
                
                {activeAlerts.length === 0 && (
                  <div className="text-center py-12">
                    <CheckCircle className="w-12 h-12 text-green-500 mx-auto mb-4" />
                    <p className="text-neutral-600 dark:text-neutral-400">
                      No active alerts. All systems are operating normally.
                    </p>
                  </div>
                )}
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default RealTimePerformanceMonitoring;