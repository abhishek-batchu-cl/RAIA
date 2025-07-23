import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  Shield,
  CheckCircle,
  Clock,
  Target,
  Activity,
  BarChart3,
  RefreshCw,
  Settings,
  Bell,
  Download,
  Calendar,
  Users,
  DollarSign,
  Zap,
  Brain,
  Eye,
  AlertCircle,
  XCircle
} from 'lucide-react';
import Card from '../components/common/Card';
import MetricCard from '../components/common/MetricCard';
import Button from '../components/common/Button';

interface ModelDriftData {
  modelId: string;
  modelName: string;
  driftDetected: boolean;
  driftSeverity: 'none' | 'low' | 'medium' | 'high' | 'critical';
  driftConfidence: number;
  businessImpactScore: number;
  retrainingUrgencyScore: number;
  detectedAt: string;
  performanceDegradation: Record<string, number>;
  recommendations: string[];
  primaryCauses: string[];
  estimatedCostImpact: number;
  affectedPredictions: number;
}

interface BaselineData {
  modelId: string;
  baselineMetrics: Record<string, number>;
  currentMetrics: Record<string, number>;
  performanceDelta: Record<string, number>;
  confidenceIntervals: Record<string, [number, number]>;
}

const ModelDriftMonitoringDashboard: React.FC = () => {
  const [selectedModel, setSelectedModel] = useState<string>('all');
  const [selectedPeriod, setSelectedPeriod] = useState<string>('7d');
  const [activeTab, setActiveTab] = useState<'overview' | 'detection' | 'impact' | 'recommendations' | 'trends'>('overview');

  // Mock data - would come from API
  const [driftData] = useState<ModelDriftData[]>([
    {
      modelId: 'model_1',
      modelName: 'Credit Risk Model v2.3',
      driftDetected: true,
      driftSeverity: 'high',
      driftConfidence: 0.87,
      businessImpactScore: 0.72,
      retrainingUrgencyScore: 0.89,
      detectedAt: '2024-01-20T14:30:00Z',
      performanceDegradation: {
        accuracy: 0.15,
        precision: 0.12,
        recall: 0.18,
        f1_score: 0.14
      },
      recommendations: [
        'Schedule model retraining within 1-2 weeks',
        'Increase data collection for affected segments',
        'Review feature engineering pipeline',
        'Consider ensemble approaches'
      ],
      primaryCauses: [
        'Data distribution drift detected',
        'Statistically significant changes in: accuracy, precision, recall',
        'Severe performance degradation'
      ],
      estimatedCostImpact: 25400,
      affectedPredictions: 45600
    },
    {
      modelId: 'model_2',
      modelName: 'Fraud Detection Model v1.8',
      driftDetected: true,
      driftSeverity: 'medium',
      driftConfidence: 0.64,
      businessImpactScore: 0.45,
      retrainingUrgencyScore: 0.52,
      detectedAt: '2024-01-20T09:15:00Z',
      performanceDegradation: {
        accuracy: 0.08,
        precision: 0.09,
        recall: 0.07,
        f1_score: 0.08
      },
      recommendations: [
        'Plan model retraining within 1 month',
        'Investigate data quality issues',
        'Consider ensemble approaches'
      ],
      primaryCauses: [
        'Minor performance variation within normal range',
        'Seasonal pattern detected'
      ],
      estimatedCostImpact: 8900,
      affectedPredictions: 23400
    },
    {
      modelId: 'model_3',
      modelName: 'Customer Churn Predictor v3.1',
      driftDetected: false,
      driftSeverity: 'none',
      driftConfidence: 0.12,
      businessImpactScore: 0.08,
      retrainingUrgencyScore: 0.15,
      detectedAt: '2024-01-20T16:45:00Z',
      performanceDegradation: {
        accuracy: 0.02,
        precision: 0.01,
        recall: 0.03,
        f1_score: 0.02
      },
      recommendations: [
        'Continue monitoring, retraining not urgent',
        'Analyze root causes for preventive measures'
      ],
      primaryCauses: [
        'Minor performance variation within normal range'
      ],
      estimatedCostImpact: 450,
      affectedPredictions: 12800
    }
  ]);

  const [baselineData] = useState<BaselineData[]>([
    {
      modelId: 'model_1',
      baselineMetrics: {
        accuracy: 0.892,
        precision: 0.887,
        recall: 0.894,
        f1_score: 0.890
      },
      currentMetrics: {
        accuracy: 0.758,
        precision: 0.781,
        recall: 0.732,
        f1_score: 0.765
      },
      performanceDelta: {
        accuracy: -0.134,
        precision: -0.106,
        recall: -0.162,
        f1_score: -0.125
      },
      confidenceIntervals: {
        accuracy: [0.745, 0.771],
        precision: [0.769, 0.793],
        recall: [0.718, 0.746],
        f1_score: [0.752, 0.778]
      }
    }
  ]);

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'red';
      case 'high': return 'red';
      case 'medium': return 'amber';
      case 'low': return 'yellow';
      default: return 'green';
    }
  };

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'critical': return XCircle;
      case 'high': return AlertTriangle;
      case 'medium': return AlertCircle;
      case 'low': return Clock;
      default: return CheckCircle;
    }
  };

  const overviewMetrics = [
    {
      title: 'Models with Drift',
      value: driftData.filter(d => d.driftDetected).length.toString(),
      change: '+1',
      changeType: 'negative' as const,
      icon: AlertTriangle,
      description: 'Models showing performance drift'
    },
    {
      title: 'Critical Alerts',
      value: driftData.filter(d => d.driftSeverity === 'critical' || d.driftSeverity === 'high').length.toString(),
      change: '+1',
      changeType: 'negative' as const,
      icon: AlertCircle,
      description: 'Models requiring immediate attention'
    },
    {
      title: 'Total Impact Cost',
      value: `$${(driftData.reduce((sum, d) => sum + d.estimatedCostImpact, 0) / 1000).toFixed(1)}K`,
      change: '+$12K',
      changeType: 'negative' as const,
      icon: DollarSign,
      description: 'Estimated business impact from drift'
    },
    {
      title: 'Avg Urgency Score',
      value: (driftData.reduce((sum, d) => sum + d.retrainingUrgencyScore, 0) / driftData.length * 100).toFixed(0) + '%',
      change: '+15%',
      changeType: 'negative' as const,
      icon: Zap,
      description: 'Average retraining urgency across models'
    }
  ];

  const renderOverview = () => (
    <div className="space-y-6">
      {/* Overview Metrics */}
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
              icon={<metric.icon className="w-5 h-5" />}
              description={metric.description}
            />
          </motion.div>
        ))}
      </div>

      {/* Model Status Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
        {driftData.map((model, index) => {
          const SeverityIcon = getSeverityIcon(model.driftSeverity);
          const severityColor = getSeverityColor(model.driftSeverity);
          
          return (
            <motion.div
              key={model.modelId}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
            >
              <Card className={`border-l-4 ${
                severityColor === 'red' ? 'border-l-red-500' :
                severityColor === 'amber' ? 'border-l-amber-500' :
                severityColor === 'yellow' ? 'border-l-yellow-500' :
                'border-l-green-500'
              }`}>
                <div className="p-6">
                  <div className="flex items-start justify-between mb-4">
                    <div>
                      <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
                        {model.modelName}
                      </h3>
                      <p className="text-sm text-neutral-600 dark:text-neutral-400">
                        Model ID: {model.modelId}
                      </p>
                    </div>
                    <div className="flex items-center space-x-2">
                      <SeverityIcon className={`w-5 h-5 ${
                        severityColor === 'red' ? 'text-red-500' :
                        severityColor === 'amber' ? 'text-amber-500' :
                        severityColor === 'yellow' ? 'text-yellow-500' :
                        'text-green-500'
                      }`} />
                      <span className={`px-2 py-1 rounded text-xs font-medium ${
                        severityColor === 'red' ? 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300' :
                        severityColor === 'amber' ? 'bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-300' :
                        severityColor === 'yellow' ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300' :
                        'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300'
                      }`}>
                        {model.driftSeverity.charAt(0).toUpperCase() + model.driftSeverity.slice(1)}
                      </span>
                    </div>
                  </div>

                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-neutral-600 dark:text-neutral-400">Drift Confidence:</span>
                      <span className="text-sm font-medium text-neutral-900 dark:text-neutral-100">
                        {(model.driftConfidence * 100).toFixed(1)}%
                      </span>
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-neutral-600 dark:text-neutral-400">Business Impact:</span>
                      <span className="text-sm font-medium text-neutral-900 dark:text-neutral-100">
                        {(model.businessImpactScore * 100).toFixed(1)}%
                      </span>
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-neutral-600 dark:text-neutral-400">Retraining Urgency:</span>
                      <span className="text-sm font-medium text-neutral-900 dark:text-neutral-100">
                        {(model.retrainingUrgencyScore * 100).toFixed(1)}%
                      </span>
                    </div>

                    <div className="pt-2 border-t border-neutral-200 dark:border-neutral-700">
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-neutral-500 dark:text-neutral-400">
                          Cost Impact: ${model.estimatedCostImpact.toLocaleString()}
                        </span>
                        <span className="text-neutral-500 dark:text-neutral-400">
                          {model.affectedPredictions.toLocaleString()} predictions
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </Card>
            </motion.div>
          );
        })}
      </div>
    </div>
  );

  const renderDetection = () => (
    <div className="space-y-6">
      {driftData.filter(d => d.driftDetected).map((model) => (
        <Card key={model.modelId}>
          <div className="p-6">
            <div className="flex items-center justify-between mb-6">
              <div>
                <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
                  {model.modelName}
                </h3>
                <p className="text-sm text-neutral-600 dark:text-neutral-400">
                  Detected at {new Date(model.detectedAt).toLocaleString()}
                </p>
              </div>
              <div className={`px-3 py-2 rounded-lg ${
                model.driftSeverity === 'critical' || model.driftSeverity === 'high' 
                  ? 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300'
                  : 'bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-300'
              }`}>
                {model.driftSeverity.toUpperCase()} DRIFT
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Performance Degradation */}
              <div>
                <h4 className="text-md font-medium text-neutral-900 dark:text-neutral-100 mb-3">
                  Performance Degradation
                </h4>
                <div className="space-y-3">
                  {Object.entries(model.performanceDegradation).map(([metric, degradation]) => (
                    <div key={metric} className="flex items-center justify-between">
                      <span className="text-sm text-neutral-600 dark:text-neutral-400 capitalize">
                        {metric.replace('_', ' ')}:
                      </span>
                      <div className="flex items-center space-x-2">
                        <span className={`text-sm font-medium ${
                          degradation > 0.15 ? 'text-red-600 dark:text-red-400' :
                          degradation > 0.10 ? 'text-amber-600 dark:text-amber-400' :
                          'text-yellow-600 dark:text-yellow-400'
                        }`}>
                          -{(degradation * 100).toFixed(1)}%
                        </span>
                        <div className="w-20 bg-neutral-200 dark:bg-neutral-700 rounded-full h-2">
                          <div
                            className={`h-2 rounded-full ${
                              degradation > 0.15 ? 'bg-red-500' :
                              degradation > 0.10 ? 'bg-amber-500' :
                              'bg-yellow-500'
                            }`}
                            style={{ width: `${Math.min(100, degradation * 500)}%` }}
                          />
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Root Causes */}
              <div>
                <h4 className="text-md font-medium text-neutral-900 dark:text-neutral-100 mb-3">
                  Root Cause Analysis
                </h4>
                <div className="space-y-2">
                  {model.primaryCauses.map((cause, index) => (
                    <div key={index} className="flex items-start space-x-2">
                      <AlertTriangle className="w-4 h-4 text-amber-500 mt-0.5 flex-shrink-0" />
                      <span className="text-sm text-neutral-600 dark:text-neutral-400">
                        {cause}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </Card>
      ))}
    </div>
  );

  const renderImpact = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <div className="p-6">
            <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4">
              Business Impact Summary
            </h3>
            
            <div className="space-y-4">
              <div className="flex items-center justify-between p-3 bg-red-50 dark:bg-red-900/20 rounded-lg">
                <div>
                  <p className="font-medium text-red-900 dark:text-red-100">Total Cost Impact</p>
                  <p className="text-sm text-red-700 dark:text-red-300">Estimated losses from drift</p>
                </div>
                <div className="text-right">
                  <p className="text-2xl font-bold text-red-600 dark:text-red-400">
                    ${(driftData.reduce((sum, d) => sum + d.estimatedCostImpact, 0) / 1000).toFixed(1)}K
                  </p>
                  <p className="text-sm text-red-500">Monthly projection</p>
                </div>
              </div>
              
              <div className="flex items-center justify-between p-3 bg-amber-50 dark:bg-amber-900/20 rounded-lg">
                <div>
                  <p className="font-medium text-amber-900 dark:text-amber-100">Affected Predictions</p>
                  <p className="text-sm text-amber-700 dark:text-amber-300">Total predictions impacted</p>
                </div>
                <div className="text-right">
                  <p className="text-2xl font-bold text-amber-600 dark:text-amber-400">
                    {(driftData.reduce((sum, d) => sum + d.affectedPredictions, 0) / 1000).toFixed(0)}K
                  </p>
                  <p className="text-sm text-amber-500">Last 7 days</p>
                </div>
              </div>
            </div>
          </div>
        </Card>

        {/* Performance Comparison for first model with baseline data */}
        {baselineData.length > 0 && (
          <Card>
            <div className="p-6">
              <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4">
                Performance Baseline vs Current
              </h3>
              
              <div className="space-y-4">
                {Object.entries(baselineData[0].baselineMetrics).map(([metric, baseline]) => {
                  const current = baselineData[0].currentMetrics[metric];
                  const delta = baselineData[0].performanceDelta[metric];
                  const [ciLower, ciUpper] = baselineData[0].confidenceIntervals[metric];
                  
                  return (
                    <div key={metric}>
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium text-neutral-900 dark:text-neutral-100 capitalize">
                          {metric.replace('_', ' ')}
                        </span>
                        <div className="text-right">
                          <span className="text-sm text-neutral-600 dark:text-neutral-400">
                            {baseline.toFixed(3)} → {current.toFixed(3)}
                          </span>
                          <div className={`text-xs ${delta < 0 ? 'text-red-500' : 'text-green-500'}`}>
                            {delta > 0 ? '+' : ''}{delta.toFixed(3)} ({((delta / baseline) * 100).toFixed(1)}%)
                          </div>
                        </div>
                      </div>
                      
                      <div className="flex items-center space-x-2">
                        <div className="flex-1 bg-neutral-200 dark:bg-neutral-700 rounded-full h-2">
                          <div
                            className="bg-blue-500 h-2 rounded-full"
                            style={{ width: `${baseline * 100}%` }}
                          />
                        </div>
                        <div className="flex-1 bg-neutral-200 dark:bg-neutral-700 rounded-full h-2">
                          <div
                            className={`h-2 rounded-full ${current < baseline ? 'bg-red-500' : 'bg-green-500'}`}
                            style={{ width: `${current * 100}%` }}
                          />
                        </div>
                      </div>
                      
                      <div className="text-xs text-neutral-500 dark:text-neutral-400 mt-1">
                        95% CI: [{ciLower.toFixed(3)}, {ciUpper.toFixed(3)}]
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </Card>
        )}
      </div>
    </div>
  );

  const renderRecommendations = () => (
    <div className="space-y-6">
      {driftData.filter(d => d.driftDetected).map((model) => (
        <Card key={model.modelId}>
          <div className="p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
                {model.modelName}
              </h3>
              <div className="flex items-center space-x-2">
                <Clock className="w-4 h-4 text-neutral-500" />
                <span className="text-sm text-neutral-600 dark:text-neutral-400">
                  Urgency: {(model.retrainingUrgencyScore * 100).toFixed(0)}%
                </span>
              </div>
            </div>
            
            <div className="space-y-3">
              {model.recommendations.map((recommendation, index) => (
                <div key={index} className="flex items-start space-x-3 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                  <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold text-white ${
                    index === 0 ? 'bg-red-500' : 
                    index === 1 ? 'bg-amber-500' : 
                    'bg-blue-500'
                  }`}>
                    {index + 1}
                  </div>
                  <div className="flex-1">
                    <p className="text-sm text-neutral-900 dark:text-neutral-100">
                      {recommendation}
                    </p>
                    {index === 0 && model.driftSeverity === 'high' && (
                      <p className="text-xs text-red-600 dark:text-red-400 mt-1 font-medium">
                        HIGH PRIORITY - Immediate action required
                      </p>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </Card>
      ))}
    </div>
  );

  const renderTrends = () => (
    <Card>
      <div className="p-6">
        <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4">
          Drift Trends (7 Days)
        </h3>
        
        <div className="h-64 bg-gradient-to-br from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-lg flex items-center justify-center border-2 border-dashed border-blue-200 dark:border-blue-700">
          <div className="text-center">
            <BarChart3 className="w-12 h-12 text-blue-400 mx-auto mb-2" />
            <p className="text-blue-600 dark:text-blue-400 font-medium">Drift Monitoring Chart</p>
            <p className="text-sm text-blue-500 dark:text-blue-300 mt-1">
              Performance degradation, drift confidence, and impact trends
            </p>
          </div>
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
            Model Drift Monitoring
          </h1>
          <p className="text-neutral-600 dark:text-neutral-400 mt-1">
            Detect performance drift and assess business impact
          </p>
        </div>
        
        <div className="flex items-center space-x-4">
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="px-4 py-2 border border-neutral-300 dark:border-neutral-600 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
          >
            <option value="all">All Models</option>
            {driftData.map(model => (
              <option key={model.modelId} value={model.modelId}>
                {model.modelName}
              </option>
            ))}
          </select>
          
          <select
            value={selectedPeriod}
            onChange={(e) => setSelectedPeriod(e.target.value)}
            className="px-4 py-2 border border-neutral-300 dark:border-neutral-600 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
          >
            <option value="1d">Last 24 Hours</option>
            <option value="7d">Last 7 Days</option>
            <option value="30d">Last 30 Days</option>
            <option value="90d">Last 90 Days</option>
          </select>
          
          <div className="flex items-center space-x-2 px-3 py-2 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
            <Activity className="w-4 h-4 text-blue-600" />
            <span className="text-sm text-blue-700 dark:text-blue-300 font-medium">
              Real-time Monitoring
            </span>
          </div>
          
          <Button variant="outline">
            <RefreshCw className="w-4 h-4 mr-2" />
            Refresh
          </Button>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="flex space-x-1 bg-neutral-100 dark:bg-neutral-800 p-1 rounded-lg w-fit">
        {[
          { key: 'overview', label: 'Overview', icon: BarChart3 },
          { key: 'detection', label: 'Drift Detection', icon: AlertTriangle },
          { key: 'impact', label: 'Impact Analysis', icon: Target },
          { key: 'recommendations', label: 'Recommendations', icon: Bell },
          { key: 'trends', label: 'Trends', icon: TrendingUp }
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
      {activeTab === 'detection' && renderDetection()}
      {activeTab === 'impact' && renderImpact()}
      {activeTab === 'recommendations' && renderRecommendations()}
      {activeTab === 'trends' && renderTrends()}
    </div>
  );
};

export default ModelDriftMonitoringDashboard;