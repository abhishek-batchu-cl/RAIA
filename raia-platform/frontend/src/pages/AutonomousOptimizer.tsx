import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Brain,
  Zap,
  TrendingUp,
  TrendingDown,
  Target,
  AlertTriangle,
  CheckCircle2,
  Clock,
  DollarSign,
  Shield,
  BarChart3,
  Activity,
  Settings,
  Play,
  Pause,
  RefreshCw,
  Download,
  Share2,
  Eye,
  GitBranch,
  Layers,
  Cpu,
  Database,
  Network,
  Gauge,
  Award,
  Lightbulb,
  ArrowRight,
  Star,
  Filter,
  Calendar,
  Users,
  FileText
} from 'lucide-react';
import { Card } from '@/components/common/Card';
import { Button } from '@/components/common/Button';
import { MetricCard } from '@/components/common/MetricCard';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Treemap,
  ScatterChart,
  Scatter
} from 'recharts';

interface OptimizationSuggestion {
  id: string;
  title: string;
  description: string;
  category: 'performance' | 'cost' | 'fairness' | 'reliability' | 'efficiency';
  impact: 'high' | 'medium' | 'low';
  confidence: number;
  estimatedImprovement: {
    metric: string;
    percentage: number;
    absoluteValue?: string;
  };
  implementation: {
    complexity: 'low' | 'medium' | 'high';
    timeToImplement: string;
    resources: string[];
    prerequisites: string[];
  };
  riskLevel: 'low' | 'medium' | 'high';
  status: 'pending' | 'in_progress' | 'completed' | 'rejected';
  priority: number;
  businessValue: number;
  technicalFeasibility: number;
  autoApply: boolean;
  rollbackAvailable: boolean;
  testResults?: {
    abTestId: string;
    winner: 'current' | 'optimized';
    improvementMeasured: number;
    statisticalSignificance: number;
  };
  createdAt: Date;
  updatedAt: Date;
  appliedAt?: Date;
}

interface OptimizationHistory {
  id: string;
  suggestion: OptimizationSuggestion;
  appliedAt: Date;
  measuredImpact: number;
  expectedImpact: number;
  status: 'success' | 'partial' | 'failed' | 'rolled_back';
  rollbackReason?: string;
}

interface ModelHealth {
  modelId: string;
  modelName: string;
  healthScore: number;
  performanceTrend: number;
  costTrend: number;
  fairnessTrend: number;
  lastOptimized: Date;
  optimizationOpportunities: number;
}

interface PredictiveAlert {
  id: string;
  type: 'performance_degradation' | 'cost_spike' | 'fairness_drift' | 'reliability_issue';
  severity: 'low' | 'medium' | 'high' | 'critical';
  title: string;
  description: string;
  predictedOccurrence: Date;
  confidence: number;
  recommendedActions: string[];
  preventable: boolean;
}

const AutonomousOptimizer: React.FC = () => {
  const [suggestions, setSuggestions] = useState<OptimizationSuggestion[]>([]);
  const [history, setHistory] = useState<OptimizationHistory[]>([]);
  const [modelHealth, setModelHealth] = useState<ModelHealth[]>([]);
  const [alerts, setAlerts] = useState<PredictiveAlert[]>([]);
  const [isAutoMode, setIsAutoMode] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [selectedSuggestion, setSelectedSuggestion] = useState<string | null>(null);
  const [optimizationRunning, setOptimizationRunning] = useState(false);
  const [analysisMode, setAnalysisMode] = useState<'continuous' | 'scheduled' | 'manual'>('continuous');

  useEffect(() => {
    // Mock data initialization
    const mockSuggestions: OptimizationSuggestion[] = [
      {
        id: 'opt-1',
        title: 'Enable Model Quantization',
        description: 'Apply 8-bit quantization to reduce model size by 75% while maintaining 98% accuracy',
        category: 'efficiency',
        impact: 'high',
        confidence: 0.92,
        estimatedImprovement: {
          metric: 'Inference Speed',
          percentage: 40,
          absoluteValue: '300ms → 180ms'
        },
        implementation: {
          complexity: 'low',
          timeToImplement: '2-3 hours',
          resources: ['ML Engineer', 'DevOps'],
          prerequisites: ['Model versioning', 'A/B testing setup']
        },
        riskLevel: 'low',
        status: 'pending',
        priority: 95,
        businessValue: 85,
        technicalFeasibility: 90,
        autoApply: true,
        rollbackAvailable: true,
        createdAt: new Date(),
        updatedAt: new Date()
      },
      {
        id: 'opt-2',
        title: 'Optimize Batch Size for Cost Efficiency',
        description: 'Increase batch processing size to 64 samples to reduce per-inference costs by 35%',
        category: 'cost',
        impact: 'high',
        confidence: 0.89,
        estimatedImprovement: {
          metric: 'Cost per 1K inferences',
          percentage: -35,
          absoluteValue: '$45 → $29'
        },
        implementation: {
          complexity: 'medium',
          timeToImplement: '4-6 hours',
          resources: ['ML Engineer', 'Backend Engineer'],
          prerequisites: ['Load testing', 'Memory optimization']
        },
        riskLevel: 'medium',
        status: 'in_progress',
        priority: 88,
        businessValue: 92,
        technicalFeasibility: 75,
        autoApply: false,
        rollbackAvailable: true,
        testResults: {
          abTestId: 'ab-batch-001',
          winner: 'optimized',
          improvementMeasured: 32,
          statisticalSignificance: 0.95
        },
        createdAt: new Date(Date.now() - 86400000),
        updatedAt: new Date(Date.now() - 3600000)
      },
      {
        id: 'opt-3',
        title: 'Implement Dynamic Feature Selection',
        description: 'Use automated feature importance to reduce input dimensionality and improve fairness metrics',
        category: 'fairness',
        impact: 'medium',
        confidence: 0.78,
        estimatedImprovement: {
          metric: 'Fairness Score',
          percentage: 12,
          absoluteValue: '0.78 → 0.87'
        },
        implementation: {
          complexity: 'high',
          timeToImplement: '1-2 weeks',
          resources: ['ML Researcher', 'Data Scientist', 'Ethics Officer'],
          prerequisites: ['Fairness testing framework', 'Feature analysis tools']
        },
        riskLevel: 'medium',
        status: 'pending',
        priority: 76,
        businessValue: 70,
        technicalFeasibility: 65,
        autoApply: false,
        rollbackAvailable: true,
        createdAt: new Date(Date.now() - 172800000),
        updatedAt: new Date(Date.now() - 172800000)
      },
      {
        id: 'opt-4',
        title: 'Enable Adaptive Learning Rate',
        description: 'Implement cosine annealing with warm restarts to improve convergence speed by 25%',
        category: 'performance',
        impact: 'medium',
        confidence: 0.85,
        estimatedImprovement: {
          metric: 'Training Time',
          percentage: -25,
          absoluteValue: '8 hours → 6 hours'
        },
        implementation: {
          complexity: 'low',
          timeToImplement: '1-2 hours',
          resources: ['ML Engineer'],
          prerequisites: ['Training pipeline access']
        },
        riskLevel: 'low',
        status: 'completed',
        priority: 82,
        businessValue: 65,
        technicalFeasibility: 95,
        autoApply: true,
        rollbackAvailable: true,
        appliedAt: new Date(Date.now() - 259200000),
        createdAt: new Date(Date.now() - 345600000),
        updatedAt: new Date(Date.now() - 259200000)
      }
    ];

    const mockHistory: OptimizationHistory[] = [
      {
        id: 'hist-1',
        suggestion: mockSuggestions[3],
        appliedAt: new Date(Date.now() - 259200000),
        measuredImpact: 23,
        expectedImpact: 25,
        status: 'success'
      }
    ];

    const mockModelHealth: ModelHealth[] = [
      {
        modelId: 'gpt-4o',
        modelName: 'GPT-4o',
        healthScore: 87,
        performanceTrend: 2.1,
        costTrend: -5.3,
        fairnessTrend: 1.8,
        lastOptimized: new Date(Date.now() - 259200000),
        optimizationOpportunities: 3
      },
      {
        modelId: 'claude-3.5',
        modelName: 'Claude 3.5 Sonnet',
        healthScore: 92,
        performanceTrend: -1.2,
        costTrend: 3.4,
        fairnessTrend: 4.1,
        lastOptimized: new Date(Date.now() - 86400000),
        optimizationOpportunities: 2
      }
    ];

    const mockAlerts: PredictiveAlert[] = [
      {
        id: 'alert-1',
        type: 'performance_degradation',
        severity: 'medium',
        title: 'Predicted Performance Drop',
        description: 'Model accuracy likely to decrease by 3-5% in the next 48 hours due to data drift',
        predictedOccurrence: new Date(Date.now() + 172800000),
        confidence: 0.78,
        recommendedActions: ['Retrain model with recent data', 'Implement drift detection', 'Schedule model refresh'],
        preventable: true
      },
      {
        id: 'alert-2',
        type: 'cost_spike',
        severity: 'high',
        title: 'Cost Anomaly Predicted',
        description: 'Inference costs expected to increase by 40% due to increased traffic volume',
        predictedOccurrence: new Date(Date.now() + 86400000),
        confidence: 0.84,
        recommendedActions: ['Enable auto-scaling', 'Optimize batch processing', 'Consider model caching'],
        preventable: true
      }
    ];

    setSuggestions(mockSuggestions);
    setHistory(mockHistory);
    setModelHealth(mockModelHealth);
    setAlerts(mockAlerts);
  }, []);

  // Simulate autonomous optimization
  useEffect(() => {
    if (isAutoMode) {
      const interval = setInterval(() => {
        setSuggestions(prev => {
          const autoApplySuggestion = prev.find(s => s.autoApply && s.status === 'pending');
          if (autoApplySuggestion) {
            return prev.map(s => 
              s.id === autoApplySuggestion.id 
                ? { ...s, status: 'in_progress', updatedAt: new Date() }
                : s
            );
          }
          return prev;
        });
      }, 10000);

      return () => clearInterval(interval);
    }
  }, [isAutoMode]);

  const applySuggestion = async (suggestionId: string) => {
    setOptimizationRunning(true);
    
    // Simulate optimization process
    setSuggestions(prev => prev.map(s => 
      s.id === suggestionId 
        ? { ...s, status: 'in_progress', updatedAt: new Date() }
        : s
    ));

    // Simulate delay
    await new Promise(resolve => setTimeout(resolve, 3000));

    setSuggestions(prev => prev.map(s => 
      s.id === suggestionId 
        ? { ...s, status: 'completed', appliedAt: new Date(), updatedAt: new Date() }
        : s
    ));

    setOptimizationRunning(false);
  };

  const filteredSuggestions = suggestions.filter(s => 
    selectedCategory === 'all' || s.category === selectedCategory
  ).sort((a, b) => b.priority - a.priority);

  const getCategoryIcon = (category: OptimizationSuggestion['category']) => {
    switch (category) {
      case 'performance': return <Zap className="w-4 h-4" />;
      case 'cost': return <DollarSign className="w-4 h-4" />;
      case 'fairness': return <Shield className="w-4 h-4" />;
      case 'reliability': return <CheckCircle2 className="w-4 h-4" />;
      case 'efficiency': return <Gauge className="w-4 h-4" />;
      default: return <Brain className="w-4 h-4" />;
    }
  };

  const getImpactColor = (impact: string) => {
    switch (impact) {
      case 'high': return 'text-red-600 bg-red-50 dark:text-red-400 dark:bg-red-900/20';
      case 'medium': return 'text-yellow-600 bg-yellow-50 dark:text-yellow-400 dark:bg-yellow-900/20';
      case 'low': return 'text-green-600 bg-green-50 dark:text-green-400 dark:bg-green-900/20';
      default: return 'text-neutral-600 bg-neutral-50 dark:text-neutral-400 dark:bg-neutral-900/20';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'text-green-600 bg-green-50 dark:text-green-400 dark:bg-green-900/20';
      case 'in_progress': return 'text-blue-600 bg-blue-50 dark:text-blue-400 dark:bg-blue-900/20';
      case 'pending': return 'text-orange-600 bg-orange-50 dark:text-orange-400 dark:bg-orange-900/20';
      case 'rejected': return 'text-red-600 bg-red-50 dark:text-red-400 dark:bg-red-900/20';
      default: return 'text-neutral-600 bg-neutral-50 dark:text-neutral-400 dark:bg-neutral-900/20';
    }
  };

  // Prepare chart data
  const optimizationImpactData = history.map(h => ({
    date: h.appliedAt.toLocaleDateString(),
    expected: h.expectedImpact,
    measured: h.measuredImpact,
    name: h.suggestion.title.slice(0, 20) + '...'
  }));

  const categoryDistribution = suggestions.reduce((acc: any, s) => {
    acc[s.category] = (acc[s.category] || 0) + 1;
    return acc;
  }, {});

  const categoryPieData = Object.entries(categoryDistribution).map(([key, value]) => ({
    name: key,
    value: value as number
  }));

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between">
        <div>
          <h1 className="text-3xl font-bold text-neutral-900 dark:text-white flex items-center gap-3">
            <div className="p-2 bg-gradient-to-br from-green-500 to-blue-500 rounded-lg">
              <Brain className="w-6 h-6 text-white" />
            </div>
            Autonomous Model Optimizer
          </h1>
          <p className="text-neutral-600 dark:text-neutral-400 mt-2">
            AI-powered optimization recommendations with predictive insights and auto-application
          </p>
        </div>

        <div className="flex items-center gap-3 mt-4 lg:mt-0">
          <div className="flex items-center gap-2">
            <span className="text-sm text-neutral-600 dark:text-neutral-400">Auto Mode:</span>
            <Button
              variant={isAutoMode ? "default" : "outline"}
              size="sm"
              onClick={() => setIsAutoMode(!isAutoMode)}
              className="flex items-center gap-2"
            >
              {isAutoMode ? (
                <>
                  <Pause className="w-4 h-4" />
                  Disable
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  Enable
                </>
              )}
            </Button>
          </div>

          <select
            value={analysisMode}
            onChange={(e) => setAnalysisMode(e.target.value as any)}
            className="px-3 py-2 border border-neutral-300 dark:border-neutral-600 rounded-lg bg-white dark:bg-neutral-800 text-sm"
          >
            <option value="continuous">Continuous Analysis</option>
            <option value="scheduled">Scheduled Analysis</option>
            <option value="manual">Manual Analysis</option>
          </select>

          <Button variant="outline" className="flex items-center gap-2">
            <Download className="w-4 h-4" />
            Export Report
          </Button>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Active Suggestions"
          value={filteredSuggestions.filter(s => s.status === 'pending').length.toString()}
          icon={<Lightbulb className="w-5 h-5" />}
          trend={8}
          className="bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20"
        />
        
        <MetricCard
          title="Auto-Applied Today"
          value={suggestions.filter(s => s.status === 'completed' && s.autoApply).length.toString()}
          icon={CheckCircle2}
          trend={12}
          className="bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20"
        />
        
        <MetricCard
          title="Avg Confidence"
          value={`${(suggestions.reduce((acc, s) => acc + s.confidence, 0) / suggestions.length * 100).toFixed(1)}%`}
          icon={<Target className="w-5 h-5" />}
          trend={3.2}
          className="bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20"
        />
        
        <MetricCard
          title="Predicted Savings"
          value="$12.4K/month"
          icon={<DollarSign className="w-5 h-5" />}
          trend={25.8}
          className="bg-gradient-to-br from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20"
        />
      </div>

      {/* Predictive Alerts */}
      {alerts.length > 0 && (
        <Card className="p-6 border-l-4 border-l-orange-500">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold flex items-center gap-2">
              <AlertTriangle className="w-5 h-5 text-orange-600" />
              Predictive Alerts ({alerts.length})
            </h3>
            <Button variant="outline" size="sm">
              View All
            </Button>
          </div>

          <div className="space-y-3">
            {alerts.map((alert) => (
              <div key={alert.id} className={`p-4 rounded-lg border-l-4 ${
                alert.severity === 'critical' ? 'border-l-red-500 bg-red-50 dark:bg-red-900/20' :
                alert.severity === 'high' ? 'border-l-orange-500 bg-orange-50 dark:bg-orange-900/20' :
                'border-l-yellow-500 bg-yellow-50 dark:bg-yellow-900/20'
              }`}>
                <div className="flex items-start justify-between mb-2">
                  <div>
                    <h4 className="font-medium text-neutral-900 dark:text-white">
                      {alert.title}
                    </h4>
                    <p className="text-sm text-neutral-600 dark:text-neutral-400 mt-1">
                      {alert.description}
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className={`px-2 py-1 text-xs rounded-full ${
                      alert.severity === 'critical' ? 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300' :
                      alert.severity === 'high' ? 'bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-300' :
                      'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300'
                    }`}>
                      {alert.severity}
                    </span>
                    <span className="text-xs text-neutral-500">
                      {(alert.confidence * 100).toFixed(0)}% confidence
                    </span>
                  </div>
                </div>

                <div className="flex items-center justify-between text-sm">
                  <span className="text-neutral-600 dark:text-neutral-400">
                    Expected: {alert.predictedOccurrence.toLocaleDateString()}
                  </span>
                  {alert.preventable && (
                    <Button variant="outline" size="sm">
                      Prevent
                    </Button>
                  )}
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Optimization Suggestions */}
        <div className="lg:col-span-2 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-semibold text-neutral-900 dark:text-white">
              Optimization Suggestions
            </h2>
            <div className="flex items-center gap-2">
              <select
                value={selectedCategory}
                onChange={(e) => setSelectedCategory(e.target.value)}
                className="px-3 py-1 border border-neutral-300 dark:border-neutral-600 rounded bg-white dark:bg-neutral-800 text-sm"
              >
                <option value="all">All Categories</option>
                <option value="performance">Performance</option>
                <option value="cost">Cost</option>
                <option value="fairness">Fairness</option>
                <option value="reliability">Reliability</option>
                <option value="efficiency">Efficiency</option>
              </select>
              <Button variant="outline" size="sm">
                <RefreshCw className="w-4 h-4" />
              </Button>
            </div>
          </div>

          <div className="space-y-4">
            <AnimatePresence mode="popLayout">
              {filteredSuggestions.map((suggestion) => (
                <motion.div
                  key={suggestion.id}
                  layout
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className="bg-white dark:bg-neutral-800 rounded-lg border border-neutral-200 dark:border-neutral-700 p-6"
                >
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-start gap-3">
                      <div className="p-2 bg-neutral-100 dark:bg-neutral-700 rounded-lg">
                        {getCategoryIcon(suggestion.category)}
                      </div>
                      <div>
                        <h3 className="font-semibold text-neutral-900 dark:text-white">
                          {suggestion.title}
                        </h3>
                        <p className="text-sm text-neutral-600 dark:text-neutral-400 mt-1">
                          {suggestion.description}
                        </p>
                      </div>
                    </div>
                    
                    <div className="flex items-center gap-2">
                      <span className={`px-2 py-1 text-xs rounded-full ${getImpactColor(suggestion.impact)}`}>
                        {suggestion.impact} impact
                      </span>
                      <span className={`px-2 py-1 text-xs rounded-full ${getStatusColor(suggestion.status)}`}>
                        {suggestion.status.replace('_', ' ')}
                      </span>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-4 mb-4">
                    <div>
                      <div className="text-sm text-neutral-600 dark:text-neutral-400">Expected Improvement</div>
                      <div className="font-medium text-neutral-900 dark:text-white">
                        {suggestion.estimatedImprovement.percentage > 0 ? '+' : ''}{suggestion.estimatedImprovement.percentage}% {suggestion.estimatedImprovement.metric}
                      </div>
                      {suggestion.estimatedImprovement.absoluteValue && (
                        <div className="text-xs text-neutral-500">
                          {suggestion.estimatedImprovement.absoluteValue}
                        </div>
                      )}
                    </div>
                    
                    <div>
                      <div className="text-sm text-neutral-600 dark:text-neutral-400">Implementation</div>
                      <div className="font-medium text-neutral-900 dark:text-white">
                        {suggestion.implementation.complexity} complexity
                      </div>
                      <div className="text-xs text-neutral-500">
                        {suggestion.implementation.timeToImplement}
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                      <div className="text-sm">
                        <span className="text-neutral-600 dark:text-neutral-400">Confidence: </span>
                        <span className="font-medium">{(suggestion.confidence * 100).toFixed(0)}%</span>
                      </div>
                      <div className="text-sm">
                        <span className="text-neutral-600 dark:text-neutral-400">Priority: </span>
                        <span className="font-medium">{suggestion.priority}</span>
                      </div>
                      {suggestion.autoApply && (
                        <span className="text-xs bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300 px-2 py-1 rounded">
                          Auto-apply enabled
                        </span>
                      )}
                    </div>

                    <div className="flex items-center gap-2">
                      {suggestion.status === 'pending' && (
                        <Button
                          size="sm"
                          onClick={() => applySuggestion(suggestion.id)}
                          disabled={optimizationRunning}
                          className="flex items-center gap-2"
                        >
                          {optimizationRunning ? (
                            <>
                              <RefreshCw className="w-4 h-4 animate-spin" />
                              Applying...
                            </>
                          ) : (
                            <>
                              <Play className="w-4 h-4" />
                              Apply
                            </>
                          )}
                        </Button>
                      )}
                      <Button variant="outline" size="sm">
                        Details
                      </Button>
                    </div>
                  </div>

                  {suggestion.testResults && (
                    <div className="mt-4 p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
                      <div className="text-sm font-medium text-green-800 dark:text-green-300 mb-1">
                        A/B Test Results
                      </div>
                      <div className="text-sm text-green-700 dark:text-green-400">
                        Measured improvement: {suggestion.testResults.improvementMeasured}% 
                        (p={suggestion.testResults.statisticalSignificance})
                      </div>
                    </div>
                  )}
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          {/* Model Health */}
          <Card className="p-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Activity className="w-5 h-5 text-green-600" />
              Model Health
            </h3>
            
            <div className="space-y-4">
              {modelHealth.map((model) => (
                <div key={model.modelId} className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="font-medium text-neutral-900 dark:text-white">
                      {model.modelName}
                    </span>
                    <span className={`text-sm font-medium ${
                      model.healthScore >= 90 ? 'text-green-600' :
                      model.healthScore >= 75 ? 'text-yellow-600' :
                      'text-red-600'
                    }`}>
                      {model.healthScore}%
                    </span>
                  </div>
                  
                  <div className="w-full bg-neutral-200 dark:bg-neutral-700 rounded-full h-2">
                    <div
                      className={`h-2 rounded-full ${
                        model.healthScore >= 90 ? 'bg-green-500' :
                        model.healthScore >= 75 ? 'bg-yellow-500' :
                        'bg-red-500'
                      }`}
                      style={{ width: `${model.healthScore}%` }}
                    />
                  </div>
                  
                  <div className="flex justify-between text-xs text-neutral-600 dark:text-neutral-400">
                    <span>Performance: {model.performanceTrend > 0 ? '+' : ''}{model.performanceTrend.toFixed(1)}%</span>
                    <span>{model.optimizationOpportunities} opportunities</span>
                  </div>
                </div>
              ))}
            </div>
          </Card>

          {/* Optimization Impact */}
          <Card className="p-6">
            <h3 className="text-lg font-semibold mb-4">Optimization Impact</h3>
            
            {optimizationImpactData.length > 0 ? (
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={optimizationImpactData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="name" fontSize={12} />
                  <YAxis fontSize={12} />
                  <Tooltip />
                  <Bar dataKey="expected" fill="#e2e8f0" name="Expected" />
                  <Bar dataKey="measured" fill="#8b5cf6" name="Measured" />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="text-center py-8 text-neutral-500 dark:text-neutral-400">
                No optimization history yet
              </div>
            )}
          </Card>

          {/* Category Distribution */}
          <Card className="p-6">
            <h3 className="text-lg font-semibold mb-4">Suggestions by Category</h3>
            
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie
                  data={categoryPieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={40}
                  outerRadius={80}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {categoryPieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={['#8b5cf6', '#06d6a0', '#f72585', '#4cc9f0', '#7209b7'][index % 5]} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default AutonomousOptimizer;