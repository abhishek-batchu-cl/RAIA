import React, { useState, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  BarChart3,
  TrendingUp,
  TrendingDown,
  DollarSign,
  Clock,
  Zap,
  Target,
  Award,
  AlertTriangle,
  CheckCircle2,
  Play,
  Pause,
  RefreshCw,
  Download,
  Share2,
  Settings,
  Filter,
  Eye,
  EyeOff,
  Plus,
  Minus,
  Maximize2,
  Grid3X3,
  BarChart,
  PieChart,
  Activity,
  Cpu
} from 'lucide-react';
import { Card } from '@/components/common/Card';
import { Button } from '@/components/common/Button';
import { MetricCard } from '@/components/common/MetricCard';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart as ReBarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ScatterChart,
  Scatter,
  Cell,
  PieChart as RePieChart,
  Pie,
  Treemap
} from 'recharts';

interface ModelMetrics {
  id: string;
  name: string;
  type: 'llm' | 'classical' | 'ensemble' | 'neural';
  version: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  auc: number;
  latency: number; // ms
  throughput: number; // requests/sec
  costPerInference: number; // $
  memoryUsage: number; // MB
  cpuUsage: number; // %
  trainingTime: number; // hours
  modelSize: number; // MB
  carbonFootprint: number; // kg CO2
  fairnessScore: number;
  interpretabilityScore: number;
  robustnessScore: number;
  lastUpdated: Date;
  isActive: boolean;
  tags: string[];
  deploymentStatus: 'production' | 'staging' | 'experimental' | 'deprecated';
}

interface BenchmarkTest {
  id: string;
  name: string;
  description: string;
  testType: 'accuracy' | 'speed' | 'robustness' | 'fairness' | 'cost';
  results: { [modelId: string]: number };
  unit: string;
  higherIsBetter: boolean;
  weight: number;
}

interface OptimizationRecommendation {
  id: string;
  modelId: string;
  type: 'performance' | 'cost' | 'fairness' | 'efficiency';
  title: string;
  description: string;
  impact: 'high' | 'medium' | 'low';
  effort: 'high' | 'medium' | 'low';
  estimatedImprovement: number;
  priority: number;
}

const AdvancedModelComparison: React.FC = () => {
  const [models, setModels] = useState<ModelMetrics[]>([]);
  const [benchmarks, setBenchmarks] = useState<BenchmarkTest[]>([]);
  const [recommendations, setRecommendations] = useState<OptimizationRecommendation[]>([]);
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [comparisonView, setComparisonView] = useState<'grid' | 'radar' | 'scatter' | 'heatmap'>('grid');
  const [filterBy, setFilterBy] = useState<'all' | 'production' | 'experimental'>('all');
  const [sortBy, setSortBy] = useState<keyof ModelMetrics>('accuracy');
  const [isLiveBenchmarking, setIsLiveBenchmarking] = useState(false);
  const [selectedMetrics, setSelectedMetrics] = useState<string[]>(['accuracy', 'latency', 'costPerInference']);
  const [showAdvancedFilters, setShowAdvancedFilters] = useState(false);

  // Initialize mock data
  useEffect(() => {
    const mockModels: ModelMetrics[] = [
      {
        id: 'gpt-4o',
        name: 'GPT-4o',
        type: 'llm',
        version: '2024-08-06',
        accuracy: 0.94,
        precision: 0.92,
        recall: 0.91,
        f1Score: 0.915,
        auc: 0.96,
        latency: 1200,
        throughput: 15,
        costPerInference: 0.045,
        memoryUsage: 2500,
        cpuUsage: 85,
        trainingTime: 720,
        modelSize: 1750,
        carbonFootprint: 850,
        fairnessScore: 0.78,
        interpretabilityScore: 0.65,
        robustnessScore: 0.88,
        lastUpdated: new Date(),
        isActive: true,
        tags: ['conversation', 'reasoning', 'code'],
        deploymentStatus: 'production'
      },
      {
        id: 'claude-3.5-sonnet',
        name: 'Claude 3.5 Sonnet',
        type: 'llm',
        version: '20241022',
        accuracy: 0.91,
        precision: 0.89,
        recall: 0.93,
        f1Score: 0.91,
        auc: 0.94,
        latency: 950,
        throughput: 18,
        costPerInference: 0.035,
        memoryUsage: 2200,
        cpuUsage: 78,
        trainingTime: 680,
        modelSize: 1600,
        carbonFootprint: 720,
        fairnessScore: 0.82,
        interpretabilityScore: 0.71,
        robustnessScore: 0.85,
        lastUpdated: new Date(Date.now() - 86400000),
        isActive: true,
        tags: ['analysis', 'writing', 'reasoning'],
        deploymentStatus: 'production'
      },
      {
        id: 'llama-3.1-70b',
        name: 'Llama 3.1 70B',
        type: 'llm',
        version: '405B',
        accuracy: 0.89,
        precision: 0.87,
        recall: 0.90,
        f1Score: 0.885,
        auc: 0.92,
        latency: 2200,
        throughput: 8,
        costPerInference: 0.018,
        memoryUsage: 3200,
        cpuUsage: 92,
        trainingTime: 450,
        modelSize: 2800,
        carbonFootprint: 420,
        fairnessScore: 0.85,
        interpretabilityScore: 0.68,
        robustnessScore: 0.83,
        lastUpdated: new Date(Date.now() - 172800000),
        isActive: true,
        tags: ['open-source', 'multilingual', 'code'],
        deploymentStatus: 'production'
      },
      {
        id: 'mistral-large',
        name: 'Mistral Large',
        type: 'llm',
        version: '2407',
        accuracy: 0.87,
        precision: 0.85,
        recall: 0.88,
        f1Score: 0.865,
        auc: 0.90,
        latency: 800,
        throughput: 22,
        costPerInference: 0.028,
        memoryUsage: 1800,
        cpuUsage: 72,
        trainingTime: 380,
        modelSize: 1400,
        carbonFootprint: 340,
        fairnessScore: 0.79,
        interpretabilityScore: 0.74,
        robustnessScore: 0.81,
        lastUpdated: new Date(Date.now() - 259200000),
        isActive: true,
        tags: ['efficient', 'multilingual', 'reasoning'],
        deploymentStatus: 'production'
      },
      {
        id: 'gemini-1.5-pro',
        name: 'Gemini 1.5 Pro',
        type: 'llm',
        version: '002',
        accuracy: 0.90,
        precision: 0.88,
        recall: 0.89,
        f1Score: 0.885,
        auc: 0.93,
        latency: 1100,
        throughput: 16,
        costPerInference: 0.041,
        memoryUsage: 2100,
        cpuUsage: 80,
        trainingTime: 520,
        modelSize: 1650,
        carbonFootprint: 580,
        fairnessScore: 0.76,
        interpretabilityScore: 0.69,
        robustnessScore: 0.87,
        lastUpdated: new Date(Date.now() - 345600000),
        isActive: true,
        tags: ['multimodal', 'context', 'reasoning'],
        deploymentStatus: 'staging'
      }
    ];

    const mockBenchmarks: BenchmarkTest[] = [
      {
        id: 'mmlu',
        name: 'MMLU',
        description: 'Massive Multitask Language Understanding',
        testType: 'accuracy',
        results: {
          'gpt-4o': 88.7,
          'claude-3.5-sonnet': 88.3,
          'llama-3.1-70b': 85.2,
          'mistral-large': 81.2,
          'gemini-1.5-pro': 85.9
        },
        unit: '%',
        higherIsBetter: true,
        weight: 0.3
      },
      {
        id: 'humaneval',
        name: 'HumanEval',
        description: 'Code generation benchmark',
        testType: 'accuracy',
        results: {
          'gpt-4o': 90.2,
          'claude-3.5-sonnet': 92.0,
          'llama-3.1-70b': 80.5,
          'mistral-large': 75.3,
          'gemini-1.5-pro': 84.7
        },
        unit: '%',
        higherIsBetter: true,
        weight: 0.25
      },
      {
        id: 'latency',
        name: 'Response Latency',
        description: 'Average time to first token',
        testType: 'speed',
        results: {
          'gpt-4o': 1200,
          'claude-3.5-sonnet': 950,
          'llama-3.1-70b': 2200,
          'mistral-large': 800,
          'gemini-1.5-pro': 1100
        },
        unit: 'ms',
        higherIsBetter: false,
        weight: 0.2
      },
      {
        id: 'cost-efficiency',
        name: 'Cost per 1M tokens',
        description: 'Cost efficiency for inference',
        testType: 'cost',
        results: {
          'gpt-4o': 45.0,
          'claude-3.5-sonnet': 35.0,
          'llama-3.1-70b': 18.0,
          'mistral-large': 28.0,
          'gemini-1.5-pro': 41.0
        },
        unit: '$',
        higherIsBetter: false,
        weight: 0.25
      }
    ];

    const mockRecommendations: OptimizationRecommendation[] = [
      {
        id: 'rec-1',
        modelId: 'gpt-4o',
        type: 'cost',
        title: 'Optimize batch processing',
        description: 'Implement request batching to reduce per-inference costs by up to 30%',
        impact: 'high',
        effort: 'medium',
        estimatedImprovement: 30,
        priority: 1
      },
      {
        id: 'rec-2',
        modelId: 'llama-3.1-70b',
        type: 'performance',
        title: 'Enable quantization',
        description: 'Use 4-bit quantization to reduce latency while maintaining 95% accuracy',
        impact: 'high',
        effort: 'low',
        estimatedImprovement: 45,
        priority: 2
      }
    ];

    setModels(mockModels);
    setBenchmarks(mockBenchmarks);
    setRecommendations(mockRecommendations);
    setSelectedModels(['gpt-4o', 'claude-3.5-sonnet', 'llama-3.1-70b']);
  }, []);

  // Filter models based on current filters
  const filteredModels = useMemo(() => {
    let filtered = models;
    
    if (filterBy !== 'all') {
      filtered = filtered.filter(m => m.deploymentStatus === filterBy);
    }
    
    return filtered.sort((a, b) => {
      const aValue = a[sortBy] as number;
      const bValue = b[sortBy] as number;
      return bValue - aValue;
    });
  }, [models, filterBy, sortBy]);

  // Prepare radar chart data
  const radarData = useMemo(() => {
    const metrics = ['accuracy', 'precision', 'recall', 'fairnessScore', 'robustnessScore'];
    return metrics.map(metric => {
      const dataPoint: any = { metric: metric.replace(/([A-Z])/g, ' $1').trim() };
      selectedModels.forEach(modelId => {
        const model = models.find(m => m.id === modelId);
        if (model) {
          dataPoint[model.name] = (model[metric as keyof ModelMetrics] as number) * 100;
        }
      });
      return dataPoint;
    });
  }, [models, selectedModels]);

  // Prepare scatter plot data
  const scatterData = useMemo(() => {
    return selectedModels.map(modelId => {
      const model = models.find(m => m.id === modelId);
      if (model) {
        return {
          name: model.name,
          x: model.accuracy * 100,
          y: model.costPerInference * 1000,
          z: model.latency
        };
      }
      return null;
    }).filter(Boolean);
  }, [models, selectedModels]);

  // Calculate composite scores
  const calculateCompositeScore = (model: ModelMetrics) => {
    const weights = {
      accuracy: 0.25,
      latency: 0.2,
      cost: 0.2,
      fairness: 0.15,
      robustness: 0.2
    };
    
    // Normalize values (higher is better)
    const normalizedAccuracy = model.accuracy;
    const normalizedLatency = 1 - (model.latency / 3000); // Assuming max latency is 3000ms
    const normalizedCost = 1 - (model.costPerInference / 0.05); // Assuming max cost is $0.05
    const normalizedFairness = model.fairnessScore;
    const normalizedRobustness = model.robustnessScore;
    
    return (
      normalizedAccuracy * weights.accuracy +
      normalizedLatency * weights.latency +
      normalizedCost * weights.cost +
      normalizedFairness * weights.fairness +
      normalizedRobustness * weights.robustness
    ) * 100;
  };

  const handleModelToggle = (modelId: string) => {
    setSelectedModels(prev => 
      prev.includes(modelId) 
        ? prev.filter(id => id !== modelId)
        : [...prev, modelId]
    );
  };

  const getModelColor = (modelId: string, index: number) => {
    const colors = ['#8b5cf6', '#06d6a0', '#f72585', '#4cc9f0', '#7209b7', '#f77f00'];
    return colors[index % colors.length];
  };

  const renderComparisonView = () => {
    const selectedModelData = models.filter(m => selectedModels.includes(m.id));

    switch (comparisonView) {
      case 'radar':
        return (
          <Card className="p-6">
            <h3 className="text-lg font-semibold mb-4">Performance Radar</h3>
            <ResponsiveContainer width="100%" height={400}>
              <RadarChart data={radarData}>
                <PolarGrid stroke="#e5e7eb" />
                <PolarAngleAxis dataKey="metric" />
                <PolarRadiusAxis angle={30} domain={[0, 100]} />
                {selectedModelData.map((model, index) => (
                  <Radar
                    key={model.id}
                    name={model.name}
                    dataKey={model.name}
                    stroke={getModelColor(model.id, index)}
                    fill={getModelColor(model.id, index)}
                    fillOpacity={0.2}
                    strokeWidth={2}
                  />
                ))}
                <Legend />
              </RadarChart>
            </ResponsiveContainer>
          </Card>
        );

      case 'scatter':
        return (
          <Card className="p-6">
            <h3 className="text-lg font-semibold mb-4">Accuracy vs Cost vs Latency</h3>
            <ResponsiveContainer width="100%" height={400}>
              <ScatterChart data={scatterData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="x" 
                  name="Accuracy"
                  unit="%" 
                  label={{ value: 'Accuracy (%)', position: 'insideBottom', offset: -10 }}
                />
                <YAxis 
                  dataKey="y" 
                  name="Cost"
                  unit="$" 
                  label={{ value: 'Cost per 1K inferences ($)', angle: -90, position: 'insideLeft' }}
                />
                <Tooltip 
                  formatter={(value, name) => {
                    if (name === 'x') return [`${value}%`, 'Accuracy'];
                    if (name === 'y') return [`$${(value / 1000).toFixed(3)}`, 'Cost per inference'];
                    if (name === 'z') return [`${value}ms`, 'Latency'];
                    return [value, name];
                  }}
                />
                <Scatter name="Models" dataKey="y" fill="#8b5cf6">
                  {scatterData.map((entry, index) => (
                    <Cell key={index} fill={getModelColor('', index)} />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </Card>
        );

      case 'heatmap':
        return (
          <Card className="p-6">
            <h3 className="text-lg font-semibold mb-4">Performance Heatmap</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b">
                    <th className="text-left p-2">Model</th>
                    <th className="text-center p-2">Accuracy</th>
                    <th className="text-center p-2">Latency</th>
                    <th className="text-center p-2">Cost</th>
                    <th className="text-center p-2">Fairness</th>
                    <th className="text-center p-2">Robustness</th>
                    <th className="text-center p-2">Score</th>
                  </tr>
                </thead>
                <tbody>
                  {selectedModelData.map((model) => {
                    const compositeScore = calculateCompositeScore(model);
                    return (
                      <tr key={model.id} className="border-b">
                        <td className="p-2 font-medium">{model.name}</td>
                        <td className="p-2 text-center">
                          <div 
                            className={`px-2 py-1 rounded text-xs font-medium ${
                              model.accuracy >= 0.9 ? 'bg-green-100 text-green-800' :
                              model.accuracy >= 0.85 ? 'bg-yellow-100 text-yellow-800' :
                              'bg-red-100 text-red-800'
                            }`}
                          >
                            {(model.accuracy * 100).toFixed(1)}%
                          </div>
                        </td>
                        <td className="p-2 text-center">
                          <div 
                            className={`px-2 py-1 rounded text-xs font-medium ${
                              model.latency <= 1000 ? 'bg-green-100 text-green-800' :
                              model.latency <= 1500 ? 'bg-yellow-100 text-yellow-800' :
                              'bg-red-100 text-red-800'
                            }`}
                          >
                            {model.latency}ms
                          </div>
                        </td>
                        <td className="p-2 text-center">
                          <div 
                            className={`px-2 py-1 rounded text-xs font-medium ${
                              model.costPerInference <= 0.03 ? 'bg-green-100 text-green-800' :
                              model.costPerInference <= 0.04 ? 'bg-yellow-100 text-yellow-800' :
                              'bg-red-100 text-red-800'
                            }`}
                          >
                            ${(model.costPerInference * 1000).toFixed(1)}
                          </div>
                        </td>
                        <td className="p-2 text-center">
                          <div 
                            className={`px-2 py-1 rounded text-xs font-medium ${
                              model.fairnessScore >= 0.8 ? 'bg-green-100 text-green-800' :
                              model.fairnessScore >= 0.75 ? 'bg-yellow-100 text-yellow-800' :
                              'bg-red-100 text-red-800'
                            }`}
                          >
                            {(model.fairnessScore * 100).toFixed(0)}%
                          </div>
                        </td>
                        <td className="p-2 text-center">
                          <div 
                            className={`px-2 py-1 rounded text-xs font-medium ${
                              model.robustnessScore >= 0.85 ? 'bg-green-100 text-green-800' :
                              model.robustnessScore >= 0.8 ? 'bg-yellow-100 text-yellow-800' :
                              'bg-red-100 text-red-800'
                            }`}
                          >
                            {(model.robustnessScore * 100).toFixed(0)}%
                          </div>
                        </td>
                        <td className="p-2 text-center">
                          <div 
                            className={`px-2 py-1 rounded text-xs font-medium ${
                              compositeScore >= 80 ? 'bg-green-100 text-green-800' :
                              compositeScore >= 70 ? 'bg-yellow-100 text-yellow-800' :
                              'bg-red-100 text-red-800'
                            }`}
                          >
                            {compositeScore.toFixed(0)}
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </Card>
        );

      default: // grid
        return (
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
            {selectedModelData.map((model) => {
              const compositeScore = calculateCompositeScore(model);
              return (
                <motion.div
                  key={model.id}
                  layout
                  className="bg-white dark:bg-neutral-800 rounded-lg border border-neutral-200 dark:border-neutral-700 p-6"
                >
                  <div className="flex items-center justify-between mb-4">
                    <div>
                      <h3 className="text-lg font-semibold text-neutral-900 dark:text-white">
                        {model.name}
                      </h3>
                      <div className="flex items-center gap-2 mt-1">
                        <span className={`px-2 py-1 text-xs rounded-full ${
                          model.deploymentStatus === 'production' 
                            ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300'
                            : model.deploymentStatus === 'staging'
                              ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300'
                              : 'bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300'
                        }`}>
                          {model.deploymentStatus}
                        </span>
                        <span className="text-xs text-neutral-500">{model.type}</span>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-2xl font-bold text-purple-600">
                        {compositeScore.toFixed(0)}
                      </div>
                      <div className="text-xs text-neutral-500">Composite Score</div>
                    </div>
                  </div>

                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-neutral-600 dark:text-neutral-400">Accuracy</span>
                      <span className="font-medium">{(model.accuracy * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-neutral-600 dark:text-neutral-400">Latency</span>
                      <span className="font-medium">{model.latency}ms</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-neutral-600 dark:text-neutral-400">Cost/inference</span>
                      <span className="font-medium">${(model.costPerInference * 1000).toFixed(1)}/1K</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-neutral-600 dark:text-neutral-400">Fairness</span>
                      <span className="font-medium">{(model.fairnessScore * 100).toFixed(0)}%</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-neutral-600 dark:text-neutral-400">Model Size</span>
                      <span className="font-medium">{model.modelSize}MB</span>
                    </div>
                  </div>

                  <div className="mt-4 flex flex-wrap gap-1">
                    {model.tags.map(tag => (
                      <span key={tag} className="px-2 py-1 text-xs bg-neutral-100 dark:bg-neutral-700 text-neutral-700 dark:text-neutral-300 rounded">
                        {tag}
                      </span>
                    ))}
                  </div>
                </motion.div>
              );
            })}
          </div>
        );
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between">
        <div>
          <h1 className="text-3xl font-bold text-neutral-900 dark:text-white flex items-center gap-3">
            <div className="p-2 bg-gradient-to-br from-blue-500 to-purple-500 rounded-lg">
              <BarChart3 className="w-6 h-6 text-white" />
            </div>
            Advanced Model Comparison
          </h1>
          <p className="text-neutral-600 dark:text-neutral-400 mt-2">
            Comprehensive multi-dimensional model performance analysis and optimization
          </p>
        </div>

        <div className="flex items-center gap-3 mt-4 lg:mt-0">
          <Button
            variant={isLiveBenchmarking ? "destructive" : "secondary"}
            onClick={() => setIsLiveBenchmarking(!isLiveBenchmarking)}
            className="flex items-center gap-2"
          >
            {isLiveBenchmarking ? (
              <>
                <Pause className="w-4 h-4" />
                Stop Live
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                Live Benchmarks
              </>
            )}
          </Button>

          <Button variant="outline" className="flex items-center gap-2">
            <Download className="w-4 h-4" />
            Export Report
          </Button>

          <Button variant="outline" className="flex items-center gap-2">
            <Share2 className="w-4 h-4" />
            Share Analysis
          </Button>
        </div>
      </div>

      {/* Filters and Controls */}
      <Card className="p-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex flex-wrap items-center gap-3">
            <select 
              value={filterBy}
              onChange={(e) => setFilterBy(e.target.value as any)}
              className="px-3 py-2 border border-neutral-300 dark:border-neutral-600 rounded-lg bg-white dark:bg-neutral-800 text-sm"
            >
              <option value="all">All Models</option>
              <option value="production">Production</option>
              <option value="experimental">Experimental</option>
            </select>

            <select 
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as keyof ModelMetrics)}
              className="px-3 py-2 border border-neutral-300 dark:border-neutral-600 rounded-lg bg-white dark:bg-neutral-800 text-sm"
            >
              <option value="accuracy">Sort by Accuracy</option>
              <option value="latency">Sort by Latency</option>
              <option value="costPerInference">Sort by Cost</option>
              <option value="fairnessScore">Sort by Fairness</option>
            </select>

            <Button
              variant="outline"
              onClick={() => setShowAdvancedFilters(!showAdvancedFilters)}
              className="flex items-center gap-2"
            >
              <Filter className="w-4 h-4" />
              Advanced Filters
            </Button>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-sm text-neutral-600 dark:text-neutral-400">View:</span>
            {(['grid', 'radar', 'scatter', 'heatmap'] as const).map((view) => (
              <Button
                key={view}
                variant={comparisonView === view ? "default" : "outline"}
                size="sm"
                onClick={() => setComparisonView(view)}
              >
                {view === 'grid' && <Grid3X3 className="w-4 h-4" />}
                {view === 'radar' && <Target className="w-4 h-4" />}
                {view === 'scatter' && <BarChart className="w-4 h-4" />}
                {view === 'heatmap' && <Activity className="w-4 h-4" />}
              </Button>
            ))}
          </div>
        </div>

        {/* Model Selection */}
        <div className="mt-4 pt-4 border-t border-neutral-200 dark:border-neutral-700">
          <div className="flex flex-wrap gap-2">
            <span className="text-sm font-medium text-neutral-700 dark:text-neutral-300 py-2">
              Select Models to Compare:
            </span>
            {filteredModels.map((model) => (
              <button
                key={model.id}
                onClick={() => handleModelToggle(model.id)}
                className={`px-3 py-1 text-sm rounded-full border transition-all ${
                  selectedModels.includes(model.id)
                    ? 'bg-purple-100 border-purple-300 text-purple-800 dark:bg-purple-900/30 dark:border-purple-700 dark:text-purple-300'
                    : 'bg-neutral-100 border-neutral-300 text-neutral-700 dark:bg-neutral-700 dark:border-neutral-600 dark:text-neutral-300 hover:bg-neutral-200 dark:hover:bg-neutral-600'
                }`}
              >
                {selectedModels.includes(model.id) ? (
                  <Eye className="w-3 h-3 inline mr-1" />
                ) : (
                  <EyeOff className="w-3 h-3 inline mr-1" />
                )}
                {model.name}
              </button>
            ))}
          </div>
        </div>
      </Card>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Models Compared"
          value={selectedModels.length.toString()}
          icon={<Cpu className="w-5 h-5" />}
          trend={0}
          className="bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20"
        />
        
        <MetricCard
          title="Best Accuracy"
          value={`${Math.max(...models.filter(m => selectedModels.includes(m.id)).map(m => m.accuracy * 100)).toFixed(1)}%`}
          icon={<Target className="w-5 h-5" />}
          trend={2.1}
          className="bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20"
        />
        
        <MetricCard
          title="Lowest Cost"
          value={`$${Math.min(...models.filter(m => selectedModels.includes(m.id)).map(m => m.costPerInference * 1000)).toFixed(1)}/1K`}
          icon={<DollarSign className="w-5 h-5" />}
          trend={-15.3}
          isDecreaseBetter
          className="bg-gradient-to-br from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20"
        />
        
        <MetricCard
          title="Fastest Response"
          value={`${Math.min(...models.filter(m => selectedModels.includes(m.id)).map(m => m.latency))}ms`}
          icon={<Zap className="w-5 h-5" />}
          trend={-8.7}
          isDecreaseBetter
          className="bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20"
        />
      </div>

      {/* Main Comparison View */}
      {renderComparisonView()}

      {/* Optimization Recommendations */}
      <Card className="p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <Award className="w-5 h-5 text-orange-600" />
            AI-Powered Optimization Recommendations
          </h3>
          <Button variant="outline" size="sm">
            Generate More
          </Button>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {recommendations.map((rec) => {
            const model = models.find(m => m.id === rec.modelId);
            return (
              <div key={rec.id} className="border border-neutral-200 dark:border-neutral-700 rounded-lg p-4">
                <div className="flex items-start justify-between mb-2">
                  <div>
                    <h4 className="font-medium text-neutral-900 dark:text-white">
                      {rec.title}
                    </h4>
                    <p className="text-sm text-neutral-600 dark:text-neutral-400">
                      {model?.name}
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className={`px-2 py-1 text-xs rounded-full ${
                      rec.impact === 'high' 
                        ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300'
                        : rec.impact === 'medium'
                          ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300'
                          : 'bg-gray-100 text-gray-800 dark:bg-gray-900/30 dark:text-gray-300'
                    }`}>
                      {rec.impact} impact
                    </span>
                    <span className={`px-2 py-1 text-xs rounded-full ${
                      rec.effort === 'low' 
                        ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300'
                        : rec.effort === 'medium'
                          ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300'
                          : 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300'
                    }`}>
                      {rec.effort} effort
                    </span>
                  </div>
                </div>
                <p className="text-sm text-neutral-700 dark:text-neutral-300 mb-3">
                  {rec.description}
                </p>
                <div className="flex items-center justify-between">
                  <div className="text-sm">
                    <span className="text-neutral-600 dark:text-neutral-400">Est. improvement: </span>
                    <span className="font-medium text-green-600 dark:text-green-400">
                      +{rec.estimatedImprovement}%
                    </span>
                  </div>
                  <Button variant="outline" size="sm">
                    Apply
                  </Button>
                </div>
              </div>
            );
          })}
        </div>
      </Card>

      {/* Live Benchmarking Status */}
      {isLiveBenchmarking && (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="fixed bottom-6 right-6 max-w-md"
        >
          <Card className="p-4 bg-gradient-to-br from-blue-500 to-purple-500 text-white border-none shadow-2xl">
            <div className="flex items-center gap-2 mb-2">
              <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
              <span className="font-semibold">Live Benchmarking Active</span>
            </div>
            <p className="text-sm text-blue-100">
              Running continuous performance tests. Next benchmark in {Math.floor(Math.random() * 60) + 30}s.
            </p>
          </Card>
        </motion.div>
      )}
    </div>
  );
};

export default AdvancedModelComparison;