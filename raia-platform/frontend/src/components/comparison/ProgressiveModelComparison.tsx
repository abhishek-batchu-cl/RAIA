import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Plus, X, BarChart3, TrendingUp, TrendingDown, 
  Target, Activity, Brain, Zap, Award, AlertCircle,
  Download, Settings, Maximize2, Play, Pause, RotateCcw
} from 'lucide-react';
import Card from '@/components/common/Card';
import Button from '@/components/common/Button';
import { apiClient } from '@/services/api';

interface ModelMetrics {
  model_id: string;
  model_name: string;
  model_type: 'classification' | 'regression';
  accuracy: number;
  precision?: number;
  recall?: number;
  f1_score?: number;
  mse?: number;
  mae?: number;
  rmse?: number;
  auc_roc?: number;
  training_time: number;
  inference_time: number;
  data_size: number;
  feature_count: number;
  last_updated: Date;
  performance_trend: Array<{date: string; value: number}>;
  business_impact: {
    revenue_impact: number;
    cost_savings: number;
    roi: number;
  };
  status: 'active' | 'training' | 'testing' | 'deprecated';
}

interface ComparisonResult {
  winner: string;
  metrics: {
    [key: string]: {
      model_a: number;
      model_b: number;
      winner: string;
      improvement: number;
    };
  };
  overall_score: {
    model_a: number;
    model_b: number;
  };
  recommendation: string;
}

const ProgressiveModelComparison: React.FC = () => {
  const [selectedModels, setSelectedModels] = useState<ModelMetrics[]>([]);
  const [availableModels, setAvailableModels] = useState<ModelMetrics[]>([]);
  const [comparisonResults, setComparisonResults] = useState<ComparisonResult[]>([]);
  const [isComparing, setIsComparing] = useState(false);
  const [showAnimation, setShowAnimation] = useState(false);
  const [viewMode, setViewMode] = useState<'metrics' | 'trends' | 'business'>('metrics');
  const [isRacing, setIsRacing] = useState(false);
  const [raceProgress, setRaceProgress] = useState<{[key: string]: number}>({});

  // Mock available models
  const mockModels: ModelMetrics[] = [
    {
      model_id: 'credit-scoring-v2',
      model_name: 'Credit Scoring V2',
      model_type: 'classification',
      accuracy: 0.847,
      precision: 0.823,
      recall: 0.871,
      f1_score: 0.846,
      auc_roc: 0.892,
      training_time: 245,
      inference_time: 12,
      data_size: 150000,
      feature_count: 28,
      last_updated: new Date(Date.now() - 2 * 60 * 60 * 1000),
      performance_trend: [
        {date: '2024-01', value: 0.832},
        {date: '2024-02', value: 0.841},
        {date: '2024-03', value: 0.847}
      ],
      business_impact: {
        revenue_impact: 1850000,
        cost_savings: 320000,
        roi: 0.15
      },
      status: 'active'
    },
    {
      model_id: 'credit-scoring-v3',
      model_name: 'Credit Scoring V3 (Beta)',
      model_type: 'classification',
      accuracy: 0.893,
      precision: 0.867,
      recall: 0.912,
      f1_score: 0.889,
      auc_roc: 0.925,
      training_time: 312,
      inference_time: 18,
      data_size: 200000,
      feature_count: 35,
      last_updated: new Date(Date.now() - 30 * 60 * 1000),
      performance_trend: [
        {date: '2024-01', value: 0.856},
        {date: '2024-02', value: 0.878},
        {date: '2024-03', value: 0.893}
      ],
      business_impact: {
        revenue_impact: 2200000,
        cost_savings: 450000,
        roi: 0.22
      },
      status: 'testing'
    },
    {
      model_id: 'fraud-detection-v1',
      model_name: 'Fraud Detection V1',
      model_type: 'classification',
      accuracy: 0.934,
      precision: 0.912,
      recall: 0.889,
      f1_score: 0.900,
      auc_roc: 0.967,
      training_time: 180,
      inference_time: 8,
      data_size: 80000,
      feature_count: 22,
      last_updated: new Date(Date.now() - 4 * 60 * 60 * 1000),
      performance_trend: [
        {date: '2024-01', value: 0.923},
        {date: '2024-02', value: 0.928},
        {date: '2024-03', value: 0.934}
      ],
      business_impact: {
        revenue_impact: 650000,
        cost_savings: 180000,
        roi: 0.08
      },
      status: 'active'
    }
  ];

  useEffect(() => {
    loadAvailableModels();
  }, []);

  const loadAvailableModels = async () => {
    try {
      // In production, this would call the API
      // const response = await apiClient.listModels();
      setAvailableModels(mockModels);
    } catch (error) {
      console.error('Error loading models:', error);
      setAvailableModels(mockModels); // Fallback to mock data
    }
  };

  const addModelToComparison = (model: ModelMetrics) => {
    if (selectedModels.length < 4 && !selectedModels.some(m => m.model_id === model.model_id)) {
      setSelectedModels([...selectedModels, model]);
    }
  };

  const removeModelFromComparison = (modelId: string) => {
    setSelectedModels(selectedModels.filter(m => m.model_id !== modelId));
  };

  const startComparison = () => {
    if (selectedModels.length < 2) return;
    
    setIsComparing(true);
    setShowAnimation(true);
    
    // Simulate comparison calculation
    setTimeout(() => {
      const results = generateComparisonResults();
      setComparisonResults(results);
      setIsComparing(false);
    }, 2000);
  };

  const startRace = () => {
    if (selectedModels.length < 2) return;
    
    setIsRacing(true);
    const progress: {[key: string]: number} = {};
    
    selectedModels.forEach(model => {
      progress[model.model_id] = 0;
    });
    setRaceProgress(progress);
    
    // Animate race progress
    const raceInterval = setInterval(() => {
      setRaceProgress(prev => {
        const newProgress = { ...prev };
        let raceComplete = false;
        
        selectedModels.forEach(model => {
          if (newProgress[model.model_id] < 100) {
            // Speed based on model accuracy (winner bias)
            const speedMultiplier = model.accuracy;
            newProgress[model.model_id] = Math.min(100, 
              newProgress[model.model_id] + (Math.random() * 5 * speedMultiplier)
            );
          }
          
          if (newProgress[model.model_id] >= 100) {
            raceComplete = true;
          }
        });
        
        if (raceComplete) {
          clearInterval(raceInterval);
          setTimeout(() => setIsRacing(false), 2000);
        }
        
        return newProgress;
      });
    }, 100);
  };

  const generateComparisonResults = (): ComparisonResult[] => {
    const results: ComparisonResult[] = [];
    
    for (let i = 0; i < selectedModels.length - 1; i++) {
      for (let j = i + 1; j < selectedModels.length; j++) {
        const modelA = selectedModels[i];
        const modelB = selectedModels[j];
        
        const metrics: {[key: string]: any} = {};
        
        // Compare accuracy
        metrics.accuracy = {
          model_a: modelA.accuracy,
          model_b: modelB.accuracy,
          winner: modelA.accuracy > modelB.accuracy ? modelA.model_name : modelB.model_name,
          improvement: Math.abs(modelA.accuracy - modelB.accuracy)
        };
        
        // Compare precision (if available)
        if (modelA.precision && modelB.precision) {
          metrics.precision = {
            model_a: modelA.precision,
            model_b: modelB.precision,
            winner: modelA.precision > modelB.precision ? modelA.model_name : modelB.model_name,
            improvement: Math.abs(modelA.precision - modelB.precision)
          };
        }
        
        // Compare inference time
        metrics.inference_time = {
          model_a: modelA.inference_time,
          model_b: modelB.inference_time,
          winner: modelA.inference_time < modelB.inference_time ? modelA.model_name : modelB.model_name,
          improvement: Math.abs(modelA.inference_time - modelB.inference_time) / Math.min(modelA.inference_time, modelB.inference_time)
        };
        
        // Compare business impact
        metrics.roi = {
          model_a: modelA.business_impact.roi,
          model_b: modelB.business_impact.roi,
          winner: modelA.business_impact.roi > modelB.business_impact.roi ? modelA.model_name : modelB.model_name,
          improvement: Math.abs(modelA.business_impact.roi - modelB.business_impact.roi)
        };
        
        // Calculate overall scores
        const scoreA = (modelA.accuracy * 0.4) + (modelA.business_impact.roi * 0.3) + ((100 - modelA.inference_time) / 100 * 0.3);
        const scoreB = (modelB.accuracy * 0.4) + (modelB.business_impact.roi * 0.3) + ((100 - modelB.inference_time) / 100 * 0.3);
        
        const winner = scoreA > scoreB ? modelA : modelB;
        const recommendation = `Based on overall performance, ${winner.model_name} shows superior results with ${((winner === modelA ? scoreA : scoreB) * 100).toFixed(1)}% effectiveness score. ${winner === modelA && modelA.status === 'testing' ? 'Consider promoting to production.' : winner === modelB && modelB.status === 'testing' ? 'Consider promoting to production.' : 'Current production model is performing well.'}`;
        
        results.push({
          winner: winner.model_name,
          metrics,
          overall_score: {
            model_a: scoreA,
            model_b: scoreB
          },
          recommendation
        });
      }
    }
    
    return results;
  };

  const getModelStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200';
      case 'training': return 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200';
      case 'testing': return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200';
      case 'deprecated': return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200';
      default: return 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200';
    }
  };

  const getMetricIcon = (metric: string) => {
    switch (metric) {
      case 'accuracy': return <Target className="w-4 h-4" />;
      case 'precision': return <Zap className="w-4 h-4" />;
      case 'recall': return <Activity className="w-4 h-4" />;
      case 'f1_score': return <Award className="w-4 h-4" />;
      case 'inference_time': return <TrendingUp className="w-4 h-4" />;
      case 'roi': return <TrendingUp className="w-4 h-4" />;
      default: return <BarChart3 className="w-4 h-4" />;
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-neutral-900 dark:text-neutral-100 flex items-center">
            <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-500 rounded-lg flex items-center justify-center mr-3">
              <BarChart3 className="w-5 h-5 text-white" />
            </div>
            Progressive Model Comparison
          </h1>
          <p className="text-neutral-600 dark:text-neutral-400 mt-1">
            Compare models side-by-side with interactive analysis and performance racing
          </p>
        </div>
        
        <div className="flex items-center space-x-2">
          <Button
            variant="outline"
            size="sm"
            leftIcon={<Settings className="w-4 h-4" />}
          >
            Configure
          </Button>
          <Button
            variant="outline"
            size="sm"
            leftIcon={<Download className="w-4 h-4" />}
          >
            Export
          </Button>
        </div>
      </div>

      {/* View Mode Toggle */}
      <div className="flex items-center justify-center">
        <div className="bg-neutral-100 dark:bg-neutral-800 rounded-lg p-1 flex">
          {[
            { id: 'metrics', label: 'Metrics', icon: <BarChart3 className="w-4 h-4" /> },
            { id: 'trends', label: 'Trends', icon: <TrendingUp className="w-4 h-4" /> },
            { id: 'business', label: 'Business', icon: <Target className="w-4 h-4" /> },
          ].map((mode) => (
            <button
              key={mode.id}
              onClick={() => setViewMode(mode.id as any)}
              className={`flex items-center space-x-2 px-4 py-2 rounded-md text-sm font-medium transition-all duration-200 ${
                viewMode === mode.id
                  ? 'bg-white dark:bg-neutral-700 text-primary-600 dark:text-primary-400 shadow-sm'
                  : 'text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-neutral-100'
              }`}
            >
              {mode.icon}
              <span>{mode.label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Model Selection */}
      <Card title="Select Models to Compare" icon={<Plus className="w-5 h-5 text-primary-500" />}>
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {availableModels.map((model) => (
              <motion.div
                key={model.model_id}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className={`p-4 border rounded-lg cursor-pointer transition-all duration-200 ${
                  selectedModels.some(m => m.model_id === model.model_id)
                    ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20 shadow-sm'
                    : 'border-neutral-200 dark:border-neutral-700 hover:border-primary-300 dark:hover:border-primary-700'
                }`}
                onClick={() => addModelToComparison(model)}
              >
                <div className="flex items-start justify-between mb-2">
                  <h3 className="font-semibold text-neutral-900 dark:text-neutral-100">
                    {model.model_name}
                  </h3>
                  <span className={`px-2 py-1 text-xs rounded-full font-medium ${getModelStatusColor(model.status)}`}>
                    {model.status}
                  </span>
                </div>
                
                <div className="space-y-2 text-sm text-neutral-600 dark:text-neutral-400">
                  <div className="flex justify-between">
                    <span>Accuracy:</span>
                    <span className="font-medium">{(model.accuracy * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Inference:</span>
                    <span className="font-medium">{model.inference_time}ms</span>
                  </div>
                  <div className="flex justify-between">
                    <span>ROI:</span>
                    <span className="font-medium">{(model.business_impact.roi * 100).toFixed(1)}%</span>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
          
          {/* Selected Models */}
          {selectedModels.length > 0 && (
            <div className="mt-6">
              <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-3">
                Selected Models ({selectedModels.length}/4)
              </h3>
              <div className="flex flex-wrap gap-2">
                <AnimatePresence>
                  {selectedModels.map((model) => (
                    <motion.div
                      key={model.model_id}
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: 0.8 }}
                      className="flex items-center space-x-2 bg-primary-100 dark:bg-primary-900/20 text-primary-800 dark:text-primary-200 px-3 py-1 rounded-full text-sm font-medium"
                    >
                      <span>{model.model_name}</span>
                      <button
                        onClick={() => removeModelFromComparison(model.model_id)}
                        className="text-primary-600 dark:text-primary-400 hover:text-primary-800 dark:hover:text-primary-200"
                      >
                        <X className="w-3 h-3" />
                      </button>
                    </motion.div>
                  ))}
                </AnimatePresence>
              </div>
              
              <div className="flex items-center space-x-3 mt-4">
                <Button
                  variant="primary"
                  onClick={startComparison}
                  disabled={selectedModels.length < 2 || isComparing}
                  leftIcon={isComparing ? <Brain className="w-4 h-4 animate-pulse" /> : <BarChart3 className="w-4 h-4" />}
                >
                  {isComparing ? 'Analyzing...' : 'Compare Models'}
                </Button>
                
                <Button
                  variant="outline"
                  onClick={startRace}
                  disabled={selectedModels.length < 2 || isRacing}
                  leftIcon={isRacing ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                >
                  {isRacing ? 'Racing...' : 'Start Race'}
                </Button>
                
                <Button
                  variant="ghost"
                  onClick={() => {
                    setSelectedModels([]);
                    setComparisonResults([]);
                    setRaceProgress({});
                  }}
                  leftIcon={<RotateCcw className="w-4 h-4" />}
                >
                  Reset
                </Button>
              </div>
            </div>
          )}
        </div>
      </Card>

      {/* Performance Race Animation */}
      <AnimatePresence>
        {isRacing && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
          >
            <Card title="Model Performance Race" icon={<Play className="w-5 h-5 text-green-500" />}>
              <div className="space-y-4">
                {selectedModels.map((model, index) => (
                  <div key={model.model_id} className="space-y-2">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        <div className={`w-8 h-8 rounded-full flex items-center justify-center text-white font-bold ${
                          index === 0 ? 'bg-blue-500' :
                          index === 1 ? 'bg-green-500' :
                          index === 2 ? 'bg-purple-500' : 'bg-orange-500'
                        }`}>
                          {index + 1}
                        </div>
                        <span className="font-medium text-neutral-900 dark:text-neutral-100">
                          {model.model_name}
                        </span>
                      </div>
                      <span className="text-sm font-medium text-neutral-600 dark:text-neutral-400">
                        {(raceProgress[model.model_id] || 0).toFixed(1)}%
                      </span>
                    </div>
                    
                    <div className="relative h-6 bg-neutral-200 dark:bg-neutral-700 rounded-full overflow-hidden">
                      <motion.div
                        className={`h-full rounded-full ${
                          index === 0 ? 'bg-gradient-to-r from-blue-500 to-blue-600' :
                          index === 1 ? 'bg-gradient-to-r from-green-500 to-green-600' :
                          index === 2 ? 'bg-gradient-to-r from-purple-500 to-purple-600' : 
                          'bg-gradient-to-r from-orange-500 to-orange-600'
                        }`}
                        style={{ width: `${raceProgress[model.model_id] || 0}%` }}
                        transition={{ duration: 0.1, ease: 'easeOut' }}
                      />
                      
                      {raceProgress[model.model_id] >= 100 && (
                        <motion.div
                          initial={{ scale: 0 }}
                          animate={{ scale: 1 }}
                          className="absolute right-2 top-1/2 transform -translate-y-1/2"
                        >
                          <Award className="w-4 h-4 text-yellow-500" />
                        </motion.div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Comparison Results */}
      <AnimatePresence>
        {comparisonResults.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="space-y-6"
          >
            {comparisonResults.map((result, index) => (
              <Card key={index} title="Comparison Results" icon={<TrendingUp className="w-5 h-5 text-primary-500" />}>
                <div className="space-y-6">
                  {/* Metrics Comparison */}
                  {viewMode === 'metrics' && (
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                      {Object.entries(result.metrics).map(([metricName, data]) => (
                        <div key={metricName} className="space-y-3">
                          <div className="flex items-center space-x-2">
                            {getMetricIcon(metricName)}
                            <h4 className="font-semibold text-neutral-900 dark:text-neutral-100 capitalize">
                              {metricName.replace('_', ' ')}
                            </h4>
                          </div>
                          
                          <div className="space-y-2">
                            <div className="flex items-center justify-between">
                              <span className="text-sm text-neutral-600 dark:text-neutral-400">Model A</span>
                              <span className={`font-medium ${
                                data.winner.includes('A') || data.model_a > data.model_b ? 
                                'text-green-600 dark:text-green-400' : 'text-neutral-600 dark:text-neutral-400'
                              }`}>
                                {typeof data.model_a === 'number' ? 
                                  (metricName === 'inference_time' ? `${data.model_a}ms` : 
                                   metricName === 'roi' ? `${(data.model_a * 100).toFixed(1)}%` :
                                   `${(data.model_a * 100).toFixed(1)}%`) : 
                                  data.model_a}
                              </span>
                            </div>
                            
                            <div className="flex items-center justify-between">
                              <span className="text-sm text-neutral-600 dark:text-neutral-400">Model B</span>
                              <span className={`font-medium ${
                                data.winner.includes('B') || data.model_b > data.model_a ? 
                                'text-green-600 dark:text-green-400' : 'text-neutral-600 dark:text-neutral-400'
                              }`}>
                                {typeof data.model_b === 'number' ? 
                                  (metricName === 'inference_time' ? `${data.model_b}ms` : 
                                   metricName === 'roi' ? `${(data.model_b * 100).toFixed(1)}%` :
                                   `${(data.model_b * 100).toFixed(1)}%`) : 
                                  data.model_b}
                              </span>
                            </div>
                            
                            <div className="pt-2 border-t border-neutral-200 dark:border-neutral-700">
                              <div className="flex items-center justify-between">
                                <span className="text-sm font-medium text-neutral-700 dark:text-neutral-300">Winner:</span>
                                <span className="font-bold text-green-600 dark:text-green-400">
                                  {data.winner}
                                </span>
                              </div>
                              <div className="text-xs text-neutral-500 dark:text-neutral-400 mt-1">
                                Improvement: {(data.improvement * 100).toFixed(1)}%
                              </div>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Overall Recommendation */}
                  <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
                    <div className="flex items-start space-x-3">
                      <Brain className="w-6 h-6 text-blue-500 flex-shrink-0 mt-1" />
                      <div>
                        <h4 className="font-semibold text-blue-900 dark:text-blue-100 mb-2">
                          AI Recommendation
                        </h4>
                        <p className="text-blue-800 dark:text-blue-200 text-sm leading-relaxed">
                          {result.recommendation}
                        </p>
                        
                        <div className="mt-3 flex items-center space-x-4 text-xs">
                          <div className="flex items-center space-x-2">
                            <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                            <span className="text-blue-700 dark:text-blue-300">
                              Model A Score: {(result.overall_score.model_a * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="flex items-center space-x-2">
                            <div className="w-3 h-3 bg-purple-500 rounded-full"></div>
                            <span className="text-blue-700 dark:text-blue-300">
                              Model B Score: {(result.overall_score.model_b * 100).toFixed(1)}%
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </Card>
            ))}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Empty State */}
      {selectedModels.length === 0 && (
        <Card>
          <div className="text-center py-12">
            <BarChart3 className="w-16 h-16 text-neutral-400 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-neutral-900 dark:text-neutral-100 mb-2">
              Ready to Compare Models
            </h3>
            <p className="text-neutral-600 dark:text-neutral-400 mb-6">
              Select 2-4 models above to start comparing their performance, trends, and business impact.
            </p>
            <div className="flex justify-center space-x-2">
              <div className="w-3 h-3 bg-blue-500 rounded-full animate-bounce"></div>
              <div className="w-3 h-3 bg-green-500 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
              <div className="w-3 h-3 bg-purple-500 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
            </div>
          </div>
        </Card>
      )}
    </div>
  );
};

export default ProgressiveModelComparison;