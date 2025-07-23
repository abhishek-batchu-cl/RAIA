import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  TrendingUp, TrendingDown, Calendar, BarChart3, LineChart,
  Clock, Target, AlertTriangle, CheckCircle, Settings,
  Play, Pause, FastForward, Rewind, Download, Upload,
  Brain, Zap, Activity, Database, Filter, RefreshCw,
  ArrowUp, ArrowDown, Minus, Info, Eye, EyeOff,
  PieChart, Layers, Globe, MapPin, Users, Package
} from 'lucide-react';

interface TimeSeriesData {
  timestamp: string;
  value: number;
  predicted?: number;
  confidence_lower?: number;
  confidence_upper?: number;
  anomaly_score?: number;
  is_anomaly?: boolean;
}

interface TimeSeriesForecast {
  id: string;
  name: string;
  description: string;
  data_source: string;
  target_variable: string;
  frequency: 'hourly' | 'daily' | 'weekly' | 'monthly';
  horizon: number; // forecast periods ahead
  model_type: 'arima' | 'prophet' | 'lstm' | 'transformer' | 'ensemble';
  accuracy_metrics: {
    mae: number;
    mape: number;
    rmse: number;
    r2: number;
  };
  status: 'training' | 'ready' | 'forecasting' | 'error';
  created_at: string;
  last_updated: string;
  next_forecast: string;
  historical_data: TimeSeriesData[];
  forecast_data: TimeSeriesData[];
  seasonality?: {
    daily?: number;
    weekly?: number;
    monthly?: number;
    yearly?: number;
  };
  trends?: {
    current_trend: 'increasing' | 'decreasing' | 'stable';
    trend_strength: number;
    change_points: string[];
  };
  anomalies?: {
    count: number;
    severity: 'low' | 'medium' | 'high';
    recent_anomalies: TimeSeriesData[];
  };
}

interface ForecastingJob {
  id: string;
  forecast_id: string;
  forecast_name: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  progress: number;
  started_at: string;
  completed_at?: string;
  error_message?: string;
  model_config: {
    model_type: string;
    parameters: Record<string, any>;
  };
}

interface TimeSeriesInsight {
  id: string;
  type: 'trend' | 'seasonality' | 'anomaly' | 'forecast' | 'correlation';
  title: string;
  description: string;
  importance: 'low' | 'medium' | 'high' | 'critical';
  confidence: number;
  data_points: any[];
  recommended_actions: string[];
  created_at: string;
}

const TimeSeriesAnalytics: React.FC = () => {
  const [forecasts, setForecasts] = useState<TimeSeriesForecast[]>([]);
  const [selectedForecast, setSelectedForecast] = useState<TimeSeriesForecast | null>(null);
  const [activeJobs, setActiveJobs] = useState<ForecastingJob[]>([]);
  const [insights, setInsights] = useState<TimeSeriesInsight[]>([]);
  
  const [activeTab, setActiveTab] = useState<'overview' | 'forecasts' | 'insights' | 'anomalies' | 'config'>('overview');
  const [timeRange, setTimeRange] = useState<'1d' | '7d' | '30d' | '90d' | '1y'>('30d');
  const [showConfidenceIntervals, setShowConfidenceIntervals] = useState(true);
  const [showAnomalies, setShowAnomalies] = useState(true);
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  // Mock data
  useEffect(() => {
    const generateTimeSeriesData = (days: number, baseValue: number = 100, trend: number = 0.1, noise: number = 10) => {
      const data: TimeSeriesData[] = [];
      const now = new Date();
      
      for (let i = days; i >= 0; i--) {
        const date = new Date(now.getTime() - i * 24 * 60 * 60 * 1000);
        const trendValue = baseValue + trend * (days - i);
        const seasonality = Math.sin((days - i) * 2 * Math.PI / 30) * 5; // Monthly cycle
        const weeklyPattern = Math.sin((days - i) * 2 * Math.PI / 7) * 3; // Weekly cycle
        const noiseValue = (Math.random() - 0.5) * noise;
        
        const value = trendValue + seasonality + weeklyPattern + noiseValue;
        const predicted = trendValue + seasonality + weeklyPattern;
        
        // Add some anomalies
        const isAnomaly = Math.random() < 0.05; // 5% chance
        const anomalyValue = isAnomaly ? value + (Math.random() > 0.5 ? 30 : -30) : value;
        
        data.push({
          timestamp: date.toISOString(),
          value: anomalyValue,
          predicted: predicted,
          confidence_lower: predicted - 15,
          confidence_upper: predicted + 15,
          is_anomaly: isAnomaly,
          anomaly_score: isAnomaly ? Math.random() * 0.4 + 0.6 : Math.random() * 0.3
        });
      }
      return data;
    };

    const generateForecastData = (days: number, lastValue: number) => {
      const data: TimeSeriesData[] = [];
      const now = new Date();
      
      for (let i = 1; i <= days; i++) {
        const date = new Date(now.getTime() + i * 24 * 60 * 60 * 1000);
        const trend = 0.1 * i;
        const seasonality = Math.sin(i * 2 * Math.PI / 30) * 5;
        const weeklyPattern = Math.sin(i * 2 * Math.PI / 7) * 3;
        
        const predicted = lastValue + trend + seasonality + weeklyPattern;
        const uncertainty = Math.min(5 + i * 0.5, 20); // Increasing uncertainty
        
        data.push({
          timestamp: date.toISOString(),
          value: predicted,
          predicted: predicted,
          confidence_lower: predicted - uncertainty,
          confidence_upper: predicted + uncertainty
        });
      }
      return data;
    };

    const mockForecasts: TimeSeriesForecast[] = [
      {
        id: 'forecast_001',
        name: 'Revenue Forecasting',
        description: 'Daily revenue prediction for the next 30 days with seasonal adjustments',
        data_source: 'revenue_database',
        target_variable: 'daily_revenue',
        frequency: 'daily',
        horizon: 30,
        model_type: 'prophet',
        accuracy_metrics: {
          mae: 1250.45,
          mape: 8.3,
          rmse: 1890.22,
          r2: 0.87
        },
        status: 'ready',
        created_at: '2024-01-10T09:00:00Z',
        last_updated: '2024-01-16T14:30:00Z',
        next_forecast: '2024-01-17T09:00:00Z',
        historical_data: generateTimeSeriesData(90, 15000, 25),
        forecast_data: generateForecastData(30, 15000),
        seasonality: {
          daily: 0.15,
          weekly: 0.45,
          monthly: 0.25,
          yearly: 0.8
        },
        trends: {
          current_trend: 'increasing',
          trend_strength: 0.7,
          change_points: ['2024-01-05T00:00:00Z', '2024-01-12T00:00:00Z']
        },
        anomalies: {
          count: 4,
          severity: 'medium',
          recent_anomalies: []
        }
      },
      {
        id: 'forecast_002',
        name: 'User Engagement Prediction',
        description: 'Hourly active user count forecast with event impact analysis',
        data_source: 'user_analytics',
        target_variable: 'active_users',
        frequency: 'hourly',
        horizon: 168, // 7 days in hours
        model_type: 'lstm',
        accuracy_metrics: {
          mae: 45.2,
          mape: 12.1,
          rmse: 67.8,
          r2: 0.82
        },
        status: 'forecasting',
        created_at: '2024-01-12T16:20:00Z',
        last_updated: '2024-01-16T15:45:00Z',
        next_forecast: '2024-01-16T17:00:00Z',
        historical_data: generateTimeSeriesData(30, 850, 5),
        forecast_data: generateForecastData(7, 850),
        seasonality: {
          daily: 0.6,
          weekly: 0.4
        },
        trends: {
          current_trend: 'stable',
          trend_strength: 0.3,
          change_points: ['2024-01-08T00:00:00Z']
        },
        anomalies: {
          count: 2,
          severity: 'low',
          recent_anomalies: []
        }
      },
      {
        id: 'forecast_003',
        name: 'Inventory Demand Forecasting',
        description: 'Weekly inventory demand prediction with supply chain optimization',
        data_source: 'inventory_system',
        target_variable: 'units_demanded',
        frequency: 'weekly',
        horizon: 12,
        model_type: 'ensemble',
        accuracy_metrics: {
          mae: 234.7,
          mape: 15.4,
          rmse: 312.5,
          r2: 0.79
        },
        status: 'training',
        created_at: '2024-01-15T11:30:00Z',
        last_updated: '2024-01-16T12:15:00Z',
        next_forecast: '2024-01-17T12:00:00Z',
        historical_data: generateTimeSeriesData(52, 2400, 15),
        forecast_data: generateForecastData(12, 2400),
        seasonality: {
          monthly: 0.3,
          yearly: 0.7
        },
        trends: {
          current_trend: 'decreasing',
          trend_strength: 0.4,
          change_points: ['2024-01-01T00:00:00Z']
        },
        anomalies: {
          count: 7,
          severity: 'high',
          recent_anomalies: []
        }
      }
    ];

    const mockJobs: ForecastingJob[] = [
      {
        id: 'job_001',
        forecast_id: 'forecast_003',
        forecast_name: 'Inventory Demand Forecasting',
        status: 'running',
        progress: 65,
        started_at: '2024-01-16T14:30:00Z',
        model_config: {
          model_type: 'ensemble',
          parameters: {
            'models': ['arima', 'prophet', 'lstm'],
            'ensemble_method': 'weighted_average',
            'cross_validation_folds': 5
          }
        }
      },
      {
        id: 'job_002',
        forecast_id: 'forecast_002',
        forecast_name: 'User Engagement Prediction',
        status: 'queued',
        progress: 0,
        started_at: '2024-01-16T15:45:00Z',
        model_config: {
          model_type: 'lstm',
          parameters: {
            'lstm_units': 50,
            'dropout_rate': 0.2,
            'epochs': 100
          }
        }
      }
    ];

    const mockInsights: TimeSeriesInsight[] = [
      {
        id: 'insight_001',
        type: 'trend',
        title: 'Strong Upward Revenue Trend Detected',
        description: 'Revenue has increased by 15% over the past 30 days with high statistical significance',
        importance: 'high',
        confidence: 0.92,
        data_points: [
          { metric: 'trend_strength', value: 0.7 },
          { metric: 'p_value', value: 0.001 },
          { metric: 'growth_rate', value: 15.3 }
        ],
        recommended_actions: [
          'Consider scaling marketing efforts to capitalize on growth',
          'Review inventory levels to meet increased demand',
          'Analyze attribution sources for this trend'
        ],
        created_at: '2024-01-16T10:30:00Z'
      },
      {
        id: 'insight_002',
        type: 'anomaly',
        title: 'Unusual User Activity Spike Detected',
        description: 'Active user count exceeded normal ranges by 200% on January 15th',
        importance: 'critical',
        confidence: 0.98,
        data_points: [
          { metric: 'anomaly_score', value: 0.95 },
          { metric: 'deviation_from_normal', value: 2.1 },
          { metric: 'duration_hours', value: 6 }
        ],
        recommended_actions: [
          'Investigate potential viral content or marketing campaign impact',
          'Check system capacity and performance during spike',
          'Document learnings for future capacity planning'
        ],
        created_at: '2024-01-16T08:45:00Z'
      },
      {
        id: 'insight_003',
        type: 'seasonality',
        title: 'Weekly Seasonality Pattern Confirmed',
        description: 'Strong weekly patterns identified with 40% higher activity on weekends',
        importance: 'medium',
        confidence: 0.85,
        data_points: [
          { metric: 'seasonality_strength', value: 0.4 },
          { metric: 'weekend_boost', value: 40.2 },
          { metric: 'pattern_consistency', value: 0.85 }
        ],
        recommended_actions: [
          'Adjust staffing levels for weekend peaks',
          'Optimize marketing spend allocation by day of week',
          'Consider weekend-specific product offerings'
        ],
        created_at: '2024-01-16T07:20:00Z'
      }
    ];

    setForecasts(mockForecasts);
    setSelectedForecast(mockForecasts[0]);
    setActiveJobs(mockJobs);
    setInsights(mockInsights);
    setIsLoading(false);
  }, []);

  const handleCreateForecast = useCallback(() => {
    // Implement forecast creation
    console.log('Creating new forecast...');
  }, []);

  const handleRefreshForecast = useCallback((forecastId: string) => {
    // Implement forecast refresh
    console.log('Refreshing forecast:', forecastId);
  }, []);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'ready': case 'completed': return 'text-green-600 bg-green-50';
      case 'training': case 'running': case 'forecasting': return 'text-blue-600 bg-blue-50';
      case 'queued': return 'text-yellow-600 bg-yellow-50';
      case 'error': case 'failed': return 'text-red-600 bg-red-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'increasing': return <TrendingUp className="w-4 h-4 text-green-500" />;
      case 'decreasing': return <TrendingDown className="w-4 h-4 text-red-500" />;
      case 'stable': return <Minus className="w-4 h-4 text-gray-500" />;
      default: return <Activity className="w-4 h-4 text-gray-500" />;
    }
  };

  const getImportanceColor = (importance: string) => {
    switch (importance) {
      case 'critical': return 'text-red-600 bg-red-50 border-red-200';
      case 'high': return 'text-orange-600 bg-orange-50 border-orange-200';
      case 'medium': return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      case 'low': return 'text-blue-600 bg-blue-50 border-blue-200';
      default: return 'text-gray-600 bg-gray-50 border-gray-200';
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
              <p className="text-sm text-gray-600">Active Forecasts</p>
              <p className="text-3xl font-bold text-blue-600">
                {forecasts.filter(f => f.status === 'ready' || f.status === 'forecasting').length}
              </p>
            </div>
            <BarChart3 className="w-8 h-8 text-blue-500" />
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
              <p className="text-sm text-gray-600">Critical Insights</p>
              <p className="text-3xl font-bold text-orange-600">
                {insights.filter(i => i.importance === 'critical' || i.importance === 'high').length}
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
              <p className="text-sm text-gray-600">Avg Accuracy</p>
              <p className="text-3xl font-bold text-purple-600">
                {Math.round(forecasts.reduce((acc, f) => acc + f.accuracy_metrics.r2, 0) / forecasts.length * 100)}%
              </p>
            </div>
            <Target className="w-8 h-8 text-purple-500" />
          </div>
        </motion.div>
      </div>

      {/* Main Forecast Chart */}
      <div className="lg:col-span-2 bg-white rounded-lg shadow-sm border border-gray-200">
        <div className="p-6 border-b border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold text-gray-900">
                {selectedForecast?.name || 'Select a Forecast'}
              </h3>
              {selectedForecast && (
                <p className="text-sm text-gray-600 mt-1">{selectedForecast.description}</p>
              )}
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setShowConfidenceIntervals(!showConfidenceIntervals)}
                className={`text-sm px-3 py-1 rounded-full transition-colors ${
                  showConfidenceIntervals ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-600'
                }`}
              >
                Confidence Bands
              </button>
              <button
                onClick={() => setShowAnomalies(!showAnomalies)}
                className={`text-sm px-3 py-1 rounded-full transition-colors ${
                  showAnomalies ? 'bg-red-100 text-red-700' : 'bg-gray-100 text-gray-600'
                }`}
              >
                Anomalies
              </button>
            </div>
          </div>
        </div>
        
        <div className="p-6">
          {selectedForecast ? (
            <div className="h-80 bg-gray-50 rounded-lg flex items-center justify-center">
              {/* This would be replaced with actual chart component like Chart.js, D3, or Recharts */}
              <div className="text-center">
                <LineChart className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-600">Time Series Chart</p>
                <p className="text-sm text-gray-500 mt-1">
                  {selectedForecast.historical_data.length} historical points, {selectedForecast.forecast_data.length} forecast points
                </p>
                <div className="mt-4 flex items-center justify-center gap-4 text-sm">
                  <div className="flex items-center gap-1">
                    <div className="w-3 h-3 bg-blue-500 rounded"></div>
                    <span>Historical</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="w-3 h-3 bg-green-500 rounded"></div>
                    <span>Forecast</span>
                  </div>
                  {showConfidenceIntervals && (
                    <div className="flex items-center gap-1">
                      <div className="w-3 h-3 bg-green-200 rounded"></div>
                      <span>Confidence</span>
                    </div>
                  )}
                  {showAnomalies && (
                    <div className="flex items-center gap-1">
                      <div className="w-3 h-3 bg-red-500 rounded"></div>
                      <span>Anomalies</span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          ) : (
            <div className="h-80 bg-gray-50 rounded-lg flex items-center justify-center">
              <div className="text-center">
                <BarChart3 className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-600">Select a forecast to view visualization</p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Forecast List & Controls */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200">
        <div className="p-6 border-b border-gray-200">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold text-gray-900">Forecasts</h3>
            <button
              onClick={handleCreateForecast}
              className="bg-blue-600 text-white text-sm px-3 py-2 rounded-lg hover:bg-blue-700 transition-colors"
            >
              New Forecast
            </button>
          </div>
        </div>
        
        <div className="p-6">
          <div className="space-y-3">
            {forecasts.map((forecast) => (
              <motion.div
                key={forecast.id}
                className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${
                  selectedForecast?.id === forecast.id
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-gray-200 hover:border-gray-300'
                }`}
                onClick={() => setSelectedForecast(forecast)}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-medium text-gray-900">{forecast.name}</h4>
                  <div className="flex items-center gap-2">
                    <span className={`text-xs px-2 py-1 rounded-full ${getStatusColor(forecast.status)}`}>
                      {forecast.status}
                    </span>
                    {getTrendIcon(forecast.trends?.current_trend || 'stable')}
                  </div>
                </div>
                
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-gray-600">Frequency:</span>
                    <p className="font-medium capitalize">{forecast.frequency}</p>
                  </div>
                  <div>
                    <span className="text-gray-600">Horizon:</span>
                    <p className="font-medium">{forecast.horizon} {forecast.frequency === 'hourly' ? 'hours' : 'periods'}</p>
                  </div>
                  <div>
                    <span className="text-gray-600">Accuracy (R²):</span>
                    <p className="font-medium">{(forecast.accuracy_metrics.r2 * 100).toFixed(1)}%</p>
                  </div>
                  <div>
                    <span className="text-gray-600">Model:</span>
                    <p className="font-medium uppercase">{forecast.model_type}</p>
                  </div>
                </div>
                
                {forecast.anomalies && forecast.anomalies.count > 0 && (
                  <div className="mt-2 flex items-center gap-1 text-xs">
                    <AlertTriangle className="w-3 h-3 text-orange-500" />
                    <span className="text-orange-600">
                      {forecast.anomalies.count} anomalies ({forecast.anomalies.severity})
                    </span>
                  </div>
                )}
                
                <div className="mt-2 flex items-center justify-between">
                  <span className="text-xs text-gray-500">
                    Updated {new Date(forecast.last_updated).toLocaleString()}
                  </span>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleRefreshForecast(forecast.id);
                    }}
                    className="text-blue-600 hover:text-blue-800"
                  >
                    <RefreshCw className="w-4 h-4" />
                  </button>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </div>

      {/* Recent Insights */}
      <div className="lg:col-span-3 bg-white rounded-lg shadow-sm border border-gray-200">
        <div className="p-6 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900">Recent Insights</h3>
        </div>
        <div className="p-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {insights.slice(0, 6).map((insight) => (
              <motion.div
                key={insight.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className={`p-4 rounded-lg border-2 ${getImportanceColor(insight.importance)}`}
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-center gap-2">
                    {insight.type === 'trend' && <TrendingUp className="w-5 h-5" />}
                    {insight.type === 'anomaly' && <AlertTriangle className="w-5 h-5" />}
                    {insight.type === 'seasonality' && <Calendar className="w-5 h-5" />}
                    {insight.type === 'forecast' && <Target className="w-5 h-5" />}
                    {insight.type === 'correlation' && <Activity className="w-5 h-5" />}
                    <span className="text-xs font-medium uppercase tracking-wide">
                      {insight.type}
                    </span>
                  </div>
                  <span className="text-xs font-medium">
                    {(insight.confidence * 100).toFixed(0)}% confident
                  </span>
                </div>
                
                <h4 className="font-semibold text-gray-900 mb-2">{insight.title}</h4>
                <p className="text-sm text-gray-600 mb-3">{insight.description}</p>
                
                <div className="text-xs text-gray-500">
                  <p>{insight.recommended_actions.length} recommended actions</p>
                  <p>{new Date(insight.created_at).toLocaleString()}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );

  const renderForecasts = () => (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-xl font-semibold text-gray-900">Time Series Forecasts</h2>
        <div className="flex items-center gap-3">
          <select
            value={timeRange}
            onChange={(e) => setTimeRange(e.target.value as any)}
            className="px-3 py-2 border border-gray-300 rounded-lg text-sm"
          >
            <option value="1d">Last 24 Hours</option>
            <option value="7d">Last 7 Days</option>
            <option value="30d">Last 30 Days</option>
            <option value="90d">Last 90 Days</option>
            <option value="1y">Last Year</option>
          </select>
          <button
            onClick={handleCreateForecast}
            className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors flex items-center gap-2"
          >
            <Brain className="w-4 h-4" />
            Create Forecast
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {forecasts.map((forecast) => (
          <motion.div
            key={forecast.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white rounded-lg shadow-sm border border-gray-200"
          >
            <div className="p-6 border-b border-gray-200">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-lg font-medium text-gray-900">{forecast.name}</h3>
                <div className="flex items-center gap-2">
                  <span className={`text-sm px-2 py-1 rounded-full ${getStatusColor(forecast.status)}`}>
                    {forecast.status}
                  </span>
                  {getTrendIcon(forecast.trends?.current_trend || 'stable')}
                </div>
              </div>
              <p className="text-gray-600 text-sm">{forecast.description}</p>
            </div>

            <div className="p-6">
              {/* Accuracy Metrics */}
              <div className="grid grid-cols-2 gap-4 mb-6">
                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-600">
                    {(forecast.accuracy_metrics.r2 * 100).toFixed(1)}%
                  </div>
                  <div className="text-sm text-gray-600">R² Score</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-600">
                    {forecast.accuracy_metrics.mape.toFixed(1)}%
                  </div>
                  <div className="text-sm text-gray-600">MAPE</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-purple-600">
                    {forecast.accuracy_metrics.mae.toFixed(0)}
                  </div>
                  <div className="text-sm text-gray-600">MAE</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-orange-600">
                    {forecast.accuracy_metrics.rmse.toFixed(0)}
                  </div>
                  <div className="text-sm text-gray-600">RMSE</div>
                </div>
              </div>

              {/* Configuration Details */}
              <div className="space-y-3 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600">Model Type:</span>
                  <span className="font-medium uppercase">{forecast.model_type}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Frequency:</span>
                  <span className="font-medium capitalize">{forecast.frequency}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Forecast Horizon:</span>
                  <span className="font-medium">{forecast.horizon} periods</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Data Source:</span>
                  <span className="font-medium">{forecast.data_source}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Next Update:</span>
                  <span className="font-medium">{new Date(forecast.next_forecast).toLocaleString()}</span>
                </div>
              </div>

              {/* Seasonality Information */}
              {forecast.seasonality && (
                <div className="mt-4 p-3 bg-gray-50 rounded-lg">
                  <h4 className="text-sm font-medium text-gray-900 mb-2">Seasonality Strength</h4>
                  <div className="space-y-2">
                    {Object.entries(forecast.seasonality).map(([period, strength]) => (
                      <div key={period} className="flex items-center justify-between text-sm">
                        <span className="text-gray-600 capitalize">{period}:</span>
                        <div className="flex items-center gap-2">
                          <div className="w-16 bg-gray-200 rounded-full h-2">
                            <div
                              className="bg-blue-500 h-2 rounded-full"
                              style={{ width: `${(strength as number) * 100}%` }}
                            />
                          </div>
                          <span className="text-gray-700 w-10 text-right">
                            {((strength as number) * 100).toFixed(0)}%
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Action Buttons */}
              <div className="mt-6 flex gap-2">
                <button
                  onClick={() => setSelectedForecast(forecast)}
                  className="flex-1 bg-blue-600 text-white py-2 px-3 rounded-lg hover:bg-blue-700 transition-colors text-sm flex items-center justify-center gap-2"
                >
                  <Eye className="w-4 h-4" />
                  View Details
                </button>
                <button
                  onClick={() => handleRefreshForecast(forecast.id)}
                  className="bg-gray-100 text-gray-700 py-2 px-3 rounded-lg hover:bg-gray-200 transition-colors"
                >
                  <RefreshCw className="w-4 h-4" />
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
          <TrendingUp className="w-8 h-8 text-blue-500" />
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
              Time Series Analytics
            </h1>
            <p className="text-gray-600">
              Advanced forecasting, anomaly detection, and trend analysis for your time series data
            </p>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={() => setAutoRefresh(!autoRefresh)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                autoRefresh 
                  ? 'bg-green-100 text-green-700 hover:bg-green-200' 
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              {autoRefresh ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              Auto Refresh
            </button>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="flex space-x-1 mb-6 bg-gray-100 p-1 rounded-lg">
        {[
          { key: 'overview', label: 'Overview', icon: BarChart3 },
          { key: 'forecasts', label: 'Forecasts', icon: TrendingUp },
          { key: 'insights', label: 'Insights', icon: Brain },
          { key: 'anomalies', label: 'Anomalies', icon: AlertTriangle }
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
          {activeTab === 'forecasts' && renderForecasts()}
          {activeTab === 'insights' && (
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-6">Time Series Insights</h2>
              {/* Insights content would go here */}
            </div>
          )}
          {activeTab === 'anomalies' && (
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-6">Anomaly Detection</h2>
              {/* Anomaly detection content would go here */}
            </div>
          )}
        </motion.div>
      </AnimatePresence>

      {/* Running Jobs Status */}
      {activeJobs.length > 0 && (
        <div className="fixed bottom-4 right-4 bg-white rounded-lg shadow-lg border border-gray-200 p-4 max-w-sm">
          <h4 className="font-medium text-gray-900 mb-2 flex items-center gap-2">
            <Activity className="w-4 h-4 text-blue-500" />
            Running Jobs ({activeJobs.length})
          </h4>
          <div className="space-y-2">
            {activeJobs.map((job) => (
              <div key={job.id} className="text-sm">
                <div className="flex justify-between items-center mb-1">
                  <span className="text-gray-700">{job.forecast_name}</span>
                  <span className={`text-xs px-2 py-1 rounded-full ${getStatusColor(job.status)}`}>
                    {job.status}
                  </span>
                </div>
                {job.status === 'running' && (
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-blue-600 h-2 rounded-full transition-all duration-500"
                      style={{ width: `${job.progress}%` }}
                    />
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default TimeSeriesAnalytics;