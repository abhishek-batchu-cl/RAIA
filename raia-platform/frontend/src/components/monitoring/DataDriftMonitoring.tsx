import React, { useState, useEffect, useRef } from 'react';
import { Card, CardHeader, CardContent } from '../ui/card';
import { Badge } from '../ui/badge';
import { Button } from '../ui/button';
import { Alert, AlertDescription } from '../ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import { Progress } from '../ui/progress';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar, ScatterChart, Scatter, Cell, PieChart, Pie,
  AreaChart, Area, ComposedChart
} from 'recharts';
import { 
  AlertTriangle, TrendingUp, TrendingDown, Clock, 
  Activity, Settings, Download, RefreshCw, Play, Pause,
  BarChart3, PieChart as PieIcon, LineChart as LineIcon
} from 'lucide-react';

interface DriftMetric {
  feature: string;
  driftScore: number;
  severity: 'none' | 'low' | 'medium' | 'high' | 'critical';
  pValue: number;
  threshold: number;
  testMethod: string;
  lastUpdated: string;
}

interface DriftReport {
  reportId: string;
  modelId: string;
  detectionTime: string;
  overallDriftScore: number;
  driftDetected: boolean;
  featureDrift: Record<string, any>;
  performanceDrift?: any;
  recommendations: string[];
  evidently?: any;
}

interface TemporalDriftData {
  timestamp: string;
  driftScore: number;
  driftDetected: boolean;
  featureDriftCount: number;
}

const DataDriftMonitoring: React.FC = () => {
  const [activeModel, setActiveModel] = useState<string>('');
  const [driftReports, setDriftReports] = useState<DriftReport[]>([]);
  const [currentReport, setCurrentReport] = useState<DriftReport | null>(null);
  const [temporalData, setTemporalData] = useState<TemporalDriftData[]>([]);
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [refreshInterval, setRefreshInterval] = useState(30);
  const [loading, setLoading] = useState(false);
  const [selectedTimeRange, setSelectedTimeRange] = useState('24h');
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  // Mock data for demonstration
  useEffect(() => {
    const mockReport: DriftReport = {
      reportId: 'dr-001',
      modelId: 'fraud-detection-v1',
      detectionTime: new Date().toISOString(),
      overallDriftScore: 0.35,
      driftDetected: true,
      featureDrift: {
        'transaction_amount': {
          drift_detected: true,
          severity: 'high',
          test_results: {
            ks_test: { p_value: 0.02, statistic: 0.23, drift_detected: true }
          }
        },
        'user_age': {
          drift_detected: false,
          severity: 'low',
          test_results: {
            ks_test: { p_value: 0.12, statistic: 0.08, drift_detected: false }
          }
        },
        'location_risk_score': {
          drift_detected: true,
          severity: 'medium',
          test_results: {
            ks_test: { p_value: 0.03, statistic: 0.18, drift_detected: true }
          }
        }
      },
      recommendations: [
        "⚠️ Data drift detected. Consider retraining your model.",
        "🎯 High drift detected in features: transaction_amount. Investigate data collection process.",
        "⏰ Set up automated drift detection for continuous monitoring."
      ]
    };

    const mockTemporalData: TemporalDriftData[] = [
      { timestamp: '2024-01-15T10:00:00Z', driftScore: 0.12, driftDetected: false, featureDriftCount: 0 },
      { timestamp: '2024-01-15T11:00:00Z', driftScore: 0.18, driftDetected: false, featureDriftCount: 1 },
      { timestamp: '2024-01-15T12:00:00Z', driftScore: 0.25, driftDetected: true, featureDriftCount: 2 },
      { timestamp: '2024-01-15T13:00:00Z', driftScore: 0.31, driftDetected: true, featureDriftCount: 2 },
      { timestamp: '2024-01-15T14:00:00Z', driftScore: 0.35, driftDetected: true, featureDriftCount: 3 }
    ];

    setCurrentReport(mockReport);
    setTemporalData(mockTemporalData);
    setDriftReports([mockReport]);
    setActiveModel('fraud-detection-v1');
  }, []);

  const startMonitoring = () => {
    setIsMonitoring(true);
    if (autoRefresh && intervalRef.current === null) {
      intervalRef.current = setInterval(() => {
        fetchDriftData();
      }, refreshInterval * 1000);
    }
  };

  const stopMonitoring = () => {
    setIsMonitoring(false);
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  };

  const fetchDriftData = async () => {
    setLoading(true);
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      // In real implementation, fetch data from API
    } catch (error) {
      console.error('Error fetching drift data:', error);
    } finally {
      setLoading(false);
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'text-red-600 bg-red-50 border-red-200';
      case 'high': return 'text-orange-600 bg-orange-50 border-orange-200';
      case 'medium': return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      case 'low': return 'text-blue-600 bg-blue-50 border-blue-200';
      default: return 'text-green-600 bg-green-50 border-green-200';
    }
  };

  const getDriftStatusIcon = (detected: boolean) => {
    return detected ? (
      <AlertTriangle className="h-4 w-4 text-red-500" />
    ) : (
      <Activity className="h-4 w-4 text-green-500" />
    );
  };

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString();
  };

  const downloadReport = async (reportId: string) => {
    try {
      // Simulate report download
      const blob = new Blob([JSON.stringify(currentReport, null, 2)], {
        type: 'application/json'
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `drift-report-${reportId}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Error downloading report:', error);
    }
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Data Drift Monitoring</h1>
          <p className="text-gray-600 mt-1">
            Monitor and analyze data drift patterns for your ML models
          </p>
        </div>
        <div className="flex items-center space-x-3">
          <Badge 
            className={`${isMonitoring ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'}`}
          >
            {isMonitoring ? 'Monitoring Active' : 'Monitoring Inactive'}
          </Badge>
          <Button
            onClick={isMonitoring ? stopMonitoring : startMonitoring}
            variant={isMonitoring ? "destructive" : "default"}
            className="flex items-center space-x-2"
          >
            {isMonitoring ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
            <span>{isMonitoring ? 'Stop' : 'Start'} Monitoring</span>
          </Button>
          <Button 
            onClick={fetchDriftData}
            disabled={loading}
            variant="outline"
            className="flex items-center space-x-2"
          >
            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            <span>Refresh</span>
          </Button>
        </div>
      </div>

      {/* Model Selection & Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center space-x-2">
              <Activity className="h-5 w-5 text-blue-600" />
              <div>
                <p className="text-sm font-medium text-gray-600">Active Model</p>
                <p className="text-lg font-bold text-gray-900">{activeModel || 'None'}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center space-x-2">
              <TrendingUp className="h-5 w-5 text-red-600" />
              <div>
                <p className="text-sm font-medium text-gray-600">Drift Score</p>
                <p className="text-lg font-bold text-gray-900">
                  {currentReport?.overallDriftScore.toFixed(3) || '0.000'}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center space-x-2">
              <BarChart3 className="h-5 w-5 text-orange-600" />
              <div>
                <p className="text-sm font-medium text-gray-600">Features Drifted</p>
                <p className="text-lg font-bold text-gray-900">
                  {currentReport ? Object.values(currentReport.featureDrift).filter((f: any) => f.drift_detected).length : 0}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center space-x-2">
              <Clock className="h-5 w-5 text-green-600" />
              <div>
                <p className="text-sm font-medium text-gray-600">Last Check</p>
                <p className="text-lg font-bold text-gray-900">
                  {currentReport ? formatTimestamp(currentReport.detectionTime) : 'Never'}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main Content Tabs */}
      <Tabs defaultValue="overview" className="space-y-4">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="temporal">Temporal Analysis</TabsTrigger>
          <TabsTrigger value="features">Feature Drift</TabsTrigger>
          <TabsTrigger value="performance">Performance</TabsTrigger>
          <TabsTrigger value="settings">Settings</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          {/* Drift Status Alert */}
          {currentReport?.driftDetected && (
            <Alert className="border-orange-200 bg-orange-50">
              <AlertTriangle className="h-4 w-4 text-orange-600" />
              <AlertDescription className="text-orange-800">
                Data drift detected! Overall drift score: {currentReport.overallDriftScore.toFixed(3)}
              </AlertDescription>
            </Alert>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Drift Score Trend */}
            <Card>
              <CardHeader>
                <h3 className="text-lg font-semibold flex items-center space-x-2">
                  <LineIcon className="h-5 w-5" />
                  <span>Drift Score Trend</span>
                </h3>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={250}>
                  <LineChart data={temporalData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="timestamp" 
                      tickFormatter={formatTimestamp}
                    />
                    <YAxis />
                    <Tooltip 
                      labelFormatter={(value) => formatTimestamp(value)}
                      formatter={(value: number) => [value.toFixed(3), 'Drift Score']}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="driftScore" 
                      stroke="#ef4444" 
                      strokeWidth={2}
                      dot={{ fill: '#ef4444', strokeWidth: 2, r: 4 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* Feature Drift Distribution */}
            <Card>
              <CardHeader>
                <h3 className="text-lg font-semibold flex items-center space-x-2">
                  <PieIcon className="h-5 w-5" />
                  <span>Feature Drift Distribution</span>
                </h3>
              </CardHeader>
              <CardContent>
                {currentReport && (
                  <div className="space-y-3">
                    {Object.entries(currentReport.featureDrift).map(([feature, data]: [string, any]) => (
                      <div key={feature} className="flex items-center justify-between">
                        <div className="flex items-center space-x-2">
                          {getDriftStatusIcon(data.drift_detected)}
                          <span className="font-medium text-sm">{feature}</span>
                        </div>
                        <div className="flex items-center space-x-2">
                          <Badge className={getSeverityColor(data.severity)}>
                            {data.severity}
                          </Badge>
                          <span className="text-xs text-gray-500">
                            p={data.test_results?.ks_test?.p_value?.toFixed(3) || 'N/A'}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Recommendations */}
          {currentReport?.recommendations && (
            <Card>
              <CardHeader>
                <h3 className="text-lg font-semibold">Recommendations</h3>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {currentReport.recommendations.map((rec, index) => (
                    <Alert key={index} className="border-blue-200 bg-blue-50">
                      <AlertDescription className="text-blue-800">
                        {rec}
                      </AlertDescription>
                    </Alert>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="temporal" className="space-y-4">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold">Temporal Drift Analysis</h3>
                <div className="flex items-center space-x-2">
                  <label className="text-sm font-medium">Time Range:</label>
                  <select 
                    value={selectedTimeRange} 
                    onChange={(e) => setSelectedTimeRange(e.target.value)}
                    className="border border-gray-300 rounded px-2 py-1 text-sm"
                  >
                    <option value="1h">Last Hour</option>
                    <option value="6h">Last 6 Hours</option>
                    <option value="24h">Last 24 Hours</option>
                    <option value="7d">Last 7 Days</option>
                  </select>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={400}>
                <ComposedChart data={temporalData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="timestamp" tickFormatter={formatTimestamp} />
                  <YAxis yAxisId="left" />
                  <YAxis yAxisId="right" orientation="right" />
                  <Tooltip labelFormatter={(value) => formatTimestamp(value)} />
                  <Legend />
                  <Area
                    yAxisId="left"
                    type="monotone"
                    dataKey="driftScore"
                    fill="#ef4444"
                    fillOpacity={0.1}
                    stroke="#ef4444"
                    strokeWidth={2}
                    name="Drift Score"
                  />
                  <Bar
                    yAxisId="right"
                    dataKey="featureDriftCount"
                    fill="#f97316"
                    name="Drifted Features"
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="features" className="space-y-4">
          <Card>
            <CardHeader>
              <h3 className="text-lg font-semibold">Feature-Level Drift Analysis</h3>
            </CardHeader>
            <CardContent>
              {currentReport && (
                <div className="overflow-x-auto">
                  <table className="w-full border-collapse">
                    <thead>
                      <tr className="border-b">
                        <th className="text-left py-2 px-3 font-medium">Feature</th>
                        <th className="text-left py-2 px-3 font-medium">Status</th>
                        <th className="text-left py-2 px-3 font-medium">Severity</th>
                        <th className="text-left py-2 px-3 font-medium">Test Method</th>
                        <th className="text-left py-2 px-3 font-medium">P-Value</th>
                        <th className="text-left py-2 px-3 font-medium">Statistic</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(currentReport.featureDrift).map(([feature, data]: [string, any]) => (
                        <tr key={feature} className="border-b hover:bg-gray-50">
                          <td className="py-2 px-3 font-medium">{feature}</td>
                          <td className="py-2 px-3">
                            {getDriftStatusIcon(data.drift_detected)}
                          </td>
                          <td className="py-2 px-3">
                            <Badge className={getSeverityColor(data.severity)}>
                              {data.severity}
                            </Badge>
                          </td>
                          <td className="py-2 px-3 text-sm text-gray-600">
                            KS Test
                          </td>
                          <td className="py-2 px-3 text-sm">
                            {data.test_results?.ks_test?.p_value?.toFixed(4) || 'N/A'}
                          </td>
                          <td className="py-2 px-3 text-sm">
                            {data.test_results?.ks_test?.statistic?.toFixed(4) || 'N/A'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="performance" className="space-y-4">
          <Card>
            <CardHeader>
              <h3 className="text-lg font-semibold">Model Performance Drift</h3>
            </CardHeader>
            <CardContent>
              <div className="text-center py-8 text-gray-500">
                <Activity className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>Performance drift tracking will be available once you enable performance monitoring.</p>
                <Button className="mt-4" variant="outline">
                  <Settings className="h-4 w-4 mr-2" />
                  Configure Performance Tracking
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="settings" className="space-y-4">
          <Card>
            <CardHeader>
              <h3 className="text-lg font-semibold">Drift Monitoring Settings</h3>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium mb-2">Auto Refresh</label>
                  <div className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      checked={autoRefresh}
                      onChange={(e) => setAutoRefresh(e.target.checked)}
                      className="rounded"
                    />
                    <span className="text-sm">Enable automatic refresh</span>
                  </div>
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">Refresh Interval (seconds)</label>
                  <input
                    type="number"
                    value={refreshInterval}
                    onChange={(e) => setRefreshInterval(Number(e.target.value))}
                    min="10"
                    max="300"
                    className="w-full border border-gray-300 rounded px-3 py-2"
                  />
                </div>
              </div>

              <div className="flex items-center justify-between pt-4 border-t">
                <Button
                  onClick={() => currentReport && downloadReport(currentReport.reportId)}
                  variant="outline"
                  className="flex items-center space-x-2"
                >
                  <Download className="h-4 w-4" />
                  <span>Download Current Report</span>
                </Button>
                <Button variant="outline">
                  <Settings className="h-4 w-4 mr-2" />
                  Advanced Settings
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default DataDriftMonitoring;