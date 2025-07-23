import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardContent } from '../ui/card';
import { Badge } from '../ui/badge';
import { Button } from '../ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar, ScatterChart, Scatter, Cell, PieChart, Pie,
  AreaChart, Area, ComposedChart, Heatmap, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar
} from 'recharts';
import { 
  TrendingUp, BarChart3, PieChart as PieIcon, Activity,
  Target, Zap, AlertCircle, CheckCircle2, Clock, Layers
} from 'lucide-react';

interface DriftVisualizationProps {
  modelId: string;
  timeRange: string;
  refreshInterval?: number;
}

const DriftVisualizationDashboard: React.FC<DriftVisualizationProps> = ({
  modelId,
  timeRange = '24h',
  refreshInterval = 30
}) => {
  const [driftData, setDriftData] = useState<any[]>([]);
  const [featureCorrelations, setFeatureCorrelations] = useState<any[]>([]);
  const [distributionChanges, setDistributionChanges] = useState<any[]>([]);
  const [performanceMetrics, setPerformanceMetrics] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);

  // Mock data generation for demonstration
  useEffect(() => {
    const generateMockData = () => {
      // Time series drift data
      const timeSeriesData = Array.from({ length: 24 }, (_, i) => ({
        time: new Date(Date.now() - (23 - i) * 3600000).toISOString().split('T')[1].split(':')[0] + ':00',
        overallDrift: 0.1 + Math.random() * 0.4,
        featureDrift: {
          transaction_amount: Math.random() * 0.6,
          user_age: Math.random() * 0.3,
          location_risk: Math.random() * 0.5,
          payment_method: Math.random() * 0.4,
          session_duration: Math.random() * 0.35
        },
        alertCount: Math.floor(Math.random() * 5),
        dataQuality: 0.85 + Math.random() * 0.15
      }));

      // Feature correlation matrix
      const features = ['transaction_amount', 'user_age', 'location_risk', 'payment_method', 'session_duration'];
      const correlationData = features.map(feature1 => ({
        feature: feature1,
        ...features.reduce((acc, feature2) => ({
          ...acc,
          [feature2]: feature1 === feature2 ? 1 : Math.random() * 2 - 1
        }), {})
      }));

      // Distribution changes
      const distributionData = features.map(feature => ({
        feature,
        baseline: {
          mean: Math.random() * 100,
          std: Math.random() * 20,
          skewness: Math.random() * 2 - 1
        },
        current: {
          mean: Math.random() * 100,
          std: Math.random() * 20,
          skewness: Math.random() * 2 - 1
        },
        driftScore: Math.random() * 0.8
      }));

      // Performance metrics over time
      const performanceData = Array.from({ length: 7 }, (_, i) => ({
        date: new Date(Date.now() - (6 - i) * 24 * 3600000).toLocaleDateString(),
        accuracy: 0.85 + Math.random() * 0.1,
        precision: 0.80 + Math.random() * 0.15,
        recall: 0.82 + Math.random() * 0.12,
        f1Score: 0.83 + Math.random() * 0.10,
        driftImpact: Math.random() * 0.3
      }));

      setDriftData(timeSeriesData);
      setFeatureCorrelations(correlationData);
      setDistributionChanges(distributionData);
      setPerformanceMetrics(performanceData);
    };

    generateMockData();
  }, [modelId, timeRange]);

  const getDriftSeverityColor = (score: number) => {
    if (score > 0.5) return '#ef4444'; // red
    if (score > 0.3) return '#f97316'; // orange
    if (score > 0.15) return '#eab308'; // yellow
    return '#22c55e'; // green
  };

  const HeatmapCell = ({ payload, x, y, width, height }: any) => {
    const value = payload?.value || 0;
    const intensity = Math.abs(value);
    const color = value > 0 ? `rgba(239, 68, 68, ${intensity})` : `rgba(59, 130, 246, ${intensity})`;
    
    return (
      <rect
        x={x}
        y={y}
        width={width}
        height={height}
        fill={color}
        stroke="#e5e7eb"
        strokeWidth={1}
      />
    );
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Drift Visualization Dashboard</h2>
          <p className="text-gray-600">Comprehensive visual analysis of data drift patterns</p>
        </div>
        <div className="flex items-center space-x-2">
          <Badge className="bg-blue-100 text-blue-800">
            Model: {modelId}
          </Badge>
          <Badge className="bg-gray-100 text-gray-800">
            {timeRange}
          </Badge>
        </div>
      </div>

      <Tabs defaultValue="overview" className="space-y-4">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="temporal">Temporal</TabsTrigger>
          <TabsTrigger value="features">Features</TabsTrigger>
          <TabsTrigger value="correlations">Correlations</TabsTrigger>
          <TabsTrigger value="performance">Performance</TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Overall Drift Timeline */}
            <Card>
              <CardHeader>
                <h3 className="text-lg font-semibold flex items-center space-x-2">
                  <TrendingUp className="h-5 w-5" />
                  <span>Overall Drift Score Timeline</span>
                </h3>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <AreaChart data={driftData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" />
                    <YAxis domain={[0, 1]} />
                    <Tooltip 
                      formatter={(value: number) => [value.toFixed(3), 'Drift Score']}
                    />
                    <Area
                      type="monotone"
                      dataKey="overallDrift"
                      stroke="#ef4444"
                      fill="#ef4444"
                      fillOpacity={0.2}
                    />
                    {/* Threshold line */}
                    <Line 
                      type="monotone" 
                      dataKey={() => 0.3} 
                      stroke="#f97316" 
                      strokeDasharray="5 5"
                      dot={false}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* Feature Drift Heatmap */}
            <Card>
              <CardHeader>
                <h3 className="text-lg font-semibold flex items-center space-x-2">
                  <Layers className="h-5 w-5" />
                  <span>Feature Drift Heatmap</span>
                </h3>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart 
                    data={driftData.slice(-12)} 
                    layout="horizontal"
                    margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" domain={[0, 1]} />
                    <YAxis dataKey="time" type="category" />
                    <Tooltip />
                    <Bar dataKey="featureDrift.transaction_amount" stackId="a" fill="#ef4444" />
                    <Bar dataKey="featureDrift.user_age" stackId="a" fill="#f97316" />
                    <Bar dataKey="featureDrift.location_risk" stackId="a" fill="#eab308" />
                    <Bar dataKey="featureDrift.payment_method" stackId="a" fill="#22c55e" />
                    <Bar dataKey="featureDrift.session_duration" stackId="a" fill="#3b82f6" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* Drift Distribution by Severity */}
            <Card>
              <CardHeader>
                <h3 className="text-lg font-semibold flex items-center space-x-2">
                  <PieIcon className="h-5 w-5" />
                  <span>Drift Severity Distribution</span>
                </h3>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={[
                        { name: 'Low', value: 35, fill: '#22c55e' },
                        { name: 'Medium', value: 25, fill: '#eab308' },
                        { name: 'High', value: 20, fill: '#f97316' },
                        { name: 'Critical', value: 10, fill: '#ef4444' },
                        { name: 'None', value: 10, fill: '#6b7280' }
                      ]}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={100}
                      paddingAngle={2}
                      dataKey="value"
                    >
                    </Pie>
                    <Tooltip />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* Data Quality Metrics */}
            <Card>
              <CardHeader>
                <h3 className="text-lg font-semibold flex items-center space-x-2">
                  <CheckCircle2 className="h-5 w-5" />
                  <span>Data Quality Metrics</span>
                </h3>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <ComposedChart data={driftData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" />
                    <YAxis yAxisId="left" domain={[0, 1]} />
                    <YAxis yAxisId="right" orientation="right" />
                    <Tooltip />
                    <Legend />
                    <Area
                      yAxisId="left"
                      type="monotone"
                      dataKey="dataQuality"
                      fill="#22c55e"
                      fillOpacity={0.2}
                      stroke="#22c55e"
                      name="Data Quality"
                    />
                    <Bar
                      yAxisId="right"
                      dataKey="alertCount"
                      fill="#f97316"
                      name="Alert Count"
                    />
                  </ComposedChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Temporal Analysis Tab */}
        <TabsContent value="temporal" className="space-y-4">
          <Card>
            <CardHeader>
              <h3 className="text-lg font-semibold">Temporal Drift Patterns</h3>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={driftData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="featureDrift.transaction_amount" 
                    stroke="#ef4444" 
                    name="Transaction Amount"
                  />
                  <Line 
                    type="monotone" 
                    dataKey="featureDrift.user_age" 
                    stroke="#f97316" 
                    name="User Age"
                  />
                  <Line 
                    type="monotone" 
                    dataKey="featureDrift.location_risk" 
                    stroke="#eab308" 
                    name="Location Risk"
                  />
                  <Line 
                    type="monotone" 
                    dataKey="featureDrift.payment_method" 
                    stroke="#22c55e" 
                    name="Payment Method"
                  />
                  <Line 
                    type="monotone" 
                    dataKey="featureDrift.session_duration" 
                    stroke="#3b82f6" 
                    name="Session Duration"
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Feature Analysis Tab */}
        <TabsContent value="features" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Feature Drift Scores */}
            <Card>
              <CardHeader>
                <h3 className="text-lg font-semibold">Current Feature Drift Scores</h3>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart 
                    data={distributionChanges}
                    layout="vertical"
                    margin={{ top: 5, right: 30, left: 50, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" domain={[0, 1]} />
                    <YAxis dataKey="feature" type="category" width={100} />
                    <Tooltip />
                    <Bar dataKey="driftScore">
                      {distributionChanges.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={getDriftSeverityColor(entry.driftScore)} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* Distribution Changes */}
            <Card>
              <CardHeader>
                <h3 className="text-lg font-semibold">Distribution Shifts</h3>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {distributionChanges.map((feature, index) => (
                    <div key={index} className="border-b pb-3">
                      <div className="flex justify-between items-center mb-2">
                        <span className="font-medium">{feature.feature}</span>
                        <Badge 
                          className={
                            feature.driftScore > 0.5 ? 'bg-red-100 text-red-800' :
                            feature.driftScore > 0.3 ? 'bg-orange-100 text-orange-800' :
                            'bg-green-100 text-green-800'
                          }
                        >
                          {feature.driftScore.toFixed(3)}
                        </Badge>
                      </div>
                      <div className="text-sm text-gray-600 grid grid-cols-2 gap-4">
                        <div>
                          <span className="font-medium">Baseline:</span>
                          <br />μ = {feature.baseline.mean.toFixed(2)}
                          <br />σ = {feature.baseline.std.toFixed(2)}
                        </div>
                        <div>
                          <span className="font-medium">Current:</span>
                          <br />μ = {feature.current.mean.toFixed(2)}
                          <br />σ = {feature.current.std.toFixed(2)}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Correlations Tab */}
        <TabsContent value="correlations" className="space-y-4">
          <Card>
            <CardHeader>
              <h3 className="text-lg font-semibold">Feature Correlation Changes</h3>
              <p className="text-sm text-gray-600">
                Correlation matrix showing relationships between features
              </p>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Correlation Heatmap Placeholder */}
                <div className="space-y-4">
                  <h4 className="font-medium">Correlation Matrix</h4>
                  <div className="bg-gray-50 p-4 rounded-lg text-center">
                    <p className="text-gray-500 mb-2">Correlation heatmap visualization</p>
                    <p className="text-sm text-gray-400">
                      Interactive heatmap showing feature correlations would be displayed here
                    </p>
                  </div>
                </div>

                {/* Correlation Changes */}
                <div className="space-y-4">
                  <h4 className="font-medium">Significant Changes</h4>
                  <div className="space-y-2">
                    {[
                      { pair: 'transaction_amount ↔ location_risk', change: '+0.23', severity: 'high' },
                      { pair: 'user_age ↔ payment_method', change: '-0.15', severity: 'medium' },
                      { pair: 'session_duration ↔ transaction_amount', change: '+0.08', severity: 'low' }
                    ].map((item, index) => (
                      <div key={index} className="flex justify-between items-center p-2 bg-gray-50 rounded">
                        <span className="text-sm">{item.pair}</span>
                        <div className="flex items-center space-x-2">
                          <span className="text-sm font-mono">{item.change}</span>
                          <Badge 
                            className={
                              item.severity === 'high' ? 'bg-red-100 text-red-800' :
                              item.severity === 'medium' ? 'bg-orange-100 text-orange-800' :
                              'bg-yellow-100 text-yellow-800'
                            }
                          >
                            {item.severity}
                          </Badge>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Performance Impact Tab */}
        <TabsContent value="performance" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Performance Metrics Over Time */}
            <Card>
              <CardHeader>
                <h3 className="text-lg font-semibold">Performance Metrics Trend</h3>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={performanceMetrics}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis domain={[0.7, 1]} />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="accuracy" stroke="#22c55e" name="Accuracy" />
                    <Line type="monotone" dataKey="precision" stroke="#3b82f6" name="Precision" />
                    <Line type="monotone" dataKey="recall" stroke="#f97316" name="Recall" />
                    <Line type="monotone" dataKey="f1Score" stroke="#8b5cf6" name="F1 Score" />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* Drift vs Performance Impact */}
            <Card>
              <CardHeader>
                <h3 className="text-lg font-semibold">Drift Impact on Performance</h3>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <ScatterChart data={performanceMetrics}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="driftImpact" name="Drift Impact" />
                    <YAxis dataKey="accuracy" name="Accuracy" domain={[0.7, 1]} />
                    <Tooltip 
                      formatter={(value, name) => [
                        typeof value === 'number' ? value.toFixed(3) : value,
                        name
                      ]}
                    />
                    <Scatter dataKey="accuracy" fill="#ef4444" />
                  </ScatterChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* Performance Alerts */}
            <Card>
              <CardHeader>
                <h3 className="text-lg font-semibold">Performance Alerts</h3>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {[
                    { 
                      type: 'warning', 
                      message: 'Accuracy dropped by 3% in the last 24 hours',
                      timestamp: '2 hours ago'
                    },
                    { 
                      type: 'info', 
                      message: 'Precision remains stable despite feature drift',
                      timestamp: '6 hours ago'
                    },
                    { 
                      type: 'critical', 
                      message: 'F1 score below acceptable threshold',
                      timestamp: '12 hours ago'
                    }
                  ].map((alert, index) => (
                    <div key={index} className={`p-3 rounded-lg border ${
                      alert.type === 'critical' ? 'border-red-200 bg-red-50' :
                      alert.type === 'warning' ? 'border-orange-200 bg-orange-50' :
                      'border-blue-200 bg-blue-50'
                    }`}>
                      <div className="flex items-start space-x-2">
                        {alert.type === 'critical' ? <AlertCircle className="h-4 w-4 text-red-600 mt-0.5" /> :
                         alert.type === 'warning' ? <AlertCircle className="h-4 w-4 text-orange-600 mt-0.5" /> :
                         <Activity className="h-4 w-4 text-blue-600 mt-0.5" />}
                        <div className="flex-1">
                          <p className="text-sm font-medium">{alert.message}</p>
                          <p className="text-xs text-gray-500 mt-1">{alert.timestamp}</p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default DriftVisualizationDashboard;