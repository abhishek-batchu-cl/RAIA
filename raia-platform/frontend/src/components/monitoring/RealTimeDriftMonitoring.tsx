import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Card, CardHeader, CardContent } from '../ui/card';
import { Badge } from '../ui/badge';
import { Button } from '../ui/button';
import { Alert, AlertDescription } from '../ui/alert';
import { Progress } from '../ui/progress';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  AreaChart, Area, ComposedChart, Bar
} from 'recharts';
import { 
  Play, Pause, Square, AlertTriangle, Zap, Activity, 
  Wifi, WifiOff, RefreshCw, Settings, Bell, Volume2, VolumeX
} from 'lucide-react';

interface RealTimeDriftPoint {
  timestamp: string;
  overallScore: number;
  featureScores: Record<string, number>;
  alertLevel: 'none' | 'low' | 'medium' | 'high' | 'critical';
  dataQuality: number;
  sampleSize: number;
}

interface StreamingAlert {
  id: string;
  timestamp: string;
  type: 'drift' | 'quality' | 'performance';
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  feature?: string;
  value?: number;
  threshold?: number;
}

interface RealTimeDriftMonitoringProps {
  modelId: string;
  websocketUrl?: string;
  bufferSize?: number;
  alertThresholds?: {
    low: number;
    medium: number;
    high: number;
    critical: number;
  };
}

const RealTimeDriftMonitoring: React.FC<RealTimeDriftMonitoringProps> = ({
  modelId,
  websocketUrl = 'ws://localhost:8080/drift-stream',
  bufferSize = 100,
  alertThresholds = { low: 0.15, medium: 0.25, high: 0.4, critical: 0.6 }
}) => {
  const [isConnected, setIsConnected] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [driftBuffer, setDriftBuffer] = useState<RealTimeDriftPoint[]>([]);
  const [alerts, setAlerts] = useState<StreamingAlert[]>([]);
  const [currentMetrics, setCurrentMetrics] = useState<RealTimeDriftPoint | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<'disconnected' | 'connecting' | 'connected' | 'error'>('disconnected');
  const [soundEnabled, setSoundEnabled] = useState(true);
  const [autoScroll, setAutoScroll] = useState(true);
  
  const websocketRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const alertsContainerRef = useRef<HTMLDivElement>(null);
  const dataGeneratorRef = useRef<NodeJS.Timeout | null>(null);

  // Initialize audio context for alert sounds
  useEffect(() => {
    if (soundEnabled && !audioContextRef.current) {
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
    }
  }, [soundEnabled]);

  // Play alert sound
  const playAlertSound = useCallback((severity: string) => {
    if (!soundEnabled || !audioContextRef.current) return;
    
    const frequencies = {
      low: 440,
      medium: 523,
      high: 659,
      critical: 880
    };
    
    const frequency = frequencies[severity as keyof typeof frequencies] || 440;
    const oscillator = audioContextRef.current.createOscillator();
    const gainNode = audioContextRef.current.createGain();
    
    oscillator.connect(gainNode);
    gainNode.connect(audioContextRef.current.destination);
    
    oscillator.frequency.value = frequency;
    oscillator.type = 'sine';
    
    gainNode.gain.setValueAtTime(0.1, audioContextRef.current.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.01, audioContextRef.current.currentTime + 0.5);
    
    oscillator.start(audioContextRef.current.currentTime);
    oscillator.stop(audioContextRef.current.currentTime + 0.5);
  }, [soundEnabled]);

  // Generate mock real-time data for demonstration
  const generateMockDataPoint = useCallback((): RealTimeDriftPoint => {
    const baseScore = 0.1 + Math.random() * 0.5;
    const timestamp = new Date().toISOString();
    
    const featureScores = {
      transaction_amount: Math.random() * 0.8,
      user_age: Math.random() * 0.4,
      location_risk: Math.random() * 0.6,
      payment_method: Math.random() * 0.3,
      session_duration: Math.random() * 0.5
    };
    
    const overallScore = Object.values(featureScores).reduce((a, b) => a + b, 0) / Object.keys(featureScores).length;
    
    let alertLevel: 'none' | 'low' | 'medium' | 'high' | 'critical' = 'none';
    if (overallScore >= alertThresholds.critical) alertLevel = 'critical';
    else if (overallScore >= alertThresholds.high) alertLevel = 'high';
    else if (overallScore >= alertThresholds.medium) alertLevel = 'medium';
    else if (overallScore >= alertThresholds.low) alertLevel = 'low';
    
    return {
      timestamp,
      overallScore,
      featureScores,
      alertLevel,
      dataQuality: 0.85 + Math.random() * 0.15,
      sampleSize: 100 + Math.floor(Math.random() * 900)
    };
  }, [alertThresholds]);

  // Add new alert
  const addAlert = useCallback((dataPoint: RealTimeDriftPoint) => {
    if (dataPoint.alertLevel === 'none') return;
    
    const alert: StreamingAlert = {
      id: `alert-${Date.now()}-${Math.random()}`,
      timestamp: dataPoint.timestamp,
      type: 'drift',
      severity: dataPoint.alertLevel as 'low' | 'medium' | 'high' | 'critical',
      message: `Drift detected: Overall score ${dataPoint.overallScore.toFixed(3)} exceeds ${dataPoint.alertLevel} threshold`,
      value: dataPoint.overallScore,
      threshold: alertThresholds[dataPoint.alertLevel as keyof typeof alertThresholds]
    };
    
    setAlerts(prev => {
      const updated = [alert, ...prev].slice(0, 50); // Keep last 50 alerts
      return updated;
    });
    
    // Play sound for medium and above alerts
    if (['medium', 'high', 'critical'].includes(dataPoint.alertLevel)) {
      playAlertSound(dataPoint.alertLevel);
    }
  }, [alertThresholds, playAlertSound]);

  // Start mock data generation
  const startMockStreaming = useCallback(() => {
    if (dataGeneratorRef.current) return;
    
    dataGeneratorRef.current = setInterval(() => {
      const dataPoint = generateMockDataPoint();
      
      setDriftBuffer(prev => {
        const updated = [...prev, dataPoint].slice(-bufferSize);
        return updated;
      });
      
      setCurrentMetrics(dataPoint);
      addAlert(dataPoint);
      
    }, 2000); // Generate data every 2 seconds
  }, [generateMockDataPoint, bufferSize, addAlert]);

  // Stop mock data generation
  const stopMockStreaming = useCallback(() => {
    if (dataGeneratorRef.current) {
      clearInterval(dataGeneratorRef.current);
      dataGeneratorRef.current = null;
    }
  }, []);

  // WebSocket connection management (mock for demo)
  const connectWebSocket = useCallback(() => {
    setConnectionStatus('connecting');
    
    // Simulate connection delay
    setTimeout(() => {
      setIsConnected(true);
      setConnectionStatus('connected');
      startMockStreaming();
    }, 1000);
  }, [startMockStreaming]);

  const disconnectWebSocket = useCallback(() => {
    setIsConnected(false);
    setConnectionStatus('disconnected');
    setIsStreaming(false);
    stopMockStreaming();
  }, [stopMockStreaming]);

  const startStreaming = () => {
    setIsStreaming(true);
  };

  const stopStreaming = () => {
    setIsStreaming(false);
  };

  const clearAlerts = () => {
    setAlerts([]);
  };

  // Auto-scroll alerts
  useEffect(() => {
    if (autoScroll && alertsContainerRef.current) {
      alertsContainerRef.current.scrollTop = 0;
    }
  }, [alerts, autoScroll]);

  const getAlertColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'bg-red-100 border-red-300 text-red-800';
      case 'high': return 'bg-orange-100 border-orange-300 text-orange-800';
      case 'medium': return 'bg-yellow-100 border-yellow-300 text-yellow-800';
      default: return 'bg-blue-100 border-blue-300 text-blue-800';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'connected': return 'text-green-600';
      case 'connecting': return 'text-yellow-600';
      case 'error': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  return (
    <div className="space-y-6">
      {/* Header & Controls */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Real-Time Drift Monitoring</h2>
          <p className="text-gray-600">Live monitoring of data drift patterns</p>
        </div>
        <div className="flex items-center space-x-3">
          <Badge className={`${isConnected ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'}`}>
            {isConnected ? <Wifi className="h-3 w-3 mr-1" /> : <WifiOff className="h-3 w-3 mr-1" />}
            {connectionStatus.charAt(0).toUpperCase() + connectionStatus.slice(1)}
          </Badge>
          <Button
            onClick={soundEnabled ? () => setSoundEnabled(false) : () => setSoundEnabled(true)}
            variant="outline"
            size="sm"
          >
            {soundEnabled ? <Volume2 className="h-4 w-4" /> : <VolumeX className="h-4 w-4" />}
          </Button>
          <Button
            onClick={isConnected ? disconnectWebSocket : connectWebSocket}
            variant={isConnected ? "destructive" : "default"}
            disabled={connectionStatus === 'connecting'}
          >
            {connectionStatus === 'connecting' ? (
              <RefreshCw className="h-4 w-4 animate-spin mr-2" />
            ) : isConnected ? (
              <Square className="h-4 w-4 mr-2" />
            ) : (
              <Play className="h-4 w-4 mr-2" />
            )}
            {connectionStatus === 'connecting' ? 'Connecting...' : isConnected ? 'Disconnect' : 'Connect'}
          </Button>
        </div>
      </div>

      {/* Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center space-x-2">
              <Activity className="h-5 w-5 text-blue-600" />
              <div>
                <p className="text-sm font-medium text-gray-600">Current Score</p>
                <p className="text-2xl font-bold text-gray-900">
                  {currentMetrics?.overallScore.toFixed(3) || '0.000'}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center space-x-2">
              <Zap className="h-5 w-5 text-orange-600" />
              <div>
                <p className="text-sm font-medium text-gray-600">Alert Level</p>
                <Badge className={`text-sm ${
                  currentMetrics?.alertLevel === 'critical' ? 'bg-red-100 text-red-800' :
                  currentMetrics?.alertLevel === 'high' ? 'bg-orange-100 text-orange-800' :
                  currentMetrics?.alertLevel === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                  currentMetrics?.alertLevel === 'low' ? 'bg-blue-100 text-blue-800' :
                  'bg-green-100 text-green-800'
                }`}>
                  {currentMetrics?.alertLevel || 'none'}
                </Badge>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center space-x-2">
              <Bell className="h-5 w-5 text-red-600" />
              <div>
                <p className="text-sm font-medium text-gray-600">Active Alerts</p>
                <p className="text-2xl font-bold text-gray-900">{alerts.length}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center space-x-2">
              <RefreshCw className="h-5 w-5 text-green-600" />
              <div>
                <p className="text-sm font-medium text-gray-600">Data Quality</p>
                <div className="flex items-center space-x-2">
                  <Progress 
                    value={(currentMetrics?.dataQuality || 0) * 100} 
                    className="w-16" 
                  />
                  <span className="text-sm font-bold">
                    {((currentMetrics?.dataQuality || 0) * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Real-Time Chart */}
        <div className="lg:col-span-2">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold">Live Drift Stream</h3>
                <div className="flex items-center space-x-2">
                  <div className={`w-2 h-2 rounded-full ${isStreaming ? 'bg-green-500' : 'bg-gray-400'}`} />
                  <span className="text-sm text-gray-600">
                    {isStreaming ? 'Streaming' : 'Stopped'}
                  </span>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={400}>
                <ComposedChart data={driftBuffer.slice(-50)}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="timestamp" 
                    tickFormatter={(value) => new Date(value).toLocaleTimeString()}
                  />
                  <YAxis yAxisId="left" domain={[0, 1]} />
                  <YAxis yAxisId="right" orientation="right" />
                  <Tooltip 
                    labelFormatter={(value) => new Date(value).toLocaleString()}
                    formatter={(value: number) => [value.toFixed(3), 'Drift Score']}
                  />
                  
                  {/* Threshold lines */}
                  <Line 
                    yAxisId="left"
                    type="monotone" 
                    dataKey={() => alertThresholds.medium}
                    stroke="#f97316" 
                    strokeDasharray="5 5"
                    dot={false}
                    connectNulls={false}
                  />
                  <Line 
                    yAxisId="left"
                    type="monotone" 
                    dataKey={() => alertThresholds.high}
                    stroke="#ef4444" 
                    strokeDasharray="5 5"
                    dot={false}
                    connectNulls={false}
                  />
                  
                  <Area
                    yAxisId="left"
                    type="monotone"
                    dataKey="overallScore"
                    stroke="#3b82f6"
                    fill="#3b82f6"
                    fillOpacity={0.1}
                  />
                  <Bar
                    yAxisId="right"
                    dataKey="sampleSize"
                    fill="#10b981"
                    fillOpacity={0.3}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </div>

        {/* Live Alerts Panel */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold flex items-center space-x-2">
                <AlertTriangle className="h-5 w-5" />
                <span>Live Alerts</span>
              </h3>
              <Button onClick={clearAlerts} variant="outline" size="sm">
                Clear
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <div 
              ref={alertsContainerRef}
              className="space-y-2 max-h-96 overflow-y-auto"
            >
              {alerts.length === 0 ? (
                <div className="text-center py-8 text-gray-500">
                  <Bell className="h-8 w-8 mx-auto mb-2 opacity-50" />
                  <p>No alerts</p>
                </div>
              ) : (
                alerts.map(alert => (
                  <Alert key={alert.id} className={`text-xs ${getAlertColor(alert.severity)}`}>
                    <AlertDescription>
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <div className="font-medium">{alert.message}</div>
                          <div className="text-xs opacity-75 mt-1">
                            {new Date(alert.timestamp).toLocaleTimeString()}
                          </div>
                        </div>
                        <Badge className={`ml-2 ${getAlertColor(alert.severity)}`}>
                          {alert.severity}
                        </Badge>
                      </div>
                    </AlertDescription>
                  </Alert>
                ))
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Feature-Level Real-Time Metrics */}
      <Card>
        <CardHeader>
          <h3 className="text-lg font-semibold">Feature Drift Streams</h3>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={driftBuffer.slice(-20)}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="timestamp" 
                tickFormatter={(value) => new Date(value).toLocaleTimeString()}
              />
              <YAxis domain={[0, 1]} />
              <Tooltip 
                labelFormatter={(value) => new Date(value).toLocaleString()}
              />
              <Line 
                type="monotone" 
                dataKey="featureScores.transaction_amount" 
                stroke="#ef4444" 
                strokeWidth={2}
                name="Transaction Amount"
                dot={false}
              />
              <Line 
                type="monotone" 
                dataKey="featureScores.user_age" 
                stroke="#f97316" 
                strokeWidth={2}
                name="User Age"
                dot={false}
              />
              <Line 
                type="monotone" 
                dataKey="featureScores.location_risk" 
                stroke="#eab308" 
                strokeWidth={2}
                name="Location Risk"
                dot={false}
              />
              <Line 
                type="monotone" 
                dataKey="featureScores.payment_method" 
                stroke="#22c55e" 
                strokeWidth={2}
                name="Payment Method"
                dot={false}
              />
              <Line 
                type="monotone" 
                dataKey="featureScores.session_duration" 
                stroke="#3b82f6" 
                strokeWidth={2}
                name="Session Duration"
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </div>
  );
};

export default RealTimeDriftMonitoring;