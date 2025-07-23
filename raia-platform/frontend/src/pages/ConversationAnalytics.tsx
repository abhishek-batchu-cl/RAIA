import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  MessageSquare,
  Brain,
  TrendingUp,
  Clock,
  Users,
  Target,
  AlertTriangle,
  CheckCircle2,
  Activity,
  Zap,
  Filter,
  Download,
  Play,
  Pause,
  RotateCcw,
  Settings,
  Share2,
  Eye
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
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar
} from 'recharts';

interface ConversationMetrics {
  id: string;
  timestamp: Date;
  sessionId: string;
  agentName: string;
  turnCount: number;
  avgResponseTime: number;
  coherenceScore: number;
  contextRetention: number;
  intentAccuracy: number;
  userSatisfaction: number;
  topicShifts: number;
  personalityConsistency: number;
}

interface ConversationFlow {
  turn: number;
  userIntent: string;
  agentResponse: string;
  confidence: number;
  coherenceScore: number;
  contextPreserved: boolean;
  emotionalTone: 'positive' | 'neutral' | 'negative';
  responseTime: number;
}

const ConversationAnalytics: React.FC = () => {
  const [selectedTimeRange, setSelectedTimeRange] = useState('24h');
  const [selectedAgent, setSelectedAgent] = useState('all');
  const [activeAnalysis, setActiveAnalysis] = useState<string | null>(null);
  const [isLiveMode, setIsLiveMode] = useState(false);
  const [conversationMetrics, setConversationMetrics] = useState<ConversationMetrics[]>([]);
  const [conversationFlows, setConversationFlows] = useState<ConversationFlow[]>([]);

  // Mock data generation
  useEffect(() => {
    const generateMockData = () => {
      const mockMetrics: ConversationMetrics[] = Array.from({ length: 50 }, (_, i) => ({
        id: `conv-${i}`,
        timestamp: new Date(Date.now() - Math.random() * 86400000 * 7),
        sessionId: `session-${Math.floor(Math.random() * 100)}`,
        agentName: ['CustomerSupport', 'TechnicalHelp', 'SalesAssistant', 'GeneralChat'][Math.floor(Math.random() * 4)],
        turnCount: Math.floor(Math.random() * 20) + 3,
        avgResponseTime: Math.random() * 3 + 0.5,
        coherenceScore: Math.random() * 0.3 + 0.7,
        contextRetention: Math.random() * 0.2 + 0.8,
        intentAccuracy: Math.random() * 0.15 + 0.85,
        userSatisfaction: Math.random() * 0.25 + 0.75,
        topicShifts: Math.floor(Math.random() * 5),
        personalityConsistency: Math.random() * 0.1 + 0.9
      }));

      const mockFlows: ConversationFlow[] = Array.from({ length: 15 }, (_, i) => ({
        turn: i + 1,
        userIntent: ['question', 'complaint', 'request', 'greeting', 'clarification'][Math.floor(Math.random() * 5)],
        agentResponse: `Response ${i + 1}`,
        confidence: Math.random() * 0.3 + 0.7,
        coherenceScore: Math.random() * 0.2 + 0.8,
        contextPreserved: Math.random() > 0.2,
        emotionalTone: (['positive', 'neutral', 'negative'] as const)[Math.floor(Math.random() * 3)],
        responseTime: Math.random() * 2 + 0.5
      }));

      setConversationMetrics(mockMetrics);
      setConversationFlows(mockFlows);
    };

    generateMockData();
    const interval = isLiveMode ? setInterval(generateMockData, 5000) : null;

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isLiveMode]);

  // Aggregate metrics
  const aggregatedMetrics = React.useMemo(() => {
    const filtered = conversationMetrics.filter(m => 
      selectedAgent === 'all' || m.agentName === selectedAgent
    );

    return {
      totalConversations: filtered.length,
      avgCoherence: filtered.reduce((sum, m) => sum + m.coherenceScore, 0) / filtered.length,
      avgContextRetention: filtered.reduce((sum, m) => sum + m.contextRetention, 0) / filtered.length,
      avgUserSatisfaction: filtered.reduce((sum, m) => sum + m.userSatisfaction, 0) / filtered.length,
      avgResponseTime: filtered.reduce((sum, m) => sum + m.avgResponseTime, 0) / filtered.length,
      avgPersonalityConsistency: filtered.reduce((sum, m) => sum + m.personalityConsistency, 0) / filtered.length
    };
  }, [conversationMetrics, selectedAgent]);

  // Time series data for charts
  const timeSeriesData = React.useMemo(() => {
    const hourlyData: { [key: string]: any } = {};
    
    conversationMetrics.forEach(metric => {
      const hour = metric.timestamp.getHours();
      const key = `${hour}:00`;
      
      if (!hourlyData[key]) {
        hourlyData[key] = {
          time: key,
          coherence: [],
          contextRetention: [],
          satisfaction: [],
          responseTime: []
        };
      }
      
      hourlyData[key].coherence.push(metric.coherenceScore);
      hourlyData[key].contextRetention.push(metric.contextRetention);
      hourlyData[key].satisfaction.push(metric.userSatisfaction);
      hourlyData[key].responseTime.push(metric.avgResponseTime);
    });

    return Object.values(hourlyData).map((item: any) => ({
      time: item.time,
      coherence: item.coherence.reduce((a: number, b: number) => a + b, 0) / item.coherence.length,
      contextRetention: item.contextRetention.reduce((a: number, b: number) => a + b, 0) / item.contextRetention.length,
      satisfaction: item.satisfaction.reduce((a: number, b: number) => a + b, 0) / item.satisfaction.length,
      responseTime: item.responseTime.reduce((a: number, b: number) => a + b, 0) / item.responseTime.length
    })).sort((a, b) => parseInt(a.time) - parseInt(b.time));
  }, [conversationMetrics]);

  // Agent comparison data
  const agentComparisonData = React.useMemo(() => {
    const agentMap: { [key: string]: any } = {};
    
    conversationMetrics.forEach(metric => {
      if (!agentMap[metric.agentName]) {
        agentMap[metric.agentName] = {
          name: metric.agentName,
          coherence: [],
          contextRetention: [],
          satisfaction: [],
          personalityConsistency: []
        };
      }
      
      agentMap[metric.agentName].coherence.push(metric.coherenceScore);
      agentMap[metric.agentName].contextRetention.push(metric.contextRetention);
      agentMap[metric.agentName].satisfaction.push(metric.userSatisfaction);
      agentMap[metric.agentName].personalityConsistency.push(metric.personalityConsistency);
    });

    return Object.values(agentMap).map((agent: any) => ({
      name: agent.name,
      coherence: (agent.coherence.reduce((a: number, b: number) => a + b, 0) / agent.coherence.length * 100).toFixed(1),
      contextRetention: (agent.contextRetention.reduce((a: number, b: number) => a + b, 0) / agent.contextRetention.length * 100).toFixed(1),
      satisfaction: (agent.satisfaction.reduce((a: number, b: number) => a + b, 0) / agent.satisfaction.length * 100).toFixed(1),
      personalityConsistency: (agent.personalityConsistency.reduce((a: number, b: number) => a + b, 0) / agent.personalityConsistency.length * 100).toFixed(1)
    }));
  }, [conversationMetrics]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between">
        <div>
          <h1 className="text-3xl font-bold text-neutral-900 dark:text-white flex items-center gap-3">
            <div className="p-2 bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg">
              <MessageSquare className="w-6 h-6 text-white" />
            </div>
            Conversation Analytics
          </h1>
          <p className="text-neutral-600 dark:text-neutral-400 mt-2">
            Advanced AI agent conversation flow analysis and optimization insights
          </p>
        </div>

        <div className="flex items-center gap-3 mt-4 lg:mt-0">
          <Button
            variant={isLiveMode ? "destructive" : "secondary"}
            onClick={() => setIsLiveMode(!isLiveMode)}
            className="flex items-center gap-2"
          >
            {isLiveMode ? (
              <>
                <Pause className="w-4 h-4" />
                Stop Live
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                Live Mode
              </>
            )}
          </Button>
          
          <select 
            value={selectedAgent} 
            onChange={(e) => setSelectedAgent(e.target.value)}
            className="px-3 py-2 border border-neutral-300 dark:border-neutral-600 rounded-lg bg-white dark:bg-neutral-800 text-sm"
          >
            <option value="all">All Agents</option>
            <option value="CustomerSupport">Customer Support</option>
            <option value="TechnicalHelp">Technical Help</option>
            <option value="SalesAssistant">Sales Assistant</option>
            <option value="GeneralChat">General Chat</option>
          </select>

          <select 
            value={selectedTimeRange} 
            onChange={(e) => setSelectedTimeRange(e.target.value)}
            className="px-3 py-2 border border-neutral-300 dark:border-neutral-600 rounded-lg bg-white dark:bg-neutral-800 text-sm"
          >
            <option value="1h">Last Hour</option>
            <option value="24h">Last 24 Hours</option>
            <option value="7d">Last 7 Days</option>
            <option value="30d">Last 30 Days</option>
          </select>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-6 gap-4">
        <MetricCard
          title="Total Conversations"
          value={aggregatedMetrics.totalConversations.toLocaleString()}
          icon={<MessageSquare className="w-5 h-5" />}
          trend={12}
          className="bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20"
        />
        
        <MetricCard
          title="Avg Coherence"
          value={`${(aggregatedMetrics.avgCoherence * 100).toFixed(1)}%`}
          icon={<Brain className="w-5 h-5" />}
          trend={5.2}
          className="bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20"
        />
        
        <MetricCard
          title="Context Retention"
          value={`${(aggregatedMetrics.avgContextRetention * 100).toFixed(1)}%`}
          icon={<Target className="w-5 h-5" />}
          trend={3.8}
          className="bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20"
        />
        
        <MetricCard
          title="User Satisfaction"
          value={`${(aggregatedMetrics.avgUserSatisfaction * 100).toFixed(1)}%`}
          icon={CheckCircle2}
          trend={7.1}
          className="bg-gradient-to-br from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20"
        />
        
        <MetricCard
          title="Avg Response Time"
          value={`${aggregatedMetrics.avgResponseTime.toFixed(2)}s`}
          icon={<Clock className="w-5 h-5" />}
          trend={-8.3}
          isDecreaseBetter
          className="bg-gradient-to-br from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20"
        />
        
        <MetricCard
          title="Personality Consistency"
          value={`${(aggregatedMetrics.avgPersonalityConsistency * 100).toFixed(1)}%`}
          icon={<Users className="w-5 h-5" />}
          trend={2.4}
          className="bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20"
        />
      </div>

      {/* Conversation Flow Analysis */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Activity className="w-5 h-5 text-purple-600" />
            Conversation Quality Trends
          </h3>
          
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={timeSeriesData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="time" />
              <YAxis domain={[0.5, 1]} />
              <Tooltip 
                formatter={(value: number) => [`${(value * 100).toFixed(1)}%`, '']}
                labelFormatter={(label) => `Time: ${label}`}
              />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="coherence" 
                stroke="#8b5cf6" 
                strokeWidth={2}
                name="Coherence"
                dot={{ fill: '#8b5cf6' }}
              />
              <Line 
                type="monotone" 
                dataKey="contextRetention" 
                stroke="#10b981" 
                strokeWidth={2}
                name="Context Retention"
                dot={{ fill: '#10b981' }}
              />
              <Line 
                type="monotone" 
                dataKey="satisfaction" 
                stroke="#f59e0b" 
                strokeWidth={2}
                name="User Satisfaction"
                dot={{ fill: '#f59e0b' }}
              />
            </LineChart>
          </ResponsiveContainer>
        </Card>

        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Zap className="w-5 h-5 text-blue-600" />
            Agent Performance Radar
          </h3>
          
          <ResponsiveContainer width="100%" height={300}>
            <RadarChart data={agentComparisonData}>
              <PolarGrid stroke="#e5e7eb" />
              <PolarAngleAxis dataKey="name" />
              <PolarRadiusAxis 
                angle={30} 
                domain={[80, 100]} 
                tickFormatter={(value) => `${value}%`}
              />
              <Radar
                name="Coherence"
                dataKey="coherence"
                stroke="#8b5cf6"
                fill="#8b5cf6"
                fillOpacity={0.3}
                strokeWidth={2}
              />
              <Radar
                name="Context Retention"
                dataKey="contextRetention"
                stroke="#10b981"
                fill="#10b981"
                fillOpacity={0.3}
                strokeWidth={2}
              />
              <Radar
                name="User Satisfaction"
                dataKey="satisfaction"
                stroke="#f59e0b"
                fill="#f59e0b"
                fillOpacity={0.3}
                strokeWidth={2}
              />
              <Legend />
            </RadarChart>
          </ResponsiveContainer>
        </Card>
      </div>

      {/* Conversation Flow Visualization */}
      <Card className="p-6">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <Eye className="w-5 h-5 text-indigo-600" />
            Live Conversation Flow Analysis
          </h3>
          <Button variant="outline" size="sm" className="flex items-center gap-2">
            <Share2 className="w-4 h-4" />
            Share Analysis
          </Button>
        </div>

        <div className="space-y-4">
          {conversationFlows.slice(0, 8).map((flow, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
              className="flex items-center justify-between p-4 bg-gradient-to-r from-neutral-50 to-neutral-100 dark:from-neutral-800 dark:to-neutral-700 rounded-lg border border-neutral-200 dark:border-neutral-600"
            >
              <div className="flex items-center gap-4">
                <div className="w-8 h-8 bg-gradient-to-br from-purple-500 to-pink-500 rounded-full flex items-center justify-center text-white font-bold text-sm">
                  {flow.turn}
                </div>
                
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-sm font-medium text-neutral-900 dark:text-white">
                      {flow.userIntent.charAt(0).toUpperCase() + flow.userIntent.slice(1)}
                    </span>
                    <span className={`px-2 py-0.5 text-xs rounded-full ${
                      flow.emotionalTone === 'positive' 
                        ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300' 
                        : flow.emotionalTone === 'negative' 
                          ? 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300'
                          : 'bg-neutral-100 text-neutral-700 dark:bg-neutral-900/30 dark:text-neutral-300'
                    }`}>
                      {flow.emotionalTone}
                    </span>
                  </div>
                  <div className="text-xs text-neutral-600 dark:text-neutral-400">
                    Response time: {flow.responseTime.toFixed(2)}s | Coherence: {(flow.coherenceScore * 100).toFixed(1)}%
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-3">
                <div className={`w-3 h-3 rounded-full ${
                  flow.contextPreserved ? 'bg-green-500' : 'bg-red-500'
                }`} title={flow.contextPreserved ? 'Context preserved' : 'Context lost'} />
                
                <div className="w-16 bg-neutral-200 dark:bg-neutral-600 rounded-full h-2">
                  <div 
                    className="bg-gradient-to-r from-purple-500 to-pink-500 h-2 rounded-full"
                    style={{ width: `${flow.confidence * 100}%` }}
                  />
                </div>
                
                <span className="text-sm font-medium text-neutral-700 dark:text-neutral-300">
                  {(flow.confidence * 100).toFixed(0)}%
                </span>
              </div>
            </motion.div>
          ))}
        </div>
      </Card>

      {/* Real-time Insights */}
      {isLiveMode && (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="fixed bottom-6 right-6 max-w-md"
        >
          <Card className="p-4 bg-gradient-to-br from-purple-500 to-pink-500 text-white border-none shadow-2xl">
            <div className="flex items-center gap-2 mb-2">
              <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
              <span className="font-semibold">Live Analysis Active</span>
            </div>
            <p className="text-sm text-purple-100">
              Monitoring conversation flows in real-time. Detecting {Math.floor(Math.random() * 5) + 1} active sessions.
            </p>
          </Card>
        </motion.div>
      )}
    </div>
  );
};

export default ConversationAnalytics;