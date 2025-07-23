import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Database,
  Zap,
  Activity,
  Trash2,
  RefreshCw,
  Settings,
  BarChart3,
  Clock,
  HardDrive,
  Cpu,
  AlertTriangle,
  CheckCircle,
  TrendingUp,
  Search
} from 'lucide-react';
import Card from '../components/common/Card';
import MetricCard from '../components/common/MetricCard';
import Button from '../components/common/Button';

interface CacheCategory {
  name: string;
  prefix: string;
  defaultTtl: number;
  currentEntries: number;
  description: string;
  hitRate?: number;
}

interface CacheStats {
  usedMemory: number;
  usedMemoryHuman: string;
  connectedClients: number;
  totalCommandsProcessed: number;
  keyspaceHits: number;
  keyspaceMisses: number;
  hitRate: number;
}

const CacheManagementDashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'overview' | 'categories' | 'performance' | 'settings'>('overview');
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [searchKey, setSearchKey] = useState<string>('');
  const [isClearing, setIsClearing] = useState(false);

  // Mock data - would come from API
  const [cacheStats] = useState<CacheStats>({
    usedMemory: 52428800, // 50MB in bytes
    usedMemoryHuman: '50.0MB',
    connectedClients: 12,
    totalCommandsProcessed: 1245680,
    keyspaceHits: 89340,
    keyspaceMisses: 12450,
    hitRate: 87.8
  });

  const [categories] = useState<CacheCategory[]>([
    {
      name: 'model_predictions',
      prefix: 'pred:',
      defaultTtl: 1800,
      currentEntries: 15420,
      description: 'Cached model predictions and inference results',
      hitRate: 92.3
    },
    {
      name: 'evaluation_results',
      prefix: 'eval:',
      defaultTtl: 7200,
      currentEntries: 3240,
      description: 'Model evaluation results and metrics',
      hitRate: 88.7
    },
    {
      name: 'user_sessions',
      prefix: 'session:',
      defaultTtl: 3600,
      currentEntries: 1890,
      description: 'User session data and preferences',
      hitRate: 95.1
    },
    {
      name: 'dashboard_data',
      prefix: 'dashboard:',
      defaultTtl: 900,
      currentEntries: 5640,
      description: 'Dashboard widgets and analytics data',
      hitRate: 76.4
    },
    {
      name: 'model_metadata',
      prefix: 'model:',
      defaultTtl: 14400,
      currentEntries: 890,
      description: 'Model configurations and metadata',
      hitRate: 98.2
    },
    {
      name: 'analytics_data',
      prefix: 'analytics:',
      defaultTtl: 1800,
      currentEntries: 7230,
      description: 'Performance analytics and aggregated metrics',
      hitRate: 84.6
    }
  ]);

  const overviewMetrics = [
    {
      title: 'Hit Rate',
      value: `${cacheStats.hitRate}%`,
      change: '+2.1%',
      changeType: 'positive' as const,
      icon: Target,
      description: 'Cache hit rate across all categories'
    },
    {
      title: 'Memory Usage',
      value: cacheStats.usedMemoryHuman,
      change: '+5MB',
      changeType: 'neutral' as const,
      icon: HardDrive,
      description: 'Redis memory consumption'
    },
    {
      title: 'Active Entries',
      value: categories.reduce((sum, cat) => sum + cat.currentEntries, 0).toLocaleString(),
      change: '+1.2K',
      changeType: 'positive' as const,
      icon: Database,
      description: 'Total cached entries across all categories'
    },
    {
      title: 'Operations/sec',
      value: '847',
      change: '+12%',
      changeType: 'positive' as const,
      icon: Zap,
      description: 'Cache operations per second'
    }
  ];

  const handleClearCategory = async (categoryName: string) => {
    setIsClearing(true);
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 2000));
    setIsClearing(false);
  };

  const handleFlushAll = async () => {
    if (!confirm('Are you sure? This will clear ALL cache entries.')) return;
    setIsClearing(true);
    await new Promise(resolve => setTimeout(resolve, 3000));
    setIsClearing(false);
  };

  const renderOverview = () => (
    <div className="space-y-6">
      {/* Metrics Grid */}
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

      {/* Health Status */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <div className="p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
                Cache Health
              </h3>
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                <span className="text-sm font-medium text-green-600 dark:text-green-400">Healthy</span>
              </div>
            </div>
            
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm text-neutral-600 dark:text-neutral-400">Connection Status</span>
                <div className="flex items-center space-x-2">
                  <CheckCircle className="w-4 h-4 text-green-500" />
                  <span className="text-sm font-medium text-neutral-900 dark:text-neutral-100">Connected</span>
                </div>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-sm text-neutral-600 dark:text-neutral-400">Active Clients</span>
                <span className="text-sm font-medium text-neutral-900 dark:text-neutral-100">
                  {cacheStats.connectedClients}
                </span>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-sm text-neutral-600 dark:text-neutral-400">Commands Processed</span>
                <span className="text-sm font-medium text-neutral-900 dark:text-neutral-100">
                  {cacheStats.totalCommandsProcessed.toLocaleString()}
                </span>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-sm text-neutral-600 dark:text-neutral-400">Memory Efficiency</span>
                <div className="flex items-center space-x-2">
                  <div className="text-sm font-medium text-green-600 dark:text-green-400">Good</div>
                  <div className="w-16 bg-neutral-200 dark:bg-neutral-700 rounded-full h-2">
                    <div className="bg-green-500 h-2 rounded-full" style={{ width: '85%' }} />
                  </div>
                </div>
              </div>
            </div>
          </div>
        </Card>

        <Card>
          <div className="p-6">
            <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4">
              Performance Metrics
            </h3>
            
            <div className="space-y-4">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-neutral-600 dark:text-neutral-400">Hit Rate</span>
                  <span className="text-sm font-medium text-neutral-900 dark:text-neutral-100">
                    {cacheStats.hitRate}%
                  </span>
                </div>
                <div className="w-full bg-neutral-200 dark:bg-neutral-700 rounded-full h-2">
                  <div
                    className="bg-blue-500 h-2 rounded-full"
                    style={{ width: `${cacheStats.hitRate}%` }}
                  />
                </div>
              </div>
              
              <div className="grid grid-cols-2 gap-4 text-center">
                <div className="bg-green-50 dark:bg-green-900/20 p-3 rounded-lg">
                  <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                    {cacheStats.keyspaceHits.toLocaleString()}
                  </div>
                  <div className="text-xs text-green-700 dark:text-green-300">Hits</div>
                </div>
                <div className="bg-red-50 dark:bg-red-900/20 p-3 rounded-lg">
                  <div className="text-2xl font-bold text-red-600 dark:text-red-400">
                    {cacheStats.keyspaceMisses.toLocaleString()}
                  </div>
                  <div className="text-xs text-red-700 dark:text-red-300">Misses</div>
                </div>
              </div>
            </div>
          </div>
        </Card>
      </div>

      {/* Top Categories by Usage */}
      <Card>
        <div className="p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
              Categories by Usage
            </h3>
            <Button variant="outline" size="sm" onClick={() => setActiveTab('categories')}>
              Manage Categories
            </Button>
          </div>
          
          <div className="space-y-3">
            {categories
              .sort((a, b) => b.currentEntries - a.currentEntries)
              .slice(0, 4)
              .map((category, index) => (
                <motion.div
                  key={category.name}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="flex items-center justify-between p-3 border border-neutral-200 dark:border-neutral-700 rounded-lg"
                >
                  <div>
                    <div className="font-medium text-neutral-900 dark:text-neutral-100">
                      {category.name.replace('_', ' ')}
                    </div>
                    <div className="text-sm text-neutral-600 dark:text-neutral-400">
                      {category.description}
                    </div>
                  </div>
                  
                  <div className="text-right">
                    <div className="text-lg font-bold text-neutral-900 dark:text-neutral-100">
                      {category.currentEntries.toLocaleString()}
                    </div>
                    <div className="text-xs text-neutral-600 dark:text-neutral-400">
                      Hit Rate: {category.hitRate}%
                    </div>
                  </div>
                </motion.div>
              ))}
          </div>
        </div>
      </Card>
    </div>
  );

  const renderCategories = () => (
    <div className="space-y-6">
      <Card>
        <div className="p-6">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
              Cache Categories Management
            </h3>
            <div className="flex items-center space-x-4">
              <div className="relative">
                <Search className="w-4 h-4 absolute left-3 top-3 text-neutral-400" />
                <input
                  type="text"
                  placeholder="Search categories..."
                  value={searchKey}
                  onChange={(e) => setSearchKey(e.target.value)}
                  className="pl-10 pr-4 py-2 border border-neutral-300 dark:border-neutral-600 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
                />
              </div>
              <Button 
                variant="danger"
                onClick={handleFlushAll}
                disabled={isClearing}
              >
                {isClearing ? 'Clearing...' : 'Flush All'}
              </Button>
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-neutral-200 dark:border-neutral-700">
                  <th className="text-left p-3 font-medium text-neutral-900 dark:text-neutral-100">Category</th>
                  <th className="text-left p-3 font-medium text-neutral-900 dark:text-neutral-100">Prefix</th>
                  <th className="text-left p-3 font-medium text-neutral-900 dark:text-neutral-100">Entries</th>
                  <th className="text-left p-3 font-medium text-neutral-900 dark:text-neutral-100">TTL</th>
                  <th className="text-left p-3 font-medium text-neutral-900 dark:text-neutral-100">Hit Rate</th>
                  <th className="text-left p-3 font-medium text-neutral-900 dark:text-neutral-100">Actions</th>
                </tr>
              </thead>
              <tbody>
                {categories
                  .filter(cat => 
                    searchKey === '' || 
                    cat.name.toLowerCase().includes(searchKey.toLowerCase()) ||
                    cat.description.toLowerCase().includes(searchKey.toLowerCase())
                  )
                  .map((category) => (
                    <tr key={category.name} className="border-b border-neutral-100 dark:border-neutral-800 hover:bg-neutral-50 dark:hover:bg-neutral-800/50">
                      <td className="p-3">
                        <div>
                          <div className="font-medium text-neutral-900 dark:text-neutral-100">
                            {category.name.replace('_', ' ')}
                          </div>
                          <div className="text-xs text-neutral-600 dark:text-neutral-400">
                            {category.description}
                          </div>
                        </div>
                      </td>
                      <td className="p-3">
                        <code className="px-2 py-1 bg-neutral-100 dark:bg-neutral-800 rounded text-xs">
                          {category.prefix}
                        </code>
                      </td>
                      <td className="p-3">
                        <div className="font-medium text-neutral-900 dark:text-neutral-100">
                          {category.currentEntries.toLocaleString()}
                        </div>
                      </td>
                      <td className="p-3 text-neutral-600 dark:text-neutral-400">
                        {Math.floor(category.defaultTtl / 60)}min
                      </td>
                      <td className="p-3">
                        <div className="flex items-center space-x-2">
                          <div className="w-12 bg-neutral-200 dark:bg-neutral-700 rounded-full h-2">
                            <div
                              className={`h-2 rounded-full ${
                                (category.hitRate || 0) >= 90 ? 'bg-green-500' :
                                (category.hitRate || 0) >= 80 ? 'bg-yellow-500' : 'bg-red-500'
                              }`}
                              style={{ width: `${category.hitRate || 0}%` }}
                            />
                          </div>
                          <span className="text-sm font-medium text-neutral-900 dark:text-neutral-100">
                            {category.hitRate}%
                          </span>
                        </div>
                      </td>
                      <td className="p-3">
                        <div className="flex items-center space-x-2">
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => handleClearCategory(category.name)}
                            disabled={isClearing}
                          >
                            <Trash2 className="w-3 h-3 mr-1" />
                            Clear
                          </Button>
                          <Button variant="ghost" size="sm">
                            <Settings className="w-3 h-3" />
                          </Button>
                        </div>
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </div>
      </Card>
    </div>
  );

  const renderPerformance = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <div className="p-6">
            <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4">
              Real-time Performance
            </h3>
            
            <div className="space-y-6">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-neutral-600 dark:text-neutral-400">Operations/sec</span>
                  <span className="text-lg font-bold text-blue-600 dark:text-blue-400">847</span>
                </div>
                <div className="w-full bg-neutral-200 dark:bg-neutral-700 rounded-full h-2">
                  <div className="bg-blue-500 h-2 rounded-full" style={{ width: '84%' }} />
                </div>
              </div>
              
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-neutral-600 dark:text-neutral-400">Response Time (avg)</span>
                  <span className="text-lg font-bold text-green-600 dark:text-green-400">0.3ms</span>
                </div>
                <div className="w-full bg-neutral-200 dark:bg-neutral-700 rounded-full h-2">
                  <div className="bg-green-500 h-2 rounded-full" style={{ width: '95%' }} />
                </div>
              </div>
              
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-neutral-600 dark:text-neutral-400">CPU Usage</span>
                  <span className="text-lg font-bold text-yellow-600 dark:text-yellow-400">23%</span>
                </div>
                <div className="w-full bg-neutral-200 dark:bg-neutral-700 rounded-full h-2">
                  <div className="bg-yellow-500 h-2 rounded-full" style={{ width: '23%' }} />
                </div>
              </div>
            </div>
          </div>
        </Card>

        <Card>
          <div className="p-6">
            <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4">
              Memory Analysis
            </h3>
            
            <div className="space-y-4">
              <div className="text-center">
                <div className="text-3xl font-bold text-neutral-900 dark:text-neutral-100 mb-1">
                  {cacheStats.usedMemoryHuman}
                </div>
                <div className="text-sm text-neutral-600 dark:text-neutral-400">
                  of 512MB allocated
                </div>
              </div>
              
              <div className="w-full bg-neutral-200 dark:bg-neutral-700 rounded-full h-3">
                <div
                  className="bg-gradient-to-r from-blue-500 to-purple-500 h-3 rounded-full"
                  style={{ width: '10%' }}
                />
              </div>
              
              <div className="grid grid-cols-2 gap-4 text-center">
                <div className="bg-blue-50 dark:bg-blue-900/20 p-3 rounded-lg">
                  <div className="text-lg font-bold text-blue-600 dark:text-blue-400">78%</div>
                  <div className="text-xs text-blue-700 dark:text-blue-300">Data</div>
                </div>
                <div className="bg-purple-50 dark:bg-purple-900/20 p-3 rounded-lg">
                  <div className="text-lg font-bold text-purple-600 dark:text-purple-400">22%</div>
                  <div className="text-xs text-purple-700 dark:text-purple-300">Overhead</div>
                </div>
              </div>
            </div>
          </div>
        </Card>
      </div>

      <Card>
        <div className="p-6">
          <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4">
            Performance Trends (24h)
          </h3>
          
          <div className="h-64 bg-gradient-to-br from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-lg flex items-center justify-center border-2 border-dashed border-blue-200 dark:border-blue-700">
            <div className="text-center">
              <BarChart3 className="w-12 h-12 text-blue-400 mx-auto mb-2" />
              <p className="text-blue-600 dark:text-blue-400 font-medium">Performance Chart</p>
              <p className="text-sm text-blue-500 dark:text-blue-300 mt-1">
                Hit rate, memory usage, and operation trends
              </p>
            </div>
          </div>
        </div>
      </Card>
    </div>
  );

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-neutral-900 dark:text-neutral-100">
            Cache Management
          </h1>
          <p className="text-neutral-600 dark:text-neutral-400 mt-1">
            Monitor and manage Redis cache performance and storage
          </p>
        </div>
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2 px-3 py-2 bg-green-50 dark:bg-green-900/20 rounded-lg">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
            <span className="text-sm text-green-700 dark:text-green-300 font-medium">
              Redis Connected
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
          { key: 'categories', label: 'Categories', icon: Database },
          { key: 'performance', label: 'Performance', icon: Activity },
          { key: 'settings', label: 'Settings', icon: Settings }
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
      {activeTab === 'categories' && renderCategories()}
      {activeTab === 'performance' && renderPerformance()}
      {activeTab === 'settings' && (
        <Card>
          <div className="p-6">
            <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4">
              Cache Settings (Coming Soon)
            </h3>
            <p className="text-neutral-600 dark:text-neutral-400">
              Configure cache policies, TTL defaults, and connection settings.
            </p>
          </div>
        </Card>
      )}
    </div>
  );
};

export default CacheManagementDashboard;