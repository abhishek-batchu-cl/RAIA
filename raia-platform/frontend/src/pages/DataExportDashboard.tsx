import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Download,
  FileText,
  File,
  Table,
  Clock,
  CheckCircle,
  AlertCircle,
  RefreshCw,
  Settings,
  BarChart3,
  Calendar,
  Filter,
  Eye
} from 'lucide-react';
import Card from '../components/common/Card';
import MetricCard from '../components/common/MetricCard';
import Button from '../components/common/Button';

interface ExportJob {
  id: string;
  filename: string;
  format: 'pdf' | 'excel' | 'csv';
  type: 'evaluation_report' | 'model_performance' | 'fairness_analysis' | 'data_drift';
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  fileSizeBytes: number;
  createdAt: string;
  completedAt?: string;
  downloadUrl?: string;
  expiresAt: string;
}

interface ExportFormat {
  key: string;
  name: string;
  description: string;
  icon: React.ComponentType<any>;
  features: string[];
  extension: string;
}

const DataExportDashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'overview' | 'jobs' | 'formats' | 'settings'>('overview');
  const [selectedFormat, setSelectedFormat] = useState<string>('pdf');
  const [selectedType, setSelectedType] = useState<string>('evaluation_report');
  const [isExporting, setIsExporting] = useState(false);
  const [filterStatus, setFilterStatus] = useState<string>('all');

  const exportFormats: ExportFormat[] = [
    {
      key: 'pdf',
      name: 'PDF Report',
      description: 'Professional formatted report with charts and tables',
      icon: FileText,
      features: ['formatted_layout', 'charts', 'tables', 'branding'],
      extension: '.pdf'
    },
    {
      key: 'excel',
      name: 'Excel Workbook',
      description: 'Multi-sheet workbook with data and analysis',
      icon: Table,
      features: ['multiple_sheets', 'charts', 'formulas', 'formatting'],
      extension: '.xlsx'
    },
    {
      key: 'csv',
      name: 'CSV Data',
      description: 'Raw data in comma-separated values format',
      icon: File,
      features: ['raw_data', 'lightweight', 'universal_compatibility'],
      extension: '.csv'
    }
  ];

  const exportTypes = [
    {
      key: 'evaluation_report',
      name: 'Evaluation Report',
      description: 'Comprehensive model evaluation results'
    },
    {
      key: 'model_performance',
      name: 'Model Performance Report',
      description: 'Detailed model performance analysis over time'
    },
    {
      key: 'fairness_analysis',
      name: 'Fairness Analysis Report',
      description: 'Bias detection and fairness metrics analysis'
    },
    {
      key: 'data_drift',
      name: 'Data Drift Report',
      description: 'Data distribution changes and drift analysis'
    }
  ];

  // Mock data - would come from API
  const [exportJobs] = useState<ExportJob[]>([
    {
      id: 'export_001',
      filename: 'model_evaluation_20240130.pdf',
      format: 'pdf',
      type: 'evaluation_report',
      status: 'completed',
      progress: 100,
      fileSizeBytes: 2457600,
      createdAt: '2024-01-30T10:30:00Z',
      completedAt: '2024-01-30T10:32:15Z',
      downloadUrl: '/downloads/export_001',
      expiresAt: '2024-02-06T10:30:00Z'
    },
    {
      id: 'export_002',
      filename: 'fairness_analysis_20240130.xlsx',
      format: 'excel',
      type: 'fairness_analysis',
      status: 'processing',
      progress: 67,
      fileSizeBytes: 0,
      createdAt: '2024-01-30T11:15:00Z',
      expiresAt: '2024-02-06T11:15:00Z'
    },
    {
      id: 'export_003',
      filename: 'drift_data_20240129.csv',
      format: 'csv',
      type: 'data_drift',
      status: 'completed',
      progress: 100,
      fileSizeBytes: 856432,
      createdAt: '2024-01-29T16:45:00Z',
      completedAt: '2024-01-29T16:46:30Z',
      downloadUrl: '/downloads/export_003',
      expiresAt: '2024-02-05T16:45:00Z'
    },
    {
      id: 'export_004',
      filename: 'performance_report_20240130.pdf',
      format: 'pdf',
      type: 'model_performance',
      status: 'failed',
      progress: 0,
      fileSizeBytes: 0,
      createdAt: '2024-01-30T12:00:00Z',
      expiresAt: '2024-02-06T12:00:00Z'
    }
  ]);

  const overviewMetrics = [
    {
      title: 'Total Exports',
      value: exportJobs.length.toString(),
      change: '+4',
      changeType: 'positive' as const,
      icon: Download,
      description: 'Reports exported this month'
    },
    {
      title: 'Success Rate',
      value: '87%',
      change: '+5%',
      changeType: 'positive' as const,
      icon: CheckCircle,
      description: 'Successful export completion rate'
    },
    {
      title: 'Avg Export Time',
      value: '2.3min',
      change: '-15%',
      changeType: 'positive' as const,
      icon: Clock,
      description: 'Average time to complete exports'
    },
    {
      title: 'Storage Used',
      value: '1.2GB',
      change: '+12%',
      changeType: 'neutral' as const,
      icon: BarChart3,
      description: 'Total export file storage'
    }
  ];

  const handleStartExport = async () => {
    setIsExporting(true);
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 2000));
    setIsExporting(false);
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatTimeAgo = (dateString: string): string => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    const diffDays = Math.floor(diffHours / 24);
    
    if (diffDays > 0) return `${diffDays} day${diffDays > 1 ? 's' : ''} ago`;
    if (diffHours > 0) return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
    return 'Recently';
  };

  const getStatusIcon = (status: ExportJob['status']) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'processing':
        return <RefreshCw className="w-4 h-4 text-blue-500 animate-spin" />;
      case 'failed':
        return <AlertCircle className="w-4 h-4 text-red-500" />;
      default:
        return <Clock className="w-4 h-4 text-yellow-500" />;
    }
  };

  const getFormatIcon = (format: string) => {
    const formatData = exportFormats.find(f => f.key === format);
    if (!formatData) return File;
    return formatData.icon;
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
              icon={React.createElement(metric.icon, { className: "w-5 h-5" })}
              description={metric.description}
            />
          </motion.div>
        ))}
      </div>

      {/* Quick Export */}
      <Card>
        <div className="p-6">
          <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-6">
            Quick Export
          </h3>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                  Export Type
                </label>
                <select
                  value={selectedType}
                  onChange={(e) => setSelectedType(e.target.value)}
                  className="w-full px-3 py-2 border border-neutral-300 dark:border-neutral-600 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
                >
                  {exportTypes.map(type => (
                    <option key={type.key} value={type.key}>{type.name}</option>
                  ))}
                </select>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                  Export Format
                </label>
                <div className="grid grid-cols-3 gap-2">
                  {exportFormats.map(format => {
                    const Icon = format.icon;
                    return (
                      <button
                        key={format.key}
                        onClick={() => setSelectedFormat(format.key)}
                        className={`p-3 rounded-lg border text-center transition-all ${
                          selectedFormat === format.key
                            ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400'
                            : 'border-neutral-200 dark:border-neutral-700 hover:border-neutral-300 dark:hover:border-neutral-600'
                        }`}
                      >
                        <Icon className="w-5 h-5 mx-auto mb-1" />
                        <div className="text-xs font-medium">{format.name}</div>
                      </button>
                    );
                  })}
                </div>
              </div>
            </div>

            <div className="space-y-4">
              <div className="bg-neutral-50 dark:bg-neutral-800 rounded-lg p-4">
                <h4 className="font-medium text-neutral-900 dark:text-neutral-100 mb-2">
                  Export Preview
                </h4>
                <div className="text-sm text-neutral-600 dark:text-neutral-400 space-y-1">
                  <div>Type: {exportTypes.find(t => t.key === selectedType)?.name}</div>
                  <div>Format: {exportFormats.find(f => f.key === selectedFormat)?.name}</div>
                  <div>Estimated size: ~2.4 MB</div>
                  <div>Estimated time: ~30 seconds</div>
                </div>
              </div>
              
              <Button
                variant="primary"
                onClick={handleStartExport}
                disabled={isExporting}
                className="w-full"
              >
                {isExporting ? (
                  <>
                    <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                    Exporting...
                  </>
                ) : (
                  <>
                    <Download className="w-4 h-4 mr-2" />
                    Start Export
                  </>
                )}
              </Button>
            </div>
          </div>
        </div>
      </Card>

      {/* Recent Exports */}
      <Card>
        <div className="p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
              Recent Exports
            </h3>
            <Button variant="outline" size="sm" onClick={() => setActiveTab('jobs')}>
              View All
            </Button>
          </div>
          
          <div className="space-y-3">
            {exportJobs.slice(0, 4).map((job, index) => {
              const FormatIcon = getFormatIcon(job.format);
              return (
                <motion.div
                  key={job.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="flex items-center justify-between p-3 border border-neutral-200 dark:border-neutral-700 rounded-lg"
                >
                  <div className="flex items-center space-x-3">
                    <FormatIcon className="w-5 h-5 text-neutral-500" />
                    <div>
                      <div className="font-medium text-neutral-900 dark:text-neutral-100 text-sm">
                        {job.filename}
                      </div>
                      <div className="text-xs text-neutral-600 dark:text-neutral-400">
                        {exportTypes.find(t => t.key === job.type)?.name} • {formatTimeAgo(job.createdAt)}
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-3">
                    {job.status === 'processing' && (
                      <div className="flex items-center space-x-2">
                        <div className="w-16 bg-neutral-200 dark:bg-neutral-700 rounded-full h-1.5">
                          <div
                            className="bg-blue-500 h-1.5 rounded-full transition-all duration-300"
                            style={{ width: `${job.progress}%` }}
                          />
                        </div>
                        <span className="text-xs text-neutral-600 dark:text-neutral-400">
                          {job.progress}%
                        </span>
                      </div>
                    )}
                    
                    <div className="flex items-center space-x-1">
                      {getStatusIcon(job.status)}
                      <span className="text-xs text-neutral-600 dark:text-neutral-400 capitalize">
                        {job.status}
                      </span>
                    </div>
                    
                    {job.status === 'completed' && job.downloadUrl && (
                      <Button variant="outline" size="sm">
                        <Download className="w-3 h-3 mr-1" />
                        Download
                      </Button>
                    )}
                  </div>
                </motion.div>
              );
            })}
          </div>
        </div>
      </Card>
    </div>
  );

  const renderJobs = () => (
    <Card>
      <div className="p-6">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
            Export Jobs
          </h3>
          <div className="flex items-center space-x-4">
            <select
              value={filterStatus}
              onChange={(e) => setFilterStatus(e.target.value)}
              className="px-3 py-2 border border-neutral-300 dark:border-neutral-600 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
            >
              <option value="all">All Status</option>
              <option value="completed">Completed</option>
              <option value="processing">Processing</option>
              <option value="failed">Failed</option>
              <option value="pending">Pending</option>
            </select>
            <Button variant="primary" onClick={handleStartExport}>
              New Export
            </Button>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-neutral-200 dark:border-neutral-700">
                <th className="text-left p-3 font-medium text-neutral-900 dark:text-neutral-100">File</th>
                <th className="text-left p-3 font-medium text-neutral-900 dark:text-neutral-100">Type</th>
                <th className="text-left p-3 font-medium text-neutral-900 dark:text-neutral-100">Status</th>
                <th className="text-left p-3 font-medium text-neutral-900 dark:text-neutral-100">Size</th>
                <th className="text-left p-3 font-medium text-neutral-900 dark:text-neutral-100">Created</th>
                <th className="text-left p-3 font-medium text-neutral-900 dark:text-neutral-100">Expires</th>
                <th className="text-left p-3 font-medium text-neutral-900 dark:text-neutral-100">Actions</th>
              </tr>
            </thead>
            <tbody>
              {exportJobs
                .filter(job => filterStatus === 'all' || job.status === filterStatus)
                .map((job) => {
                  const FormatIcon = getFormatIcon(job.format);
                  return (
                    <tr key={job.id} className="border-b border-neutral-100 dark:border-neutral-800 hover:bg-neutral-50 dark:hover:bg-neutral-800/50">
                      <td className="p-3">
                        <div className="flex items-center space-x-3">
                          <FormatIcon className="w-4 h-4 text-neutral-500" />
                          <div>
                            <div className="font-medium text-neutral-900 dark:text-neutral-100">
                              {job.filename}
                            </div>
                            <div className="text-xs text-neutral-600 dark:text-neutral-400">
                              {job.format.toUpperCase()}
                            </div>
                          </div>
                        </div>
                      </td>
                      <td className="p-3 text-neutral-600 dark:text-neutral-400">
                        {exportTypes.find(t => t.key === job.type)?.name}
                      </td>
                      <td className="p-3">
                        <div className="flex items-center space-x-2">
                          {getStatusIcon(job.status)}
                          <span className={`px-2 py-1 rounded-full text-xs font-medium capitalize ${
                            job.status === 'completed' ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300' :
                            job.status === 'processing' ? 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300' :
                            job.status === 'failed' ? 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300' :
                            'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300'
                          }`}>
                            {job.status}
                          </span>
                        </div>
                        {job.status === 'processing' && (
                          <div className="mt-1 w-20 bg-neutral-200 dark:bg-neutral-700 rounded-full h-1">
                            <div
                              className="bg-blue-500 h-1 rounded-full transition-all duration-300"
                              style={{ width: `${job.progress}%` }}
                            />
                          </div>
                        )}
                      </td>
                      <td className="p-3 text-neutral-600 dark:text-neutral-400">
                        {job.fileSizeBytes > 0 ? formatFileSize(job.fileSizeBytes) : '-'}
                      </td>
                      <td className="p-3 text-neutral-600 dark:text-neutral-400">
                        {new Date(job.createdAt).toLocaleDateString()}
                      </td>
                      <td className="p-3 text-neutral-600 dark:text-neutral-400">
                        {new Date(job.expiresAt).toLocaleDateString()}
                      </td>
                      <td className="p-3">
                        <div className="flex items-center space-x-2">
                          {job.status === 'completed' && job.downloadUrl && (
                            <Button variant="outline" size="sm">
                              <Download className="w-3 h-3 mr-1" />
                              Download
                            </Button>
                          )}
                          <Button variant="ghost" size="sm">
                            <Eye className="w-3 h-3" />
                          </Button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
            </tbody>
          </table>
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
            Data Export Dashboard
          </h1>
          <p className="text-neutral-600 dark:text-neutral-400 mt-1">
            Export reports and data in multiple formats (PDF, Excel, CSV)
          </p>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="flex space-x-1 bg-neutral-100 dark:bg-neutral-800 p-1 rounded-lg w-fit">
        {[
          { key: 'overview', label: 'Overview', icon: BarChart3 },
          { key: 'jobs', label: 'Export Jobs', icon: Clock },
          { key: 'formats', label: 'Formats', icon: Settings },
          { key: 'settings', label: 'Settings', icon: Filter }
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
      {activeTab === 'jobs' && renderJobs()}
      {activeTab === 'formats' && (
        <Card>
          <div className="p-6">
            <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-6">
              Supported Export Formats
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {exportFormats.map((format, index) => {
                const Icon = format.icon;
                return (
                  <motion.div
                    key={format.key}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.1 }}
                  >
                    <Card>
                      <div className="p-4">
                        <div className="flex items-center space-x-3 mb-4">
                          <Icon className="w-6 h-6 text-blue-600 dark:text-blue-400" />
                          <div>
                            <h4 className="font-medium text-neutral-900 dark:text-neutral-100">
                              {format.name}
                            </h4>
                            <p className="text-sm text-neutral-600 dark:text-neutral-400">
                              {format.extension}
                            </p>
                          </div>
                        </div>
                        <p className="text-sm text-neutral-600 dark:text-neutral-400 mb-4">
                          {format.description}
                        </p>
                        <div className="space-y-2">
                          {format.features.map(feature => (
                            <div key={feature} className="flex items-center space-x-2">
                              <CheckCircle className="w-3 h-3 text-green-500" />
                              <span className="text-xs text-neutral-600 dark:text-neutral-400">
                                {feature.replace('_', ' ')}
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    </Card>
                  </motion.div>
                );
              })}
            </div>
          </div>
        </Card>
      )}
      {activeTab === 'settings' && (
        <Card>
          <div className="p-6">
            <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4">
              Export Settings (Coming Soon)
            </h3>
            <p className="text-neutral-600 dark:text-neutral-400">
              Configure export preferences, retention policies, and notification settings.
            </p>
          </div>
        </Card>
      )}
    </div>
  );
};

export default DataExportDashboard;