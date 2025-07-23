import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  FileText, Download, Calendar, Clock, Settings, 
  Share2, Eye, Edit3, Copy, Mail, Printer, 
  BarChart3, TrendingUp, AlertTriangle, CheckCircle,
  Brain, Target, Users, Shield, Zap, Star,
  Play, Pause, RefreshCw, Filter, Search,
  ChevronDown, ChevronRight, Maximize2, Minimize2
} from 'lucide-react';
import Card from '@/components/common/Card';
import Button from '@/components/common/Button';
import { apiClient } from '@/services/api';

interface ReportTemplate {
  id: string;
  name: string;
  description: string;
  type: 'executive' | 'technical' | 'compliance' | 'performance';
  frequency: 'daily' | 'weekly' | 'monthly' | 'quarterly' | 'on-demand';
  sections: ReportSection[];
  created_at: Date;
  last_generated: Date;
  next_scheduled?: Date;
  is_active: boolean;
  recipients: string[];
}

interface ReportSection {
  id: string;
  title: string;
  type: 'summary' | 'metrics' | 'charts' | 'insights' | 'recommendations' | 'compliance';
  content_source: string;
  ai_enhanced: boolean;
  order: number;
}

interface GeneratedReport {
  id: string;
  template_id: string;
  title: string;
  type: string;
  status: 'generating' | 'completed' | 'failed' | 'scheduled';
  generated_at: Date;
  file_size: string;
  format: 'pdf' | 'html' | 'powerpoint' | 'word';
  download_url?: string;
  preview_url?: string;
  shared_with: string[];
}

interface ReportMetrics {
  total_reports: number;
  scheduled_reports: number;
  reports_this_month: number;
  avg_generation_time: number;
  most_popular_template: string;
  success_rate: number;
}

const AutomatedReportGeneration: React.FC = () => {
  const [reportTemplates, setReportTemplates] = useState<ReportTemplate[]>([]);
  const [generatedReports, setGeneratedReports] = useState<GeneratedReport[]>([]);
  const [reportMetrics, setReportMetrics] = useState<ReportMetrics | null>(null);
  const [selectedTemplate, setSelectedTemplate] = useState<ReportTemplate | null>(null);
  const [showTemplateEditor, setShowTemplateEditor] = useState(false);
  const [viewMode, setViewMode] = useState<'templates' | 'reports' | 'scheduled'>('templates');
  const [isGenerating, setIsGenerating] = useState(false);
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set());

  // Mock data
  const mockMetrics: ReportMetrics = {
    total_reports: 247,
    scheduled_reports: 12,
    reports_this_month: 38,
    avg_generation_time: 2.3,
    most_popular_template: 'Executive Performance Dashboard',
    success_rate: 0.972
  };

  const mockTemplates: ReportTemplate[] = [
    {
      id: 'tmpl-1',
      name: 'Executive Performance Dashboard',
      description: 'High-level performance summary for executives and stakeholders',
      type: 'executive',
      frequency: 'weekly',
      sections: [
        { id: 'sec-1', title: 'Executive Summary', type: 'summary', content_source: 'ai_generated', ai_enhanced: true, order: 1 },
        { id: 'sec-2', title: 'Key Metrics Overview', type: 'metrics', content_source: 'system_metrics', ai_enhanced: true, order: 2 },
        { id: 'sec-3', title: 'Performance Trends', type: 'charts', content_source: 'performance_data', ai_enhanced: false, order: 3 },
        { id: 'sec-4', title: 'AI Insights & Recommendations', type: 'insights', content_source: 'ml_insights', ai_enhanced: true, order: 4 },
        { id: 'sec-5', title: 'Strategic Recommendations', type: 'recommendations', content_source: 'ai_analysis', ai_enhanced: true, order: 5 }
      ],
      created_at: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000),
      last_generated: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000),
      next_scheduled: new Date(Date.now() + 5 * 24 * 60 * 60 * 1000),
      is_active: true,
      recipients: ['executives@company.com', 'board@company.com']
    },
    {
      id: 'tmpl-2',
      name: 'Model Performance Report',
      description: 'Detailed technical analysis of ML model performance and metrics',
      type: 'technical',
      frequency: 'daily',
      sections: [
        { id: 'sec-6', title: 'Model Performance Summary', type: 'metrics', content_source: 'model_metrics', ai_enhanced: true, order: 1 },
        { id: 'sec-7', title: 'Accuracy & Precision Trends', type: 'charts', content_source: 'performance_charts', ai_enhanced: false, order: 2 },
        { id: 'sec-8', title: 'Data Drift Analysis', type: 'insights', content_source: 'drift_detection', ai_enhanced: true, order: 3 },
        { id: 'sec-9', title: 'Anomaly Detection Results', type: 'insights', content_source: 'anomaly_data', ai_enhanced: true, order: 4 }
      ],
      created_at: new Date(Date.now() - 45 * 24 * 60 * 60 * 1000),
      last_generated: new Date(Date.now() - 6 * 60 * 60 * 1000),
      next_scheduled: new Date(Date.now() + 18 * 60 * 60 * 1000),
      is_active: true,
      recipients: ['data-team@company.com', 'ml-engineers@company.com']
    },
    {
      id: 'tmpl-3',
      name: 'Compliance & Bias Report',
      description: 'Regulatory compliance, fairness metrics, and bias assessment report',
      type: 'compliance',
      frequency: 'monthly',
      sections: [
        { id: 'sec-10', title: 'Compliance Overview', type: 'compliance', content_source: 'compliance_data', ai_enhanced: true, order: 1 },
        { id: 'sec-11', title: 'Fairness Metrics', type: 'metrics', content_source: 'bias_metrics', ai_enhanced: true, order: 2 },
        { id: 'sec-12', title: 'Regulatory Alignment', type: 'compliance', content_source: 'regulatory_check', ai_enhanced: true, order: 3 },
        { id: 'sec-13', title: 'Risk Assessment', type: 'insights', content_source: 'risk_analysis', ai_enhanced: true, order: 4 }
      ],
      created_at: new Date(Date.now() - 60 * 24 * 60 * 60 * 1000),
      last_generated: new Date(Date.now() - 15 * 24 * 60 * 60 * 1000),
      next_scheduled: new Date(Date.now() + 15 * 24 * 60 * 60 * 1000),
      is_active: true,
      recipients: ['compliance@company.com', 'legal@company.com', 'risk@company.com']
    }
  ];

  const mockGeneratedReports: GeneratedReport[] = [
    {
      id: 'report-1',
      template_id: 'tmpl-1',
      title: 'Executive Performance Dashboard - Week 42',
      type: 'executive',
      status: 'completed',
      generated_at: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000),
      file_size: '2.4 MB',
      format: 'pdf',
      download_url: '/reports/exec-dashboard-w42.pdf',
      preview_url: '/reports/exec-dashboard-w42.html',
      shared_with: ['executives@company.com']
    },
    {
      id: 'report-2',
      template_id: 'tmpl-2',
      title: 'Model Performance Report - October 25',
      type: 'technical',
      status: 'completed',
      generated_at: new Date(Date.now() - 6 * 60 * 60 * 1000),
      file_size: '5.7 MB',
      format: 'pdf',
      download_url: '/reports/model-perf-oct25.pdf',
      shared_with: ['data-team@company.com']
    },
    {
      id: 'report-3',
      template_id: 'tmpl-1',
      title: 'Executive Performance Dashboard - Week 41',
      type: 'executive',
      status: 'completed',
      generated_at: new Date(Date.now() - 9 * 24 * 60 * 60 * 1000),
      file_size: '2.1 MB',
      format: 'pdf',
      download_url: '/reports/exec-dashboard-w41.pdf',
      shared_with: ['executives@company.com']
    }
  ];

  useEffect(() => {
    loadReportData();
  }, []);

  const loadReportData = async () => {
    try {
      // In production, this would call the API
      setReportTemplates(mockTemplates);
      setGeneratedReports(mockGeneratedReports);
      setReportMetrics(mockMetrics);
    } catch (error) {
      console.error('Error loading report data:', error);
      setReportTemplates(mockTemplates);
      setGeneratedReports(mockGeneratedReports);
      setReportMetrics(mockMetrics);
    }
  };

  const generateReport = async (templateId: string) => {
    setIsGenerating(true);
    
    try {
      // Simulate report generation
      await new Promise(resolve => setTimeout(resolve, 3000));
      
      const template = reportTemplates.find(t => t.id === templateId);
      if (template) {
        const newReport: GeneratedReport = {
          id: `report-${Date.now()}`,
          template_id: templateId,
          title: `${template.name} - ${new Date().toLocaleDateString()}`,
          type: template.type,
          status: 'completed',
          generated_at: new Date(),
          file_size: '3.2 MB',
          format: 'pdf',
          download_url: `/reports/generated-${Date.now()}.pdf`,
          shared_with: template.recipients
        };
        
        setGeneratedReports(prev => [newReport, ...prev]);
      }
    } catch (error) {
      console.error('Error generating report:', error);
    } finally {
      setIsGenerating(false);
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'executive': return <Crown className="w-5 h-5" />;
      case 'technical': return <Brain className="w-5 h-5" />;
      case 'compliance': return <Shield className="w-5 h-5" />;
      case 'performance': return <TrendingUp className="w-5 h-5" />;
      default: return <FileText className="w-5 h-5" />;
    }
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'executive': return 'bg-purple-100 text-purple-600 dark:bg-purple-900/20 dark:text-purple-400';
      case 'technical': return 'bg-blue-100 text-blue-600 dark:bg-blue-900/20 dark:text-blue-400';
      case 'compliance': return 'bg-green-100 text-green-600 dark:bg-green-900/20 dark:text-green-400';
      case 'performance': return 'bg-orange-100 text-orange-600 dark:bg-orange-900/20 dark:text-orange-400';
      default: return 'bg-gray-100 text-gray-600 dark:bg-gray-900/20 dark:text-gray-400';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'generating': return <RefreshCw className="w-4 h-4 text-blue-500 animate-spin" />;
      case 'failed': return <AlertTriangle className="w-4 h-4 text-red-500" />;
      case 'scheduled': return <Clock className="w-4 h-4 text-yellow-500" />;
      default: return <Clock className="w-4 h-4 text-gray-500" />;
    }
  };

  const formatTimeAgo = (date: Date) => {
    const now = new Date();
    const diffInHours = Math.floor((now.getTime() - date.getTime()) / (1000 * 60 * 60));
    
    if (diffInHours < 1) return 'Just now';
    if (diffInHours < 24) return `${diffInHours}h ago`;
    return `${Math.floor(diffInHours / 24)}d ago`;
  };

  const toggleSection = (sectionId: string) => {
    setExpandedSections(prev => {
      const newSet = new Set(prev);
      if (newSet.has(sectionId)) {
        newSet.delete(sectionId);
      } else {
        newSet.add(sectionId);
      }
      return newSet;
    });
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-neutral-900 dark:text-neutral-100 flex items-center">
            <div className="w-8 h-8 bg-gradient-to-r from-orange-500 to-red-500 rounded-lg flex items-center justify-center mr-3">
              <FileText className="w-5 h-5 text-white" />
            </div>
            Automated Report Generation
          </h1>
          <p className="text-neutral-600 dark:text-neutral-400 mt-1">
            AI-powered report generation with executive summaries and automated scheduling
          </p>
        </div>
        
        <div className="flex items-center space-x-2">
          <Button
            variant="outline"
            size="sm"
            leftIcon={<Settings className="w-4 h-4" />}
          >
            Settings
          </Button>
          <Button
            variant="outline"
            size="sm"
            leftIcon={<Calendar className="w-4 h-4" />}
          >
            Schedule
          </Button>
          <Button
            variant="primary"
            size="sm"
            leftIcon={<Plus className="w-4 h-4" />}
            onClick={() => setShowTemplateEditor(true)}
          >
            New Template
          </Button>
        </div>
      </div>

      {/* Metrics Cards */}
      {reportMetrics && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-6 gap-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
          >
            <Card className="bg-gradient-to-r from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 border-blue-200 dark:border-blue-800">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-blue-700 dark:text-blue-300">Total Reports</p>
                  <p className="text-2xl font-bold text-blue-900 dark:text-blue-100">
                    {reportMetrics.total_reports}
                  </p>
                </div>
                <FileText className="w-8 h-8 text-blue-500" />
              </div>
            </Card>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            <Card className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 border-green-200 dark:border-green-800">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-green-700 dark:text-green-300">Scheduled</p>
                  <p className="text-2xl font-bold text-green-900 dark:text-green-100">
                    {reportMetrics.scheduled_reports}
                  </p>
                </div>
                <Calendar className="w-8 h-8 text-green-500" />
              </div>
            </Card>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
          >
            <Card className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 border-purple-200 dark:border-purple-800">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-purple-700 dark:text-purple-300">This Month</p>
                  <p className="text-2xl font-bold text-purple-900 dark:text-purple-100">
                    {reportMetrics.reports_this_month}
                  </p>
                </div>
                <TrendingUp className="w-8 h-8 text-purple-500" />
              </div>
            </Card>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
          >
            <Card className="bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 border-yellow-200 dark:border-yellow-800">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-yellow-700 dark:text-yellow-300">Avg Time</p>
                  <p className="text-2xl font-bold text-yellow-900 dark:text-yellow-100">
                    {reportMetrics.avg_generation_time}m
                  </p>
                </div>
                <Clock className="w-8 h-8 text-yellow-500" />
              </div>
            </Card>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
          >
            <Card className="bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 border-indigo-200 dark:border-indigo-800">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-indigo-700 dark:text-indigo-300">Success Rate</p>
                  <p className="text-2xl font-bold text-indigo-900 dark:text-indigo-100">
                    {(reportMetrics.success_rate * 100).toFixed(1)}%
                  </p>
                </div>
                <CheckCircle className="w-8 h-8 text-indigo-500" />
              </div>
            </Card>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6 }}
          >
            <Card className="bg-gradient-to-r from-rose-50 to-pink-50 dark:from-rose-900/20 dark:to-pink-900/20 border-rose-200 dark:border-rose-800">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-rose-700 dark:text-rose-300 text-xs">Most Popular</p>
                  <p className="text-sm font-bold text-rose-900 dark:text-rose-100 leading-tight">
                    Executive Dashboard
                  </p>
                </div>
                <Star className="w-8 h-8 text-rose-500" />
              </div>
            </Card>
          </motion.div>
        </div>
      )}

      {/* View Mode Tabs */}
      <div className="flex items-center justify-center">
        <div className="bg-neutral-100 dark:bg-neutral-800 rounded-lg p-1 flex">
          {[
            { id: 'templates', label: 'Templates', icon: <FileText className="w-4 h-4" /> },
            { id: 'reports', label: 'Generated Reports', icon: <BarChart3 className="w-4 h-4" /> },
            { id: 'scheduled', label: 'Scheduled', icon: <Calendar className="w-4 h-4" /> },
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

      {/* Content based on view mode */}
      <AnimatePresence mode="wait">
        {viewMode === 'templates' && (
          <motion.div
            key="templates"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="grid grid-cols-1 lg:grid-cols-2 gap-6"
          >
            {reportTemplates.map((template, index) => (
              <motion.div
                key={template.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
              >
                <Card className="hover:shadow-lg transition-all duration-200">
                  <div className="space-y-4">
                    {/* Template Header */}
                    <div className="flex items-start justify-between">
                      <div className="flex items-start space-x-3">
                        <div className={`p-2 rounded-lg ${getTypeColor(template.type)}`}>
                          {getTypeIcon(template.type)}
                        </div>
                        <div className="flex-1">
                          <h3 className="font-semibold text-neutral-900 dark:text-neutral-100 mb-1">
                            {template.name}
                          </h3>
                          <p className="text-sm text-neutral-600 dark:text-neutral-400 line-clamp-2">
                            {template.description}
                          </p>
                        </div>
                      </div>
                      
                      <div className="flex items-center space-x-1">
                        <div className={`w-2 h-2 rounded-full ${template.is_active ? 'bg-green-500' : 'bg-gray-400'}`}></div>
                        <span className="text-xs text-neutral-500">
                          {template.is_active ? 'Active' : 'Inactive'}
                        </span>
                      </div>
                    </div>
                    
                    {/* Template Info */}
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <p className="text-neutral-500 dark:text-neutral-400">Frequency</p>
                        <p className="font-medium text-neutral-900 dark:text-neutral-100 capitalize">
                          {template.frequency}
                        </p>
                      </div>
                      <div>
                        <p className="text-neutral-500 dark:text-neutral-400">Recipients</p>
                        <p className="font-medium text-neutral-900 dark:text-neutral-100">
                          {template.recipients.length} recipients
                        </p>
                      </div>
                    </div>
                    
                    {/* Sections Preview */}
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <h4 className="text-sm font-medium text-neutral-900 dark:text-neutral-100">
                          Sections ({template.sections.length})
                        </h4>
                        <button
                          onClick={() => toggleSection(template.id)}
                          className="text-neutral-400 hover:text-neutral-600 dark:hover:text-neutral-300"
                        >
                          {expandedSections.has(template.id) ? 
                            <ChevronDown className="w-4 h-4" /> : 
                            <ChevronRight className="w-4 h-4" />
                          }
                        </button>
                      </div>
                      
                      <AnimatePresence>
                        {expandedSections.has(template.id) && (
                          <motion.div
                            initial={{ opacity: 0, height: 0 }}
                            animate={{ opacity: 1, height: 'auto' }}
                            exit={{ opacity: 0, height: 0 }}
                            className="space-y-2"
                          >
                            {template.sections.map((section) => (
                              <div
                                key={section.id}
                                className="flex items-center justify-between p-2 bg-neutral-50 dark:bg-neutral-800 rounded"
                              >
                                <div className="flex items-center space-x-2">
                                  <span className="text-xs font-medium text-neutral-600 dark:text-neutral-400">
                                    {section.order}.
                                  </span>
                                  <span className="text-sm text-neutral-900 dark:text-neutral-100">
                                    {section.title}
                                  </span>
                                  {section.ai_enhanced && (
                                    <Brain className="w-3 h-3 text-purple-500" title="AI Enhanced" />
                                  )}
                                </div>
                                <span className="text-xs px-2 py-1 bg-neutral-200 dark:bg-neutral-700 text-neutral-600 dark:text-neutral-300 rounded">
                                  {section.type}
                                </span>
                              </div>
                            ))}
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>
                    
                    {/* Template Actions */}
                    <div className="flex items-center justify-between pt-3 border-t border-neutral-200 dark:border-neutral-700">
                      <div className="text-xs text-neutral-500 dark:text-neutral-400">
                        Last generated: {formatTimeAgo(template.last_generated)}
                      </div>
                      
                      <div className="flex items-center space-x-2">
                        <Button
                          variant="outline"
                          size="sm"
                          leftIcon={<Eye className="w-4 h-4" />}
                          onClick={() => setSelectedTemplate(template)}
                        >
                          Preview
                        </Button>
                        <Button
                          variant="primary"
                          size="sm"
                          leftIcon={isGenerating ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
                          onClick={() => generateReport(template.id)}
                          disabled={isGenerating}
                        >
                          {isGenerating ? 'Generating...' : 'Generate'}
                        </Button>
                      </div>
                    </div>
                  </div>
                </Card>
              </motion.div>
            ))}
          </motion.div>
        )}

        {viewMode === 'reports' && (
          <motion.div
            key="reports"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
          >
            <Card title="Generated Reports" icon={<BarChart3 className="w-5 h-5 text-primary-500" />}>
              <div className="space-y-4">
                {generatedReports.map((report, index) => (
                  <motion.div
                    key={report.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.1 }}
                    className="flex items-center justify-between p-4 bg-neutral-50 dark:bg-neutral-800 rounded-lg hover:shadow-md transition-shadow duration-200"
                  >
                    <div className="flex items-start space-x-4 flex-1">
                      <div className={`p-2 rounded-lg ${getTypeColor(report.type)}`}>
                        {getTypeIcon(report.type)}
                      </div>
                      
                      <div className="flex-1">
                        <div className="flex items-center space-x-2 mb-1">
                          <h4 className="font-medium text-neutral-900 dark:text-neutral-100">
                            {report.title}
                          </h4>
                          {getStatusIcon(report.status)}
                        </div>
                        
                        <div className="flex items-center space-x-4 text-sm text-neutral-600 dark:text-neutral-400">
                          <span>Generated: {formatTimeAgo(report.generated_at)}</span>
                          <span>Size: {report.file_size}</span>
                          <span>Format: {report.format.toUpperCase()}</span>
                          <span>Shared with: {report.shared_with.length} recipients</span>
                        </div>
                      </div>
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      {report.preview_url && (
                        <Button
                          variant="ghost"
                          size="sm"
                          leftIcon={<Eye className="w-4 h-4" />}
                        >
                          Preview
                        </Button>
                      )}
                      <Button
                        variant="outline"
                        size="sm"
                        leftIcon={<Share2 className="w-4 h-4" />}
                      >
                        Share
                      </Button>
                      {report.download_url && (
                        <Button
                          variant="primary"
                          size="sm"
                          leftIcon={<Download className="w-4 h-4" />}
                        >
                          Download
                        </Button>
                      )}
                    </div>
                  </motion.div>
                ))}
              </div>
            </Card>
          </motion.div>
        )}

        {viewMode === 'scheduled' && (
          <motion.div
            key="scheduled"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
          >
            <Card title="Scheduled Reports" icon={<Calendar className="w-5 h-5 text-primary-500" />}>
              <div className="space-y-4">
                {reportTemplates
                  .filter(template => template.is_active && template.next_scheduled)
                  .map((template, index) => (
                    <motion.div
                      key={template.id}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.1 }}
                      className="flex items-center justify-between p-4 bg-neutral-50 dark:bg-neutral-800 rounded-lg"
                    >
                      <div className="flex items-center space-x-4">
                        <div className={`p-2 rounded-lg ${getTypeColor(template.type)}`}>
                          {getTypeIcon(template.type)}
                        </div>
                        
                        <div>
                          <h4 className="font-medium text-neutral-900 dark:text-neutral-100 mb-1">
                            {template.name}
                          </h4>
                          <div className="flex items-center space-x-4 text-sm text-neutral-600 dark:text-neutral-400">
                            <span>Frequency: {template.frequency}</span>
                            <span>Recipients: {template.recipients.length}</span>
                            {template.next_scheduled && (
                              <span>
                                Next run: {template.next_scheduled.toLocaleDateString()} at {template.next_scheduled.toLocaleTimeString()}
                              </span>
                            )}
                          </div>
                        </div>
                      </div>
                      
                      <div className="flex items-center space-x-2">
                        <Button
                          variant="ghost"
                          size="sm"
                          leftIcon={<Edit3 className="w-4 h-4" />}
                        >
                          Edit
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          leftIcon={<Pause className="w-4 h-4" />}
                        >
                          Pause
                        </Button>
                        <Button
                          variant="primary"
                          size="sm"
                          leftIcon={<Play className="w-4 h-4" />}
                        >
                          Run Now
                        </Button>
                      </div>
                    </motion.div>
                  ))}
              </div>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Empty states can be added here */}
    </div>
  );
};

export default AutomatedReportGeneration;