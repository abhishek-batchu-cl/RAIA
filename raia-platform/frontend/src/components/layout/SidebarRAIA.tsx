import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  // Data Flow Icons
  Database, Plug, Upload, FileCheck, GitBranch, CheckCircle2,
  // Analytics Icons
  BarChart3, TrendingUp, Activity, PieChart, Target, Eye,
  // Responsible AI Icons
  Shield, Scale, Users, Heart, Lock, AlertTriangle,
  // Monitoring Icons
  Bell, AlertCircle, Gauge, TrendingDown, RefreshCw,
  // Business Icons
  Briefcase, DollarSign, Building2, FileText, Award,
  // System Icons
  Settings, Users2, HelpCircle, ChevronRight, ChevronDown,
  Home, Zap, Brain, Search, Filter, BookOpen, PlayCircle,
  Sparkles, Cpu, Network, Cloud, Server, Check, X, ArrowRight
} from 'lucide-react';
import { cn } from '../../utils';
import { useAuth } from '../../contexts/AuthContext';

interface SidebarRAIAProps {
  isOpen: boolean;
  onClose: () => void;
  activeTab: string;
  onTabChange: (tabId: string) => void;
  modelType: 'classification' | 'regression';
}

interface DataConnectionStatus {
  connected: boolean;
  source: string;
  lastSync?: Date;
  recordCount?: number;
}

// Enterprise Data Flow Architecture
const SidebarRAIA: React.FC<SidebarRAIAProps> = ({
  isOpen,
  onClose,
  activeTab,
  onTabChange,
  modelType,
}) => {
  const { user, hasPermission } = useAuth();
  const [expandedSections, setExpandedSections] = useState<string[]>(['data-pipeline']);
  const [dataStatus, setDataStatus] = useState<DataConnectionStatus | null>(null);
  const [showOnboardingWizard, setShowOnboardingWizard] = useState(false);

  // Check data connection status
  useEffect(() => {
    // In real app, this would check actual data connections
    const mockDataStatus: DataConnectionStatus = {
      connected: true,
      source: 'AWS S3 Data Lake',
      lastSync: new Date(),
      recordCount: 1250000,
    };
    setDataStatus(mockDataStatus);
  }, []);

  const toggleSection = (sectionId: string) => {
    setExpandedSections(prev =>
      prev.includes(sectionId)
        ? prev.filter(id => id !== sectionId)
        : [...prev, sectionId]
    );
  };

  // Enterprise Navigation Structure - Data Flow First
  const navigationStructure = [
    {
      id: 'data-pipeline',
      title: '📊 Data Pipeline',
      subtitle: 'Connect, validate, and prepare',
      icon: Database,
      priority: 'critical',
      items: [
        {
          id: 'data-connectivity',
          label: 'Data Sources',
          icon: Plug,
          description: 'Connect to enterprise data',
          status: dataStatus?.connected ? 'connected' : 'disconnected',
          badge: dataStatus?.connected ? 'Live' : 'Setup Required',
          priority: 'high',
        },
        {
          id: 'data-quality',
          label: 'Data Quality',
          icon: FileCheck,
          description: 'Validate and clean data',
          status: 'warning',
          badge: '3 Issues',
        },
        {
          id: 'data-preprocessing',
          label: 'Data Preparation',
          icon: GitBranch,
          description: 'Transform and engineer features',
          status: 'ready',
        },
        {
          id: 'stream-data-management',
          label: 'Real-time Streams',
          icon: Activity,
          description: 'Manage streaming data',
          status: 'active',
          badge: 'Beta',
        },
      ]
    },
    {
      id: 'responsible-ai',
      title: '🛡️ Responsible AI',
      subtitle: 'Fairness, bias, and compliance',
      icon: Shield,
      priority: 'critical',
      items: [
        {
          id: 'enterprise-fairness',
          label: 'Fairness Assessment',
          icon: Scale,
          description: '80+ bias metrics',
          badge: 'Compliant',
          status: 'success',
        },
        {
          id: 'bias-mitigation',
          label: 'Bias Mitigation',
          icon: Heart,
          description: 'Algorithmic debiasing',
          status: 'ready',
        },
        {
          id: 'compliance-reports',
          label: 'Compliance Hub',
          icon: FileText,
          description: 'GDPR, CCPA, SOC2',
          badge: 'Auto-Generated',
        },
        {
          id: 'ethical-ai',
          label: 'Ethical Guidelines',
          icon: BookOpen,
          description: 'AI governance policies',
          status: 'ready',
        },
      ]
    },
    {
      id: 'model-insights',
      title: '🧠 Model Intelligence',
      subtitle: 'Understand and explain',
      icon: Brain,
      items: [
        {
          id: 'overview',
          label: 'Executive Dashboard',
          icon: Home,
          description: 'High-level metrics',
          badge: 'Start Here',
        },
        {
          id: 'model-performance',
          label: 'Performance Analytics',
          icon: TrendingUp,
          description: 'Model accuracy & metrics',
        },
        {
          id: 'feature-importance',
          label: 'Feature Analysis',
          icon: BarChart3,
          description: 'What drives predictions',
        },
        {
          id: modelType === 'classification' ? 'classification-stats' : 'regression-stats',
          label: modelType === 'classification' ? 'Classification Metrics' : 'Regression Metrics',
          icon: modelType === 'classification' ? Target : PieChart,
          description: 'Detailed model statistics',
        },
      ]
    },
    {
      id: 'explainability',
      title: '💡 Explainability Suite',
      subtitle: 'Transparent AI decisions',
      icon: Eye,
      items: [
        {
          id: 'predictions',
          label: 'Prediction Explorer',
          icon: Search,
          description: 'Individual explanations',
        },
        {
          id: 'what-if',
          label: 'Scenario Analysis',
          icon: Zap,
          description: 'Test hypotheticals',
          badge: 'Interactive',
        },
        {
          id: 'feature-dependence',
          label: 'Dependencies',
          icon: Network,
          description: 'Feature relationships',
        },
        {
          id: 'root-cause-analysis',
          label: 'Root Cause Analysis',
          icon: GitBranch,
          description: 'Decision trees & paths',
        },
      ]
    },
    {
      id: 'monitoring',
      title: '🚨 AI Operations',
      subtitle: 'Monitor, alert, and maintain',
      icon: Bell,
      priority: 'high',
      items: [
        {
          id: 'model-monitoring',
          label: 'Model Health',
          icon: Activity,
          description: 'Real-time monitoring',
          status: 'active',
          badge: '↑ 99.8%',
        },
        {
          id: 'data-drift',
          label: 'Drift Detection',
          icon: TrendingDown,
          description: 'Data & concept drift',
          status: 'alert',
          badge: 'Alert',
        },
        {
          id: 'enterprise-alerts',
          label: 'Alert Center',
          icon: AlertCircle,
          description: 'Intelligent alerting',
          badge: '5 Active',
        },
        {
          id: 'ab-testing',
          label: 'A/B Testing',
          icon: GitBranch,
          description: 'Model experiments',
        },
      ]
    },
    {
      id: 'business',
      title: '💼 Business Impact',
      subtitle: 'ROI and value metrics',
      icon: Briefcase,
      requiredRole: ['admin', 'analyst'],
      items: [
        {
          id: 'executive-dashboard',
          label: 'C-Suite Analytics',
          icon: Building2,
          description: 'Business KPIs',
          badge: 'Executive',
        },
        {
          id: 'roi-calculator',
          label: 'ROI Calculator',
          icon: DollarSign,
          description: 'Value measurement',
        },
        {
          id: 'custom-dashboard',
          label: 'Custom Dashboards',
          icon: Filter,
          description: 'Build your views',
        },
      ]
    },
    {
      id: 'governance',
      title: '⚙️ Governance',
      subtitle: 'Control and configure',
      icon: Settings,
      requiredRole: ['admin'],
      items: [
        {
          id: 'user-management',
          label: 'User Access',
          icon: Users2,
          description: 'Manage permissions',
          requiredPermission: 'admin:users',
        },
        {
          id: 'model-management',
          label: 'Model Registry',
          icon: Server,
          description: 'Version control',
        },
        {
          id: 'system-health',
          label: 'System Health',
          icon: Gauge,
          description: 'Infrastructure status',
        },
        {
          id: 'settings',
          label: 'Configuration',
          icon: Settings,
          description: 'System settings',
        },
      ]
    },
  ];

  // Filter sections based on user role
  const filteredSections = navigationStructure.filter(section => {
    if (!section.requiredRole) return true;
    return user && section.requiredRole.includes(user.role);
  });

  const getStatusColor = (status?: string) => {
    switch (status) {
      case 'connected':
      case 'success':
      case 'ready':
        return 'text-green-600 bg-green-100 dark:text-green-400 dark:bg-green-900/30';
      case 'warning':
      case 'active':
        return 'text-amber-600 bg-amber-100 dark:text-amber-400 dark:bg-amber-900/30';
      case 'alert':
      case 'disconnected':
        return 'text-red-600 bg-red-100 dark:text-red-400 dark:bg-red-900/30';
      default:
        return 'text-neutral-600 bg-neutral-100 dark:text-neutral-400 dark:bg-neutral-900/30';
    }
  };

  const getBadgeStyle = (badge?: string) => {
    if (badge?.includes('Alert')) return 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300 animate-pulse';
    if (badge?.includes('Live')) return 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300';
    if (badge?.includes('Setup')) return 'bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-300';
    if (badge?.includes('Beta')) return 'bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-300';
    if (badge?.includes('Executive')) return 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300';
    return 'bg-neutral-100 text-neutral-700 dark:bg-neutral-900/30 dark:text-neutral-300';
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Mobile Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/20 z-40 lg:hidden"
            onClick={onClose}
          />

          {/* Sidebar */}
          <motion.div
            initial={{ x: -300 }}
            animate={{ x: 0 }}
            exit={{ x: -300 }}
            transition={{ type: 'spring', stiffness: 300, damping: 30 }}
            className="fixed left-0 top-0 bottom-0 w-80 bg-white dark:bg-neutral-900 border-r border-neutral-200 dark:border-neutral-800 z-50 overflow-hidden flex flex-col"
          >
            {/* Header */}
            <div className="p-4 border-b border-neutral-200 dark:border-neutral-800">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center space-x-3">
                  <div className="w-10 h-10 bg-gradient-to-br from-primary-500 to-accent-500 rounded-lg flex items-center justify-center">
                    <Shield className="w-5 h-5 text-white" />
                  </div>
                  <div>
                    <h2 className="text-lg font-bold text-neutral-900 dark:text-white">RAIA</h2>
                    <p className="text-xs text-neutral-600 dark:text-neutral-400">Responsible AI Analytics</p>
                  </div>
                </div>
                <button
                  onClick={() => setShowOnboardingWizard(true)}
                  className="p-2 rounded-lg hover:bg-neutral-100 dark:hover:bg-neutral-800 text-neutral-500"
                >
                  <PlayCircle className="w-5 h-5" />
                </button>
              </div>

              {/* Data Connection Status */}
              {dataStatus && (
                <div className={cn(
                  "p-3 rounded-lg border-2 transition-all",
                  dataStatus.connected
                    ? "bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800"
                    : "bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800"
                )}>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <Cloud className={cn(
                        "w-4 h-4",
                        dataStatus.connected ? "text-green-600 dark:text-green-400" : "text-red-600 dark:text-red-400"
                      )} />
                      <div>
                        <p className="text-sm font-medium text-neutral-900 dark:text-white">
                          {dataStatus.connected ? 'Connected' : 'Disconnected'}
                        </p>
                        <p className="text-xs text-neutral-600 dark:text-neutral-400">
                          {dataStatus.source}
                        </p>
                      </div>
                    </div>
                    {dataStatus.connected ? (
                      <Check className="w-4 h-4 text-green-600 dark:text-green-400" />
                    ) : (
                      <button className="px-2 py-1 text-xs bg-red-600 text-white rounded hover:bg-red-700">
                        Connect
                      </button>
                    )}
                  </div>
                  {dataStatus.connected && dataStatus.recordCount && (
                    <div className="mt-2 pt-2 border-t border-green-200 dark:border-green-800">
                      <div className="flex items-center justify-between text-xs text-neutral-600 dark:text-neutral-400">
                        <span>Records: {dataStatus.recordCount.toLocaleString()}</span>
                        <span>Last sync: Just now</span>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Navigation Sections */}
            <div className="flex-1 overflow-y-auto">
              <div className="p-4 space-y-4">
                {filteredSections.map((section) => (
                  <div key={section.id} className="space-y-2">
                    <button
                      onClick={() => toggleSection(section.id)}
                      className={cn(
                        "w-full flex items-center justify-between p-3 rounded-lg transition-all",
                        section.priority === 'critical' 
                          ? "bg-gradient-to-r from-primary-50 to-accent-50 dark:from-primary-900/20 dark:to-accent-900/20 border-2 border-primary-200 dark:border-primary-800"
                          : "hover:bg-neutral-100 dark:hover:bg-neutral-800"
                      )}
                    >
                      <div className="flex items-center space-x-3">
                        <section.icon className={cn(
                          "w-5 h-5",
                          section.priority === 'critical' 
                            ? "text-primary-600 dark:text-primary-400" 
                            : "text-neutral-600 dark:text-neutral-400"
                        )} />
                        <div className="text-left">
                          <p className="font-semibold text-sm text-neutral-900 dark:text-white">
                            {section.title}
                          </p>
                          <p className="text-xs text-neutral-500 dark:text-neutral-400">
                            {section.subtitle}
                          </p>
                        </div>
                      </div>
                      {expandedSections.includes(section.id) ? (
                        <ChevronDown className="w-4 h-4 text-neutral-400" />
                      ) : (
                        <ChevronRight className="w-4 h-4 text-neutral-400" />
                      )}
                    </button>

                    <AnimatePresence>
                      {expandedSections.includes(section.id) && (
                        <motion.div
                          initial={{ height: 0, opacity: 0 }}
                          animate={{ height: 'auto', opacity: 1 }}
                          exit={{ height: 0, opacity: 0 }}
                          transition={{ duration: 0.2 }}
                          className="overflow-hidden"
                        >
                          <div className="ml-8 space-y-1">
                            {section.items
                              .filter(item => !item.requiredPermission || hasPermission(item.requiredPermission))
                              .map((item) => (
                                <button
                                  key={item.id}
                                  onClick={() => onTabChange(item.id)}
                                  className={cn(
                                    "w-full text-left p-3 rounded-lg transition-all duration-200 border",
                                    activeTab === item.id
                                      ? 'bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300 border-primary-200 dark:border-primary-700'
                                      : 'text-neutral-700 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-800 border-transparent'
                                  )}
                                >
                                  <div className="flex items-start justify-between">
                                    <div className="flex items-start space-x-3">
                                      <item.icon className={cn(
                                        "w-4 h-4 mt-0.5",
                                        activeTab === item.id
                                          ? 'text-primary-600 dark:text-primary-400'
                                          : 'text-neutral-500 dark:text-neutral-400'
                                      )} />
                                      <div>
                                        <p className="font-medium text-sm">{item.label}</p>
                                        <p className="text-xs text-neutral-500 dark:text-neutral-400 mt-0.5">
                                          {item.description}
                                        </p>
                                      </div>
                                    </div>
                                    <div className="flex flex-col items-end space-y-1">
                                      {item.badge && (
                                        <span className={cn("text-xs px-2 py-0.5 rounded-full font-medium", getBadgeStyle(item.badge))}>
                                          {item.badge}
                                        </span>
                                      )}
                                      {item.status && (
                                        <span className={cn("text-xs px-2 py-0.5 rounded-full", getStatusColor(item.status))}>
                                          {item.status}
                                        </span>
                                      )}
                                    </div>
                                  </div>
                                </button>
                              ))}
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                ))}
              </div>
            </div>

            {/* Footer Quick Actions */}
            <div className="p-4 border-t border-neutral-200 dark:border-neutral-800 space-y-2">
              <button className="w-full flex items-center justify-center space-x-2 px-4 py-2 bg-primary-500 hover:bg-primary-600 text-white rounded-lg transition-colors">
                <Sparkles className="w-4 h-4" />
                <span className="text-sm font-medium">Quick Setup Wizard</span>
              </button>
              <div className="text-xs text-center text-neutral-500 dark:text-neutral-400">
                Enterprise v2.1.0 • SOC2 Compliant
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
};

export default SidebarRAIA;