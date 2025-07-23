import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  BarChart3, Target, TrendingUp, Users, User, Zap, GitBranch, Network,
  ChevronRight, ChevronDown, Activity, PieChart, Settings, FileText,
  Database, Brain, Eye, CheckCircle, Clock, AlertCircle, Shield, Bell,
  BarChart, FileCheck, Search, Plug, Crown, Layout, FolderOpen,
  Settings as SettingsIcon, TrendingDown, Home, HelpCircle, Lightbulb,
  PlayCircle, Bookmark, Star, Award, AlertTriangle
} from 'lucide-react';
import { cn } from '../../utils';
import { useAuth } from '../../contexts/AuthContext';

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
  activeTab: string;
  onTabChange: (tabId: string) => void;
  modelType: 'classification' | 'regression';
}

interface NavigationItem {
  id: string;
  label: string;
  icon: React.ComponentType<any>;
  description: string;
  badge?: string;
  urgent?: boolean;
  difficulty?: 'beginner' | 'intermediate' | 'advanced';
  estimatedTime?: string;
  requiredPermission?: string;
}

interface NavigationSection {
  id: string;
  title: string;
  subtitle: string;
  icon: React.ComponentType<any>;
  items: NavigationItem[];
  defaultExpanded?: boolean;
  userType?: ('all' | 'admin' | 'analyst' | 'viewer')[];
}

const SidebarUX: React.FC<SidebarProps> = ({
  isOpen,
  onClose,
  activeTab,
  onTabChange,
  modelType,
}) => {
  const { user, hasPermission } = useAuth();
  const [expandedSections, setExpandedSections] = useState<string[]>(['get-started', 'insights']);
  const [searchQuery, setSearchQuery] = useState('');
  const [showOnboarding, setShowOnboarding] = useState(false);

  // UX-Optimized Navigation with User Personas in Mind
  const navigationStructure: NavigationSection[] = [
    {
      id: 'get-started',
      title: '🚀 Get Started',
      subtitle: 'Quick wins and immediate insights',
      icon: PlayCircle,
      defaultExpanded: true,
      userType: ['all'],
      items: [
        {
          id: 'overview',
          label: 'Dashboard Home',
          icon: Home,
          description: 'Your model at a glance',
          difficulty: 'beginner',
          estimatedTime: '2 min',
        },
        {
          id: 'model-performance',
          label: 'How Good is My Model?',
          icon: Award,
          description: 'Performance summary & key metrics',
          difficulty: 'beginner',
          estimatedTime: '3 min',
          badge: 'Popular',
        },
      ]
    },
    {
      id: 'insights',
      title: '💡 Understand Predictions',
      subtitle: 'Why did my model decide this?',
      icon: Lightbulb,
      defaultExpanded: true,
      userType: ['all'],
      items: [
        {
          id: 'feature-importance',
          label: 'What Features Matter?',
          icon: BarChart3,
          description: 'Feature importance ranking',
          difficulty: 'beginner',
          estimatedTime: '5 min',
        },
        {
          id: modelType === 'classification' ? 'classification-stats' : 'regression-stats',
          label: modelType === 'classification' ? 'Classification Deep Dive' : 'Regression Analysis',
          icon: modelType === 'classification' ? Target : PieChart,
          description: modelType === 'classification' 
            ? 'Accuracy, confusion matrix, ROC curves' 
            : 'R², residuals, error analysis',
          difficulty: 'intermediate',
          estimatedTime: '8 min',
        },
        {
          id: 'predictions',
          label: 'Individual Cases',
          icon: Eye,
          description: 'Why this specific prediction?',
          difficulty: 'intermediate',
          estimatedTime: '6 min',
          badge: 'Interactive',
        },
      ]
    },
    {
      id: 'explore',
      title: '🔬 Advanced Analysis',
      subtitle: 'Deep dive into model behavior',
      icon: Brain,
      defaultExpanded: false,
      userType: ['all'],
      items: [
        {
          id: 'what-if',
          label: 'What-If Scenarios',
          icon: Zap,
          description: 'Test different input combinations',
          difficulty: 'intermediate',
          estimatedTime: '10 min',
          badge: 'Powerful',
        },
        {
          id: 'feature-dependence',
          label: 'Feature Relationships',
          icon: Network,
          description: 'How features interact with each other',
          difficulty: 'advanced',
          estimatedTime: '12 min',
        },
        {
          id: 'feature-interactions',
          label: 'Complex Interactions',
          icon: Plug,
          description: 'Advanced feature interaction analysis',
          difficulty: 'advanced',
          estimatedTime: '15 min',
        },
        {
          id: 'decision-trees',
          label: 'Decision Logic',
          icon: GitBranch,
          description: 'How decisions flow through the model',
          difficulty: 'intermediate',
          estimatedTime: '8 min',
        },
      ]
    },
    {
      id: 'monitoring',
      title: '📊 Monitor & Maintain',
      subtitle: 'Keep your model healthy',
      icon: Activity,
      defaultExpanded: false,
      userType: ['admin', 'analyst'],
      items: [
        {
          id: 'data-drift',
          label: 'Data Drift Detection',
          icon: TrendingDown,
          description: 'Is your data changing over time?',
          difficulty: 'intermediate',
          estimatedTime: '7 min',
          urgent: true,
          badge: 'Alert',
        },
        {
          id: 'model-monitoring',
          label: 'Performance Monitoring',
          icon: Activity,
          description: 'Track model health over time',
          difficulty: 'intermediate',
          estimatedTime: '6 min',
        },
        {
          id: 'data-quality',
          label: 'Data Quality Check',
          icon: Shield,
          description: 'Validate input data quality',
          difficulty: 'beginner',
          estimatedTime: '4 min',
        },
      ]
    },
    {
      id: 'business',
      title: '💼 Business Impact',
      subtitle: 'Connect ML to business value',
      icon: Crown,
      defaultExpanded: false,
      userType: ['admin', 'analyst'],
      items: [
        {
          id: 'executive-dashboard',
          label: 'Executive Summary',
          icon: BarChart,
          description: 'C-suite KPIs and business metrics',
          difficulty: 'beginner',
          estimatedTime: '5 min',
          requiredPermission: 'read:executive',
        },
        {
          id: 'enterprise-fairness',
          label: 'Fairness & Bias Analysis',
          icon: Shield,
          description: 'Ensure fair and ethical AI',
          difficulty: 'advanced',
          estimatedTime: '20 min',
          badge: 'Compliance',
        },
        {
          id: 'compliance-reports',
          label: 'Compliance Reporting',
          icon: FileCheck,
          description: 'Regulatory compliance documentation',
          difficulty: 'intermediate',
          estimatedTime: '10 min',
          requiredPermission: 'read:compliance',
        },
      ]
    },
    {
      id: 'system',
      title: '⚙️ System Management',
      subtitle: 'Configure and maintain',
      icon: Settings,
      defaultExpanded: false,
      userType: ['admin'],
      items: [
        {
          id: 'user-management',
          label: 'User Management',
          icon: Users,
          description: 'Manage users and permissions',
          difficulty: 'beginner',
          estimatedTime: '5 min',
          requiredPermission: 'admin:users',
        },
        {
          id: 'system-health',
          label: 'System Health',
          icon: Activity,
          description: 'Monitor system performance',
          difficulty: 'intermediate',
          estimatedTime: '7 min',
          requiredPermission: 'system:monitor',
        },
        {
          id: 'settings',
          label: 'Settings',
          icon: SettingsIcon,
          description: 'Configure system preferences',
          difficulty: 'beginner',
          estimatedTime: '3 min',
          requiredPermission: 'system:config',
        },
      ]
    },
  ];

  const toggleSection = (sectionId: string) => {
    setExpandedSections(prev =>
      prev.includes(sectionId)
        ? prev.filter(id => id !== sectionId)
        : [...prev, sectionId]
    );
  };

  const getDifficultyColor = (difficulty?: string) => {
    switch (difficulty) {
      case 'beginner': return 'text-green-600 bg-green-100 dark:text-green-400 dark:bg-green-900/30';
      case 'intermediate': return 'text-amber-600 bg-amber-100 dark:text-amber-400 dark:bg-amber-900/30';
      case 'advanced': return 'text-red-600 bg-red-100 dark:text-red-400 dark:bg-red-900/30';
      default: return 'text-neutral-600 bg-neutral-100 dark:text-neutral-400 dark:bg-neutral-900/30';
    }
  };

  const getBadgeColor = (badge?: string) => {
    switch (badge) {
      case 'Popular': return 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300';
      case 'Interactive': return 'bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-300';
      case 'Powerful': return 'bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-300';
      case 'Alert': return 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300';
      case 'Compliance': return 'bg-indigo-100 text-indigo-700 dark:bg-indigo-900/30 dark:text-indigo-300';
      default: return 'bg-neutral-100 text-neutral-700 dark:bg-neutral-900/30 dark:text-neutral-300';
    }
  };

  const filteredSections = navigationStructure.filter(section => {
    if (!section.userType) return true;
    if (section.userType.includes('all')) return true;
    return user && section.userType.includes(user.role);
  });

  const searchResults = searchQuery.length > 0 
    ? filteredSections.flatMap(section => 
        section.items.filter(item => 
          item.label.toLowerCase().includes(searchQuery.toLowerCase()) ||
          item.description.toLowerCase().includes(searchQuery.toLowerCase())
        ).map(item => ({ ...item, sectionTitle: section.title }))
      )
    : [];

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
                <h2 className="text-lg font-bold text-neutral-900 dark:text-white">Navigation</h2>
                <button
                  onClick={() => setShowOnboarding(!showOnboarding)}
                  className="p-2 rounded-lg hover:bg-neutral-100 dark:hover:bg-neutral-800 text-neutral-500"
                >
                  <HelpCircle className="w-5 h-5" />
                </button>
              </div>

              {/* Search */}
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-neutral-400" />
                <input
                  type="text"
                  placeholder="Find features..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pl-10 pr-4 py-2 border border-neutral-200 dark:border-neutral-700 rounded-lg bg-neutral-50 dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100 placeholder-neutral-500 focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                />
              </div>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto">
              {/* Onboarding Banner */}
              {showOnboarding && (
                <div className="m-4 p-3 bg-primary-50 dark:bg-primary-900/20 border border-primary-200 dark:border-primary-800 rounded-lg">
                  <div className="flex items-start space-x-2">
                    <Lightbulb className="w-4 h-4 text-primary-600 dark:text-primary-400 mt-0.5 flex-shrink-0" />
                    <div className="text-sm text-primary-700 dark:text-primary-300">
                      <p className="font-medium">New to ML Explainability?</p>
                      <p className="text-xs mt-1">Start with "Get Started" section. Green badges = beginner-friendly!</p>
                    </div>
                  </div>
                </div>
              )}

              {/* Search Results */}
              {searchQuery.length > 0 && (
                <div className="p-4">
                  <h3 className="text-sm font-medium text-neutral-600 dark:text-neutral-400 mb-3">
                    Search Results ({searchResults.length})
                  </h3>
                  {searchResults.map((item) => (
                    <button
                      key={item.id}
                      onClick={() => {
                        onTabChange(item.id);
                        setSearchQuery('');
                      }}
                      className="w-full text-left p-3 rounded-lg hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-colors mb-2"
                    >
                      <div className="flex items-center space-x-3">
                        <item.icon className="w-4 h-4 text-primary-600 dark:text-primary-400" />
                        <div>
                          <p className="font-medium text-sm">{item.label}</p>
                          <p className="text-xs text-neutral-500">{item.sectionTitle} • {item.description}</p>
                        </div>
                      </div>
                    </button>
                  ))}
                </div>
              )}

              {/* Navigation Sections */}
              {searchQuery.length === 0 && (
                <div className="p-4 space-y-6">
                  {filteredSections.map((section) => (
                    <div key={section.id}>
                      <button
                        onClick={() => toggleSection(section.id)}
                        className="w-full flex items-center justify-between p-2 text-left rounded-lg hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-colors"
                      >
                        <div className="flex items-center space-x-3">
                          <section.icon className="w-5 h-5 text-primary-600 dark:text-primary-400" />
                          <div>
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
                            <div className="ml-8 mt-2 space-y-2">
                              {section.items
                                .filter(item => !item.requiredPermission || hasPermission(item.requiredPermission))
                                .map((item) => (
                                  <button
                                    key={item.id}
                                    onClick={() => onTabChange(item.id)}
                                    className={cn(
                                      "w-full text-left p-3 rounded-lg transition-all duration-200 border-2",
                                      activeTab === item.id
                                        ? 'bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300 border-primary-200 dark:border-primary-700 shadow-sm'
                                        : 'text-neutral-700 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-800 border-transparent hover:border-primary-200 dark:hover:border-primary-700',
                                      item.urgent && 'ring-2 ring-red-200 dark:ring-red-800'
                                    )}
                                  >
                                    <div className="flex items-start justify-between">
                                      <div className="flex items-start space-x-3">
                                        <item.icon className={cn(
                                          "w-4 h-4 mt-0.5 flex-shrink-0",
                                          activeTab === item.id
                                            ? 'text-primary-600 dark:text-primary-400'
                                            : 'text-neutral-500 dark:text-neutral-400'
                                        )} />
                                        <div className="min-w-0">
                                          <p className="font-medium text-sm">{item.label}</p>
                                          <p className="text-xs text-neutral-500 dark:text-neutral-400 mt-1">
                                            {item.description}
                                          </p>
                                          <div className="flex items-center space-x-2 mt-2">
                                            {item.estimatedTime && (
                                              <span className="text-xs px-2 py-0.5 bg-neutral-100 dark:bg-neutral-800 rounded-full">
                                                ⏱️ {item.estimatedTime}
                                              </span>
                                            )}
                                            {item.difficulty && (
                                              <span className={cn("text-xs px-2 py-0.5 rounded-full", getDifficultyColor(item.difficulty))}>
                                                {item.difficulty}
                                              </span>
                                            )}
                                          </div>
                                        </div>
                                      </div>
                                      <div className="flex flex-col items-end space-y-1">
                                        {item.badge && (
                                          <span className={cn("text-xs px-2 py-0.5 rounded-full font-medium", getBadgeColor(item.badge))}>
                                            {item.badge}
                                          </span>
                                        )}
                                        {item.urgent && (
                                          <AlertTriangle className="w-3 h-3 text-red-500" />
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
              )}
            </div>

            {/* Footer */}
            <div className="p-4 border-t border-neutral-200 dark:border-neutral-800">
              <div className="text-xs text-neutral-500 dark:text-neutral-400 text-center">
                💡 Tip: Use search to quickly find features
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
};

export default SidebarUX;