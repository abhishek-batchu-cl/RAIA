import React, { useState } from 'react';
import { motion } from 'framer-motion';
import {
  Settings,
  Moon,
  Sun,
  Monitor,
  Download,
  User,
  Bell,
  Search,
  Menu,
  X,
  ChevronDown,
  LogOut,
  HelpCircle,
  Zap,
  Database,
  Command
} from 'lucide-react';
import { useTheme } from '../../contexts/ThemeContext';
import { useAuth } from '../../contexts/AuthContext';
import RateLimitStatus from '../common/RateLimitStatus';
import NotificationCenter from '../common/NotificationCenter';
import Breadcrumb from '../common/Breadcrumb';
import ContextualHelp from '../common/ContextualHelp';
import SmartSearchBar from '../common/SmartSearchBar';
import ThemeToggle from '../common/ThemeToggle';
import KeyboardShortcutsModal from '../common/KeyboardShortcutsModal';
import { useRAIAShortcuts } from '../../hooks/useKeyboardShortcuts';
import { cn } from '../../utils';

interface HeaderProps {
  onToggleSidebar: () => void;
  isSidebarOpen: boolean;
  modelType: 'classification' | 'regression';
  onModelTypeChange: (type: 'classification' | 'regression') => void;
  onShowHelp: () => void;
  onShowTour?: () => void;
  onTabChange?: (tabId: string) => void;
  activeTab?: string;
}

const Header: React.FC<HeaderProps> = ({
  onToggleSidebar,
  isSidebarOpen,
  modelType,
  onModelTypeChange,
  onShowHelp,
  onShowTour,
  onTabChange,
  activeTab = 'overview',
}) => {
  const { theme, setMode, isDark } = useTheme();
  const { user, logout } = useAuth();
  const [showUserMenu, setShowUserMenu] = useState(false);
  const [showThemeMenu, setShowThemeMenu] = useState(false);
  const [showNotifications, setShowNotifications] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [showShortcutsModal, setShowShortcutsModal] = useState(false);
  const [showSmartSearch, setShowSmartSearch] = useState(false);

  // Initialize keyboard shortcuts
  useRAIAShortcuts({
    onSearch: () => setShowSmartSearch(true),
    onHelp: () => setShowShortcutsModal(true),
    onToggleSidebar: onToggleSidebar,
    onCloseModal: () => {
      setShowShortcutsModal(false);
      setShowSmartSearch(false);
      setShowUserMenu(false);
      setShowThemeMenu(false);
    },
    onExport: handleExport,
    onSettings: () => setShowUserMenu(true),
    onNotifications: () => setShowNotifications(true)
  });

  const themeOptions = [
    { value: 'light', label: 'Light', icon: Sun },
    { value: 'dark', label: 'Dark', icon: Moon },
    { value: 'system', label: 'System', icon: Monitor },
  ];

  // Notification count for the indicator
  const notificationCount = 3; // This would come from your notification context/API

  // Generate breadcrumb items based on current tab - RAIA Enterprise
  const getBreadcrumbItems = () => {
    const pageMap: Record<string, { label: string; description: string }> = {
      // Data Pipeline
      'data-connectivity': { label: 'Data Sources', description: 'Connect enterprise data sources' },
      'data-quality': { label: 'Data Quality', description: 'Validate and monitor data integrity' },
      'data-preprocessing': { label: 'Data Preparation', description: 'Transform and engineer features' },
      'stream-data-management': { label: 'Real-time Streams', description: 'Manage streaming data pipelines' },
      
      // Responsible AI
      'enterprise-fairness': { label: 'Fairness Assessment', description: 'Bias detection and fairness metrics' },
      'bias-mitigation': { label: 'Bias Mitigation', description: 'Algorithmic debiasing strategies' },
      'compliance-reports': { label: 'Compliance Hub', description: 'Regulatory compliance documentation' },
      'ethical-ai': { label: 'Ethical Guidelines', description: 'AI governance and ethics policies' },
      
      // Model Intelligence
      'overview': { label: 'Executive Dashboard', description: 'High-level AI performance metrics' },
      'model-performance': { label: 'Performance Analytics', description: 'Model accuracy and business impact' },
      'feature-importance': { label: 'Feature Analysis', description: 'What drives AI decisions' },
      'classification-stats': { label: 'Classification Metrics', description: 'Precision, recall, and accuracy analysis' },
      'regression-stats': { label: 'Regression Metrics', description: 'R², residuals, and error analysis' },
      
      // Explainability Suite
      'predictions': { label: 'Prediction Explorer', description: 'Individual prediction explanations' },
      'what-if': { label: 'Scenario Analysis', description: 'Test hypothetical scenarios' },
      'feature-dependence': { label: 'Dependencies', description: 'Feature relationship analysis' },
      'root-cause-analysis': { label: 'Root Cause Analysis', description: 'Trace decision paths' },
      'decision-trees': { label: 'Decision Trees', description: 'Visualize decision logic' },
      
      // AI Operations
      'model-monitoring': { label: 'Model Health', description: 'Real-time model monitoring' },
      'data-drift': { label: 'Drift Detection', description: 'Data and concept drift monitoring' },
      'enterprise-alerts': { label: 'Alert Center', description: 'Intelligent alerting system' },
      'ab-testing': { label: 'A/B Testing', description: 'Model experimentation platform' },
      
      // Business Impact
      'executive-dashboard': { label: 'C-Suite Analytics', description: 'Executive AI performance dashboard' },
      'roi-calculator': { label: 'ROI Calculator', description: 'Measure AI business value' },
      'custom-dashboard': { label: 'Custom Dashboards', description: 'Build personalized views' },
      
      // Governance
      'user-management': { label: 'User Access', description: 'Manage team permissions' },
      'model-management': { label: 'Model Registry', description: 'Version control and lifecycle' },
      'system-health': { label: 'System Health', description: 'Infrastructure monitoring' },
      'settings': { label: 'Configuration', description: 'System preferences and settings' },
    };

    const currentPage = pageMap[activeTab] || { label: 'Unknown', description: '' };
    
    return [
      { id: 'overview', label: 'Home', description: 'Dashboard overview' },
      ...(activeTab !== 'overview' ? [{ 
        id: activeTab, 
        label: currentPage.label, 
        description: currentPage.description 
      }] : [])
    ];
  };

  const handleExport = () => {
    // TODO: Implement export functionality
    console.log('Export functionality to be implemented');
  };

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    // TODO: Implement search functionality
    console.log('Search query:', searchQuery);
  };

  return (
    <header className="bg-white/80 dark:bg-neutral-900/80 backdrop-blur-md border-b border-neutral-200 dark:border-neutral-800 sticky top-0 z-50">
      {/* Main Header Row */}
      <div className="max-w-full mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-14">
          {/* Left Section */}
          <div className="flex items-center space-x-4">
            {/* Sidebar Toggle */}
            <button
              onClick={onToggleSidebar}
              className="p-2 rounded-lg hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-colors lg:hidden"
              aria-label="Toggle sidebar"
            >
              {isSidebarOpen ? (
                <X className="w-5 h-5 text-neutral-700 dark:text-neutral-300" />
              ) : (
                <Menu className="w-5 h-5 text-neutral-700 dark:text-neutral-300" />
              )}
            </button>

            {/* Logo */}
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="flex items-center space-x-2"
            >
              <div className="w-8 h-8 bg-gradient-to-br from-primary-500 to-accent-500 rounded-lg flex items-center justify-center">
                <Zap className="w-4 h-4 text-white" />
              </div>
              <div className="hidden sm:block">
                <h1 className="text-lg font-bold text-gradient">
                  RAIA
                </h1>
                <p className="text-xs text-neutral-600 dark:text-neutral-400">
                  Responsible AI Analytics
                </p>
              </div>
            </motion.div>

            {/* Model Type Selector */}
            <div className="hidden md:flex items-center space-x-2">
              <span className="text-sm text-neutral-600 dark:text-neutral-400">
                Model Type:
              </span>
              <div className="flex bg-neutral-100 dark:bg-neutral-800 rounded-lg p-1">
                {['classification', 'regression'].map((type) => (
                  <button
                    key={type}
                    onClick={() => onModelTypeChange(type as 'classification' | 'regression')}
                    className={cn(
                      'px-3 py-1 text-sm font-medium rounded-md transition-all duration-200',
                      modelType === type
                        ? 'bg-white dark:bg-neutral-700 text-primary-600 dark:text-primary-400 shadow-sm'
                        : 'text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-neutral-200'
                    )}
                  >
                    {type.charAt(0).toUpperCase() + type.slice(1)}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Center Section - Smart Search */}
          <div className="hidden md:flex flex-1 max-w-md mx-4">
            <SmartSearchBar
              placeholder="Ask me anything about your AI models... (⌘K)"
              onSearch={(query) => {
                console.log('Smart search:', query);
                // TODO: Implement AI-powered search
              }}
              onSuggestionClick={(suggestion) => {
                console.log('Suggestion clicked:', suggestion);
                // TODO: Handle suggestion navigation
              }}
              showVoiceInput={true}
              showRecentSearches={true}
            />
          </div>

          {/* Right Section */}
          <div className="flex items-center space-x-2">
            {/* Data Connectivity Quick Access */}
            {onTabChange && (
              <button
                onClick={() => onTabChange('data-connectivity')}
                className="flex items-center space-x-2 px-3 py-2 rounded-lg bg-blue-50 hover:bg-blue-100 dark:bg-blue-900/20 dark:hover:bg-blue-900/30 text-blue-700 dark:text-blue-300 transition-all duration-200 border border-blue-200 dark:border-blue-700 hover:border-blue-300 dark:hover:border-blue-600"
                aria-label="Data Connectivity"
              >
                <Database className="w-4 h-4" />
                <span className="text-sm font-medium hidden sm:inline">Data</span>
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
              </button>
            )}

            {/* Export Button */}
            <button
              onClick={handleExport}
              className="p-2 rounded-lg hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-colors"
              aria-label="Export data"
            >
              <Download className="w-5 h-5 text-neutral-700 dark:text-neutral-300" />
            </button>

            {/* Rate Limit Status */}
            <div className="hidden sm:block">
              <RateLimitStatus className="mr-2" />
            </div>

            {/* Notifications */}
            <div className="relative">
              <button
                onClick={() => setShowNotifications(!showNotifications)}
                className="p-2 rounded-lg hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-colors relative"
                aria-label="Notifications"
              >
                <Bell className="w-5 h-5 text-neutral-700 dark:text-neutral-300" />
                {notificationCount > 0 && (
                  <span className="absolute -top-1 -right-1 w-3 h-3 bg-red-500 rounded-full"></span>
                )}
              </button>

              {/* NotificationCenter */}
              <NotificationCenter
                isOpen={showNotifications}
                onClose={() => setShowNotifications(false)}
                onNotificationClick={(notification) => {
                  // Handle navigation based on notification action
                  if (notification.actionUrl && onTabChange) {
                    if (notification.actionUrl === '/data-drift') {
                      onTabChange('data-drift');
                    } else if (notification.actionUrl === '/user-management') {
                      onTabChange('user-management');
                    } else if (notification.actionUrl === '/system-health') {
                      onTabChange('system-health');
                    }
                  }
                  setShowNotifications(false);
                }}
              />
            </div>

            {/* Tour Button */}
            {onShowTour && (
              <button
                onClick={onShowTour}
                className="p-2 rounded-lg hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-colors"
                aria-label="Show tour"
              >
                <Zap className="w-5 h-5 text-neutral-700 dark:text-neutral-300" />
              </button>
            )}
            
            {/* Help Button */}
            <button
              onClick={onShowHelp}
              className="p-2 rounded-lg hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-colors"
              aria-label="Show help guide"
            >
              <HelpCircle className="w-5 h-5 text-neutral-700 dark:text-neutral-300" />
            </button>

            {/* Shortcuts Button */}
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => setShowShortcutsModal(true)}
              className="p-2 rounded-lg hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-colors relative group"
              aria-label="Keyboard shortcuts (⌘/)"
            >
              <Command className="w-5 h-5 text-neutral-700 dark:text-neutral-300" />
              <div className="absolute -bottom-8 left-1/2 transform -translate-x-1/2 opacity-0 group-hover:opacity-100 transition-opacity">
                <div className="bg-black text-white text-xs rounded px-2 py-1 whitespace-nowrap">
                  ⌘/
                </div>
              </div>
            </motion.button>

            {/* Enhanced Theme Toggle */}
            <ThemeToggle
              currentTheme={theme.mode as 'light' | 'dark' | 'auto'}
              onThemeChange={(newTheme) => setMode(newTheme)}
              showCustomThemes={true}
              showScheduling={true}
              showAccessibility={true}
            />

            {/* User Menu */}
            <div className="relative">
              <button
                onClick={() => setShowUserMenu(!showUserMenu)}
                className="flex items-center space-x-2 p-2 rounded-lg hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-colors"
                aria-label="User menu"
              >
                <div className="w-8 h-8 bg-gradient-to-br from-primary-500 to-accent-500 rounded-full flex items-center justify-center">
                  <User className="w-4 h-4 text-white" />
                </div>
                <ChevronDown className="w-4 h-4 text-neutral-700 dark:text-neutral-300" />
              </button>

              {/* User Menu Dropdown */}
              {showUserMenu && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="absolute right-0 mt-2 w-48 bg-white dark:bg-neutral-800 rounded-lg shadow-lg border border-neutral-200 dark:border-neutral-700 z-50"
                >
                  <div className="p-4 border-b border-neutral-200 dark:border-neutral-700">
                    <p className="font-medium text-neutral-900 dark:text-neutral-100">
                      John Doe
                    </p>
                    <p className="text-sm text-neutral-600 dark:text-neutral-400">
                      john.doe@company.com
                    </p>
                  </div>
                  <div className="p-2">
                    <button className="w-full flex items-center space-x-2 px-3 py-2 text-sm text-neutral-700 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-700 rounded-lg transition-colors">
                      <Settings className="w-4 h-4" />
                      <span>Settings</span>
                    </button>
                    <button className="w-full flex items-center space-x-2 px-3 py-2 text-sm text-neutral-700 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-700 rounded-lg transition-colors">
                      <HelpCircle className="w-4 h-4" />
                      <span>Help</span>
                    </button>
                    <hr className="my-2 border-neutral-200 dark:border-neutral-700" />
                    <button className="w-full flex items-center space-x-2 px-3 py-2 text-sm text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-lg transition-colors">
                      <LogOut className="w-4 h-4" />
                      <span>Sign out</span>
                    </button>
                  </div>
                </motion.div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Mobile Search */}
      <div className="md:hidden px-4 pb-4">
        <form onSubmit={handleSearch} className="relative">
          <div className="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none">
            <Search className="w-4 h-4 text-neutral-400" />
          </div>
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search..."
            className="w-full pl-10 pr-4 py-2 border border-neutral-200 dark:border-neutral-700 rounded-lg 
                     bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100 
                     placeholder-neutral-500 dark:placeholder-neutral-400
                     focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent
                     transition-all duration-200"
          />
        </form>
      </div>

      {/* Breadcrumb and Contextual Help Row */}
      <div className="border-t border-neutral-200 dark:border-neutral-800 bg-neutral-50/50 dark:bg-neutral-900/50">
        <div className="max-w-full mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-12">
            {/* Breadcrumb Navigation */}
            <div className="flex-1 min-w-0">
              <Breadcrumb
                items={getBreadcrumbItems()}
                onNavigate={(tabId) => onTabChange && onTabChange(tabId)}
                className="min-w-0"
              />
            </div>

            {/* Contextual Help */}
            <div className="flex items-center space-x-3">
              <ContextualHelp pageId={activeTab} />
              
              {/* Quick Action Buttons */}
              <div className="hidden md:flex items-center space-x-2">
                <button
                  onClick={handleExport}
                  className="flex items-center space-x-1 px-3 py-1.5 text-sm rounded-md bg-neutral-100 dark:bg-neutral-800 text-neutral-700 dark:text-neutral-300 hover:bg-neutral-200 dark:hover:bg-neutral-700 transition-colors"
                >
                  <Download className="w-3 h-3" />
                  <span>Export</span>
                </button>
                
                {onShowTour && (
                  <button
                    onClick={onShowTour}
                    className="flex items-center space-x-1 px-3 py-1.5 text-sm rounded-md bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300 hover:bg-primary-200 dark:hover:bg-primary-900/50 transition-colors"
                  >
                    <Zap className="w-3 h-3" />
                    <span>Tour</span>
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Smart Search Modal */}
      {showSmartSearch && (
        <div className="fixed inset-0 bg-black/20 backdrop-blur-sm z-50 flex items-start justify-center pt-20">
          <motion.div
            initial={{ opacity: 0, y: -20, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -20, scale: 0.95 }}
            className="w-full max-w-2xl mx-4"
          >
            <SmartSearchBar
              placeholder="What would you like to know about your AI models?"
              onSearch={(query) => {
                console.log('Global search:', query);
                setShowSmartSearch(false);
              }}
              onSuggestionClick={(suggestion) => {
                console.log('Global suggestion:', suggestion);
                setShowSmartSearch(false);
              }}
              showVoiceInput={true}
              showRecentSearches={true}
              className="shadow-2xl"
            />
          </motion.div>
          {/* Click outside to close */}
          <div 
            className="absolute inset-0 -z-10" 
            onClick={() => setShowSmartSearch(false)}
          />
        </div>
      )}

      {/* Keyboard Shortcuts Modal */}
      <KeyboardShortcutsModal
        open={showShortcutsModal}
        onClose={() => setShowShortcutsModal(false)}
      />
    </header>
  );
};

export default Header;