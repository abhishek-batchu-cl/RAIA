import React, { useState } from 'react';
import { motion } from 'framer-motion';
import Header from './Header';
import Sidebar from './Sidebar';
import SidebarUX from './Sidebar-UX';
import SidebarRAIA from './SidebarRAIA';
import HelpGuide from '@/components/common/HelpGuide';
import DataConnectivityFAB from '@/components/common/DataConnectivityFAB';
import EnterpriseBreadcrumbs from '@/components/common/EnterpriseBreadcrumbs';
import EnterpriseNotificationCenter from '@/components/common/EnterpriseNotificationCenter';
import { cn } from '@/utils';

interface LayoutProps {
  children: React.ReactNode;
  activeTab: string;
  onTabChange: (tabId: string) => void;
  modelType: 'classification' | 'regression';
  onModelTypeChange: (type: 'classification' | 'regression') => void;
  onShowTour?: () => void;
}

const Layout: React.FC<LayoutProps> = ({
  children,
  activeTab,
  onTabChange,
  modelType,
  onModelTypeChange,
  onShowTour,
}) => {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isHelpOpen, setIsHelpOpen] = useState(false);

  const toggleSidebar = () => {
    setIsSidebarOpen(!isSidebarOpen);
  };

  const closeSidebar = () => {
    setIsSidebarOpen(false);
  };

  const pageVariants = {
    initial: {
      opacity: 0,
      x: -20,
    },
    animate: {
      opacity: 1,
      x: 0,
      transition: {
        duration: 0.3,
        ease: "easeOut",
      },
    },
    exit: {
      opacity: 0,
      x: 20,
      transition: {
        duration: 0.2,
        ease: "easeIn",
      },
    },
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-neutral-50 via-blue-50/30 to-indigo-50/30 dark:from-neutral-900 dark:via-blue-900/10 dark:to-indigo-900/10">
      {/* Header */}
      <Header
        onToggleSidebar={toggleSidebar}
        isSidebarOpen={isSidebarOpen}
        modelType={modelType}
        onModelTypeChange={onModelTypeChange}
        onShowHelp={() => setIsHelpOpen(true)}
        onShowTour={onShowTour}
        onTabChange={onTabChange}
        activeTab={activeTab}
      />
      
      {/* Enterprise Notification Center */}
      <div className="fixed top-4 right-4 z-50">
        <EnterpriseNotificationCenter />
      </div>

      <div className="flex h-[calc(100vh-6.5rem)]">
        {/* RAIA Enterprise Sidebar */}
        <SidebarRAIA
          isOpen={isSidebarOpen}
          onClose={closeSidebar}
          activeTab={activeTab}
          onTabChange={onTabChange}
          modelType={modelType}
        />

        {/* Main Content */}
        <main 
          className={cn(
            "flex-1 overflow-hidden",
            "lg:ml-0" // Sidebar is handled by its own positioning
          )}
        >
          <div className="h-full overflow-y-auto">
            <motion.div
              key={activeTab}
              variants={pageVariants}
              initial="initial"
              animate="animate"
              exit="exit"
              className="p-6 max-w-full"
            >
              {/* Enterprise Breadcrumb */}
              <div className="mb-6">
                <EnterpriseBreadcrumbs 
                  items={[
                    { label: 'Dashboard', path: '/dashboard', icon: 'dashboard' },
                    { label: 'RAIA Platform', path: '/models', icon: 'brain' },
                    { label: getTabLabel(activeTab), active: true }
                  ]}
                />
              </div>

              {/* Page Content */}
              <div className="space-y-6 enterprise-card">
                <div className="glass rounded-xl border border-white/20 p-1">
                  {children}
                </div>
              </div>
            </motion.div>
          </div>
        </main>
      </div>

      {/* Help Guide */}
      <HelpGuide
        isOpen={isHelpOpen}
        onClose={() => setIsHelpOpen(false)}
        activeTab={activeTab}
      />

      {/* Data Connectivity FAB */}
      <DataConnectivityFAB
        onNavigateToDataConnectivity={() => onTabChange('data-connectivity')}
        isDataConnectivityActive={activeTab === 'data-connectivity'}
      />
    </div>
  );
};

// Helper function to get tab label
function getTabLabel(tabId: string): string {
  const tabLabels: Record<string, string> = {
    'overview': 'Model Overview',
    'feature-importance': 'Feature Importance',
    'classification-stats': 'Classification Statistics',
    'regression-stats': 'Regression Statistics',
    'predictions': 'Individual Predictions',
    'what-if': 'What-If Analysis',
    'feature-dependence': 'Feature Dependence',
    'feature-interactions': 'Feature Interactions',
    'decision-trees': 'Decision Trees',
  };
  
  return tabLabels[tabId] || 'Dashboard';
}

export default Layout;