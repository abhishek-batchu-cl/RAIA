import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import { ThemeProvider } from './contexts/ThemeContext';
import { Toaster } from 'react-hot-toast';

// Layout Components
import Layout from './components/layout/Layout';
import AuthGuard from './components/common/AuthGuard';

// Page Components - Unified Dashboard
import ModelOverview from './pages/ModelOverview';
import FeatureImportance from './pages/FeatureImportance';
import ClassificationStats from './pages/ClassificationStats';
import RegressionStats from './pages/RegressionStats';
import IndividualPredictions from './pages/IndividualPredictions';
import WhatIfAnalysis from './pages/WhatIfAnalysis';
import FeatureDependence from './pages/FeatureDependence';
import FeatureInteractions from './pages/FeatureInteractions';
import DecisionTrees from './pages/DecisionTrees';
import DataConnectivity from './pages/DataConnectivity';
import DataManagement from './pages/DataManagement';
import DataQualityDashboard from './pages/DataQualityDashboard';
import DataDrift from './pages/DataDrift';
import EnterpriseFairness from './pages/EnterpriseFairness';
import BiasMetigationDashboard from './pages/BiasMetigationDashboard';
import ModelPerformance from './pages/ModelPerformance';
import RootCauseAnalysis from './pages/RootCauseAnalysis';
import EnterpriseAlerts from './pages/EnterpriseAlerts';
import ABTestingDashboard from './pages/ABTestingDashboard';
import ExecutiveDashboard from './pages/ExecutiveDashboard';
import CustomDashboardBuilder from './pages/CustomDashboardBuilder';
import UserManagement from './pages/UserManagement';
import ModelManagement from './pages/ModelManagement';
import SystemHealth from './pages/SystemHealth';
import Settings from './pages/Settings';

// Agent Evaluation Pages
import EvaluationChat from './pages/evaluation/Chat';
import EvaluationComparison from './pages/evaluation/Comparison';
import EvaluationConfiguration from './pages/evaluation/Configuration';
import EvaluationDashboard from './pages/evaluation/Dashboard';
import EvaluationDetails from './pages/evaluation/Details';
import EvaluationDocuments from './pages/evaluation/Documents';
import EvaluationMain from './pages/evaluation/Evaluation';
import EvaluationMonitoring from './pages/evaluation/Monitoring';

// Auth Pages
import { AuthProvider } from './contexts/AuthContext';

// Styles
import './styles/globals.css';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
      staleTime: 5 * 60 * 1000, // 5 minutes
    },
  },
});

function App() {
  const [activeTab, setActiveTab] = useState('overview');
  const [modelType, setModelType] = useState<'classification' | 'regression'>('classification');

  const handleTabChange = (tabId: string) => {
    setActiveTab(tabId);
  };

  const handleModelTypeChange = (type: 'classification' | 'regression') => {
    setModelType(type);
  };

  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <AuthProvider>
            <Router>
              <div className="min-h-screen bg-neutral-50 dark:bg-neutral-900">
                <Routes>
                  {/* Public Routes */}
                  <Route path="/login" element={<div>Login Page</div>} />
                  
                  {/* Protected Routes - Using Unified RAIA Layout */}
                  <Route path="/" element={
                    <AuthGuard requireAuth={false}>
                      <Layout
                        activeTab={activeTab}
                        onTabChange={handleTabChange}
                        modelType={modelType}
                        onModelTypeChange={handleModelTypeChange}
                      >
                        <Navigate to="/overview" replace />
                      </Layout>
                    </AuthGuard>
                  } />
                  
                  {/* Model Explainability & Analytics Routes */}
                  <Route path="/overview" element={
                    <AuthGuard requireAuth={false}>
                      <Layout
                        activeTab="overview"
                        onTabChange={handleTabChange}
                        modelType={modelType}
                        onModelTypeChange={handleModelTypeChange}
                      >
                        <ModelOverview />
                      </Layout>
                    </AuthGuard>
                  } />
                  
                  <Route path="/feature-importance" element={
                    <AuthGuard requireAuth={false}>
                      <Layout
                        activeTab="feature-importance"
                        onTabChange={handleTabChange}
                        modelType={modelType}
                        onModelTypeChange={handleModelTypeChange}
                      >
                        <FeatureImportance />
                      </Layout>
                    </AuthGuard>
                  } />
                  
                  <Route path="/classification-stats" element={
                    <AuthGuard requireAuth={false}>
                      <Layout
                        activeTab="classification-stats"
                        onTabChange={handleTabChange}
                        modelType={modelType}
                        onModelTypeChange={handleModelTypeChange}
                      >
                        <ClassificationStats />
                      </Layout>
                    </AuthGuard>
                  } />
                  
                  <Route path="/regression-stats" element={
                    <AuthGuard requireAuth={false}>
                      <Layout
                        activeTab="regression-stats"
                        onTabChange={handleTabChange}
                        modelType={modelType}
                        onModelTypeChange={handleModelTypeChange}
                      >
                        <RegressionStats />
                      </Layout>
                    </AuthGuard>
                  } />
                  
                  <Route path="/predictions" element={
                    <AuthGuard requireAuth={false}>
                      <Layout
                        activeTab="predictions"
                        onTabChange={handleTabChange}
                        modelType={modelType}
                        onModelTypeChange={handleModelTypeChange}
                      >
                        <IndividualPredictions />
                      </Layout>
                    </AuthGuard>
                  } />
                  
                  <Route path="/what-if" element={
                    <AuthGuard requireAuth={false}>
                      <Layout
                        activeTab="what-if"
                        onTabChange={handleTabChange}
                        modelType={modelType}
                        onModelTypeChange={handleModelTypeChange}
                      >
                        <WhatIfAnalysis />
                      </Layout>
                    </AuthGuard>
                  } />
                  
                  <Route path="/feature-dependence" element={
                    <AuthGuard requireAuth={false}>
                      <Layout
                        activeTab="feature-dependence"
                        onTabChange={handleTabChange}
                        modelType={modelType}
                        onModelTypeChange={handleModelTypeChange}
                      >
                        <FeatureDependence />
                      </Layout>
                    </AuthGuard>
                  } />
                  
                  <Route path="/feature-interactions" element={
                    <AuthGuard requireAuth={false}>
                      <Layout
                        activeTab="feature-interactions"
                        onTabChange={handleTabChange}
                        modelType={modelType}
                        onModelTypeChange={handleModelTypeChange}
                      >
                        <FeatureInteractions />
                      </Layout>
                    </AuthGuard>
                  } />
                  
                  <Route path="/decision-trees" element={
                    <AuthGuard requireAuth={false}>
                      <Layout
                        activeTab="decision-trees"
                        onTabChange={handleTabChange}
                        modelType={modelType}
                        onModelTypeChange={handleModelTypeChange}
                      >
                        <DecisionTrees />
                      </Layout>
                    </AuthGuard>
                  } />
                  
                  {/* Data Pipeline Routes */}
                  <Route path="/data-connectivity" element={
                    <AuthGuard requireAuth={false}>
                      <Layout
                        activeTab="data-connectivity"
                        onTabChange={handleTabChange}
                        modelType={modelType}
                        onModelTypeChange={handleModelTypeChange}
                      >
                        <DataConnectivity />
                      </Layout>
                    </AuthGuard>
                  } />
                  
                  <Route path="/data-quality" element={
                    <AuthGuard requireAuth={false}>
                      <Layout
                        activeTab="data-quality"
                        onTabChange={handleTabChange}
                        modelType={modelType}
                        onModelTypeChange={handleModelTypeChange}
                      >
                        <DataQualityDashboard />
                      </Layout>
                    </AuthGuard>
                  } />
                  
                  <Route path="/data-drift" element={
                    <AuthGuard requireAuth={false}>
                      <Layout
                        activeTab="data-drift"
                        onTabChange={handleTabChange}
                        modelType={modelType}
                        onModelTypeChange={handleModelTypeChange}
                      >
                        <DataDrift />
                      </Layout>
                    </AuthGuard>
                  } />
                  
                  {/* Responsible AI Routes */}
                  <Route path="/enterprise-fairness" element={
                    <AuthGuard requireAuth={false}>
                      <Layout
                        activeTab="enterprise-fairness"
                        onTabChange={handleTabChange}
                        modelType={modelType}
                        onModelTypeChange={handleModelTypeChange}
                      >
                        <EnterpriseFairness />
                      </Layout>
                    </AuthGuard>
                  } />
                  
                  <Route path="/bias-mitigation" element={
                    <AuthGuard requireAuth={false}>
                      <Layout
                        activeTab="bias-mitigation"
                        onTabChange={handleTabChange}
                        modelType={modelType}
                        onModelTypeChange={handleModelTypeChange}
                      >
                        <BiasMetigationDashboard />
                      </Layout>
                    </AuthGuard>
                  } />
                  
                  {/* Agent Evaluation Routes */}
                  <Route path="/agent-chat" element={
                    <AuthGuard requireAuth={false}>
                      <Layout
                        activeTab="agent-chat"
                        onTabChange={handleTabChange}
                        modelType={modelType}
                        onModelTypeChange={handleModelTypeChange}
                      >
                        <EvaluationChat />
                      </Layout>
                    </AuthGuard>
                  } />
                  
                  <Route path="/agent-comparison" element={
                    <AuthGuard requireAuth={false}>
                      <Layout
                        activeTab="agent-comparison"
                        onTabChange={handleTabChange}
                        modelType={modelType}
                        onModelTypeChange={handleModelTypeChange}
                      >
                        <EvaluationComparison />
                      </Layout>
                    </AuthGuard>
                  } />
                  
                  <Route path="/agent-evaluation" element={
                    <AuthGuard requireAuth={false}>
                      <Layout
                        activeTab="agent-evaluation"
                        onTabChange={handleTabChange}
                        modelType={modelType}
                        onModelTypeChange={handleModelTypeChange}
                      >
                        <EvaluationMain />
                      </Layout>
                    </AuthGuard>
                  } />
                  
                  {/* Executive & Business Routes */}
                  <Route path="/executive-dashboard" element={
                    <AuthGuard requireAuth={false}>
                      <Layout
                        activeTab="executive-dashboard"
                        onTabChange={handleTabChange}
                        modelType={modelType}
                        onModelTypeChange={handleModelTypeChange}
                      >
                        <ExecutiveDashboard />
                      </Layout>
                    </AuthGuard>
                  } />
                  
                  {/* System Management Routes */}
                  <Route path="/user-management" element={
                    <AuthGuard requireAuth={false}>
                      <Layout
                        activeTab="user-management"
                        onTabChange={handleTabChange}
                        modelType={modelType}
                        onModelTypeChange={handleModelTypeChange}
                      >
                        <UserManagement />
                      </Layout>
                    </AuthGuard>
                  } />
                  
                  <Route path="/settings" element={
                    <AuthGuard requireAuth={false}>
                      <Layout
                        activeTab="settings"
                        onTabChange={handleTabChange}
                        modelType={modelType}
                        onModelTypeChange={handleModelTypeChange}
                      >
                        <Settings />
                      </Layout>
                    </AuthGuard>
                  } />
                  
                  {/* Catch all route */}
                  <Route path="*" element={<Navigate to="/overview" replace />} />
                </Routes>
                
                <Toaster 
                  position="top-right"
                  toastOptions={{
                    duration: 4000,
                    className: 'bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100',
                  }}
                />
              </div>
            </Router>
        </AuthProvider>
      </ThemeProvider>
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  );
}

export default App;