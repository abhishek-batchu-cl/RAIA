/**
 * Custom React Hooks for API Data Fetching
 * 
 * This module provides custom React hooks that encapsulate API calls using React Query.
 * These hooks offer powerful data fetching capabilities with built-in state management:
 * 
 * Core Features:
 * - Automatic data fetching with intelligent caching
 * - Background refetching for real-time data updates
 * - Optimistic updates for better user experience
 * - Built-in error handling with toast notifications
 * - Loading states and error boundaries
 * - Type-safe API responses with TypeScript
 * - Automatic retries for failed requests
 * - Stale-while-revalidate caching strategy
 * 
 * Hook Categories:
 * - Dashboard Hooks: Real-time metrics and analytics
 * - Configuration Hooks: Agent configuration management
 * - Evaluation Hooks: Evaluation results and management
 * - Document Hooks: Document upload and management
 * - Chat Hooks: Real-time chat interactions
 * - Monitoring Hooks: System performance metrics
 * 
 * Performance Optimizations:
 * - Smart refetch intervals based on data criticality
 * - Query invalidation for data consistency
 * - Background updates to maintain fresh data
 * - Efficient caching to reduce API calls
 * 
 * @author LLM Agent Evaluation Team
 * @version 2.0.0
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '../services/api';
import { toast } from 'react-hot-toast';

// ========================================
// DASHBOARD HOOKS
// ========================================

/**
 * Hook to fetch dashboard overview metrics
 * Automatically refetches every 30 seconds for real-time updates
 * 
 * @returns useQuery result with dashboard metrics data, loading state, and error handling
 */
export const useDashboardMetrics = () => {
  return useQuery({
    queryKey: ['dashboard', 'metrics'],
    queryFn: api.getDashboardMetrics,
    refetchInterval: 30000, // Refetch every 30 seconds for real-time updates
    onError: (error: any) => {
      console.error('Failed to fetch dashboard metrics:', error);
      toast.error('Failed to load dashboard metrics');
    },
  });
};

/**
 * Hook to fetch recent dashboard activity
 * 
 * @param limit Number of recent activities to fetch (default: 10)
 * @returns useQuery result with activity data
 */
export const useDashboardActivity = (limit = 10) => {
  return useQuery({
    queryKey: ['dashboard', 'activity', limit],
    queryFn: () => api.getDashboardActivity(limit),
    refetchInterval: 60000, // Refetch every minute
    onError: (error: any) => {
      console.error('Failed to fetch dashboard activity:', error);
      toast.error('Failed to load recent activity');
    },
  });
};

/**
 * Hook to fetch evaluation metrics over time for trend analysis
 * Used for line charts showing metric trends
 * 
 * @returns useQuery result with time-series evaluation data
 */
export const useDashboardEvaluationMetrics = () => {
  return useQuery({
    queryKey: ['dashboard', 'evaluation-metrics'],
    queryFn: api.getDashboardEvaluationMetrics,
    refetchInterval: 30000,
    onError: (error: any) => {
      console.error('Failed to fetch evaluation metrics:', error);
      toast.error('Failed to load evaluation metrics');
    },
  });
};

/**
 * Hook to fetch token consumption data by model
 * Used for cost analysis and resource monitoring
 * 
 * @returns useQuery result with token consumption data
 */
export const useDashboardTokenConsumption = () => {
  return useQuery({
    queryKey: ['dashboard', 'token-consumption'],
    queryFn: api.getDashboardTokenConsumption,
    refetchInterval: 30000,
    onError: (error: any) => {
      console.error('Failed to fetch token consumption:', error);
      toast.error('Failed to load token consumption data');
    },
  });
};

/**
 * Hook to fetch performance metrics by configuration
 * Used for configuration comparison and optimization analysis
 * 
 * @returns useQuery result with performance data grouped by configuration
 */
export const useDashboardPerformanceByConfig = () => {
  return useQuery({
    queryKey: ['dashboard', 'performance-by-config'],
    queryFn: api.getDashboardPerformanceByConfig,
    refetchInterval: 60000, // Less frequent updates for historical data
    onError: (error: any) => {
      console.error('Failed to fetch performance by config:', error);
      toast.error('Failed to load configuration performance');
    },
  });
};

export const useDashboardModelComparison = () => {
  return useQuery({
    queryKey: ['dashboard', 'model-comparison'],
    queryFn: api.getDashboardModelComparison,
    refetchInterval: 60000,
    onError: (error: any) => {
      console.error('Failed to fetch model comparison:', error);
      toast.error('Failed to load model comparison data');
    },
  });
};

export const useDashboardAggregatedMetrics = () => {
  return useQuery({
    queryKey: ['dashboard', 'aggregated-metrics'],
    queryFn: api.getDashboardAggregatedMetrics,
    refetchInterval: 30000,
    onError: (error: any) => {
      console.error('Failed to fetch aggregated metrics:', error);
      toast.error('Failed to load aggregated metrics');
    },
  });
};

export const useDashboardEvaluationDetails = () => {
  return useQuery({
    queryKey: ['dashboard', 'evaluation-details'],
    queryFn: api.getDashboardEvaluationDetails,
    refetchInterval: 60000,
    onError: (error: any) => {
      console.error('Failed to fetch evaluation details:', error);
      toast.error('Failed to load evaluation details');
    },
  });
};

// Configuration hooks
export const useConfigurations = () => {
  return useQuery({
    queryKey: ['configurations'],
    queryFn: api.getConfigurations,
    onError: (error: any) => {
      console.error('Failed to fetch configurations:', error);
      toast.error('Failed to load configurations');
    },
  });
};

export const useConfiguration = (id: string) => {
  return useQuery({
    queryKey: ['configurations', id],
    queryFn: () => api.getConfiguration(id),
    enabled: !!id,
    onError: (error: any) => {
      console.error('Failed to fetch configuration:', error);
      toast.error('Failed to load configuration');
    },
  });
};

export const useCreateConfiguration = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: api.createConfiguration,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['configurations'] });
      toast.success('Configuration created successfully');
    },
    onError: (error: any) => {
      console.error('Failed to create configuration:', error);
      toast.error('Failed to create configuration');
    },
  });
};

export const useUpdateConfiguration = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: ({ id, data }: { id: string; data: any }) => 
      api.updateConfiguration(id, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['configurations'] });
      toast.success('Configuration updated successfully');
    },
    onError: (error: any) => {
      console.error('Failed to update configuration:', error);
      toast.error('Failed to update configuration');
    },
  });
};

export const useDeleteConfiguration = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: api.deleteConfiguration,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['configurations'] });
      toast.success('Configuration deleted successfully');
    },
    onError: (error: any) => {
      console.error('Failed to delete configuration:', error);
      toast.error('Failed to delete configuration');
    },
  });
};

// Document hooks
export const useDocuments = () => {
  return useQuery({
    queryKey: ['documents'],
    queryFn: api.getDocuments,
    onError: (error: any) => {
      console.error('Failed to fetch documents:', error);
      toast.error('Failed to load documents');
    },
  });
};

export const useUploadDocument = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: api.uploadDocument,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['documents'] });
      toast.success('Document uploaded successfully');
    },
    onError: (error: any) => {
      console.error('Failed to upload document:', error);
      toast.error('Failed to upload document');
    },
  });
};

export const useDeleteDocument = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: api.deleteDocument,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['documents'] });
      toast.success('Document deleted successfully');
    },
    onError: (error: any) => {
      console.error('Failed to delete document:', error);
      toast.error('Failed to delete document');
    },
  });
};

// Evaluation hooks
export const useEvaluations = () => {
  return useQuery({
    queryKey: ['evaluations'],
    queryFn: api.getEvaluations,
    refetchInterval: 10000, // Refetch every 10 seconds for active evaluations
    onError: (error: any) => {
      console.error('Failed to fetch evaluations:', error);
      toast.error('Failed to load evaluations');
    },
  });
};

export const useEvaluation = (id: string) => {
  return useQuery({
    queryKey: ['evaluations', id],
    queryFn: () => api.getEvaluation(id),
    enabled: !!id,
    refetchInterval: 5000, // Refetch every 5 seconds for active evaluation
    onError: (error: any) => {
      console.error('Failed to fetch evaluation:', error);
      toast.error('Failed to load evaluation');
    },
  });
};

export const useStartEvaluation = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: ({ configId, dataset }: { configId: string; dataset: string }) =>
      api.startEvaluation(configId, dataset),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['evaluations'] });
      toast.success('Evaluation started successfully');
    },
    onError: (error: any) => {
      console.error('Failed to start evaluation:', error);
      toast.error('Failed to start evaluation');
    },
  });
};

export const useStopEvaluation = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: api.stopEvaluation,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['evaluations'] });
      toast.success('Evaluation stopped successfully');
    },
    onError: (error: any) => {
      console.error('Failed to stop evaluation:', error);
      toast.error('Failed to stop evaluation');
    },
  });
};

// Monitoring hooks
export const useMonitoring = () => {
  return useQuery({
    queryKey: ['monitoring'],
    queryFn: api.getMonitoring,
    refetchInterval: 30000, // Refetch every 30 seconds
    onError: (error: any) => {
      console.error('Failed to fetch monitoring data:', error);
      toast.error('Failed to load monitoring data');
    },
  });
};

// Chat hooks
export const useChat = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: ({ message, configId }: { message: string; configId?: string }) =>
      api.chat(message, configId),
    onSuccess: () => {
      // Optionally invalidate chat history or other related queries
      toast.success('Message sent successfully');
    },
    onError: (error: any) => {
      console.error('Failed to send message:', error);
      toast.error('Failed to send message');
    },
  });
};

// Dataset hooks
export const useDatasets = () => {
  return useQuery({
    queryKey: ['datasets'],
    queryFn: api.getDatasets,
    onError: (error: any) => {
      console.error('Failed to fetch datasets:', error);
      toast.error('Failed to load datasets');
    },
  });
};

// Comparison hooks
export const useCompareConfigurations = () => {
  return useMutation({
    mutationFn: ({ configIds, dataset }: { configIds: string[]; dataset: string }) =>
      api.compareConfigurations(configIds, dataset),
    onError: (error: any) => {
      console.error('Failed to compare configurations:', error);
      toast.error('Failed to compare configurations');
    },
  });
};

// Export hooks
export const useExportResults = () => {
  return useMutation({
    mutationFn: api.exportResults,
    onSuccess: () => {
      toast.success('Results exported successfully');
    },
    onError: (error: any) => {
      console.error('Failed to export results:', error);
      toast.error('Failed to export results');
    },
  });
};
