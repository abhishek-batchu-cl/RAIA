/**
 * API Service Module for LLM Agent Evaluation Framework
 * 
 * This module provides a centralized interface for all API communications
 * between the React frontend and the FastAPI backend. It includes:
 * 
 * Core Features:
 * - Axios HTTP client with interceptors for authentication and error handling
 * - Type-safe API methods with TypeScript interfaces
 * - Centralized error handling and request/response logging
 * - Environment-based API URL configuration
 * - Organized endpoint groups for better maintainability
 * 
 * Endpoint Categories:
 * - Health & System: Service health checks and status
 * - Dashboard: Metrics, charts, and analytics data
 * - Configurations: Agent configuration CRUD operations
 * - Evaluations: Evaluation management and results
 * - Documents: Document upload, processing, and management
 * - Chat: Real-time chat interface with agents
 * - Monitoring: System performance and usage metrics
 * 
 * Authentication:
 * - Ready for Bearer token authentication (future feature)
 * - Automatic token attachment via request interceptors
 * - Token refresh handling (when implemented)
 * 
 * Error Handling:
 * - Response interceptor for centralized error logging
 * - Error propagation to calling components
 * - Network error handling and retry logic
 * 
 * @author LLM Agent Evaluation Team
 * @version 2.0.0
 */

import axios from 'axios';
import type { 
  Configuration, 
  Document, 
  EvaluationResult, 
  ComparisonResult, 
  MonitoringData,
  ChatResponse,
  Dataset 
} from '../../types/evaluation';

// API base URL - defaults to localhost for development
const API_BASE_URL = (import.meta as any).env?.VITE_API_URL || 'http://localhost:8000/api';

/**
 * Create axios instance with default configuration
 * Includes base URL and default headers for JSON communication
 */
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * Request interceptor for authentication
 * Automatically adds Bearer token to requests if available in localStorage
 */
apiClient.interceptors.request.use(
  (config) => {
    // Add auth token if available (for future authentication implementation)
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

/**
 * Response interceptor for centralized error handling
 * Logs errors and passes them to the calling component for handling
 */
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error);
    // Additional error handling can be added here (e.g., token refresh)
    return Promise.reject(error);
  }
);

/**
 * Main API service object containing all endpoint methods
 * Organized by feature area for better maintainability
 */
export const api = {
  // ========================================
  // HEALTH & SYSTEM ENDPOINTS
  // ========================================
  
  /**
   * Check the health status of the backend API
   * @returns Promise<any> Health status and timestamp
   */
  async getHealth() {
    const response = await apiClient.get('/health');
    return response.data;
  },
  
  // ========================================
  // DASHBOARD ENDPOINTS
  // ========================================
  
  /**
   * Get comprehensive dashboard metrics including evaluations, documents, and configurations
   * @returns Promise<any> Dashboard overview metrics
   */
  async getDashboardMetrics() {
    const response = await apiClient.get('/dashboard/metrics');
    return response.data;
  },
  
  /**
   * Get recent dashboard activity with optional limit
   * @param limit Number of recent activities to retrieve (default: 10)
   * @returns Promise<any> Recent activity data
   */
  async getDashboardActivity(limit = 10) {
    const response = await apiClient.get(`/dashboard/activity?limit=${limit}`);
    return response.data;
  },

  /**
   * Get evaluation metrics over time for trend analysis
   * @returns Promise<any> Time-series evaluation metrics data
   */
  async getDashboardEvaluationMetrics() {
    const response = await apiClient.get('/dashboard/evaluation-metrics');
    return response.data;
  },

  /**
   * Get token consumption data by model for cost analysis
   * @returns Promise<any> Token consumption breakdown by model
   */
  async getDashboardTokenConsumption() {
    const response = await apiClient.get('/dashboard/token-consumption');
    return response.data;
  },

  /**
   * Get performance metrics grouped by configuration
   * @returns Promise<any> Performance data by configuration
   */
  async getDashboardPerformanceByConfig() {
    const response = await apiClient.get('/dashboard/performance-by-config');
    return response.data;
  },

  /**
   * Get comparative performance data across different models
   * @returns Promise<any> Model comparison metrics
   */
  async getDashboardModelComparison() {
    const response = await apiClient.get('/dashboard/model-comparison');
    return response.data;
  },

  /**
   * Get aggregated metrics for comprehensive analysis
   * @returns Promise<any> Aggregated evaluation metrics
   */
  async getDashboardAggregatedMetrics() {
    const response = await apiClient.get('/dashboard/aggregated-metrics');
    return response.data;
  },

  async getDashboardEvaluationDetails() {
    const response = await apiClient.get('/dashboard/evaluation-details');
    return response.data;
  },
  
  // Configurations
  async getConfigurations(): Promise<Configuration[]> {
    const response = await apiClient.get('/configurations');
    return response.data;
  },
  
  async createConfiguration(data: Omit<Configuration, 'id' | 'created_at' | 'updated_at'>): Promise<Configuration> {
    const response = await apiClient.post('/configurations', data);
    return response.data;
  },
  
  async updateConfiguration(id: string, data: Partial<Configuration>): Promise<Configuration> {
    const response = await apiClient.put(`/configurations/${id}`, data);
    return response.data;
  },
  
  async deleteConfiguration(id: string): Promise<void> {
    await apiClient.delete(`/configurations/${id}`);
  },
  
  async getConfiguration(id: string): Promise<Configuration> {
    const response = await apiClient.get(`/configurations/${id}`);
    return response.data;
  },
  
  // Documents
  async getDocuments(): Promise<Document[]> {
    const response = await apiClient.get('/documents');
    return response.data;
  },
  
  async uploadDocument(file: File): Promise<Document> {
    const formData = new FormData();
    formData.append('file', file);
    const response = await apiClient.post('/documents/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },
  
  async deleteDocument(id: string): Promise<void> {
    await apiClient.delete(`/documents/${id}`);
  },
  
  // Chat
  async chat(message: string, configId?: string): Promise<ChatResponse> {
    const response = await apiClient.post('/chat', {
      message,
      config_id: configId,
    });
    return response.data;
  },
  
  // Evaluations
  async getEvaluations(): Promise<EvaluationResult[]> {
    const response = await apiClient.get('/evaluations');
    return response.data;
  },
  
  async startEvaluation(configId: string, dataset: string): Promise<EvaluationResult> {
    const response = await apiClient.post('/evaluations/start', {
      config_id: configId,
      dataset_name: dataset,
    });
    return response.data;
  },
  
  async stopEvaluation(id: string): Promise<void> {
    await apiClient.post(`/evaluations/${id}/stop`);
  },
  
  async getEvaluation(id: string): Promise<EvaluationResult> {
    const response = await apiClient.get(`/evaluations/${id}`);
    return response.data;
  },
  
  // Datasets
  async getDatasets(): Promise<string[]> {
    const response = await apiClient.get('/datasets');
    return response.data;
  },
  
  // Monitoring
  async getMonitoring(): Promise<MonitoringData> {
    const response = await apiClient.get('/monitoring');
    return response.data;
  },
  
  // Comparison
  async compareConfigurations(configIds: string[], dataset: string): Promise<ComparisonResult> {
    const response = await apiClient.post('/comparison', {
      config_ids: configIds,
      dataset_name: dataset,
    });
    return response.data;
  },
  
  // Export
  async exportResults(evaluationIds: string[]): Promise<Blob> {
    const response = await apiClient.post('/export', {
      evaluation_ids: evaluationIds,
    }, {
      responseType: 'blob',
    });
    return response.data;
  },
};

export default api;
