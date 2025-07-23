/**
 * Main API Service for RAIA Platform
 * Centralized API service that exports all API functionality
 */

import axios from 'axios';

// Export types for convenience
export type { ModelMetadata } from '../types';

// API base URL - defaults to localhost for development
const API_BASE_URL = (import.meta as any).env?.VITE_API_URL || 'http://localhost:8000/api';

/**
 * Create axios instance with default configuration
 */
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * Request interceptor for authentication
 */
apiClient.interceptors.request.use(
  (config) => {
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
 */
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error);
    return Promise.reject(error);
  }
);

/**
 * Main API service object containing all endpoint methods
 */
// We'll add the enhanced client after the api object is defined
let enhancedApiClient: any;

export const api = {
  // Health & System
  async getHealth() {
    const response = await apiClient.get('/health');
    return response.data;
  },

  // Dashboard & Metrics
  async getDashboardMetrics() {
    const response = await apiClient.get('/dashboard/metrics');
    return response.data;
  },

  async getMetrics() {
    const response = await apiClient.get('/metrics');
    return response.data;
  },

  // Models
  async getModels() {
    const response = await apiClient.get('/models');
    return response.data;
  },

  async getModel(id: string) {
    const response = await apiClient.get(`/models/${id}`);
    return response.data;
  },

  // Feature Importance
  async getFeatureImportance(modelId?: string) {
    const url = modelId ? `/feature-importance?model_id=${modelId}` : '/feature-importance';
    const response = await apiClient.get(url);
    return response.data;
  },

  // Data Quality & Connectivity
  async getDataQuality() {
    const response = await apiClient.get('/data-quality');
    return response.data;
  },

  async getDataConnections() {
    const response = await apiClient.get('/data-connections');
    return response.data;
  },

  async testDataConnection(connectionData: any) {
    const response = await apiClient.post('/data-connections/test', connectionData);
    return response.data;
  },

  // Bias & Fairness
  async getBiasMetrics(modelId?: string) {
    const url = modelId ? `/bias-metrics?model_id=${modelId}` : '/bias-metrics';
    const response = await apiClient.get(url);
    return response.data;
  },

  // Models API methods
  async listModels() {
    // Return mock data for demo - replace with real API call when backend is available
    return {
      success: true,
      data: [
        {
          id: "model_1",
          name: "Credit Risk Classifier",
          type: "classification", 
          model_type: "classification",
          status: "deployed",
          accuracy: 0.94,
          version: "1.2.0",
          created_at: "2024-01-15T10:30:00Z",
          updated_at: "2024-02-20T15:45:00Z",
          metrics: {
            precision: 0.92,
            recall: 0.95,
            f1_score: 0.94,
            auc_roc: 0.97
          }
        },
        {
          id: "model_2", 
          name: "Customer Churn Predictor",
          type: "classification",
          model_type: "classification",
          status: "training",
          accuracy: 0.89,
          version: "2.1.0",
          created_at: "2024-02-10T08:15:00Z",
          updated_at: "2024-02-22T12:30:00Z",
          metrics: {
            precision: 0.87,
            recall: 0.91,
            f1_score: 0.89,
            auc_roc: 0.93
          }
        },
        {
          id: "model_3",
          name: "Price Prediction Model",
          type: "regression", 
          model_type: "regression",
          status: "deployed",
          accuracy: 0.91,
          version: "1.5.2",
          created_at: "2024-01-20T14:20:00Z",
          updated_at: "2024-02-18T09:10:00Z",
          metrics: {
            mae: 0.12,
            mse: 0.05,
            rmse: 0.22,
            r2_score: 0.91
          }
        }
      ]
    };
  },

  // General purpose API calls for pages that need basic data
  async get(endpoint: string) {
    const response = await apiClient.get(endpoint);
    return response.data;
  },

  async post(endpoint: string, data: any) {
    const response = await apiClient.post(endpoint, data);
    return response.data;
  },

  async put(endpoint: string, data: any) {
    const response = await apiClient.put(endpoint, data);
    return response.data;
  },

  async delete(endpoint: string) {
    const response = await apiClient.delete(endpoint);
    return response.data;
  },
};

// Create enhanced API client with additional methods after api is defined
enhancedApiClient = {
  ...apiClient,
  listModels: api.listModels,
  getModels: api.getModels,
  getModel: api.getModel,
  getMetrics: api.getMetrics,
  getFeatureImportance: api.getFeatureImportance,
  getBiasMetrics: api.getBiasMetrics,
  getDataQuality: api.getDataQuality,
  getDataConnections: api.getDataConnections,
};

// Export the enhanced client for direct use
export { enhancedApiClient as apiClient };

export default api;