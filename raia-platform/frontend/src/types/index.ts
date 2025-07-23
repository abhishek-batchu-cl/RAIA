// Core Types
export interface User {
  id: string;
  email: string;
  name: string;
  role: string;
  avatar?: string;
  createdAt: string;
  lastLogin?: string;
}

export interface Model {
  id: string;
  name: string;
  type: 'classification' | 'regression';
  status: 'training' | 'deployed' | 'archived';
  accuracy?: number;
  createdAt: string;
  updatedAt: string;
}

export interface ModelMetadata {
  id: string;
  name: string;
  type: 'classification' | 'regression';
  version: string;
  description?: string;
  status: 'training' | 'deployed' | 'archived';
  accuracy?: number;
  createdAt: string;
  updatedAt: string;
  metrics?: ClassificationMetrics | RegressionMetrics;
}

// Theme Types
export interface ThemeSettings {
  mode: 'light' | 'dark' | 'system';
  primaryColor: string;
  secondaryColor: string;
  accentColor: string;
  animations: boolean;
  reducedMotion: boolean;
}

// API Response Types
export interface ApiResponse<T = any> {
  data: T;
  message: string;
  success: boolean;
  error?: string;
}

export interface PaginatedResponse<T = any> extends ApiResponse<T[]> {
  pagination: {
    page: number;
    limit: number;
    total: number;
    totalPages: number;
  };
}

// Chart Types
export interface ChartDataPoint {
  x: number | string;
  y: number;
  label?: string;
}

export interface ChartSeries {
  name: string;
  data: ChartDataPoint[];
  color?: string;
}

// Feature Types
export interface Feature {
  name: string;
  type: 'numerical' | 'categorical' | 'boolean';
  importance?: number;
  values?: (string | number)[];
}

// Explanation Types
export interface ShapValue {
  feature: string;
  value: number;
  base_value: number;
  data_value: number | string;
}

export interface LimeExplanation {
  feature: string;
  weight: number;
  value: number | string;
}

// Performance Metrics Types
export interface ClassificationMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  auc_roc?: number;
  confusion_matrix?: number[][];
}

export interface RegressionMetrics {
  mae: number;
  mse: number;
  rmse: number;
  r2_score: number;
  mape?: number;
}

// Data Drift Types
export interface DriftDetection {
  feature: string;
  drift_score: number;
  p_value: number;
  is_drift: boolean;
  drift_type: 'numerical' | 'categorical';
}

// Fairness Types
export interface FairnessMetric {
  metric_name: string;
  value: number;
  threshold: number;
  is_fair: boolean;
  protected_attribute: string;
}

// Alert Types
export interface Alert {
  id: string;
  type: 'drift' | 'performance' | 'bias' | 'system';
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  timestamp: string;
  model_id?: string;
  is_read: boolean;
}

// Export all evaluation types
export * from './evaluation';