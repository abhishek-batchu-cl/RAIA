// Type definitions for the frontend application

export interface Configuration {
  id: string;
  name: string;
  model_name: string;
  temperature: number;
  max_tokens: number;
  system_prompt: string;
  created_at: string;
  updated_at: string;
}

export interface Document {
  id: string;
  filename: string;
  content: string;
  metadata: Record<string, any>;
  uploaded_at: string;
}

export interface EvaluationResult {
  id: string;
  configuration_id: string;
  dataset_name: string;
  results: {
    accuracy: number;
    relevance: number;
    coherence: number;
    groundedness: number;
    fluency: number;
    overall_score: number;
  };
  detailed_results: Array<{
    question: string;
    expected_answer: string;
    actual_answer: string;
    scores: Record<string, number>;
  }>;
  status: 'running' | 'completed' | 'failed';
  created_at: string;
  completed_at?: string;
}

export interface ComparisonResult {
  configurations: Configuration[];
  results: EvaluationResult[];
  comparison_metrics: {
    winner: string;
    performance_diff: Record<string, number>;
    statistical_significance: Record<string, boolean>;
  };
}

export interface MonitoringData {
  total_evaluations: number;
  avg_response_time: number;
  success_rate: number;
  recent_activity: Array<{
    timestamp: string;
    action: string;
    details: string;
  }>;
  performance_trends: {
    labels: string[];
    accuracy_trend: number[];
    response_time_trend: number[];
  };
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
}

export interface ChatResponse {
  response: string;
  sources: Array<{
    document: string;
    chunk: string;
    score: number;
  }>;
}

export interface Dataset {
  name: string;
  questions: Array<{
    question: string;
    expected_answer: string;
    metadata?: Record<string, any>;
  }>;
}

export interface ApiError {
  detail: string;
  status_code?: number;
}
