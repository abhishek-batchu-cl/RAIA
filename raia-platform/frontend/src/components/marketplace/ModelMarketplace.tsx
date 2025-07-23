import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Search, Filter, Star, Download, Upload, Share2, User,
  Tag, Calendar, TrendingUp, Award, Heart, Eye,
  Package, Shield, CheckCircle, AlertTriangle, Info,
  MoreVertical, ExternalLink, GitBranch, Users, Zap,
  BookOpen, Code, Database, Brain, Target, Activity
} from 'lucide-react';

interface ModelListing {
  id: string;
  name: string;
  description: string;
  author: {
    username: string;
    avatar?: string;
    verified: boolean;
    reputation: number;
  };
  version: string;
  category: string;
  subcategory: string;
  tags: string[];
  
  // Performance metrics
  accuracy?: number;
  performance_metrics: Record<string, number>;
  dataset_info: {
    name: string;
    size: number;
    features: number;
  };
  
  // Usage stats
  downloads: number;
  stars: number;
  views: number;
  forks: number;
  
  // Model details
  model_type: 'classification' | 'regression' | 'clustering' | 'nlp' | 'computer_vision' | 'time_series';
  framework: string;
  framework_version: string;
  model_size_mb: number;
  inference_time_ms: number;
  
  // Licensing and security
  license: string;
  is_public: boolean;
  security_scan: {
    status: 'passed' | 'warning' | 'failed';
    last_scan: string;
    issues?: string[];
  };
  
  // Dates
  created_at: string;
  updated_at: string;
  published_at: string;
  
  // Additional metadata
  documentation_url?: string;
  demo_url?: string;
  github_url?: string;
  paper_url?: string;
  use_cases: string[];
  requirements: string[];
  pricing?: {
    type: 'free' | 'paid' | 'freemium';
    price?: number;
    currency?: string;
  };
}

interface MarketplaceStats {
  total_models: number;
  total_downloads: number;
  active_publishers: number;
  categories: Record<string, number>;
  trending_models: string[];
  featured_models: string[];
}

const ModelMarketplace: React.FC = () => {
  const [models, setModels] = useState<ModelListing[]>([]);
  const [filteredModels, setFilteredModels] = useState<ModelListing[]>([]);
  const [marketplaceStats, setMarketplaceStats] = useState<MarketplaceStats | null>(null);
  
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [selectedFramework, setSelectedFramework] = useState('all');
  const [sortBy, setSortBy] = useState<'popular' | 'recent' | 'stars' | 'downloads'>('popular');
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [showFilters, setShowFilters] = useState(false);
  
  const [selectedModel, setSelectedModel] = useState<ModelListing | null>(null);
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  // Mock data
  useEffect(() => {
    const mockStats: MarketplaceStats = {
      total_models: 1247,
      total_downloads: 89234,
      active_publishers: 523,
      categories: {
        'Computer Vision': 342,
        'Natural Language Processing': 289,
        'Time Series': 156,
        'Classification': 234,
        'Regression': 189,
        'Clustering': 87
      },
      trending_models: ['model_001', 'model_003', 'model_007'],
      featured_models: ['model_002', 'model_005', 'model_009']
    };

    const mockModels: ModelListing[] = [
      {
        id: 'model_001',
        name: 'Advanced Customer Churn Predictor',
        description: 'State-of-the-art ensemble model for predicting customer churn across various industries. Achieves 94.2% accuracy with interpretable features and built-in fairness constraints.',
        author: {
          username: 'ml_expert_sarah',
          verified: true,
          reputation: 4.8
        },
        version: '2.1.0',
        category: 'Classification',
        subcategory: 'Business Analytics',
        tags: ['churn', 'customer-analytics', 'ensemble', 'interpretable', 'production-ready'],
        accuracy: 0.942,
        performance_metrics: {
          'accuracy': 0.942,
          'precision': 0.915,
          'recall': 0.928,
          'f1_score': 0.921,
          'auc': 0.967
        },
        dataset_info: {
          name: 'Multi-Industry Customer Dataset',
          size: 125000,
          features: 47
        },
        downloads: 8934,
        stars: 456,
        views: 12847,
        forks: 89,
        model_type: 'classification',
        framework: 'scikit-learn',
        framework_version: '1.3.0',
        model_size_mb: 15.7,
        inference_time_ms: 2.3,
        license: 'MIT',
        is_public: true,
        security_scan: {
          status: 'passed',
          last_scan: '2024-01-15T10:30:00Z'
        },
        created_at: '2023-11-20T14:20:00Z',
        updated_at: '2024-01-15T09:15:00Z',
        published_at: '2023-12-01T16:45:00Z',
        documentation_url: 'https://github.com/ml_expert_sarah/churn-predictor',
        demo_url: 'https://churn-demo.example.com',
        github_url: 'https://github.com/ml_expert_sarah/churn-predictor',
        use_cases: ['E-commerce', 'SaaS', 'Telecom', 'Banking'],
        requirements: ['pandas>=1.5.0', 'scikit-learn>=1.3.0', 'numpy>=1.24.0'],
        pricing: {
          type: 'free'
        }
      },
      {
        id: 'model_002',
        name: 'NLP Sentiment Analysis Pro',
        description: 'Fine-tuned transformer model for multi-language sentiment analysis with support for 15 languages and domain adaptation capabilities.',
        author: {
          username: 'nlp_researcher_alex',
          verified: true,
          reputation: 4.9
        },
        version: '1.4.2',
        category: 'Natural Language Processing',
        subcategory: 'Sentiment Analysis',
        tags: ['sentiment', 'transformer', 'multilingual', 'fine-tuned', 'production'],
        accuracy: 0.891,
        performance_metrics: {
          'accuracy': 0.891,
          'macro_f1': 0.887,
          'weighted_f1': 0.893
        },
        dataset_info: {
          name: 'Multi-Language Sentiment Corpus',
          size: 500000,
          features: 768
        },
        downloads: 12456,
        stars: 678,
        views: 18923,
        forks: 134,
        model_type: 'nlp',
        framework: 'transformers',
        framework_version: '4.25.1',
        model_size_mb: 438.2,
        inference_time_ms: 15.7,
        license: 'Apache-2.0',
        is_public: true,
        security_scan: {
          status: 'passed',
          last_scan: '2024-01-16T08:20:00Z'
        },
        created_at: '2023-09-15T11:30:00Z',
        updated_at: '2024-01-10T14:22:00Z',
        published_at: '2023-10-01T09:15:00Z',
        documentation_url: 'https://huggingface.co/nlp_researcher_alex/sentiment-pro',
        github_url: 'https://github.com/nlp_researcher_alex/sentiment-analysis-pro',
        paper_url: 'https://arxiv.org/abs/2310.12345',
        use_cases: ['Social Media Monitoring', 'Product Reviews', 'Customer Feedback', 'Brand Analytics'],
        requirements: ['transformers>=4.25.0', 'torch>=1.13.0', 'tokenizers>=0.13.0'],
        pricing: {
          type: 'freemium'
        }
      },
      {
        id: 'model_003',
        name: 'Computer Vision Object Detector',
        description: 'Real-time object detection model optimized for edge deployment. Supports 80+ object classes with fast inference and low memory footprint.',
        author: {
          username: 'vision_ai_team',
          verified: true,
          reputation: 4.7
        },
        version: '3.0.1',
        category: 'Computer Vision',
        subcategory: 'Object Detection',
        tags: ['object-detection', 'real-time', 'edge', 'yolo', 'optimized'],
        accuracy: 0.847,
        performance_metrics: {
          'mAP@0.5': 0.847,
          'mAP@0.5:0.95': 0.623,
          'fps': 45.2
        },
        dataset_info: {
          name: 'COCO 2017',
          size: 118287,
          features: 80
        },
        downloads: 15632,
        stars: 892,
        views: 24567,
        forks: 203,
        model_type: 'computer_vision',
        framework: 'pytorch',
        framework_version: '1.13.1',
        model_size_mb: 47.3,
        inference_time_ms: 22.1,
        license: 'GPL-3.0',
        is_public: true,
        security_scan: {
          status: 'passed',
          last_scan: '2024-01-14T16:45:00Z'
        },
        created_at: '2023-08-10T09:20:00Z',
        updated_at: '2024-01-12T11:30:00Z',
        published_at: '2023-09-01T13:45:00Z',
        documentation_url: 'https://vision-ai-team.github.io/object-detector',
        demo_url: 'https://detector-demo.example.com',
        github_url: 'https://github.com/vision-ai-team/object-detector',
        use_cases: ['Security Systems', 'Autonomous Vehicles', 'Retail Analytics', 'Manufacturing QC'],
        requirements: ['torch>=1.13.0', 'torchvision>=0.14.0', 'opencv-python>=4.7.0'],
        pricing: {
          type: 'paid',
          price: 99,
          currency: 'USD'
        }
      },
      {
        id: 'model_004',
        name: 'Time Series Forecaster',
        description: 'Advanced neural network model for multi-variate time series forecasting with automatic seasonality detection and confidence intervals.',
        author: {
          username: 'time_series_guru',
          verified: false,
          reputation: 4.3
        },
        version: '1.2.5',
        category: 'Time Series',
        subcategory: 'Forecasting',
        tags: ['forecasting', 'neural-network', 'seasonality', 'confidence-intervals'],
        performance_metrics: {
          'mae': 0.087,
          'mse': 0.012,
          'mape': 4.2
        },
        dataset_info: {
          name: 'Energy Consumption Dataset',
          size: 87500,
          features: 12
        },
        downloads: 3421,
        stars: 187,
        views: 5432,
        forks: 34,
        model_type: 'time_series',
        framework: 'tensorflow',
        framework_version: '2.11.0',
        model_size_mb: 28.9,
        inference_time_ms: 8.4,
        license: 'MIT',
        is_public: true,
        security_scan: {
          status: 'warning',
          last_scan: '2024-01-13T14:20:00Z',
          issues: ['Potential dependency vulnerability in tensorflow<2.11.1']
        },
        created_at: '2023-12-05T16:30:00Z',
        updated_at: '2024-01-08T10:15:00Z',
        published_at: '2023-12-15T12:00:00Z',
        github_url: 'https://github.com/time_series_guru/ts-forecaster',
        use_cases: ['Energy Management', 'Demand Planning', 'Financial Forecasting', 'IoT Analytics'],
        requirements: ['tensorflow>=2.11.0', 'pandas>=1.5.0', 'numpy>=1.24.0'],
        pricing: {
          type: 'free'
        }
      }
    ];

    setMarketplaceStats(mockStats);
    setModels(mockModels);
    setFilteredModels(mockModels);
    setIsLoading(false);
  }, []);

  // Filter and search logic
  useEffect(() => {
    let filtered = models;

    // Search filter
    if (searchQuery) {
      filtered = filtered.filter(model =>
        model.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        model.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
        model.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase())) ||
        model.author.username.toLowerCase().includes(searchQuery.toLowerCase())
      );
    }

    // Category filter
    if (selectedCategory !== 'all') {
      filtered = filtered.filter(model => model.category === selectedCategory);
    }

    // Framework filter
    if (selectedFramework !== 'all') {
      filtered = filtered.filter(model => model.framework === selectedFramework);
    }

    // Sort
    filtered.sort((a, b) => {
      switch (sortBy) {
        case 'popular':
          return (b.downloads + b.stars) - (a.downloads + a.stars);
        case 'recent':
          return new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime();
        case 'stars':
          return b.stars - a.stars;
        case 'downloads':
          return b.downloads - a.downloads;
        default:
          return 0;
      }
    });

    setFilteredModels(filtered);
  }, [models, searchQuery, selectedCategory, selectedFramework, sortBy]);

  const handleDownloadModel = useCallback(async (modelId: string) => {
    // Implement model download logic
    console.log('Downloading model:', modelId);
  }, []);

  const handleStarModel = useCallback(async (modelId: string) => {
    setModels(prev => prev.map(model =>
      model.id === modelId
        ? { ...model, stars: model.stars + 1 }
        : model
    ));
  }, []);

  const getSecurityBadgeColor = (status: string) => {
    switch (status) {
      case 'passed': return 'bg-green-100 text-green-800';
      case 'warning': return 'bg-yellow-100 text-yellow-800';
      case 'failed': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getSecurityIcon = (status: string) => {
    switch (status) {
      case 'passed': return <CheckCircle className="w-3 h-3" />;
      case 'warning': return <AlertTriangle className="w-3 h-3" />;
      case 'failed': return <AlertTriangle className="w-3 h-3" />;
      default: return <Info className="w-3 h-3" />;
    }
  };

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'Computer Vision': return <Eye className="w-4 h-4" />;
      case 'Natural Language Processing': return <BookOpen className="w-4 h-4" />;
      case 'Time Series': return <TrendingUp className="w-4 h-4" />;
      case 'Classification': return <Target className="w-4 h-4" />;
      case 'Regression': return <Activity className="w-4 h-4" />;
      case 'Clustering': return <Database className="w-4 h-4" />;
      default: return <Brain className="w-4 h-4" />;
    }
  };

  const renderModelCard = (model: ModelListing) => (
    <motion.div
      key={model.id}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden hover:shadow-md transition-shadow"
    >
      <div className="p-6">
        {/* Header */}
        <div className="flex items-start justify-between mb-4">
          <div className="flex items-center gap-3">
            {getCategoryIcon(model.category)}
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-1">{model.name}</h3>
              <div className="flex items-center gap-2">
                <span className="text-sm text-gray-600">by {model.author.username}</span>
                {model.author.verified && <CheckCircle className="w-4 h-4 text-blue-500" />}
                <div className="flex items-center gap-1">
                  <Star className="w-3 h-3 text-yellow-500 fill-current" />
                  <span className="text-xs text-gray-600">{model.author.reputation}</span>
                </div>
              </div>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            <span className={`text-xs px-2 py-1 rounded-full font-medium ${getSecurityBadgeColor(model.security_scan.status)}`}>
              {getSecurityIcon(model.security_scan.status)}
              <span className="ml-1">{model.security_scan.status}</span>
            </span>
            <button className="text-gray-400 hover:text-gray-600">
              <MoreVertical className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Description */}
        <p className="text-gray-600 text-sm mb-4 line-clamp-3">{model.description}</p>

        {/* Tags */}
        <div className="flex flex-wrap gap-1 mb-4">
          {model.tags.slice(0, 4).map((tag, index) => (
            <span
              key={index}
              className="text-xs bg-blue-50 text-blue-700 px-2 py-1 rounded-full"
            >
              {tag}
            </span>
          ))}
          {model.tags.length > 4 && (
            <span className="text-xs text-gray-500">+{model.tags.length - 4} more</span>
          )}
        </div>

        {/* Performance metrics */}
        {model.accuracy && (
          <div className="mb-4">
            <div className="flex justify-between text-sm mb-1">
              <span className="text-gray-600">Accuracy</span>
              <span className="font-medium">{(model.accuracy * 100).toFixed(1)}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-green-500 h-2 rounded-full"
                style={{ width: `${model.accuracy * 100}%` }}
              />
            </div>
          </div>
        )}

        {/* Stats */}
        <div className="grid grid-cols-3 gap-4 mb-4 text-sm">
          <div className="text-center">
            <div className="flex items-center justify-center gap-1 text-gray-600">
              <Download className="w-3 h-3" />
              <span>{model.downloads.toLocaleString()}</span>
            </div>
            <span className="text-xs text-gray-500">Downloads</span>
          </div>
          <div className="text-center">
            <div className="flex items-center justify-center gap-1 text-gray-600">
              <Star className="w-3 h-3" />
              <span>{model.stars}</span>
            </div>
            <span className="text-xs text-gray-500">Stars</span>
          </div>
          <div className="text-center">
            <div className="flex items-center justify-center gap-1 text-gray-600">
              <GitBranch className="w-3 h-3" />
              <span>{model.forks}</span>
            </div>
            <span className="text-xs text-gray-500">Forks</span>
          </div>
        </div>

        {/* Technical details */}
        <div className="grid grid-cols-2 gap-4 mb-4 text-xs text-gray-600">
          <div>
            <span className="font-medium">Framework:</span> {model.framework}
          </div>
          <div>
            <span className="font-medium">Size:</span> {model.model_size_mb}MB
          </div>
          <div>
            <span className="font-medium">License:</span> {model.license}
          </div>
          <div>
            <span className="font-medium">Inference:</span> {model.inference_time_ms}ms
          </div>
        </div>

        {/* Actions */}
        <div className="flex gap-2">
          <button
            onClick={() => handleDownloadModel(model.id)}
            className="flex-1 bg-blue-600 text-white text-sm py-2 px-3 rounded-lg hover:bg-blue-700 transition-colors flex items-center justify-center gap-2"
          >
            <Download className="w-4 h-4" />
            Download
          </button>
          <button
            onClick={() => handleStarModel(model.id)}
            className="bg-gray-100 text-gray-700 text-sm py-2 px-3 rounded-lg hover:bg-gray-200 transition-colors flex items-center gap-1"
          >
            <Star className="w-4 h-4" />
            Star
          </button>
          <button
            onClick={() => setSelectedModel(model)}
            className="bg-gray-100 text-gray-700 text-sm py-2 px-3 rounded-lg hover:bg-gray-200 transition-colors"
          >
            <Eye className="w-4 h-4" />
          </button>
        </div>
      </div>
    </motion.div>
  );

  const renderModelDetail = (model: ModelListing) => (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        className="bg-white rounded-lg shadow-xl max-w-4xl max-h-[90vh] overflow-auto"
      >
        <div className="p-8">
          <div className="flex items-start justify-between mb-6">
            <div>
              <h2 className="text-2xl font-bold text-gray-900 mb-2">{model.name}</h2>
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-2">
                  <User className="w-4 h-4 text-gray-500" />
                  <span className="text-gray-700">{model.author.username}</span>
                  {model.author.verified && <CheckCircle className="w-4 h-4 text-blue-500" />}
                </div>
                <div className="flex items-center gap-1">
                  <Star className="w-4 h-4 text-yellow-500 fill-current" />
                  <span>{model.author.reputation}</span>
                </div>
                <span className="text-gray-500">v{model.version}</span>
              </div>
            </div>
            <button
              onClick={() => setSelectedModel(null)}
              className="text-gray-400 hover:text-gray-600"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Main content */}
            <div className="lg:col-span-2 space-y-6">
              <div>
                <h3 className="text-lg font-semibold mb-3">Description</h3>
                <p className="text-gray-600">{model.description}</p>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-3">Performance Metrics</h3>
                <div className="grid grid-cols-2 gap-4">
                  {Object.entries(model.performance_metrics).map(([metric, value]) => (
                    <div key={metric} className="bg-gray-50 p-3 rounded-lg">
                      <div className="text-sm text-gray-600 capitalize">{metric.replace('_', ' ')}</div>
                      <div className="text-lg font-semibold">
                        {typeof value === 'number' ? value.toFixed(3) : value}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-3">Use Cases</h3>
                <div className="grid grid-cols-2 gap-2">
                  {model.use_cases.map((useCase, index) => (
                    <div key={index} className="flex items-center gap-2 text-sm">
                      <Zap className="w-4 h-4 text-blue-500" />
                      {useCase}
                    </div>
                  ))}
                </div>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-3">Requirements</h3>
                <div className="bg-gray-900 text-gray-100 p-4 rounded-lg font-mono text-sm">
                  {model.requirements.map((req, index) => (
                    <div key={index}>{req}</div>
                  ))}
                </div>
              </div>
            </div>

            {/* Sidebar */}
            <div className="space-y-6">
              <div className="bg-gray-50 p-4 rounded-lg">
                <h4 className="font-semibold mb-3">Quick Stats</h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span>Downloads:</span>
                    <span className="font-medium">{model.downloads.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Stars:</span>
                    <span className="font-medium">{model.stars}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Views:</span>
                    <span className="font-medium">{model.views.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Forks:</span>
                    <span className="font-medium">{model.forks}</span>
                  </div>
                </div>
              </div>

              <div className="bg-gray-50 p-4 rounded-lg">
                <h4 className="font-semibold mb-3">Technical Details</h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span>Framework:</span>
                    <span className="font-medium">{model.framework}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Version:</span>
                    <span className="font-medium">{model.framework_version}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Model Size:</span>
                    <span className="font-medium">{model.model_size_mb}MB</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Inference Time:</span>
                    <span className="font-medium">{model.inference_time_ms}ms</span>
                  </div>
                  <div className="flex justify-between">
                    <span>License:</span>
                    <span className="font-medium">{model.license}</span>
                  </div>
                </div>
              </div>

              <div className="bg-gray-50 p-4 rounded-lg">
                <h4 className="font-semibold mb-3">Dataset Info</h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span>Name:</span>
                    <span className="font-medium">{model.dataset_info.name}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Size:</span>
                    <span className="font-medium">{model.dataset_info.size.toLocaleString()} samples</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Features:</span>
                    <span className="font-medium">{model.dataset_info.features}</span>
                  </div>
                </div>
              </div>

              <div className="space-y-3">
                <button
                  onClick={() => handleDownloadModel(model.id)}
                  className="w-full bg-blue-600 text-white py-3 px-4 rounded-lg hover:bg-blue-700 transition-colors flex items-center justify-center gap-2"
                >
                  <Download className="w-5 h-5" />
                  Download Model
                </button>
                
                <div className="grid grid-cols-2 gap-2">
                  <button className="bg-gray-100 text-gray-700 py-2 px-3 rounded-lg hover:bg-gray-200 transition-colors flex items-center justify-center gap-1">
                    <Star className="w-4 h-4" />
                    Star
                  </button>
                  <button className="bg-gray-100 text-gray-700 py-2 px-3 rounded-lg hover:bg-gray-200 transition-colors flex items-center justify-center gap-1">
                    <GitBranch className="w-4 h-4" />
                    Fork
                  </button>
                </div>

                {model.github_url && (
                  <button className="w-full bg-gray-800 text-white py-2 px-4 rounded-lg hover:bg-gray-900 transition-colors flex items-center justify-center gap-2">
                    <Code className="w-4 h-4" />
                    View Source
                  </button>
                )}

                {model.demo_url && (
                  <button className="w-full bg-green-600 text-white py-2 px-4 rounded-lg hover:bg-green-700 transition-colors flex items-center justify-center gap-2">
                    <ExternalLink className="w-4 h-4" />
                    Try Demo
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  );

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
        >
          <Package className="w-8 h-8 text-blue-500" />
        </motion.div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto p-6">
      <div className="mb-8">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 mb-2">
              Model Marketplace
            </h1>
            <p className="text-gray-600">
              Discover, share, and deploy pre-trained machine learning models from the community
            </p>
          </div>
          <button
            onClick={() => setShowUploadModal(true)}
            className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition-colors flex items-center gap-2"
          >
            <Upload className="w-5 h-5" />
            Upload Model
          </button>
        </div>

        {/* Stats */}
        {marketplaceStats && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div className="bg-white p-4 rounded-lg shadow-sm border border-gray-200">
              <div className="flex items-center gap-3">
                <Package className="w-8 h-8 text-blue-500" />
                <div>
                  <p className="text-2xl font-bold text-gray-900">{marketplaceStats.total_models.toLocaleString()}</p>
                  <p className="text-sm text-gray-600">Models</p>
                </div>
              </div>
            </div>
            <div className="bg-white p-4 rounded-lg shadow-sm border border-gray-200">
              <div className="flex items-center gap-3">
                <Download className="w-8 h-8 text-green-500" />
                <div>
                  <p className="text-2xl font-bold text-gray-900">{marketplaceStats.total_downloads.toLocaleString()}</p>
                  <p className="text-sm text-gray-600">Downloads</p>
                </div>
              </div>
            </div>
            <div className="bg-white p-4 rounded-lg shadow-sm border border-gray-200">
              <div className="flex items-center gap-3">
                <Users className="w-8 h-8 text-purple-500" />
                <div>
                  <p className="text-2xl font-bold text-gray-900">{marketplaceStats.active_publishers.toLocaleString()}</p>
                  <p className="text-sm text-gray-600">Publishers</p>
                </div>
              </div>
            </div>
            <div className="bg-white p-4 rounded-lg shadow-sm border border-gray-200">
              <div className="flex items-center gap-3">
                <Award className="w-8 h-8 text-yellow-500" />
                <div>
                  <p className="text-2xl font-bold text-gray-900">{Object.keys(marketplaceStats.categories).length}</p>
                  <p className="text-sm text-gray-600">Categories</p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Search and filters */}
      <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200 mb-6">
        <div className="flex flex-col lg:flex-row gap-4">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
            <input
              type="text"
              placeholder="Search models, authors, or tags..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>
          
          <div className="flex gap-3">
            <select
              value={selectedCategory}
              onChange={(e) => setSelectedCategory(e.target.value)}
              className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            >
              <option value="all">All Categories</option>
              {marketplaceStats && Object.keys(marketplaceStats.categories).map(category => (
                <option key={category} value={category}>{category}</option>
              ))}
            </select>
            
            <select
              value={selectedFramework}
              onChange={(e) => setSelectedFramework(e.target.value)}
              className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            >
              <option value="all">All Frameworks</option>
              <option value="scikit-learn">Scikit-learn</option>
              <option value="pytorch">PyTorch</option>
              <option value="tensorflow">TensorFlow</option>
              <option value="transformers">Transformers</option>
            </select>
            
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as any)}
              className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            >
              <option value="popular">Most Popular</option>
              <option value="recent">Recently Updated</option>
              <option value="stars">Most Stars</option>
              <option value="downloads">Most Downloads</option>
            </select>
          </div>
        </div>
        
        <div className="mt-4 flex items-center justify-between">
          <p className="text-sm text-gray-600">
            Showing {filteredModels.length} of {models.length} models
          </p>
          <div className="flex gap-2">
            <button
              onClick={() => setViewMode('grid')}
              className={`p-2 rounded ${viewMode === 'grid' ? 'bg-blue-100 text-blue-600' : 'text-gray-400'}`}
            >
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                <path d="M5 3a2 2 0 00-2 2v2a2 2 0 002 2h2a2 2 0 002-2V5a2 2 0 00-2-2H5zM5 11a2 2 0 00-2 2v2a2 2 0 002 2h2a2 2 0 002-2v-2a2 2 0 00-2-2H5zM11 5a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V5zM11 13a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
              </svg>
            </button>
            <button
              onClick={() => setViewMode('list')}
              className={`p-2 rounded ${viewMode === 'list' ? 'bg-blue-100 text-blue-600' : 'text-gray-400'}`}
            >
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M3 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1z" clipRule="evenodd" />
              </svg>
            </button>
          </div>
        </div>
      </div>

      {/* Models grid */}
      <div className={viewMode === 'grid' ? 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6' : 'space-y-4'}>
        {filteredModels.map(renderModelCard)}
      </div>

      {filteredModels.length === 0 && (
        <div className="text-center py-12">
          <Package className="w-16 h-16 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No models found</h3>
          <p className="text-gray-500">Try adjusting your search criteria or browse all categories.</p>
        </div>
      )}

      {/* Model detail modal */}
      <AnimatePresence>
        {selectedModel && renderModelDetail(selectedModel)}
      </AnimatePresence>
    </div>
  );
};

export default ModelMarketplace;