import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Search, Command, ArrowRight, Clock, Star, Folder, FileText,
  Database, Brain, TrendingUp, Settings, Users, Package,
  Play, GitBranch, AlertTriangle, Target, Activity, Zap,
  BookOpen, Code, Eye, Download, Upload, Calendar, BarChart3
} from 'lucide-react';

interface SearchResult {
  id: string;
  type: 'model' | 'experiment' | 'dataset' | 'workflow' | 'report' | 'user' | 'command' | 'page';
  title: string;
  description: string;
  category: string;
  url?: string;
  action?: () => void;
  metadata?: {
    author?: string;
    created_at?: string;
    updated_at?: string;
    status?: string;
    tags?: string[];
    accuracy?: number;
    downloads?: number;
    version?: string;
  };
  icon?: React.ComponentType<any>;
  score: number; // Search relevance score
}

interface SearchCategory {
  id: string;
  name: string;
  icon: React.ComponentType<any>;
  count: number;
}

interface RecentSearch {
  query: string;
  timestamp: string;
  results_count: number;
}

interface QuickAction {
  id: string;
  title: string;
  description: string;
  icon: React.ComponentType<any>;
  shortcut?: string;
  action: () => void;
  category: string;
}

const GlobalSearchCommandPalette: React.FC<{
  isOpen: boolean;
  onClose: () => void;
}> = ({ isOpen, onClose }) => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [activeCategory, setActiveCategory] = useState<string>('all');
  const [recentSearches, setRecentSearches] = useState<RecentSearch[]>([]);
  const [searchHistory, setSearchHistory] = useState<string[]>([]);

  // Mock data - in real app, this would come from APIs
  const mockResults: SearchResult[] = useMemo(() => [
    // Models
    {
      id: 'model_001',
      type: 'model',
      title: 'Customer Churn Predictor v2.1',
      description: 'Advanced ensemble model for predicting customer churn with 94.2% accuracy',
      category: 'Classification Models',
      url: '/models/model_001',
      metadata: {
        author: 'sarah_ml',
        accuracy: 0.942,
        downloads: 8934,
        status: 'active',
        tags: ['churn', 'classification', 'ensemble']
      },
      icon: Brain,
      score: 0.95
    },
    {
      id: 'model_002',
      type: 'model',
      title: 'NLP Sentiment Analysis Pro',
      description: 'Multi-language sentiment analysis with transformer architecture',
      category: 'NLP Models',
      url: '/models/model_002',
      metadata: {
        author: 'alex_nlp',
        accuracy: 0.891,
        downloads: 12456,
        status: 'active',
        tags: ['nlp', 'sentiment', 'transformer']
      },
      icon: BookOpen,
      score: 0.92
    },
    
    // Experiments
    {
      id: 'exp_001',
      type: 'experiment',
      title: 'Churn Model Hyperparameter Tuning',
      description: 'Experiment comparing different hyperparameter combinations',
      category: 'Active Experiments',
      url: '/experiments/exp_001',
      metadata: {
        author: 'data_scientist',
        status: 'running',
        created_at: '2024-01-15T10:30:00Z'
      },
      icon: Target,
      score: 0.88
    },
    
    // Datasets
    {
      id: 'dataset_001',
      type: 'dataset',
      title: 'Customer Behavior Dataset 2024',
      description: 'Comprehensive customer interaction data with 500K samples',
      category: 'Training Data',
      url: '/datasets/dataset_001',
      metadata: {
        author: 'data_team',
        created_at: '2024-01-10T14:20:00Z',
        tags: ['customers', 'behavior', 'interactions']
      },
      icon: Database,
      score: 0.85
    },
    
    // Workflows
    {
      id: 'workflow_001',
      type: 'workflow',
      title: 'ML Pipeline: Data to Production',
      description: 'End-to-end ML pipeline from data ingestion to model deployment',
      category: 'ML Workflows',
      url: '/workflows/workflow_001',
      metadata: {
        author: 'ml_engineer',
        status: 'active',
        updated_at: '2024-01-16T09:15:00Z'
      },
      icon: GitBranch,
      score: 0.82
    },
    
    // Reports
    {
      id: 'report_001',
      type: 'report',
      title: 'Monthly Model Performance Report',
      description: 'Comprehensive analysis of all production model performance',
      category: 'Performance Reports',
      url: '/reports/report_001',
      metadata: {
        author: 'analyst_team',
        created_at: '2024-01-15T08:00:00Z',
        tags: ['performance', 'monthly', 'production']
      },
      icon: BarChart3,
      score: 0.78
    },
    
    // Users
    {
      id: 'user_001',
      type: 'user',
      title: 'Sarah Chen - ML Engineer',
      description: 'Senior ML Engineer specializing in computer vision',
      category: 'Team Members',
      url: '/team/user_001',
      metadata: {
        status: 'online',
        tags: ['computer-vision', 'deep-learning', 'pytorch']
      },
      icon: Users,
      score: 0.75
    }
  ], []);

  const quickActions: QuickAction[] = useMemo(() => [
    {
      id: 'create_model',
      title: 'Upload New Model',
      description: 'Upload and register a new ML model',
      icon: Upload,
      shortcut: '⌘+U',
      action: () => console.log('Navigate to model upload'),
      category: 'Create'
    },
    {
      id: 'new_experiment',
      title: 'Start Experiment',
      description: 'Create a new ML experiment',
      icon: Play,
      shortcut: '⌘+E',
      action: () => console.log('Navigate to new experiment'),
      category: 'Create'
    },
    {
      id: 'create_workflow',
      title: 'Build Workflow',
      description: 'Create a new ML pipeline workflow',
      icon: GitBranch,
      shortcut: '⌘+W',
      action: () => console.log('Navigate to workflow builder'),
      category: 'Create'
    },
    {
      id: 'generate_report',
      title: 'Generate Report',
      description: 'Create automated performance report',
      icon: FileText,
      shortcut: '⌘+R',
      action: () => console.log('Navigate to report generator'),
      category: 'Analytics'
    },
    {
      id: 'marketplace',
      title: 'Browse Marketplace',
      description: 'Discover community models',
      icon: Package,
      shortcut: '⌘+M',
      action: () => console.log('Navigate to marketplace'),
      category: 'Discover'
    },
    {
      id: 'monitoring',
      title: 'View Monitoring',
      description: 'Check real-time model performance',
      icon: Activity,
      shortcut: '⌘+L',
      action: () => console.log('Navigate to monitoring'),
      category: 'Monitor'
    },
    {
      id: 'settings',
      title: 'Open Settings',
      description: 'Configure platform settings',
      icon: Settings,
      shortcut: '⌘+,',
      action: () => console.log('Navigate to settings'),
      category: 'System'
    }
  ], []);

  const categories: SearchCategory[] = useMemo(() => [
    { id: 'all', name: 'All', icon: Search, count: mockResults.length },
    { id: 'model', name: 'Models', icon: Brain, count: mockResults.filter(r => r.type === 'model').length },
    { id: 'experiment', name: 'Experiments', icon: Target, count: mockResults.filter(r => r.type === 'experiment').length },
    { id: 'dataset', name: 'Datasets', icon: Database, count: mockResults.filter(r => r.type === 'dataset').length },
    { id: 'workflow', name: 'Workflows', icon: GitBranch, count: mockResults.filter(r => r.type === 'workflow').length },
    { id: 'report', name: 'Reports', icon: BarChart3, count: mockResults.filter(r => r.type === 'report').length },
    { id: 'user', name: 'People', icon: Users, count: mockResults.filter(r => r.type === 'user').length }
  ], [mockResults]);

  // Search logic
  const performSearch = useCallback(async (searchQuery: string) => {
    setIsLoading(true);
    
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 150));
    
    if (!searchQuery.trim()) {
      setResults([]);
      setIsLoading(false);
      return;
    }
    
    // Simple text search with scoring
    const filtered = mockResults.filter(result => {
      const searchText = `${result.title} ${result.description} ${result.category} ${result.metadata?.tags?.join(' ') || ''}`.toLowerCase();
      const queryLower = searchQuery.toLowerCase();
      
      // Calculate relevance score
      let score = 0;
      
      // Title match (highest weight)
      if (result.title.toLowerCase().includes(queryLower)) {
        score += 3;
      }
      
      // Description match
      if (result.description.toLowerCase().includes(queryLower)) {
        score += 2;
      }
      
      // Category match
      if (result.category.toLowerCase().includes(queryLower)) {
        score += 1;
      }
      
      // Tag match
      if (result.metadata?.tags?.some(tag => tag.toLowerCase().includes(queryLower))) {
        score += 1;
      }
      
      // Author match
      if (result.metadata?.author?.toLowerCase().includes(queryLower)) {
        score += 1;
      }
      
      return score > 0;
    }).map(result => ({ ...result, score: result.score }));
    
    // Sort by relevance score and type priority
    const sorted = filtered.sort((a, b) => {
      // Type priority
      const typePriority = { model: 4, experiment: 3, dataset: 2, workflow: 2, report: 1, user: 1, command: 5, page: 0 };
      const aPriority = typePriority[a.type] || 0;
      const bPriority = typePriority[b.type] || 0;
      
      if (aPriority !== bPriority) {
        return bPriority - aPriority;
      }
      
      // Then by search score
      return b.score - a.score;
    });
    
    // Filter by active category
    const categoryFiltered = activeCategory === 'all' 
      ? sorted 
      : sorted.filter(r => r.type === activeCategory);
    
    setResults(categoryFiltered);
    setSelectedIndex(0);
    setIsLoading(false);
  }, [mockResults, activeCategory]);

  // Debounced search
  useEffect(() => {
    const timer = setTimeout(() => {
      performSearch(query);
    }, 200);
    
    return () => clearTimeout(timer);
  }, [query, performSearch]);

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!isOpen) return;

      switch (e.key) {
        case 'ArrowDown':
          e.preventDefault();
          setSelectedIndex(prev => 
            prev < (query ? results.length - 1 : quickActions.length - 1) ? prev + 1 : 0
          );
          break;
        case 'ArrowUp':
          e.preventDefault();
          setSelectedIndex(prev => 
            prev > 0 ? prev - 1 : (query ? results.length - 1 : quickActions.length - 1)
          );
          break;
        case 'Enter':
          e.preventDefault();
          handleSelectItem();
          break;
        case 'Escape':
          e.preventDefault();
          onClose();
          break;
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, results.length, quickActions.length, selectedIndex, query]);

  const handleSelectItem = useCallback(() => {
    const items = query ? results : quickActions;
    const selectedItem = items[selectedIndex];
    
    if (!selectedItem) return;
    
    if ('action' in selectedItem) {
      selectedItem.action();
    } else if (selectedItem.url) {
      // Navigate to URL
      console.log('Navigate to:', selectedItem.url);
    } else if (selectedItem.action) {
      selectedItem.action();
    }
    
    // Save to recent searches
    if (query && query !== recentSearches[0]?.query) {
      const newSearch: RecentSearch = {
        query,
        timestamp: new Date().toISOString(),
        results_count: results.length
      };
      setRecentSearches(prev => [newSearch, ...prev.slice(0, 9)]);
    }
    
    onClose();
  }, [query, results, selectedIndex, quickActions, recentSearches, onClose]);

  const getResultIcon = (result: SearchResult) => {
    if (result.icon) return result.icon;
    
    switch (result.type) {
      case 'model': return Brain;
      case 'experiment': return Target;
      case 'dataset': return Database;
      case 'workflow': return GitBranch;
      case 'report': return BarChart3;
      case 'user': return Users;
      default: return FileText;
    }
  };

  const getStatusColor = (status?: string) => {
    switch (status) {
      case 'active': case 'online': case 'running': return 'text-green-600 bg-green-50';
      case 'inactive': case 'offline': case 'paused': return 'text-yellow-600 bg-yellow-50';
      case 'failed': case 'error': return 'text-red-600 bg-red-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  const highlightMatch = (text: string, query: string) => {
    if (!query) return text;
    
    const parts = text.split(new RegExp(`(${query})`, 'gi'));
    return parts.map((part, index) => 
      part.toLowerCase() === query.toLowerCase() ? 
        <mark key={index} className="bg-yellow-200 text-yellow-900">{part}</mark> : 
        part
    );
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-start justify-center pt-20 z-50">
        <motion.div
          initial={{ opacity: 0, scale: 0.95, y: -20 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95, y: -20 }}
          className="bg-white rounded-lg shadow-xl w-full max-w-2xl mx-4 max-h-[70vh] flex flex-col overflow-hidden"
        >
          {/* Search Header */}
          <div className="flex items-center px-4 py-3 border-b border-gray-200">
            <Search className="w-5 h-5 text-gray-400 mr-3" />
            <input
              type="text"
              placeholder="Search models, experiments, datasets, or type a command..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="flex-1 outline-none text-lg"
              autoFocus
            />
            {isLoading && (
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full"
              />
            )}
            <div className="flex items-center gap-1 text-xs text-gray-500 ml-3">
              <kbd className="bg-gray-100 px-2 py-1 rounded">↑↓</kbd>
              <span>to navigate</span>
              <kbd className="bg-gray-100 px-2 py-1 rounded ml-2">↵</kbd>
              <span>to select</span>
            </div>
          </div>

          {/* Categories */}
          {query && (
            <div className="flex gap-1 px-4 py-2 border-b border-gray-200 overflow-x-auto">
              {categories.map((category) => (
                <button
                  key={category.id}
                  onClick={() => setActiveCategory(category.id)}
                  className={`flex items-center gap-2 px-3 py-1 rounded-full text-sm whitespace-nowrap transition-colors ${
                    activeCategory === category.id
                      ? 'bg-blue-100 text-blue-700'
                      : 'text-gray-600 hover:bg-gray-100'
                  }`}
                >
                  <category.icon className="w-4 h-4" />
                  {category.name}
                  <span className="text-xs bg-gray-200 text-gray-600 px-1 rounded-full min-w-[16px] text-center">
                    {category.count}
                  </span>
                </button>
              ))}
            </div>
          )}

          {/* Results */}
          <div className="flex-1 overflow-auto">
            {!query ? (
              /* Quick Actions */
              <div className="p-4">
                <div className="text-sm text-gray-500 mb-3 flex items-center gap-2">
                  <Zap className="w-4 h-4" />
                  Quick Actions
                </div>
                <div className="space-y-1">
                  {quickActions.map((action, index) => (
                    <motion.div
                      key={action.id}
                      className={`flex items-center gap-3 p-3 rounded-lg cursor-pointer transition-colors ${
                        index === selectedIndex ? 'bg-blue-50 border-l-4 border-l-blue-500' : 'hover:bg-gray-50'
                      }`}
                      onClick={() => {
                        setSelectedIndex(index);
                        handleSelectItem();
                      }}
                    >
                      <action.icon className="w-5 h-5 text-gray-500" />
                      <div className="flex-1">
                        <div className="font-medium text-gray-900">{action.title}</div>
                        <div className="text-sm text-gray-600">{action.description}</div>
                      </div>
                      {action.shortcut && (
                        <kbd className="bg-gray-100 text-gray-600 px-2 py-1 rounded text-xs">
                          {action.shortcut}
                        </kbd>
                      )}
                      <ArrowRight className="w-4 h-4 text-gray-400" />
                    </motion.div>
                  ))}
                </div>

                {/* Recent Searches */}
                {recentSearches.length > 0 && (
                  <div className="mt-6">
                    <div className="text-sm text-gray-500 mb-3 flex items-center gap-2">
                      <Clock className="w-4 h-4" />
                      Recent Searches
                    </div>
                    <div className="space-y-1">
                      {recentSearches.slice(0, 3).map((search, index) => (
                        <div
                          key={index}
                          className="flex items-center gap-3 p-2 rounded-lg cursor-pointer hover:bg-gray-50"
                          onClick={() => setQuery(search.query)}
                        >
                          <Search className="w-4 h-4 text-gray-400" />
                          <span className="flex-1 text-gray-700">{search.query}</span>
                          <span className="text-xs text-gray-500">
                            {search.results_count} results
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              /* Search Results */
              <div className="p-4">
                {results.length === 0 && !isLoading ? (
                  <div className="text-center py-8 text-gray-500">
                    <Search className="w-12 h-12 mx-auto mb-4 text-gray-300" />
                    <p>No results found for "{query}"</p>
                    <p className="text-sm mt-1">Try different keywords or browse quick actions</p>
                  </div>
                ) : (
                  <div className="space-y-1">
                    {results.map((result, index) => {
                      const Icon = getResultIcon(result);
                      return (
                        <motion.div
                          key={result.id}
                          className={`flex items-start gap-3 p-3 rounded-lg cursor-pointer transition-colors ${
                            index === selectedIndex ? 'bg-blue-50 border-l-4 border-l-blue-500' : 'hover:bg-gray-50'
                          }`}
                          onClick={() => {
                            setSelectedIndex(index);
                            handleSelectItem();
                          }}
                          initial={{ opacity: 0, y: 20 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: index * 0.05 }}
                        >
                          <Icon className="w-5 h-5 text-gray-500 mt-0.5" />
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 mb-1">
                              <span className="font-medium text-gray-900">
                                {highlightMatch(result.title, query)}
                              </span>
                              {result.metadata?.status && (
                                <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${getStatusColor(result.metadata.status)}`}>
                                  {result.metadata.status}
                                </span>
                              )}
                              {result.metadata?.version && (
                                <span className="text-xs text-gray-500">
                                  {result.metadata.version}
                                </span>
                              )}
                            </div>
                            <div className="text-sm text-gray-600 mb-2">
                              {highlightMatch(result.description, query)}
                            </div>
                            <div className="flex items-center gap-4 text-xs text-gray-500">
                              <span className="text-blue-600">{result.category}</span>
                              {result.metadata?.author && (
                                <span>by {result.metadata.author}</span>
                              )}
                              {result.metadata?.accuracy && (
                                <span>{(result.metadata.accuracy * 100).toFixed(1)}% accuracy</span>
                              )}
                              {result.metadata?.downloads && (
                                <span>{result.metadata.downloads.toLocaleString()} downloads</span>
                              )}
                              {result.metadata?.created_at && (
                                <span>{new Date(result.metadata.created_at).toLocaleDateString()}</span>
                              )}
                            </div>
                            {result.metadata?.tags && result.metadata.tags.length > 0 && (
                              <div className="flex flex-wrap gap-1 mt-2">
                                {result.metadata.tags.slice(0, 3).map((tag, tagIndex) => (
                                  <span
                                    key={tagIndex}
                                    className="text-xs bg-gray-100 text-gray-600 px-2 py-0.5 rounded-full"
                                  >
                                    {tag}
                                  </span>
                                ))}
                                {result.metadata.tags.length > 3 && (
                                  <span className="text-xs text-gray-400">
                                    +{result.metadata.tags.length - 3} more
                                  </span>
                                )}
                              </div>
                            )}
                          </div>
                          <ArrowRight className="w-4 h-4 text-gray-400 mt-0.5" />
                        </motion.div>
                      );
                    })}
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Footer */}
          <div className="border-t border-gray-200 px-4 py-2">
            <div className="flex items-center justify-between text-xs text-gray-500">
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-1">
                  <Command className="w-3 h-3" />
                  <span>Search across all resources</span>
                </div>
                {results.length > 0 && (
                  <span>{results.length} results found</span>
                )}
              </div>
              <div className="flex items-center gap-1">
                <kbd className="bg-gray-100 px-2 py-1 rounded">ESC</kbd>
                <span>to close</span>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
};

export default GlobalSearchCommandPalette;