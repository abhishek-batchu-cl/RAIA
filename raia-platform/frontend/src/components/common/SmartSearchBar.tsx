import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  TextField,
  InputAdornment,
  Paper,
  List,
  ListItem,
  ListItemText,
  ListItemAvatar,
  Avatar,
  Chip,
  IconButton,
  Typography,
  Divider,
  Box
} from '@mui/material';
import {
  Search,
  Mic,
  MicOff,
  Sparkles,
  Clock,
  TrendingUp,
  AlertTriangle,
  Brain,
  Database,
  Shield,
  FileText,
  Settings
} from 'lucide-react';

interface SearchSuggestion {
  id: string;
  title: string;
  description: string;
  category: 'model' | 'metric' | 'data' | 'compliance' | 'recent' | 'action';
  icon: React.ReactNode;
  relevanceScore?: number;
  isRecent?: boolean;
  onClick?: () => void;
}

interface SmartSearchBarProps {
  placeholder?: string;
  onSearch?: (query: string) => void;
  onSuggestionClick?: (suggestion: SearchSuggestion) => void;
  showVoiceInput?: boolean;
  showRecentSearches?: boolean;
  maxSuggestions?: number;
  className?: string;
}

const SmartSearchBar: React.FC<SmartSearchBarProps> = ({
  placeholder = "Ask me anything about your models...",
  onSearch,
  onSuggestionClick,
  showVoiceInput = true,
  showRecentSearches = true,
  maxSuggestions = 8,
  className = ''
}) => {
  const [query, setQuery] = useState('');
  const [isOpen, setIsOpen] = useState(false);
  const [suggestions, setSuggestions] = useState<SearchSuggestion[]>([]);
  const [isListening, setIsListening] = useState(false);
  const [recentSearches, setRecentSearches] = useState<SearchSuggestion[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);
  const searchRef = useRef<HTMLDivElement>(null);

  // Mock data for suggestions
  const allSuggestions: SearchSuggestion[] = [
    {
      id: '1',
      title: 'Show bias metrics for Credit Risk model',
      description: 'View fairness analysis and bias indicators',
      category: 'model',
      icon: <Brain size={18} className="text-blue-500" />,
      relevanceScore: 0.95
    },
    {
      id: '2', 
      title: 'Models with accuracy below 85%',
      description: 'Find underperforming models requiring attention',
      category: 'metric',
      icon: <TrendingUp size={18} className="text-orange-500" />,
      relevanceScore: 0.88
    },
    {
      id: '3',
      title: 'Data quality issues in training set',
      description: 'Identify missing values and anomalies',
      category: 'data',
      icon: <Database size={18} className="text-green-500" />,
      relevanceScore: 0.82
    },
    {
      id: '4',
      title: 'GDPR compliance report',
      description: 'Generate regulatory compliance documentation',
      category: 'compliance',
      icon: <Shield size={18} className="text-purple-500" />,
      relevanceScore: 0.76
    },
    {
      id: '5',
      title: 'Export model performance dashboard',
      description: 'Download charts and metrics as PDF',
      category: 'action',
      icon: <FileText size={18} className="text-indigo-500" />,
      relevanceScore: 0.70
    }
  ];

  // Recent searches mock data
  const mockRecentSearches: SearchSuggestion[] = [
    {
      id: 'recent1',
      title: 'bias detection customer churn',
      description: '2 hours ago',
      category: 'recent',
      icon: <Clock size={18} className="text-gray-400" />,
      isRecent: true
    },
    {
      id: 'recent2', 
      title: 'model drift analysis',
      description: 'Yesterday',
      category: 'recent',
      icon: <Clock size={18} className="text-gray-400" />,
      isRecent: true
    }
  ];

  useEffect(() => {
    setRecentSearches(mockRecentSearches);
  }, []);

  // Smart search logic
  useEffect(() => {
    if (query.trim() === '') {
      if (showRecentSearches && recentSearches.length > 0) {
        setSuggestions(recentSearches.slice(0, 3));
      } else {
        setSuggestions([]);
      }
      return;
    }

    // AI-powered search suggestions
    const filtered = allSuggestions
      .filter(s => 
        s.title.toLowerCase().includes(query.toLowerCase()) ||
        s.description.toLowerCase().includes(query.toLowerCase())
      )
      .sort((a, b) => (b.relevanceScore || 0) - (a.relevanceScore || 0))
      .slice(0, maxSuggestions);

    setSuggestions(filtered);
  }, [query, recentSearches, showRecentSearches, maxSuggestions]);

  // Voice recognition
  useEffect(() => {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
      return;
    }

    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    const recognition = new SpeechRecognition();
    
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';

    recognition.onstart = () => setIsListening(true);
    recognition.onend = () => setIsListening(false);
    recognition.onresult = (event: any) => {
      const transcript = event.results[0][0].transcript;
      setQuery(transcript);
      handleSearch(transcript);
    };

    if (isListening) {
      recognition.start();
    }

    return () => recognition.stop();
  }, [isListening]);

  // Close suggestions when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (searchRef.current && !searchRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleSearch = (searchQuery: string) => {
    if (!searchQuery.trim()) return;

    // Add to recent searches
    const newSearch: SearchSuggestion = {
      id: `recent-${Date.now()}`,
      title: searchQuery,
      description: 'Just searched',
      category: 'recent',
      icon: <Clock size={18} className="text-gray-400" />,
      isRecent: true
    };

    setRecentSearches(prev => [newSearch, ...prev.slice(0, 4)]);
    setIsOpen(false);
    onSearch?.(searchQuery);
  };

  const handleSuggestionClick = (suggestion: SearchSuggestion) => {
    setQuery(suggestion.title);
    setIsOpen(false);
    onSuggestionClick?.(suggestion);
    handleSearch(suggestion.title);
  };

  const handleVoiceToggle = () => {
    setIsListening(!isListening);
  };

  const getCategoryColor = (category: SearchSuggestion['category']) => {
    switch (category) {
      case 'model': return 'primary';
      case 'metric': return 'warning';
      case 'data': return 'success';
      case 'compliance': return 'secondary';
      case 'action': return 'info';
      default: return 'default';
    }
  };

  return (
    <div ref={searchRef} className={`relative w-full max-w-2xl ${className}`}>
      <TextField
        ref={inputRef}
        fullWidth
        variant="outlined"
        placeholder={placeholder}
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onFocus={() => setIsOpen(true)}
        onKeyDown={(e) => {
          if (e.key === 'Enter') {
            handleSearch(query);
          }
          if (e.key === 'Escape') {
            setIsOpen(false);
          }
        }}
        InputProps={{
          startAdornment: (
            <InputAdornment position="start">
              <Search size={20} className="text-gray-400" />
            </InputAdornment>
          ),
          endAdornment: showVoiceInput ? (
            <InputAdornment position="end">
              <motion.div
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <IconButton
                  onClick={handleVoiceToggle}
                  className={isListening ? 'text-red-500' : 'text-gray-400'}
                  size="small"
                >
                  {isListening ? <MicOff size={18} /> : <Mic size={18} />}
                </IconButton>
              </motion.div>
            </InputAdornment>
          ) : undefined,
        }}
        sx={{
          '& .MuiOutlinedInput-root': {
            backgroundColor: 'rgba(255, 255, 255, 0.9)',
            backdropFilter: 'blur(10px)',
            '&:hover': {
              backgroundColor: 'rgba(255, 255, 255, 0.95)',
            },
            '&.Mui-focused': {
              backgroundColor: 'white',
              boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)',
            }
          }
        }}
      />

      <AnimatePresence>
        {isOpen && (suggestions.length > 0 || isListening) && (
          <motion.div
            initial={{ opacity: 0, y: -10, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -10, scale: 0.95 }}
            transition={{ duration: 0.2, ease: "easeOut" }}
            className="absolute top-full left-0 right-0 mt-2 z-50"
          >
            <Paper 
              elevation={8}
              className="glass rounded-xl border border-white/20 max-h-96 overflow-hidden"
            >
              {isListening && (
                <Box className="p-4 text-center border-b border-gray-200 dark:border-gray-700">
                  <motion.div
                    animate={{ scale: [1, 1.1, 1] }}
                    transition={{ duration: 1.5, repeat: Infinity }}
                    className="inline-flex items-center space-x-2 text-red-500"
                  >
                    <Mic size={20} />
                    <Typography variant="body2">Listening...</Typography>
                  </motion.div>
                </Box>
              )}

              <List className="p-0 max-h-80 overflow-y-auto custom-scrollbar">
                {query.trim() === '' && recentSearches.length > 0 && (
                  <>
                    <ListItem className="bg-gray-50 dark:bg-gray-800 px-4 py-2">
                      <Typography variant="caption" className="font-medium text-gray-600 dark:text-gray-400 uppercase tracking-wide">
                        Recent Searches
                      </Typography>
                    </ListItem>
                    <Divider />
                  </>
                )}

                {suggestions.map((suggestion, index) => (
                  <motion.div
                    key={suggestion.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.05 }}
                  >
                    <ListItem
                      button
                      onClick={() => handleSuggestionClick(suggestion)}
                      className="hover:bg-gray-50 dark:hover:bg-gray-700 px-4 py-3"
                    >
                      <ListItemAvatar>
                        <Avatar className="w-8 h-8 bg-transparent">
                          {suggestion.icon}
                        </Avatar>
                      </ListItemAvatar>
                      <ListItemText
                        primary={
                          <div className="flex items-center justify-between">
                            <Typography variant="body2" className="font-medium">
                              {suggestion.title}
                            </Typography>
                            {suggestion.relevanceScore && (
                              <Chip
                                size="small"
                                label={`${Math.round(suggestion.relevanceScore * 100)}%`}
                                color={getCategoryColor(suggestion.category)}
                                variant="outlined"
                                className="ml-2"
                              />
                            )}
                          </div>
                        }
                        secondary={
                          <Typography variant="caption" color="textSecondary">
                            {suggestion.description}
                          </Typography>
                        }
                      />
                      {suggestion.category !== 'recent' && (
                        <Sparkles size={16} className="text-blue-400 ml-2" />
                      )}
                    </ListItem>
                    {index < suggestions.length - 1 && <Divider />}
                  </motion.div>
                ))}

                {query.trim() !== '' && suggestions.length === 0 && (
                  <ListItem className="px-4 py-8 text-center">
                    <div className="w-full text-center">
                      <Typography variant="body2" color="textSecondary" className="mb-2">
                        No suggestions found
                      </Typography>
                      <Typography variant="caption" color="textSecondary">
                        Try a different search term or ask a question
                      </Typography>
                    </div>
                  </ListItem>
                )}
              </List>

              {/* AI-powered suggestions footer */}
              <div className="p-3 bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 border-t border-gray-200 dark:border-gray-700">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <Sparkles size={14} className="text-blue-500" />
                    <Typography variant="caption" className="text-blue-600 dark:text-blue-400">
                      AI-powered suggestions
                    </Typography>
                  </div>
                  <Typography variant="caption" color="textSecondary">
                    Press Enter to search
                  </Typography>
                </div>
              </div>
            </Paper>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default SmartSearchBar;