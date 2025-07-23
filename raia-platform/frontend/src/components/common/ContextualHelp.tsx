import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  HelpCircle, X, Lightbulb, ArrowRight, BookOpen, 
  PlayCircle, ExternalLink, CheckCircle, Star,
  Clock, Users, Zap
} from 'lucide-react';
import { cn } from '../../utils';

interface HelpTip {
  id: string;
  title: string;
  description: string;
  type: 'quick-tip' | 'tutorial' | 'best-practice' | 'warning';
  estimatedTime?: string;
  difficulty?: 'beginner' | 'intermediate' | 'advanced';
  actionLabel?: string;
  actionUrl?: string;
  videoUrl?: string;
}

interface ContextualHelpProps {
  pageId: string;
  className?: string;
}

// Help content database - would typically come from CMS or API
const helpContent: Record<string, HelpTip[]> = {
  'overview': [
    {
      id: 'overview-basics',
      title: 'Understanding Your Model Dashboard',
      description: 'This page shows your model\'s key performance indicators at a glance. Start by checking the accuracy score and confidence intervals.',
      type: 'quick-tip',
      estimatedTime: '2 min',
      difficulty: 'beginner',
    },
    {
      id: 'overview-alerts',
      title: 'Pay Attention to Alerts',
      description: 'Red indicators mean your model needs attention. Yellow warnings suggest monitoring closely.',
      type: 'warning',
      difficulty: 'beginner',
    },
  ],
  'feature-importance': [
    {
      id: 'feature-basics',
      title: 'What is Feature Importance?',
      description: 'Feature importance tells you which input variables have the biggest impact on your model\'s predictions. Higher values = more important.',
      type: 'tutorial',
      estimatedTime: '5 min',
      difficulty: 'beginner',
      actionLabel: 'Watch Tutorial',
      videoUrl: '#tutorial-feature-importance',
    },
    {
      id: 'feature-interpretation',
      title: 'How to Interpret the Rankings',
      description: 'Features at the top drive most decisions. Focus on the top 5-10 features for model improvements.',
      type: 'best-practice',
      estimatedTime: '3 min',
      difficulty: 'beginner',
    },
    {
      id: 'feature-advanced',
      title: 'SHAP vs Permutation Importance',
      description: 'SHAP shows how features contribute to individual predictions. Permutation shows overall feature impact.',
      type: 'tutorial',
      estimatedTime: '8 min',
      difficulty: 'advanced',
    },
  ],
  'classification-stats': [
    {
      id: 'confusion-matrix',
      title: 'Reading the Confusion Matrix',
      description: 'The confusion matrix shows correct vs incorrect predictions. Diagonal cells = correct predictions.',
      type: 'tutorial',
      estimatedTime: '4 min',
      difficulty: 'beginner',
    },
    {
      id: 'precision-recall',
      title: 'Precision vs Recall Trade-off',
      description: 'High precision = fewer false positives. High recall = fewer false negatives. You usually can\'t have both perfect.',
      type: 'best-practice',
      estimatedTime: '6 min',
      difficulty: 'intermediate',
    },
  ],
  'regression-stats': [
    {
      id: 'r2-score',
      title: 'Understanding R² Score',
      description: 'R² measures how well your model explains the data. 1.0 = perfect, 0.0 = no better than average.',
      type: 'tutorial',
      estimatedTime: '3 min',
      difficulty: 'beginner',
    },
    {
      id: 'residual-plots',
      title: 'Reading Residual Plots',
      description: 'Residuals should be randomly scattered. Patterns indicate model problems or missing features.',
      type: 'best-practice',
      estimatedTime: '7 min',
      difficulty: 'intermediate',
    },
  ],
  'predictions': [
    {
      id: 'shap-waterfall',
      title: 'How to Read SHAP Waterfall Charts',
      description: 'Start from the baseline (average). Each bar shows how much each feature pushes the prediction up or down.',
      type: 'tutorial',
      estimatedTime: '5 min',
      difficulty: 'beginner',
    },
    {
      id: 'local-explanations',
      title: 'Local vs Global Explanations',
      description: 'This page shows local explanations (why this specific prediction). Global explanations show overall model behavior.',
      type: 'quick-tip',
      estimatedTime: '2 min',
      difficulty: 'intermediate',
    },
  ],
  'what-if': [
    {
      id: 'scenario-testing',
      title: 'Effective Scenario Testing',
      description: 'Change one feature at a time to see its isolated impact. Then try realistic combinations.',
      type: 'best-practice',
      estimatedTime: '4 min',
      difficulty: 'beginner',
    },
    {
      id: 'counterfactual',
      title: 'Finding Counterfactual Examples',
      description: 'Ask "What\'s the smallest change needed to flip the prediction?" This helps find model decision boundaries.',
      type: 'tutorial',
      estimatedTime: '8 min',
      difficulty: 'advanced',
    },
  ],
};

const ContextualHelp: React.FC<ContextualHelpProps> = ({ pageId, className }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [currentTipIndex, setCurrentTipIndex] = useState(0);
  const [completedTips, setCompletedTips] = useState<Set<string>>(new Set());

  const tips = helpContent[pageId] || [];
  const currentTip = tips[currentTipIndex];
  const hasMultipleTips = tips.length > 1;

  useEffect(() => {
    setCurrentTipIndex(0);
  }, [pageId]);

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'tutorial': return PlayCircle;
      case 'best-practice': return Star;
      case 'warning': return HelpCircle;
      default: return Lightbulb;
    }
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'tutorial': return 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300';
      case 'best-practice': return 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300';
      case 'warning': return 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300';
      default: return 'bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-300';
    }
  };

  const getDifficultyColor = (difficulty?: string) => {
    switch (difficulty) {
      case 'beginner': return 'text-green-600 bg-green-50 dark:text-green-400 dark:bg-green-900/20';
      case 'intermediate': return 'text-amber-600 bg-amber-50 dark:text-amber-400 dark:bg-amber-900/20';
      case 'advanced': return 'text-red-600 bg-red-50 dark:text-red-400 dark:bg-red-900/20';
      default: return 'text-neutral-600 bg-neutral-50 dark:text-neutral-400 dark:bg-neutral-900/20';
    }
  };

  const markAsCompleted = (tipId: string) => {
    setCompletedTips(prev => new Set([...prev, tipId]));
  };

  const nextTip = () => {
    if (currentTipIndex < tips.length - 1) {
      setCurrentTipIndex(prev => prev + 1);
    }
  };

  const prevTip = () => {
    if (currentTipIndex > 0) {
      setCurrentTipIndex(prev => prev - 1);
    }
  };

  if (tips.length === 0) return null;

  return (
    <div className={cn("relative", className)}>
      {/* Help Trigger Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={cn(
          "flex items-center space-x-2 px-3 py-2 rounded-lg transition-all duration-200 border-2",
          isOpen
            ? "bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300 border-primary-200 dark:border-primary-700"
            : "bg-white dark:bg-neutral-800 text-neutral-700 dark:text-neutral-300 border-neutral-200 dark:border-neutral-700 hover:border-primary-200 dark:hover:border-primary-700 shadow-sm"
        )}
      >
        <HelpCircle className="w-4 h-4" />
        <span className="text-sm font-medium hidden sm:inline">
          Help ({tips.length} {tips.length === 1 ? 'tip' : 'tips'})
        </span>
        {tips.filter(tip => !completedTips.has(tip.id)).length > 0 && (
          <div className="w-2 h-2 bg-primary-500 rounded-full animate-pulse"></div>
        )}
      </button>

      {/* Help Panel */}
      <AnimatePresence>
        {isOpen && currentTip && (
          <motion.div
            initial={{ opacity: 0, y: 10, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 10, scale: 0.95 }}
            transition={{ duration: 0.2 }}
            className="absolute top-full right-0 mt-2 w-80 bg-white dark:bg-neutral-800 rounded-lg shadow-lg border border-neutral-200 dark:border-neutral-700 z-50"
          >
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b border-neutral-200 dark:border-neutral-700">
              <div className="flex items-center space-x-2">
                {React.createElement(getTypeIcon(currentTip.type), {
                  className: "w-4 h-4 text-primary-600 dark:text-primary-400"
                })}
                <span className="text-sm font-medium text-neutral-900 dark:text-white">
                  Helpful Tips
                </span>
              </div>
              <button
                onClick={() => setIsOpen(false)}
                className="p-1 rounded-md hover:bg-neutral-100 dark:hover:bg-neutral-700 text-neutral-500"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Content */}
            <div className="p-4">
              <div className="flex items-start justify-between mb-3">
                <h3 className="font-semibold text-neutral-900 dark:text-white text-sm">
                  {currentTip.title}
                </h3>
                <div className="flex items-center space-x-1 ml-2">
                  <span className={cn("text-xs px-2 py-0.5 rounded-full font-medium", getTypeColor(currentTip.type))}>
                    {currentTip.type.replace('-', ' ')}
                  </span>
                </div>
              </div>

              <p className="text-sm text-neutral-600 dark:text-neutral-400 mb-4">
                {currentTip.description}
              </p>

              {/* Metadata */}
              <div className="flex items-center space-x-3 mb-4 text-xs text-neutral-500">
                {currentTip.estimatedTime && (
                  <div className="flex items-center space-x-1">
                    <Clock className="w-3 h-3" />
                    <span>{currentTip.estimatedTime}</span>
                  </div>
                )}
                {currentTip.difficulty && (
                  <span className={cn("px-2 py-0.5 rounded-full", getDifficultyColor(currentTip.difficulty))}>
                    {currentTip.difficulty}
                  </span>
                )}
              </div>

              {/* Actions */}
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  {currentTip.actionLabel && (
                    <button className="flex items-center space-x-1 px-3 py-1 bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300 rounded-md hover:bg-primary-200 dark:hover:bg-primary-900/50 transition-colors text-sm">
                      <span>{currentTip.actionLabel}</span>
                      <ExternalLink className="w-3 h-3" />
                    </button>
                  )}
                  
                  {!completedTips.has(currentTip.id) && (
                    <button
                      onClick={() => markAsCompleted(currentTip.id)}
                      className="flex items-center space-x-1 px-3 py-1 bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300 rounded-md hover:bg-green-200 dark:hover:bg-green-900/50 transition-colors text-sm"
                    >
                      <CheckCircle className="w-3 h-3" />
                      <span>Got it</span>
                    </button>
                  )}
                </div>

                {/* Navigation */}
                {hasMultipleTips && (
                  <div className="flex items-center space-x-2">
                    <span className="text-xs text-neutral-500">
                      {currentTipIndex + 1} of {tips.length}
                    </span>
                    <div className="flex space-x-1">
                      <button
                        onClick={prevTip}
                        disabled={currentTipIndex === 0}
                        className="p-1 rounded-md hover:bg-neutral-100 dark:hover:bg-neutral-700 disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        <ArrowRight className="w-3 h-3 rotate-180" />
                      </button>
                      <button
                        onClick={nextTip}
                        disabled={currentTipIndex === tips.length - 1}
                        className="p-1 rounded-md hover:bg-neutral-100 dark:hover:bg-neutral-700 disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        <ArrowRight className="w-3 h-3" />
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default ContextualHelp;