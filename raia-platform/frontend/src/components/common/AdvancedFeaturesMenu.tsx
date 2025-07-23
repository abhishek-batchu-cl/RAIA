import React from 'react';
import { motion } from 'framer-motion';
import {
  Search,
  Brain,
  Eye,
  GitBranch,
  Download,
  Database,
  BarChart3,
  Zap,
  Shield,
  Target,
  TrendingUp,
  FileText
} from 'lucide-react';
import Card from './Card';
import Button from './Button';

interface FeatureCard {
  key: string;
  title: string;
  description: string;
  icon: React.ComponentType<any>;
  status: 'active' | 'beta' | 'new';
  path: string;
  color: string;
  metrics?: {
    primary: string;
    secondary: string;
  };
}

const AdvancedFeaturesMenu: React.FC = () => {
  const features: FeatureCard[] = [
    {
      key: 'rag_evaluation',
      title: 'RAG Evaluation',
      description: 'Comprehensive evaluation of Retrieval-Augmented Generation systems with precision, recall, and generation quality metrics.',
      icon: Search,
      status: 'new',
      path: '/rag-evaluation',
      color: 'blue',
      metrics: {
        primary: '3 Systems',
        secondary: '89% Avg Score'
      }
    },
    {
      key: 'llm_evaluation',
      title: 'LLM Evaluation',
      description: 'Advanced Large Language Model evaluation with content quality, language quality, and safety analysis.',
      icon: Brain,
      status: 'new',
      path: '/llm-evaluation',
      color: 'purple',
      metrics: {
        primary: '5 Models',
        secondary: '91% Quality'
      }
    },
    {
      key: 'advanced_explainability',
      title: 'Advanced Explainability',
      description: 'Deep model insights using Anchor explanations, ALE plots, prototype analysis, and counterfactual generation.',
      icon: Eye,
      status: 'active',
      path: '/advanced-explainability',
      color: 'indigo',
      metrics: {
        primary: '6 Methods',
        secondary: '24s Avg Time'
      }
    },
    {
      key: 'what_if_analysis',
      title: 'What-If Analysis',
      description: 'Interactive scenario analysis with decision tree extraction, feature impact assessment, and optimization.',
      icon: GitBranch,
      status: 'active',
      path: '/what-if-analysis',
      color: 'green',
      metrics: {
        primary: '12 Sessions',
        secondary: '95% Accuracy'
      }
    },
    {
      key: 'data_export',
      title: 'Data Export',
      description: 'Professional report generation in PDF, Excel, and CSV formats with automated scheduling and delivery.',
      icon: Download,
      status: 'active',
      path: '/data-export',
      color: 'yellow',
      metrics: {
        primary: '24 Reports',
        secondary: '1.2GB Stored'
      }
    },
    {
      key: 'cache_management',
      title: 'Cache Management',
      description: 'High-performance Redis cache monitoring with hit rate analytics, memory optimization, and category management.',
      icon: Database,
      status: 'active',
      path: '/cache-management',
      color: 'red',
      metrics: {
        primary: '87% Hit Rate',
        secondary: '50MB Used'
      }
    },
    {
      key: 'enterprise_dashboard',
      title: 'Enterprise Dashboard',
      description: 'Executive-level insights with KPIs, model performance trends, compliance status, and business impact metrics.',
      icon: BarChart3,
      status: 'active',
      path: '/executive-dashboard',
      color: 'gray',
      metrics: {
        primary: '12 Models',
        secondary: '98% Uptime'
      }
    },
    {
      key: 'model_statistics',
      title: 'Model Statistics',
      description: 'Comprehensive classification and regression statistics with cross-validation, learning curves, and statistical tests.',
      icon: TrendingUp,
      status: 'active',
      path: '/model-statistics',
      color: 'cyan',
      metrics: {
        primary: '15 Metrics',
        secondary: '92% Accuracy'
      }
    },
    {
      key: 'model_drift',
      title: 'Model Drift Monitoring',
      description: 'Real-time model performance drift detection with impact analysis, root cause identification, and retraining recommendations.',
      icon: AlertTriangle,
      status: 'new',
      path: '/model-drift-monitoring',
      color: 'orange',
      metrics: {
        primary: '2 Alerts',
        secondary: '87% Confidence'
      }
    }
  ];

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'new':
        return (
          <span className="px-2 py-1 bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300 text-xs font-medium rounded-full">
            New
          </span>
        );
      case 'beta':
        return (
          <span className="px-2 py-1 bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300 text-xs font-medium rounded-full">
            Beta
          </span>
        );
      default:
        return null;
    }
  };

  const getColorClasses = (color: string, type: 'icon' | 'bg' | 'border') => {
    const colorMap = {
      blue: {
        icon: 'text-blue-600 dark:text-blue-400',
        bg: 'bg-blue-50 dark:bg-blue-900/20',
        border: 'border-blue-200 dark:border-blue-700'
      },
      purple: {
        icon: 'text-purple-600 dark:text-purple-400',
        bg: 'bg-purple-50 dark:bg-purple-900/20',
        border: 'border-purple-200 dark:border-purple-700'
      },
      indigo: {
        icon: 'text-indigo-600 dark:text-indigo-400',
        bg: 'bg-indigo-50 dark:bg-indigo-900/20',
        border: 'border-indigo-200 dark:border-indigo-700'
      },
      green: {
        icon: 'text-green-600 dark:text-green-400',
        bg: 'bg-green-50 dark:bg-green-900/20',
        border: 'border-green-200 dark:border-green-700'
      },
      yellow: {
        icon: 'text-yellow-600 dark:text-yellow-400',
        bg: 'bg-yellow-50 dark:bg-yellow-900/20',
        border: 'border-yellow-200 dark:border-yellow-700'
      },
      red: {
        icon: 'text-red-600 dark:text-red-400',
        bg: 'bg-red-50 dark:bg-red-900/20',
        border: 'border-red-200 dark:border-red-700'
      },
      gray: {
        icon: 'text-gray-600 dark:text-gray-400',
        bg: 'bg-gray-50 dark:bg-gray-900/20',
        border: 'border-gray-200 dark:border-gray-700'
      },
      cyan: {
        icon: 'text-cyan-600 dark:text-cyan-400',
        bg: 'bg-cyan-50 dark:bg-cyan-900/20',
        border: 'border-cyan-200 dark:border-cyan-700'
      },
      orange: {
        icon: 'text-orange-600 dark:text-orange-400',
        bg: 'bg-orange-50 dark:bg-orange-900/20',
        border: 'border-orange-200 dark:border-orange-700'
      }
    };
    return colorMap[color as keyof typeof colorMap]?.[type] || colorMap.gray[type];
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-neutral-900 dark:text-neutral-100 mb-2">
          Advanced AI Features
        </h2>
        <p className="text-neutral-600 dark:text-neutral-400">
          Comprehensive evaluation, explainability, and enterprise management tools
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
        {features.map((feature, index) => {
          const Icon = feature.icon;
          return (
            <motion.div
              key={feature.key}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className="group"
            >
              <Card className={`h-full border-2 transition-all duration-300 hover:shadow-lg hover:scale-105 ${getColorClasses(feature.color, 'border')} hover:${getColorClasses(feature.color, 'bg')}`}>
                <div className="p-6">
                  <div className="flex items-start justify-between mb-4">
                    <div className={`p-3 rounded-xl ${getColorClasses(feature.color, 'bg')}`}>
                      <Icon className={`w-6 h-6 ${getColorClasses(feature.color, 'icon')}`} />
                    </div>
                    {getStatusBadge(feature.status)}
                  </div>

                  <div className="mb-4">
                    <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-2 group-hover:text-neutral-800 dark:group-hover:text-neutral-200">
                      {feature.title}
                    </h3>
                    <p className="text-sm text-neutral-600 dark:text-neutral-400 leading-relaxed">
                      {feature.description}
                    </p>
                  </div>

                  {feature.metrics && (
                    <div className="mb-4 p-3 bg-neutral-50 dark:bg-neutral-800/50 rounded-lg">
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-neutral-600 dark:text-neutral-400">Status:</span>
                        <span className="font-medium text-neutral-900 dark:text-neutral-100">
                          {feature.metrics.primary}
                        </span>
                      </div>
                      <div className="flex items-center justify-between text-sm mt-1">
                        <span className="text-neutral-600 dark:text-neutral-400">Performance:</span>
                        <span className={`font-medium ${getColorClasses(feature.color, 'icon')}`}>
                          {feature.metrics.secondary}
                        </span>
                      </div>
                    </div>
                  )}

                  <Button
                    variant="outline"
                    className="w-full group-hover:bg-neutral-900 group-hover:text-white dark:group-hover:bg-neutral-100 dark:group-hover:text-neutral-900"
                    onClick={() => {
                      // In a real app, this would use React Router
                      console.log(`Navigate to ${feature.path}`);
                    }}
                  >
                    Explore Feature
                  </Button>
                </div>
              </Card>
            </motion.div>
          );
        })}
      </div>

      {/* Quick Stats Section */}
      <Card>
        <div className="p-6">
          <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4">
            Platform Overview
          </h3>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">9</div>
              <div className="text-sm text-blue-700 dark:text-blue-300">Advanced Features</div>
            </div>
            <div className="text-center p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
              <div className="text-2xl font-bold text-green-600 dark:text-green-400">20</div>
              <div className="text-sm text-green-700 dark:text-green-300">Active Models</div>
            </div>
            <div className="text-center p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
              <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">1.2M</div>
              <div className="text-sm text-purple-700 dark:text-purple-300">Predictions/Day</div>
            </div>
            <div className="text-center p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
              <div className="text-2xl font-bold text-yellow-600 dark:text-yellow-400">99.7%</div>
              <div className="text-sm text-yellow-700 dark:text-yellow-300">Uptime</div>
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default AdvancedFeaturesMenu;