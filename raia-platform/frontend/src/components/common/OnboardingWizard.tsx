import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Database, Shield, Brain, BarChart3, CheckCircle, 
  ArrowRight, ArrowLeft, X, Zap, Upload, Cloud,
  Server, FileCheck, Users, Lock, Gauge, AlertCircle,
  Sparkles, Target, TrendingUp, Award
} from 'lucide-react';
import { cn } from '../../utils';

interface OnboardingStep {
  id: string;
  title: string;
  subtitle: string;
  icon: React.ComponentType<any>;
  description: string;
  actions?: {
    label: string;
    action: () => void;
    primary?: boolean;
  }[];
  checklist?: {
    label: string;
    completed: boolean;
  }[];
}

interface OnboardingWizardProps {
  isOpen: boolean;
  onClose: () => void;
  onComplete: (selectedPath: string) => void;
}

const OnboardingWizard: React.FC<OnboardingWizardProps> = ({
  isOpen,
  onClose,
  onComplete,
}) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [selectedPath, setSelectedPath] = useState<string>('');
  const [completedSteps, setCompletedSteps] = useState<Set<number>>(new Set());

  const onboardingSteps: OnboardingStep[] = [
    {
      id: 'welcome',
      title: 'Welcome to RAIA',
      subtitle: 'Responsible AI Analytics Platform',
      icon: Sparkles,
      description: 'Enterprise-grade AI governance, explainability, and monitoring in one unified platform.',
      actions: [
        {
          label: 'Start Setup',
          action: () => nextStep(),
          primary: true,
        },
        {
          label: 'Import Existing',
          action: () => setSelectedPath('import'),
        },
      ],
    },
    {
      id: 'data-connection',
      title: 'Connect Your Data',
      subtitle: 'Step 1 of 5',
      icon: Database,
      description: 'RAIA supports enterprise data sources including data lakes, warehouses, and real-time streams.',
      checklist: [
        { label: 'AWS S3 / Data Lake', completed: false },
        { label: 'Snowflake / BigQuery', completed: false },
        { label: 'PostgreSQL / MySQL', completed: false },
        { label: 'Kafka / Streaming', completed: false },
        { label: 'REST APIs', completed: false },
      ],
      actions: [
        {
          label: 'Configure Data Sources',
          action: () => nextStep(),
          primary: true,
        },
      ],
    },
    {
      id: 'model-import',
      title: 'Import Your Models',
      subtitle: 'Step 2 of 5',
      icon: Brain,
      description: 'Import ML models from popular frameworks and platforms.',
      checklist: [
        { label: 'TensorFlow / PyTorch', completed: false },
        { label: 'Scikit-learn / XGBoost', completed: false },
        { label: 'SageMaker / Vertex AI', completed: false },
        { label: 'MLflow / Kubeflow', completed: false },
        { label: 'Custom Models', completed: false },
      ],
      actions: [
        {
          label: 'Import Models',
          action: () => nextStep(),
          primary: true,
        },
      ],
    },
    {
      id: 'compliance-setup',
      title: 'Compliance & Governance',
      subtitle: 'Step 3 of 5',
      icon: Shield,
      description: 'Configure compliance frameworks and responsible AI policies.',
      checklist: [
        { label: 'GDPR Compliance', completed: true },
        { label: 'CCPA/CPRA', completed: true },
        { label: 'SOC 2 Type II', completed: true },
        { label: 'HIPAA (if applicable)', completed: false },
        { label: 'Custom Policies', completed: false },
      ],
      actions: [
        {
          label: 'Configure Compliance',
          action: () => nextStep(),
          primary: true,
        },
      ],
    },
    {
      id: 'monitoring',
      title: 'Set Up Monitoring',
      subtitle: 'Step 4 of 5',
      icon: Gauge,
      description: 'Configure real-time monitoring and alerting for your AI systems.',
      checklist: [
        { label: 'Performance Thresholds', completed: false },
        { label: 'Data Drift Detection', completed: false },
        { label: 'Bias Monitoring', completed: false },
        { label: 'Alert Channels', completed: false },
        { label: 'SLA Configuration', completed: false },
      ],
      actions: [
        {
          label: 'Configure Monitoring',
          action: () => nextStep(),
          primary: true,
        },
      ],
    },
    {
      id: 'team-setup',
      title: 'Invite Your Team',
      subtitle: 'Step 5 of 5',
      icon: Users,
      description: 'Set up your team with appropriate roles and permissions.',
      checklist: [
        { label: 'Admin Users', completed: true },
        { label: 'Data Scientists', completed: false },
        { label: 'Business Analysts', completed: false },
        { label: 'Compliance Officers', completed: false },
        { label: 'Executive Viewers', completed: false },
      ],
      actions: [
        {
          label: 'Invite Team',
          action: () => completeOnboarding(),
          primary: true,
        },
      ],
    },
  ];

  const nextStep = () => {
    setCompletedSteps(prev => new Set([...prev, currentStep]));
    setCurrentStep(prev => Math.min(prev + 1, onboardingSteps.length - 1));
  };

  const prevStep = () => {
    setCurrentStep(prev => Math.max(prev - 1, 0));
  };

  const completeOnboarding = () => {
    setCompletedSteps(prev => new Set([...prev, currentStep]));
    onComplete(selectedPath || 'standard');
  };

  const currentStepData = onboardingSteps[currentStep];
  const progress = ((currentStep + 1) / onboardingSteps.length) * 100;

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4"
      >
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.9, opacity: 0 }}
          transition={{ type: 'spring', stiffness: 300, damping: 30 }}
          className="bg-white dark:bg-neutral-900 rounded-2xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-hidden"
        >
          {/* Header */}
          <div className="relative p-6 pb-4 border-b border-neutral-200 dark:border-neutral-800">
            <button
              onClick={onClose}
              className="absolute top-6 right-6 p-2 rounded-lg hover:bg-neutral-100 dark:hover:bg-neutral-800 text-neutral-500"
            >
              <X className="w-5 h-5" />
            </button>

            {/* Progress Bar */}
            <div className="w-full bg-neutral-200 dark:bg-neutral-800 rounded-full h-2 mb-6">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${progress}%` }}
                transition={{ duration: 0.3 }}
                className="bg-gradient-to-r from-primary-500 to-accent-500 h-2 rounded-full"
              />
            </div>

            <div className="flex items-center space-x-4">
              <div className={cn(
                "w-16 h-16 rounded-2xl flex items-center justify-center",
                "bg-gradient-to-br from-primary-100 to-accent-100 dark:from-primary-900/30 dark:to-accent-900/30"
              )}>
                <currentStepData.icon className="w-8 h-8 text-primary-600 dark:text-primary-400" />
              </div>
              <div>
                <h2 className="text-2xl font-bold text-neutral-900 dark:text-white">
                  {currentStepData.title}
                </h2>
                <p className="text-neutral-600 dark:text-neutral-400">
                  {currentStepData.subtitle}
                </p>
              </div>
            </div>
          </div>

          {/* Content */}
          <div className="p-6 max-h-[50vh] overflow-y-auto">
            <p className="text-neutral-700 dark:text-neutral-300 mb-6">
              {currentStepData.description}
            </p>

            {/* Checklist */}
            {currentStepData.checklist && (
              <div className="space-y-3">
                {currentStepData.checklist.map((item, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.1 }}
                    className={cn(
                      "flex items-center space-x-3 p-3 rounded-lg border-2 transition-all",
                      item.completed
                        ? "bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800"
                        : "bg-neutral-50 dark:bg-neutral-800/50 border-neutral-200 dark:border-neutral-700"
                    )}
                  >
                    {item.completed ? (
                      <CheckCircle className="w-5 h-5 text-green-600 dark:text-green-400" />
                    ) : (
                      <div className="w-5 h-5 rounded-full border-2 border-neutral-300 dark:border-neutral-600" />
                    )}
                    <span className={cn(
                      "text-sm",
                      item.completed
                        ? "text-green-700 dark:text-green-300 font-medium"
                        : "text-neutral-700 dark:text-neutral-300"
                    )}>
                      {item.label}
                    </span>
                  </motion.div>
                ))}
              </div>
            )}

            {/* Special content for welcome step */}
            {currentStep === 0 && (
              <div className="grid grid-cols-3 gap-4 mt-6">
                <motion.div
                  whileHover={{ scale: 1.05 }}
                  className="p-4 bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-lg text-center"
                >
                  <Target className="w-8 h-8 text-blue-600 dark:text-blue-400 mx-auto mb-2" />
                  <p className="text-sm font-medium text-blue-700 dark:text-blue-300">80% Faster</p>
                  <p className="text-xs text-blue-600 dark:text-blue-400">Model Deployment</p>
                </motion.div>
                <motion.div
                  whileHover={{ scale: 1.05 }}
                  className="p-4 bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-lg text-center"
                >
                  <Shield className="w-8 h-8 text-green-600 dark:text-green-400 mx-auto mb-2" />
                  <p className="text-sm font-medium text-green-700 dark:text-green-300">100% Compliant</p>
                  <p className="text-xs text-green-600 dark:text-green-400">Out of the Box</p>
                </motion.div>
                <motion.div
                  whileHover={{ scale: 1.05 }}
                  className="p-4 bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-lg text-center"
                >
                  <Award className="w-8 h-8 text-purple-600 dark:text-purple-400 mx-auto mb-2" />
                  <p className="text-sm font-medium text-purple-700 dark:text-purple-300">Enterprise</p>
                  <p className="text-xs text-purple-600 dark:text-purple-400">Grade Security</p>
                </motion.div>
              </div>
            )}
          </div>

          {/* Footer */}
          <div className="p-6 pt-4 border-t border-neutral-200 dark:border-neutral-800">
            <div className="flex items-center justify-between">
              <button
                onClick={prevStep}
                disabled={currentStep === 0}
                className={cn(
                  "flex items-center space-x-2 px-4 py-2 rounded-lg transition-all",
                  currentStep === 0
                    ? "opacity-50 cursor-not-allowed"
                    : "hover:bg-neutral-100 dark:hover:bg-neutral-800 text-neutral-700 dark:text-neutral-300"
                )}
              >
                <ArrowLeft className="w-4 h-4" />
                <span>Back</span>
              </button>

              <div className="flex items-center space-x-3">
                {currentStepData.actions?.map((action, index) => (
                  <button
                    key={index}
                    onClick={action.action}
                    className={cn(
                      "flex items-center space-x-2 px-4 py-2 rounded-lg transition-all",
                      action.primary
                        ? "bg-primary-500 hover:bg-primary-600 text-white"
                        : "bg-neutral-100 dark:bg-neutral-800 hover:bg-neutral-200 dark:hover:bg-neutral-700 text-neutral-700 dark:text-neutral-300"
                    )}
                  >
                    <span>{action.label}</span>
                    {action.primary && <ArrowRight className="w-4 h-4" />}
                  </button>
                ))}
              </div>
            </div>

            {/* Step indicators */}
            <div className="flex items-center justify-center space-x-2 mt-4">
              {onboardingSteps.map((_, index) => (
                <div
                  key={index}
                  className={cn(
                    "w-2 h-2 rounded-full transition-all",
                    index === currentStep
                      ? "w-8 bg-primary-500"
                      : completedSteps.has(index)
                      ? "bg-green-500"
                      : "bg-neutral-300 dark:bg-neutral-600"
                  )}
                />
              ))}
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};

export default OnboardingWizard;