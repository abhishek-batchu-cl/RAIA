import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Shield, AlertTriangle, CheckCircle, X, Eye, FileText,
  Scale, BookOpen, Users, Lock, Unlock, Globe, 
  Gavel, AlertCircle, TrendingUp, TrendingDown, Clock,
  Download, Upload, Share2, Settings, Filter, Search,
  Brain, Target, Activity, BarChart3, PieChart, LineChart,
  Bell, Calendar, User, Tag, ArrowRight, ChevronDown,
  ExternalLink, RefreshCw, Database, Code, Zap
} from 'lucide-react';
import Card from '@/components/common/Card';
import Button from '@/components/common/Button';
import { apiClient } from '@/services/api';

interface ComplianceFramework {
  id: string;
  name: string;
  description: string;
  category: 'data_protection' | 'ai_ethics' | 'industry_specific' | 'internal';
  requirements: ComplianceRequirement[];
  overall_score: number;
  status: 'compliant' | 'partial' | 'non_compliant';
  last_assessed: Date;
  next_review: Date;
  mandatory: boolean;
}

interface ComplianceRequirement {
  id: string;
  title: string;
  description: string;
  category: string;
  priority: 'critical' | 'high' | 'medium' | 'low';
  status: 'compliant' | 'partial' | 'non_compliant' | 'not_applicable';
  evidence: Evidence[];
  controls: Control[];
  last_assessed: Date;
  assessor: string;
  remediation_plan?: RemediationPlan;
}

interface Evidence {
  id: string;
  type: 'documentation' | 'technical_control' | 'audit_log' | 'test_result';
  title: string;
  description: string;
  file_path?: string;
  created_at: Date;
  valid_until?: Date;
  status: 'valid' | 'expired' | 'under_review';
}

interface Control {
  id: string;
  name: string;
  type: 'preventive' | 'detective' | 'corrective';
  implementation_status: 'implemented' | 'planned' | 'not_implemented';
  effectiveness: number;
  automated: boolean;
  last_tested: Date;
  responsible_party: string;
}

interface RemediationPlan {
  id: string;
  issue: string;
  priority: 'critical' | 'high' | 'medium' | 'low';
  steps: RemediationStep[];
  assigned_to: string;
  due_date: Date;
  status: 'open' | 'in_progress' | 'completed' | 'overdue';
  estimated_effort: string;
}

interface RemediationStep {
  id: string;
  description: string;
  status: 'pending' | 'in_progress' | 'completed';
  assigned_to: string;
  due_date: Date;
  dependencies: string[];
}

interface RiskAssessment {
  id: string;
  model_id: string;
  model_name: string;
  risk_category: 'bias' | 'fairness' | 'privacy' | 'security' | 'explainability' | 'robustness';
  risk_level: 'low' | 'medium' | 'high' | 'critical';
  risk_score: number;
  description: string;
  impact: string;
  likelihood: string;
  mitigation_measures: string[];
  status: 'open' | 'mitigated' | 'accepted' | 'transferred';
  identified_by: string;
  identified_at: Date;
  last_reviewed: Date;
  next_review: Date;
}

interface AuditTrail {
  id: string;
  timestamp: Date;
  user: string;
  action: string;
  resource: string;
  resource_id: string;
  details: any;
  ip_address: string;
  user_agent: string;
  compliance_relevant: boolean;
}

interface GovernanceMetrics {
  overall_compliance_score: number;
  compliant_frameworks: number;
  total_frameworks: number;
  critical_issues: number;
  high_risk_models: number;
  overdue_reviews: number;
  automation_coverage: number;
  recent_audits: number;
}

const AIGovernanceDashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'overview' | 'frameworks' | 'risks' | 'audit'>('overview');
  const [frameworks, setFrameworks] = useState<ComplianceFramework[]>([]);
  const [riskAssessments, setRiskAssessments] = useState<RiskAssessment[]>([]);
  const [auditTrail, setAuditTrail] = useState<AuditTrail[]>([]);
  const [metrics, setMetrics] = useState<GovernanceMetrics | null>(null);
  const [selectedFramework, setSelectedFramework] = useState<ComplianceFramework | null>(null);
  const [showComplianceModal, setShowComplianceModal] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [filterStatus, setFilterStatus] = useState<string>('all');
  const [timeRange, setTimeRange] = useState<string>('30d');

  // Mock data
  const mockMetrics: GovernanceMetrics = {
    overall_compliance_score: 87.3,
    compliant_frameworks: 12,
    total_frameworks: 15,
    critical_issues: 3,
    high_risk_models: 7,
    overdue_reviews: 5,
    automation_coverage: 78.5,
    recent_audits: 24
  };

  const mockFrameworks: ComplianceFramework[] = [
    {
      id: 'gdpr',
      name: 'GDPR - General Data Protection Regulation',
      description: 'EU data protection and privacy regulation compliance',
      category: 'data_protection',
      overall_score: 92.5,
      status: 'compliant',
      last_assessed: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
      next_review: new Date(Date.now() + 83 * 24 * 60 * 60 * 1000),
      mandatory: true,
      requirements: [
        {
          id: 'gdpr-1',
          title: 'Data Processing Lawfulness',
          description: 'Ensure all personal data processing has a lawful basis',
          category: 'Data Processing',
          priority: 'critical',
          status: 'compliant',
          evidence: [],
          controls: [],
          last_assessed: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
          assessor: 'compliance@company.com'
        },
        {
          id: 'gdpr-2',
          title: 'Right to Explanation',
          description: 'Provide meaningful information about automated decision-making',
          category: 'Automated Decision Making',
          priority: 'high',
          status: 'partial',
          evidence: [],
          controls: [],
          last_assessed: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
          assessor: 'compliance@company.com',
          remediation_plan: {
            id: 'rem-1',
            issue: 'Insufficient explainability documentation',
            priority: 'high',
            steps: [],
            assigned_to: 'ai-team@company.com',
            due_date: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000),
            status: 'in_progress',
            estimated_effort: '2 weeks'
          }
        }
      ]
    },
    {
      id: 'ccpa',
      name: 'CCPA - California Consumer Privacy Act',
      description: 'California state privacy law compliance',
      category: 'data_protection',
      overall_score: 85.2,
      status: 'partial',
      last_assessed: new Date(Date.now() - 14 * 24 * 60 * 60 * 1000),
      next_review: new Date(Date.now() + 76 * 24 * 60 * 60 * 1000),
      mandatory: true,
      requirements: []
    },
    {
      id: 'ai-ethics',
      name: 'AI Ethics Framework',
      description: 'Internal AI ethics and responsible AI guidelines',
      category: 'ai_ethics',
      overall_score: 78.9,
      status: 'partial',
      last_assessed: new Date(Date.now() - 21 * 24 * 60 * 60 * 1000),
      next_review: new Date(Date.now() + 69 * 24 * 60 * 60 * 1000),
      mandatory: false,
      requirements: []
    },
    {
      id: 'iso-27001',
      name: 'ISO 27001 - Information Security',
      description: 'International standard for information security management',
      category: 'industry_specific',
      overall_score: 94.1,
      status: 'compliant',
      last_assessed: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000),
      next_review: new Date(Date.now() + 60 * 24 * 60 * 60 * 1000),
      mandatory: true,
      requirements: []
    }
  ];

  const mockRisks: RiskAssessment[] = [
    {
      id: 'risk-1',
      model_id: 'model-001',
      model_name: 'Credit Risk Assessment Model',
      risk_category: 'bias',
      risk_level: 'high',
      risk_score: 8.2,
      description: 'Potential gender bias detected in loan approval decisions',
      impact: 'Discriminatory outcomes affecting loan approvals',
      likelihood: 'High - Statistical evidence of bias patterns',
      mitigation_measures: [
        'Implement fairness constraints in model training',
        'Add bias detection monitoring',
        'Regular fairness audits'
      ],
      status: 'open',
      identified_by: 'ai-audit@company.com',
      identified_at: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000),
      last_reviewed: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000),
      next_review: new Date(Date.now() + 28 * 24 * 60 * 60 * 1000)
    },
    {
      id: 'risk-2',
      model_id: 'model-002',
      model_name: 'Customer Segmentation Model',
      risk_category: 'privacy',
      risk_level: 'medium',
      risk_score: 5.7,
      description: 'Model inference could potentially reveal sensitive customer information',
      impact: 'Privacy breach risk through model inversion attacks',
      likelihood: 'Medium - Requires technical expertise',
      mitigation_measures: [
        'Implement differential privacy',
        'Add noise to model outputs',
        'Access controls and monitoring'
      ],
      status: 'mitigated',
      identified_by: 'security@company.com',
      identified_at: new Date(Date.now() - 15 * 24 * 60 * 60 * 1000),
      last_reviewed: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000),
      next_review: new Date(Date.now() + 25 * 24 * 60 * 60 * 1000)
    },
    {
      id: 'risk-3',
      model_id: 'model-003',
      model_name: 'Fraud Detection Model',
      risk_category: 'robustness',
      risk_level: 'critical',
      risk_score: 9.1,
      description: 'Model vulnerable to adversarial attacks that could bypass fraud detection',
      impact: 'Critical - Could result in significant financial losses',
      likelihood: 'High - Known attack vectors exist',
      mitigation_measures: [
        'Implement adversarial training',
        'Deploy ensemble methods',
        'Real-time attack detection'
      ],
      status: 'in_progress',
      identified_by: 'security@company.com',
      identified_at: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000),
      last_reviewed: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000),
      next_review: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000)
    }
  ];

  const mockAuditTrail: AuditTrail[] = [
    {
      id: 'audit-1',
      timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000),
      user: 'sarah.chen@company.com',
      action: 'MODEL_DEPLOYED',
      resource: 'model',
      resource_id: 'model-001',
      details: { version: 'v2.1.0', endpoint: '/api/credit-risk' },
      ip_address: '192.168.1.100',
      user_agent: 'Mozilla/5.0...',
      compliance_relevant: true
    },
    {
      id: 'audit-2',
      timestamp: new Date(Date.now() - 4 * 60 * 60 * 1000),
      user: 'compliance@company.com',
      action: 'RISK_ASSESSMENT_CREATED',
      resource: 'risk_assessment',
      resource_id: 'risk-3',
      details: { risk_level: 'critical', category: 'robustness' },
      ip_address: '192.168.1.101',
      user_agent: 'Mozilla/5.0...',
      compliance_relevant: true
    },
    {
      id: 'audit-3',
      timestamp: new Date(Date.now() - 6 * 60 * 60 * 1000),
      user: 'michael.torres@company.com',
      action: 'DATA_ACCESS',
      resource: 'dataset',
      resource_id: 'ds-customer-data',
      details: { records_accessed: 50000, purpose: 'model_training' },
      ip_address: '192.168.1.102',
      user_agent: 'Python/requests',
      compliance_relevant: true
    }
  ];

  useEffect(() => {
    loadGovernanceData();
  }, [timeRange]);

  const loadGovernanceData = async () => {
    try {
      // In production, this would call the API
      setFrameworks(mockFrameworks);
      setRiskAssessments(mockRisks);
      setAuditTrail(mockAuditTrail);
      setMetrics(mockMetrics);
    } catch (error) {
      console.error('Error loading governance data:', error);
      setFrameworks(mockFrameworks);
      setRiskAssessments(mockRisks);
      setAuditTrail(mockAuditTrail);
      setMetrics(mockMetrics);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'compliant':
      case 'implemented':
      case 'valid':
      case 'mitigated':
      case 'completed':
        return 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400';
      case 'partial':
      case 'planned':
      case 'in_progress':
      case 'under_review':
        return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400';
      case 'non_compliant':
      case 'not_implemented':
      case 'expired':
      case 'open':
      case 'overdue':
        return 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400';
      default:
        return 'bg-gray-100 text-gray-800 dark:bg-gray-900/20 dark:text-gray-400';
    }
  };

  const getRiskLevelColor = (level: string) => {
    switch (level) {
      case 'critical':
        return 'bg-red-500 text-white';
      case 'high':
        return 'bg-orange-500 text-white';
      case 'medium':
        return 'bg-yellow-500 text-white';
      case 'low':
        return 'bg-green-500 text-white';
      default:
        return 'bg-gray-500 text-white';
    }
  };

  const formatTimeAgo = (date: Date) => {
    const now = new Date();
    const diffInHours = Math.floor((now.getTime() - date.getTime()) / (1000 * 60 * 60));
    
    if (diffInHours < 1) return 'Just now';
    if (diffInHours < 24) return `${diffInHours}h ago`;
    if (diffInHours < 168) return `${Math.floor(diffInHours / 24)}d ago`;
    return `${Math.floor(diffInHours / 168)}w ago`;
  };

  const calculateComplianceScore = (framework: ComplianceFramework) => {
    const compliantReqs = framework.requirements.filter(r => r.status === 'compliant').length;
    const totalReqs = framework.requirements.length;
    return totalReqs > 0 ? (compliantReqs / totalReqs) * 100 : 100;
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-neutral-900 dark:text-neutral-100 flex items-center">
            <div className="w-8 h-8 bg-gradient-to-r from-green-500 to-blue-500 rounded-lg flex items-center justify-center mr-3">
              <Shield className="w-5 h-5 text-white" />
            </div>
            AI Governance Dashboard
          </h1>
          <p className="text-neutral-600 dark:text-neutral-400 mt-1">
            Comprehensive compliance tracking, risk management, and regulatory oversight
          </p>
        </div>
        
        <div className="flex items-center space-x-2">
          <div className="flex items-center space-x-1 bg-neutral-100 dark:bg-neutral-800 rounded-lg p-1">
            {['7d', '30d', '90d', '1y'].map((range) => (
              <button
                key={range}
                onClick={() => setTimeRange(range)}
                className={`px-3 py-1 rounded-md text-sm font-medium transition-colors ${
                  timeRange === range
                    ? 'bg-white dark:bg-neutral-700 text-primary-600 dark:text-primary-400 shadow-sm'
                    : 'text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-neutral-100'
                }`}
              >
                {range}
              </button>
            ))}
          </div>
          <Button variant="outline" size="sm" leftIcon={<Download className="w-4 h-4" />}>
            Export Report
          </Button>
          <Button variant="outline" size="sm" leftIcon={<Settings className="w-4 h-4" />}>
            Settings
          </Button>
        </div>
      </div>

      {/* Metrics Overview */}
      {metrics && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
            <Card className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 border-green-200 dark:border-green-800">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-green-700 dark:text-green-300">Overall Compliance</p>
                  <p className="text-3xl font-bold text-green-900 dark:text-green-100">
                    {metrics.overall_compliance_score}%
                  </p>
                  <p className="text-xs text-green-600 dark:text-green-400">
                    {metrics.compliant_frameworks}/{metrics.total_frameworks} frameworks
                  </p>
                </div>
                <div className="p-3 bg-green-100 dark:bg-green-900/30 rounded-full">
                  <CheckCircle className="w-8 h-8 text-green-600 dark:text-green-400" />
                </div>
              </div>
            </Card>
          </motion.div>

          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
            <Card className="bg-gradient-to-r from-red-50 to-pink-50 dark:from-red-900/20 dark:to-pink-900/20 border-red-200 dark:border-red-800">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-red-700 dark:text-red-300">Critical Issues</p>
                  <p className="text-3xl font-bold text-red-900 dark:text-red-100">
                    {metrics.critical_issues}
                  </p>
                  <p className="text-xs text-red-600 dark:text-red-400">
                    {metrics.high_risk_models} high-risk models
                  </p>
                </div>
                <div className="p-3 bg-red-100 dark:bg-red-900/30 rounded-full">
                  <AlertTriangle className="w-8 h-8 text-red-600 dark:text-red-400" />
                </div>
              </div>
            </Card>
          </motion.div>

          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}>
            <Card className="bg-gradient-to-r from-orange-50 to-yellow-50 dark:from-orange-900/20 dark:to-yellow-900/20 border-orange-200 dark:border-orange-800">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-orange-700 dark:text-orange-300">Overdue Reviews</p>
                  <p className="text-3xl font-bold text-orange-900 dark:text-orange-100">
                    {metrics.overdue_reviews}
                  </p>
                  <p className="text-xs text-orange-600 dark:text-orange-400">
                    Require immediate attention
                  </p>
                </div>
                <div className="p-3 bg-orange-100 dark:bg-orange-900/30 rounded-full">
                  <Clock className="w-8 h-8 text-orange-600 dark:text-orange-400" />
                </div>
              </div>
            </Card>
          </motion.div>

          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }}>
            <Card className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 border-blue-200 dark:border-blue-800">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-blue-700 dark:text-blue-300">Automation Coverage</p>
                  <p className="text-3xl font-bold text-blue-900 dark:text-blue-100">
                    {metrics.automation_coverage}%
                  </p>
                  <p className="text-xs text-blue-600 dark:text-blue-400">
                    {metrics.recent_audits} recent audits
                  </p>
                </div>
                <div className="p-3 bg-blue-100 dark:bg-blue-900/30 rounded-full">
                  <Zap className="w-8 h-8 text-blue-600 dark:text-blue-400" />
                </div>
              </div>
            </Card>
          </motion.div>
        </div>
      )}

      {/* Tabs */}
      <div className="flex space-x-1 bg-neutral-100 dark:bg-neutral-800 rounded-lg p-1">
        {[
          { id: 'overview', label: 'Overview', icon: <BarChart3 className="w-4 h-4" /> },
          { id: 'frameworks', label: 'Compliance Frameworks', icon: <Scale className="w-4 h-4" /> },
          { id: 'risks', label: 'Risk Assessment', icon: <AlertTriangle className="w-4 h-4" /> },
          { id: 'audit', label: 'Audit Trail', icon: <FileText className="w-4 h-4" /> }
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as any)}
            className={`flex-1 flex items-center justify-center space-x-2 px-4 py-2 rounded-md text-sm font-medium transition-all duration-200 ${
              activeTab === tab.id
                ? 'bg-white dark:bg-neutral-700 text-primary-600 dark:text-primary-400 shadow-sm'
                : 'text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-neutral-100'
            }`}
          >
            {tab.icon}
            <span className="hidden sm:inline">{tab.label}</span>
          </button>
        ))}
      </div>

      {/* Content */}
      <AnimatePresence mode="wait">
        {activeTab === 'overview' && (
          <motion.div
            key="overview"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="grid grid-cols-1 lg:grid-cols-2 gap-6"
          >
            {/* Compliance Status */}
            <Card title="Compliance Framework Status" icon={<Scale className="w-5 h-5 text-primary-500" />}>
              <div className="space-y-4">
                {frameworks.slice(0, 4).map((framework) => (
                  <div key={framework.id} className="flex items-center justify-between p-3 bg-neutral-50 dark:bg-neutral-800 rounded-lg">
                    <div className="flex-1">
                      <div className="flex items-center justify-between mb-2">
                        <h4 className="font-medium text-neutral-900 dark:text-neutral-100 text-sm">
                          {framework.name}
                        </h4>
                        <span className={`px-2 py-1 text-xs rounded-full ${getStatusColor(framework.status)}`}>
                          {framework.status.replace('_', ' ')}
                        </span>
                      </div>
                      <div className="w-full bg-neutral-200 dark:bg-neutral-700 rounded-full h-2 mb-2">
                        <motion.div
                          className="bg-gradient-to-r from-green-500 to-emerald-500 h-2 rounded-full"
                          initial={{ width: 0 }}
                          animate={{ width: `${framework.overall_score}%` }}
                          transition={{ duration: 1, delay: 0.2 }}
                        />
                      </div>
                      <div className="flex items-center justify-between text-xs text-neutral-500 dark:text-neutral-400">
                        <span>{framework.overall_score}% compliant</span>
                        <span>Review due: {new Date(framework.next_review).toLocaleDateString()}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </Card>

            {/* Risk Heat Map */}
            <Card title="Risk Assessment Overview" icon={<AlertTriangle className="w-5 h-5 text-primary-500" />}>
              <div className="space-y-4">
                {/* Risk Categories */}
                <div className="grid grid-cols-2 gap-3">
                  {[
                    { category: 'Bias & Fairness', count: 3, level: 'high' },
                    { category: 'Privacy', count: 2, level: 'medium' },
                    { category: 'Security', count: 1, level: 'critical' },
                    { category: 'Explainability', count: 4, level: 'low' }
                  ].map((item, index) => (
                    <motion.div
                      key={item.category}
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ delay: index * 0.1 }}
                      className={`p-3 rounded-lg text-center ${
                        item.level === 'critical' ? 'bg-red-100 dark:bg-red-900/20' :
                        item.level === 'high' ? 'bg-orange-100 dark:bg-orange-900/20' :
                        item.level === 'medium' ? 'bg-yellow-100 dark:bg-yellow-900/20' :
                        'bg-green-100 dark:bg-green-900/20'
                      }`}
                    >
                      <div className={`text-2xl font-bold ${
                        item.level === 'critical' ? 'text-red-700 dark:text-red-300' :
                        item.level === 'high' ? 'text-orange-700 dark:text-orange-300' :
                        item.level === 'medium' ? 'text-yellow-700 dark:text-yellow-300' :
                        'text-green-700 dark:text-green-300'
                      }`}>
                        {item.count}
                      </div>
                      <div className="text-xs text-neutral-600 dark:text-neutral-400">
                        {item.category}
                      </div>
                    </motion.div>
                  ))}
                </div>

                {/* Recent High-Risk Items */}
                <div className="border-t border-neutral-200 dark:border-neutral-700 pt-4">
                  <h5 className="font-medium text-neutral-900 dark:text-neutral-100 mb-3">Recent High-Risk Items</h5>
                  <div className="space-y-2">
                    {riskAssessments.filter(r => r.risk_level === 'high' || r.risk_level === 'critical').slice(0, 3).map((risk) => (
                      <div key={risk.id} className="flex items-center justify-between text-sm">
                        <div className="flex items-center space-x-2">
                          <div className={`w-2 h-2 rounded-full ${getRiskLevelColor(risk.risk_level).replace('text-white', '')}`} />
                          <span className="text-neutral-900 dark:text-neutral-100">{risk.model_name}</span>
                        </div>
                        <span className="text-neutral-500 dark:text-neutral-400">{formatTimeAgo(risk.identified_at)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </Card>

            {/* Recent Activity */}
            <Card title="Recent Governance Activity" icon={<Activity className="w-5 h-5 text-primary-500" />}>
              <div className="space-y-3">
                {auditTrail.slice(0, 5).map((entry) => (
                  <div key={entry.id} className="flex items-start space-x-3 p-3 bg-neutral-50 dark:bg-neutral-800 rounded-lg">
                    <div className="p-1.5 bg-primary-100 dark:bg-primary-900/20 rounded-lg">
                      {entry.action.includes('MODEL') ? <Brain className="w-4 h-4 text-primary-600 dark:text-primary-400" /> :
                       entry.action.includes('RISK') ? <AlertTriangle className="w-4 h-4 text-orange-600 dark:text-orange-400" /> :
                       <FileText className="w-4 h-4 text-blue-600 dark:text-blue-400" />}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-neutral-900 dark:text-neutral-100">
                        {entry.action.replace('_', ' ').toLowerCase()}
                      </p>
                      <p className="text-xs text-neutral-600 dark:text-neutral-400">
                        {entry.user} • {formatTimeAgo(entry.timestamp)}
                      </p>
                      <p className="text-xs text-neutral-500 dark:text-neutral-400 mt-1">
                        {entry.resource}: {entry.resource_id}
                      </p>
                    </div>
                    {entry.compliance_relevant && (
                      <div className="p-1 bg-green-100 dark:bg-green-900/20 rounded">
                        <Shield className="w-3 h-3 text-green-600 dark:text-green-400" />
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </Card>

            {/* Automation Status */}
            <Card title="Governance Automation" icon={<Zap className="w-5 h-5 text-primary-500" />}>
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="text-center p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                    <div className="text-2xl font-bold text-blue-700 dark:text-blue-300">
                      {metrics?.automation_coverage}%
                    </div>
                    <div className="text-xs text-blue-600 dark:text-blue-400">
                      Automated Controls
                    </div>
                  </div>
                  <div className="text-center p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                    <div className="text-2xl font-bold text-green-700 dark:text-green-300">
                      24/7
                    </div>
                    <div className="text-xs text-green-600 dark:text-green-400">
                      Continuous Monitoring
                    </div>
                  </div>
                </div>
                
                <div className="space-y-3">
                  <h5 className="font-medium text-neutral-900 dark:text-neutral-100">Active Automated Controls</h5>
                  {[
                    'Bias Detection Monitoring',
                    'Data Access Logging',
                    'Model Performance Tracking',
                    'Compliance Report Generation'
                  ].map((control, index) => (
                    <div key={control} className="flex items-center justify-between text-sm">
                      <div className="flex items-center space-x-2">
                        <CheckCircle className="w-4 h-4 text-green-500" />
                        <span className="text-neutral-900 dark:text-neutral-100">{control}</span>
                      </div>
                      <span className="text-green-600 dark:text-green-400">Active</span>
                    </div>
                  ))}
                </div>
              </div>
            </Card>
          </motion.div>
        )}

        {activeTab === 'frameworks' && (
          <motion.div
            key="frameworks"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
          >
            <Card>
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center space-x-4">
                  <div className="flex items-center space-x-2">
                    <Search className="w-4 h-4 text-neutral-400" />
                    <input
                      type="text"
                      placeholder="Search frameworks..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="border-0 bg-transparent text-neutral-900 dark:text-neutral-100 placeholder-neutral-400 focus:outline-none"
                    />
                  </div>
                  
                  <div className="flex items-center space-x-1 bg-neutral-100 dark:bg-neutral-800 rounded-lg p-1">
                    {['all', 'compliant', 'partial', 'non_compliant'].map((status) => (
                      <button
                        key={status}
                        onClick={() => setFilterStatus(status)}
                        className={`px-3 py-1 rounded-md text-sm font-medium capitalize transition-colors ${
                          filterStatus === status
                            ? 'bg-white dark:bg-neutral-700 text-primary-600 dark:text-primary-400 shadow-sm'
                            : 'text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-neutral-100'
                        }`}
                      >
                        {status.replace('_', ' ')}
                      </button>
                    ))}
                  </div>
                </div>
                
                <Button
                  variant="primary"
                  size="sm"
                  leftIcon={<Plus className="w-4 h-4" />}
                  onClick={() => setShowComplianceModal(true)}
                >
                  Add Framework
                </Button>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {frameworks.map((framework, index) => (
                  <motion.div
                    key={framework.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.1 }}
                    onClick={() => setSelectedFramework(framework)}
                    className="cursor-pointer"
                  >
                    <Card className="hover:shadow-lg transition-all duration-200">
                      <div className="space-y-4">
                        {/* Framework Header */}
                        <div className="flex items-start justify-between">
                          <div className="flex-1">
                            <div className="flex items-center space-x-2 mb-2">
                              <h3 className="font-semibold text-neutral-900 dark:text-neutral-100">
                                {framework.name}
                              </h3>
                              {framework.mandatory && (
                                <span className="px-2 py-1 text-xs bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400 rounded-full">
                                  Mandatory
                                </span>
                              )}
                            </div>
                            <p className="text-sm text-neutral-600 dark:text-neutral-400 mb-3">
                              {framework.description}
                            </p>
                          </div>
                          
                          <span className={`px-3 py-1 text-sm rounded-full ${getStatusColor(framework.status)}`}>
                            {framework.status.replace('_', ' ')}
                          </span>
                        </div>
                        
                        {/* Compliance Score */}
                        <div>
                          <div className="flex items-center justify-between text-sm mb-1">
                            <span className="font-medium text-neutral-700 dark:text-neutral-300">Compliance Score</span>
                            <span className="font-bold text-neutral-900 dark:text-neutral-100">{framework.overall_score}%</span>
                          </div>
                          <div className="w-full bg-neutral-200 dark:bg-neutral-700 rounded-full h-3">
                            <motion.div
                              className={`h-3 rounded-full ${
                                framework.overall_score >= 90 ? 'bg-green-500' :
                                framework.overall_score >= 70 ? 'bg-yellow-500' : 'bg-red-500'
                              }`}
                              initial={{ width: 0 }}
                              animate={{ width: `${framework.overall_score}%` }}
                              transition={{ duration: 1, delay: index * 0.2 }}
                            />
                          </div>
                        </div>
                        
                        {/* Framework Details */}
                        <div className="grid grid-cols-2 gap-4 text-sm">
                          <div>
                            <span className="text-neutral-500 dark:text-neutral-400">Category:</span>
                            <span className="ml-2 capitalize text-neutral-900 dark:text-neutral-100">
                              {framework.category.replace('_', ' ')}
                            </span>
                          </div>
                          <div>
                            <span className="text-neutral-500 dark:text-neutral-400">Requirements:</span>
                            <span className="ml-2 text-neutral-900 dark:text-neutral-100">
                              {framework.requirements.length}
                            </span>
                          </div>
                          <div>
                            <span className="text-neutral-500 dark:text-neutral-400">Last Assessed:</span>
                            <span className="ml-2 text-neutral-900 dark:text-neutral-100">
                              {formatTimeAgo(framework.last_assessed)}
                            </span>
                          </div>
                          <div>
                            <span className="text-neutral-500 dark:text-neutral-400">Next Review:</span>
                            <span className="ml-2 text-neutral-900 dark:text-neutral-100">
                              {new Date(framework.next_review).toLocaleDateString()}
                            </span>
                          </div>
                        </div>
                        
                        {/* Actions */}
                        <div className="flex items-center space-x-2 pt-3 border-t border-neutral-200 dark:border-neutral-700">
                          <Button variant="outline" size="sm" leftIcon={<Eye className="w-4 h-4" />}>
                            View Details
                          </Button>
                          <Button variant="outline" size="sm" leftIcon={<FileText className="w-4 h-4" />}>
                            Generate Report
                          </Button>
                          {framework.status !== 'compliant' && (
                            <Button variant="primary" size="sm" leftIcon={<Settings className="w-4 h-4" />}>
                              Remediate
                            </Button>
                          )}
                        </div>
                      </div>
                    </Card>
                  </motion.div>
                ))}
              </div>
            </Card>
          </motion.div>
        )}

        {activeTab === 'risks' && (
          <motion.div
            key="risks"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
          >
            <Card title="AI Risk Assessment" icon={<AlertTriangle className="w-5 h-5 text-primary-500" />}>
              <div className="space-y-4">
                {riskAssessments.map((risk, index) => (
                  <motion.div
                    key={risk.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.1 }}
                    className="p-4 border border-neutral-200 dark:border-neutral-700 rounded-lg"
                  >
                    <div className="flex items-start justify-between mb-3">
                      <div className="flex-1">
                        <div className="flex items-center space-x-3 mb-2">
                          <h4 className="font-semibold text-neutral-900 dark:text-neutral-100">
                            {risk.model_name}
                          </h4>
                          <span className={`px-2 py-1 text-xs rounded-full font-medium ${getRiskLevelColor(risk.risk_level)}`}>
                            {risk.risk_level} risk
                          </span>
                          <span className={`px-2 py-1 text-xs rounded-full ${getStatusColor(risk.status)}`}>
                            {risk.status.replace('_', ' ')}
                          </span>
                        </div>
                        
                        <p className="text-sm text-neutral-600 dark:text-neutral-400 mb-2">
                          <span className="font-medium">Risk:</span> {risk.description}
                        </p>
                        
                        <div className="grid grid-cols-2 gap-4 text-sm">
                          <div>
                            <span className="font-medium text-neutral-700 dark:text-neutral-300">Impact:</span>
                            <p className="text-neutral-600 dark:text-neutral-400 mt-1">{risk.impact}</p>
                          </div>
                          <div>
                            <span className="font-medium text-neutral-700 dark:text-neutral-300">Likelihood:</span>
                            <p className="text-neutral-600 dark:text-neutral-400 mt-1">{risk.likelihood}</p>
                          </div>
                        </div>
                      </div>
                      
                      <div className="text-right">
                        <div className="text-2xl font-bold text-neutral-900 dark:text-neutral-100">
                          {risk.risk_score}/10
                        </div>
                        <div className="text-xs text-neutral-500 dark:text-neutral-400">
                          Risk Score
                        </div>
                      </div>
                    </div>
                    
                    {/* Mitigation Measures */}
                    {risk.mitigation_measures.length > 0 && (
                      <div className="mb-3">
                        <h5 className="font-medium text-neutral-900 dark:text-neutral-100 mb-2">Mitigation Measures:</h5>
                        <div className="space-y-1">
                          {risk.mitigation_measures.map((measure, idx) => (
                            <div key={idx} className="flex items-center space-x-2 text-sm">
                              <CheckCircle className="w-4 h-4 text-green-500 flex-shrink-0" />
                              <span className="text-neutral-600 dark:text-neutral-400">{measure}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                    
                    {/* Footer */}
                    <div className="flex items-center justify-between pt-3 border-t border-neutral-200 dark:border-neutral-700">
                      <div className="text-xs text-neutral-500 dark:text-neutral-400">
                        Identified by {risk.identified_by.split('@')[0]} • {formatTimeAgo(risk.identified_at)}
                      </div>
                      <div className="flex items-center space-x-2">
                        <Button variant="outline" size="sm">
                          View Details
                        </Button>
                        {risk.status === 'open' && (
                          <Button variant="primary" size="sm">
                            Mitigate
                          </Button>
                        )}
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
            </Card>
          </motion.div>
        )}

        {activeTab === 'audit' && (
          <motion.div
            key="audit"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
          >
            <Card title="Compliance Audit Trail" icon={<FileText className="w-5 h-5 text-primary-500" />}>
              <div className="space-y-3">
                {auditTrail.map((entry, index) => (
                  <motion.div
                    key={entry.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.05 }}
                    className="flex items-start space-x-4 p-4 bg-neutral-50 dark:bg-neutral-800 rounded-lg"
                  >
                    <div className={`p-2 rounded-lg ${
                      entry.compliance_relevant 
                        ? 'bg-green-100 dark:bg-green-900/20' 
                        : 'bg-gray-100 dark:bg-gray-900/20'
                    }`}>
                      {entry.action.includes('MODEL') ? <Brain className="w-5 h-5 text-blue-600 dark:text-blue-400" /> :
                       entry.action.includes('RISK') ? <AlertTriangle className="w-5 h-5 text-orange-600 dark:text-orange-400" /> :
                       entry.action.includes('DATA') ? <Database className="w-5 h-5 text-purple-600 dark:text-purple-400" /> :
                       <FileText className="w-5 h-5 text-neutral-600 dark:text-neutral-400" />}
                    </div>
                    
                    <div className="flex-1">
                      <div className="flex items-center justify-between mb-1">
                        <h4 className="font-medium text-neutral-900 dark:text-neutral-100">
                          {entry.action.replace('_', ' ').toLowerCase()}
                        </h4>
                        {entry.compliance_relevant && (
                          <span className="px-2 py-1 text-xs bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400 rounded-full">
                            Compliance Relevant
                          </span>
                        )}
                      </div>
                      
                      <div className="text-sm text-neutral-600 dark:text-neutral-400 mb-2">
                        <span className="font-medium">{entry.user}</span> performed action on{' '}
                        <span className="font-medium">{entry.resource}</span> ({entry.resource_id})
                      </div>
                      
                      {entry.details && (
                        <div className="text-xs text-neutral-500 dark:text-neutral-400 font-mono bg-neutral-100 dark:bg-neutral-900 p-2 rounded">
                          {JSON.stringify(entry.details, null, 2)}
                        </div>
                      )}
                      
                      <div className="flex items-center justify-between mt-2">
                        <div className="text-xs text-neutral-500 dark:text-neutral-400">
                          {entry.timestamp.toLocaleString()}
                        </div>
                        <div className="text-xs text-neutral-500 dark:text-neutral-400">
                          {entry.ip_address}
                        </div>
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default AIGovernanceDashboard;