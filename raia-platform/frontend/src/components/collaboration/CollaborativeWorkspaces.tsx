import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Users, Plus, Share2, MessageSquare, Eye, Edit3, 
  Clock, Star, Lock, Unlock, Settings, Download,
  Search, Filter, Calendar, Bell, UserPlus, Crown,
  GitBranch, History, FileText, BarChart3, Brain,
  CheckCircle, AlertCircle, Play, Pause, Archive
} from 'lucide-react';
import Card from '@/components/common/Card';
import Button from '@/components/common/Button';
import { apiClient } from '@/services/api';

interface WorkspaceMember {
  user_id: string;
  name: string;
  email: string;
  role: 'owner' | 'admin' | 'editor' | 'viewer';
  avatar?: string;
  last_active: Date;
  online: boolean;
}

interface WorkspaceActivity {
  id: string;
  type: 'analysis_created' | 'model_updated' | 'comment_added' | 'insight_shared' | 'member_added';
  user: {
    name: string;
    avatar?: string;
  };
  title: string;
  description: string;
  timestamp: Date;
  metadata?: any;
}

interface Workspace {
  id: string;
  name: string;
  description: string;
  type: 'model_development' | 'data_analysis' | 'research' | 'compliance';
  status: 'active' | 'archived' | 'paused';
  privacy: 'private' | 'team' | 'public';
  owner: WorkspaceMember;
  members: WorkspaceMember[];
  created_at: Date;
  last_activity: Date;
  stats: {
    analyses: number;
    models: number;
    discussions: number;
    insights: number;
  };
  tags: string[];
}

interface SharedAnalysis {
  id: string;
  title: string;
  type: 'model_comparison' | 'drift_analysis' | 'bias_report' | 'performance_report';
  author: WorkspaceMember;
  created_at: Date;
  shared_with: string[];
  comments_count: number;
  status: 'draft' | 'shared' | 'reviewed';
  tags: string[];
}

const CollaborativeWorkspaces: React.FC = () => {
  const [workspaces, setWorkspaces] = useState<Workspace[]>([]);
  const [selectedWorkspace, setSelectedWorkspace] = useState<Workspace | null>(null);
  const [sharedAnalyses, setSharedAnalyses] = useState<SharedAnalysis[]>([]);
  const [recentActivity, setRecentActivity] = useState<WorkspaceActivity[]>([]);
  const [showCreateWorkspace, setShowCreateWorkspace] = useState(false);
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [filterType, setFilterType] = useState<'all' | 'owned' | 'member'>('all');

  // Mock workspaces data
  const mockWorkspaces: Workspace[] = [
    {
      id: 'ws-1',
      name: 'Credit Risk Models',
      description: 'Collaborative development and monitoring of credit risk assessment models',
      type: 'model_development',
      status: 'active',
      privacy: 'team',
      owner: {
        user_id: 'user-1',
        name: 'Sarah Chen',
        email: 'sarah.chen@company.com',
        role: 'owner',
        avatar: '',
        last_active: new Date(Date.now() - 30 * 60 * 1000),
        online: true
      },
      members: [
        {
          user_id: 'user-2',
          name: 'Michael Torres',
          email: 'michael.torres@company.com',
          role: 'admin',
          last_active: new Date(Date.now() - 2 * 60 * 60 * 1000),
          online: false
        },
        {
          user_id: 'user-3',
          name: 'Emily Wang',
          email: 'emily.wang@company.com',
          role: 'editor',
          last_active: new Date(Date.now() - 15 * 60 * 1000),
          online: true
        },
        {
          user_id: 'user-4',
          name: 'David Kumar',
          email: 'david.kumar@company.com',
          role: 'viewer',
          last_active: new Date(Date.now() - 24 * 60 * 60 * 1000),
          online: false
        }
      ],
      created_at: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000),
      last_activity: new Date(Date.now() - 30 * 60 * 1000),
      stats: {
        analyses: 24,
        models: 8,
        discussions: 47,
        insights: 15
      },
      tags: ['credit-risk', 'lending', 'compliance']
    },
    {
      id: 'ws-2',
      name: 'Fraud Detection Research',
      description: 'Research and experimentation workspace for fraud detection algorithms',
      type: 'research',
      status: 'active',
      privacy: 'private',
      owner: {
        user_id: 'user-5',
        name: 'Alex Thompson',
        email: 'alex.thompson@company.com',
        role: 'owner',
        last_active: new Date(Date.now() - 60 * 60 * 1000),
        online: false
      },
      members: [
        {
          user_id: 'user-6',
          name: 'Lisa Rodriguez',
          email: 'lisa.rodriguez@company.com',
          role: 'editor',
          last_active: new Date(Date.now() - 3 * 60 * 60 * 1000),
          online: false
        }
      ],
      created_at: new Date(Date.now() - 15 * 24 * 60 * 60 * 1000),
      last_activity: new Date(Date.now() - 60 * 60 * 1000),
      stats: {
        analyses: 12,
        models: 3,
        discussions: 18,
        insights: 8
      },
      tags: ['fraud', 'research', 'experimental']
    },
    {
      id: 'ws-3',
      name: 'Data Quality Initiative',
      description: 'Organization-wide data quality monitoring and improvement project',
      type: 'data_analysis',
      status: 'active',
      privacy: 'public',
      owner: {
        user_id: 'user-7',
        name: 'Jennifer Park',
        email: 'jennifer.park@company.com',
        role: 'owner',
        last_active: new Date(Date.now() - 4 * 60 * 60 * 1000),
        online: false
      },
      members: [
        {
          user_id: 'user-8',
          name: 'Robert Johnson',
          email: 'robert.johnson@company.com',
          role: 'admin',
          last_active: new Date(Date.now() - 45 * 60 * 1000),
          online: true
        }
      ],
      created_at: new Date(Date.now() - 45 * 24 * 60 * 60 * 1000),
      last_activity: new Date(Date.now() - 45 * 60 * 1000),
      stats: {
        analyses: 35,
        models: 5,
        discussions: 62,
        insights: 28
      },
      tags: ['data-quality', 'monitoring', 'organization-wide']
    }
  ];

  const mockRecentActivity: WorkspaceActivity[] = [
    {
      id: 'act-1',
      type: 'analysis_created',
      user: { name: 'Sarah Chen' },
      title: 'Credit Risk Model Bias Analysis',
      description: 'Created comprehensive bias analysis for credit scoring model v2',
      timestamp: new Date(Date.now() - 30 * 60 * 1000),
      metadata: { workspace: 'Credit Risk Models', analysis_type: 'bias_report' }
    },
    {
      id: 'act-2',
      type: 'comment_added',
      user: { name: 'Emily Wang' },
      title: 'Performance Review Discussion',
      description: 'Added comments to Q3 model performance review',
      timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000),
      metadata: { workspace: 'Credit Risk Models' }
    },
    {
      id: 'act-3',
      type: 'model_updated',
      user: { name: 'Alex Thompson' },
      title: 'Fraud Detection Model v1.3',
      description: 'Updated model parameters and retrained with new data',
      timestamp: new Date(Date.now() - 4 * 60 * 60 * 1000),
      metadata: { workspace: 'Fraud Detection Research' }
    }
  ];

  const mockSharedAnalyses: SharedAnalysis[] = [
    {
      id: 'analysis-1',
      title: 'Q3 Credit Model Performance Report',
      type: 'performance_report',
      author: {
        user_id: 'user-1',
        name: 'Sarah Chen',
        email: 'sarah.chen@company.com',
        role: 'owner',
        last_active: new Date(),
        online: true
      },
      created_at: new Date(Date.now() - 24 * 60 * 60 * 1000),
      shared_with: ['Credit Risk Models', 'Data Quality Initiative'],
      comments_count: 8,
      status: 'reviewed',
      tags: ['quarterly', 'performance', 'credit']
    },
    {
      id: 'analysis-2',
      title: 'Fraud Pattern Analysis - October',
      type: 'drift_analysis',
      author: {
        user_id: 'user-5',
        name: 'Alex Thompson',
        email: 'alex.thompson@company.com',
        role: 'owner',
        last_active: new Date(),
        online: false
      },
      created_at: new Date(Date.now() - 48 * 60 * 60 * 1000),
      shared_with: ['Fraud Detection Research'],
      comments_count: 12,
      status: 'shared',
      tags: ['fraud', 'patterns', 'monthly']
    }
  ];

  useEffect(() => {
    loadWorkspaces();
  }, []);

  const loadWorkspaces = async () => {
    try {
      // In production, this would call the API
      setWorkspaces(mockWorkspaces);
      setRecentActivity(mockRecentActivity);
      setSharedAnalyses(mockSharedAnalyses);
    } catch (error) {
      console.error('Error loading workspaces:', error);
      setWorkspaces(mockWorkspaces);
      setRecentActivity(mockRecentActivity);
      setSharedAnalyses(mockSharedAnalyses);
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'model_development': return <Brain className="w-5 h-5" />;
      case 'data_analysis': return <BarChart3 className="w-5 h-5" />;
      case 'research': return <FileText className="w-5 h-5" />;
      case 'compliance': return <CheckCircle className="w-5 h-5" />;
      default: return <Users className="w-5 h-5" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'text-green-500';
      case 'paused': return 'text-yellow-500';
      case 'archived': return 'text-gray-500';
      default: return 'text-gray-500';
    }
  };

  const getPrivacyIcon = (privacy: string) => {
    switch (privacy) {
      case 'private': return <Lock className="w-4 h-4" />;
      case 'team': return <Users className="w-4 h-4" />;
      case 'public': return <Unlock className="w-4 h-4" />;
      default: return <Lock className="w-4 h-4" />;
    }
  };

  const formatTimeAgo = (timestamp: Date) => {
    const now = new Date();
    const diffInMinutes = Math.floor((now.getTime() - timestamp.getTime()) / (1000 * 60));
    
    if (diffInMinutes < 1) return 'Just now';
    if (diffInMinutes < 60) return `${diffInMinutes}m ago`;
    if (diffInMinutes < 1440) return `${Math.floor(diffInMinutes / 60)}h ago`;
    return `${Math.floor(diffInMinutes / 1440)}d ago`;
  };

  const getActivityIcon = (type: string) => {
    switch (type) {
      case 'analysis_created': return <FileText className="w-4 h-4" />;
      case 'model_updated': return <Brain className="w-4 h-4" />;
      case 'comment_added': return <MessageSquare className="w-4 h-4" />;
      case 'insight_shared': return <Share2 className="w-4 h-4" />;
      case 'member_added': return <UserPlus className="w-4 h-4" />;
      default: return <Activity className="w-4 h-4" />;
    }
  };

  const filteredWorkspaces = workspaces.filter(workspace => {
    if (filterType === 'owned') {
      return workspace.owner.user_id === 'current-user-id'; // Would be current user's ID
    } else if (filterType === 'member') {
      return workspace.members.some(member => member.user_id === 'current-user-id');
    }
    return true;
  });

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-neutral-900 dark:text-neutral-100 flex items-center">
            <div className="w-8 h-8 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg flex items-center justify-center mr-3">
              <Users className="w-5 h-5 text-white" />
            </div>
            Collaborative Workspaces
          </h1>
          <p className="text-neutral-600 dark:text-neutral-400 mt-1">
            Team-based analysis environments with shared insights and discussions
          </p>
        </div>
        
        <div className="flex items-center space-x-2">
          <Button
            variant="outline"
            size="sm"
            leftIcon={<Bell className="w-4 h-4" />}
          >
            Notifications
          </Button>
          <Button
            variant="outline"
            size="sm"
            leftIcon={<Settings className="w-4 h-4" />}
          >
            Settings
          </Button>
          <Button
            variant="primary"
            size="sm"
            leftIcon={<Plus className="w-4 h-4" />}
            onClick={() => setShowCreateWorkspace(true)}
          >
            New Workspace
          </Button>
        </div>
      </div>

      {/* Filters and View Controls */}
      <Card>
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <Search className="w-4 h-4 text-neutral-400" />
              <input
                type="text"
                placeholder="Search workspaces..."
                className="border-0 bg-transparent text-neutral-900 dark:text-neutral-100 placeholder-neutral-400 focus:outline-none"
              />
            </div>
            
            <div className="flex items-center space-x-1 bg-neutral-100 dark:bg-neutral-800 rounded-lg p-1">
              {[
                { id: 'all', label: 'All' },
                { id: 'owned', label: 'Owned' },
                { id: 'member', label: 'Member' },
              ].map((filter) => (
                <button
                  key={filter.id}
                  onClick={() => setFilterType(filter.id as any)}
                  className={`px-3 py-1 rounded-md text-sm font-medium transition-colors ${
                    filterType === filter.id
                      ? 'bg-white dark:bg-neutral-700 text-primary-600 dark:text-primary-400 shadow-sm'
                      : 'text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-neutral-100'
                  }`}
                >
                  {filter.label}
                </button>
              ))}
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            <Button
              variant="outline"
              size="sm"
              leftIcon={<Filter className="w-4 h-4" />}
            >
              Filter
            </Button>
            
            <div className="flex items-center bg-neutral-100 dark:bg-neutral-800 rounded-lg p-1">
              <button
                onClick={() => setViewMode('grid')}
                className={`p-1 rounded ${viewMode === 'grid' ? 'bg-white dark:bg-neutral-700 shadow-sm' : ''}`}
              >
                <BarChart3 className="w-4 h-4" />
              </button>
              <button
                onClick={() => setViewMode('list')}
                className={`p-1 rounded ${viewMode === 'list' ? 'bg-white dark:bg-neutral-700 shadow-sm' : ''}`}
              >
                <Users className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>
      </Card>

      {/* Workspaces Grid/List */}
      <div className={viewMode === 'grid' ? 'grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6' : 'space-y-4'}>
        <AnimatePresence>
          {filteredWorkspaces.map((workspace, index) => (
            <motion.div
              key={workspace.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ delay: index * 0.1 }}
              onClick={() => setSelectedWorkspace(workspace)}
              className="cursor-pointer"
            >
              <Card className="hover:shadow-lg transition-all duration-200 h-full">
                <div className="space-y-4">
                  {/* Workspace Header */}
                  <div className="flex items-start justify-between">
                    <div className="flex items-start space-x-3">
                      <div className={`p-2 rounded-lg ${
                        workspace.type === 'model_development' ? 'bg-blue-100 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400' :
                        workspace.type === 'data_analysis' ? 'bg-green-100 dark:bg-green-900/20 text-green-600 dark:text-green-400' :
                        workspace.type === 'research' ? 'bg-purple-100 dark:bg-purple-900/20 text-purple-600 dark:text-purple-400' :
                        'bg-orange-100 dark:bg-orange-900/20 text-orange-600 dark:text-orange-400'
                      }`}>
                        {getTypeIcon(workspace.type)}
                      </div>
                      <div className="flex-1">
                        <h3 className="font-semibold text-neutral-900 dark:text-neutral-100 mb-1">
                          {workspace.name}
                        </h3>
                        <p className="text-sm text-neutral-600 dark:text-neutral-400 line-clamp-2">
                          {workspace.description}
                        </p>
                      </div>
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      <div className="flex items-center space-x-1 text-neutral-400">
                        {getPrivacyIcon(workspace.privacy)}
                        <span className={`w-2 h-2 rounded-full ${getStatusColor(workspace.status)} bg-current`}></span>
                      </div>
                    </div>
                  </div>
                  
                  {/* Stats */}
                  <div className="grid grid-cols-4 gap-2">
                    <div className="text-center">
                      <p className="text-lg font-bold text-neutral-900 dark:text-neutral-100">
                        {workspace.stats.analyses}
                      </p>
                      <p className="text-xs text-neutral-500 dark:text-neutral-400">Analyses</p>
                    </div>
                    <div className="text-center">
                      <p className="text-lg font-bold text-neutral-900 dark:text-neutral-100">
                        {workspace.stats.models}
                      </p>
                      <p className="text-xs text-neutral-500 dark:text-neutral-400">Models</p>
                    </div>
                    <div className="text-center">
                      <p className="text-lg font-bold text-neutral-900 dark:text-neutral-100">
                        {workspace.stats.discussions}
                      </p>
                      <p className="text-xs text-neutral-500 dark:text-neutral-400">Discussions</p>
                    </div>
                    <div className="text-center">
                      <p className="text-lg font-bold text-neutral-900 dark:text-neutral-100">
                        {workspace.stats.insights}
                      </p>
                      <p className="text-xs text-neutral-500 dark:text-neutral-400">Insights</p>
                    </div>
                  </div>
                  
                  {/* Members */}
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <div className="flex -space-x-2">
                        {workspace.members.slice(0, 3).map((member, idx) => (
                          <div
                            key={member.user_id}
                            className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-medium text-white border-2 border-white dark:border-neutral-800 ${
                              idx === 0 ? 'bg-blue-500' :
                              idx === 1 ? 'bg-green-500' :
                              'bg-purple-500'
                            }`}
                            title={member.name}
                          >
                            {member.name.split(' ').map(n => n[0]).join('')}
                          </div>
                        ))}
                        {workspace.members.length > 3 && (
                          <div className="w-8 h-8 rounded-full bg-neutral-300 dark:bg-neutral-600 flex items-center justify-center text-xs font-medium text-neutral-700 dark:text-neutral-300 border-2 border-white dark:border-neutral-800">
                            +{workspace.members.length - 3}
                          </div>
                        )}
                      </div>
                      <span className="text-sm text-neutral-500 dark:text-neutral-400">
                        {workspace.members.length + 1} members
                      </span>
                    </div>
                    
                    <div className="flex items-center space-x-2 text-xs text-neutral-500 dark:text-neutral-400">
                      <Clock className="w-3 h-3" />
                      <span>{formatTimeAgo(workspace.last_activity)}</span>
                    </div>
                  </div>
                  
                  {/* Tags */}
                  {workspace.tags.length > 0 && (
                    <div className="flex flex-wrap gap-1">
                      {workspace.tags.slice(0, 3).map((tag) => (
                        <span
                          key={tag}
                          className="px-2 py-1 text-xs bg-neutral-100 dark:bg-neutral-800 text-neutral-600 dark:text-neutral-400 rounded"
                        >
                          {tag}
                        </span>
                      ))}
                      {workspace.tags.length > 3 && (
                        <span className="px-2 py-1 text-xs bg-neutral-100 dark:bg-neutral-800 text-neutral-600 dark:text-neutral-400 rounded">
                          +{workspace.tags.length - 3}
                        </span>
                      )}
                    </div>
                  )}
                </div>
              </Card>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>

      {/* Recent Activity & Shared Analyses */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recent Activity */}
        <Card title="Recent Activity" icon={<Activity className="w-5 h-5 text-primary-500" />}>
          <div className="space-y-4">
            {recentActivity.map((activity) => (
              <div key={activity.id} className="flex items-start space-x-3">
                <div className="p-2 bg-neutral-100 dark:bg-neutral-800 rounded-lg">
                  {getActivityIcon(activity.type)}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="font-medium text-neutral-900 dark:text-neutral-100 text-sm">
                    {activity.title}
                  </p>
                  <p className="text-sm text-neutral-600 dark:text-neutral-400 mt-1">
                    <span className="font-medium">{activity.user.name}</span> {activity.description}
                  </p>
                  <div className="flex items-center space-x-2 mt-2 text-xs text-neutral-500 dark:text-neutral-400">
                    <Clock className="w-3 h-3" />
                    <span>{formatTimeAgo(activity.timestamp)}</span>
                    {activity.metadata?.workspace && (
                      <>
                        <span>•</span>
                        <span>{activity.metadata.workspace}</span>
                      </>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Card>

        {/* Shared Analyses */}
        <Card title="Shared Analyses" icon={<Share2 className="w-5 h-5 text-primary-500" />}>
          <div className="space-y-4">
            {sharedAnalyses.map((analysis) => (
              <div key={analysis.id} className="p-3 bg-neutral-50 dark:bg-neutral-800 rounded-lg">
                <div className="flex items-start justify-between mb-2">
                  <h4 className="font-medium text-neutral-900 dark:text-neutral-100 text-sm">
                    {analysis.title}
                  </h4>
                  <span className={`px-2 py-1 text-xs rounded-full ${
                    analysis.status === 'reviewed' ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' :
                    analysis.status === 'shared' ? 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200' :
                    'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200'
                  }`}>
                    {analysis.status}
                  </span>
                </div>
                
                <div className="flex items-center justify-between text-xs text-neutral-500 dark:text-neutral-400">
                  <div className="flex items-center space-x-2">
                    <span>by {analysis.author.name}</span>
                    <span>•</span>
                    <span>{formatTimeAgo(analysis.created_at)}</span>
                  </div>
                  <div className="flex items-center space-x-1">
                    <MessageSquare className="w-3 h-3" />
                    <span>{analysis.comments_count}</span>
                  </div>
                </div>
                
                <div className="flex flex-wrap gap-1 mt-2">
                  {analysis.tags.slice(0, 3).map((tag) => (
                    <span
                      key={tag}
                      className="px-1.5 py-0.5 text-xs bg-primary-100 dark:bg-primary-900/20 text-primary-600 dark:text-primary-400 rounded"
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>

      {/* Create Workspace Modal */}
      <AnimatePresence>
        {showCreateWorkspace && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
            onClick={() => setShowCreateWorkspace(false)}
          >
            <motion.div
              initial={{ scale: 0.9 }}
              animate={{ scale: 1 }}
              exit={{ scale: 0.9 }}
              className="bg-white dark:bg-neutral-800 rounded-lg p-6 w-full max-w-lg mx-4"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
                  Create New Workspace
                </h3>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowCreateWorkspace(false)}
                >
                  ×
                </Button>
              </div>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                    Workspace Name
                  </label>
                  <input
                    type="text"
                    placeholder="Enter workspace name"
                    className="w-full p-3 border border-neutral-200 dark:border-neutral-700 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                    Description
                  </label>
                  <textarea
                    rows={3}
                    placeholder="Describe the purpose of this workspace"
                    className="w-full p-3 border border-neutral-200 dark:border-neutral-700 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100 resize-none"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                    Type
                  </label>
                  <select className="w-full p-3 border border-neutral-200 dark:border-neutral-700 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100">
                    <option value="model_development">Model Development</option>
                    <option value="data_analysis">Data Analysis</option>
                    <option value="research">Research</option>
                    <option value="compliance">Compliance</option>
                  </select>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                    Privacy
                  </label>
                  <select className="w-full p-3 border border-neutral-200 dark:border-neutral-700 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100">
                    <option value="private">Private (Only you)</option>
                    <option value="team">Team (Invited members only)</option>
                    <option value="public">Public (All organization members)</option>
                  </select>
                </div>
              </div>
              
              <div className="flex justify-end space-x-3 mt-6">
                <Button
                  variant="outline"
                  onClick={() => setShowCreateWorkspace(false)}
                >
                  Cancel
                </Button>
                <Button variant="primary">
                  Create Workspace
                </Button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default CollaborativeWorkspaces;