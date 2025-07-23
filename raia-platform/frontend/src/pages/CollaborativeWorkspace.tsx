import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Users,
  MessageCircle,
  Share2,
  Eye,
  Edit3,
  Save,
  Download,
  Video,
  Mic,
  MicOff,
  VideoOff,
  UserPlus,
  Settings,
  Lock,
  Unlock,
  Bookmark,
  Flag,
  Clock,
  CheckCircle2,
  AlertCircle,
  Camera,
  FileText,
  Layers,
  Zap,
  Brain,
  Target,
  TrendingUp
} from 'lucide-react';
import { Card } from '@/components/common/Card';
import { Button } from '@/components/common/Button';

interface CollaboratorCursor {
  id: string;
  name: string;
  color: string;
  x: number;
  y: number;
  lastSeen: Date;
}

interface Annotation {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
  text: string;
  author: string;
  timestamp: Date;
  type: 'note' | 'question' | 'finding' | 'action';
  resolved: boolean;
  replies: AnnotationReply[];
}

interface AnnotationReply {
  id: string;
  text: string;
  author: string;
  timestamp: Date;
}

interface KnowledgeItem {
  id: string;
  title: string;
  description: string;
  tags: string[];
  author: string;
  timestamp: Date;
  type: 'insight' | 'pattern' | 'anomaly' | 'recommendation';
  upvotes: number;
  category: string;
}

interface Collaborator {
  id: string;
  name: string;
  avatar: string;
  role: string;
  isOnline: boolean;
  permissions: 'view' | 'edit' | 'admin';
  lastActive: Date;
  cursor?: CollaboratorCursor;
}

const CollaborativeWorkspace: React.FC = () => {
  const [collaborators, setCollaborators] = useState<Collaborator[]>([]);
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [knowledgeBase, setKnowledgeBase] = useState<KnowledgeItem[]>([]);
  const [activeAnnotation, setActiveAnnotation] = useState<string | null>(null);
  const [newAnnotation, setNewAnnotation] = useState<{ x: number; y: number } | null>(null);
  const [isVideoCall, setIsVideoCall] = useState(false);
  const [isMicOn, setIsMicOn] = useState(false);
  const [isScreenShare, setIsScreenShare] = useState(false);
  const [selectedVisualization, setSelectedVisualization] = useState('feature-importance');
  const [workspaceMode, setWorkspaceMode] = useState<'explore' | 'investigate' | 'present'>('explore');
  const [cursors, setCursors] = useState<CollaboratorCursor[]>([]);
  const workspaceRef = useRef<HTMLDivElement>(null);

  // Initialize mock data
  useEffect(() => {
    const mockCollaborators: Collaborator[] = [
      {
        id: '1',
        name: 'Dr. Sarah Chen',
        avatar: '/avatars/sarah.jpg',
        role: 'ML Researcher',
        isOnline: true,
        permissions: 'admin',
        lastActive: new Date(),
      },
      {
        id: '2', 
        name: 'Alex Rodriguez',
        avatar: '/avatars/alex.jpg',
        role: 'Data Scientist',
        isOnline: true,
        permissions: 'edit',
        lastActive: new Date(),
      },
      {
        id: '3',
        name: 'Emily Zhang',
        avatar: '/avatars/emily.jpg',
        role: 'ML Engineer',
        isOnline: false,
        permissions: 'view',
        lastActive: new Date(Date.now() - 1800000), // 30 min ago
      }
    ];

    const mockAnnotations: Annotation[] = [
      {
        id: 'ann-1',
        x: 200,
        y: 150,
        width: 200,
        height: 100,
        text: 'This feature shows unexpected importance. Need to investigate data quality.',
        author: 'Dr. Sarah Chen',
        timestamp: new Date(),
        type: 'finding',
        resolved: false,
        replies: [
          {
            id: 'reply-1',
            text: 'I noticed this too. Could be data leakage?',
            author: 'Alex Rodriguez',
            timestamp: new Date(Date.now() - 600000)
          }
        ]
      },
      {
        id: 'ann-2',
        x: 400,
        y: 300,
        width: 180,
        height: 80,
        text: 'SHAP values look correct here. Good baseline.',
        author: 'Emily Zhang',
        timestamp: new Date(Date.now() - 1800000),
        type: 'note',
        resolved: true,
        replies: []
      }
    ];

    const mockKnowledge: KnowledgeItem[] = [
      {
        id: 'kb-1',
        title: 'Feature Correlation Pattern',
        description: 'Discovered strong correlation between age and income features affecting model predictions',
        tags: ['correlation', 'features', 'bias'],
        author: 'Dr. Sarah Chen',
        timestamp: new Date(),
        type: 'pattern',
        upvotes: 5,
        category: 'Feature Analysis'
      },
      {
        id: 'kb-2',
        title: 'Data Drift Alert',
        description: 'Model performance degradation linked to seasonal data distribution changes',
        tags: ['drift', 'seasonal', 'performance'],
        author: 'Alex Rodriguez',
        timestamp: new Date(Date.now() - 3600000),
        type: 'anomaly',
        upvotes: 3,
        category: 'Model Monitoring'
      }
    ];

    setCollaborators(mockCollaborators);
    setAnnotations(mockAnnotations);
    setKnowledgeBase(mockKnowledge);
  }, []);

  // Handle mouse movement for cursor tracking
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (workspaceRef.current) {
        const rect = workspaceRef.current.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        // Update cursor position for current user (simulated)
        setCursors(prev => [
          ...prev.filter(c => c.id !== 'current-user'),
          {
            id: 'current-user',
            name: 'You',
            color: '#3b82f6',
            x,
            y,
            lastSeen: new Date()
          }
        ]);
      }
    };

    const current = workspaceRef.current;
    if (current) {
      current.addEventListener('mousemove', handleMouseMove);
      return () => current.removeEventListener('mousemove', handleMouseMove);
    }
  }, []);

  // Handle annotation creation
  const handleWorkspaceClick = (e: React.MouseEvent) => {
    if (e.detail === 2) { // Double click
      const rect = e.currentTarget.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      setNewAnnotation({ x, y });
    }
  };

  const createAnnotation = (text: string, type: Annotation['type']) => {
    if (newAnnotation) {
      const annotation: Annotation = {
        id: `ann-${Date.now()}`,
        x: newAnnotation.x,
        y: newAnnotation.y,
        width: 200,
        height: 100,
        text,
        author: 'You',
        timestamp: new Date(),
        type,
        resolved: false,
        replies: []
      };
      setAnnotations(prev => [...prev, annotation]);
      setNewAnnotation(null);
    }
  };

  const addKnowledgeItem = (item: Omit<KnowledgeItem, 'id' | 'timestamp' | 'upvotes' | 'author'>) => {
    const knowledgeItem: KnowledgeItem = {
      ...item,
      id: `kb-${Date.now()}`,
      timestamp: new Date(),
      upvotes: 0,
      author: 'You'
    };
    setKnowledgeBase(prev => [...prev, knowledgeItem]);
  };

  const getAnnotationIcon = (type: Annotation['type']) => {
    switch (type) {
      case 'note': return <FileText className="w-4 h-4" />;
      case 'question': return <AlertCircle className="w-4 h-4" />;
      case 'finding': return <Eye className="w-4 h-4" />;
      case 'action': return <Flag className="w-4 h-4" />;
      default: return <FileText className="w-4 h-4" />;
    }
  };

  const getAnnotationColor = (type: Annotation['type']) => {
    switch (type) {
      case 'note': return 'border-blue-500 bg-blue-50';
      case 'question': return 'border-orange-500 bg-orange-50';
      case 'finding': return 'border-green-500 bg-green-50';
      case 'action': return 'border-red-500 bg-red-50';
      default: return 'border-gray-500 bg-gray-50';
    }
  };

  return (
    <div className="flex h-screen bg-neutral-50 dark:bg-neutral-900">
      {/* Left Sidebar - Collaborators & Tools */}
      <div className="w-80 bg-white dark:bg-neutral-800 border-r border-neutral-200 dark:border-neutral-700 flex flex-col">
        {/* Header */}
        <div className="p-4 border-b border-neutral-200 dark:border-neutral-700">
          <h2 className="text-lg font-semibold text-neutral-900 dark:text-white flex items-center gap-2">
            <Users className="w-5 h-5 text-purple-600" />
            Collaborative Workspace
          </h2>
          <p className="text-sm text-neutral-600 dark:text-neutral-400 mt-1">
            Model explanation session
          </p>
        </div>

        {/* Video Call Controls */}
        <div className="p-4 border-b border-neutral-200 dark:border-neutral-700">
          <div className="flex items-center gap-2 mb-3">
            <Button
              variant={isVideoCall ? "destructive" : "secondary"}
              size="sm"
              onClick={() => setIsVideoCall(!isVideoCall)}
              className="flex items-center gap-2"
            >
              {isVideoCall ? <VideoOff className="w-4 h-4" /> : <Video className="w-4 h-4" />}
            </Button>
            
            <Button
              variant={isMicOn ? "secondary" : "destructive"}
              size="sm"
              onClick={() => setIsMicOn(!isMicOn)}
              className="flex items-center gap-2"
            >
              {isMicOn ? <Mic className="w-4 h-4" /> : <MicOff className="w-4 h-4" />}
            </Button>
            
            <Button
              variant={isScreenShare ? "default" : "outline"}
              size="sm"
              onClick={() => setIsScreenShare(!isScreenShare)}
              className="flex items-center gap-2"
            >
              <Share2 className="w-4 h-4" />
            </Button>
          </div>

          {/* Mode Selection */}
          <div className="flex gap-1 p-1 bg-neutral-100 dark:bg-neutral-700 rounded-lg">
            {(['explore', 'investigate', 'present'] as const).map((mode) => (
              <button
                key={mode}
                onClick={() => setWorkspaceMode(mode)}
                className={`px-3 py-1 text-xs rounded-md transition-all ${
                  workspaceMode === mode
                    ? 'bg-white dark:bg-neutral-800 shadow-sm text-neutral-900 dark:text-white'
                    : 'text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-white'
                }`}
              >
                {mode.charAt(0).toUpperCase() + mode.slice(1)}
              </button>
            ))}
          </div>
        </div>

        {/* Active Collaborators */}
        <div className="p-4 border-b border-neutral-200 dark:border-neutral-700">
          <h3 className="text-sm font-medium text-neutral-900 dark:text-white mb-3">
            Active Collaborators ({collaborators.filter(c => c.isOnline).length})
          </h3>
          <div className="space-y-2">
            {collaborators.map((collaborator) => (
              <div key={collaborator.id} className="flex items-center gap-3">
                <div className="relative">
                  <div className="w-8 h-8 bg-gradient-to-br from-purple-500 to-pink-500 rounded-full flex items-center justify-center text-white text-sm font-medium">
                    {collaborator.name.charAt(0)}
                  </div>
                  <div className={`absolute -bottom-1 -right-1 w-3 h-3 rounded-full border-2 border-white dark:border-neutral-800 ${
                    collaborator.isOnline ? 'bg-green-500' : 'bg-gray-400'
                  }`} />
                </div>
                
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium text-neutral-900 dark:text-white truncate">
                    {collaborator.name}
                  </div>
                  <div className="text-xs text-neutral-600 dark:text-neutral-400">
                    {collaborator.role} • {collaborator.permissions}
                  </div>
                </div>

                <div className="flex items-center gap-1">
                  {collaborator.permissions === 'admin' && (
                    <Lock className="w-3 h-3 text-neutral-400" />
                  )}
                  {collaborator.isOnline && (
                    <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                  )}
                </div>
              </div>
            ))}
          </div>

          <Button variant="outline" size="sm" className="w-full mt-3 flex items-center gap-2">
            <UserPlus className="w-4 h-4" />
            Invite Collaborator
          </Button>
        </div>

        {/* Annotations Panel */}
        <div className="flex-1 overflow-y-auto p-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-medium text-neutral-900 dark:text-white">
              Annotations ({annotations.length})
            </h3>
            <Button variant="outline" size="sm">
              <Download className="w-4 h-4" />
            </Button>
          </div>

          <div className="space-y-3">
            {annotations.map((annotation) => (
              <motion.div
                key={annotation.id}
                layout
                className={`p-3 rounded-lg border cursor-pointer transition-all ${
                  activeAnnotation === annotation.id 
                    ? 'ring-2 ring-purple-500 border-purple-300' 
                    : getAnnotationColor(annotation.type)
                }`}
                onClick={() => setActiveAnnotation(
                  activeAnnotation === annotation.id ? null : annotation.id
                )}
              >
                <div className="flex items-start gap-2 mb-2">
                  {getAnnotationIcon(annotation.type)}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-xs font-medium text-neutral-900 dark:text-white">
                        {annotation.author}
                      </span>
                      <span className="text-xs text-neutral-500">
                        {annotation.timestamp.toLocaleTimeString()}
                      </span>
                      {annotation.resolved && (
                        <CheckCircle2 className="w-3 h-3 text-green-500" />
                      )}
                    </div>
                    <p className="text-sm text-neutral-700 dark:text-neutral-300">
                      {annotation.text}
                    </p>
                  </div>
                </div>

                {annotation.replies.length > 0 && (
                  <div className="ml-6 space-y-2">
                    {annotation.replies.map((reply) => (
                      <div key={reply.id} className="bg-neutral-100 dark:bg-neutral-700 p-2 rounded">
                        <div className="flex items-center gap-2 mb-1">
                          <span className="text-xs font-medium">{reply.author}</span>
                          <span className="text-xs text-neutral-500">
                            {reply.timestamp.toLocaleTimeString()}
                          </span>
                        </div>
                        <p className="text-xs text-neutral-600 dark:text-neutral-400">
                          {reply.text}
                        </p>
                      </div>
                    ))}
                  </div>
                )}
              </motion.div>
            ))}
          </div>
        </div>

        {/* Knowledge Base Quick Access */}
        <div className="p-4 border-t border-neutral-200 dark:border-neutral-700">
          <h3 className="text-sm font-medium text-neutral-900 dark:text-white mb-2">
            Session Insights ({knowledgeBase.length})
          </h3>
          <div className="space-y-2">
            {knowledgeBase.slice(0, 2).map((item) => (
              <div key={item.id} className="p-2 bg-neutral-100 dark:bg-neutral-700 rounded">
                <div className="text-xs font-medium text-neutral-900 dark:text-white">
                  {item.title}
                </div>
                <div className="text-xs text-neutral-600 dark:text-neutral-400 mt-1">
                  {item.type} • {item.upvotes} upvotes
                </div>
              </div>
            ))}
          </div>
          <Button variant="link" size="sm" className="p-0 mt-2 text-xs">
            View All Insights →
          </Button>
        </div>
      </div>

      {/* Main Workspace */}
      <div className="flex-1 flex flex-col">
        {/* Top Bar */}
        <div className="h-14 bg-white dark:bg-neutral-800 border-b border-neutral-200 dark:border-neutral-700 flex items-center justify-between px-4">
          <div className="flex items-center gap-4">
            <select 
              value={selectedVisualization} 
              onChange={(e) => setSelectedVisualization(e.target.value)}
              className="px-3 py-1 border border-neutral-300 dark:border-neutral-600 rounded bg-white dark:bg-neutral-700 text-sm"
            >
              <option value="feature-importance">Feature Importance</option>
              <option value="shap-values">SHAP Values</option>
              <option value="partial-dependence">Partial Dependence</option>
              <option value="model-performance">Model Performance</option>
            </select>

            <div className="flex items-center gap-2">
              <span className="text-sm text-neutral-600 dark:text-neutral-400">
                Double-click to annotate
              </span>
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
            </div>
          </div>

          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm">
              <Camera className="w-4 h-4 mr-2" />
              Screenshot
            </Button>
            <Button variant="outline" size="sm">
              <Save className="w-4 h-4 mr-2" />
              Save Session
            </Button>
          </div>
        </div>

        {/* Workspace Canvas */}
        <div 
          ref={workspaceRef}
          className="flex-1 relative bg-white dark:bg-neutral-900 overflow-hidden"
          onDoubleClick={handleWorkspaceClick}
        >
          {/* Visualization Content */}
          <div className="absolute inset-4 bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-lg border-2 border-dashed border-neutral-300 dark:border-neutral-600 flex items-center justify-center">
            <div className="text-center">
              <Brain className="w-12 h-12 text-purple-500 mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-neutral-900 dark:text-white mb-2">
                {selectedVisualization.split('-').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')}
              </h3>
              <p className="text-neutral-600 dark:text-neutral-400">
                Interactive model explanation visualization would be rendered here
              </p>
            </div>
          </div>

          {/* Render Annotations */}
          {annotations.map((annotation) => (
            <motion.div
              key={annotation.id}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              className={`absolute border-2 rounded-lg p-2 bg-white dark:bg-neutral-800 shadow-lg cursor-pointer z-10 ${
                activeAnnotation === annotation.id 
                  ? 'ring-2 ring-purple-500' 
                  : 'border-neutral-300 dark:border-neutral-600'
              }`}
              style={{ 
                left: annotation.x, 
                top: annotation.y,
                width: annotation.width,
                minHeight: annotation.height
              }}
              onClick={() => setActiveAnnotation(annotation.id)}
            >
              <div className="flex items-center gap-2 mb-1">
                {getAnnotationIcon(annotation.type)}
                <span className="text-xs font-medium">{annotation.author}</span>
                {annotation.resolved && (
                  <CheckCircle2 className="w-3 h-3 text-green-500" />
                )}
              </div>
              <p className="text-xs text-neutral-700 dark:text-neutral-300">
                {annotation.text}
              </p>
            </motion.div>
          ))}

          {/* Render Collaborator Cursors */}
          {cursors.filter(c => c.id !== 'current-user').map((cursor) => (
            <motion.div
              key={cursor.id}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="absolute z-20 pointer-events-none"
              style={{ left: cursor.x, top: cursor.y }}
            >
              <div 
                className="w-4 h-4 rounded-full border-2 border-white shadow-lg"
                style={{ backgroundColor: cursor.color }}
              />
              <div 
                className="absolute top-4 left-0 px-2 py-1 rounded text-xs text-white shadow-lg whitespace-nowrap"
                style={{ backgroundColor: cursor.color }}
              >
                {cursor.name}
              </div>
            </motion.div>
          ))}

          {/* New Annotation Dialog */}
          <AnimatePresence>
            {newAnnotation && (
              <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
                className="absolute bg-white dark:bg-neutral-800 border border-neutral-300 dark:border-neutral-600 rounded-lg shadow-xl p-4 z-30"
                style={{ left: newAnnotation.x, top: newAnnotation.y }}
              >
                <h4 className="font-medium mb-3">Add Annotation</h4>
                <div className="space-y-3">
                  <textarea 
                    placeholder="Enter your observation..."
                    className="w-64 h-20 p-2 border border-neutral-300 dark:border-neutral-600 rounded text-sm resize-none"
                    autoFocus
                  />
                  <div className="flex gap-2">
                    <Button 
                      size="sm" 
                      onClick={() => createAnnotation('Sample finding', 'finding')}
                    >
                      Finding
                    </Button>
                    <Button 
                      variant="outline" 
                      size="sm"
                      onClick={() => createAnnotation('Sample note', 'note')}
                    >
                      Note
                    </Button>
                    <Button 
                      variant="outline" 
                      size="sm"
                      onClick={() => createAnnotation('Sample question', 'question')}
                    >
                      Question
                    </Button>
                    <Button 
                      variant="destructive" 
                      size="sm"
                      onClick={() => setNewAnnotation(null)}
                    >
                      Cancel
                    </Button>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Status Bar */}
        <div className="h-8 bg-neutral-100 dark:bg-neutral-800 border-t border-neutral-200 dark:border-neutral-700 flex items-center justify-between px-4 text-xs text-neutral-600 dark:text-neutral-400">
          <div className="flex items-center gap-4">
            <span>Session: {new Date().toLocaleTimeString()}</span>
            <span>Mode: {workspaceMode}</span>
            <span>{annotations.length} annotations</span>
          </div>
          <div className="flex items-center gap-4">
            <span>{collaborators.filter(c => c.isOnline).length} online</span>
            <div className="flex items-center gap-1">
              <div className="w-2 h-2 bg-green-500 rounded-full" />
              <span>Saved</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CollaborativeWorkspace;