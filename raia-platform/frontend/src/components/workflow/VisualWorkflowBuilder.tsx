import React, { useState, useCallback, useRef, useEffect } from 'react';
import ReactFlow, {
  Node,
  Edge,
  Controls,
  Background,
  MiniMap,
  ReactFlowProvider,
  useNodesState,
  useEdgesState,
  addEdge,
  Connection,
  ConnectionMode,
  NodeTypes,
  Handle,
  Position,
  NodeProps,
  EdgeTypes,
  getBezierPath,
  EdgeProps,
  MarkerType
} from 'reactflow';
import 'reactflow/dist/style.css';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Database, Brain, GitBranch, Filter, Layers, 
  BarChart3, Save, Play, Pause, Download, Upload,
  Settings, Zap, FileText, Share2, Code, Eye,
  Plus, X, CheckCircle, AlertTriangle, Clock,
  Cpu, Cloud, HardDrive, Activity, Target,
  Shuffle, Binary, Calculator, ChevronRight
} from 'lucide-react';
import Card from '@/components/common/Card';
import Button from '@/components/common/Button';
import { apiClient } from '@/services/api';

// Node type definitions
interface NodeData {
  label: string;
  type: string;
  config?: any;
  status?: 'idle' | 'running' | 'completed' | 'error';
  error?: string;
  output?: any;
  icon?: React.ReactNode;
}

// Custom node components
const DataSourceNode: React.FC<NodeProps<NodeData>> = ({ data, selected }) => {
  return (
    <div className={`px-4 py-3 rounded-lg border-2 ${
      selected ? 'border-blue-500 shadow-lg' : 'border-neutral-300 dark:border-neutral-600'
    } bg-gradient-to-r from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 min-w-[180px]`}>
      <Handle
        type="source"
        position={Position.Right}
        className="w-3 h-3 bg-blue-500 border-2 border-white dark:border-neutral-800"
      />
      <div className="flex items-center space-x-2">
        <Database className="w-5 h-5 text-blue-600 dark:text-blue-400" />
        <div>
          <div className="font-medium text-neutral-900 dark:text-neutral-100">{data.label}</div>
          <div className="text-xs text-neutral-600 dark:text-neutral-400">Data Source</div>
        </div>
      </div>
      {data.status === 'running' && (
        <div className="mt-2 text-xs text-blue-600 dark:text-blue-400 flex items-center">
          <Clock className="w-3 h-3 mr-1 animate-pulse" />
          Loading...
        </div>
      )}
    </div>
  );
};

const TransformNode: React.FC<NodeProps<NodeData>> = ({ data, selected }) => {
  return (
    <div className={`px-4 py-3 rounded-lg border-2 ${
      selected ? 'border-purple-500 shadow-lg' : 'border-neutral-300 dark:border-neutral-600'
    } bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 min-w-[180px]`}>
      <Handle
        type="target"
        position={Position.Left}
        className="w-3 h-3 bg-purple-500 border-2 border-white dark:border-neutral-800"
      />
      <Handle
        type="source"
        position={Position.Right}
        className="w-3 h-3 bg-purple-500 border-2 border-white dark:border-neutral-800"
      />
      <div className="flex items-center space-x-2">
        <Filter className="w-5 h-5 text-purple-600 dark:text-purple-400" />
        <div>
          <div className="font-medium text-neutral-900 dark:text-neutral-100">{data.label}</div>
          <div className="text-xs text-neutral-600 dark:text-neutral-400">Transform</div>
        </div>
      </div>
    </div>
  );
};

const ModelNode: React.FC<NodeProps<NodeData>> = ({ data, selected }) => {
  return (
    <div className={`px-4 py-3 rounded-lg border-2 ${
      selected ? 'border-green-500 shadow-lg' : 'border-neutral-300 dark:border-neutral-600'
    } bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 min-w-[180px]`}>
      <Handle
        type="target"
        position={Position.Left}
        className="w-3 h-3 bg-green-500 border-2 border-white dark:border-neutral-800"
      />
      <Handle
        type="source"
        position={Position.Right}
        className="w-3 h-3 bg-green-500 border-2 border-white dark:border-neutral-800"
      />
      <div className="flex items-center space-x-2">
        <Brain className="w-5 h-5 text-green-600 dark:text-green-400" />
        <div>
          <div className="font-medium text-neutral-900 dark:text-neutral-100">{data.label}</div>
          <div className="text-xs text-neutral-600 dark:text-neutral-400">ML Model</div>
        </div>
      </div>
      {data.status === 'completed' && (
        <div className="mt-2 text-xs text-green-600 dark:text-green-400 flex items-center">
          <CheckCircle className="w-3 h-3 mr-1" />
          Trained
        </div>
      )}
    </div>
  );
};

const OutputNode: React.FC<NodeProps<NodeData>> = ({ data, selected }) => {
  return (
    <div className={`px-4 py-3 rounded-lg border-2 ${
      selected ? 'border-orange-500 shadow-lg' : 'border-neutral-300 dark:border-neutral-600'
    } bg-gradient-to-r from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 min-w-[180px]`}>
      <Handle
        type="target"
        position={Position.Left}
        className="w-3 h-3 bg-orange-500 border-2 border-white dark:border-neutral-800"
      />
      <div className="flex items-center space-x-2">
        <BarChart3 className="w-5 h-5 text-orange-600 dark:text-orange-400" />
        <div>
          <div className="font-medium text-neutral-900 dark:text-neutral-100">{data.label}</div>
          <div className="text-xs text-neutral-600 dark:text-neutral-400">Output</div>
        </div>
      </div>
    </div>
  );
};

// Custom edge component
const CustomEdge: React.FC<EdgeProps> = ({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  style = {},
  markerEnd,
}) => {
  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  return (
    <>
      <path
        id={id}
        style={style}
        className="react-flow__edge-path stroke-2 stroke-neutral-400 dark:stroke-neutral-600"
        d={edgePath}
        markerEnd={markerEnd}
      />
      <circle
        cx={labelX}
        cy={labelY}
        r={10}
        className="fill-white dark:fill-neutral-800 stroke-neutral-400 dark:stroke-neutral-600"
        strokeWidth={2}
      >
        <animate
          attributeName="r"
          values="8;12;8"
          dur="2s"
          repeatCount="indefinite"
        />
      </circle>
    </>
  );
};

// Node types configuration
const nodeTypes: NodeTypes = {
  dataSource: DataSourceNode,
  transform: TransformNode,
  model: ModelNode,
  output: OutputNode,
};

const edgeTypes: EdgeTypes = {
  custom: CustomEdge,
};

// Component library
interface ComponentItem {
  id: string;
  type: string;
  label: string;
  icon: React.ReactNode;
  category: 'data' | 'transform' | 'model' | 'output';
  description: string;
}

const componentLibrary: ComponentItem[] = [
  // Data Sources
  { id: 'csv-reader', type: 'dataSource', label: 'CSV Reader', icon: <FileText className="w-5 h-5" />, category: 'data', description: 'Load data from CSV files' },
  { id: 'database', type: 'dataSource', label: 'Database', icon: <Database className="w-5 h-5" />, category: 'data', description: 'Connect to SQL databases' },
  { id: 'api-connector', type: 'dataSource', label: 'API Connector', icon: <Globe className="w-5 h-5" />, category: 'data', description: 'Fetch data from APIs' },
  { id: 's3-reader', type: 'dataSource', label: 'S3 Reader', icon: <Cloud className="w-5 h-5" />, category: 'data', description: 'Read from Amazon S3' },
  
  // Transformations
  { id: 'filter', type: 'transform', label: 'Filter', icon: <Filter className="w-5 h-5" />, category: 'transform', description: 'Filter rows by condition' },
  { id: 'aggregator', type: 'transform', label: 'Aggregator', icon: <Calculator className="w-5 h-5" />, category: 'transform', description: 'Group and aggregate data' },
  { id: 'joiner', type: 'transform', label: 'Joiner', icon: <GitBranch className="w-5 h-5" />, category: 'transform', description: 'Join multiple datasets' },
  { id: 'scaler', type: 'transform', label: 'Feature Scaler', icon: <Layers className="w-5 h-5" />, category: 'transform', description: 'Normalize features' },
  { id: 'encoder', type: 'transform', label: 'Encoder', icon: <Binary className="w-5 h-5" />, category: 'transform', description: 'Encode categorical variables' },
  { id: 'splitter', type: 'transform', label: 'Train/Test Split', icon: <Shuffle className="w-5 h-5" />, category: 'transform', description: 'Split data for training' },
  
  // Models
  { id: 'classification', type: 'model', label: 'Classifier', icon: <Target className="w-5 h-5" />, category: 'model', description: 'Classification models' },
  { id: 'regression', type: 'model', label: 'Regressor', icon: <Activity className="w-5 h-5" />, category: 'model', description: 'Regression models' },
  { id: 'clustering', type: 'model', label: 'Clustering', icon: <Cpu className="w-5 h-5" />, category: 'model', description: 'Clustering algorithms' },
  { id: 'neural-net', type: 'model', label: 'Neural Network', icon: <Brain className="w-5 h-5" />, category: 'model', description: 'Deep learning models' },
  { id: 'custom-model', type: 'model', label: 'Custom Model', icon: <Code className="w-5 h-5" />, category: 'model', description: 'Import custom model' },
  
  // Outputs
  { id: 'visualizer', type: 'output', label: 'Visualizer', icon: <BarChart3 className="w-5 h-5" />, category: 'output', description: 'Create visualizations' },
  { id: 'exporter', type: 'output', label: 'Data Exporter', icon: <Download className="w-5 h-5" />, category: 'output', description: 'Export results' },
  { id: 'model-saver', type: 'output', label: 'Model Saver', icon: <Save className="w-5 h-5" />, category: 'output', description: 'Save trained model' },
  { id: 'api-endpoint', type: 'output', label: 'API Endpoint', icon: <Share2 className="w-5 h-5" />, category: 'output', description: 'Deploy as API' },
];

interface WorkflowPipeline {
  id: string;
  name: string;
  description: string;
  nodes: Node<NodeData>[];
  edges: Edge[];
  created_at: Date;
  updated_at: Date;
  status: 'draft' | 'running' | 'completed' | 'error';
  execution_time?: number;
  output_summary?: any;
}

const VisualWorkflowBuilder: React.FC = () => {
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const [nodes, setNodes, onNodesChange] = useNodesState<NodeData>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [selectedCategory, setSelectedCategory] = useState<'data' | 'transform' | 'model' | 'output'>('data');
  const [workflowName, setWorkflowName] = useState('Untitled Workflow');
  const [isExecuting, setIsExecuting] = useState(false);
  const [showSaveModal, setShowSaveModal] = useState(false);
  const [showConfigPanel, setShowConfigPanel] = useState(false);
  const [selectedNode, setSelectedNode] = useState<Node<NodeData> | null>(null);
  const [executionProgress, setExecutionProgress] = useState(0);
  const [savedWorkflows, setSavedWorkflows] = useState<WorkflowPipeline[]>([]);

  // Initialize with sample workflow
  useEffect(() => {
    const sampleNodes: Node<NodeData>[] = [
      {
        id: '1',
        type: 'dataSource',
        position: { x: 100, y: 200 },
        data: { label: 'Customer Data', type: 'database', icon: <Database className="w-5 h-5" /> },
      },
      {
        id: '2',
        type: 'transform',
        position: { x: 350, y: 200 },
        data: { label: 'Clean & Scale', type: 'scaler', icon: <Layers className="w-5 h-5" /> },
      },
      {
        id: '3',
        type: 'model',
        position: { x: 600, y: 200 },
        data: { label: 'Random Forest', type: 'classification', icon: <Brain className="w-5 h-5" /> },
      },
      {
        id: '4',
        type: 'output',
        position: { x: 850, y: 200 },
        data: { label: 'Results Dashboard', type: 'visualizer', icon: <BarChart3 className="w-5 h-5" /> },
      },
    ];

    const sampleEdges: Edge[] = [
      { id: 'e1-2', source: '1', target: '2', type: 'custom', animated: true },
      { id: 'e2-3', source: '2', target: '3', type: 'custom', animated: true },
      { id: 'e3-4', source: '3', target: '4', type: 'custom', animated: true },
    ];

    setNodes(sampleNodes);
    setEdges(sampleEdges);
  }, []);

  const onConnect = useCallback(
    (params: Connection) => {
      setEdges((eds) => addEdge({ ...params, type: 'custom', animated: true }, eds));
    },
    [setEdges]
  );

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();

      const reactFlowBounds = reactFlowWrapper.current?.getBoundingClientRect();
      const type = event.dataTransfer.getData('application/reactflow');

      if (typeof type === 'undefined' || !type || !reactFlowBounds) {
        return;
      }

      const item = JSON.parse(type) as ComponentItem;
      const position = {
        x: event.clientX - reactFlowBounds.left,
        y: event.clientY - reactFlowBounds.top,
      };

      const newNode: Node<NodeData> = {
        id: `${item.id}-${nodes.length + 1}`,
        type: item.type,
        position,
        data: { 
          label: item.label, 
          type: item.id,
          icon: item.icon,
          status: 'idle'
        },
      };

      setNodes((nds) => nds.concat(newNode));
    },
    [nodes]
  );

  const onDragStart = (event: React.DragEvent, item: ComponentItem) => {
    event.dataTransfer.setData('application/reactflow', JSON.stringify(item));
    event.dataTransfer.effectAllowed = 'move';
  };

  const onNodeClick = useCallback((event: React.MouseEvent, node: Node<NodeData>) => {
    setSelectedNode(node);
    setShowConfigPanel(true);
  }, []);

  const executeWorkflow = async () => {
    setIsExecuting(true);
    setExecutionProgress(0);

    // Simulate workflow execution
    const totalSteps = nodes.length;
    for (let i = 0; i < totalSteps; i++) {
      await new Promise(resolve => setTimeout(resolve, 1000));
      setExecutionProgress((i + 1) / totalSteps * 100);
      
      // Update node status
      setNodes((nds) =>
        nds.map((node, index) => {
          if (index <= i) {
            return {
              ...node,
              data: {
                ...node.data,
                status: index === i ? 'running' : 'completed'
              }
            };
          }
          return node;
        })
      );
    }

    setIsExecuting(false);
  };

  const saveWorkflow = () => {
    const workflow: WorkflowPipeline = {
      id: `workflow-${Date.now()}`,
      name: workflowName,
      description: 'Visual ML pipeline',
      nodes,
      edges,
      created_at: new Date(),
      updated_at: new Date(),
      status: 'draft'
    };

    setSavedWorkflows([...savedWorkflows, workflow]);
    setShowSaveModal(false);
  };

  const exportWorkflow = () => {
    const workflow = {
      name: workflowName,
      nodes,
      edges,
      version: '1.0.0',
      created_at: new Date().toISOString()
    };

    const dataStr = JSON.stringify(workflow, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    
    const exportFileDefaultName = `${workflowName.replace(/\s+/g, '_').toLowerCase()}_workflow.json`;
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
  };

  const generateCode = () => {
    // Generate Python code from the workflow
    let code = `# Auto-generated ML Pipeline\n# ${workflowName}\n\n`;
    code += `import pandas as pd\nimport numpy as np\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler\n\n`;
    
    nodes.forEach((node, index) => {
      switch (node.data.type) {
        case 'csv-reader':
          code += `# Step ${index + 1}: Load data\ndata = pd.read_csv('data.csv')\n\n`;
          break;
        case 'scaler':
          code += `# Step ${index + 1}: Scale features\nscaler = StandardScaler()\ndata_scaled = scaler.fit_transform(data)\n\n`;
          break;
        case 'classification':
          code += `# Step ${index + 1}: Train classifier\nfrom sklearn.ensemble import RandomForestClassifier\nmodel = RandomForestClassifier(n_estimators=100)\nmodel.fit(X_train, y_train)\n\n`;
          break;
        case 'visualizer':
          code += `# Step ${index + 1}: Visualize results\nimport matplotlib.pyplot as plt\nplt.figure(figsize=(10, 6))\n# Add visualization code here\nplt.show()\n`;
          break;
      }
    });

    // Copy code to clipboard
    navigator.clipboard.writeText(code);
  };

  return (
    <div className="h-screen flex flex-col">
      {/* Header */}
      <div className="bg-white dark:bg-neutral-800 border-b border-neutral-200 dark:border-neutral-700 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-gradient-to-r from-indigo-500 to-purple-500 rounded-lg flex items-center justify-center">
                <GitBranch className="w-5 h-5 text-white" />
              </div>
              <input
                type="text"
                value={workflowName}
                onChange={(e) => setWorkflowName(e.target.value)}
                className="text-xl font-bold bg-transparent border-0 focus:outline-none text-neutral-900 dark:text-neutral-100"
              />
            </div>
            
            <div className="flex items-center space-x-2 text-sm text-neutral-500 dark:text-neutral-400">
              <span>Nodes: {nodes.length}</span>
              <span>•</span>
              <span>Edges: {edges.length}</span>
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            <Button variant="outline" size="sm" leftIcon={<Code className="w-4 h-4" />} onClick={generateCode}>
              Generate Code
            </Button>
            <Button variant="outline" size="sm" leftIcon={<Download className="w-4 h-4" />} onClick={exportWorkflow}>
              Export
            </Button>
            <Button variant="outline" size="sm" leftIcon={<Save className="w-4 h-4" />} onClick={() => setShowSaveModal(true)}>
              Save
            </Button>
            <Button 
              variant="primary" 
              size="sm" 
              leftIcon={isExecuting ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              onClick={executeWorkflow}
              disabled={isExecuting}
            >
              {isExecuting ? 'Running...' : 'Run Pipeline'}
            </Button>
          </div>
        </div>
        
        {/* Execution Progress */}
        {isExecuting && (
          <div className="mt-4">
            <div className="flex items-center justify-between text-sm text-neutral-600 dark:text-neutral-400 mb-1">
              <span>Executing pipeline...</span>
              <span>{Math.round(executionProgress)}%</span>
            </div>
            <div className="w-full bg-neutral-200 dark:bg-neutral-700 rounded-full h-2">
              <motion.div
                className="bg-gradient-to-r from-indigo-500 to-purple-500 h-2 rounded-full"
                initial={{ width: 0 }}
                animate={{ width: `${executionProgress}%` }}
                transition={{ duration: 0.3 }}
              />
            </div>
          </div>
        )}
      </div>

      <div className="flex-1 flex">
        {/* Component Library Sidebar */}
        <div className="w-64 bg-neutral-50 dark:bg-neutral-900 border-r border-neutral-200 dark:border-neutral-700 p-4 overflow-y-auto">
          <h3 className="font-semibold text-neutral-900 dark:text-neutral-100 mb-4">Components</h3>
          
          {/* Category Tabs */}
          <div className="flex space-x-1 mb-4 bg-neutral-200 dark:bg-neutral-800 rounded-lg p-1">
            {(['data', 'transform', 'model', 'output'] as const).map((category) => (
              <button
                key={category}
                onClick={() => setSelectedCategory(category)}
                className={`flex-1 px-2 py-1 rounded text-xs font-medium capitalize transition-colors ${
                  selectedCategory === category
                    ? 'bg-white dark:bg-neutral-700 text-primary-600 dark:text-primary-400 shadow-sm'
                    : 'text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-neutral-100'
                }`}
              >
                {category}
              </button>
            ))}
          </div>
          
          {/* Component List */}
          <div className="space-y-2">
            {componentLibrary
              .filter(item => item.category === selectedCategory)
              .map((item) => (
                <motion.div
                  key={item.id}
                  draggable
                  onDragStart={(e) => onDragStart(e as any, item)}
                  whileHover={{ scale: 1.02 }}
                  whileDrag={{ scale: 1.05, opacity: 0.8 }}
                  className="p-3 bg-white dark:bg-neutral-800 rounded-lg border border-neutral-200 dark:border-neutral-700 cursor-move hover:shadow-md transition-shadow"
                >
                  <div className="flex items-center space-x-2 mb-1">
                    <div className={`p-1.5 rounded ${
                      item.category === 'data' ? 'bg-blue-100 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400' :
                      item.category === 'transform' ? 'bg-purple-100 dark:bg-purple-900/20 text-purple-600 dark:text-purple-400' :
                      item.category === 'model' ? 'bg-green-100 dark:bg-green-900/20 text-green-600 dark:text-green-400' :
                      'bg-orange-100 dark:bg-orange-900/20 text-orange-600 dark:text-orange-400'
                    }`}>
                      {item.icon}
                    </div>
                    <span className="font-medium text-sm text-neutral-900 dark:text-neutral-100">
                      {item.label}
                    </span>
                  </div>
                  <p className="text-xs text-neutral-600 dark:text-neutral-400">
                    {item.description}
                  </p>
                </motion.div>
              ))}
          </div>
        </div>

        {/* React Flow Canvas */}
        <div className="flex-1" ref={reactFlowWrapper}>
          <ReactFlowProvider>
            <ReactFlow
              nodes={nodes}
              edges={edges}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onConnect={onConnect}
              onNodeClick={onNodeClick}
              onDrop={onDrop}
              onDragOver={onDragOver}
              nodeTypes={nodeTypes}
              edgeTypes={edgeTypes}
              connectionMode={ConnectionMode.Loose}
              defaultEdgeOptions={{
                type: 'custom',
                animated: true,
                markerEnd: {
                  type: MarkerType.ArrowClosed,
                  width: 20,
                  height: 20,
                },
              }}
              fitView
            >
              <Background variant="dots" gap={12} size={1} />
              <Controls />
              <MiniMap 
                nodeColor={(node) => {
                  switch (node.type) {
                    case 'dataSource': return '#3B82F6';
                    case 'transform': return '#8B5CF6';
                    case 'model': return '#10B981';
                    case 'output': return '#F97316';
                    default: return '#6B7280';
                  }
                }}
                style={{
                  backgroundColor: 'rgba(0, 0, 0, 0.1)',
                  border: '1px solid rgba(0, 0, 0, 0.2)',
                }}
              />
            </ReactFlow>
          </ReactFlowProvider>
        </div>

        {/* Configuration Panel */}
        <AnimatePresence>
          {showConfigPanel && selectedNode && (
            <motion.div
              initial={{ x: 300 }}
              animate={{ x: 0 }}
              exit={{ x: 300 }}
              className="w-80 bg-white dark:bg-neutral-800 border-l border-neutral-200 dark:border-neutral-700 p-4 overflow-y-auto"
            >
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold text-neutral-900 dark:text-neutral-100">
                  Configure {selectedNode.data.label}
                </h3>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowConfigPanel(false)}
                >
                  <X className="w-4 h-4" />
                </Button>
              </div>
              
              {/* Configuration options based on node type */}
              <div className="space-y-4">
                {selectedNode.data.type === 'database' && (
                  <>
                    <div>
                      <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                        Connection
                      </label>
                      <select className="w-full p-2 border border-neutral-200 dark:border-neutral-700 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100">
                        <option>PostgreSQL - Analytics DB</option>
                        <option>MySQL - Customer DB</option>
                        <option>MongoDB - Logs DB</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                        Table/Query
                      </label>
                      <textarea
                        rows={3}
                        className="w-full p-2 border border-neutral-200 dark:border-neutral-700 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
                        placeholder="SELECT * FROM customers"
                      />
                    </div>
                  </>
                )}
                
                {selectedNode.data.type === 'classification' && (
                  <>
                    <div>
                      <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                        Algorithm
                      </label>
                      <select className="w-full p-2 border border-neutral-200 dark:border-neutral-700 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100">
                        <option>Random Forest</option>
                        <option>Gradient Boosting</option>
                        <option>SVM</option>
                        <option>Neural Network</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                        Target Column
                      </label>
                      <input
                        type="text"
                        className="w-full p-2 border border-neutral-200 dark:border-neutral-700 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
                        placeholder="target"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                        Training Split
                      </label>
                      <input
                        type="range"
                        min="60"
                        max="90"
                        defaultValue="80"
                        className="w-full"
                      />
                      <span className="text-xs text-neutral-500">80%</span>
                    </div>
                  </>
                )}
                
                <div className="pt-4 border-t border-neutral-200 dark:border-neutral-700">
                  <Button variant="primary" size="sm" className="w-full">
                    Apply Configuration
                  </Button>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Save Modal */}
      <AnimatePresence>
        {showSaveModal && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
            onClick={() => setShowSaveModal(false)}
          >
            <motion.div
              initial={{ scale: 0.9 }}
              animate={{ scale: 1 }}
              exit={{ scale: 0.9 }}
              className="bg-white dark:bg-neutral-800 rounded-lg p-6 w-full max-w-md mx-4"
              onClick={(e) => e.stopPropagation()}
            >
              <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4">
                Save Workflow
              </h3>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                    Workflow Name
                  </label>
                  <input
                    type="text"
                    value={workflowName}
                    onChange={(e) => setWorkflowName(e.target.value)}
                    className="w-full p-3 border border-neutral-200 dark:border-neutral-700 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                    Description
                  </label>
                  <textarea
                    rows={3}
                    placeholder="Describe what this workflow does..."
                    className="w-full p-3 border border-neutral-200 dark:border-neutral-700 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100 resize-none"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                    Tags
                  </label>
                  <input
                    type="text"
                    placeholder="Add tags separated by commas"
                    className="w-full p-3 border border-neutral-200 dark:border-neutral-700 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
                  />
                </div>
              </div>
              
              <div className="flex justify-end space-x-3 mt-6">
                <Button variant="outline" onClick={() => setShowSaveModal(false)}>
                  Cancel
                </Button>
                <Button variant="primary" onClick={saveWorkflow}>
                  Save Workflow
                </Button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default VisualWorkflowBuilder;