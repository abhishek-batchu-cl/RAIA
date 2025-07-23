import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Database, Cloud, HardDrive, Globe, FileText, 
  Plus, Settings, TestTube, Eye, Play, Pause,
  CheckCircle, AlertTriangle, X, RefreshCw,
  Upload, Download, Filter, Search, Key,
  Server, Wifi, Shield, Zap, Activity
} from 'lucide-react';
import Card from '@/components/common/Card';
import Button from '@/components/common/Button';
import { apiClient } from '@/services/api';

interface DataSource {
  id: string;
  name: string;
  type: DataSourceType;
  status: 'connected' | 'disconnected' | 'error' | 'testing';
  connection_config: Record<string, any>;
  last_sync: Date;
  data_preview?: any[];
  schema?: SchemaField[];
  created_at: Date;
  updated_at: Date;
}

interface DataSourceType {
  id: string;
  name: string;
  category: 'database' | 'cloud' | 'file' | 'api' | 'streaming';
  icon: React.ReactNode;
  color: string;
  description: string;
  connection_fields: ConnectionField[];
  supports_streaming: boolean;
  supports_batch: boolean;
}

interface ConnectionField {
  name: string;
  label: string;
  type: 'text' | 'password' | 'number' | 'select' | 'boolean' | 'file';
  required: boolean;
  options?: string[];
  placeholder?: string;
  help_text?: string;
}

interface SchemaField {
  name: string;
  type: string;
  nullable: boolean;
  description?: string;
}

const DataSourceConnectors: React.FC = () => {
  const [dataSources, setDataSources] = useState<DataSource[]>([]);
  const [availableConnectors, setAvailableConnectors] = useState<DataSourceType[]>([]);
  const [showAddConnector, setShowAddConnector] = useState(false);
  const [selectedConnectorType, setSelectedConnectorType] = useState<DataSourceType | null>(null);
  const [isTestingConnection, setIsTestingConnection] = useState(false);
  const [connectionForm, setConnectionForm] = useState<Record<string, any>>({});
  const [previewData, setPreviewData] = useState<any[] | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterCategory, setFilterCategory] = useState<string>('all');

  // Available connector types
  const connectorTypes: DataSourceType[] = [
    {
      id: 'postgresql',
      name: 'PostgreSQL',
      category: 'database',
      icon: <Database className="w-6 h-6" />,
      color: 'bg-blue-500 text-white',
      description: 'Connect to PostgreSQL database',
      supports_streaming: false,
      supports_batch: true,
      connection_fields: [
        { name: 'host', label: 'Host', type: 'text', required: true, placeholder: 'localhost' },
        { name: 'port', label: 'Port', type: 'number', required: true, placeholder: '5432' },
        { name: 'database', label: 'Database', type: 'text', required: true },
        { name: 'username', label: 'Username', type: 'text', required: true },
        { name: 'password', label: 'Password', type: 'password', required: true },
        { name: 'ssl', label: 'Use SSL', type: 'boolean', required: false }
      ]
    },
    {
      id: 'mysql',
      name: 'MySQL',
      category: 'database',
      icon: <Database className="w-6 h-6" />,
      color: 'bg-orange-500 text-white',
      description: 'Connect to MySQL database',
      supports_streaming: false,
      supports_batch: true,
      connection_fields: [
        { name: 'host', label: 'Host', type: 'text', required: true, placeholder: 'localhost' },
        { name: 'port', label: 'Port', type: 'number', required: true, placeholder: '3306' },
        { name: 'database', label: 'Database', type: 'text', required: true },
        { name: 'username', label: 'Username', type: 'text', required: true },
        { name: 'password', label: 'Password', type: 'password', required: true }
      ]
    },
    {
      id: 'snowflake',
      name: 'Snowflake',
      category: 'cloud',
      icon: <Cloud className="w-6 h-6" />,
      color: 'bg-blue-600 text-white',
      description: 'Connect to Snowflake data warehouse',
      supports_streaming: false,
      supports_batch: true,
      connection_fields: [
        { name: 'account', label: 'Account', type: 'text', required: true, help_text: 'Your Snowflake account identifier' },
        { name: 'username', label: 'Username', type: 'text', required: true },
        { name: 'password', label: 'Password', type: 'password', required: true },
        { name: 'database', label: 'Database', type: 'text', required: true },
        { name: 'warehouse', label: 'Warehouse', type: 'text', required: true },
        { name: 'role', label: 'Role', type: 'text', required: false }
      ]
    },
    {
      id: 'bigquery',
      name: 'Google BigQuery',
      category: 'cloud',
      icon: <Cloud className="w-6 h-6" />,
      color: 'bg-green-500 text-white',
      description: 'Connect to Google BigQuery',
      supports_streaming: false,
      supports_batch: true,
      connection_fields: [
        { name: 'project_id', label: 'Project ID', type: 'text', required: true },
        { name: 'credentials_json', label: 'Service Account JSON', type: 'file', required: true, help_text: 'Upload your service account key file' },
        { name: 'dataset', label: 'Dataset', type: 'text', required: false }
      ]
    },
    {
      id: 'redshift',
      name: 'Amazon Redshift',
      category: 'cloud',
      icon: <Cloud className="w-6 h-6" />,
      color: 'bg-yellow-600 text-white',
      description: 'Connect to Amazon Redshift',
      supports_streaming: false,
      supports_batch: true,
      connection_fields: [
        { name: 'host', label: 'Host', type: 'text', required: true },
        { name: 'port', label: 'Port', type: 'number', required: true, placeholder: '5439' },
        { name: 'database', label: 'Database', type: 'text', required: true },
        { name: 'username', label: 'Username', type: 'text', required: true },
        { name: 'password', label: 'Password', type: 'password', required: true }
      ]
    },
    {
      id: 'mongodb',
      name: 'MongoDB',
      category: 'database',
      icon: <Database className="w-6 h-6" />,
      color: 'bg-green-600 text-white',
      description: 'Connect to MongoDB database',
      supports_streaming: true,
      supports_batch: true,
      connection_fields: [
        { name: 'connection_string', label: 'Connection String', type: 'text', required: true, placeholder: 'mongodb://localhost:27017' },
        { name: 'database', label: 'Database', type: 'text', required: true },
        { name: 'collection', label: 'Collection', type: 'text', required: false }
      ]
    },
    {
      id: 's3',
      name: 'Amazon S3',
      category: 'cloud',
      icon: <HardDrive className="w-6 h-6" />,
      color: 'bg-orange-600 text-white',
      description: 'Connect to Amazon S3 bucket',
      supports_streaming: false,
      supports_batch: true,
      connection_fields: [
        { name: 'access_key_id', label: 'Access Key ID', type: 'text', required: true },
        { name: 'secret_access_key', label: 'Secret Access Key', type: 'password', required: true },
        { name: 'bucket_name', label: 'Bucket Name', type: 'text', required: true },
        { name: 'region', label: 'Region', type: 'select', required: true, options: ['us-east-1', 'us-west-2', 'eu-west-1'] },
        { name: 'prefix', label: 'Prefix', type: 'text', required: false }
      ]
    },
    {
      id: 'csv',
      name: 'CSV Files',
      category: 'file',
      icon: <FileText className="w-6 h-6" />,
      color: 'bg-purple-500 text-white',
      description: 'Upload and process CSV files',
      supports_streaming: false,
      supports_batch: true,
      connection_fields: [
        { name: 'file', label: 'CSV File', type: 'file', required: true },
        { name: 'delimiter', label: 'Delimiter', type: 'select', required: false, options: [',', ';', '\t', '|'], placeholder: 'Auto-detect' },
        { name: 'has_header', label: 'Has Header Row', type: 'boolean', required: false }
      ]
    },
    {
      id: 'api',
      name: 'REST API',
      category: 'api',
      icon: <Globe className="w-6 h-6" />,
      color: 'bg-indigo-500 text-white',
      description: 'Connect to REST API endpoints',
      supports_streaming: false,
      supports_batch: true,
      connection_fields: [
        { name: 'base_url', label: 'Base URL', type: 'text', required: true, placeholder: 'https://api.example.com' },
        { name: 'auth_type', label: 'Authentication', type: 'select', required: true, options: ['none', 'bearer', 'api_key', 'basic'] },
        { name: 'auth_value', label: 'Auth Value', type: 'password', required: false },
        { name: 'headers', label: 'Custom Headers (JSON)', type: 'text', required: false }
      ]
    },
    {
      id: 'kafka',
      name: 'Apache Kafka',
      category: 'streaming',
      icon: <Activity className="w-6 h-6" />,
      color: 'bg-red-600 text-white',
      description: 'Connect to Kafka streaming platform',
      supports_streaming: true,
      supports_batch: false,
      connection_fields: [
        { name: 'bootstrap_servers', label: 'Bootstrap Servers', type: 'text', required: true, placeholder: 'localhost:9092' },
        { name: 'topic', label: 'Topic', type: 'text', required: true },
        { name: 'group_id', label: 'Consumer Group ID', type: 'text', required: false },
        { name: 'security_protocol', label: 'Security Protocol', type: 'select', required: false, options: ['PLAINTEXT', 'SSL', 'SASL_PLAINTEXT', 'SASL_SSL'] }
      ]
    }
  ];

  // Mock existing data sources
  const mockDataSources: DataSource[] = [
    {
      id: 'ds-1',
      name: 'Customer Database',
      type: connectorTypes.find(c => c.id === 'postgresql')!,
      status: 'connected',
      connection_config: { host: 'prod-db.company.com', database: 'customers' },
      last_sync: new Date(Date.now() - 2 * 60 * 60 * 1000),
      created_at: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
      updated_at: new Date(Date.now() - 2 * 60 * 60 * 1000),
      schema: [
        { name: 'customer_id', type: 'integer', nullable: false },
        { name: 'name', type: 'varchar', nullable: false },
        { name: 'email', type: 'varchar', nullable: false },
        { name: 'age', type: 'integer', nullable: true },
        { name: 'created_at', type: 'timestamp', nullable: false }
      ]
    },
    {
      id: 'ds-2',
      name: 'Transaction Data Warehouse',
      type: connectorTypes.find(c => c.id === 'snowflake')!,
      status: 'connected',
      connection_config: { account: 'company.snowflakecomputing.com', database: 'ANALYTICS' },
      last_sync: new Date(Date.now() - 30 * 60 * 1000),
      created_at: new Date(Date.now() - 15 * 24 * 60 * 60 * 1000),
      updated_at: new Date(Date.now() - 30 * 60 * 1000)
    },
    {
      id: 'ds-3',
      name: 'ML Training Data',
      type: connectorTypes.find(c => c.id === 's3')!,
      status: 'error',
      connection_config: { bucket_name: 'ml-training-data', region: 'us-east-1' },
      last_sync: new Date(Date.now() - 24 * 60 * 60 * 1000),
      created_at: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000),
      updated_at: new Date(Date.now() - 24 * 60 * 60 * 1000)
    }
  ];

  useEffect(() => {
    loadDataSources();
    setAvailableConnectors(connectorTypes);
  }, []);

  const loadDataSources = async () => {
    try {
      // In production, this would call the API
      setDataSources(mockDataSources);
    } catch (error) {
      console.error('Error loading data sources:', error);
      setDataSources(mockDataSources);
    }
  };

  const testConnection = async () => {
    if (!selectedConnectorType) return;

    setIsTestingConnection(true);
    
    try {
      // Simulate connection test
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Mock success response with schema preview
      const mockSchema = [
        { name: 'id', type: 'integer', nullable: false },
        { name: 'name', type: 'varchar', nullable: false },
        { name: 'value', type: 'decimal', nullable: true },
        { name: 'timestamp', type: 'timestamp', nullable: false }
      ];
      
      const mockPreview = [
        { id: 1, name: 'Sample 1', value: 123.45, timestamp: '2024-01-15T10:30:00Z' },
        { id: 2, name: 'Sample 2', value: 678.90, timestamp: '2024-01-15T11:15:00Z' },
        { id: 3, name: 'Sample 3', value: null, timestamp: '2024-01-15T12:00:00Z' }
      ];
      
      setPreviewData(mockPreview);
      
    } catch (error) {
      console.error('Connection test failed:', error);
    } finally {
      setIsTestingConnection(false);
    }
  };

  const createDataSource = async () => {
    if (!selectedConnectorType || !connectionForm.name) return;

    const newDataSource: DataSource = {
      id: `ds-${Date.now()}`,
      name: connectionForm.name,
      type: selectedConnectorType,
      status: 'connected',
      connection_config: { ...connectionForm },
      last_sync: new Date(),
      created_at: new Date(),
      updated_at: new Date(),
      schema: previewData ? [
        { name: 'id', type: 'integer', nullable: false },
        { name: 'name', type: 'varchar', nullable: false },
        { name: 'value', type: 'decimal', nullable: true },
        { name: 'timestamp', type: 'timestamp', nullable: false }
      ] : undefined
    };

    setDataSources(prev => [newDataSource, ...prev]);
    setShowAddConnector(false);
    setSelectedConnectorType(null);
    setConnectionForm({});
    setPreviewData(null);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'connected': return 'text-green-500';
      case 'disconnected': return 'text-gray-500';
      case 'error': return 'text-red-500';
      case 'testing': return 'text-blue-500';
      default: return 'text-gray-500';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'connected': return <CheckCircle className="w-4 h-4" />;
      case 'disconnected': return <Pause className="w-4 h-4" />;
      case 'error': return <AlertTriangle className="w-4 h-4" />;
      case 'testing': return <RefreshCw className="w-4 h-4 animate-spin" />;
      default: return <Pause className="w-4 h-4" />;
    }
  };

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'database': return <Database className="w-4 h-4" />;
      case 'cloud': return <Cloud className="w-4 h-4" />;
      case 'file': return <FileText className="w-4 h-4" />;
      case 'api': return <Globe className="w-4 h-4" />;
      case 'streaming': return <Activity className="w-4 h-4" />;
      default: return <Database className="w-4 h-4" />;
    }
  };

  const formatTimeAgo = (date: Date) => {
    const now = new Date();
    const diffInMinutes = Math.floor((now.getTime() - date.getTime()) / (1000 * 60));
    
    if (diffInMinutes < 1) return 'Just now';
    if (diffInMinutes < 60) return `${diffInMinutes}m ago`;
    if (diffInMinutes < 1440) return `${Math.floor(diffInMinutes / 60)}h ago`;
    return `${Math.floor(diffInMinutes / 1440)}d ago`;
  };

  const filteredConnectors = availableConnectors.filter(connector => {
    if (filterCategory !== 'all' && connector.category !== filterCategory) return false;
    if (searchTerm && !connector.name.toLowerCase().includes(searchTerm.toLowerCase())) return false;
    return true;
  });

  const renderConnectionField = (field: ConnectionField) => {
    const value = connectionForm[field.name] || '';
    
    return (
      <div key={field.name} className="space-y-2">
        <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300">
          {field.label} {field.required && <span className="text-red-500">*</span>}
        </label>
        
        {field.type === 'text' && (
          <input
            type="text"
            value={value}
            onChange={(e) => setConnectionForm(prev => ({ ...prev, [field.name]: e.target.value }))}
            placeholder={field.placeholder}
            className="w-full p-3 border border-neutral-200 dark:border-neutral-700 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
            required={field.required}
          />
        )}
        
        {field.type === 'password' && (
          <input
            type="password"
            value={value}
            onChange={(e) => setConnectionForm(prev => ({ ...prev, [field.name]: e.target.value }))}
            placeholder={field.placeholder}
            className="w-full p-3 border border-neutral-200 dark:border-neutral-700 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
            required={field.required}
          />
        )}
        
        {field.type === 'number' && (
          <input
            type="number"
            value={value}
            onChange={(e) => setConnectionForm(prev => ({ ...prev, [field.name]: parseInt(e.target.value) }))}
            placeholder={field.placeholder}
            className="w-full p-3 border border-neutral-200 dark:border-neutral-700 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
            required={field.required}
          />
        )}
        
        {field.type === 'select' && (
          <select
            value={value}
            onChange={(e) => setConnectionForm(prev => ({ ...prev, [field.name]: e.target.value }))}
            className="w-full p-3 border border-neutral-200 dark:border-neutral-700 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
            required={field.required}
          >
            <option value="">Select {field.label}</option>
            {field.options?.map(option => (
              <option key={option} value={option}>{option}</option>
            ))}
          </select>
        )}
        
        {field.type === 'boolean' && (
          <label className="flex items-center space-x-2">
            <input
              type="checkbox"
              checked={!!value}
              onChange={(e) => setConnectionForm(prev => ({ ...prev, [field.name]: e.target.checked }))}
              className="rounded border-gray-300"
            />
            <span className="text-sm text-neutral-600 dark:text-neutral-400">Enable {field.label}</span>
          </label>
        )}
        
        {field.type === 'file' && (
          <div className="border-2 border-dashed border-neutral-300 dark:border-neutral-600 rounded-lg p-4 text-center">
            <Upload className="w-8 h-8 text-neutral-400 mx-auto mb-2" />
            <p className="text-sm text-neutral-600 dark:text-neutral-400 mb-2">
              Click to upload or drag and drop
            </p>
            <input
              type="file"
              onChange={(e) => setConnectionForm(prev => ({ ...prev, [field.name]: e.target.files?.[0] }))}
              className="hidden"
              id={`file-${field.name}`}
            />
            <Button
              variant="outline"
              size="sm"
              onClick={() => document.getElementById(`file-${field.name}`)?.click()}
            >
              Choose File
            </Button>
          </div>
        )}
        
        {field.help_text && (
          <p className="text-xs text-neutral-500 dark:text-neutral-400">{field.help_text}</p>
        )}
      </div>
    );
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-neutral-900 dark:text-neutral-100 flex items-center">
            <div className="w-8 h-8 bg-gradient-to-r from-green-500 to-blue-500 rounded-lg flex items-center justify-center mr-3">
              <Database className="w-5 h-5 text-white" />
            </div>
            Data Source Connectors
          </h1>
          <p className="text-neutral-600 dark:text-neutral-400 mt-1">
            Connect to databases, cloud services, APIs, and streaming platforms
          </p>
        </div>
        
        <Button
          variant="primary"
          leftIcon={<Plus className="w-4 h-4" />}
          onClick={() => setShowAddConnector(true)}
        >
          Add Connector
        </Button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-green-700 dark:text-green-300">Total Connectors</p>
              <p className="text-2xl font-bold text-green-900 dark:text-green-100">
                {dataSources.length}
              </p>
            </div>
            <Database className="w-8 h-8 text-green-500" />
          </div>
        </Card>

        <Card className="bg-gradient-to-r from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-blue-700 dark:text-blue-300">Connected</p>
              <p className="text-2xl font-bold text-blue-900 dark:text-blue-100">
                {dataSources.filter(ds => ds.status === 'connected').length}
              </p>
            </div>
            <CheckCircle className="w-8 h-8 text-blue-500" />
          </div>
        </Card>

        <Card className="bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-yellow-700 dark:text-yellow-300">With Errors</p>
              <p className="text-2xl font-bold text-yellow-900 dark:text-yellow-100">
                {dataSources.filter(ds => ds.status === 'error').length}
              </p>
            </div>
            <AlertTriangle className="w-8 h-8 text-yellow-500" />
          </div>
        </Card>

        <Card className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-purple-700 dark:text-purple-300">Data Sources</p>
              <p className="text-2xl font-bold text-purple-900 dark:text-purple-100">
                {availableConnectors.length}
              </p>
            </div>
            <Server className="w-8 h-8 text-purple-500" />
          </div>
        </Card>
      </div>

      {/* Existing Data Sources */}
      <Card title="Connected Data Sources" icon={<Wifi className="w-5 h-5 text-primary-500" />}>
        <div className="space-y-4">
          {dataSources.map((dataSource) => (
            <motion.div
              key={dataSource.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex items-center justify-between p-4 bg-neutral-50 dark:bg-neutral-800 rounded-lg hover:shadow-md transition-shadow duration-200"
            >
              <div className="flex items-center space-x-4">
                <div className={`p-3 rounded-lg ${dataSource.type.color}`}>
                  {dataSource.type.icon}
                </div>
                
                <div>
                  <div className="flex items-center space-x-2 mb-1">
                    <h3 className="font-semibold text-neutral-900 dark:text-neutral-100">
                      {dataSource.name}
                    </h3>
                    <span className={`flex items-center space-x-1 ${getStatusColor(dataSource.status)}`}>
                      {getStatusIcon(dataSource.status)}
                      <span className="text-sm capitalize">{dataSource.status}</span>
                    </span>
                  </div>
                  
                  <div className="flex items-center space-x-4 text-sm text-neutral-600 dark:text-neutral-400">
                    <span className="flex items-center space-x-1">
                      {getCategoryIcon(dataSource.type.category)}
                      <span>{dataSource.type.name}</span>
                    </span>
                    <span>Last sync: {formatTimeAgo(dataSource.last_sync)}</span>
                    {dataSource.schema && (
                      <span>{dataSource.schema.length} fields</span>
                    )}
                  </div>
                </div>
              </div>
              
              <div className="flex items-center space-x-2">
                <Button variant="ghost" size="sm" leftIcon={<Eye className="w-4 h-4" />}>
                  View
                </Button>
                <Button variant="ghost" size="sm" leftIcon={<TestTube className="w-4 h-4" />}>
                  Test
                </Button>
                <Button variant="ghost" size="sm" leftIcon={<Settings className="w-4 h-4" />}>
                  Configure
                </Button>
                <Button variant="ghost" size="sm" leftIcon={<RefreshCw className="w-4 h-4" />}>
                  Sync
                </Button>
              </div>
            </motion.div>
          ))}
          
          {dataSources.length === 0 && (
            <div className="text-center py-12">
              <Database className="w-16 h-16 text-neutral-400 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-neutral-900 dark:text-neutral-100 mb-2">
                No Data Sources Connected
              </h3>
              <p className="text-neutral-600 dark:text-neutral-400 mb-6">
                Connect your first data source to start analyzing your data
              </p>
              <Button
                variant="primary"
                leftIcon={<Plus className="w-4 h-4" />}
                onClick={() => setShowAddConnector(true)}
              >
                Add Data Source
              </Button>
            </div>
          )}
        </div>
      </Card>

      {/* Add Connector Modal */}
      <AnimatePresence>
        {showAddConnector && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
            onClick={() => setShowAddConnector(false)}
          >
            <motion.div
              initial={{ scale: 0.9 }}
              animate={{ scale: 1 }}
              exit={{ scale: 0.9 }}
              className="bg-white dark:bg-neutral-800 rounded-lg w-full max-w-4xl mx-4 max-h-[90vh] overflow-hidden"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between p-6 border-b border-neutral-200 dark:border-neutral-700">
                <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
                  {selectedConnectorType ? `Configure ${selectedConnectorType.name}` : 'Add Data Source Connector'}
                </h3>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => {
                    setShowAddConnector(false);
                    setSelectedConnectorType(null);
                    setConnectionForm({});
                    setPreviewData(null);
                  }}
                >
                  <X className="w-4 h-4" />
                </Button>
              </div>
              
              <div className="overflow-y-auto" style={{ maxHeight: 'calc(90vh - 120px)' }}>
                {!selectedConnectorType ? (
                  // Connector Selection
                  <div className="p-6 space-y-6">
                    {/* Search and Filter */}
                    <div className="flex items-center space-x-4">
                      <div className="flex-1 relative">
                        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-neutral-400" />
                        <input
                          type="text"
                          placeholder="Search connectors..."
                          value={searchTerm}
                          onChange={(e) => setSearchTerm(e.target.value)}
                          className="pl-10 pr-4 py-2 w-full border border-neutral-200 dark:border-neutral-700 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
                        />
                      </div>
                      <select
                        value={filterCategory}
                        onChange={(e) => setFilterCategory(e.target.value)}
                        className="px-3 py-2 border border-neutral-200 dark:border-neutral-700 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
                      >
                        <option value="all">All Categories</option>
                        <option value="database">Databases</option>
                        <option value="cloud">Cloud Services</option>
                        <option value="file">File Sources</option>
                        <option value="api">APIs</option>
                        <option value="streaming">Streaming</option>
                      </select>
                    </div>

                    {/* Connector Grid */}
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                      {filteredConnectors.map((connector) => (
                        <motion.div
                          key={connector.id}
                          whileHover={{ scale: 1.02 }}
                          whileTap={{ scale: 0.98 }}
                          onClick={() => setSelectedConnectorType(connector)}
                          className="p-4 border border-neutral-200 dark:border-neutral-700 rounded-lg hover:shadow-md transition-all duration-200 cursor-pointer"
                        >
                          <div className="flex items-start space-x-3">
                            <div className={`p-2 rounded-lg ${connector.color}`}>
                              {connector.icon}
                            </div>
                            <div className="flex-1">
                              <h4 className="font-semibold text-neutral-900 dark:text-neutral-100 mb-1">
                                {connector.name}
                              </h4>
                              <p className="text-sm text-neutral-600 dark:text-neutral-400 mb-2">
                                {connector.description}
                              </p>
                              <div className="flex items-center space-x-2 text-xs">
                                <span className={`px-2 py-1 rounded-full ${
                                  connector.category === 'database' ? 'bg-blue-100 text-blue-800' :
                                  connector.category === 'cloud' ? 'bg-green-100 text-green-800' :
                                  connector.category === 'file' ? 'bg-purple-100 text-purple-800' :
                                  connector.category === 'api' ? 'bg-orange-100 text-orange-800' :
                                  'bg-red-100 text-red-800'
                                }`}>
                                  {connector.category}
                                </span>
                                {connector.supports_streaming && (
                                  <span className="px-2 py-1 bg-gray-100 text-gray-800 rounded-full">
                                    streaming
                                  </span>
                                )}
                              </div>
                            </div>
                          </div>
                        </motion.div>
                      ))}
                    </div>
                  </div>
                ) : (
                  // Configuration Form
                  <div className="p-6 space-y-6">
                    <div className="flex items-center space-x-3 p-4 bg-neutral-50 dark:bg-neutral-700 rounded-lg">
                      <div className={`p-2 rounded-lg ${selectedConnectorType.color}`}>
                        {selectedConnectorType.icon}
                      </div>
                      <div>
                        <h4 className="font-semibold text-neutral-900 dark:text-neutral-100">
                          {selectedConnectorType.name}
                        </h4>
                        <p className="text-sm text-neutral-600 dark:text-neutral-400">
                          {selectedConnectorType.description}
                        </p>
                      </div>
                    </div>

                    {/* Connection Name */}
                    <div className="space-y-2">
                      <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300">
                        Connection Name <span className="text-red-500">*</span>
                      </label>
                      <input
                        type="text"
                        value={connectionForm.name || ''}
                        onChange={(e) => setConnectionForm(prev => ({ ...prev, name: e.target.value }))}
                        placeholder="My Data Source"
                        className="w-full p-3 border border-neutral-200 dark:border-neutral-700 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
                        required
                      />
                    </div>

                    {/* Connection Fields */}
                    <div className="space-y-4">
                      {selectedConnectorType.connection_fields.map(renderConnectionField)}
                    </div>

                    {/* Data Preview */}
                    {previewData && (
                      <div className="space-y-3">
                        <h4 className="font-medium text-neutral-900 dark:text-neutral-100">Data Preview</h4>
                        <div className="overflow-x-auto">
                          <table className="w-full border border-neutral-200 dark:border-neutral-700 rounded-lg">
                            <thead className="bg-neutral-50 dark:bg-neutral-700">
                              <tr>
                                {Object.keys(previewData[0] || {}).map((key) => (
                                  <th key={key} className="px-4 py-2 text-left text-sm font-medium text-neutral-900 dark:text-neutral-100">
                                    {key}
                                  </th>
                                ))}
                              </tr>
                            </thead>
                            <tbody>
                              {previewData.slice(0, 3).map((row, index) => (
                                <tr key={index} className="border-t border-neutral-200 dark:border-neutral-700">
                                  {Object.values(row).map((value: any, cellIndex) => (
                                    <td key={cellIndex} className="px-4 py-2 text-sm text-neutral-600 dark:text-neutral-400">
                                      {value === null ? '(null)' : String(value)}
                                    </td>
                                  ))}
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    )}

                    {/* Action Buttons */}
                    <div className="flex justify-between pt-4 border-t border-neutral-200 dark:border-neutral-700">
                      <Button
                        variant="outline"
                        onClick={() => {
                          setSelectedConnectorType(null);
                          setConnectionForm({});
                          setPreviewData(null);
                        }}
                      >
                        Back
                      </Button>
                      <div className="flex space-x-3">
                        <Button
                          variant="outline"
                          leftIcon={isTestingConnection ? <RefreshCw className="w-4 h-4 animate-spin" /> : <TestTube className="w-4 h-4" />}
                          onClick={testConnection}
                          disabled={isTestingConnection}
                        >
                          {isTestingConnection ? 'Testing...' : 'Test Connection'}
                        </Button>
                        <Button
                          variant="primary"
                          onClick={createDataSource}
                          disabled={!connectionForm.name || !previewData}
                        >
                          Create Connection
                        </Button>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default DataSourceConnectors;