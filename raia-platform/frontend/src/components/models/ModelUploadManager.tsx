import React, { useState, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Upload, File, CheckCircle, AlertTriangle, X, 
  Brain, Zap, Settings, Eye, Download, Copy,
  FileText, Database, Cloud, HardDrive, Globe,
  Layers, Code, Activity, Shield, Target
} from 'lucide-react';
import Card from '@/components/common/Card';
import Button from '@/components/common/Button';
import { apiClient } from '@/services/api';

interface ModelFile {
  id: string;
  name: string;
  size: number;
  type: string;
  format: 'onnx' | 'pytorch' | 'tensorflow' | 'sklearn' | 'xgboost' | 'lightgbm' | 'custom';
  status: 'uploading' | 'processing' | 'ready' | 'error';
  progress: number;
  metadata?: {
    framework: string;
    version: string;
    input_shape: number[];
    output_shape: number[];
    model_type: 'classification' | 'regression' | 'clustering';
    features: string[];
    performance_metrics?: any;
  };
  error_message?: string;
  uploaded_at: Date;
}

interface SupportedFormat {
  id: string;
  name: string;
  extensions: string[];
  icon: React.ReactNode;
  description: string;
  features: string[];
  conversion_supported: boolean;
}

const ModelUploadManager: React.FC = () => {
  const [uploadedModels, setUploadedModels] = useState<ModelFile[]>([]);
  const [dragActive, setDragActive] = useState(false);
  const [showConversionModal, setShowConversionModal] = useState(false);
  const [selectedModel, setSelectedModel] = useState<ModelFile | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Supported model formats
  const supportedFormats: SupportedFormat[] = [
    {
      id: 'onnx',
      name: 'ONNX',
      extensions: ['.onnx'],
      icon: <Brain className="w-6 h-6 text-blue-500" />,
      description: 'Open Neural Network Exchange - Universal model format',
      features: ['Cross-platform', 'Hardware acceleration', 'Production ready'],
      conversion_supported: false
    },
    {
      id: 'pytorch',
      name: 'PyTorch',
      extensions: ['.pt', '.pth', '.pkl'],
      icon: <Zap className="w-6 h-6 text-orange-500" />,
      description: 'PyTorch model files and state dictionaries',
      features: ['Dynamic graphs', 'Research friendly', 'Convert to ONNX'],
      conversion_supported: true
    },
    {
      id: 'tensorflow',
      name: 'TensorFlow',
      extensions: ['.pb', '.h5', '.tf'],
      icon: <Layers className="w-6 h-6 text-green-500" />,
      description: 'TensorFlow SavedModel, HDF5, and frozen graphs',
      features: ['Production ready', 'TensorFlow Lite', 'Convert to ONNX'],
      conversion_supported: true
    },
    {
      id: 'sklearn',
      name: 'Scikit-learn',
      extensions: ['.pkl', '.joblib'],
      icon: <Target className="w-6 h-6 text-purple-500" />,
      description: 'Scikit-learn models using pickle or joblib',
      features: ['Classical ML', 'Feature engineering', 'SHAP support'],
      conversion_supported: true
    },
    {
      id: 'xgboost',
      name: 'XGBoost',
      extensions: ['.model', '.json', '.ubj'],
      icon: <Activity className="w-6 h-6 text-red-500" />,
      description: 'XGBoost gradient boosting models',
      features: ['High performance', 'Feature importance', 'SHAP support'],
      conversion_supported: true
    },
    {
      id: 'lightgbm',
      name: 'LightGBM',
      extensions: ['.txt', '.model'],
      icon: <Zap className="w-6 h-6 text-yellow-500" />,
      description: 'Microsoft LightGBM models',
      features: ['Fast training', 'Memory efficient', 'SHAP support'],
      conversion_supported: true
    }
  ];

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFiles(Array.from(e.dataTransfer.files));
    }
  }, []);

  const handleFiles = async (files: File[]) => {
    for (const file of files) {
      const modelFile: ModelFile = {
        id: `model-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        name: file.name,
        size: file.size,
        type: file.type,
        format: detectModelFormat(file.name),
        status: 'uploading',
        progress: 0,
        uploaded_at: new Date()
      };

      setUploadedModels(prev => [...prev, modelFile]);
      
      // Simulate upload and processing
      await simulateModelUpload(modelFile);
    }
  };

  const detectModelFormat = (fileName: string): ModelFile['format'] => {
    const ext = fileName.toLowerCase().substring(fileName.lastIndexOf('.'));
    
    if (ext === '.onnx') return 'onnx';
    if (['.pt', '.pth'].includes(ext)) return 'pytorch';
    if (['.pb', '.h5', '.tf'].includes(ext)) return 'tensorflow';
    if (['.pkl', '.joblib'].includes(ext)) return 'sklearn';
    if (['.model', '.json', '.ubj'].includes(ext)) return 'xgboost';
    if (ext === '.txt') return 'lightgbm';
    
    return 'custom';
  };

  const simulateModelUpload = async (modelFile: ModelFile) => {
    // Simulate upload progress
    for (let progress = 0; progress <= 100; progress += 10) {
      await new Promise(resolve => setTimeout(resolve, 200));
      setUploadedModels(prev =>
        prev.map(m => m.id === modelFile.id ? { ...m, progress } : m)
      );
    }

    // Simulate processing
    setUploadedModels(prev =>
      prev.map(m => m.id === modelFile.id ? { ...m, status: 'processing' } : m)
    );

    await new Promise(resolve => setTimeout(resolve, 2000));

    // Simulate completion with metadata
    const mockMetadata = {
      framework: getFrameworkName(modelFile.format),
      version: '1.0.0',
      input_shape: [1, 784] as number[],
      output_shape: [1, 10] as number[],
      model_type: 'classification' as const,
      features: ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5'],
      performance_metrics: {
        accuracy: 0.94,
        precision: 0.92,
        recall: 0.91,
        f1_score: 0.915
      }
    };

    setUploadedModels(prev =>
      prev.map(m => m.id === modelFile.id ? { 
        ...m, 
        status: 'ready',
        metadata: mockMetadata
      } : m)
    );
  };

  const getFrameworkName = (format: ModelFile['format']): string => {
    switch (format) {
      case 'onnx': return 'ONNX Runtime';
      case 'pytorch': return 'PyTorch';
      case 'tensorflow': return 'TensorFlow';
      case 'sklearn': return 'Scikit-learn';
      case 'xgboost': return 'XGBoost';
      case 'lightgbm': return 'LightGBM';
      default: return 'Unknown';
    }
  };

  const getStatusIcon = (status: ModelFile['status']) => {
    switch (status) {
      case 'uploading':
      case 'processing':
        return <motion.div animate={{ rotate: 360 }} transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}>
          <Settings className="w-5 h-5 text-blue-500" />
        </motion.div>;
      case 'ready':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'error':
        return <AlertTriangle className="w-5 h-5 text-red-500" />;
      default:
        return <File className="w-5 h-5 text-gray-500" />;
    }
  };

  const getFormatIcon = (format: ModelFile['format']) => {
    const formatConfig = supportedFormats.find(f => f.id === format);
    return formatConfig?.icon || <File className="w-6 h-6 text-gray-500" />;
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const removeModel = (modelId: string) => {
    setUploadedModels(prev => prev.filter(m => m.id !== modelId));
  };

  const convertModel = (model: ModelFile) => {
    setSelectedModel(model);
    setShowConversionModal(true);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-neutral-900 dark:text-neutral-100 flex items-center">
            <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-500 rounded-lg flex items-center justify-center mr-3">
              <Upload className="w-5 h-5 text-white" />
            </div>
            Model Upload & Management
          </h1>
          <p className="text-neutral-600 dark:text-neutral-400 mt-1">
            Upload, convert, and manage ML models in multiple formats with ONNX support
          </p>
        </div>
        
        <Button
          variant="primary"
          leftIcon={<Upload className="w-4 h-4" />}
          onClick={() => fileInputRef.current?.click()}
        >
          Upload Model
        </Button>
      </div>

      {/* Supported Formats */}
      <Card title="Supported Model Formats" icon={<Brain className="w-5 h-5 text-primary-500" />}>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {supportedFormats.map((format) => (
            <motion.div
              key={format.id}
              whileHover={{ scale: 1.02 }}
              className="p-4 border border-neutral-200 dark:border-neutral-700 rounded-lg hover:shadow-md transition-all duration-200"
            >
              <div className="flex items-start space-x-3">
                {format.icon}
                <div className="flex-1">
                  <h3 className="font-semibold text-neutral-900 dark:text-neutral-100 mb-1">
                    {format.name}
                  </h3>
                  <p className="text-sm text-neutral-600 dark:text-neutral-400 mb-2">
                    {format.description}
                  </p>
                  <div className="flex flex-wrap gap-1 mb-2">
                    {format.extensions.map(ext => (
                      <span
                        key={ext}
                        className="px-2 py-1 text-xs bg-neutral-100 dark:bg-neutral-800 text-neutral-600 dark:text-neutral-400 rounded"
                      >
                        {ext}
                      </span>
                    ))}
                  </div>
                  <div className="space-y-1">
                    {format.features.map(feature => (
                      <div key={feature} className="flex items-center space-x-1 text-xs text-neutral-500 dark:text-neutral-400">
                        <CheckCircle className="w-3 h-3 text-green-500" />
                        <span>{feature}</span>
                      </div>
                    ))}
                  </div>
                  {format.conversion_supported && (
                    <div className="mt-2 text-xs text-blue-600 dark:text-blue-400 font-medium">
                      ↗ Convert to ONNX supported
                    </div>
                  )}
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </Card>

      {/* Upload Area */}
      <Card>
        <div
          className={`relative border-2 border-dashed rounded-lg p-8 text-center transition-colors duration-200 ${
            dragActive
              ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
              : 'border-neutral-300 dark:border-neutral-600 hover:border-primary-400 dark:hover:border-primary-500'
          }`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".onnx,.pt,.pth,.pb,.h5,.tf,.pkl,.joblib,.model,.json,.ubj,.txt"
            onChange={(e) => e.target.files && handleFiles(Array.from(e.target.files))}
            className="hidden"
          />
          
          <div className="space-y-4">
            <div className="w-16 h-16 mx-auto bg-neutral-100 dark:bg-neutral-800 rounded-full flex items-center justify-center">
              <Upload className="w-8 h-8 text-neutral-500 dark:text-neutral-400" />
            </div>
            
            <div>
              <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-2">
                {dragActive ? 'Drop your model files here' : 'Upload your ML models'}
              </h3>
              <p className="text-neutral-600 dark:text-neutral-400 mb-4">
                Drag and drop your model files or click to browse
              </p>
              <p className="text-sm text-neutral-500 dark:text-neutral-400">
                Supports ONNX, PyTorch, TensorFlow, Scikit-learn, XGBoost, LightGBM
              </p>
            </div>
            
            <Button
              variant="outline"
              onClick={() => fileInputRef.current?.click()}
            >
              Choose Files
            </Button>
          </div>
        </div>
      </Card>

      {/* Uploaded Models */}
      {uploadedModels.length > 0 && (
        <Card title="Uploaded Models" icon={<Database className="w-5 h-5 text-primary-500" />}>
          <div className="space-y-4">
            <AnimatePresence>
              {uploadedModels.map((model) => (
                <motion.div
                  key={model.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className="border border-neutral-200 dark:border-neutral-700 rounded-lg p-4"
                >
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-start space-x-3">
                      {getFormatIcon(model.format)}
                      <div className="flex-1">
                        <h4 className="font-semibold text-neutral-900 dark:text-neutral-100 mb-1">
                          {model.name}
                        </h4>
                        <div className="flex items-center space-x-4 text-sm text-neutral-600 dark:text-neutral-400">
                          <span>{formatFileSize(model.size)}</span>
                          <span className="capitalize">{model.format}</span>
                          {model.metadata?.framework && (
                            <span>{model.metadata.framework}</span>
                          )}
                        </div>
                      </div>
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      {getStatusIcon(model.status)}
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => removeModel(model.id)}
                        className="text-red-500 hover:text-red-600"
                      >
                        <X className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>

                  {/* Progress Bar */}
                  {(model.status === 'uploading' || model.status === 'processing') && (
                    <div className="mb-3">
                      <div className="flex items-center justify-between text-sm text-neutral-600 dark:text-neutral-400 mb-1">
                        <span>
                          {model.status === 'uploading' ? 'Uploading...' : 'Processing model...'}
                        </span>
                        <span>{model.progress}%</span>
                      </div>
                      <div className="w-full bg-neutral-200 dark:bg-neutral-700 rounded-full h-2">
                        <motion.div
                          className="bg-primary-500 h-2 rounded-full"
                          initial={{ width: 0 }}
                          animate={{ width: `${model.progress}%` }}
                          transition={{ duration: 0.3 }}
                        />
                      </div>
                    </div>
                  )}

                  {/* Model Metadata */}
                  {model.status === 'ready' && model.metadata && (
                    <div className="space-y-3">
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                        <div>
                          <span className="font-medium text-neutral-700 dark:text-neutral-300">Type:</span>
                          <span className="ml-2 capitalize text-neutral-600 dark:text-neutral-400">
                            {model.metadata.model_type}
                          </span>
                        </div>
                        <div>
                          <span className="font-medium text-neutral-700 dark:text-neutral-300">Input Shape:</span>
                          <span className="ml-2 font-mono text-neutral-600 dark:text-neutral-400">
                            [{model.metadata.input_shape.join(', ')}]
                          </span>
                        </div>
                        <div>
                          <span className="font-medium text-neutral-700 dark:text-neutral-300">Output Shape:</span>
                          <span className="ml-2 font-mono text-neutral-600 dark:text-neutral-400">
                            [{model.metadata.output_shape.join(', ')}]
                          </span>
                        </div>
                      </div>

                      {/* Performance Metrics */}
                      {model.metadata.performance_metrics && (
                        <div className="p-3 bg-neutral-50 dark:bg-neutral-800 rounded-lg">
                          <h5 className="font-medium text-neutral-900 dark:text-neutral-100 mb-2">
                            Performance Metrics
                          </h5>
                          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                            {Object.entries(model.metadata.performance_metrics).map(([key, value]) => (
                              <div key={key} className="text-center">
                                <div className="text-lg font-bold text-neutral-900 dark:text-neutral-100">
                                  {typeof value === 'number' ? value.toFixed(3) : value}
                                </div>
                                <div className="text-xs text-neutral-500 dark:text-neutral-400 capitalize">
                                  {key.replace('_', ' ')}
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Action Buttons */}
                      <div className="flex items-center space-x-2 pt-2 border-t border-neutral-200 dark:border-neutral-700">
                        <Button variant="outline" size="sm" leftIcon={<Eye className="w-4 h-4" />}>
                          View Details
                        </Button>
                        <Button variant="outline" size="sm" leftIcon={<Code className="w-4 h-4" />}>
                          API Info
                        </Button>
                        {supportedFormats.find(f => f.id === model.format)?.conversion_supported && (
                          <Button
                            variant="outline"
                            size="sm"
                            leftIcon={<Zap className="w-4 h-4" />}
                            onClick={() => convertModel(model)}
                          >
                            Convert to ONNX
                          </Button>
                        )}
                        <Button variant="outline" size="sm" leftIcon={<Download className="w-4 h-4" />}>
                          Export
                        </Button>
                        <Button variant="primary" size="sm" leftIcon={<Activity className="w-4 h-4" />}>
                          Deploy
                        </Button>
                      </div>
                    </div>
                  )}

                  {/* Error Message */}
                  {model.status === 'error' && model.error_message && (
                    <div className="p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
                      <div className="flex items-start space-x-2">
                        <AlertTriangle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
                        <div>
                          <p className="font-medium text-red-900 dark:text-red-100">Upload Error</p>
                          <p className="text-sm text-red-700 dark:text-red-300 mt-1">
                            {model.error_message}
                          </p>
                        </div>
                      </div>
                    </div>
                  )}
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        </Card>
      )}

      {/* Conversion Modal */}
      <AnimatePresence>
        {showConversionModal && selectedModel && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
            onClick={() => setShowConversionModal(false)}
          >
            <motion.div
              initial={{ scale: 0.9 }}
              animate={{ scale: 1 }}
              exit={{ scale: 0.9 }}
              className="bg-white dark:bg-neutral-800 rounded-lg p-6 w-full max-w-lg mx-4"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 flex items-center">
                  <Zap className="w-5 h-5 text-primary-500 mr-2" />
                  Convert to ONNX
                </h3>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowConversionModal(false)}
                >
                  <X className="w-4 h-4" />
                </Button>
              </div>
              
              <div className="space-y-4">
                <div className="p-4 bg-neutral-50 dark:bg-neutral-700 rounded-lg">
                  <div className="flex items-center space-x-3 mb-2">
                    {getFormatIcon(selectedModel.format)}
                    <span className="font-medium text-neutral-900 dark:text-neutral-100">
                      {selectedModel.name}
                    </span>
                  </div>
                  <div className="text-sm text-neutral-600 dark:text-neutral-400">
                    Current format: {selectedModel.format.toUpperCase()}
                  </div>
                </div>
                
                <div className="text-center py-4">
                  <div className="flex items-center justify-center space-x-4">
                    {getFormatIcon(selectedModel.format)}
                    <div className="flex items-center space-x-2">
                      <div className="w-8 border-t border-neutral-300 dark:border-neutral-600"></div>
                      <Zap className="w-5 h-5 text-primary-500" />
                      <div className="w-8 border-t border-neutral-300 dark:border-neutral-600"></div>
                    </div>
                    <Brain className="w-6 h-6 text-blue-500" />
                  </div>
                  <p className="text-sm text-neutral-600 dark:text-neutral-400 mt-2">
                    Convert to ONNX for universal compatibility
                  </p>
                </div>
                
                <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
                  <h4 className="font-medium text-blue-900 dark:text-blue-100 mb-2">Benefits of ONNX:</h4>
                  <ul className="text-sm text-blue-800 dark:text-blue-200 space-y-1">
                    <li>• Cross-platform compatibility</li>
                    <li>• Hardware acceleration support</li>
                    <li>• Optimized inference performance</li>
                    <li>• Standardized model format</li>
                  </ul>
                </div>
              </div>
              
              <div className="flex justify-end space-x-3 mt-6">
                <Button
                  variant="outline"
                  onClick={() => setShowConversionModal(false)}
                >
                  Cancel
                </Button>
                <Button variant="primary" leftIcon={<Zap className="w-4 h-4" />}>
                  Convert Model
                </Button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default ModelUploadManager;