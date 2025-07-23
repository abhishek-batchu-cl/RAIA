# Model Upload and Management Service with ONNX Support
import os
import json
import uuid
import pickle
import joblib
import onnx
import torch
import tensorflow as tf
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from sqlalchemy.orm import Session
from sqlalchemy import Column, String, DateTime, Text, Boolean, JSON, UUID as PG_UUID, Float, Integer, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
import tempfile
import shutil
import warnings
warnings.filterwarnings('ignore')

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import tf2onnx
    TF2ONNX_AVAILABLE = True
except ImportError:
    TF2ONNX_AVAILABLE = False

try:
    import skl2onnx
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    SKL2ONNX_AVAILABLE = True
except ImportError:
    SKL2ONNX_AVAILABLE = False

Base = declarative_base()

@dataclass
class ModelMetadata:
    framework: str
    version: str
    input_shape: List[int]
    output_shape: List[int]
    model_type: str  # 'classification', 'regression', 'clustering'
    features: List[str]
    performance_metrics: Optional[Dict[str, float]] = None
    preprocessing_info: Optional[Dict[str, Any]] = None
    training_info: Optional[Dict[str, Any]] = None

@dataclass
class ConversionResult:
    success: bool
    onnx_model_path: Optional[str] = None
    converted_metadata: Optional[ModelMetadata] = None
    error_message: Optional[str] = None
    conversion_log: List[str] = None

class UploadedModel(Base):
    """Store uploaded model information"""
    __tablename__ = "uploaded_models"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(String(255), unique=True, nullable=False)
    name = Column(String(500), nullable=False)
    original_filename = Column(String(500), nullable=False)
    file_path = Column(String(1000), nullable=False)
    file_size = Column(Integer, nullable=False)
    format = Column(String(50), nullable=False)  # 'onnx', 'pytorch', 'tensorflow', etc.
    status = Column(String(50), default='uploaded')  # 'uploaded', 'processing', 'ready', 'error'
    
    # Model metadata
    metadata = Column(JSON)
    framework = Column(String(100))
    framework_version = Column(String(50))
    model_type = Column(String(50))  # 'classification', 'regression', 'clustering'
    input_schema = Column(JSON)
    output_schema = Column(JSON)
    feature_names = Column(JSON)
    performance_metrics = Column(JSON)
    
    # Conversion info
    onnx_converted = Column(Boolean, default=False)
    onnx_model_path = Column(String(1000))
    conversion_log = Column(Text)
    
    # Deployment info
    deployment_ready = Column(Boolean, default=False)
    api_endpoint = Column(String(500))
    
    # Timestamps
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime)
    last_accessed = Column(DateTime)
    
    # User info
    uploaded_by = Column(String(255))
    organization_id = Column(String(255))

class ModelConversionLog(Base):
    """Store model conversion attempts and results"""
    __tablename__ = "model_conversion_logs"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(String(255), nullable=False)
    source_format = Column(String(50), nullable=False)
    target_format = Column(String(50), default='onnx')
    conversion_status = Column(String(50), nullable=False)  # 'success', 'failed', 'in_progress'
    conversion_time = Column(Float)  # seconds
    error_message = Column(Text)
    conversion_details = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class ModelUploadService:
    """Comprehensive service for uploading, processing, and converting ML models"""
    
    def __init__(self, db_session: Session = None, storage_path: str = "/tmp/raia_models"):
        self.db = db_session
        self.storage_path = storage_path
        
        # Ensure storage directory exists
        os.makedirs(storage_path, exist_ok=True)
        os.makedirs(f"{storage_path}/original", exist_ok=True)
        os.makedirs(f"{storage_path}/converted", exist_ok=True)
        
        # Supported formats and their processors
        self.format_processors = {
            'onnx': self._process_onnx_model,
            'pytorch': self._process_pytorch_model,
            'tensorflow': self._process_tensorflow_model,
            'sklearn': self._process_sklearn_model,
            'xgboost': self._process_xgboost_model,
            'lightgbm': self._process_lightgbm_model
        }
        
        # Conversion functions
        self.converters = {
            'pytorch': self._convert_pytorch_to_onnx,
            'tensorflow': self._convert_tensorflow_to_onnx,
            'sklearn': self._convert_sklearn_to_onnx,
            'xgboost': self._convert_xgboost_to_onnx,
            'lightgbm': self._convert_lightgbm_to_onnx
        }

    async def upload_model(self, file_data: bytes, filename: str, 
                          uploaded_by: str = None, organization_id: str = None) -> Dict[str, Any]:
        """Upload and process a model file"""
        
        model_id = f"model_{uuid.uuid4().hex}"
        
        # Detect format
        model_format = self._detect_model_format(filename)
        if not model_format:
            return {
                'success': False,
                'error': f'Unsupported model format for file: {filename}'
            }
        
        # Save file
        original_path = os.path.join(self.storage_path, "original", f"{model_id}_{filename}")
        with open(original_path, 'wb') as f:
            f.write(file_data)
        
        # Create database record
        model_record = UploadedModel(
            model_id=model_id,
            name=os.path.splitext(filename)[0],
            original_filename=filename,
            file_path=original_path,
            file_size=len(file_data),
            format=model_format,
            status='uploaded',
            uploaded_by=uploaded_by,
            organization_id=organization_id
        )
        
        if self.db:
            self.db.add(model_record)
            self.db.commit()
        
        # Process model asynchronously
        try:
            processing_result = await self._process_uploaded_model(model_record)
            return {
                'success': True,
                'model_id': model_id,
                'processing_result': processing_result
            }
        except Exception as e:
            if self.db:
                model_record.status = 'error'
                self.db.commit()
            return {
                'success': False,
                'model_id': model_id,
                'error': str(e)
            }

    async def _process_uploaded_model(self, model_record: UploadedModel) -> Dict[str, Any]:
        """Process uploaded model to extract metadata"""
        
        if self.db:
            model_record.status = 'processing'
            self.db.commit()
        
        try:
            # Get processor for this format
            processor = self.format_processors.get(model_record.format)
            if not processor:
                raise ValueError(f"No processor available for format: {model_record.format}")
            
            # Process model
            metadata = await processor(model_record.file_path)
            
            # Update database record
            if self.db:
                model_record.metadata = asdict(metadata) if metadata else None
                model_record.framework = metadata.framework if metadata else None
                model_record.framework_version = metadata.version if metadata else None
                model_record.model_type = metadata.model_type if metadata else None
                model_record.input_schema = metadata.input_shape if metadata else None
                model_record.output_schema = metadata.output_shape if metadata else None
                model_record.feature_names = metadata.features if metadata else None
                model_record.performance_metrics = metadata.performance_metrics if metadata else None
                model_record.status = 'ready'
                model_record.processed_at = datetime.utcnow()
                model_record.deployment_ready = model_record.format == 'onnx'
                self.db.commit()
            
            return {
                'success': True,
                'metadata': asdict(metadata) if metadata else None
            }
            
        except Exception as e:
            if self.db:
                model_record.status = 'error'
                self.db.commit()
            raise e

    def _detect_model_format(self, filename: str) -> Optional[str]:
        """Detect model format based on file extension"""
        
        ext = filename.lower().split('.')[-1]
        
        format_mapping = {
            'onnx': 'onnx',
            'pt': 'pytorch',
            'pth': 'pytorch',
            'pb': 'tensorflow',
            'h5': 'tensorflow',
            'pkl': 'sklearn',  # Could also be pytorch, we'll detect in processing
            'joblib': 'sklearn',
            'model': 'xgboost',  # Could also be lightgbm
            'json': 'xgboost',
            'ubj': 'xgboost',
            'txt': 'lightgbm'
        }
        
        return format_mapping.get(ext)

    async def _process_onnx_model(self, file_path: str) -> ModelMetadata:
        """Process ONNX model file"""
        
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX not available. Install with: pip install onnx onnxruntime")
        
        try:
            # Load ONNX model
            onnx_model = onnx.load(file_path)
            
            # Get input/output info
            input_info = []
            output_info = []
            
            for input_tensor in onnx_model.graph.input:
                shape = []
                for dim in input_tensor.type.tensor_type.shape.dim:
                    if dim.dim_value:
                        shape.append(dim.dim_value)
                    else:
                        shape.append(-1)  # Dynamic dimension
                input_info.append(shape)
            
            for output_tensor in onnx_model.graph.output:
                shape = []
                for dim in output_tensor.type.tensor_type.shape.dim:
                    if dim.dim_value:
                        shape.append(dim.dim_value)
                    else:
                        shape.append(-1)  # Dynamic dimension
                output_info.append(shape)
            
            # Determine model type (heuristic)
            model_type = self._infer_model_type_from_output_shape(output_info[0] if output_info else [])
            
            return ModelMetadata(
                framework="ONNX Runtime",
                version=onnx.__version__,
                input_shape=input_info[0] if input_info else [],
                output_shape=output_info[0] if output_info else [],
                model_type=model_type,
                features=[f"feature_{i}" for i in range(input_info[0][1] if len(input_info[0]) > 1 else 10)]
            )
            
        except Exception as e:
            raise ValueError(f"Failed to process ONNX model: {str(e)}")

    async def _process_pytorch_model(self, file_path: str) -> ModelMetadata:
        """Process PyTorch model file"""
        
        try:
            # Try to load as state dict first
            try:
                state_dict = torch.load(file_path, map_location='cpu')
                if isinstance(state_dict, dict):
                    # This is a state dict, we need the model architecture
                    return ModelMetadata(
                        framework="PyTorch",
                        version=torch.__version__,
                        input_shape=[1, 784],  # Default assumption
                        output_shape=[1, 10],   # Default assumption
                        model_type="classification",
                        features=[f"feature_{i}" for i in range(784)],
                        preprocessing_info={"type": "state_dict"}
                    )
            except:
                pass
            
            # Try to load as complete model
            model = torch.load(file_path, map_location='cpu')
            
            # Try to infer input/output shapes
            input_shape = [1, 784]  # Default
            output_shape = [1, 10]  # Default
            
            # If it's a nn.Module, we could try to inspect it
            if hasattr(model, 'named_parameters'):
                params = list(model.named_parameters())
                if params:
                    # Try to infer from first layer
                    first_param = params[0][1]
                    if len(first_param.shape) >= 2:
                        input_shape = [1, first_param.shape[1]]
                    
                    # Try to infer from last layer
                    last_param = params[-1][1]
                    if len(last_param.shape) >= 1:
                        output_shape = [1, last_param.shape[0]]
            
            model_type = self._infer_model_type_from_output_shape(output_shape)
            
            return ModelMetadata(
                framework="PyTorch",
                version=torch.__version__,
                input_shape=input_shape,
                output_shape=output_shape,
                model_type=model_type,
                features=[f"feature_{i}" for i in range(input_shape[1] if len(input_shape) > 1 else 10)]
            )
            
        except Exception as e:
            raise ValueError(f"Failed to process PyTorch model: {str(e)}")

    async def _process_tensorflow_model(self, file_path: str) -> ModelMetadata:
        """Process TensorFlow model file"""
        
        try:
            if file_path.endswith('.h5'):
                # Load Keras model
                model = tf.keras.models.load_model(file_path)
                
                input_shape = model.input_shape if hasattr(model, 'input_shape') else [None, 784]
                output_shape = model.output_shape if hasattr(model, 'output_shape') else [None, 10]
                
            else:
                # Try to load SavedModel
                model = tf.saved_model.load(file_path)
                # For SavedModel, shape inference is more complex
                input_shape = [1, 784]  # Default
                output_shape = [1, 10]  # Default
            
            model_type = self._infer_model_type_from_output_shape(output_shape)
            
            return ModelMetadata(
                framework="TensorFlow",
                version=tf.__version__,
                input_shape=list(input_shape) if input_shape else [1, 784],
                output_shape=list(output_shape) if output_shape else [1, 10],
                model_type=model_type,
                features=[f"feature_{i}" for i in range(input_shape[1] if len(input_shape) > 1 else 784)]
            )
            
        except Exception as e:
            raise ValueError(f"Failed to process TensorFlow model: {str(e)}")

    async def _process_sklearn_model(self, file_path: str) -> ModelMetadata:
        """Process Scikit-learn model file"""
        
        try:
            # Try joblib first, then pickle
            try:
                model = joblib.load(file_path)
            except:
                with open(file_path, 'rb') as f:
                    model = pickle.load(f)
            
            # Get model info
            model_name = type(model).__name__
            n_features = getattr(model, 'n_features_in_', 10)  # Default to 10
            
            # Determine model type
            if hasattr(model, 'classes_'):
                model_type = 'classification'
                n_outputs = len(model.classes_)
            elif hasattr(model, 'n_outputs_'):
                model_type = 'regression'
                n_outputs = model.n_outputs_
            else:
                model_type = 'classification'  # Default
                n_outputs = 2
            
            # Get feature names if available
            feature_names = getattr(model, 'feature_names_in_', None)
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(n_features)]
            
            return ModelMetadata(
                framework="Scikit-learn",
                version="1.0.0",  # Default version
                input_shape=[1, n_features],
                output_shape=[1, n_outputs],
                model_type=model_type,
                features=list(feature_names),
                preprocessing_info={"model_class": model_name}
            )
            
        except Exception as e:
            raise ValueError(f"Failed to process Scikit-learn model: {str(e)}")

    async def _process_xgboost_model(self, file_path: str) -> ModelMetadata:
        """Process XGBoost model file"""
        
        try:
            import xgboost as xgb
            
            # Load XGBoost model
            if file_path.endswith('.json') or file_path.endswith('.ubj'):
                model = xgb.Booster()
                model.load_model(file_path)
            else:
                model = xgb.Booster(model_file=file_path)
            
            # Get model info
            config = json.loads(model.save_config())
            learner_config = config.get('learner', {})
            
            # Infer shapes (XGBoost doesn't store this directly)
            n_features = 10  # Default, should be inferred from training data
            objective = learner_config.get('objective', {}).get('name', 'binary:logistic')
            
            if 'multi:' in objective:
                model_type = 'classification'
                n_outputs = 10  # Default for multiclass
            elif 'binary:' in objective or 'rank:' in objective:
                model_type = 'classification'
                n_outputs = 1
            else:
                model_type = 'regression'
                n_outputs = 1
            
            return ModelMetadata(
                framework="XGBoost",
                version=xgb.__version__,
                input_shape=[1, n_features],
                output_shape=[1, n_outputs],
                model_type=model_type,
                features=[f"feature_{i}" for i in range(n_features)],
                training_info={"objective": objective}
            )
            
        except ImportError:
            raise ValueError("XGBoost not available. Install with: pip install xgboost")
        except Exception as e:
            raise ValueError(f"Failed to process XGBoost model: {str(e)}")

    async def _process_lightgbm_model(self, file_path: str) -> ModelMetadata:
        """Process LightGBM model file"""
        
        try:
            import lightgbm as lgb
            
            # Load LightGBM model
            model = lgb.Booster(model_file=file_path)
            
            # Get model info
            n_features = model.num_feature()
            n_classes = model.num_model_per_iteration()
            
            model_type = 'classification' if n_classes > 1 else 'regression'
            n_outputs = n_classes if model_type == 'classification' else 1
            
            # Get feature names
            feature_names = model.feature_name()
            if not feature_names:
                feature_names = [f"feature_{i}" for i in range(n_features)]
            
            return ModelMetadata(
                framework="LightGBM",
                version=lgb.__version__,
                input_shape=[1, n_features],
                output_shape=[1, n_outputs],
                model_type=model_type,
                features=feature_names
            )
            
        except ImportError:
            raise ValueError("LightGBM not available. Install with: pip install lightgbm")
        except Exception as e:
            raise ValueError(f"Failed to process LightGBM model: {str(e)}")

    def _infer_model_type_from_output_shape(self, output_shape: List[int]) -> str:
        """Infer model type from output shape"""
        
        if len(output_shape) < 2:
            return "regression"  # Single output
        
        output_size = output_shape[-1] if output_shape else 1
        
        if output_size == 1:
            return "regression"
        elif output_size > 1:
            return "classification"
        else:
            return "regression"

    async def convert_to_onnx(self, model_id: str) -> ConversionResult:
        """Convert uploaded model to ONNX format"""
        
        if not ONNX_AVAILABLE:
            return ConversionResult(
                success=False,
                error_message="ONNX not available. Install with: pip install onnx onnxruntime"
            )
        
        # Get model record
        model_record = None
        if self.db:
            model_record = self.db.query(UploadedModel).filter(
                UploadedModel.model_id == model_id
            ).first()
        
        if not model_record:
            return ConversionResult(
                success=False,
                error_message=f"Model {model_id} not found"
            )
        
        if model_record.format == 'onnx':
            return ConversionResult(
                success=True,
                onnx_model_path=model_record.file_path,
                converted_metadata=ModelMetadata(**model_record.metadata) if model_record.metadata else None
            )
        
        # Log conversion attempt
        conversion_log = ModelConversionLog(
            model_id=model_id,
            source_format=model_record.format,
            target_format='onnx',
            conversion_status='in_progress'
        )
        
        if self.db:
            self.db.add(conversion_log)
            self.db.commit()
        
        start_time = datetime.utcnow()
        
        try:
            # Get converter for this format
            converter = self.converters.get(model_record.format)
            if not converter:
                raise ValueError(f"No ONNX converter available for format: {model_record.format}")
            
            # Perform conversion
            result = await converter(model_record.file_path, model_record.metadata)
            
            if result.success:
                # Update model record
                if self.db:
                    model_record.onnx_converted = True
                    model_record.onnx_model_path = result.onnx_model_path
                    model_record.deployment_ready = True
                    model_record.conversion_log = json.dumps(result.conversion_log or [])
                    
                    # Update conversion log
                    conversion_log.conversion_status = 'success'
                    conversion_log.conversion_time = (datetime.utcnow() - start_time).total_seconds()
                    
                    self.db.commit()
            
            return result
            
        except Exception as e:
            if self.db:
                conversion_log.conversion_status = 'failed'
                conversion_log.error_message = str(e)
                conversion_log.conversion_time = (datetime.utcnow() - start_time).total_seconds()
                self.db.commit()
            
            return ConversionResult(
                success=False,
                error_message=str(e)
            )

    async def _convert_pytorch_to_onnx(self, model_path: str, metadata: Dict[str, Any]) -> ConversionResult:
        """Convert PyTorch model to ONNX"""
        
        try:
            # Load PyTorch model
            model = torch.load(model_path, map_location='cpu')
            model.eval()
            
            # Create dummy input
            input_shape = metadata.get('input_shape', [1, 784])
            dummy_input = torch.randn(*input_shape)
            
            # Export to ONNX
            onnx_path = model_path.replace('.pt', '.onnx').replace('.pth', '.onnx')
            
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output']
            )
            
            # Verify the conversion
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            
            return ConversionResult(
                success=True,
                onnx_model_path=onnx_path,
                conversion_log=["PyTorch model successfully converted to ONNX"]
            )
            
        except Exception as e:
            return ConversionResult(
                success=False,
                error_message=f"PyTorch to ONNX conversion failed: {str(e)}"
            )

    async def _convert_sklearn_to_onnx(self, model_path: str, metadata: Dict[str, Any]) -> ConversionResult:
        """Convert Scikit-learn model to ONNX"""
        
        if not SKL2ONNX_AVAILABLE:
            return ConversionResult(
                success=False,
                error_message="skl2onnx not available. Install with: pip install skl2onnx"
            )
        
        try:
            # Load sklearn model
            try:
                model = joblib.load(model_path)
            except:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            
            # Get input shape
            input_shape = metadata.get('input_shape', [1, 10])
            n_features = input_shape[1] if len(input_shape) > 1 else 10
            
            # Define input type
            initial_type = [('float_input', FloatTensorType([None, n_features]))]
            
            # Convert to ONNX
            onnx_model = convert_sklearn(model, initial_types=initial_type)
            
            # Save ONNX model
            onnx_path = model_path.replace('.pkl', '.onnx').replace('.joblib', '.onnx')
            with open(onnx_path, 'wb') as f:
                f.write(onnx_model.SerializeToString())
            
            return ConversionResult(
                success=True,
                onnx_model_path=onnx_path,
                conversion_log=["Scikit-learn model successfully converted to ONNX"]
            )
            
        except Exception as e:
            return ConversionResult(
                success=False,
                error_message=f"Scikit-learn to ONNX conversion failed: {str(e)}"
            )

    async def _convert_tensorflow_to_onnx(self, model_path: str, metadata: Dict[str, Any]) -> ConversionResult:
        """Convert TensorFlow model to ONNX"""
        
        if not TF2ONNX_AVAILABLE:
            return ConversionResult(
                success=False,
                error_message="tf2onnx not available. Install with: pip install tf2onnx"
            )
        
        try:
            import subprocess
            
            # Use tf2onnx command line tool
            onnx_path = model_path.replace('.h5', '.onnx').replace('.pb', '.onnx')
            
            if model_path.endswith('.h5'):
                cmd = [
                    'python', '-m', 'tf2onnx.convert',
                    '--keras', model_path,
                    '--output', onnx_path,
                    '--opset', '11'
                ]
            else:
                cmd = [
                    'python', '-m', 'tf2onnx.convert',
                    '--saved-model', model_path,
                    '--output', onnx_path,
                    '--opset', '11'
                ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                return ConversionResult(
                    success=True,
                    onnx_model_path=onnx_path,
                    conversion_log=["TensorFlow model successfully converted to ONNX", result.stdout]
                )
            else:
                return ConversionResult(
                    success=False,
                    error_message=f"TensorFlow to ONNX conversion failed: {result.stderr}"
                )
                
        except Exception as e:
            return ConversionResult(
                success=False,
                error_message=f"TensorFlow to ONNX conversion failed: {str(e)}"
            )

    async def _convert_xgboost_to_onnx(self, model_path: str, metadata: Dict[str, Any]) -> ConversionResult:
        """Convert XGBoost model to ONNX"""
        
        try:
            import xgboost as xgb
            from onnxmltools import convert_xgboost
            from onnxmltools.convert.common.data_types import FloatTensorType
            
            # Load XGBoost model
            model = xgb.Booster(model_file=model_path)
            
            # Get input shape
            input_shape = metadata.get('input_shape', [1, 10])
            n_features = input_shape[1] if len(input_shape) > 1 else 10
            
            # Define input type
            initial_type = [('float_input', FloatTensorType([None, n_features]))]
            
            # Convert to ONNX
            onnx_model = convert_xgboost(model, initial_types=initial_type)
            
            # Save ONNX model
            onnx_path = model_path.replace('.model', '.onnx').replace('.json', '.onnx').replace('.ubj', '.onnx')
            with open(onnx_path, 'wb') as f:
                f.write(onnx_model.SerializeToString())
            
            return ConversionResult(
                success=True,
                onnx_model_path=onnx_path,
                conversion_log=["XGBoost model successfully converted to ONNX"]
            )
            
        except ImportError:
            return ConversionResult(
                success=False,
                error_message="onnxmltools not available. Install with: pip install onnxmltools"
            )
        except Exception as e:
            return ConversionResult(
                success=False,
                error_message=f"XGBoost to ONNX conversion failed: {str(e)}"
            )

    async def _convert_lightgbm_to_onnx(self, model_path: str, metadata: Dict[str, Any]) -> ConversionResult:
        """Convert LightGBM model to ONNX"""
        
        try:
            import lightgbm as lgb
            from onnxmltools import convert_lightgbm
            from onnxmltools.convert.common.data_types import FloatTensorType
            
            # Load LightGBM model
            model = lgb.Booster(model_file=model_path)
            
            # Get input shape
            input_shape = metadata.get('input_shape', [1, 10])
            n_features = input_shape[1] if len(input_shape) > 1 else 10
            
            # Define input type
            initial_type = [('float_input', FloatTensorType([None, n_features]))]
            
            # Convert to ONNX
            onnx_model = convert_lightgbm(model, initial_types=initial_type)
            
            # Save ONNX model
            onnx_path = model_path.replace('.txt', '.onnx').replace('.model', '.onnx')
            with open(onnx_path, 'wb') as f:
                f.write(onnx_model.SerializeToString())
            
            return ConversionResult(
                success=True,
                onnx_model_path=onnx_path,
                conversion_log=["LightGBM model successfully converted to ONNX"]
            )
            
        except ImportError:
            return ConversionResult(
                success=False,
                error_message="onnxmltools not available. Install with: pip install onnxmltools"
            )
        except Exception as e:
            return ConversionResult(
                success=False,
                error_message=f"LightGBM to ONNX conversion failed: {str(e)}"
            )

    async def list_uploaded_models(self, user_id: str = None, 
                                 organization_id: str = None) -> List[Dict[str, Any]]:
        """List all uploaded models for a user or organization"""
        
        if not self.db:
            return []
        
        query = self.db.query(UploadedModel)
        
        if user_id:
            query = query.filter(UploadedModel.uploaded_by == user_id)
        if organization_id:
            query = query.filter(UploadedModel.organization_id == organization_id)
        
        models = query.order_by(UploadedModel.uploaded_at.desc()).all()
        
        return [
            {
                'model_id': model.model_id,
                'name': model.name,
                'format': model.format,
                'status': model.status,
                'file_size': model.file_size,
                'metadata': model.metadata,
                'onnx_converted': model.onnx_converted,
                'deployment_ready': model.deployment_ready,
                'uploaded_at': model.uploaded_at.isoformat(),
                'processed_at': model.processed_at.isoformat() if model.processed_at else None
            }
            for model in models
        ]

    async def delete_model(self, model_id: str) -> Dict[str, Any]:
        """Delete uploaded model and associated files"""
        
        if self.db:
            model = self.db.query(UploadedModel).filter(
                UploadedModel.model_id == model_id
            ).first()
            
            if model:
                # Delete files
                try:
                    if os.path.exists(model.file_path):
                        os.remove(model.file_path)
                    if model.onnx_model_path and os.path.exists(model.onnx_model_path):
                        os.remove(model.onnx_model_path)
                except Exception as e:
                    print(f"Error deleting files: {e}")
                
                # Delete database record
                self.db.delete(model)
                self.db.commit()
                
                return {'success': True, 'message': 'Model deleted successfully'}
        
        return {'success': False, 'message': 'Model not found'}

# Factory function
def create_model_upload_service(db_session: Session = None, 
                               storage_path: str = "/tmp/raia_models") -> ModelUploadService:
    """Create and return a ModelUploadService instance"""
    return ModelUploadService(db_session, storage_path)