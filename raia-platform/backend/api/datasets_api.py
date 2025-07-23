# Dataset Management API
import os
import uuid
import json
import shutil
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from io import BytesIO, StringIO

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form, BackgroundTasks, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, validator
from sqlalchemy.orm import Session
from sqlalchemy import Column, String, DateTime, Text, Boolean, JSON, UUID as PG_UUID, Integer, Float, ForeignKey, func, desc
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

# Data processing libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.datasets import load_iris, load_boston, load_wine

# Data validation and profiling
import great_expectations as ge
from great_expectations.dataset import PandasDataset
from pandas_profiling import ProfileReport

# Data formats
import pyarrow as pa
import pyarrow.parquet as pq
import h5py
import sqlite3

Base = declarative_base()
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/datasets", tags=["datasets"])

# Database Models
class Dataset(Base):
    """Dataset metadata and tracking"""
    __tablename__ = "datasets"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    display_name = Column(String(255))
    description = Column(Text)
    
    # Dataset characteristics
    dataset_type = Column(String(100))  # tabular, image, text, time_series, etc.
    data_format = Column(String(50))  # csv, json, parquet, hdf5, etc.
    file_size_bytes = Column(Integer)
    row_count = Column(Integer)
    column_count = Column(Integer)
    
    # Schema information
    schema = Column(JSON)  # Column names, types, constraints
    column_metadata = Column(JSON)  # Detailed column information
    statistics = Column(JSON)  # Basic statistical summary
    data_quality_report = Column(JSON)  # Data quality metrics
    
    # File information
    file_path = Column(String(1000))
    checksum = Column(String(128))
    compression = Column(String(50))
    encoding = Column(String(50), default="utf-8")
    
    # Data lineage and versioning
    version = Column(String(50), default="1.0.0")
    source_type = Column(String(100))  # upload, api, database, generated, etc.
    source_config = Column(JSON)  # Configuration for data source
    parent_dataset_id = Column(String(255))  # For derived datasets
    transformation_steps = Column(JSON)  # Applied transformations
    
    # ML-specific metadata
    target_column = Column(String(255))  # For supervised learning
    feature_columns = Column(JSON)  # List of feature column names
    categorical_columns = Column(JSON)  # List of categorical columns
    numerical_columns = Column(JSON)  # List of numerical columns
    text_columns = Column(JSON)  # List of text columns
    datetime_columns = Column(JSON)  # List of datetime columns
    
    # Data splits
    train_split_path = Column(String(1000))
    validation_split_path = Column(String(1000))
    test_split_path = Column(String(1000))
    split_config = Column(JSON)  # Train/validation/test split configuration
    
    # Status and lifecycle
    status = Column(String(50), default="draft")  # draft, validated, ready, processing, error
    processing_status = Column(String(50), default="none")  # none, processing, completed, failed
    validation_status = Column(String(50), default="not_validated")  # not_validated, validating, passed, failed
    
    # Usage tracking
    download_count = Column(Integer, default=0)
    last_accessed = Column(DateTime)
    usage_stats = Column(JSON)  # Usage analytics
    
    # Metadata and tagging
    tags = Column(JSON)  # List of tags
    metadata = Column(JSON)  # Additional metadata
    license = Column(String(255))  # Data license
    citation = Column(Text)  # How to cite this dataset
    
    # Ownership and permissions
    created_by = Column(String(255), nullable=False)
    organization_id = Column(String(255))
    is_public = Column(Boolean, default=False)
    access_level = Column(String(50), default="private")  # private, organization, public
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_modified = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    versions = relationship("DatasetVersion", back_populates="dataset")
    transformations = relationship("DatasetTransformation", back_populates="dataset")

class DatasetVersion(Base):
    """Dataset versioning"""
    __tablename__ = "dataset_versions"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_id = Column(PG_UUID(as_uuid=True), ForeignKey('datasets.id'), nullable=False)
    
    version_number = Column(String(50), nullable=False)
    version_name = Column(String(255))
    description = Column(Text)
    
    # Version-specific metadata
    file_path = Column(String(1000))
    file_size_bytes = Column(Integer)
    checksum = Column(String(128))
    row_count = Column(Integer)
    column_count = Column(Integer)
    
    # Changes from previous version
    changes = Column(JSON)  # Description of changes
    diff_summary = Column(JSON)  # Statistical differences
    
    # Status
    is_current = Column(Boolean, default=False)
    created_by = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    dataset = relationship("Dataset", back_populates="versions")

class DatasetTransformation(Base):
    """Track dataset transformations"""
    __tablename__ = "dataset_transformations"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_id = Column(PG_UUID(as_uuid=True), ForeignKey('datasets.id'), nullable=False)
    
    # Transformation details
    transformation_type = Column(String(100))  # clean, normalize, feature_engineer, split, etc.
    transformation_name = Column(String(255))
    description = Column(Text)
    
    # Configuration
    config = Column(JSON)  # Transformation parameters
    code = Column(Text)  # Transformation code/script
    
    # Input/Output
    input_columns = Column(JSON)
    output_columns = Column(JSON)
    output_dataset_id = Column(String(255))  # If creates new dataset
    
    # Execution details
    status = Column(String(50), default="pending")  # pending, running, completed, failed
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    execution_log = Column(Text)
    error_message = Column(Text)
    
    # Metadata
    created_by = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    dataset = relationship("Dataset", back_populates="transformations")

# Pydantic Models
class DatasetCreate(BaseModel):
    name: str
    display_name: Optional[str] = None
    description: Optional[str] = None
    dataset_type: str = "tabular"
    source_type: str = "upload"
    source_config: Optional[Dict[str, Any]] = {}
    target_column: Optional[str] = None
    tags: Optional[List[str]] = []
    license: Optional[str] = None
    is_public: bool = False
    access_level: str = "private"

class DatasetUpdate(BaseModel):
    display_name: Optional[str] = None
    description: Optional[str] = None
    target_column: Optional[str] = None
    tags: Optional[List[str]] = None
    license: Optional[str] = None
    citation: Optional[str] = None
    is_public: Optional[bool] = None
    access_level: Optional[str] = None

class DatasetResponse(BaseModel):
    id: str
    name: str
    display_name: Optional[str]
    description: Optional[str]
    dataset_type: str
    data_format: Optional[str]
    file_size_bytes: Optional[int]
    row_count: Optional[int]
    column_count: Optional[int]
    version: str
    status: str
    created_by: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True

class DatasetTransformationCreate(BaseModel):
    transformation_type: str
    transformation_name: str
    description: Optional[str] = None
    config: Dict[str, Any]
    input_columns: Optional[List[str]] = None

class DataSplitConfig(BaseModel):
    train_size: float = 0.7
    validation_size: float = 0.15
    test_size: float = 0.15
    random_state: Optional[int] = 42
    stratify: bool = False
    stratify_column: Optional[str] = None

# Dependency injection
def get_db():
    pass

def get_current_user():
    return "current_user_id"

# Dataset Management Service
class DatasetManagementService:
    """Service for managing datasets and data operations"""
    
    def __init__(self, db: Session, storage_path: str = "/tmp/raia_datasets"):
        self.db = db
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Supported file formats
        self.format_handlers = {
            'csv': self._handle_csv,
            'json': self._handle_json,
            'parquet': self._handle_parquet,
            'excel': self._handle_excel,
            'hdf5': self._handle_hdf5,
            'sqlite': self._handle_sqlite
        }
        
        # Data processors
        self.processors = {
            'clean': self._process_clean,
            'normalize': self._process_normalize,
            'feature_engineer': self._process_feature_engineer,
            'split': self._process_split,
            'sample': self._process_sample
        }
    
    async def create_dataset(self, dataset_data: DatasetCreate, user_id: str) -> Dataset:
        """Create a new dataset record"""
        
        dataset = Dataset(
            name=dataset_data.name,
            display_name=dataset_data.display_name or dataset_data.name,
            description=dataset_data.description,
            dataset_type=dataset_data.dataset_type,
            source_type=dataset_data.source_type,
            source_config=dataset_data.source_config,
            target_column=dataset_data.target_column,
            tags=dataset_data.tags,
            license=dataset_data.license,
            is_public=dataset_data.is_public,
            access_level=dataset_data.access_level,
            created_by=user_id,
            organization_id=self._get_user_org(user_id)
        )
        
        self.db.add(dataset)
        self.db.commit()
        self.db.refresh(dataset)
        
        logger.info(f"Created dataset {dataset.name} (ID: {dataset.id})")
        return dataset
    
    async def upload_dataset_file(self, dataset_id: str, file: UploadFile, user_id: str) -> Dict[str, Any]:
        """Upload and process dataset file"""
        
        dataset = self._get_dataset_by_id(dataset_id, user_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Create dataset directory
        dataset_dir = self.storage_path / str(dataset.id)
        dataset_dir.mkdir(exist_ok=True)
        
        # Save file
        file_path = dataset_dir / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Calculate checksum
        checksum = hashlib.sha256(file_path.read_bytes()).hexdigest()
        
        # Detect format and process
        data_format = self._detect_file_format(file.filename)
        processing_result = await self._process_dataset_file(file_path, data_format, dataset)
        
        # Update dataset record
        dataset.file_path = str(file_path)
        dataset.file_size_bytes = file_path.stat().st_size
        dataset.data_format = data_format
        dataset.checksum = checksum
        dataset.row_count = processing_result.get('row_count')
        dataset.column_count = processing_result.get('column_count')
        dataset.schema = processing_result.get('schema')
        dataset.column_metadata = processing_result.get('column_metadata')
        dataset.statistics = processing_result.get('statistics')
        dataset.categorical_columns = processing_result.get('categorical_columns')
        dataset.numerical_columns = processing_result.get('numerical_columns')
        dataset.text_columns = processing_result.get('text_columns')
        dataset.datetime_columns = processing_result.get('datetime_columns')
        dataset.status = "ready" if processing_result.get('success') else "error"
        dataset.processing_status = "completed"
        dataset.last_modified = datetime.utcnow()
        
        self.db.commit()
        
        return {
            'success': processing_result.get('success', False),
            'message': processing_result.get('message', 'Dataset processed'),
            'file_size': file_path.stat().st_size,
            'row_count': processing_result.get('row_count'),
            'column_count': processing_result.get('column_count'),
            'checksum': checksum
        }
    
    async def get_datasets(self, user_id: str, skip: int = 0, limit: int = 100, filters: Optional[Dict[str, Any]] = None) -> List[Dataset]:
        """Get datasets with filtering and pagination"""
        
        query = self.db.query(Dataset).filter(
            (Dataset.created_by == user_id) |
            (Dataset.is_public == True) |
            (Dataset.organization_id == self._get_user_org(user_id))
        )
        
        # Apply filters
        if filters:
            if filters.get('dataset_type'):
                query = query.filter(Dataset.dataset_type == filters['dataset_type'])
            if filters.get('status'):
                query = query.filter(Dataset.status == filters['status'])
            if filters.get('tags'):
                for tag in filters['tags']:
                    query = query.filter(Dataset.tags.contains([tag]))
        
        return query.order_by(desc(Dataset.created_at)).offset(skip).limit(limit).all()
    
    async def get_dataset(self, dataset_id: str, user_id: str) -> Dataset:
        """Get a specific dataset"""
        
        dataset = self._get_dataset_by_id(dataset_id, user_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Update last accessed
        dataset.last_accessed = datetime.utcnow()
        self.db.commit()
        
        return dataset
    
    async def get_dataset_preview(self, dataset_id: str, user_id: str, rows: int = 100) -> Dict[str, Any]:
        """Get a preview of dataset contents"""
        
        dataset = await self.get_dataset(dataset_id, user_id)
        
        if not dataset.file_path or not Path(dataset.file_path).exists():
            raise HTTPException(status_code=404, detail="Dataset file not found")
        
        try:
            # Load data based on format
            df = await self._load_dataset(dataset.file_path, dataset.data_format, rows)
            
            preview_data = {
                'columns': df.columns.tolist(),
                'data': df.head(rows).to_dict('records'),
                'total_rows': len(df),
                'sample_rows': min(rows, len(df)),
                'dtypes': df.dtypes.astype(str).to_dict()
            }
            
            # Add basic statistics for numerical columns
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numerical_cols:
                preview_data['statistics'] = df[numerical_cols].describe().to_dict()
            
            return preview_data
            
        except Exception as e:
            logger.error(f"Failed to generate dataset preview: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to load dataset: {str(e)}")
    
    async def split_dataset(self, dataset_id: str, split_config: DataSplitConfig, user_id: str) -> Dict[str, Any]:
        """Split dataset into train/validation/test sets"""
        
        dataset = await self.get_dataset(dataset_id, user_id)
        
        if not dataset.file_path or not Path(dataset.file_path).exists():
            raise HTTPException(status_code=404, detail="Dataset file not found")
        
        try:
            # Load full dataset
            df = await self._load_dataset(dataset.file_path, dataset.data_format)
            
            # Prepare target for stratification
            y = None
            if split_config.stratify and dataset.target_column and dataset.target_column in df.columns:
                y = df[dataset.target_column]
            
            # Create splits
            train_size = split_config.train_size
            val_size = split_config.validation_size
            test_size = split_config.test_size
            
            # Normalize sizes if they don't sum to 1
            total = train_size + val_size + test_size
            if abs(total - 1.0) > 1e-6:
                train_size /= total
                val_size /= total
                test_size /= total
            
            # First split: separate test set
            train_val_df, test_df = train_test_split(
                df,
                test_size=test_size,
                random_state=split_config.random_state,
                stratify=y if split_config.stratify else None
            )
            
            # Second split: separate train and validation
            if val_size > 0:
                relative_val_size = val_size / (train_size + val_size)
                y_train_val = None
                if split_config.stratify and dataset.target_column and dataset.target_column in train_val_df.columns:
                    y_train_val = train_val_df[dataset.target_column]
                
                train_df, val_df = train_test_split(
                    train_val_df,
                    test_size=relative_val_size,
                    random_state=split_config.random_state,
                    stratify=y_train_val if split_config.stratify else None
                )
            else:
                train_df = train_val_df
                val_df = pd.DataFrame()
            
            # Save split files
            dataset_dir = Path(dataset.file_path).parent
            
            # Save train split
            train_path = dataset_dir / f"{dataset.name}_train.{dataset.data_format}"
            await self._save_dataset(train_df, train_path, dataset.data_format)
            dataset.train_split_path = str(train_path)
            
            # Save validation split
            if not val_df.empty:
                val_path = dataset_dir / f"{dataset.name}_val.{dataset.data_format}"
                await self._save_dataset(val_df, val_path, dataset.data_format)
                dataset.validation_split_path = str(val_path)
            
            # Save test split
            test_path = dataset_dir / f"{dataset.name}_test.{dataset.data_format}"
            await self._save_dataset(test_df, test_path, dataset.data_format)
            dataset.test_split_path = str(test_path)
            
            # Update dataset record
            dataset.split_config = split_config.dict()
            dataset.updated_at = datetime.utcnow()
            
            self.db.commit()
            
            return {
                'success': True,
                'train_size': len(train_df),
                'validation_size': len(val_df) if not val_df.empty else 0,
                'test_size': len(test_df),
                'train_path': str(train_path),
                'validation_path': str(val_path) if not val_df.empty else None,
                'test_path': str(test_path)
            }
            
        except Exception as e:
            logger.error(f"Dataset splitting failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Dataset splitting failed: {str(e)}")
    
    async def generate_data_profile(self, dataset_id: str, user_id: str) -> Dict[str, Any]:
        """Generate comprehensive data profiling report"""
        
        dataset = await self.get_dataset(dataset_id, user_id)
        
        if not dataset.file_path or not Path(dataset.file_path).exists():
            raise HTTPException(status_code=404, detail="Dataset file not found")
        
        try:
            # Load dataset
            df = await self._load_dataset(dataset.file_path, dataset.data_format)
            
            # Generate profile report
            profile = ProfileReport(
                df,
                title=f"Data Profile - {dataset.display_name or dataset.name}",
                explorative=True,
                minimal=False
            )
            
            # Save HTML report
            dataset_dir = Path(dataset.file_path).parent
            report_path = dataset_dir / f"{dataset.name}_profile.html"
            profile.to_file(report_path)
            
            # Extract key insights
            description = profile.get_description()
            
            profile_summary = {
                'overview': {
                    'n_vars': description['table']['n_var'],
                    'n_obs': description['table']['n'],
                    'missing_cells': description['table']['n_cells_missing'],
                    'missing_cells_perc': description['table']['p_cells_missing'],
                    'duplicate_rows': description['table']['n_duplicates'],
                    'duplicate_rows_perc': description['table']['p_duplicates']
                },
                'variables': {},
                'report_path': str(report_path)
            }
            
            # Process variable information
            for var_name, var_info in description['variables'].items():
                profile_summary['variables'][var_name] = {
                    'type': var_info.get('type', 'unknown'),
                    'missing_count': var_info.get('n_missing', 0),
                    'missing_perc': var_info.get('p_missing', 0.0),
                    'unique_count': var_info.get('n_unique', 0),
                    'unique_perc': var_info.get('p_unique', 0.0)
                }
                
                # Add type-specific statistics
                if var_info.get('type') == 'Numeric':
                    profile_summary['variables'][var_name].update({
                        'mean': var_info.get('mean'),
                        'std': var_info.get('std'),
                        'min': var_info.get('min'),
                        'max': var_info.get('max'),
                        'median': var_info.get('50%')
                    })
                elif var_info.get('type') == 'Categorical':
                    profile_summary['variables'][var_name].update({
                        'mode': var_info.get('mode'),
                        'frequency': var_info.get('max_count')
                    })
            
            # Update dataset with profile data
            dataset.data_quality_report = profile_summary
            dataset.updated_at = datetime.utcnow()
            self.db.commit()
            
            return profile_summary
            
        except Exception as e:
            logger.error(f"Data profiling failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Data profiling failed: {str(e)}")
    
    async def transform_dataset(self, dataset_id: str, transformation: DatasetTransformationCreate, user_id: str) -> Dict[str, Any]:
        """Apply transformation to dataset"""
        
        dataset = self._get_dataset_by_id(dataset_id, user_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Create transformation record
        transform_record = DatasetTransformation(
            dataset_id=dataset.id,
            transformation_type=transformation.transformation_type,
            transformation_name=transformation.transformation_name,
            description=transformation.description,
            config=transformation.config,
            input_columns=transformation.input_columns,
            status="running",
            start_time=datetime.utcnow(),
            created_by=user_id
        )
        
        self.db.add(transform_record)
        self.db.commit()
        self.db.refresh(transform_record)
        
        try:
            # Apply transformation
            processor = self.processors.get(transformation.transformation_type)
            if not processor:
                raise ValueError(f"Unsupported transformation type: {transformation.transformation_type}")
            
            result = await processor(dataset, transformation.config, user_id)
            
            # Update transformation record
            transform_record.status = "completed"
            transform_record.end_time = datetime.utcnow()
            transform_record.output_columns = result.get('output_columns')
            transform_record.output_dataset_id = result.get('output_dataset_id')
            
            self.db.commit()
            
            return {
                'success': True,
                'transformation_id': str(transform_record.id),
                'result': result
            }
            
        except Exception as e:
            # Update transformation record with error
            transform_record.status = "failed"
            transform_record.end_time = datetime.utcnow()
            transform_record.error_message = str(e)
            self.db.commit()
            
            logger.error(f"Dataset transformation failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Transformation failed: {str(e)}")
    
    # Private methods
    def _get_dataset_by_id(self, dataset_id: str, user_id: str) -> Optional[Dataset]:
        """Get dataset by ID with access control"""
        
        return self.db.query(Dataset).filter(
            Dataset.id == dataset_id,
            (
                (Dataset.created_by == user_id) |
                (Dataset.is_public == True) |
                (Dataset.organization_id == self._get_user_org(user_id))
            )
        ).first()
    
    def _get_user_org(self, user_id: str) -> Optional[str]:
        """Get user's organization ID"""
        return "default_org"
    
    def _detect_file_format(self, filename: str) -> str:
        """Detect file format from filename"""
        
        extension = filename.split('.')[-1].lower()
        
        format_map = {
            'csv': 'csv',
            'json': 'json',
            'jsonl': 'json',
            'parquet': 'parquet',
            'xlsx': 'excel',
            'xls': 'excel',
            'h5': 'hdf5',
            'hdf5': 'hdf5',
            'db': 'sqlite',
            'sqlite': 'sqlite',
            'sqlite3': 'sqlite'
        }
        
        return format_map.get(extension, 'csv')
    
    async def _process_dataset_file(self, file_path: Path, data_format: str, dataset: Dataset) -> Dict[str, Any]:
        """Process uploaded dataset file"""
        
        try:
            # Load data
            df = await self._load_dataset(file_path, data_format)
            
            # Generate basic statistics
            stats = self._generate_basic_statistics(df)
            
            # Detect column types
            column_info = self._analyze_columns(df)
            
            # Generate schema
            schema = self._generate_schema(df)
            
            return {
                'success': True,
                'row_count': len(df),
                'column_count': len(df.columns),
                'statistics': stats,
                'column_metadata': column_info['metadata'],
                'categorical_columns': column_info['categorical'],
                'numerical_columns': column_info['numerical'],
                'text_columns': column_info['text'],
                'datetime_columns': column_info['datetime'],
                'schema': schema,
                'message': 'Dataset processed successfully'
            }
            
        except Exception as e:
            logger.error(f"Dataset processing failed: {str(e)}")
            return {
                'success': False,
                'message': f'Dataset processing failed: {str(e)}'
            }
    
    async def _load_dataset(self, file_path: Path, data_format: str, nrows: Optional[int] = None) -> pd.DataFrame:
        """Load dataset from file"""
        
        if data_format == 'csv':
            return pd.read_csv(file_path, nrows=nrows)
        elif data_format == 'json':
            return pd.read_json(file_path, lines=True, nrows=nrows)
        elif data_format == 'parquet':
            df = pd.read_parquet(file_path)
            return df.head(nrows) if nrows else df
        elif data_format == 'excel':
            return pd.read_excel(file_path, nrows=nrows)
        elif data_format == 'hdf5':
            # Assume first key for simplicity
            with pd.HDFStore(file_path, mode='r') as store:
                key = store.keys()[0]
                return pd.read_hdf(file_path, key=key, nrows=nrows)
        elif data_format == 'sqlite':
            # Assume first table for simplicity
            conn = sqlite3.connect(file_path)
            tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
            if not tables.empty:
                table_name = tables.iloc[0]['name']
                query = f"SELECT * FROM {table_name}"
                if nrows:
                    query += f" LIMIT {nrows}"
                return pd.read_sql_query(query, conn)
            else:
                raise ValueError("No tables found in SQLite database")
        else:
            raise ValueError(f"Unsupported data format: {data_format}")
    
    async def _save_dataset(self, df: pd.DataFrame, file_path: Path, data_format: str):
        """Save dataset to file"""
        
        if data_format == 'csv':
            df.to_csv(file_path, index=False)
        elif data_format == 'json':
            df.to_json(file_path, orient='records', lines=True)
        elif data_format == 'parquet':
            df.to_parquet(file_path, index=False)
        elif data_format == 'excel':
            df.to_excel(file_path, index=False)
        elif data_format == 'hdf5':
            df.to_hdf(file_path, key='data', mode='w', index=False)
        else:
            # Default to CSV
            df.to_csv(file_path, index=False)
    
    def _generate_basic_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate basic dataset statistics"""
        
        stats = {
            'row_count': len(df),
            'column_count': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'data_types': df.dtypes.astype(str).to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
        }
        
        # Numerical statistics
        numerical_df = df.select_dtypes(include=[np.number])
        if not numerical_df.empty:
            stats['numerical_summary'] = numerical_df.describe().to_dict()
        
        # Categorical statistics
        categorical_df = df.select_dtypes(include=['object', 'category'])
        if not categorical_df.empty:
            stats['categorical_summary'] = {}
            for col in categorical_df.columns:
                stats['categorical_summary'][col] = {
                    'unique_count': df[col].nunique(),
                    'top_values': df[col].value_counts().head(5).to_dict()
                }
        
        return stats
    
    def _analyze_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze column types and properties"""
        
        column_metadata = {}
        categorical_columns = []
        numerical_columns = []
        text_columns = []
        datetime_columns = []
        
        for col in df.columns:
            col_info = {
                'dtype': str(df[col].dtype),
                'null_count': int(df[col].isnull().sum()),
                'null_percentage': float(df[col].isnull().sum() / len(df) * 100),
                'unique_count': int(df[col].nunique()),
                'unique_percentage': float(df[col].nunique() / len(df) * 100)
            }
            
            # Detect column type
            if pd.api.types.is_numeric_dtype(df[col]):
                numerical_columns.append(col)
                col_info['column_type'] = 'numerical'
                if not df[col].isnull().all():
                    col_info.update({
                        'min': float(df[col].min()),
                        'max': float(df[col].max()),
                        'mean': float(df[col].mean()),
                        'std': float(df[col].std())
                    })
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                datetime_columns.append(col)
                col_info['column_type'] = 'datetime'
                if not df[col].isnull().all():
                    col_info.update({
                        'min_date': df[col].min().isoformat(),
                        'max_date': df[col].max().isoformat()
                    })
            else:
                # Check if it's categorical or text
                if df[col].nunique() / len(df) < 0.5 and df[col].nunique() < 100:
                    categorical_columns.append(col)
                    col_info['column_type'] = 'categorical'
                    col_info['categories'] = df[col].value_counts().head(10).to_dict()
                else:
                    text_columns.append(col)
                    col_info['column_type'] = 'text'
                    if not df[col].isnull().all():
                        text_lengths = df[col].astype(str).str.len()
                        col_info.update({
                            'avg_length': float(text_lengths.mean()),
                            'max_length': int(text_lengths.max()),
                            'min_length': int(text_lengths.min())
                        })
            
            column_metadata[col] = col_info
        
        return {
            'metadata': column_metadata,
            'categorical': categorical_columns,
            'numerical': numerical_columns,
            'text': text_columns,
            'datetime': datetime_columns
        }
    
    def _generate_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate JSON schema for the dataset"""
        
        schema = {
            'type': 'object',
            'properties': {},
            'required': []
        }
        
        for col in df.columns:
            dtype = df[col].dtype
            
            if pd.api.types.is_integer_dtype(dtype):
                col_schema = {'type': 'integer'}
            elif pd.api.types.is_float_dtype(dtype):
                col_schema = {'type': 'number'}
            elif pd.api.types.is_bool_dtype(dtype):
                col_schema = {'type': 'boolean'}
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                col_schema = {'type': 'string', 'format': 'date-time'}
            else:
                col_schema = {'type': 'string'}
            
            # Add constraints based on data
            if not df[col].isnull().any():
                schema['required'].append(col)
            
            if pd.api.types.is_numeric_dtype(df[col]) and not df[col].isnull().all():
                col_schema['minimum'] = float(df[col].min())
                col_schema['maximum'] = float(df[col].max())
            
            schema['properties'][col] = col_schema
        
        return schema
    
    # Transformation processors
    async def _process_clean(self, dataset: Dataset, config: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Clean dataset (remove nulls, duplicates, etc.)"""
        
        df = await self._load_dataset(Path(dataset.file_path), dataset.data_format)
        original_shape = df.shape
        
        # Remove duplicates if requested
        if config.get('remove_duplicates', False):
            df = df.drop_duplicates()
        
        # Handle missing values
        missing_strategy = config.get('missing_strategy', 'remove')
        if missing_strategy == 'remove':
            df = df.dropna()
        elif missing_strategy == 'fill':
            fill_method = config.get('fill_method', 'mean')
            if fill_method == 'mean':
                df = df.fillna(df.mean())
            elif fill_method == 'median':
                df = df.fillna(df.median())
            elif fill_method == 'mode':
                df = df.fillna(df.mode().iloc[0])
            elif fill_method == 'constant':
                fill_value = config.get('fill_value', 0)
                df = df.fillna(fill_value)
        
        # Save cleaned dataset
        output_path = Path(dataset.file_path).parent / f"{dataset.name}_cleaned.{dataset.data_format}"
        await self._save_dataset(df, output_path, dataset.data_format)
        
        return {
            'output_path': str(output_path),
            'original_shape': original_shape,
            'cleaned_shape': df.shape,
            'rows_removed': original_shape[0] - df.shape[0],
            'output_columns': df.columns.tolist()
        }
    
    async def _process_normalize(self, dataset: Dataset, config: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Normalize numerical columns"""
        
        df = await self._load_dataset(Path(dataset.file_path), dataset.data_format)
        
        # Get numerical columns to normalize
        columns_to_normalize = config.get('columns', dataset.numerical_columns or [])
        method = config.get('method', 'standard')  # standard, minmax, robust
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        elif method == 'robust':
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        # Apply normalization
        df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
        
        # Save normalized dataset
        output_path = Path(dataset.file_path).parent / f"{dataset.name}_normalized.{dataset.data_format}"
        await self._save_dataset(df, output_path, dataset.data_format)
        
        return {
            'output_path': str(output_path),
            'normalized_columns': columns_to_normalize,
            'method': method,
            'output_columns': df.columns.tolist()
        }
    
    async def _process_feature_engineer(self, dataset: Dataset, config: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Apply feature engineering transformations"""
        
        df = await self._load_dataset(Path(dataset.file_path), dataset.data_format)
        
        # One-hot encode categorical variables
        if config.get('one_hot_encode', False):
            categorical_cols = config.get('categorical_columns', dataset.categorical_columns or [])
            df = pd.get_dummies(df, columns=categorical_cols, prefix_sep='_')
        
        # Create polynomial features
        if config.get('polynomial_features', False):
            from sklearn.preprocessing import PolynomialFeatures
            poly_cols = config.get('polynomial_columns', dataset.numerical_columns or [])
            degree = config.get('polynomial_degree', 2)
            
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            poly_features = poly.fit_transform(df[poly_cols])
            poly_feature_names = poly.get_feature_names_out(poly_cols)
            
            # Add polynomial features to dataframe
            poly_df = pd.DataFrame(poly_features[:, len(poly_cols):], 
                                 columns=poly_feature_names[len(poly_cols):])
            df = pd.concat([df, poly_df], axis=1)
        
        # Feature selection
        if config.get('feature_selection', False) and dataset.target_column:
            n_features = config.get('n_features_to_select', 10)
            score_func = f_classif if dataset.dataset_type == 'classification' else SelectKBest
            
            selector = SelectKBest(score_func=score_func, k=n_features)
            feature_cols = [col for col in df.columns if col != dataset.target_column]
            
            X = df[feature_cols]
            y = df[dataset.target_column]
            
            selected_features = selector.fit_transform(X, y)
            selected_feature_names = selector.get_feature_names_out(feature_cols)
            
            # Reconstruct dataframe with selected features
            df = df[[dataset.target_column]].copy()
            for i, feature_name in enumerate(selected_feature_names):
                df[feature_name] = selected_features[:, i]
        
        # Save engineered dataset
        output_path = Path(dataset.file_path).parent / f"{dataset.name}_engineered.{dataset.data_format}"
        await self._save_dataset(df, output_path, dataset.data_format)
        
        return {
            'output_path': str(output_path),
            'original_features': len(dataset.feature_columns or []),
            'new_features': len(df.columns) - (1 if dataset.target_column else 0),
            'output_columns': df.columns.tolist()
        }
    
    async def _process_split(self, dataset: Dataset, config: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Split dataset (already implemented in split_dataset method)"""
        
        split_config = DataSplitConfig(**config)
        return await self.split_dataset(str(dataset.id), split_config, user_id)
    
    async def _process_sample(self, dataset: Dataset, config: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Sample dataset"""
        
        df = await self._load_dataset(Path(dataset.file_path), dataset.data_format)
        
        sample_method = config.get('method', 'random')  # random, stratified, systematic
        sample_size = config.get('sample_size', 1000)
        
        if sample_method == 'random':
            sampled_df = df.sample(n=min(sample_size, len(df)), random_state=config.get('random_state', 42))
        elif sample_method == 'stratified' and dataset.target_column:
            from sklearn.model_selection import train_test_split
            sampled_df, _ = train_test_split(
                df, 
                train_size=min(sample_size, len(df)), 
                stratify=df[dataset.target_column],
                random_state=config.get('random_state', 42)
            )
        elif sample_method == 'systematic':
            step = len(df) // sample_size
            indices = range(0, len(df), max(step, 1))[:sample_size]
            sampled_df = df.iloc[indices]
        else:
            # Default to random
            sampled_df = df.sample(n=min(sample_size, len(df)), random_state=config.get('random_state', 42))
        
        # Save sampled dataset
        output_path = Path(dataset.file_path).parent / f"{dataset.name}_sample.{dataset.data_format}"
        await self._save_dataset(sampled_df, output_path, dataset.data_format)
        
        return {
            'output_path': str(output_path),
            'original_size': len(df),
            'sample_size': len(sampled_df),
            'sample_method': sample_method,
            'output_columns': sampled_df.columns.tolist()
        }

# API Endpoints
@router.post("/", response_model=DatasetResponse)
async def create_dataset(
    dataset_data: DatasetCreate,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Create a new dataset"""
    
    service = DatasetManagementService(db)
    dataset = await service.create_dataset(dataset_data, current_user)
    return dataset

@router.post("/{dataset_id}/upload")
async def upload_dataset_file(
    dataset_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Upload dataset file"""
    
    service = DatasetManagementService(db)
    result = await service.upload_dataset_file(dataset_id, file, current_user)
    return result

@router.get("/", response_model=List[DatasetResponse])
async def get_datasets(
    skip: int = 0,
    limit: int = 100,
    dataset_type: Optional[str] = None,
    status: Optional[str] = None,
    tags: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Get datasets with filtering"""
    
    filters = {}
    if dataset_type:
        filters['dataset_type'] = dataset_type
    if status:
        filters['status'] = status
    if tags:
        filters['tags'] = tags.split(',')
    
    service = DatasetManagementService(db)
    datasets = await service.get_datasets(current_user, skip, limit, filters)
    return datasets

@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
    dataset_id: str,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Get a specific dataset"""
    
    service = DatasetManagementService(db)
    dataset = await service.get_dataset(dataset_id, current_user)
    return dataset

@router.get("/{dataset_id}/preview")
async def get_dataset_preview(
    dataset_id: str,
    rows: int = Query(default=100, le=1000),
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Get dataset preview"""
    
    service = DatasetManagementService(db)
    preview = await service.get_dataset_preview(dataset_id, current_user, rows)
    return preview

@router.post("/{dataset_id}/split")
async def split_dataset(
    dataset_id: str,
    split_config: DataSplitConfig,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Split dataset into train/validation/test sets"""
    
    service = DatasetManagementService(db)
    result = await service.split_dataset(dataset_id, split_config, current_user)
    return result

@router.post("/{dataset_id}/profile")
async def generate_data_profile(
    dataset_id: str,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Generate data profiling report"""
    
    service = DatasetManagementService(db)
    profile = await service.generate_data_profile(dataset_id, current_user)
    return profile

@router.post("/{dataset_id}/transform")
async def transform_dataset(
    dataset_id: str,
    transformation: DatasetTransformationCreate,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Apply transformation to dataset"""
    
    service = DatasetManagementService(db)
    result = await service.transform_dataset(dataset_id, transformation, current_user)
    return result

@router.get("/{dataset_id}/download")
async def download_dataset(
    dataset_id: str,
    split: Optional[str] = None,  # train, validation, test
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Download dataset file"""
    
    service = DatasetManagementService(db)
    dataset = await service.get_dataset(dataset_id, current_user)
    
    # Determine which file to download
    if split == 'train' and dataset.train_split_path:
        file_path = dataset.train_split_path
        filename = f"{dataset.name}_train.{dataset.data_format}"
    elif split == 'validation' and dataset.validation_split_path:
        file_path = dataset.validation_split_path
        filename = f"{dataset.name}_val.{dataset.data_format}"
    elif split == 'test' and dataset.test_split_path:
        file_path = dataset.test_split_path
        filename = f"{dataset.name}_test.{dataset.data_format}"
    else:
        file_path = dataset.file_path
        filename = f"{dataset.name}.{dataset.data_format}"
    
    if not file_path or not Path(file_path).exists():
        raise HTTPException(status_code=404, detail="Dataset file not found")
    
    # Update download count
    dataset.download_count += 1
    dataset.last_accessed = datetime.utcnow()
    service.db.commit()
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='application/octet-stream'
    )

@router.delete("/{dataset_id}")
async def delete_dataset(
    dataset_id: str,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Delete a dataset and its files"""
    
    service = DatasetManagementService(db)
    dataset = service._get_dataset_by_id(dataset_id, current_user)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    if dataset.created_by != current_user:
        raise HTTPException(status_code=403, detail="Permission denied")
    
    # Delete files
    if dataset.file_path:
        dataset_dir = Path(dataset.file_path).parent
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
    
    # Delete database records (cascading)
    service.db.delete(dataset)
    service.db.commit()
    
    logger.info(f"Deleted dataset {dataset.name} (ID: {dataset.id})")
    
    return {'success': True, 'message': 'Dataset deleted successfully'}