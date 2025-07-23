# Model Marketplace Service
import os
import json
import uuid
import shutil
import hashlib
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, BinaryIO
from dataclasses import dataclass, asdict
from sqlalchemy.orm import Session
from sqlalchemy import Column, String, DateTime, Text, Boolean, JSON, UUID as PG_UUID, Integer, Float, ForeignKey, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import zipfile
import tarfile
from pathlib import Path

# File handling and validation
import magic
import pickle
import joblib
import torch
import tensorflow as tf
from PIL import Image

# Model analysis and metadata extraction
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

# Security scanning
import yaml
import subprocess
from packaging import version as pkg_version

# Search and recommendation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Content moderation
import re
from urllib.parse import urlparse

Base = declarative_base()
logger = logging.getLogger(__name__)

@dataclass
class ModelUploadConfig:
    """Configuration for model upload"""
    name: str
    description: str
    category: str
    subcategory: str
    tags: List[str]
    license: str
    version: str = "1.0.0"
    is_public: bool = True
    documentation_url: Optional[str] = None
    demo_url: Optional[str] = None
    github_url: Optional[str] = None
    paper_url: Optional[str] = None
    use_cases: List[str] = None
    requirements: List[str] = None
    pricing_type: str = "free"  # free, paid, freemium
    price: Optional[float] = None

@dataclass
class ModelSearchQuery:
    """Search query for models"""
    query: Optional[str] = None
    category: Optional[str] = None
    framework: Optional[str] = None
    license: Optional[str] = None
    tags: List[str] = None
    min_rating: Optional[float] = None
    sort_by: str = "popular"  # popular, recent, stars, downloads, rating
    limit: int = 20
    offset: int = 0

@dataclass
class ModelRating:
    """Model rating and review"""
    rating: float  # 1-5 stars
    review: Optional[str] = None
    use_case: Optional[str] = None
    performance_notes: Optional[str] = None

class ModelListing(Base):
    """Store model marketplace listings"""
    __tablename__ = "model_listings"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(String(255), unique=True, nullable=False, index=True)
    
    # Basic information
    name = Column(String(500), nullable=False)
    description = Column(Text, nullable=False)
    category = Column(String(100), nullable=False, index=True)
    subcategory = Column(String(100))
    tags = Column(JSON)  # List of strings
    
    # Version and status
    version = Column(String(50), nullable=False)
    status = Column(String(50), default='published')  # draft, published, archived, suspended
    is_public = Column(Boolean, default=True)
    is_featured = Column(Boolean, default=False)
    
    # Author information
    author_id = Column(String(255), nullable=False)
    author_username = Column(String(255), nullable=False)
    organization_id = Column(String(255))
    
    # Model metadata
    model_type = Column(String(100))  # classification, regression, nlp, computer_vision, etc.
    framework = Column(String(100))
    framework_version = Column(String(50))
    model_size_mb = Column(Float)
    inference_time_ms = Column(Float)
    
    # Performance metrics
    performance_metrics = Column(JSON)
    accuracy = Column(Float)  # Primary metric for easy filtering
    
    # Dataset information
    dataset_info = Column(JSON)
    training_data_size = Column(Integer)
    feature_count = Column(Integer)
    
    # File information
    model_file_path = Column(String(1000))
    model_file_size = Column(Integer)
    model_file_checksum = Column(String(255))
    archive_path = Column(String(1000))  # Full package with docs, examples, etc.
    
    # External links
    documentation_url = Column(String(1000))
    demo_url = Column(String(1000))
    github_url = Column(String(1000))
    paper_url = Column(String(1000))
    
    # Usage and requirements
    use_cases = Column(JSON)  # List of strings
    requirements = Column(JSON)  # List of strings
    python_version = Column(String(50))
    license = Column(String(100), nullable=False)
    
    # Pricing
    pricing_type = Column(String(50), default='free')  # free, paid, freemium
    price = Column(Float)
    currency = Column(String(10), default='USD')
    
    # Statistics
    downloads = Column(Integer, default=0)
    views = Column(Integer, default=0)
    stars = Column(Integer, default=0)
    forks = Column(Integer, default=0)
    rating_average = Column(Float, default=0.0)
    rating_count = Column(Integer, default=0)
    
    # Security and validation
    security_scan_status = Column(String(50), default='pending')  # pending, passed, warning, failed
    security_scan_date = Column(DateTime)
    security_issues = Column(JSON)
    malware_scan_status = Column(String(50), default='pending')
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    published_at = Column(DateTime)
    
    # Relationships
    ratings = relationship("ModelRatingRecord", back_populates="model")
    downloads_log = relationship("ModelDownloadLog", back_populates="model")

class ModelRatingRecord(Base):
    """Store model ratings and reviews"""
    __tablename__ = "model_ratings"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(String(255), ForeignKey('model_listings.model_id'), nullable=False)
    user_id = Column(String(255), nullable=False)
    
    # Rating details
    rating = Column(Float, nullable=False)  # 1-5 stars
    review = Column(Text)
    use_case = Column(String(500))
    performance_notes = Column(Text)
    
    # Helpful votes
    helpful_votes = Column(Integer, default=0)
    total_votes = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    model = relationship("ModelListing", back_populates="ratings")

class ModelDownloadLog(Base):
    """Log model downloads for analytics"""
    __tablename__ = "model_downloads"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(String(255), ForeignKey('model_listings.model_id'), nullable=False)
    user_id = Column(String(255))
    
    # Download details
    download_type = Column(String(50))  # model_file, full_package, documentation
    ip_address = Column(String(45))
    user_agent = Column(String(1000))
    
    # Timestamp
    downloaded_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    model = relationship("ModelListing", back_populates="downloads_log")

class ModelFork(Base):
    """Track model forks and derivatives"""
    __tablename__ = "model_forks"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    original_model_id = Column(String(255), nullable=False)
    forked_model_id = Column(String(255), nullable=False)
    fork_type = Column(String(50))  # full_fork, fine_tuned, adapted
    
    # Fork details
    user_id = Column(String(255), nullable=False)
    fork_reason = Column(Text)
    changes_description = Column(Text)
    
    # Timestamps
    forked_at = Column(DateTime, default=datetime.utcnow)

class ModelMarketplaceService:
    """Service for model marketplace operations"""
    
    def __init__(self, db_session: Session = None,
                 storage_path: str = "/tmp/raia_marketplace",
                 max_file_size_mb: int = 1000):
        self.db = db_session
        self.storage_path = storage_path
        self.max_file_size_mb = max_file_size_mb
        
        # Ensure directories exist
        os.makedirs(storage_path, exist_ok=True)
        os.makedirs(f"{storage_path}/models", exist_ok=True)
        os.makedirs(f"{storage_path}/archives", exist_ok=True)
        os.makedirs(f"{storage_path}/thumbnails", exist_ok=True)
        os.makedirs(f"{storage_path}/temp", exist_ok=True)
        
        # Security configuration
        self.allowed_file_types = {
            '.pkl', '.pickle', '.joblib', '.pt', '.pth', '.pb',
            '.h5', '.onnx', '.pmml', '.zip', '.tar.gz', '.tar'
        }
        
        # Content filters
        self.inappropriate_patterns = [
            r'\b(?:hate|violence|discriminat)\w*\b',
            r'\b(?:malware|virus|trojan)\b',
            r'\b(?:illegal|fraud|scam)\b'
        ]
        
        # Search index
        self.search_vectorizer = None
        self.search_index = None

    async def upload_model(self, config: ModelUploadConfig,
                          model_file: BinaryIO,
                          additional_files: Dict[str, BinaryIO] = None,
                          uploaded_by: str = None,
                          organization_id: str = None) -> Dict[str, Any]:
        """Upload a model to the marketplace"""
        
        model_id = f"model_{uuid.uuid4().hex[:12]}"
        
        try:
            # Validate configuration
            validation_result = await self._validate_model_config(config)
            if not validation_result['valid']:
                return {
                    'success': False,
                    'error': f"Invalid configuration: {validation_result['error']}"
                }
            
            # Validate and process model file
            file_validation = await self._validate_model_file(model_file)
            if not file_validation['valid']:
                return {
                    'success': False,
                    'error': f"Invalid model file: {file_validation['error']}"
                }
            
            # Create storage paths
            model_dir = os.path.join(self.storage_path, 'models', model_id)
            archive_dir = os.path.join(self.storage_path, 'archives', model_id)
            os.makedirs(model_dir, exist_ok=True)
            os.makedirs(archive_dir, exist_ok=True)
            
            # Save model file
            model_file_path = os.path.join(model_dir, f"model{file_validation['extension']}")
            with open(model_file_path, 'wb') as f:
                shutil.copyfileobj(model_file, f)
            
            # Calculate file metrics
            file_size = os.path.getsize(model_file_path)
            file_checksum = self._calculate_checksum(model_file_path)
            
            # Extract model metadata
            model_metadata = await self._extract_model_metadata(model_file_path, file_validation['file_type'])
            
            # Process additional files
            additional_paths = {}
            if additional_files:
                for filename, file_obj in additional_files.items():
                    file_path = os.path.join(archive_dir, filename)
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    with open(file_path, 'wb') as f:
                        shutil.copyfileobj(file_obj, f)
                    additional_paths[filename] = file_path
            
            # Create archive package
            archive_path = os.path.join(archive_dir, f"{model_id}_package.zip")
            await self._create_model_archive(model_file_path, additional_paths, archive_path, config)
            
            # Run security scans
            security_scan = await self._run_security_scan(model_file_path, additional_paths)
            
            # Create model listing
            model_listing = ModelListing(
                model_id=model_id,
                name=config.name,
                description=config.description,
                category=config.category,
                subcategory=config.subcategory,
                tags=config.tags,
                version=config.version,
                is_public=config.is_public,
                author_id=uploaded_by or 'anonymous',
                author_username=uploaded_by or 'anonymous',
                organization_id=organization_id,
                
                # Model metadata
                model_type=model_metadata.get('model_type'),
                framework=model_metadata.get('framework'),
                framework_version=model_metadata.get('framework_version'),
                model_size_mb=file_size / (1024 * 1024),
                
                # Performance metrics
                performance_metrics=model_metadata.get('performance_metrics', {}),
                accuracy=model_metadata.get('accuracy'),
                
                # Dataset information
                dataset_info=model_metadata.get('dataset_info', {}),
                feature_count=model_metadata.get('feature_count'),
                
                # File information
                model_file_path=model_file_path,
                model_file_size=file_size,
                model_file_checksum=file_checksum,
                archive_path=archive_path,
                
                # External links
                documentation_url=config.documentation_url,
                demo_url=config.demo_url,
                github_url=config.github_url,
                paper_url=config.paper_url,
                
                # Usage information
                use_cases=config.use_cases or [],
                requirements=config.requirements or [],
                license=config.license,
                
                # Pricing
                pricing_type=config.pricing_type,
                price=config.price,
                
                # Security scan results
                security_scan_status=security_scan['status'],
                security_scan_date=datetime.utcnow(),
                security_issues=security_scan.get('issues', []),
                malware_scan_status=security_scan.get('malware_status', 'pending'),
                
                # Set published date if public
                published_at=datetime.utcnow() if config.is_public else None,
                status='published' if config.is_public and security_scan['status'] != 'failed' else 'draft'
            )
            
            if self.db:
                self.db.add(model_listing)
                self.db.commit()
            
            # Update search index
            await self._update_search_index()
            
            logger.info(f"Model {model_id} uploaded successfully by {uploaded_by}")
            
            return {
                'success': True,
                'model_id': model_id,
                'status': model_listing.status,
                'security_scan': security_scan,
                'message': f'Model "{config.name}" uploaded successfully'
            }
            
        except Exception as e:
            logger.error(f"Error uploading model: {str(e)}")
            return {
                'success': False,
                'error': f'Upload failed: {str(e)}'
            }

    async def search_models(self, query: ModelSearchQuery) -> Dict[str, Any]:
        """Search models in the marketplace"""
        
        if not self.db:
            return {'models': [], 'total': 0}
        
        # Build base query
        db_query = self.db.query(ModelListing).filter(
            ModelListing.status == 'published',
            ModelListing.is_public == True
        )
        
        # Apply filters
        if query.category:
            db_query = db_query.filter(ModelListing.category == query.category)
        
        if query.framework:
            db_query = db_query.filter(ModelListing.framework == query.framework)
        
        if query.license:
            db_query = db_query.filter(ModelListing.license == query.license)
        
        if query.min_rating:
            db_query = db_query.filter(ModelListing.rating_average >= query.min_rating)
        
        if query.tags:
            for tag in query.tags:
                db_query = db_query.filter(ModelListing.tags.contains([tag]))
        
        # Text search
        if query.query:
            search_term = f"%{query.query}%"
            db_query = db_query.filter(
                (ModelListing.name.ilike(search_term)) |
                (ModelListing.description.ilike(search_term)) |
                (ModelListing.author_username.ilike(search_term))
            )
        
        # Get total count
        total = db_query.count()
        
        # Apply sorting
        if query.sort_by == 'popular':
            db_query = db_query.order_by((ModelListing.downloads + ModelListing.stars).desc())
        elif query.sort_by == 'recent':
            db_query = db_query.order_by(ModelListing.updated_at.desc())
        elif query.sort_by == 'stars':
            db_query = db_query.order_by(ModelListing.stars.desc())
        elif query.sort_by == 'downloads':
            db_query = db_query.order_by(ModelListing.downloads.desc())
        elif query.sort_by == 'rating':
            db_query = db_query.order_by(ModelListing.rating_average.desc())
        
        # Apply pagination
        db_query = db_query.offset(query.offset).limit(query.limit)
        
        # Execute query
        models = db_query.all()
        
        # Convert to dictionaries
        model_list = []
        for model in models:
            model_dict = self._model_to_dict(model)
            model_list.append(model_dict)
        
        return {
            'models': model_list,
            'total': total,
            'offset': query.offset,
            'limit': query.limit
        }

    async def get_model_details(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a model"""
        
        if not self.db:
            return None
        
        model = self.db.query(ModelListing).filter(
            ModelListing.model_id == model_id,
            ModelListing.status == 'published'
        ).first()
        
        if not model:
            return None
        
        # Increment view count
        model.views += 1
        self.db.commit()
        
        # Get recent ratings
        recent_ratings = self.db.query(ModelRatingRecord).filter(
            ModelRatingRecord.model_id == model_id
        ).order_by(ModelRatingRecord.created_at.desc()).limit(10).all()
        
        model_dict = self._model_to_dict(model)
        model_dict['recent_ratings'] = [self._rating_to_dict(r) for r in recent_ratings]
        
        return model_dict

    async def download_model(self, model_id: str, download_type: str = 'model_file',
                           user_id: str = None, ip_address: str = None,
                           user_agent: str = None) -> Dict[str, Any]:
        """Download a model file"""
        
        if not self.db:
            return {'success': False, 'error': 'Database not available'}
        
        model = self.db.query(ModelListing).filter(
            ModelListing.model_id == model_id,
            ModelListing.status == 'published'
        ).first()
        
        if not model:
            return {'success': False, 'error': 'Model not found'}
        
        # Determine file path
        if download_type == 'model_file':
            file_path = model.model_file_path
        elif download_type == 'full_package':
            file_path = model.archive_path
        else:
            return {'success': False, 'error': 'Invalid download type'}
        
        if not file_path or not os.path.exists(file_path):
            return {'success': False, 'error': 'File not found'}
        
        # Log download
        download_log = ModelDownloadLog(
            model_id=model_id,
            user_id=user_id,
            download_type=download_type,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.db.add(download_log)
        
        # Increment download count
        model.downloads += 1
        self.db.commit()
        
        return {
            'success': True,
            'file_path': file_path,
            'file_size': os.path.getsize(file_path),
            'filename': os.path.basename(file_path)
        }

    async def rate_model(self, model_id: str, rating: ModelRating,
                        user_id: str) -> Dict[str, Any]:
        """Rate and review a model"""
        
        if not self.db:
            return {'success': False, 'error': 'Database not available'}
        
        # Check if model exists
        model = self.db.query(ModelListing).filter(
            ModelListing.model_id == model_id
        ).first()
        
        if not model:
            return {'success': False, 'error': 'Model not found'}
        
        # Check if user already rated this model
        existing_rating = self.db.query(ModelRatingRecord).filter(
            ModelRatingRecord.model_id == model_id,
            ModelRatingRecord.user_id == user_id
        ).first()
        
        if existing_rating:
            # Update existing rating
            existing_rating.rating = rating.rating
            existing_rating.review = rating.review
            existing_rating.use_case = rating.use_case
            existing_rating.performance_notes = rating.performance_notes
            existing_rating.updated_at = datetime.utcnow()
        else:
            # Create new rating
            rating_record = ModelRatingRecord(
                model_id=model_id,
                user_id=user_id,
                rating=rating.rating,
                review=rating.review,
                use_case=rating.use_case,
                performance_notes=rating.performance_notes
            )
            self.db.add(rating_record)
            model.rating_count += 1
        
        # Recalculate average rating
        avg_rating = self.db.query(
            ModelRatingRecord.rating
        ).filter(ModelRatingRecord.model_id == model_id).all()
        
        if avg_rating:
            model.rating_average = sum(r[0] for r in avg_rating) / len(avg_rating)
        
        self.db.commit()
        
        return {
            'success': True,
            'message': 'Rating submitted successfully',
            'new_average': model.rating_average
        }

    async def star_model(self, model_id: str, user_id: str) -> Dict[str, Any]:
        """Star/unstar a model"""
        
        if not self.db:
            return {'success': False, 'error': 'Database not available'}
        
        model = self.db.query(ModelListing).filter(
            ModelListing.model_id == model_id
        ).first()
        
        if not model:
            return {'success': False, 'error': 'Model not found'}
        
        # For simplicity, just increment star count
        # In a real implementation, you'd track individual user stars
        model.stars += 1
        self.db.commit()
        
        return {
            'success': True,
            'message': 'Model starred',
            'total_stars': model.stars
        }

    async def get_marketplace_stats(self) -> Dict[str, Any]:
        """Get marketplace statistics"""
        
        if not self.db:
            return {}
        
        # Basic counts
        total_models = self.db.query(ModelListing).filter(
            ModelListing.status == 'published',
            ModelListing.is_public == True
        ).count()
        
        total_downloads = self.db.query(ModelListing).with_entities(
            ModelListing.downloads
        ).filter(
            ModelListing.status == 'published',
            ModelListing.is_public == True
        ).all()
        total_downloads = sum(d[0] for d in total_downloads)
        
        # Categories
        categories = self.db.query(
            ModelListing.category, 
            self.db.func.count(ModelListing.id)
        ).filter(
            ModelListing.status == 'published',
            ModelListing.is_public == True
        ).group_by(ModelListing.category).all()
        
        category_counts = {cat: count for cat, count in categories}
        
        # Top models
        trending_models = self.db.query(ModelListing).filter(
            ModelListing.status == 'published',
            ModelListing.is_public == True,
            ModelListing.updated_at >= datetime.utcnow() - timedelta(days=7)
        ).order_by(ModelListing.downloads.desc()).limit(10).all()
        
        featured_models = self.db.query(ModelListing).filter(
            ModelListing.status == 'published',
            ModelListing.is_public == True,
            ModelListing.is_featured == True
        ).limit(10).all()
        
        # Active publishers (unique authors)
        active_publishers = self.db.query(ModelListing.author_id).filter(
            ModelListing.status == 'published',
            ModelListing.is_public == True
        ).distinct().count()
        
        return {
            'total_models': total_models,
            'total_downloads': total_downloads,
            'active_publishers': active_publishers,
            'categories': category_counts,
            'trending_models': [m.model_id for m in trending_models],
            'featured_models': [m.model_id for m in featured_models]
        }

    async def get_recommendations(self, user_id: str = None, 
                                model_id: str = None,
                                limit: int = 5) -> List[Dict[str, Any]]:
        """Get model recommendations"""
        
        if not self.db:
            return []
        
        # Simple recommendation based on popularity and category
        # In a real implementation, this would use collaborative filtering or content-based recommendations
        
        base_query = self.db.query(ModelListing).filter(
            ModelListing.status == 'published',
            ModelListing.is_public == True
        )
        
        if model_id:
            # Get similar models in the same category
            target_model = self.db.query(ModelListing).filter(
                ModelListing.model_id == model_id
            ).first()
            
            if target_model:
                base_query = base_query.filter(
                    ModelListing.category == target_model.category,
                    ModelListing.model_id != model_id
                )
        
        recommendations = base_query.order_by(
            (ModelListing.downloads + ModelListing.stars * 10).desc()
        ).limit(limit).all()
        
        return [self._model_to_dict(model) for model in recommendations]

    # Private helper methods
    async def _validate_model_config(self, config: ModelUploadConfig) -> Dict[str, Any]:
        """Validate model upload configuration"""
        
        if not config.name or len(config.name) < 3:
            return {'valid': False, 'error': 'Model name must be at least 3 characters'}
        
        if not config.description or len(config.description) < 10:
            return {'valid': False, 'error': 'Description must be at least 10 characters'}
        
        if not config.category:
            return {'valid': False, 'error': 'Category is required'}
        
        if not config.license:
            return {'valid': False, 'error': 'License is required'}
        
        # Check for inappropriate content
        text_to_check = f"{config.name} {config.description}"
        for pattern in self.inappropriate_patterns:
            if re.search(pattern, text_to_check, re.IGNORECASE):
                return {'valid': False, 'error': 'Content violates community guidelines'}
        
        # Validate URLs
        for url_field, url in [
            ('documentation_url', config.documentation_url),
            ('demo_url', config.demo_url),
            ('github_url', config.github_url),
            ('paper_url', config.paper_url)
        ]:
            if url and not self._is_valid_url(url):
                return {'valid': False, 'error': f'Invalid {url_field}'}
        
        return {'valid': True}

    async def _validate_model_file(self, file_obj: BinaryIO) -> Dict[str, Any]:
        """Validate uploaded model file"""
        
        # Check file size
        file_obj.seek(0, 2)  # Seek to end
        file_size = file_obj.tell()
        file_obj.seek(0)  # Reset to beginning
        
        if file_size > self.max_file_size_mb * 1024 * 1024:
            return {
                'valid': False, 
                'error': f'File size exceeds maximum allowed size of {self.max_file_size_mb}MB'
            }
        
        # Read file header to determine type
        header = file_obj.read(1024)
        file_obj.seek(0)
        
        # Basic file type detection
        file_type = None
        extension = None
        
        if header.startswith(b'\x80\x03'):  # Pickle protocol 3
            file_type = 'pickle'
            extension = '.pkl'
        elif header.startswith(b'PK'):  # ZIP archive
            file_type = 'zip'
            extension = '.zip'
        elif b'TORCH' in header or header.startswith(b'\x50\x4b'):
            file_type = 'pytorch'
            extension = '.pt'
        elif b'tensorflow' in header.lower():
            file_type = 'tensorflow'
            extension = '.pb'
        elif header.startswith(b'\x08\x0a'):  # ONNX
            file_type = 'onnx'
            extension = '.onnx'
        elif header.startswith(b'\x89HDF'):  # HDF5
            file_type = 'hdf5'
            extension = '.h5'
        else:
            # Try to detect by content
            try:
                file_obj.seek(0)
                # Try to load as pickle
                pickle.load(file_obj)
                file_type = 'pickle'
                extension = '.pkl'
                file_obj.seek(0)
            except:
                return {'valid': False, 'error': 'Unsupported file format'}
        
        return {
            'valid': True,
            'file_type': file_type,
            'extension': extension,
            'size': file_size
        }

    async def _extract_model_metadata(self, file_path: str, file_type: str) -> Dict[str, Any]:
        """Extract metadata from model file"""
        
        metadata = {}
        
        try:
            if file_type == 'pickle':
                # Load pickle file and analyze
                model = pickle.load(open(file_path, 'rb'))
                
                # Detect framework
                if hasattr(model, '_estimator_type'):  # Scikit-learn
                    metadata['framework'] = 'scikit-learn'
                    metadata['model_type'] = getattr(model, '_estimator_type', 'unknown')
                    
                    # Try to get feature count
                    if hasattr(model, 'n_features_in_'):
                        metadata['feature_count'] = model.n_features_in_
                elif hasattr(model, 'layers'):  # Keras
                    metadata['framework'] = 'tensorflow'
                    metadata['model_type'] = 'neural_network'
                elif hasattr(model, 'parameters'):  # PyTorch
                    metadata['framework'] = 'pytorch'
                    metadata['model_type'] = 'neural_network'
                else:
                    metadata['framework'] = 'unknown'
                    metadata['model_type'] = 'unknown'
                
            elif file_type == 'pytorch':
                metadata['framework'] = 'pytorch'
                metadata['model_type'] = 'neural_network'
                
                # Try to load and get info
                try:
                    model = torch.load(file_path, map_location='cpu')
                    if isinstance(model, dict) and 'model_state_dict' in model:
                        # Count parameters
                        total_params = sum(p.numel() for p in model['model_state_dict'].values())
                        metadata['parameter_count'] = total_params
                except:
                    pass
                    
            elif file_type == 'tensorflow':
                metadata['framework'] = 'tensorflow'
                metadata['model_type'] = 'neural_network'
                
            elif file_type == 'onnx':
                metadata['framework'] = 'onnx'
                metadata['model_type'] = 'neural_network'
                
            elif file_type == 'hdf5':
                metadata['framework'] = 'tensorflow'  # Likely Keras
                metadata['model_type'] = 'neural_network'
                
        except Exception as e:
            logger.warning(f"Could not extract metadata from {file_path}: {str(e)}")
        
        return metadata

    async def _create_model_archive(self, model_path: str, 
                                  additional_files: Dict[str, str],
                                  archive_path: str,
                                  config: ModelUploadConfig):
        """Create a complete model package archive"""
        
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Add model file
            zf.write(model_path, f"model{os.path.splitext(model_path)[1]}")
            
            # Add additional files
            for filename, filepath in additional_files.items():
                zf.write(filepath, filename)
            
            # Create metadata file
            metadata = {
                'name': config.name,
                'description': config.description,
                'version': config.version,
                'category': config.category,
                'tags': config.tags,
                'license': config.license,
                'use_cases': config.use_cases,
                'requirements': config.requirements,
                'created_at': datetime.utcnow().isoformat()
            }
            
            zf.writestr('metadata.json', json.dumps(metadata, indent=2))
            
            # Create README
            readme_content = f"""# {config.name}

{config.description}

## Version
{config.version}

## Category
{config.category}

## License
{config.license}

## Use Cases
{chr(10).join(f'- {use_case}' for use_case in config.use_cases or [])}

## Requirements
{chr(10).join(f'- {req}' for req in config.requirements or [])}

## Tags
{', '.join(config.tags)}
"""
            zf.writestr('README.md', readme_content)

    async def _run_security_scan(self, model_path: str, 
                               additional_files: Dict[str, str]) -> Dict[str, Any]:
        """Run security scan on uploaded files"""
        
        scan_results = {
            'status': 'passed',
            'issues': [],
            'malware_status': 'passed'
        }
        
        try:
            # Basic file validation
            all_files = [model_path] + list(additional_files.values())
            
            for file_path in all_files:
                # Check file size (prevent zip bombs)
                file_size = os.path.getsize(file_path)
                if file_size > 500 * 1024 * 1024:  # 500MB limit
                    scan_results['issues'].append(f"File {file_path} exceeds size limit")
                    scan_results['status'] = 'warning'
                
                # Check for suspicious file extensions in archives
                if file_path.endswith(('.zip', '.tar', '.tar.gz')):
                    if zipfile.is_zipfile(file_path):
                        with zipfile.ZipFile(file_path, 'r') as zf:
                            for filename in zf.namelist():
                                if any(filename.endswith(ext) for ext in ['.exe', '.bat', '.sh', '.py']):
                                    scan_results['issues'].append(f"Suspicious file in archive: {filename}")
                                    scan_results['status'] = 'warning'
                
                # Basic content scanning for model files
                if file_path.endswith('.pkl'):
                    try:
                        with open(file_path, 'rb') as f:
                            content = f.read(1024)  # Read first 1KB
                            if b'exec(' in content or b'eval(' in content or b'__import__' in content:
                                scan_results['issues'].append("Potentially malicious code detected in pickle file")
                                scan_results['status'] = 'failed'
                    except:
                        pass
        
        except Exception as e:
            logger.error(f"Security scan error: {str(e)}")
            scan_results['status'] = 'warning'
            scan_results['issues'].append(f"Security scan incomplete: {str(e)}")
        
        return scan_results

    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA256 checksum of file"""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False

    async def _update_search_index(self):
        """Update search index for better search performance"""
        
        if not self.db:
            return
        
        # Get all published models
        models = self.db.query(ModelListing).filter(
            ModelListing.status == 'published',
            ModelListing.is_public == True
        ).all()
        
        # Create text corpus for search indexing
        documents = []
        for model in models:
            doc_text = f"{model.name} {model.description} {' '.join(model.tags or [])}"
            documents.append(doc_text)
        
        if documents:
            # Initialize or update TF-IDF vectorizer
            if self.search_vectorizer is None:
                self.search_vectorizer = TfidfVectorizer(
                    max_features=1000,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
            
            self.search_index = self.search_vectorizer.fit_transform(documents)

    def _model_to_dict(self, model: ModelListing) -> Dict[str, Any]:
        """Convert model database record to dictionary"""
        
        return {
            'id': model.model_id,
            'name': model.name,
            'description': model.description,
            'author': {
                'username': model.author_username,
                'verified': True,  # Placeholder - would need user verification system
                'reputation': 4.5  # Placeholder - would calculate from ratings
            },
            'version': model.version,
            'category': model.category,
            'subcategory': model.subcategory,
            'tags': model.tags or [],
            'accuracy': model.accuracy,
            'performance_metrics': model.performance_metrics or {},
            'dataset_info': {
                'name': model.dataset_info.get('name', 'Unknown') if model.dataset_info else 'Unknown',
                'size': model.training_data_size or 0,
                'features': model.feature_count or 0
            },
            'downloads': model.downloads,
            'stars': model.stars,
            'views': model.views,
            'forks': model.forks,
            'model_type': model.model_type,
            'framework': model.framework,
            'framework_version': model.framework_version,
            'model_size_mb': model.model_size_mb,
            'inference_time_ms': model.inference_time_ms,
            'license': model.license,
            'is_public': model.is_public,
            'security_scan': {
                'status': model.security_scan_status,
                'last_scan': model.security_scan_date.isoformat() if model.security_scan_date else None,
                'issues': model.security_issues or []
            },
            'created_at': model.created_at.isoformat(),
            'updated_at': model.updated_at.isoformat(),
            'published_at': model.published_at.isoformat() if model.published_at else None,
            'documentation_url': model.documentation_url,
            'demo_url': model.demo_url,
            'github_url': model.github_url,
            'paper_url': model.paper_url,
            'use_cases': model.use_cases or [],
            'requirements': model.requirements or [],
            'pricing': {
                'type': model.pricing_type,
                'price': model.price,
                'currency': model.currency
            } if model.pricing_type != 'free' else {'type': 'free'}
        }

    def _rating_to_dict(self, rating: ModelRatingRecord) -> Dict[str, Any]:
        """Convert rating record to dictionary"""
        
        return {
            'user_id': rating.user_id,
            'rating': rating.rating,
            'review': rating.review,
            'use_case': rating.use_case,
            'performance_notes': rating.performance_notes,
            'helpful_votes': rating.helpful_votes,
            'total_votes': rating.total_votes,
            'created_at': rating.created_at.isoformat(),
            'updated_at': rating.updated_at.isoformat()
        }

# Factory function
def create_marketplace_service(db_session: Session = None,
                             storage_path: str = "/tmp/raia_marketplace",
                             max_file_size_mb: int = 1000) -> ModelMarketplaceService:
    """Create and return a ModelMarketplaceService instance"""
    return ModelMarketplaceService(db_session, storage_path, max_file_size_mb)