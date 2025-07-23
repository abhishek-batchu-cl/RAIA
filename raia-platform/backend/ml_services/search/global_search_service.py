# Global Search Service
import os
import json
import uuid
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from sqlalchemy.orm import Session
from sqlalchemy import Column, String, DateTime, Text, Boolean, JSON, UUID as PG_UUID, Integer, Float, ForeignKey, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

# Text search and indexing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Full-text search
try:
    import whoosh
    from whoosh.index import create_index, open_dir
    from whoosh.fields import Schema, TEXT, ID, DATETIME, NUMERIC, KEYWORD
    from whoosh.qparser import QueryParser, MultifieldParser
    from whoosh.query import And, Or, Term
    WHOOSH_AVAILABLE = True
except ImportError:
    WHOOSH_AVAILABLE = False

# Fuzzy matching
from difflib import SequenceMatcher
import re

Base = declarative_base()
logger = logging.getLogger(__name__)

@dataclass
class SearchQuery:
    """Search query parameters"""
    query: str
    types: List[str] = None  # Filter by resource types
    categories: List[str] = None
    authors: List[str] = None
    tags: List[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    sort_by: str = "relevance"  # relevance, date, popularity, name
    limit: int = 20
    offset: int = 0
    user_id: Optional[str] = None

@dataclass
class SearchResult:
    """Search result item"""
    id: str
    type: str
    title: str
    description: str
    category: str
    url: Optional[str]
    metadata: Dict[str, Any]
    relevance_score: float
    created_at: datetime
    updated_at: datetime

@dataclass
class SearchStats:
    """Search analytics"""
    total_results: int
    search_time_ms: float
    results_by_type: Dict[str, int]
    suggestions: List[str]

class SearchIndex(Base):
    """Store searchable content index"""
    __tablename__ = "search_index"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    resource_id = Column(String(255), nullable=False, index=True)
    resource_type = Column(String(50), nullable=False, index=True)
    
    # Content fields
    title = Column(String(1000), nullable=False)
    description = Column(Text)
    content = Column(Text)  # Full text content
    category = Column(String(255), index=True)
    tags = Column(JSON)  # List of tags
    
    # Metadata
    author_id = Column(String(255), index=True)
    author_name = Column(String(255))
    organization_id = Column(String(255), index=True)
    
    # URLs and navigation
    url = Column(String(1000))
    parent_id = Column(String(255))
    
    # Searchable metrics
    popularity_score = Column(Float, default=0.0)  # Views, downloads, stars, etc.
    quality_score = Column(Float, default=0.0)  # User ratings, completion rate, etc.
    recency_score = Column(Float, default=0.0)  # How recently updated
    
    # Status and permissions
    is_active = Column(Boolean, default=True)
    is_public = Column(Boolean, default=True)
    access_level = Column(String(50), default='public')  # public, internal, private
    
    # Timestamps
    resource_created_at = Column(DateTime)
    resource_updated_at = Column(DateTime)
    indexed_at = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Search optimization
    search_vector = Column(Text)  # Pre-computed search vector
    boost_factor = Column(Float, default=1.0)  # Manual boost for important content

class SearchAnalytics(Base):
    """Track search queries and results"""
    __tablename__ = "search_analytics"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query_id = Column(String(255), unique=True, nullable=False)
    
    # Query details
    query_text = Column(String(1000), nullable=False)
    query_type = Column(String(50))  # search, command, filter
    user_id = Column(String(255))
    session_id = Column(String(255))
    
    # Results
    total_results = Column(Integer, default=0)
    clicked_result_id = Column(String(255))  # Which result was clicked
    clicked_position = Column(Integer)  # Position of clicked result
    search_time_ms = Column(Float)
    
    # Context
    filters_applied = Column(JSON)
    source_page = Column(String(255))
    user_agent = Column(String(1000))
    
    # Timestamps
    searched_at = Column(DateTime, default=datetime.utcnow)

class GlobalSearchService:
    """Service for global search across all resources"""
    
    def __init__(self, db_session: Session = None,
                 index_path: str = "/tmp/raia_search_index"):
        self.db = db_session
        self.index_path = index_path
        
        # Ensure directories exist
        os.makedirs(index_path, exist_ok=True)
        
        # Initialize search components
        self.tfidf_vectorizer = None
        self.search_index = None
        self.whoosh_index = None
        
        # Initialize Whoosh if available
        if WHOOSH_AVAILABLE:
            self._initialize_whoosh_index()
        
        # Search configuration
        self.max_results = 1000
        self.fuzzy_threshold = 0.6
        
        # Resource type weights for ranking
        self.type_weights = {
            'model': 1.0,
            'experiment': 0.9,
            'dataset': 0.8,
            'workflow': 0.9,
            'report': 0.7,
            'user': 0.6,
            'command': 1.2,  # Commands should rank high
            'page': 0.5
        }

    async def index_resource(self, resource_id: str, resource_type: str,
                           title: str, description: str, content: str = None,
                           category: str = None, tags: List[str] = None,
                           metadata: Dict[str, Any] = None,
                           author_id: str = None, author_name: str = None,
                           url: str = None, is_public: bool = True) -> Dict[str, Any]:
        """Index a resource for search"""
        
        try:
            # Calculate scores
            popularity_score = self._calculate_popularity_score(metadata or {})
            quality_score = self._calculate_quality_score(metadata or {})
            recency_score = self._calculate_recency_score(metadata or {})
            
            # Create or update search index record
            search_record = None
            if self.db:
                search_record = self.db.query(SearchIndex).filter(
                    SearchIndex.resource_id == resource_id
                ).first()
                
                if search_record:
                    # Update existing record
                    search_record.title = title
                    search_record.description = description
                    search_record.content = content or description
                    search_record.category = category
                    search_record.tags = tags
                    search_record.author_id = author_id
                    search_record.author_name = author_name
                    search_record.url = url
                    search_record.popularity_score = popularity_score
                    search_record.quality_score = quality_score
                    search_record.recency_score = recency_score
                    search_record.is_public = is_public
                    search_record.last_updated = datetime.utcnow()
                else:
                    # Create new record
                    search_record = SearchIndex(
                        resource_id=resource_id,
                        resource_type=resource_type,
                        title=title,
                        description=description,
                        content=content or description,
                        category=category,
                        tags=tags,
                        author_id=author_id,
                        author_name=author_name,
                        url=url,
                        popularity_score=popularity_score,
                        quality_score=quality_score,
                        recency_score=recency_score,
                        is_public=is_public,
                        resource_created_at=metadata.get('created_at'),
                        resource_updated_at=metadata.get('updated_at')
                    )
                    self.db.add(search_record)
                
                self.db.commit()
            
            # Update Whoosh index
            if WHOOSH_AVAILABLE and self.whoosh_index:
                writer = self.whoosh_index.writer()
                
                # Prepare tags for indexing
                tags_text = ' '.join(tags or [])
                
                # Add or update document
                writer.update_document(
                    id=resource_id,
                    type=resource_type,
                    title=title,
                    description=description,
                    content=content or description,
                    category=category or '',
                    tags=tags_text,
                    author=author_name or '',
                    url=url or '',
                    popularity=popularity_score,
                    quality=quality_score,
                    recency=recency_score,
                    created=metadata.get('created_at', datetime.utcnow()),
                    updated=metadata.get('updated_at', datetime.utcnow())
                )
                writer.commit()
            
            logger.info(f"Indexed resource {resource_id} ({resource_type})")
            
            return {
                'success': True,
                'resource_id': resource_id,
                'message': 'Resource indexed successfully'
            }
            
        except Exception as e:
            logger.error(f"Error indexing resource {resource_id}: {str(e)}")
            return {
                'success': False,
                'error': f'Failed to index resource: {str(e)}'
            }

    async def search(self, query: SearchQuery) -> Dict[str, Any]:
        """Perform global search"""
        
        start_time = datetime.utcnow()
        query_id = f"query_{uuid.uuid4().hex[:8]}"
        
        try:
            results = []
            
            if WHOOSH_AVAILABLE and self.whoosh_index:
                # Use Whoosh for full-text search
                results = await self._whoosh_search(query)
            else:
                # Fallback to database search
                results = await self._database_search(query)
            
            # Apply additional filtering and ranking
            results = self._apply_filters(results, query)
            results = self._rank_results(results, query)
            
            # Apply pagination
            total_results = len(results)
            paginated_results = results[query.offset:query.offset + query.limit]
            
            # Calculate search time
            search_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Generate statistics
            stats = SearchStats(
                total_results=total_results,
                search_time_ms=search_time,
                results_by_type=self._count_by_type(results),
                suggestions=await self._generate_suggestions(query.query, results)
            )
            
            # Log search analytics
            await self._log_search(query_id, query, stats)
            
            return {
                'results': [asdict(result) for result in paginated_results],
                'stats': asdict(stats),
                'query_id': query_id,
                'total': total_results,
                'offset': query.offset,
                'limit': query.limit
            }
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return {
                'results': [],
                'stats': asdict(SearchStats(0, 0, {}, [])),
                'error': str(e)
            }

    async def get_suggestions(self, partial_query: str, limit: int = 5) -> List[str]:
        """Get search suggestions for autocomplete"""
        
        if not self.db or len(partial_query) < 2:
            return []
        
        # Get suggestions from indexed titles and descriptions
        suggestions = self.db.query(SearchIndex.title).filter(
            SearchIndex.title.ilike(f'%{partial_query}%'),
            SearchIndex.is_active == True,
            SearchIndex.is_public == True
        ).distinct().limit(limit).all()
        
        # Also get suggestions from common search terms
        recent_queries = self.db.query(SearchAnalytics.query_text).filter(
            SearchAnalytics.query_text.ilike(f'%{partial_query}%'),
            SearchAnalytics.total_results > 0,
            SearchAnalytics.searched_at >= datetime.utcnow() - timedelta(days=30)
        ).distinct().limit(limit).all()
        
        # Combine and deduplicate
        all_suggestions = [s[0] for s in suggestions] + [q[0] for q in recent_queries]
        unique_suggestions = list(dict.fromkeys(all_suggestions))
        
        return unique_suggestions[:limit]

    async def get_popular_searches(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get popular search queries"""
        
        if not self.db:
            return []
        
        # Get most frequent searches from last 30 days
        popular = self.db.query(
            SearchAnalytics.query_text,
            func.count(SearchAnalytics.id).label('search_count'),
            func.avg(SearchAnalytics.total_results).label('avg_results')
        ).filter(
            SearchAnalytics.searched_at >= datetime.utcnow() - timedelta(days=30),
            SearchAnalytics.total_results > 0
        ).group_by(SearchAnalytics.query_text).order_by(
            func.count(SearchAnalytics.id).desc()
        ).limit(limit).all()
        
        return [
            {
                'query': query,
                'search_count': count,
                'avg_results': int(avg_results)
            }
            for query, count, avg_results in popular
        ]

    async def get_search_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get search analytics and insights"""
        
        if not self.db:
            return {}
        
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Basic metrics
        total_searches = self.db.query(SearchAnalytics).filter(
            SearchAnalytics.searched_at >= start_date
        ).count()
        
        successful_searches = self.db.query(SearchAnalytics).filter(
            SearchAnalytics.searched_at >= start_date,
            SearchAnalytics.total_results > 0
        ).count()
        
        clicked_searches = self.db.query(SearchAnalytics).filter(
            SearchAnalytics.searched_at >= start_date,
            SearchAnalytics.clicked_result_id.isnot(None)
        ).count()
        
        # Average search time
        avg_search_time = self.db.query(
            func.avg(SearchAnalytics.search_time_ms)
        ).filter(
            SearchAnalytics.searched_at >= start_date
        ).scalar() or 0
        
        # Top search types
        search_types = self.db.query(
            SearchAnalytics.query_type,
            func.count(SearchAnalytics.id).label('count')
        ).filter(
            SearchAnalytics.searched_at >= start_date
        ).group_by(SearchAnalytics.query_type).all()
        
        # Zero result queries (need attention)
        zero_results = self.db.query(SearchAnalytics.query_text).filter(
            SearchAnalytics.searched_at >= start_date,
            SearchAnalytics.total_results == 0
        ).distinct().limit(20).all()
        
        return {
            'period_days': days,
            'total_searches': total_searches,
            'success_rate': (successful_searches / total_searches * 100) if total_searches > 0 else 0,
            'click_through_rate': (clicked_searches / successful_searches * 100) if successful_searches > 0 else 0,
            'avg_search_time_ms': avg_search_time,
            'search_types': {qt: count for qt, count in search_types},
            'zero_result_queries': [q[0] for q in zero_results]
        }

    async def reindex_all(self, batch_size: int = 100) -> Dict[str, Any]:
        """Rebuild the entire search index"""
        
        if not self.db:
            return {'success': False, 'error': 'Database not available'}
        
        try:
            # Clear existing index
            if WHOOSH_AVAILABLE and self.whoosh_index:
                writer = self.whoosh_index.writer()
                writer.commit(mergetype=whoosh.writing.CLEAR)
            
            # Get all searchable resources
            resources = self.db.query(SearchIndex).filter(
                SearchIndex.is_active == True
            ).all()
            
            indexed_count = 0
            for resource in resources:
                await self.index_resource(
                    resource_id=resource.resource_id,
                    resource_type=resource.resource_type,
                    title=resource.title,
                    description=resource.description,
                    content=resource.content,
                    category=resource.category,
                    tags=resource.tags,
                    metadata={
                        'created_at': resource.resource_created_at,
                        'updated_at': resource.resource_updated_at
                    },
                    author_id=resource.author_id,
                    author_name=resource.author_name,
                    url=resource.url,
                    is_public=resource.is_public
                )
                indexed_count += 1
                
                if indexed_count % batch_size == 0:
                    logger.info(f"Reindexed {indexed_count} resources")
            
            logger.info(f"Reindexing completed: {indexed_count} resources")
            
            return {
                'success': True,
                'indexed_count': indexed_count,
                'message': f'Successfully reindexed {indexed_count} resources'
            }
            
        except Exception as e:
            logger.error(f"Reindexing failed: {str(e)}")
            return {
                'success': False,
                'error': f'Reindexing failed: {str(e)}'
            }

    # Private methods
    def _initialize_whoosh_index(self):
        """Initialize Whoosh search index"""
        
        try:
            # Define schema
            schema = Schema(
                id=ID(stored=True, unique=True),
                type=TEXT(stored=True),
                title=TEXT(stored=True),
                description=TEXT(stored=True),
                content=TEXT(),
                category=TEXT(stored=True),
                tags=KEYWORD(stored=True),
                author=TEXT(stored=True),
                url=TEXT(stored=True),
                popularity=NUMERIC(stored=True),
                quality=NUMERIC(stored=True),
                recency=NUMERIC(stored=True),
                created=DATETIME(stored=True),
                updated=DATETIME(stored=True)
            )
            
            index_dir = os.path.join(self.index_path, 'whoosh')
            os.makedirs(index_dir, exist_ok=True)
            
            if os.listdir(index_dir):
                # Open existing index
                from whoosh import index
                self.whoosh_index = index.open_dir(index_dir)
            else:
                # Create new index
                self.whoosh_index = create_index(schema, index_dir)
            
            logger.info("Whoosh index initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Whoosh index: {str(e)}")
            self.whoosh_index = None

    async def _whoosh_search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform search using Whoosh"""
        
        results = []
        
        try:
            with self.whoosh_index.searcher() as searcher:
                # Build query
                parser = MultifieldParser(
                    ["title", "description", "content", "tags", "category", "author"],
                    self.whoosh_index.schema
                )
                
                whoosh_query = parser.parse(query.query)
                
                # Apply filters
                if query.types:
                    type_terms = [Term("type", t) for t in query.types]
                    type_query = Or(type_terms)
                    whoosh_query = And([whoosh_query, type_query])
                
                # Execute search
                search_results = searcher.search(whoosh_query, limit=self.max_results)
                
                for hit in search_results:
                    result = SearchResult(
                        id=hit['id'],
                        type=hit['type'],
                        title=hit['title'],
                        description=hit['description'],
                        category=hit['category'] or '',
                        url=hit['url'],
                        metadata={
                            'author': hit['author'],
                            'popularity': hit['popularity'],
                            'quality': hit['quality'],
                            'recency': hit['recency'],
                            'created_at': hit['created'].isoformat() if hit['created'] else None,
                            'updated_at': hit['updated'].isoformat() if hit['updated'] else None
                        },
                        relevance_score=hit.score,
                        created_at=hit['created'] or datetime.utcnow(),
                        updated_at=hit['updated'] or datetime.utcnow()
                    )
                    results.append(result)
        
        except Exception as e:
            logger.error(f"Whoosh search error: {str(e)}")
        
        return results

    async def _database_search(self, query: SearchQuery) -> List[SearchResult]:
        """Fallback database search"""
        
        if not self.db:
            return []
        
        # Build base query
        db_query = self.db.query(SearchIndex).filter(
            SearchIndex.is_active == True,
            SearchIndex.is_public == True
        )
        
        # Text search
        search_terms = query.query.lower().split()
        for term in search_terms:
            search_pattern = f'%{term}%'
            db_query = db_query.filter(
                (SearchIndex.title.ilike(search_pattern)) |
                (SearchIndex.description.ilike(search_pattern)) |
                (SearchIndex.content.ilike(search_pattern))
            )
        
        # Apply filters
        if query.types:
            db_query = db_query.filter(SearchIndex.resource_type.in_(query.types))
        
        if query.categories:
            db_query = db_query.filter(SearchIndex.category.in_(query.categories))
        
        if query.authors:
            db_query = db_query.filter(SearchIndex.author_id.in_(query.authors))
        
        # Execute query
        records = db_query.limit(self.max_results).all()
        
        # Convert to search results
        results = []
        for record in records:
            # Calculate relevance score
            relevance = self._calculate_relevance_score(record, query.query)
            
            result = SearchResult(
                id=record.resource_id,
                type=record.resource_type,
                title=record.title,
                description=record.description,
                category=record.category or '',
                url=record.url,
                metadata={
                    'author': record.author_name,
                    'author_id': record.author_id,
                    'popularity': record.popularity_score,
                    'quality': record.quality_score,
                    'recency': record.recency_score,
                    'tags': record.tags,
                    'created_at': record.resource_created_at.isoformat() if record.resource_created_at else None,
                    'updated_at': record.resource_updated_at.isoformat() if record.resource_updated_at else None
                },
                relevance_score=relevance,
                created_at=record.resource_created_at or datetime.utcnow(),
                updated_at=record.resource_updated_at or datetime.utcnow()
            )
            results.append(result)
        
        return results

    def _apply_filters(self, results: List[SearchResult], query: SearchQuery) -> List[SearchResult]:
        """Apply additional filters to search results"""
        
        filtered = results
        
        # Date range filter
        if query.date_from or query.date_to:
            filtered = [
                r for r in filtered
                if (not query.date_from or r.updated_at >= query.date_from) and
                   (not query.date_to or r.updated_at <= query.date_to)
            ]
        
        # Tag filter
        if query.tags:
            filtered = [
                r for r in filtered
                if r.metadata.get('tags') and
                   any(tag in r.metadata['tags'] for tag in query.tags)
            ]
        
        return filtered

    def _rank_results(self, results: List[SearchResult], query: SearchQuery) -> List[SearchResult]:
        """Rank and sort search results"""
        
        # Calculate final scores
        for result in results:
            base_score = result.relevance_score
            type_weight = self.type_weights.get(result.type, 1.0)
            popularity = result.metadata.get('popularity', 0)
            quality = result.metadata.get('quality', 0)
            recency = result.metadata.get('recency', 0)
            
            # Combined ranking score
            final_score = (
                base_score * type_weight * 0.4 +  # Relevance (40%)
                popularity * 0.3 +                 # Popularity (30%)
                quality * 0.2 +                    # Quality (20%)
                recency * 0.1                      # Recency (10%)
            )
            
            result.relevance_score = final_score
        
        # Sort based on sort_by parameter
        if query.sort_by == 'relevance':
            results.sort(key=lambda x: x.relevance_score, reverse=True)
        elif query.sort_by == 'date':
            results.sort(key=lambda x: x.updated_at, reverse=True)
        elif query.sort_by == 'popularity':
            results.sort(key=lambda x: x.metadata.get('popularity', 0), reverse=True)
        elif query.sort_by == 'name':
            results.sort(key=lambda x: x.title.lower())
        
        return results

    def _calculate_relevance_score(self, record: SearchIndex, query: str) -> float:
        """Calculate relevance score for database search"""
        
        query_lower = query.lower()
        score = 0.0
        
        # Title match (highest weight)
        if query_lower in record.title.lower():
            score += 3.0
            if record.title.lower().startswith(query_lower):
                score += 1.0  # Bonus for prefix match
        
        # Description match
        if query_lower in record.description.lower():
            score += 2.0
        
        # Content match
        if record.content and query_lower in record.content.lower():
            score += 1.0
        
        # Category match
        if record.category and query_lower in record.category.lower():
            score += 1.5
        
        # Tag matches
        if record.tags:
            for tag in record.tags:
                if query_lower in tag.lower():
                    score += 1.0
        
        # Author match
        if record.author_name and query_lower in record.author_name.lower():
            score += 0.5
        
        # Fuzzy matching for typos
        for word in query.split():
            title_similarity = SequenceMatcher(None, word.lower(), record.title.lower()).ratio()
            if title_similarity > self.fuzzy_threshold:
                score += title_similarity * 0.5
        
        return score

    def _calculate_popularity_score(self, metadata: Dict[str, Any]) -> float:
        """Calculate popularity score from metadata"""
        
        score = 0.0
        
        # Various popularity indicators
        downloads = metadata.get('downloads', 0)
        views = metadata.get('views', 0)
        stars = metadata.get('stars', 0)
        forks = metadata.get('forks', 0)
        likes = metadata.get('likes', 0)
        
        # Weight different metrics
        score += downloads * 0.001  # 1000 downloads = 1 point
        score += views * 0.0001     # 10000 views = 1 point
        score += stars * 0.1        # 10 stars = 1 point
        score += forks * 0.2        # 5 forks = 1 point
        score += likes * 0.05       # 20 likes = 1 point
        
        # Cap at reasonable maximum
        return min(score, 10.0)

    def _calculate_quality_score(self, metadata: Dict[str, Any]) -> float:
        """Calculate quality score from metadata"""
        
        score = 0.0
        
        # Quality indicators
        rating = metadata.get('rating', 0)
        accuracy = metadata.get('accuracy', 0)
        completeness = metadata.get('completeness', 0)
        documentation_quality = metadata.get('documentation_quality', 0)
        
        # Combine metrics
        if rating > 0:
            score += rating  # 0-5 scale
        
        if accuracy > 0:
            score += accuracy * 5  # 0-1 scale -> 0-5 scale
        
        score += completeness * 2   # 0-1 scale -> 0-2 scale
        score += documentation_quality * 2  # 0-1 scale -> 0-2 scale
        
        # Normalize to 0-10 scale
        return min(score, 10.0)

    def _calculate_recency_score(self, metadata: Dict[str, Any]) -> float:
        """Calculate recency score based on update time"""
        
        updated_at = metadata.get('updated_at')
        if not updated_at:
            return 0.0
        
        if isinstance(updated_at, str):
            try:
                updated_at = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
            except:
                return 0.0
        
        # Days since update
        days_old = (datetime.utcnow() - updated_at.replace(tzinfo=None)).days
        
        # Decay function - newer is better
        if days_old <= 1:
            return 10.0
        elif days_old <= 7:
            return 8.0
        elif days_old <= 30:
            return 6.0
        elif days_old <= 90:
            return 4.0
        elif days_old <= 365:
            return 2.0
        else:
            return 1.0

    def _count_by_type(self, results: List[SearchResult]) -> Dict[str, int]:
        """Count results by type"""
        
        counts = {}
        for result in results:
            counts[result.type] = counts.get(result.type, 0) + 1
        
        return counts

    async def _generate_suggestions(self, query: str, results: List[SearchResult]) -> List[str]:
        """Generate search suggestions"""
        
        suggestions = []
        
        # If no results, suggest corrections
        if len(results) == 0:
            # Simple suggestions for common terms
            common_corrections = {
                'modle': 'model',
                'experment': 'experiment',
                'datset': 'dataset',
                'workfow': 'workflow',
                'algoritm': 'algorithm',
                'accurcy': 'accuracy'
            }
            
            for typo, correction in common_corrections.items():
                if typo in query.lower():
                    suggestions.append(query.lower().replace(typo, correction))
        
        # If few results, suggest broader terms
        elif len(results) < 5:
            # Extract common categories from results
            categories = [r.category for r in results if r.category]
            if categories:
                suggestions.extend(list(set(categories)))
        
        return suggestions[:3]

    async def _log_search(self, query_id: str, query: SearchQuery, stats: SearchStats):
        """Log search analytics"""
        
        if not self.db:
            return
        
        try:
            analytics = SearchAnalytics(
                query_id=query_id,
                query_text=query.query,
                query_type='search',
                user_id=query.user_id,
                total_results=stats.total_results,
                search_time_ms=stats.search_time_ms,
                filters_applied={
                    'types': query.types,
                    'categories': query.categories,
                    'authors': query.authors,
                    'tags': query.tags,
                    'date_from': query.date_from.isoformat() if query.date_from else None,
                    'date_to': query.date_to.isoformat() if query.date_to else None,
                    'sort_by': query.sort_by
                }
            )
            
            self.db.add(analytics)
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Failed to log search analytics: {str(e)}")

# Factory function
def create_global_search_service(db_session: Session = None,
                                index_path: str = "/tmp/raia_search_index") -> GlobalSearchService:
    """Create and return a GlobalSearchService instance"""
    return GlobalSearchService(db_session, index_path)