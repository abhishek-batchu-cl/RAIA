"""
RAIA Platform - Document Processing Service
Handles document upload, processing, embedding generation, and vector search
"""

import asyncio
import hashlib
import logging
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from io import BytesIO

import structlog
import chromadb
from chromadb.config import Settings as ChromaSettings
import numpy as np
from sentence_transformers import SentenceTransformer
import PyPDF2
import aiofiles
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_database
from app.models.schemas import VectorDocument, VectorDocumentResponse, DocumentSearchResult, DocumentSearchResponse
from app.core.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


class DocumentProcessor:
    """Handles document processing with embeddings and chunking"""
    
    def __init__(self):
        self.logger = logger.bind(component="document_processor")
        self.embedding_models = {}  # Cache for embedding models
        self.default_embedding_model = "all-MiniLM-L6-v2"
        
    def _get_embedding_model(self, model_name: str = None) -> SentenceTransformer:
        """Get or create embedding model"""
        model_name = model_name or self.default_embedding_model
        
        if model_name not in self.embedding_models:
            self.logger.info(f"Loading embedding model: {model_name}")
            try:
                self.embedding_models[model_name] = SentenceTransformer(model_name)
                self.logger.info(f"Successfully loaded embedding model: {model_name}")
            except Exception as e:
                self.logger.error(f"Failed to load embedding model {model_name}", error=str(e))
                # Fallback to default if available
                if model_name != self.default_embedding_model and self.default_embedding_model in self.embedding_models:
                    return self.embedding_models[self.default_embedding_model]
                raise
        
        return self.embedding_models[model_name]
    
    async def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            text = ""
            async with aiofiles.open(file_path, 'rb') as file:
                content = await file.read()
                pdf_reader = PyPDF2.PdfReader(BytesIO(content))
                
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            
            self.logger.debug(f"Extracted {len(text)} characters from PDF")
            return text
            
        except Exception as e:
            self.logger.error("Failed to extract text from PDF", error=str(e))
            raise
    
    async def extract_text_from_txt(self, file_path: str) -> str:
        """Extract text from text file"""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as file:
                text = await file.read()
            
            self.logger.debug(f"Extracted {len(text)} characters from text file")
            return text
            
        except Exception as e:
            self.logger.error("Failed to extract text from text file", error=str(e))
            raise
    
    async def extract_text(self, file_path: str, file_type: str) -> str:
        """Extract text based on file type"""
        file_type = file_type.lower()
        
        if file_type == '.pdf':
            return await self.extract_text_from_pdf(file_path)
        elif file_type in ['.txt', '.md', '.text']:
            return await self.extract_text_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    def create_chunks(
        self, 
        text: str, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200
    ) -> List[str]:
        """Create overlapping chunks from text"""
        if len(text) <= chunk_size:
            return [text.strip()] if text.strip() else []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at natural boundaries
            if end < len(text):
                # Look for paragraph break
                para_break = text.rfind('\n\n', start, end)
                if para_break != -1 and para_break > start:
                    end = para_break
                else:
                    # Look for sentence break
                    sent_break = text.rfind('.', start, end)
                    if sent_break != -1 and sent_break > start:
                        end = sent_break + 1
                    else:
                        # Look for word boundary
                        word_break = text.rfind(' ', start, end)
                        if word_break != -1 and word_break > start:
                            end = word_break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            if end >= len(text):
                break
            
            start = max(end - chunk_overlap, start + 1)
        
        self.logger.debug(f"Created {len(chunks)} chunks from text")
        return chunks
    
    async def generate_embeddings(
        self, 
        chunks: List[str], 
        embedding_model: str = None
    ) -> List[List[float]]:
        """Generate embeddings for text chunks"""
        if not chunks:
            return []
        
        try:
            model = self._get_embedding_model(embedding_model)
            
            # Run embedding generation in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                lambda: model.encode(chunks, show_progress_bar=False).tolist()
            )
            
            self.logger.debug(f"Generated embeddings for {len(chunks)} chunks")
            return embeddings
            
        except Exception as e:
            self.logger.error("Failed to generate embeddings", error=str(e))
            raise


class VectorStore:
    """Manages vector storage and retrieval using ChromaDB"""
    
    def __init__(self, collection_name: str = "documents"):
        self.collection_name = collection_name
        self.logger = logger.bind(component="vector_store")
        
        # Initialize ChromaDB
        chroma_dir = settings.vector_db_path or "./data/chroma"
        os.makedirs(chroma_dir, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=None  # We'll provide embeddings directly
            )
            self.logger.info(f"Retrieved existing ChromaDB collection: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=None,
                metadata={"hnsw:space": "cosine"}
            )
            self.logger.info(f"Created new ChromaDB collection: {collection_name}")
    
    async def add_document_chunks(
        self,
        document_id: str,
        chunks: List[str],
        embeddings: List[List[float]],
        metadata: Dict[str, Any]
    ) -> int:
        """Add document chunks to vector store"""
        if not chunks or not embeddings:
            return 0
        
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        try:
            # Create chunk IDs and metadata
            chunk_ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]
            chunk_metadata = [
                {
                    **metadata,
                    "document_id": document_id,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
                for i in range(len(chunks))
            ]
            
            # Add to collection
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.collection.add(
                    documents=chunks,
                    embeddings=embeddings,
                    metadatas=chunk_metadata,
                    ids=chunk_ids
                )
            )
            
            self.logger.info(f"Added {len(chunks)} chunks to vector store for document {document_id}")
            return len(chunks)
            
        except Exception as e:
            self.logger.error(f"Failed to add chunks to vector store", error=str(e))
            raise
    
    async def search_documents(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentSearchResult]:
        """Search for similar document chunks"""
        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    where=filter_metadata,
                    include=["documents", "metadatas", "distances"]
                )
            )
            
            # Convert to search results
            search_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    metadata = results['metadatas'][0][i]
                    search_results.append(DocumentSearchResult(
                        document_chunk=results['documents'][0][i],
                        metadata=metadata,
                        similarity_score=1.0 - results['distances'][0][i],  # Convert distance to similarity
                        chunk_index=metadata.get('chunk_index', 0),
                        source_document=metadata.get('filename', 'Unknown')
                    ))
            
            self.logger.debug(f"Found {len(search_results)} similar chunks")
            return search_results
            
        except Exception as e:
            self.logger.error("Failed to search vector store", error=str(e))
            return []
    
    async def delete_document(self, document_id: str) -> int:
        """Delete all chunks for a document"""
        try:
            # Find chunks for this document
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self.collection.get(
                    where={"document_id": document_id},
                    include=["metadatas"]
                )
            )
            
            if results['ids']:
                # Delete chunks
                await loop.run_in_executor(
                    None,
                    lambda: self.collection.delete(ids=results['ids'])
                )
                
                self.logger.info(f"Deleted {len(results['ids'])} chunks for document {document_id}")
                return len(results['ids'])
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Failed to delete document chunks", error=str(e))
            raise
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        try:
            loop = asyncio.get_event_loop()
            count = await loop.run_in_executor(None, lambda: self.collection.count())
            
            return {
                "total_chunks": count,
                "collection_name": self.collection_name
            }
            
        except Exception as e:
            self.logger.error("Failed to get collection stats", error=str(e))
            return {"total_chunks": 0, "collection_name": self.collection_name}


class DocumentService:
    """Main document service orchestrating processing and storage"""
    
    def __init__(self):
        self.logger = logger.bind(component="document_service")
        self.processor = DocumentProcessor()
        self.vector_store = VectorStore()
    
    async def upload_document(
        self,
        file_content: bytes,
        filename: str,
        organization_id: str,
        user_id: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model: str = None
    ) -> Tuple[VectorDocumentResponse, int, int]:
        """
        Upload and process a document
        
        Returns:
            Tuple of (VectorDocumentResponse, chunks_created, processing_time_ms)
        """
        start_time = time.time()
        
        try:
            # Generate content hash
            content_hash = hashlib.sha256(file_content).hexdigest()
            
            # Check if document already exists
            async with get_database() as db:
                existing_doc = await self._check_existing_document(db, content_hash, organization_id)
                if existing_doc:
                    self.logger.info(f"Document already exists: {filename}")
                    processing_time = int((time.time() - start_time) * 1000)
                    return existing_doc, existing_doc.chunk_count, processing_time
            
            # Create temporary file for processing
            file_suffix = Path(filename).suffix
            document_id = str(uuid.uuid4())
            
            with tempfile.NamedTemporaryFile(suffix=file_suffix, delete=False) as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name
            
            try:
                # Extract text
                self.logger.info(f"Processing document: {filename}")
                text = await self.processor.extract_text(temp_file_path, file_suffix)
                
                if not text.strip():
                    raise ValueError("No text content extracted from document")
                
                # Create chunks
                chunks = self.processor.create_chunks(text, chunk_size, chunk_overlap)
                if not chunks:
                    raise ValueError("No chunks created from document")
                
                # Generate embeddings
                embedding_model = embedding_model or self.processor.default_embedding_model
                embeddings = await self.processor.generate_embeddings(chunks, embedding_model)
                
                # Store in database
                async with get_database() as db:
                    doc_record = VectorDocument(
                        id=uuid.UUID(document_id),
                        document_id=document_id,
                        filename=filename,
                        file_type=file_suffix,
                        content=text,
                        chunk_count=len(chunks),
                        embedding_model=embedding_model,
                        processing_status="completed",
                        processed_at=time.time(),
                        metadata={
                            "file_size": len(file_content),
                            "content_hash": content_hash,
                            "chunk_size": chunk_size,
                            "chunk_overlap": chunk_overlap
                        },
                        organization_id=uuid.UUID(organization_id),
                        uploaded_by=uuid.UUID(user_id)
                    )
                    
                    db.add(doc_record)
                    await db.commit()
                    await db.refresh(doc_record)
                
                # Store in vector database
                chunk_metadata = {
                    "filename": filename,
                    "file_type": file_suffix,
                    "embedding_model": embedding_model,
                    "organization_id": organization_id
                }
                
                chunks_added = await self.vector_store.add_document_chunks(
                    document_id, chunks, embeddings, chunk_metadata
                )
                
                processing_time = int((time.time() - start_time) * 1000)
                
                self.logger.info(
                    f"Successfully processed document: {filename}",
                    chunks=len(chunks),
                    processing_time_ms=processing_time
                )
                
                # Create response
                doc_response = VectorDocumentResponse(
                    id=doc_record.id,
                    document_id=doc_record.document_id,
                    filename=doc_record.filename,
                    file_type=doc_record.file_type,
                    chunk_count=doc_record.chunk_count,
                    embedding_model=doc_record.embedding_model,
                    processing_status=doc_record.processing_status,
                    processing_error=doc_record.processing_error,
                    processed_at=doc_record.processed_at,
                    metadata=doc_record.metadata,
                    created_at=doc_record.created_at,
                    updated_at=doc_record.updated_at
                )
                
                return doc_response, chunks_added, processing_time
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file_path)
                except Exception as e:
                    self.logger.warning(f"Failed to delete temp file: {temp_file_path}", error=str(e))
                    
        except Exception as e:
            self.logger.error(f"Failed to process document: {filename}", error=str(e))
            
            # Store error in database
            try:
                async with get_database() as db:
                    doc_record = VectorDocument(
                        id=uuid.uuid4(),
                        document_id=str(uuid.uuid4()),
                        filename=filename,
                        file_type=Path(filename).suffix,
                        chunk_count=0,
                        embedding_model=embedding_model or self.processor.default_embedding_model,
                        processing_status="failed",
                        processing_error=str(e),
                        metadata={"file_size": len(file_content)},
                        organization_id=uuid.UUID(organization_id),
                        uploaded_by=uuid.UUID(user_id)
                    )
                    db.add(doc_record)
                    await db.commit()
            except Exception as db_error:
                self.logger.error("Failed to store error in database", error=str(db_error))
            
            raise
    
    async def _check_existing_document(
        self, 
        db: AsyncSession, 
        content_hash: str, 
        organization_id: str
    ) -> Optional[VectorDocumentResponse]:
        """Check if document already exists by content hash"""
        try:
            query = select(VectorDocument).where(
                VectorDocument.metadata['content_hash'].astext == content_hash,
                VectorDocument.organization_id == uuid.UUID(organization_id),
                VectorDocument.processing_status == "completed"
            )
            
            result = await db.execute(query)
            doc = result.scalar_one_or_none()
            
            if doc:
                return VectorDocumentResponse(
                    id=doc.id,
                    document_id=doc.document_id,
                    filename=doc.filename,
                    file_type=doc.file_type,
                    chunk_count=doc.chunk_count,
                    embedding_model=doc.embedding_model,
                    processing_status=doc.processing_status,
                    processing_error=doc.processing_error,
                    processed_at=doc.processed_at,
                    metadata=doc.metadata,
                    created_at=doc.created_at,
                    updated_at=doc.updated_at
                )
            
            return None
            
        except Exception as e:
            self.logger.error("Failed to check existing document", error=str(e))
            return None
    
    async def search_documents(
        self,
        query: str,
        organization_id: str,
        n_results: int = 5,
        embedding_model: str = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> DocumentSearchResponse:
        """Search documents using semantic similarity"""
        start_time = time.time()
        
        try:
            # Generate query embedding
            embedding_model = embedding_model or self.processor.default_embedding_model
            query_embeddings = await self.processor.generate_embeddings([query], embedding_model)
            
            if not query_embeddings:
                return DocumentSearchResponse(
                    query=query,
                    results=[],
                    total_results=0,
                    search_time_ms=0
                )
            
            query_embedding = query_embeddings[0]
            
            # Add organization filter
            if filter_metadata is None:
                filter_metadata = {}
            filter_metadata["organization_id"] = organization_id
            
            # Search vector store
            search_results = await self.vector_store.search_documents(
                query_embedding,
                n_results,
                filter_metadata
            )
            
            search_time_ms = int((time.time() - start_time) * 1000)
            
            self.logger.info(
                f"Document search completed",
                query=query[:50],
                results_count=len(search_results),
                search_time_ms=search_time_ms
            )
            
            return DocumentSearchResponse(
                query=query,
                results=search_results,
                total_results=len(search_results),
                search_time_ms=search_time_ms
            )
            
        except Exception as e:
            self.logger.error("Document search failed", error=str(e))
            return DocumentSearchResponse(
                query=query,
                results=[],
                total_results=0,
                search_time_ms=int((time.time() - start_time) * 1000)
            )
    
    async def get_documents(
        self,
        organization_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[VectorDocumentResponse]:
        """Get documents for an organization"""
        try:
            async with get_database() as db:
                query = select(VectorDocument).where(
                    VectorDocument.organization_id == uuid.UUID(organization_id)
                ).limit(limit).offset(offset).order_by(VectorDocument.created_at.desc())
                
                result = await db.execute(query)
                documents = result.scalars().all()
                
                return [
                    VectorDocumentResponse(
                        id=doc.id,
                        document_id=doc.document_id,
                        filename=doc.filename,
                        file_type=doc.file_type,
                        chunk_count=doc.chunk_count,
                        embedding_model=doc.embedding_model,
                        processing_status=doc.processing_status,
                        processing_error=doc.processing_error,
                        processed_at=doc.processed_at,
                        metadata=doc.metadata,
                        created_at=doc.created_at,
                        updated_at=doc.updated_at
                    )
                    for doc in documents
                ]
                
        except Exception as e:
            self.logger.error("Failed to get documents", error=str(e))
            return []
    
    async def delete_document(
        self,
        document_id: str,
        organization_id: str
    ) -> bool:
        """Delete a document and its chunks"""
        try:
            async with get_database() as db:
                # Get document
                query = select(VectorDocument).where(
                    VectorDocument.document_id == document_id,
                    VectorDocument.organization_id == uuid.UUID(organization_id)
                )
                
                result = await db.execute(query)
                doc = result.scalar_one_or_none()
                
                if not doc:
                    return False
                
                # Delete from vector store
                await self.vector_store.delete_document(document_id)
                
                # Delete from database
                await db.delete(doc)
                await db.commit()
                
                self.logger.info(f"Deleted document: {document_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to delete document: {document_id}", error=str(e))
            return False
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            # Get vector store stats
            vector_stats = await self.vector_store.get_collection_stats()
            
            # Get database stats
            async with get_database() as db:
                from sqlalchemy import func
                
                total_docs_result = await db.execute(
                    select(func.count(VectorDocument.id))
                )
                total_docs = total_docs_result.scalar()
                
                processed_docs_result = await db.execute(
                    select(func.count(VectorDocument.id)).where(
                        VectorDocument.processing_status == "completed"
                    )
                )
                processed_docs = processed_docs_result.scalar()
                
                return {
                    **vector_stats,
                    "total_documents": total_docs,
                    "processed_documents": processed_docs,
                    "failed_documents": total_docs - processed_docs
                }
                
        except Exception as e:
            self.logger.error("Failed to get collection stats", error=str(e))
            return {
                "total_chunks": 0,
                "total_documents": 0,
                "processed_documents": 0,
                "failed_documents": 0
            }


# Global document service instance
_document_service: Optional[DocumentService] = None


def get_document_service() -> DocumentService:
    """Get or create the global document service"""
    global _document_service
    if _document_service is None:
        _document_service = DocumentService()
    return _document_service