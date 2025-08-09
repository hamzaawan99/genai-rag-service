import os
import json
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import uuid
import shutil
from loguru import logger

from app.core.vector_store.base import VectorStore
from app.models.embedding import DocumentChunk, SearchResponse, QueryResult
from app.config.settings import settings

class FAISSStore(VectorStore):
    """FAISS-based vector store implementation."""
    
    def __init__(self, index_path: Optional[str] = None):
        """Initialize the FAISS store.
        
        Args:
            index_path: Path to the FAISS index directory. If None, uses settings.FAISS_INDEX_PATH
        """
        self.index_path = Path(index_path) if index_path else Path(settings.FAISS_INDEX_PATH)
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize FAISS index and metadata storage
        self.index = None
        self.metadata = {}
        self.chunk_id_to_idx = {}
        self.document_id_to_chunk_ids = {}
        
        # Load existing index if it exists
        self._load_index()
    
    def _load_index(self) -> None:
        """Load the FAISS index and metadata from disk if they exist."""
        index_file = self.index_path / "index.faiss"
        metadata_file = self.index_path / "metadata.json"
        
        if index_file.exists() and metadata_file.exists():
            try:
                # Load the FAISS index
                self.index = faiss.read_index(str(index_file))
                
                # Load metadata
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    self.metadata = metadata.get('metadata', {})
                    self.chunk_id_to_idx = metadata.get('chunk_id_to_idx', {})
                    self.document_id_to_chunk_ids = metadata.get('document_id_to_chunk_ids', {})
                
                logger.info(f"Loaded FAISS index with {len(self.chunk_id_to_idx)} chunks")
            except Exception as e:
                logger.error(f"Error loading FAISS index: {e}")
                self._initialize_empty_index()
        else:
            self._initialize_empty_index()
    
    def _initialize_empty_index(self, dimension: int = 384) -> None:
        """Initialize a new empty FAISS index.
        
        Args:
            dimension: Dimensionality of the embeddings
        """
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = {}
        self.chunk_id_to_idx = {}
        self.document_id_to_chunk_ids = {}
    
    def _save_index(self) -> None:
        """Save the FAISS index and metadata to disk."""
        if not self.index_path.exists():
            self.index_path.mkdir(parents=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(self.index_path / "index.faiss"))
        
        # Save metadata
        metadata = {
            'metadata': self.metadata,
            'chunk_id_to_idx': self.chunk_id_to_idx,
            'document_id_to_chunk_ids': self.document_id_to_chunk_ids
        }
        
        with open(self.index_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    async def add_documents(
        self,
        chunks: List[DocumentChunk],
        **kwargs
    ) -> List[str]:
        """Add document chunks to the vector store."""
        if not chunks:
            return []
        
        # Convert chunks to embeddings and metadata
        embeddings = []
        chunk_metadatas = []
        chunk_ids = []
        
        for chunk in chunks:
            if not chunk.embedding:
                raise ValueError(f"Chunk {chunk.chunk_id} has no embedding")
            
            embeddings.append(chunk.embedding)
            chunk_metadatas.append({
                'chunk_id': chunk.chunk_id,
                'document_id': chunk.document_id,
                'text': chunk.text,
                'chunk_index': chunk.chunk_index,
                'metadata': chunk.metadata
            })
            chunk_ids.append(chunk.chunk_id)
        
        # Convert to numpy array
        embeddings_np = self._validate_embeddings(embeddings)
        
        # Add to FAISS index
        if self.index.ntotal == 0:
            # First batch, initialize with the correct dimension
            self.index = faiss.IndexFlatL2(embeddings_np.shape[1])
        
        # Add vectors to the index
        self.index.add(embeddings_np)
        
        # Update metadata
        start_idx = self.index.ntotal - len(chunk_ids)
        for i, (chunk_id, chunk_metadata) in enumerate(zip(chunk_ids, chunk_metadatas)):
            idx = start_idx + i
            self.chunk_id_to_idx[chunk_id] = idx
            self.metadata[str(idx)] = chunk_metadata
            
            # Update document to chunk mapping
            doc_id = chunk_metadata['document_id']
            if doc_id not in self.document_id_to_chunk_ids:
                self.document_id_to_chunk_ids[doc_id] = []
            self.document_id_to_chunk_ids[doc_id].append(chunk_id)
        
        # Save the updated index and metadata
        self._save_index()
        
        return chunk_ids
    
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_conditions: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> SearchResponse:
        """Search for similar document chunks."""
        if self.index.ntotal == 0:
            return SearchResponse(
                query=kwargs.get('query_text', ''),
                results=[],
                total_results=0,
                model=kwargs.get('model', ''),
                search_duration=0.0
            )
        
        # Convert query embedding to numpy array
        query_embedding_np = np.array([query_embedding], dtype=np.float32)
        
        # Search the index
        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_embedding_np, k)
        
        # Convert results to DocumentChunk objects
        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx == -1:  # No more results
                continue
                
            chunk_metadata = self.metadata.get(str(idx))
            if not chunk_metadata:
                continue
            
            # Apply filters if provided
            if filter_conditions:
                match = True
                for key, value in filter_conditions.items():
                    if key in chunk_metadata and chunk_metadata[key] != value:
                        match = False
                        break
                    if key in chunk_metadata.get('metadata', {}) and chunk_metadata['metadata'].get(key) != value:
                        match = False
                        break
                if not match:
                    continue
            
            chunk = DocumentChunk(
                text=chunk_metadata['text'],
                chunk_id=chunk_metadata['chunk_id'],
                document_id=chunk_metadata['document_id'],
                chunk_index=chunk_metadata['chunk_index'],
                metadata=chunk_metadata.get('metadata', {})
            )
            
            results.append(QueryResult(chunk=chunk, score=float(score)))
        
        return SearchResponse(
            query=kwargs.get('query_text', ''),
            results=results,
            total_results=len(results),
            model=kwargs.get('model', ''),
            search_duration=0.0  # TODO: Implement timing
        )
    
    async def delete(
        self,
        chunk_ids: Optional[List[str]] = None,
        document_ids: Optional[List[str]] = None,
        **kwargs
    ) -> bool:
        """Delete document chunks from the vector store."""
        if not chunk_ids and not document_ids:
            return False
        
        # Get all chunk IDs to delete
        chunks_to_delete = set()
        
        if chunk_ids:
            chunks_to_delete.update(chunk_ids)
        
        if document_ids:
            for doc_id in document_ids:
                if doc_id in self.document_id_to_chunk_ids:
                    chunks_to_delete.update(self.document_id_to_chunk_ids[doc_id])
        
        if not chunks_to_delete:
            return False
        
        # Create a new index without the deleted chunks
        new_index = faiss.IndexFlatL2(self.index.d) if self.index.ntotal > 0 else None
        new_metadata = {}
        new_chunk_id_to_idx = {}
        new_doc_id_to_chunk_ids = {}
        
        # Rebuild the index and metadata
        for idx in range(self.index.ntotal):
            chunk_metadata = self.metadata.get(str(idx))
            if not chunk_metadata:
                continue
                
            chunk_id = chunk_metadata['chunk_id']
            if chunk_id in chunks_to_delete:
                continue
            
            # Add to new index
            if new_index is None:
                new_index = faiss.IndexFlatL2(self.index.d)
            
            # Get the vector from the old index
            vector = self.index.reconstruct(idx)
            new_index.add(np.array([vector], dtype=np.float32))
            
            # Update metadata
            new_idx = new_index.ntotal - 1
            new_metadata[str(new_idx)] = chunk_metadata
            new_chunk_id_to_idx[chunk_id] = new_idx
            
            # Update document to chunk mapping
            doc_id = chunk_metadata['document_id']
            if doc_id not in new_doc_id_to_chunk_ids:
                new_doc_id_to_chunk_ids[doc_id] = []
            new_doc_id_to_chunk_ids[doc_id].append(chunk_id)
        
        # Update the index and metadata
        self.index = new_index or faiss.IndexFlatL2(384)  # Fallback to empty index
        self.metadata = new_metadata
        self.chunk_id_to_idx = new_chunk_id_to_idx
        self.document_id_to_chunk_ids = new_doc_id_to_chunk_ids
        
        # Save the updated index and metadata
        self._save_index()
        
        return True
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return {
            'total_chunks': self.index.ntotal if self.index else 0,
            'dimension': self.index.d if self.index else 0,
            'documents': len(self.document_id_to_chunk_ids),
            'chunks_per_document': {
                doc_id: len(chunk_ids)
                for doc_id, chunk_ids in self.document_id_to_chunk_ids.items()
            },
            'is_trained': self.index.is_trained if self.index else False
        }
    
    async def clear(self) -> bool:
        """Clear all data from the vector store."""
        try:
            # Remove all index files
            for file in self.index_path.glob('*'):
                if file.is_file():
                    file.unlink()
                elif file.is_dir():
                    shutil.rmtree(file)
            
            # Reset in-memory state
            self._initialize_empty_index()
            return True
        except Exception as e:
            logger.error(f"Error clearing FAISS store: {e}")
            return False
