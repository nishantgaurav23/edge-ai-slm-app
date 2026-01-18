"""
Context Manager Module - Semantic Context Window for Edge AI SLM App

Provides:
- Sliding window with semantic chunking
- Embedding-based similarity for context relevance
- Archives old context instead of discarding
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional
import pickle

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Handles text embeddings using a lightweight local model.
    Uses all-MiniLM-L6-v2 (22MB) for efficient CPU-based embeddings.
    """
    
    _instance = None
    _model = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        self._model = None
    
    def _ensure_loaded(self):
        """Lazy load the embedding model."""
        if self._model is None:
            if SentenceTransformer is None:
                logger.warning("sentence-transformers not installed. Semantic search disabled.")
                return False
            
            logger.info("Loading embedding model (all-MiniLM-L6-v2)...")
            self._model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embedding model loaded.")
        return True
    
    def embed(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding for a single text."""
        if not self._ensure_loaded():
            return None
        return self._model.encode(text, convert_to_numpy=True)
    
    def embed_batch(self, texts: List[str]) -> Optional[np.ndarray]:
        """Generate embeddings for multiple texts."""
        if not self._ensure_loaded():
            return None
        return self._model.encode(texts, convert_to_numpy=True)
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        return float(np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        ))
    
    def unload(self):
        """Unload model to free memory."""
        if self._model:
            del self._model
            self._model = None
            logger.info("Embedding model unloaded.")


class ContextManager:
    """
    Manages the context window for the LLM with semantic awareness.
    
    Features:
    - Sliding window that keeps recent N interactions
    - Semantic similarity to determine what stays vs. gets archived
    - Archives old messages to DataStore for later retrieval
    """

    def __init__(
        self, 
        max_tokens: int = 2048, 
        system_prompt: str = "You are a helpful AI assistant.",
        data_store=None,
        conversation_id: int = None
    ):
        self.max_approx_tokens = max_tokens
        self.system_prompt = {"role": "system", "content": system_prompt}
        self.history: List[Dict[str, Any]] = []
        self.embedding_service = EmbeddingService.get_instance()
        self.data_store = data_store
        self.conversation_id = conversation_id
        
        # Threshold for considering context "relevant"
        self.relevance_threshold = 0.7

    def add_message(self, role: str, content: str):
        """Add a message and compute its embedding."""
        embedding = self.embedding_service.embed(content)
        
        message = {
            "role": role, 
            "content": content,
            "embedding": embedding
        }
        self.history.append(message)
        
        # Store in database if available
        if self.data_store and self.conversation_id:
            embedding_bytes = pickle.dumps(embedding) if embedding is not None else None
            self.data_store.add_message(
                self.conversation_id, 
                role, 
                content, 
                embedding_bytes
            )
        
        self._prune_context()

    def get_context(self) -> List[Dict[str, str]]:
        """
        Returns the full context formatted for Llama inference.
        Only includes role and content (not embeddings).
        """
        formatted = [self.system_prompt]
        for msg in self.history:
            formatted.append({"role": msg["role"], "content": msg["content"]})
        return formatted

    def _prune_context(self):
        """
        Smart pruning with semantic awareness:
        1. Estimate token count
        2. If over limit, find least relevant messages
        3. Archive them instead of discarding
        """
        while self._estimate_tokens() > self.max_approx_tokens:
            if len(self.history) <= 2:  # Keep at least last exchange
                break
            
            # Find least relevant message to archive (skip most recent)
            least_relevant_idx = self._find_least_relevant_message()
            
            if least_relevant_idx is not None:
                archived_msg = self.history.pop(least_relevant_idx)
                logger.debug(f"Archived message: {archived_msg['role']}")
                
                # Store in archive with embedding
                if self.data_store and self.conversation_id and archived_msg.get('embedding') is not None:
                    # Mark as archived in database
                    pass  # Already stored when added
            else:
                # Fallback: remove oldest
                removed = self.history.pop(0)
                logger.debug(f"Pruned oldest message: {removed['role']}")

    def _find_least_relevant_message(self) -> Optional[int]:
        """
        Find the message least relevant to current conversation.
        Compares each message's embedding to the most recent message.
        """
        if len(self.history) <= 2:
            return None
        
        # Get embedding of most recent message
        recent_embedding = self.history[-1].get('embedding')
        if recent_embedding is None:
            return 0  # Fallback to oldest
        
        min_similarity = float('inf')
        min_idx = 0
        
        # Check all messages except the last 2 (keep recent context)
        for i in range(len(self.history) - 2):
            msg_embedding = self.history[i].get('embedding')
            if msg_embedding is None:
                continue
            
            similarity = self.embedding_service.similarity(msg_embedding, recent_embedding)
            if similarity < min_similarity:
                min_similarity = similarity
                min_idx = i
        
        return min_idx

    def _estimate_tokens(self) -> int:
        """Rough token estimation (4 chars ≈ 1 token)."""
        text = self.system_prompt['content']
        text += "".join([m['content'] for m in self.history])
        return len(text) // 4

    def semantically_search_history(self, query: str, top_k: int = 3) -> List[Dict[str, str]]:
        """
        Search archived and current history for relevant context.
        Returns top_k most similar messages.
        """
        query_embedding = self.embedding_service.embed(query)
        if query_embedding is None:
            return []
        
        results = []
        
        # Search current history
        for msg in self.history:
            if msg.get('embedding') is not None:
                similarity = self.embedding_service.similarity(
                    query_embedding, msg['embedding']
                )
                if similarity >= self.relevance_threshold:
                    results.append({
                        'role': msg['role'],
                        'content': msg['content'],
                        'similarity': similarity
                    })
        
        # Search archived messages from database
        if self.data_store and self.conversation_id:
            archived = self.data_store.get_archived_with_embeddings(self.conversation_id)
            for item in archived:
                if item['embedding']:
                    archived_embedding = pickle.loads(item['embedding'])
                    similarity = self.embedding_service.similarity(
                        query_embedding, archived_embedding
                    )
                    if similarity >= self.relevance_threshold:
                        results.append({
                            'role': 'archived',
                            'content': item['content'],
                            'similarity': similarity
                        })
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]

    def get_enhanced_context(self, current_query: str) -> List[Dict[str, str]]:
        """
        Get context enhanced with semantically relevant archived messages.
        Useful for providing LLM with relevant past context.
        """
        base_context = self.get_context()
        
        # Find relevant archived context
        relevant = self.semantically_search_history(current_query, top_k=2)
        
        if relevant:
            # Insert relevant context after system prompt
            enhanced = [base_context[0]]  # System prompt
            
            # Add relevant archived context with a note
            for item in relevant:
                if item['role'] == 'archived':
                    enhanced.append({
                        "role": "system",
                        "content": f"[Relevant earlier context]: {item['content']}"
                    })
            
            # Add current conversation history
            enhanced.extend(base_context[1:])
            return enhanced
        
        return base_context

    def clear(self):
        """Clear all history."""
        self.history = []
