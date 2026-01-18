"""
Unit Tests for Context Manager with Semantic Embeddings
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestContextManager:
    """Tests for ContextManager functionality."""
    
    def test_basic_message_addition(self):
        """Test basic message addition to context."""
        from app.core.context_manager import ContextManager
        
        cm = ContextManager(max_tokens=2048)
        cm.add_message("user", "Hello, how are you?")
        cm.add_message("assistant", "I'm doing well, thank you!")
        
        context = cm.get_context()
        
        # Should have system prompt + 2 messages
        assert len(context) == 3
        assert context[0]["role"] == "system"
        assert context[1]["role"] == "user"
        assert context[2]["role"] == "assistant"
    
    def test_context_pruning(self):
        """Test that context is pruned when exceeding token limit."""
        from app.core.context_manager import ContextManager
        
        # Small token limit to force pruning
        cm = ContextManager(max_tokens=100)
        
        # Add many messages to exceed limit
        for i in range(10):
            cm.add_message("user", f"This is message number {i} with some extra text to use tokens")
            cm.add_message("assistant", f"Response to message {i} with additional content")
        
        # Context should be pruned
        context = cm.get_context()
        estimated_tokens = sum(len(m["content"]) for m in context) // 4
        
        # Should be under or near the limit
        assert estimated_tokens <= 150  # Allow some buffer
    
    def test_semantic_search_returns_results(self):
        """Test semantic search functionality."""
        from app.core.context_manager import ContextManager
        
        cm = ContextManager(max_tokens=4096)
        
        # Add diverse messages
        cm.add_message("user", "What is the weather like in Paris?")
        cm.add_message("assistant", "Paris is currently sunny with 22°C temperature.")
        cm.add_message("user", "Tell me about Python programming.")
        cm.add_message("assistant", "Python is a versatile programming language.")
        cm.add_message("user", "How do I cook pasta?")
        cm.add_message("assistant", "Boil water, add pasta, cook for 8-10 minutes.")
        
        # Search should return relevant results
        results = cm.semantically_search_history("What about the weather in France?")
        
        # Results should be a list
        assert isinstance(results, list)
    
    def test_clear_history(self):
        """Test clearing history."""
        from app.core.context_manager import ContextManager
        
        cm = ContextManager()
        cm.add_message("user", "Hello")
        cm.add_message("assistant", "Hi there!")
        
        cm.clear()
        
        context = cm.get_context()
        # Should only have system prompt
        assert len(context) == 1
        assert context[0]["role"] == "system"


class TestEmbeddingService:
    """Tests for EmbeddingService."""
    
    def test_singleton_pattern(self):
        """Test that EmbeddingService uses singleton pattern."""
        from app.core.context_manager import EmbeddingService
        
        instance1 = EmbeddingService.get_instance()
        instance2 = EmbeddingService.get_instance()
        
        assert instance1 is instance2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
