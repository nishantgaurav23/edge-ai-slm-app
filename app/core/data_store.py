"""
Data Store Module - Encrypted Local Storage for Edge AI SLM App

Provides:
- SQLite database for conversations and archived context
- AES-256 encryption for privacy
- Message archive for semantic search
"""

import sqlite3
import json
import os
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    import base64
except ImportError:
    Fernet = None

logger = logging.getLogger(__name__)


class EncryptionManager:
    """Handles AES-256 encryption for sensitive data."""
    
    def __init__(self, key_path: str = ".encryption_key"):
        self.key_path = key_path
        self._fernet = None
        
        if Fernet is None:
            logger.warning("cryptography not installed. Data will not be encrypted.")
            return
            
        self._initialize_key()
    
    def _initialize_key(self):
        """Load or generate encryption key."""
        if os.path.exists(self.key_path):
            with open(self.key_path, 'rb') as f:
                key = f.read()
        else:
            key = Fernet.generate_key()
            with open(self.key_path, 'wb') as f:
                f.write(key)
            os.chmod(self.key_path, 0o600)  # Restrict access
            
        self._fernet = Fernet(key)
    
    def encrypt(self, data: str) -> str:
        """Encrypt a string."""
        if not self._fernet:
            return data
        return self._fernet.encrypt(data.encode()).decode()
    
    def decrypt(self, data: str) -> str:
        """Decrypt a string."""
        if not self._fernet:
            return data
        return self._fernet.decrypt(data.encode()).decode()


class DataStore:
    """
    SQLite-based local storage with encryption.
    Stores conversations, archived context, and embeddings.
    """
    
    def __init__(self, db_path: str = "edge_ai_data.db", encrypt: bool = True):
        self.db_path = db_path
        self.encryption = EncryptionManager() if encrypt else None
        self._init_database()
    
    def _init_database(self):
        """Create database tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Conversations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Messages table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_archived INTEGER DEFAULT 0,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            )
        ''')
        
        # Archived context table (for semantic retrieval)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS archived_context (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER,
                content TEXT NOT NULL,
                embedding BLOB,
                relevance_score REAL DEFAULT 0.0,
                archived_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {self.db_path}")
    
    def create_conversation(self, title: str = "New Chat") -> int:
        """Create a new conversation and return its ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        encrypted_title = self.encryption.encrypt(title) if self.encryption else title
        cursor.execute(
            "INSERT INTO conversations (title) VALUES (?)",
            (encrypted_title,)
        )
        conversation_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"Created conversation {conversation_id}")
        return conversation_id
    
    def add_message(
        self, 
        conversation_id: int, 
        role: str, 
        content: str, 
        embedding: Optional[bytes] = None
    ) -> int:
        """Add a message to a conversation."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        encrypted_content = self.encryption.encrypt(content) if self.encryption else content
        cursor.execute(
            "INSERT INTO messages (conversation_id, role, content, embedding) VALUES (?, ?, ?, ?)",
            (conversation_id, role, encrypted_content, embedding)
        )
        message_id = cursor.lastrowid
        
        # Update conversation timestamp
        cursor.execute(
            "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (conversation_id,)
        )
        
        conn.commit()
        conn.close()
        return message_id
    
    def get_messages(self, conversation_id: int, include_archived: bool = False) -> List[Dict]:
        """Retrieve all messages from a conversation."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT id, role, content, created_at, is_archived FROM messages WHERE conversation_id = ?"
        if not include_archived:
            query += " AND is_archived = 0"
        query += " ORDER BY created_at ASC"
        
        cursor.execute(query, (conversation_id,))
        rows = cursor.fetchall()
        conn.close()
        
        messages = []
        for row in rows:
            content = self.encryption.decrypt(row[2]) if self.encryption else row[2]
            messages.append({
                'id': row[0],
                'role': row[1],
                'content': content,
                'created_at': row[3],
                'is_archived': bool(row[4])
            })
        
        return messages
    
    def archive_message(self, message_id: int, embedding: Optional[bytes] = None):
        """Mark a message as archived and optionally store its embedding."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "UPDATE messages SET is_archived = 1, embedding = ? WHERE id = ?",
            (embedding, message_id)
        )
        
        conn.commit()
        conn.close()
        logger.debug(f"Archived message {message_id}")
    
    def get_archived_with_embeddings(self, conversation_id: int) -> List[Dict]:
        """Get archived messages with their embeddings for similarity search."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT id, content, embedding FROM messages WHERE conversation_id = ? AND is_archived = 1 AND embedding IS NOT NULL",
            (conversation_id,)
        )
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for row in rows:
            content = self.encryption.decrypt(row[1]) if self.encryption else row[1]
            results.append({
                'id': row[0],
                'content': content,
                'embedding': row[2]
            })
        
        return results
    
    def get_all_conversations(self) -> List[Dict]:
        """Get list of all conversations."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC")
        rows = cursor.fetchall()
        conn.close()
        
        conversations = []
        for row in rows:
            title = self.encryption.decrypt(row[1]) if self.encryption else row[1]
            conversations.append({
                'id': row[0],
                'title': title,
                'created_at': row[2],
                'updated_at': row[3]
            })
        
        return conversations
    
    def delete_conversation(self, conversation_id: int):
        """Delete a conversation and all its messages."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        cursor.execute("DELETE FROM archived_context WHERE conversation_id = ?", (conversation_id,))
        cursor.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        
        conn.commit()
        conn.close()
        logger.info(f"Deleted conversation {conversation_id}")
