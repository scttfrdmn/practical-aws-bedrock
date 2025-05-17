"""
Conversation history management utilities for AWS Bedrock Converse API.

This module provides tools for persistent storage, retrieval, and management of
conversation histories from AWS Bedrock Converse API sessions.
"""

import os
import json
import time
import uuid
import logging
from typing import Dict, Any, Optional, List, Union, Tuple
import sqlite3


class ConversationStore:
    """
    A persistent storage manager for conversation histories.
    
    This class provides methods for saving, loading, and managing conversation
    histories from AWS Bedrock Converse API sessions. It supports multiple
    storage backends, including file-based JSON storage and SQLite.
    """
    
    def __init__(
        self,
        storage_type: str = "json",
        storage_path: str = "./conversations",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the conversation store.
        
        Args:
            storage_type: Storage backend type ('json' or 'sqlite')
            storage_path: Path to storage location
            logger: Optional logger instance
        """
        self.storage_type = storage_type.lower()
        self.storage_path = storage_path
        self.logger = logger or logging.getLogger(__name__)
        
        # Validate storage type
        valid_storage_types = ["json", "sqlite"]
        if self.storage_type not in valid_storage_types:
            raise ValueError(f"Invalid storage type: {storage_type}. Must be one of: {valid_storage_types}")
        
        # Initialize storage
        self._initialize_storage()
    
    def _initialize_storage(self) -> None:
        """Initialize the storage backend."""
        if self.storage_type == "json":
            # Create directory if it doesn't exist
            os.makedirs(self.storage_path, exist_ok=True)
        
        elif self.storage_type == "sqlite":
            # Initialize SQLite database
            self._init_sqlite_db()
    
    def _init_sqlite_db(self) -> None:
        """Initialize SQLite database schema."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        
        # Connect to database
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()
        
        # Create conversations table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            model_id TEXT,
            created_at REAL,
            updated_at REAL,
            metadata TEXT
        )
        ''')
        
        # Create messages table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            conversation_id TEXT,
            role TEXT,
            content TEXT,
            created_at REAL,
            sequence_num INTEGER,
            FOREIGN KEY (conversation_id) REFERENCES conversations (id)
        )
        ''')
        
        # Create index
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_messages_conversation
        ON messages (conversation_id, sequence_num)
        ''')
        
        # Commit changes
        conn.commit()
        conn.close()
    
    def save_conversation(self, conversation: Dict[str, Any]) -> str:
        """
        Save a conversation to persistent storage.
        
        Args:
            conversation: Conversation data dictionary
            
        Returns:
            Conversation ID
        """
        # Ensure conversation has an ID
        conversation_id = conversation.get("id", str(uuid.uuid4()))
        
        # Update conversation with ID if not present
        if "id" not in conversation:
            conversation["id"] = conversation_id
        
        # Update timestamps if not present
        if "created_at" not in conversation:
            conversation["created_at"] = time.time()
        
        if "updated_at" not in conversation:
            conversation["updated_at"] = time.time()
        
        # Save based on storage type
        if self.storage_type == "json":
            self._save_json_conversation(conversation)
        elif self.storage_type == "sqlite":
            self._save_sqlite_conversation(conversation)
        
        self.logger.debug(f"Saved conversation {conversation_id}")
        return conversation_id
    
    def _save_json_conversation(self, conversation: Dict[str, Any]) -> None:
        """Save conversation to JSON file."""
        conversation_id = conversation["id"]
        file_path = os.path.join(self.storage_path, f"{conversation_id}.json")
        
        with open(file_path, 'w') as f:
            json.dump(conversation, f, indent=2)
    
    def _save_sqlite_conversation(self, conversation: Dict[str, Any]) -> None:
        """Save conversation to SQLite database."""
        conversation_id = conversation["id"]
        model_id = conversation.get("model_id", "")
        created_at = conversation.get("created_at", time.time())
        updated_at = conversation.get("updated_at", time.time())
        
        # Extract metadata (everything except messages and core fields)
        metadata = {k: v for k, v in conversation.items() 
                   if k not in ["id", "model_id", "created_at", "updated_at", "messages"]}
        
        # Connect to database
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()
        
        # Insert or update conversation
        cursor.execute('''
        INSERT OR REPLACE INTO conversations (id, model_id, created_at, updated_at, metadata)
        VALUES (?, ?, ?, ?, ?)
        ''', (conversation_id, model_id, created_at, updated_at, json.dumps(metadata)))
        
        # Delete existing messages for this conversation
        cursor.execute('DELETE FROM messages WHERE conversation_id = ?', (conversation_id,))
        
        # Insert messages
        messages = conversation.get("messages", [])
        for i, message in enumerate(messages):
            message_id = message.get("id", str(uuid.uuid4()))
            role = message.get("role", "unknown")
            
            # Handle complex content (e.g., multimodal)
            content = message.get("content")
            if not isinstance(content, str):
                content = json.dumps(content)
            
            cursor.execute('''
            INSERT INTO messages (id, conversation_id, role, content, created_at, sequence_num)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (message_id, conversation_id, role, content, time.time(), i))
        
        # Commit changes
        conn.commit()
        conn.close()
    
    def load_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """
        Load a conversation from persistent storage.
        
        Args:
            conversation_id: ID of the conversation to load
            
        Returns:
            Conversation data dictionary
            
        Raises:
            FileNotFoundError: If conversation not found
        """
        # Load based on storage type
        if self.storage_type == "json":
            return self._load_json_conversation(conversation_id)
        elif self.storage_type == "sqlite":
            return self._load_sqlite_conversation(conversation_id)
        
        # Should never reach here due to validation in __init__
        raise ValueError(f"Unsupported storage type: {self.storage_type}")
    
    def _load_json_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """Load conversation from JSON file."""
        file_path = os.path.join(self.storage_path, f"{conversation_id}.json")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Conversation {conversation_id} not found")
        
        with open(file_path, 'r') as f:
            conversation = json.load(f)
        
        return conversation
    
    def _load_sqlite_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """Load conversation from SQLite database."""
        # Connect to database
        conn = sqlite3.connect(self.storage_path)
        conn.row_factory = sqlite3.Row  # Enable row_factory for dict-like rows
        cursor = conn.cursor()
        
        # Get conversation
        cursor.execute('''
        SELECT id, model_id, created_at, updated_at, metadata
        FROM conversations WHERE id = ?
        ''', (conversation_id,))
        
        row = cursor.fetchone()
        if not row:
            conn.close()
            raise FileNotFoundError(f"Conversation {conversation_id} not found")
        
        # Create conversation dict
        conversation = {
            "id": row["id"],
            "model_id": row["model_id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "messages": []
        }
        
        # Add metadata
        metadata = json.loads(row["metadata"])
        conversation.update(metadata)
        
        # Get messages
        cursor.execute('''
        SELECT id, role, content, created_at, sequence_num
        FROM messages 
        WHERE conversation_id = ?
        ORDER BY sequence_num ASC
        ''', (conversation_id,))
        
        for msg_row in cursor.fetchall():
            # Parse content (may be JSON for multimodal)
            content = msg_row["content"]
            try:
                content = json.loads(content)
            except (json.JSONDecodeError, TypeError):
                # If not valid JSON, keep as string
                pass
            
            message = {
                "id": msg_row["id"],
                "role": msg_row["role"],
                "content": content,
                "created_at": msg_row["created_at"]
            }
            
            conversation["messages"].append(message)
        
        conn.close()
        return conversation
    
    def list_conversations(
        self, 
        limit: int = 100, 
        offset: int = 0,
        sort_by: str = "updated_at",
        sort_order: str = "desc"
    ) -> List[Dict[str, Any]]:
        """
        List conversations in the store.
        
        Args:
            limit: Maximum number of conversations to return
            offset: Number of conversations to skip
            sort_by: Field to sort by ('created_at' or 'updated_at')
            sort_order: Sort order ('asc' or 'desc')
            
        Returns:
            List of conversation metadata dictionaries
        """
        # Validate sort parameters
        valid_sort_fields = ["created_at", "updated_at"]
        if sort_by not in valid_sort_fields:
            raise ValueError(f"Invalid sort field: {sort_by}. Must be one of: {valid_sort_fields}")
        
        valid_sort_orders = ["asc", "desc"]
        if sort_order.lower() not in valid_sort_orders:
            raise ValueError(f"Invalid sort order: {sort_order}. Must be one of: {valid_sort_orders}")
        
        # List based on storage type
        if self.storage_type == "json":
            return self._list_json_conversations(limit, offset, sort_by, sort_order)
        elif self.storage_type == "sqlite":
            return self._list_sqlite_conversations(limit, offset, sort_by, sort_order)
        
        # Should never reach here due to validation in __init__
        raise ValueError(f"Unsupported storage type: {self.storage_type}")
    
    def _list_json_conversations(
        self,
        limit: int,
        offset: int,
        sort_by: str,
        sort_order: str
    ) -> List[Dict[str, Any]]:
        """List conversations from JSON files."""
        # Get all JSON files in the directory
        if not os.path.exists(self.storage_path):
            return []
        
        files = [f for f in os.listdir(self.storage_path) if f.endswith('.json')]
        
        # Load basic metadata from each file
        conversations = []
        for filename in files:
            file_path = os.path.join(self.storage_path, filename)
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                    # Extract basic metadata
                    metadata = {
                        "id": data.get("id", filename.replace('.json', '')),
                        "model_id": data.get("model_id", "unknown"),
                        "created_at": data.get("created_at", 0),
                        "updated_at": data.get("updated_at", 0),
                        "message_count": len(data.get("messages", []))
                    }
                    
                    conversations.append(metadata)
            except Exception as e:
                self.logger.warning(f"Error loading conversation from {filename}: {str(e)}")
        
        # Sort conversations
        reverse = sort_order.lower() == "desc"
        conversations.sort(key=lambda x: x.get(sort_by, 0), reverse=reverse)
        
        # Apply pagination
        return conversations[offset:offset + limit]
    
    def _list_sqlite_conversations(
        self,
        limit: int,
        offset: int,
        sort_by: str,
        sort_order: str
    ) -> List[Dict[str, Any]]:
        """List conversations from SQLite database."""
        # Connect to database
        conn = sqlite3.connect(self.storage_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Build query with sorting and pagination
        query = f'''
        SELECT c.id, c.model_id, c.created_at, c.updated_at, c.metadata,
               COUNT(m.id) as message_count
        FROM conversations c
        LEFT JOIN messages m ON c.id = m.conversation_id
        GROUP BY c.id
        ORDER BY c.{sort_by} {sort_order}
        LIMIT ? OFFSET ?
        '''
        
        cursor.execute(query, (limit, offset))
        
        # Extract conversations
        conversations = []
        for row in cursor.fetchall():
            metadata = json.loads(row["metadata"])
            
            conversation = {
                "id": row["id"],
                "model_id": row["model_id"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "message_count": row["message_count"]
            }
            
            # Add selected metadata
            for key in ["name", "description", "tags"]:
                if key in metadata:
                    conversation[key] = metadata[key]
                    
            conversations.append(conversation)
        
        conn.close()
        return conversations
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation from persistent storage.
        
        Args:
            conversation_id: ID of the conversation to delete
            
        Returns:
            True if deleted, False if not found
        """
        # Delete based on storage type
        if self.storage_type == "json":
            return self._delete_json_conversation(conversation_id)
        elif self.storage_type == "sqlite":
            return self._delete_sqlite_conversation(conversation_id)
        
        # Should never reach here due to validation in __init__
        raise ValueError(f"Unsupported storage type: {self.storage_type}")
    
    def _delete_json_conversation(self, conversation_id: str) -> bool:
        """Delete conversation from JSON file."""
        file_path = os.path.join(self.storage_path, f"{conversation_id}.json")
        
        if not os.path.exists(file_path):
            return False
        
        os.remove(file_path)
        return True
    
    def _delete_sqlite_conversation(self, conversation_id: str) -> bool:
        """Delete conversation from SQLite database."""
        # Connect to database
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()
        
        # Check if conversation exists
        cursor.execute('SELECT id FROM conversations WHERE id = ?', (conversation_id,))
        if not cursor.fetchone():
            conn.close()
            return False
        
        # Delete messages first (foreign key constraint)
        cursor.execute('DELETE FROM messages WHERE conversation_id = ?', (conversation_id,))
        
        # Delete conversation
        cursor.execute('DELETE FROM conversations WHERE id = ?', (conversation_id,))
        
        # Commit changes
        conn.commit()
        conn.close()
        
        return True
    
    def search_conversations(
        self,
        query: str,
        field: str = "content",
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Search for conversations containing the query string.
        
        Args:
            query: Search query string
            field: Field to search in ('content', 'model_id', etc.)
            limit: Maximum number of results to return
            
        Returns:
            List of matching conversation metadata dictionaries
        """
        # Search based on storage type
        if self.storage_type == "json":
            return self._search_json_conversations(query, field, limit)
        elif self.storage_type == "sqlite":
            return self._search_sqlite_conversations(query, field, limit)
        
        # Should never reach here due to validation in __init__
        raise ValueError(f"Unsupported storage type: {self.storage_type}")
    
    def _search_json_conversations(
        self,
        query: str,
        field: str,
        limit: int
    ) -> List[Dict[str, Any]]:
        """Search conversations in JSON files."""
        # Get all JSON files in the directory
        if not os.path.exists(self.storage_path):
            return []
        
        files = [f for f in os.listdir(self.storage_path) if f.endswith('.json')]
        
        # Search each file
        matches = []
        for filename in files:
            file_path = os.path.join(self.storage_path, filename)
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                    # Search in messages if field is 'content'
                    if field == "content":
                        found = False
                        for message in data.get("messages", []):
                            content = message.get("content", "")
                            if isinstance(content, str) and query.lower() in content.lower():
                                found = True
                                break
                        
                        if found:
                            matches.append({
                                "id": data.get("id", filename.replace('.json', '')),
                                "model_id": data.get("model_id", "unknown"),
                                "created_at": data.get("created_at", 0),
                                "updated_at": data.get("updated_at", 0),
                                "message_count": len(data.get("messages", []))
                            })
                    
                    # Search in other fields
                    else:
                        field_value = data.get(field, "")
                        if isinstance(field_value, str) and query.lower() in field_value.lower():
                            matches.append({
                                "id": data.get("id", filename.replace('.json', '')),
                                "model_id": data.get("model_id", "unknown"),
                                "created_at": data.get("created_at", 0),
                                "updated_at": data.get("updated_at", 0),
                                "message_count": len(data.get("messages", []))
                            })
                    
                    # Stop searching if we have enough matches
                    if len(matches) >= limit:
                        break
            except Exception as e:
                self.logger.warning(f"Error searching conversation in {filename}: {str(e)}")
        
        return matches[:limit]
    
    def _search_sqlite_conversations(
        self,
        query: str,
        field: str,
        limit: int
    ) -> List[Dict[str, Any]]:
        """Search conversations in SQLite database."""
        # Connect to database
        conn = sqlite3.connect(self.storage_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Build appropriate search query
        if field == "content":
            query_sql = '''
            SELECT DISTINCT c.id, c.model_id, c.created_at, c.updated_at, c.metadata,
                   COUNT(DISTINCT m.id) as message_count
            FROM conversations c
            JOIN messages m ON c.id = m.conversation_id
            WHERE m.content LIKE ?
            GROUP BY c.id
            LIMIT ?
            '''
            cursor.execute(query_sql, (f"%{query}%", limit))
        else:
            # Search in conversation fields or metadata
            if field in ["id", "model_id", "created_at", "updated_at"]:
                # Direct field
                query_sql = f'''
                SELECT c.id, c.model_id, c.created_at, c.updated_at, c.metadata,
                       COUNT(m.id) as message_count
                FROM conversations c
                LEFT JOIN messages m ON c.id = m.conversation_id
                WHERE c.{field} LIKE ?
                GROUP BY c.id
                LIMIT ?
                '''
                cursor.execute(query_sql, (f"%{query}%", limit))
            else:
                # Search in metadata JSON
                query_sql = '''
                SELECT c.id, c.model_id, c.created_at, c.updated_at, c.metadata,
                       COUNT(m.id) as message_count
                FROM conversations c
                LEFT JOIN messages m ON c.id = m.conversation_id
                WHERE c.metadata LIKE ?
                GROUP BY c.id
                LIMIT ?
                '''
                cursor.execute(query_sql, (f"%{query}%", limit))
        
        # Extract conversations
        matches = []
        for row in cursor.fetchall():
            metadata = json.loads(row["metadata"])
            
            conversation = {
                "id": row["id"],
                "model_id": row["model_id"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "message_count": row["message_count"]
            }
            
            # Add selected metadata
            for key in ["name", "description", "tags"]:
                if key in metadata:
                    conversation[key] = metadata[key]
                    
            matches.append(conversation)
        
        conn.close()
        return matches


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create a conversation store
    json_store = ConversationStore(storage_type="json", storage_path="./conversation_data")
    
    # Create a sample conversation
    sample_conversation = {
        "id": "sample-convo-1",
        "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
        "created_at": time.time(),
        "updated_at": time.time(),
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant specializing in AWS services."
            },
            {
                "role": "user",
                "content": "What is AWS Bedrock?"
            },
            {
                "role": "assistant",
                "content": "AWS Bedrock is a fully managed service that provides access to foundation models (FMs) from leading AI companies through a unified API. It allows you to build generative AI applications without having to manage the underlying infrastructure or train your own models."
            }
        ]
    }
    
    try:
        # Save the conversation
        json_store.save_conversation(sample_conversation)
        print(f"Saved conversation: {sample_conversation['id']}")
        
        # Load the conversation
        loaded_conversation = json_store.load_conversation("sample-convo-1")
        print(f"Loaded conversation with {len(loaded_conversation['messages'])} messages")
        
        # List conversations
        conversations = json_store.list_conversations()
        print(f"Found {len(conversations)} conversations")
        
        # Search conversations
        matches = json_store.search_conversations("Bedrock", field="content")
        print(f"Found {len(matches)} conversations mentioning 'Bedrock'")
        
        # Create a SQLite store for comparison
        sqlite_store = ConversationStore(storage_type="sqlite", storage_path="./conversation_data/convos.db")
        
        # Save the same conversation to SQLite
        sqlite_store.save_conversation(sample_conversation)
        print(f"Saved conversation to SQLite: {sample_conversation['id']}")
        
        # Load from SQLite
        sqlite_loaded = sqlite_store.load_conversation("sample-convo-1")
        print(f"Loaded from SQLite: {len(sqlite_loaded['messages'])} messages")
        
    except Exception as e:
        print(f"Error: {str(e)}")