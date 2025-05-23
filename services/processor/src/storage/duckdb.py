# src/storage/duckdb.py
import os
import json
from typing import Dict, List, Any, Optional
import duckdb
from datetime import datetime

# Fix the import path for Docker environment
try:
    # Try container path first
    from src.models.document import Document
except ImportError:
    # Fall back to development path
    from services.processor.src.models.document import Document

class DuckDBStorage:
    """Storage interface for DuckDB database to store document metadata."""
    
    def __init__(self, db_path: str, schema_name: str = "personal_rag"):
        """Initialize DuckDB connection and ensure tables exist."""
        self.db_path = db_path
        self.schema_name = schema_name
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Connect to DuckDB
        self.conn = duckdb.connect(db_path)
        
        # Create schema if not exists
        self.conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")
        
        # Create tables if not exist
        self._create_tables()
    
    def _create_tables(self):
        """Create necessary tables if they don't exist."""
        # Documents table
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.schema_name}.documents (
                doc_id VARCHAR PRIMARY KEY,
                file_name VARCHAR,
                file_extension VARCHAR,
                content_type VARCHAR[],
                created_at TIMESTAMP,
                author VARCHAR,
                metadata JSON,
                persuasion_tags VARCHAR[],
                text_content TEXT
            )
        """)
        
        # Chunks table
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.schema_name}.chunks (
                chunk_id VARCHAR PRIMARY KEY,
                doc_id VARCHAR,
                text_chunk TEXT,
                tag_context VARCHAR[],
                FOREIGN KEY (doc_id) REFERENCES {self.schema_name}.documents(doc_id)
            )
        """)
        
        # Pipeline events table for logging
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.schema_name}.pipeline_events (
                event_id VARCHAR PRIMARY KEY,
                doc_id VARCHAR,
                event_type VARCHAR,
                event_timestamp TIMESTAMP,
                event_data JSON,
                FOREIGN KEY (doc_id) REFERENCES {self.schema_name}.documents(doc_id)
            )
        """)
        
        # Create indexes for performance
        self.conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_documents_file_name 
            ON {self.schema_name}.documents (file_name)
        """)
        
        self.conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_chunks_doc_id 
            ON {self.schema_name}.chunks (doc_id)
        """)
    
    def store_document(self, document: Document) -> bool:
        """Store document metadata in DuckDB."""
        try:
            # Create array literals for lists - fix to handle content_type correctly
            if document.content_type:
                content_type_array = f"ARRAY[{', '.join(repr(item) for item in document.content_type)}]"
            else:
                content_type_array = "ARRAY[]::VARCHAR[]"
                
            if document.persuasion_tags:
                persuasion_tags_array = f"ARRAY[{', '.join(repr(item) for item in document.persuasion_tags)}]"
            else:
                persuasion_tags_array = "ARRAY[]::VARCHAR[]"
            
            # Convert metadata to JSON
            metadata_json = json.dumps(document.metadata)
            
            # Insert document with proper SQL syntax
            self.conn.execute(f"""
                INSERT INTO {self.schema_name}.documents 
                (doc_id, file_name, file_extension, content_type, created_at, author, 
                 metadata, persuasion_tags, text_content)
                VALUES (?, ?, ?, {content_type_array}, ?, ?, ?, {persuasion_tags_array}, ?)
            """, (
                document.doc_id,
                document.file_name,
                document.file_extension,
                document.created_at if document.created_at else datetime.now(),
                document.author if document.author else None,
                metadata_json,
                document.text_content
            ))
            
            # Store chunks
            if document.chunks:
                self.store_chunks(document.doc_id, document.chunks)
            
            # Log the event
            self._log_event(
                document.doc_id, 
                "document_stored", 
                {"file_name": document.file_name}
            )
            
            return True
            
        except Exception as e:
            print(f"Error storing document: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def store_chunks(self, doc_id: str, chunks: List[Dict[str, Any]]) -> bool:
        """Store document chunks in DuckDB."""
        try:
            # Begin transaction
            self.conn.execute("BEGIN TRANSACTION")
            
            # Delete existing chunks
            self.conn.execute(f"""
                DELETE FROM {self.schema_name}.chunks 
                WHERE doc_id = ?
            """, (doc_id,))
            
            # Insert new chunks
            for chunk in chunks:
                # Fix array creation for tag_context
                if chunk.get('tag_context'):
                    tag_context_array = f"ARRAY[{', '.join(repr(item) for item in chunk.get('tag_context', []))}]"
                else:
                    tag_context_array = "ARRAY[]::VARCHAR[]"
                
                self.conn.execute(f"""
                    INSERT INTO {self.schema_name}.chunks 
                    (chunk_id, doc_id, text_chunk, tag_context)
                    VALUES (?, ?, ?, {tag_context_array})
                """, (
                    chunk["chunk_id"],
                    doc_id,
                    chunk["text_chunk"]
                ))
            
            # Commit transaction
            self.conn.execute("COMMIT")
            
            return True
            
        except Exception as e:
            # Rollback on error
            self.conn.execute("ROLLBACK")
            print(f"Error storing chunks: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve document by ID."""
        try:
            result = self.conn.execute(f"""
                SELECT * FROM {self.schema_name}.documents
                WHERE doc_id = ?
            """, (doc_id,)).fetchone()
            
            if not result:
                return None
                
            # Convert result to dict
            columns = ["doc_id", "file_name", "file_extension", "content_type", 
                       "created_at", "author", "metadata", "persuasion_tags", "text_content"]
            doc_dict = {columns[i]: result[i] for i in range(len(columns))}
            
            # Parse JSON fields
            if doc_dict["metadata"]:
                doc_dict["metadata"] = json.loads(doc_dict["metadata"])
            
            # Get chunks
            chunks = self.get_chunks(doc_id)
            doc_dict["chunks"] = chunks
            
            return doc_dict
            
        except Exception as e:
            print(f"Error retrieving document: {e}")
            return None
    
    def get_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        """Retrieve chunks for a document."""
        try:
            results = self.conn.execute(f"""
                SELECT chunk_id, doc_id, text_chunk, tag_context 
                FROM {self.schema_name}.chunks
                WHERE doc_id = ?
            """, (doc_id,)).fetchall()
            
            chunks = []
            for row in results:
                chunk = {
                    "chunk_id": row[0],
                    "doc_id": row[1],
                    "text_chunk": row[2],
                    "tag_context": row[3] if row[3] else []
                }
                chunks.append(chunk)
                
            return chunks
            
        except Exception as e:
            print(f"Error retrieving chunks: {e}")
            return []
    
    def _log_event(self, doc_id: str, event_type: str, event_data: Dict[str, Any]) -> None:
        """Log pipeline event to the events table."""
        try:
            import uuid
            event_id = str(uuid.uuid4())
            event_timestamp = datetime.now()
            event_data_json = json.dumps(event_data)
            
            self.conn.execute(f"""
                INSERT INTO {self.schema_name}.pipeline_events
                (event_id, doc_id, event_type, event_timestamp, event_data)
                VALUES (?, ?, ?, ?, ?)
            """, (
                event_id,
                doc_id,
                event_type,
                event_timestamp,
                event_data_json
            ))
            
        except Exception as e:
            print(f"Error logging event: {e}")