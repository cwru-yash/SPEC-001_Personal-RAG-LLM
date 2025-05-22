# src/storage/duckdb.py
import os
import json
from typing import Dict, List, Any, Optional
import duckdb
from datetime import datetime

from src.models.document import Document

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
            # Convert content_type and persuasion_tags lists to arrays
            content_type_array = f"ARRAY{json.dumps(document.content_type)}"
            persuasion_tags_array = f"ARRAY{json.dumps(document.persuasion_tags)}"
            
            # Convert metadata to JSON
            metadata_json = json.dumps(document.metadata)
            
            # Insert document
            self.conn.execute(f"""
                INSERT INTO {self.schema_name}.documents 
                (doc_id, file_name, file_extension, content_type, created_at, author, 
                 metadata, persuasion_tags, text_content)
                VALUES (?, ?, ?, {content_type_array}, ?, ?, ?, {persuasion_tags_array}, ?)
                ON CONFLICT (doc_id) DO UPDATE SET
                file_name = excluded.file_name,
                file_extension = excluded.file_extension,
                content_type = excluded.content_type,
                created_at = excluded.created_at,
                author = excluded.author,
                metadata = excluded.metadata,
                persuasion_tags = excluded.persuasion_tags,
                text_content = excluded.text_content
            """, (
                document.doc_id,
                document.file_name,
                document.file_extension,
                document.created_at if document.created_at else datetime.now(),
                document.author,
                metadata_json,
                document.text_content
            ))
            
            # Store chunks
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
                tag_context_array = f"ARRAY{json.dumps(chunk.get('tag_context', []))}"
                
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
    
    def search_documents(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search documents by various criteria."""
        try:
            # Build WHERE clause
            where_clauses = []
            params = []
            
            if "content_type" in query:
                where_clauses.append(f"array_contains(content_type, ?)")
                params.append(query["content_type"])
                
            if "author" in query:
                where_clauses.append("author LIKE ?")
                params.append(f"%{query['author']}%")
                
            if "created_after" in query:
                where_clauses.append("created_at >= ?")
                params.append(query["created_after"])
                
            if "created_before" in query:
                where_clauses.append("created_at <= ?")
                params.append(query["created_before"])
                
            if "text_search" in query:
                where_clauses.append("text_content LIKE ?")
                params.append(f"%{query['text_search']}%")
            
            # Construct query
            sql = f"SELECT doc_id, file_name, file_extension, content_type, created_at, author FROM {self.schema_name}.documents"
            
            if where_clauses:
                sql += " WHERE " + " AND ".join(where_clauses)
                
            # Execute query
            results = self.conn.execute(sql, params).fetchall()
            
            # Convert to list of dicts
            columns = ["doc_id", "file_name", "file_extension", "content_type", "created_at", "author"]
            documents = []
            
            for row in results:
                doc_dict = {columns[i]: row[i] for i in range(len(columns))}
                documents.append(doc_dict)
                
            return documents
            
        except Exception as e:
            print(f"Error searching documents: {e}")
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