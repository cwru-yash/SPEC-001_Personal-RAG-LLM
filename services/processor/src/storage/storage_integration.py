# services/processor/src/pipeline/storage_integration.py
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.models.document import Document
from src.storage.duckdb import DuckDBStorage
from src.storage.vector_store import VectorStore
from src.storage.graph_db import GraphDBStorage

class StorageIntegration:
    """Unified storage integration for all document types."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize storage backends."""
        self.logger = logging.getLogger(__name__)
        
        # Initialize storage systems
        self.metadata_storage = DuckDBStorage(config["duckdb"]["database"])
        self.vector_store = VectorStore(config["vector_store"])
        self.graph_storage = GraphDBStorage(config["graph_db"])
        
        self.stats = {
            "documents_stored": 0,
            "chunks_stored": 0,
            "relationships_created": 0,
            "errors": []
        }
    
    def store_document(self, document: Document) -> bool:
        """Store a document across all storage systems."""
        try:
            # 1. Store metadata in DuckDB
            self.metadata_storage.store_document(document)
            
            # 2. Store chunks in vector store with enhanced metadata
            if document.chunks:
                enhanced_chunks = self._enhance_chunks_for_storage(document)
                self.vector_store.store_chunks(enhanced_chunks)
                self.stats["chunks_stored"] += len(enhanced_chunks)
            
            # 3. Store in graph database with relationships
            self._store_in_graph(document)
            
            self.stats["documents_stored"] += 1
            return True
            
        except Exception as e:
            error_msg = f"Failed to store document {document.doc_id}: {str(e)}"
            self.logger.error(error_msg)
            self.stats["errors"].append(error_msg)
            return False
    
    def store_document_pair(self, content_doc: Document, metadata_doc: Document) -> bool:
        """Store a paired document set with relationships."""
        try:
            # Store both documents
            self.store_document(content_doc)
            self.store_document(metadata_doc)
            
            # Create special relationship in graph
            self._create_paired_relationship(content_doc, metadata_doc)
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to store document pair: {str(e)}"
            self.logger.error(error_msg)
            self.stats["errors"].append(error_msg)
            return False
    
    # def _enhance_chunks_for_storage(self, document: Document) -> List[Dict[str, Any]]:
    #     """Enhance chunks with additional metadata for better retrieval."""
    #     enhanced_chunks = []
        
    #     for chunk in document.chunks:
    #         enhanced_chunk = chunk.copy()
            
    #         # Add document-level metadata to each chunk
    #         enhanced_chunk["metadata"] = {
    #             "doc_id": document.doc_id,
    #             "file_name": document.file_name,
    #             "content_types": document.content_type,
    #             "category": document.metadata.get("dataset_category", "unknown"),
    #             "document_type": document.metadata.get("document_type", "content"),
    #             "author": document.author if document.author else "unknown",
    #             "created_at": document.created_at.isoformat() if document.created_at else None,
    #             "quality_score": document.metadata.get("quality_score", 0.5)
    #         }
            
    #         # Add special metadata for different content types
    #         if "image" in document.content_type:
    #             enhanced_chunk["metadata"]["image_type"] = document.metadata.get("image_type", "general")
    #             if "chart_analysis" in document.metadata:
    #                 enhanced_chunk["metadata"]["chart_type"] = document.metadata["chart_analysis"].get("chart_type")
            
    #         if "spreadsheet" in document.content_type:
    #             enhanced_chunk["metadata"]["has_numerical_data"] = True
    #             enhanced_chunk["metadata"]["sheet_count"] = document.metadata.get("total_sheets", 1)
            
    #         if "presentation" in document.content_type:
    #             enhanced_chunk["metadata"]["slide_count"] = document.metadata.get("total_slides", 0)
    #             enhanced_chunk["metadata"]["has_visual_content"] = "visual_presentation" in document.content_type
            
    #         enhanced_chunks.append(enhanced_chunk)
        
    #     return enhanced_chunks
    
    def _enhance_chunks_for_storage(self, document: Document) -> List[Dict[str, Any]]:
        """Enhance chunks with additional metadata for better retrieval."""
        enhanced_chunks = []
        
        for chunk in document.chunks:
            enhanced_chunk = chunk.copy()
            
            # Add document-level metadata to chunk
            enhanced_chunk["metadata"] = {
                "doc_id": document.doc_id,
                "file_name": document.file_name,
                "content_types": document.content_type,
                "created_at": document.created_at.isoformat() if document.created_at else None
            }
            
            # Add special metadata for different content types
            if "image" in document.content_type:
                enhanced_chunk["metadata"]["image_type"] = document.metadata.get("image_type", "general")
                if "chart_analysis" in document.metadata:
                    enhanced_chunk["metadata"]["chart_type"] = document.metadata["chart_analysis"].get("chart_type")
            
            if "presentation" in document.content_type:
                enhanced_chunk["metadata"]["slide_count"] = document.metadata.get("total_slides", 0)
                if "slides" in document.metadata:
                    slide_num = None
                    for tag in chunk.get("tag_context", []):
                        if tag.startswith("slide_"):
                            slide_num = tag.split("_")[1]
                            break
                    if slide_num:
                        enhanced_chunk["metadata"]["slide_number"] = slide_num
            
            enhanced_chunks.append(enhanced_chunk)
        
        return enhanced_chunks
    def _store_in_graph(self, document: Document):
        """Store document in graph database with rich relationships."""
        # Store basic document
        self.graph_storage.store_document(document.__dict__)
        
        # Create category nodes
        if "dataset_category" in document.metadata:
            self._create_category_relationship(document)
        
        # Create content type relationships
        for content_type in document.content_type:
            self._create_content_type_relationship(document, content_type)
        
        # Create author relationships if present
        if document.author:
            self._create_author_relationship(document)
        
        # Create special relationships for different document types
        if "image" in document.content_type and "chart_analysis" in document.metadata:
            self._create_chart_relationships(document)
        
        if "spreadsheet" in document.content_type and "sheets" in document.metadata:
            self._create_spreadsheet_relationships(document)
    
    def _create_paired_relationship(self, content_doc: Document, metadata_doc: Document):
        """Create relationship between paired documents."""
        try:
            with self.graph_storage.driver.session(database=self.graph_storage.database) as session:
                session.run("""
                    MATCH (c:Document {doc_id: $content_id})
                    MATCH (m:Document {doc_id: $metadata_id})
                    MERGE (m)-[r:DESCRIBES]->(c)
                    ON CREATE SET
                        r.created_at = $created_at,
                        r.relationship_type = 'metadata_content_pair'
                    ON MATCH SET
                        r.updated_at = $created_at
                """, {
                    "content_id": content_doc.doc_id,
                    "metadata_id": metadata_doc.doc_id,
                    "created_at": datetime.now().isoformat()
                })
                
                self.stats["relationships_created"] += 1
                
        except Exception as e:
            self.logger.error(f"Failed to create paired relationship: {e}")
    
    def _create_category_relationship(self, document: Document):
        """Create relationship to category node."""
        try:
            category = document.metadata.get("dataset_category", "unknown")
            
            with self.graph_storage.driver.session(database=self.graph_storage.database) as session:
                session.run("""
                    MERGE (cat:Category {name: $category})
                    WITH cat
                    MATCH (d:Document {doc_id: $doc_id})
                    MERGE (d)-[r:BELONGS_TO_CATEGORY]->(cat)
                """, {
                    "category": category,
                    "doc_id": document.doc_id
                })
                
        except Exception as e:
            self.logger.error(f"Failed to create category relationship: {e}")
    
    def _create_content_type_relationship(self, document: Document, content_type: str):
        """Create relationship to content type node."""
        try:
            with self.graph_storage.driver.session(database=self.graph_storage.database) as session:
                session.run("""
                    MERGE (ct:ContentType {name: $content_type})
                    WITH ct
                    MATCH (d:Document {doc_id: $doc_id})
                    MERGE (d)-[r:HAS_TYPE]->(ct)
                """, {
                    "content_type": content_type,
                    "doc_id": document.doc_id
                })
                
        except Exception as e:
            self.logger.error(f"Failed to create content type relationship: {e}")
    
    def _create_author_relationship(self, document: Document):
        """Create or update author relationship."""
        try:
            with self.graph_storage.driver.session(database=self.graph_storage.database) as session:
                # Enhanced author processing
                author_name = document.author
                author_email = ""
                
                if "@" in author_name:
                    author_email = author_name
                    # Extract name from email
                    name_part = author_name.split("@")[0]
                    author_name = name_part.replace(".", " ").title()
                
                session.run("""
                    MERGE (a:Person {name: $author_name})
                    ON CREATE SET
                        a.email = $author_email,
                        a.first_seen = $created_at
                    WITH a
                    MATCH (d:Document {doc_id: $doc_id})
                    MERGE (d)-[r:AUTHORED_BY]->(a)
                    ON CREATE SET
                        r.timestamp = $created_at
                """, {
                    "author_name": author_name,
                    "author_email": author_email,
                    "doc_id": document.doc_id,
                    "created_at": datetime.now().isoformat()
                })
                
        except Exception as e:
            self.logger.error(f"Failed to create author relationship: {e}")
    
    def _create_chart_relationships(self, document: Document):
        """Create relationships for chart/graph documents."""
        try:
            chart_analysis = document.metadata.get("chart_analysis", {})
            chart_type = chart_analysis.get("chart_type", "unknown")
            
            with self.graph_storage.driver.session(database=self.graph_storage.database) as session:
                # Create chart type node
                session.run("""
                    MERGE (ct:ChartType {name: $chart_type})
                    WITH ct
                    MATCH (d:Document {doc_id: $doc_id})
                    MERGE (d)-[r:IS_CHART_TYPE]->(ct)
                    SET d.has_visual_data = true
                """, {
                    "chart_type": chart_type,
                    "doc_id": document.doc_id
                })
                
                # If data values detected, create data indicator
                if chart_analysis.get("detected_values"):
                    session.run("""
                        MATCH (d:Document {doc_id: $doc_id})
                        SET d.has_numerical_data = true,
                            d.data_points_count = $count
                    """, {
                        "doc_id": document.doc_id,
                        "count": len(chart_analysis["detected_values"])
                    })
                    
        except Exception as e:
            self.logger.error(f"Failed to create chart relationships: {e}")
    
    def _create_spreadsheet_relationships(self, document: Document):
        """Create relationships for spreadsheet documents."""
        try:
            sheets_info = document.metadata.get("sheets", [])
            
            with self.graph_storage.driver.session(database=self.graph_storage.database) as session:
                for sheet in sheets_info:
                    # Create sheet nodes
                    session.run("""
                        CREATE (s:Sheet {
                            name: $sheet_name,
                            rows: $rows,
                            columns: $columns
                        })
                        WITH s
                        MATCH (d:Document {doc_id: $doc_id})
                        MERGE (d)-[r:CONTAINS_SHEET]->(s)
                    """, {
                        "sheet_name": sheet["name"],
                        "rows": sheet.get("rows", 0),
                        "columns": sheet.get("columns", 0),
                        "doc_id": document.doc_id
                    })
                    
                # Mark as data source
                session.run("""
                    MATCH (d:Document {doc_id: $doc_id})
                    SET d.is_data_source = true,
                        d.sheet_count = $sheet_count
                """, {
                    "doc_id": document.doc_id,
                    "sheet_count": len(sheets_info)
                })
                
        except Exception as e:
            self.logger.error(f"Failed to create spreadsheet relationships: {e}")
    
    def query_by_content_type(self, content_type: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Query documents by content type."""
        try:
            with self.graph_storage.driver.session(database=self.graph_storage.database) as session:
                result = session.run("""
                    MATCH (d:Document)-[:HAS_TYPE]->(ct:ContentType {name: $content_type})
                    RETURN d.doc_id as doc_id, 
                           d.file_name as file_name,
                           d.content_type as content_types,
                           d.created_at as created_at
                    ORDER BY d.created_at DESC
                    LIMIT $limit
                """, {
                    "content_type": content_type,
                    "limit": limit
                })
                
                return [dict(record) for record in result]
                
        except Exception as e:
            self.logger.error(f"Failed to query by content type: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        return self.stats.copy()