# src/storage/graph_db.py
from typing import Dict, List, Any, Optional
from neo4j import GraphDatabase
import uuid
import json
from datetime import datetime

class GraphDBStorage:
    """Storage interface for Neo4j graph database to model document relationships."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize connection to Neo4j."""
        self.config = config
        self.uri = config.get("uri", "bolt://neo4j:7687")
        self.user = config.get("user", "neo4j")
        self.password = config.get("password", "password")
        self.database = config.get("database", "documents")
        
        # Connect to Neo4j
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        
        # Initialize database schema
        self._initialize_schema()
    
    def _initialize_schema(self):
        """Initialize Neo4j schema with constraints and indexes."""
        with self.driver.session(database=self.database) as session:
            # Create constraints for unique IDs
            session.run("""
                CREATE CONSTRAINT document_id IF NOT EXISTS
                FOR (d:Document) REQUIRE d.doc_id IS UNIQUE
            """)
            
            session.run("""
                CREATE CONSTRAINT person_email IF NOT EXISTS
                FOR (p:Person) REQUIRE p.email IS UNIQUE
            """)
            
            session.run("""
                CREATE CONSTRAINT thread_id IF NOT EXISTS
                FOR (t:EmailThread) REQUIRE t.thread_id IS UNIQUE
            """)
            
            session.run("""
                CREATE CONSTRAINT topic_name IF NOT EXISTS
                FOR (t:Topic) REQUIRE t.name IS UNIQUE
            """)
            
            # Create indexes for better query performance
            session.run("""
                CREATE INDEX document_file_name IF NOT EXISTS
                FOR (d:Document) ON (d.file_name)
            """)
            
            session.run("""
                CREATE INDEX document_author IF NOT EXISTS
                FOR (d:Document) ON (d.author)
            """)
            
            session.run("""
                CREATE INDEX document_content_type IF NOT EXISTS
                FOR (d:Document) ON (d.content_type)
            """)
    
    def store_document(self, document: Dict[str, Any]) -> bool:
        """Store document node in Neo4j."""
        try:
            with self.driver.session(database=self.database) as session:
                # Convert content_type list to string for Neo4j
                content_type = ", ".join(document.get("content_type", []))
                
                # Store basic document node
                session.run("""
                    MERGE (d:Document {doc_id: $doc_id})
                    ON CREATE SET
                        d.file_name = $file_name,
                        d.file_extension = $file_extension,
                        d.content_type = $content_type,
                        d.created_at = $created_at,
                        d.author = $author,
                        d.created_timestamp = timestamp()
                    ON MATCH SET
                        d.file_name = $file_name,
                        d.file_extension = $file_extension,
                        d.content_type = $content_type,
                        d.created_at = $created_at,
                        d.author = $author
                """, {
                    "doc_id": document.get("doc_id", ""),
                    "file_name": document.get("file_name", ""),
                    "file_extension": document.get("file_extension", ""),
                    "content_type": content_type,
                    "created_at": document.get("created_at", datetime.now().isoformat()),
                    "author": document.get("author", "")
                })
                
                # Process author as Person node if available
                if document.get("author"):
                    self._process_author(session, document)
                
                # Process email-specific relationships if applicable
                if "email" in document.get("content_type", []):
                    self._process_email_document(session, document)
                
                return True
                
        except Exception as e:
            print(f"Error storing document in graph DB: {e}")
            return False
    
    def _process_author(self, session, document):
        """Create Person node for document author and relationship."""
        author = document.get("author", "")
        
        # Skip if no author
        if not author:
            return
            
        # Handle email format authors
        email = ""
        name = author
        
        if "@" in author:
            # Extract name from email if possible
            parts = author.split("@")[0].split(".")
            if len(parts) > 1:
                name = " ".join([p.capitalize() for p in parts])
            email = author
        
        # Create or update Person node
        session.run("""
            MERGE (p:Person {email: $email})
            ON CREATE SET
                p.name = $name
            WITH p
            MATCH (d:Document {doc_id: $doc_id})
            MERGE (d)-[r:AUTHORED_BY]->(p)
            ON CREATE SET
                r.timestamp = $timestamp
        """, {
            "email": email if email else name.lower().replace(" ", ".") + "@placeholder.com",
            "name": name,
            "doc_id": document.get("doc_id", ""),
            "timestamp": document.get("created_at", datetime.now().isoformat())
        })
    
    def _process_email_document(self, session, document):
        """Create email-specific nodes and relationships."""
        doc_id = document.get("doc_id", "")
        metadata = document.get("metadata", {})
        
        # Extract email metadata
        email_from = metadata.get("email_from", "")
        email_to = metadata.get("email_to", "")
        email_cc = metadata.get("email_cc", "")
        email_subject = metadata.get("email_subject", "")
        email_date = metadata.get("email_date", "")
        
        # Skip if minimal email data
        if not email_subject and not email_from:
            return
            
        # Normalize email subject for thread ID
        thread_subject = email_subject.lower().strip()
        if thread_subject.startswith("re:"):
            thread_subject = thread_subject[3:].strip()
        if thread_subject.startswith("fwd:"):
            thread_subject = thread_subject[4:].strip()
            
        # Generate thread ID from subject
        import hashlib
        thread_id = hashlib.md5(thread_subject.encode()).hexdigest()
        
        # Create or update EmailThread node
        session.run("""
            MERGE (t:EmailThread {thread_id: $thread_id})
            ON CREATE SET
                t.subject = $subject,
                t.created_at = $date
            WITH t
            MATCH (d:Document {doc_id: $doc_id})
            MERGE (d)-[r:BELONGS_TO]->(t)
            ON CREATE SET
                r.timestamp = $date
        """, {
            "thread_id": thread_id,
            "subject": email_subject,
            "date": email_date if email_date else document.get("created_at", datetime.now().isoformat()),
            "doc_id": doc_id
        })
        
        # Process email participants
        if email_from:
            self._process_email_participant(session, doc_id, email_from, "SENDER")
            
        if email_to:
            for recipient in email_to.split(","):
                recipient = recipient.strip()
                if recipient:
                    self._process_email_participant(session, doc_id, recipient, "RECIPIENT")
                    
        if email_cc:
            for cc in email_cc.split(","):
                cc = cc.strip()
                if cc:
                    self._process_email_participant(session, doc_id, cc, "CC")
    
    def _process_email_participant(self, session, doc_id, email_address, role):
        """Create Person node for email participant and relationship."""
        # Skip if invalid email
        if not email_address or "@" not in email_address:
            return
            
        # Extract name from email if possible
        name = email_address
        parts = email_address.split("@")[0].split(".")
        if len(parts) > 1:
            name = " ".join([p.capitalize() for p in parts])
        
        # Create or update Person node
        session.run("""
            MERGE (p:Person {email: $email})
            ON CREATE SET
                p.name = $name
            WITH p
            MATCH (d:Document {doc_id: $doc_id})
            MERGE (d)-[r:HAS_PARTICIPANT {role: $role}]->(p)
        """, {
            "email": email_address,
            "name": name,
            "doc_id": doc_id,
            "role": role
        })
    
    def store_document_relations(self, document: Dict[str, Any]) -> bool:
        """Store document relationships in graph database."""
        try:
            # First, store the document itself
            self.store_document(document)
            
            with self.driver.session(database=self.database) as session:
                doc_id = document.get("doc_id", "")
                
                # Process document content for topics
                content = document.get("text_content", "")
                if content:
                    topics = self._extract_topics(content)
                    for topic in topics:
                        self._link_document_to_topic(session, doc_id, topic["name"], topic["relevance"])
                
                # Link document to any referenced documents
                self._process_document_references(session, document)
                
                return True
                
        except Exception as e:
            print(f"Error storing document relations: {e}")
            return False
    
    def _extract_topics(self, content: str) -> List[Dict[str, Any]]:
        """Extract topics from document content.
        
        In a production system, this would use NLP or LLM-based topic extraction.
        For now, we'll use a simple keyword-based approach.
        """
        # Simple keyword frequency-based topic extraction
        from collections import Counter
        import re
        
        # Tokenize and clean text
        tokens = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())
        
        # Remove common stopwords
        stopwords = {"the", "and", "that", "for", "with", "this", "are", "from"}
        filtered_tokens = [token for token in tokens if token not in stopwords]
        
        # Count word frequencies
        word_counts = Counter(filtered_tokens)
        
        # Get top keywords as topics
        top_topics = word_counts.most_common(5)
        
        # Format topics with relevance scores
        max_count = max(count for _, count in top_topics) if top_topics else 1
        topics = [
            {"name": word, "relevance": count / max_count}
            for word, count in top_topics
        ]
        
        return topics
    
    def _link_document_to_topic(self, session, doc_id: str, topic_name: str, relevance: float):
        """Create Topic node and link to document."""
        session.run("""
            MERGE (t:Topic {name: $topic_name})
            WITH t
            MATCH (d:Document {doc_id: $doc_id})
            MERGE (d)-[r:CONTAINS]->(t)
            ON CREATE SET
                r.relevance_score = $relevance
            ON MATCH SET
                r.relevance_score = $relevance
        """, {
            "topic_name": topic_name,
            "doc_id": doc_id,
            "relevance": relevance
        })
    
    def _process_document_references(self, session, document):
        """Identify and create relationships for document references."""
        doc_id = document.get("doc_id", "")
        content = document.get("text_content", "")
        
        # Check for explicitly identified references in metadata
        metadata = document.get("metadata", {})
        if "references" in metadata and isinstance(metadata["references"], list):
            for ref in metadata["references"]:
                ref_id = ref.get("doc_id")
                ref_type = ref.get("type", "REFERENCES")
                
                if ref_id:
                    session.run("""
                        MATCH (d:Document {doc_id: $doc_id})
                        MATCH (ref:Document {doc_id: $ref_id})
                        MERGE (d)-[r:REFERENCES {type: $ref_type}]->(ref)
                    """, {
                        "doc_id": doc_id,
                        "ref_id": ref_id,
                        "ref_type": ref_type
                    })
    
    def find_related_documents(self, doc_id: str, max_depth: int = 2) -> List[Dict[str, Any]]:
        """Find documents related to the given document."""
        try:
            with self.driver.session(database=self.database) as session:
                # Query for related documents through various relationships
                result = session.run("""
                    MATCH (d:Document {doc_id: $doc_id})
                    CALL apoc.path.expand(d, 
                        'CONTAINS|AUTHORED_BY|BELONGS_TO|REFERENCES|HAS_PARTICIPANT', 
                        'Document', 
                        1, $max_depth) YIELD path
                    WITH LAST(NODES(path)) as related
                    WHERE related:Document AND related.doc_id <> $doc_id
                    RETURN related.doc_id as doc_id, 
                           related.file_name as file_name,
                           related.content_type as content_type,
                           related.created_at as created_at,
                           related.author as author,
                           COUNT(path) as relevance
                    ORDER BY relevance DESC
                    LIMIT 10
                """, {
                    "doc_id": doc_id,
                    "max_depth": max_depth
                })
                
                # Process results
                related_docs = []
                for record in result:
                    related_docs.append({
                        "doc_id": record["doc_id"],
                        "file_name": record["file_name"],
                        "content_type": record["content_type"],
                        "created_at": record["created_at"],
                        "author": record["author"],
                        "relevance_score": record["relevance"]
                    })
                    
                return related_docs
                
        except Exception as e:
            print(f"Error finding related documents: {e}")
            return []
    
    def get_document_graph(self, doc_id: str, include_topics: bool = True) -> Dict[str, Any]:
        """Get document and its immediate relationships as a subgraph."""
        try:
            with self.driver.session(database=self.database) as session:
                # Query for document subgraph
                query = """
                    MATCH (d:Document {doc_id: $doc_id})
                    OPTIONAL MATCH (d)-[r1]->(node1)
                    OPTIONAL MATCH (d)<-[r2]-(node2)
                    RETURN d as document,
                           collect(distinct {type: type(r1), target: node1, properties: properties(r1)}) as outgoing,
                           collect(distinct {type: type(r2), source: node2, properties: properties(r2)}) as incoming
                """
                
                result = session.run(query, {"doc_id": doc_id})
                record = result.single()
                
                if not record:
                    return {"nodes": [], "edges": []}
                
                # Process document
                doc_node = record["document"]
                doc_data = {
                    "id": doc_node["doc_id"],
                    "label": "Document",
                    "properties": dict(doc_node)
                }
                
                # Process relationships
                nodes = [doc_data]
                edges = []
                
                # Helper function to add nodes and edges
                def process_relationships(relationships, direction):
                    for rel in relationships:
                        if rel["target"] is None:
                            continue
                            
                        # Skip topic nodes if not included
                        if not include_topics and (
                            "Topic" in str(rel["target"].labels) or 
                            "CONTAINS" == rel["type"]
                        ):
                            continue
                            
                        # Add target node
                        target = rel["target"]
                        target_id = None
                        
                        # Extract the appropriate ID field based on node type
                        if "Document" in str(target.labels):
                            target_id = target["doc_id"]
                        elif "Person" in str(target.labels):
                            target_id = target["email"]
                        elif "EmailThread" in str(target.labels):
                            target_id = target["thread_id"]
                        elif "Topic" in str(target.labels):
                            target_id = target["name"]
                        else:
                            # Generate ID for other node types
                            target_id = str(uuid.uuid4())
                        
                        # Add node if not already added
                        node_exists = any(n["id"] == target_id for n in nodes)
                        if not node_exists:
                            nodes.append({
                                "id": target_id,
                                "label": str(target.labels).replace(":", ""),
                                "properties": dict(target)
                            })
                        
                        # Add edge
                        if direction == "outgoing":
                            edges.append({
                                "source": doc_id,
                                "target": target_id,
                                "label": rel["type"],
                                "properties": rel["properties"]
                            })
                        else:
                            edges.append({
                                "source": target_id,
                                "target": doc_id,
                                "label": rel["type"],
                                "properties": rel["properties"]
                            })
                
                # Process outgoing and incoming relationships
                process_relationships(record["outgoing"], "outgoing")
                process_relationships(record["incoming"], "incoming")
                
                return {
                    "nodes": nodes,
                    "edges": edges
                }
                
        except Exception as e:
            print(f"Error getting document graph: {e}")
            return {"nodes": [], "edges": []}
    
    def close(self):
        """Close the Neo4j connection."""
        self.driver.close()