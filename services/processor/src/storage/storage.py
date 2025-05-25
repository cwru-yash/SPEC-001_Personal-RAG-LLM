# storage.py

# 1) DuckDB (for fast SQL analytics)
import duckdb
duck = duckdb.connect('my_knowledge.duckdb')
duck.execute("""
CREATE TABLE IF NOT EXISTS documents (
  id    VARCHAR,
  content TEXT,
  metadata JSON
)
""")
def store_in_duck(doc_id, content, meta):
    duck.execute("INSERT INTO documents VALUES (?, ?, ?)", (doc_id, content, meta))

# 2) ChromaDB (for vector embeddings)
import chromadb
from chromadb.config import Settings
chroma = chromadb.Client(Settings())
collection = chroma.get_or_create_collection("my_rag_index")
def store_in_chroma(doc_id, content, meta, embedding):
    collection.add(
      documents=[content],
      metadatas=[meta],
      ids=[doc_id],
      embeddings=[embedding]
    )

# 3) Neo4j (for graph relationships)
from neo4j import GraphDatabase
neo = GraphDatabase.driver("bolt://localhost:7687", auth=("user","pass"))
def store_in_neo(doc_id, meta):
    with neo.session() as s:
        s.run(
            "MERGE (d:Document {id:$id}) "
            "SET d += $meta",
            id=doc_id, meta=meta
        )
