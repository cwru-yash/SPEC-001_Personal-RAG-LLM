# Personal RAG-LLM System

A local-first, personal LLM system that can query knowledge from multiple PDFs and other document types automatically synced from a Box account.

## Features

- Automatically sync documents from Box
- Extract text from various file types (.pdf, .png, .msg, etc.)
- Organize data using structured dimensions
- Run a local RAG pipeline using open-source models
- Explicit user approval for external LLM queries
- Web-based frontend for querying

## Architecture

The system consists of five layers:
1. Frontend Layer: Web UI for querying
2. API Layer (Go): GraphQL API
3. Processing Layer (Python): Text extraction and classification
4. Knowledge Base: DuckDB, Chroma, Neo4j
5. LLM Layer (Python): Local RAG engine

## Development

### Prerequisites
- Docker and Docker Compose
- Git
- VS Code with Remote Containers extension (recommended)

### Setup
1. Clone the repository
2. Open in VS Code and reopen in container
3. Run `make setup` to initialize the development environment

### Running locally
```bash
make run