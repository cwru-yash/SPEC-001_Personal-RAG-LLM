# Start the infrastructure services
cd docker
docker-compose up -d duckdb-service chroma neo4j

# Wait for services to be ready
echo "Waiting for services to start..."
sleep 30

# Check service health
echo "Checking ChromaDB..."
curl -f http://localhost:8000/api/v2/heartbeat || echo "ChromaDB not ready"

echo "Checking Neo4j..."
curl -f http://localhost:7474 || echo "Neo4j not ready"

echo "Services started!"