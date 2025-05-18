.PHONY: setup run build test clean proto

# Development environment setup
setup:
	@echo "Setting up development environment..."
	@docker-compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml build

# Run all services
run:
	@echo "Starting all services..."
	@docker-compose -f docker/docker-compose.yml up

# Build all Docker images
build:
	@echo "Building Docker images..."
	@docker-compose -f docker/docker-compose.yml build

# Run tests for all services
test:
	@echo "Running tests..."
	@docker-compose -f docker/docker-compose.yml -f docker/docker-compose.test.yml up --exit-code-from tests

# Generate code from Protocol Buffers
proto:
	@echo "Generating code from Protocol Buffers..."
	@cd proto && protoc --go_out=../services/api/pkg --go-grpc_out=../services/api/pkg \
		--python_out=../services/processor/src --grpc_python_out=../services/processor/src \
		*.proto

# Clean up generated files and containers
clean:
	@echo "Cleaning up..."
	@docker-compose -f docker/docker-compose.yml down -v
	@find . -type d -name __pycache__ -exec rm -rf {} +
	@find . -type d -name .pytest_cache -exec rm -rf {} +
	@find . -type d -name node_modules -exec rm -rf {} +
