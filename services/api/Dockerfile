FROM golang:1.22-alpine AS builder

WORKDIR /app

# Copy Go module files and download dependencies
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build the application
RUN CGO_ENABLED=0 GOOS=linux go build -o /api cmd/api/main.go

# Use a minimal Alpine image for the final container
FROM alpine:3.18

WORKDIR /app

# Copy the binary from the builder stage
COPY --from=builder /api /app/api

# Run the application
CMD ["/app/api"]