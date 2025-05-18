package main

import (
  "fmt"
  "log"
  "net/http"
)

func main() {
  http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
    fmt.Fprintf(w, "Hello from Personal RAG-LLM API!")
  })

  log.Println("API server starting on port 4000...")
  if err := http.ListenAndServe(":4000", nil); err != nil {
    log.Fatalf("Failed to start server: %v", err)
  }
}
