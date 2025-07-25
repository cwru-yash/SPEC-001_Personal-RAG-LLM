#!/usr/bin/env python3
import requests
import json
import sys

def test_chroma_connection():
    """Test ChromaDB connection and API endpoints."""
    base_url = "http://localhost:8000"
    collection_name = "test_collection"
    
    print("Testing ChromaDB connection...")
    
    # 1. Test heartbeat
    try:
        response = requests.get(f"{base_url}/api/v2/heartbeat")
        print(f"V2 Heartbeat: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"V2 Heartbeat failed: {e}")
    
    try:
        response = requests.get(f"{base_url}/api/v1/heartbeat")
        print(f"V1 Heartbeat: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"V1 Heartbeat failed: {e}")
    
    # 2. Test collections API
    try:
        response = requests.get(f"{base_url}/api/v2/collections")
        print(f"V2 Collections: {response.status_code}")
        if response.status_code == 200:
            print(f"Collections: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"V2 Collections failed: {e}")
    
    try:
        response = requests.get(f"{base_url}/api/v1/collections")
        print(f"V1 Collections: {response.status_code}")
        if response.status_code == 200:
            print(f"Collections: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"V1 Collections failed: {e}")
    
    # 3. Create a test collection
    try:
        create_data = {
            "name": collection_name,
            "metadata": {"test": True}
        }
        response = requests.post(f"{base_url}/api/v2/collections", json=create_data)
        print(f"Create collection: {response.status_code}")
        if response.status_code in (200, 201):
            print(f"Collection created: {response.text}")
    except Exception as e:
        print(f"Create collection failed: {e}")
    
    print("\nTest complete!")

if __name__ == "__main__":
    test_chroma_connection()