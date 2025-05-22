import requests

def test_chroma_endpoints():
    """Test various ChromaDB API endpoints to determine what's available."""
    base_url = "http://chroma:8000"
    
    # Test endpoints to try
    endpoints = [
        "/api/v1/heartbeat",
        "/api/v2/heartbeat",
        "/api/v1/collections",
        "/api/v2/collections",
        "/api/v1",
        "/api/v2",
        "/"
    ]
    
    print("Testing Chroma API endpoints:")
    
    for endpoint in endpoints:
        url = f"{base_url}{endpoint}"
        try:
            response = requests.get(url)
            status = response.status_code
            print(f"{url}: Status {status}")
            
            if status == 200:
                try:
                    # Try to parse as JSON if possible
                    data = response.json()
                    print(f"  Response: {data}")
                except:
                    # If not JSON, print first 100 chars
                    print(f"  Response: {response.text[:100]}...")
        except Exception as e:
            print(f"{url}: Error - {str(e)}")
    
    # Try the openapi spec which might have API details
    try:
        response = requests.get(f"{base_url}/openapi.json")
        if response.status_code == 200:
            spec = response.json()
            # Extract available paths
            paths = list(spec.get("paths", {}).keys())
            print("\nAvailable API paths from OpenAPI spec:")
            for path in sorted(paths):
                print(f"  {path}")
    except Exception as e:
        print(f"Error getting OpenAPI spec: {e}")

if __name__ == "__main__":
    test_chroma_endpoints()