import os
import sys
import requests
import time
import subprocess
from pathlib import Path

project_root = Path(file).parent.parent
sys.path.insert(0, str(project_root))
def check_python_dependencies():
    """Check if required Python packages are available."""
    print("🐍 Checking Python Dependencies...")
    required_packages = {
        'pymupdf': 'fitz',
        'requests': 'requests',
        'pathlib': 'pathlib',
        'uuid': 'uuid',
        'json': 'json',
        'tempfile': 'tempfile'
    }

    missing_packages = []
    available_packages = []

    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
            available_packages.append(package_name)
            print(f"  ✅ {package_name}")
        except ImportError:
            missing_packages.append(package_name)
            print(f"  ❌ {package_name}")

    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    else:
        print(f"✅ All {len(available_packages)} required packages are available")
        return True
def check_docker_services():
    """Check if Docker services are accessible."""
    print("\n🐳 Checking Docker Services...")
    services = {
        'DuckDB Service': 'http://localhost:3000',
        'ChromaDB': 'http://localhost:8000/api/v1/heartbeat',
        'Neo4j': 'http://localhost:7474',
        'API Service': 'http://localhost:4000'
    }

    service_status = {}

    for service_name, url in services.items():
        try:
            response = requests.get(url, timeout=5)
            if response.status_code < 500:  # Accept any non-server-error status
                print(f"  ✅ {service_name} - Accessible")
                service_status[service_name] = True
            else:
                print(f"  ⚠️ {service_name} - Server Error (Status: {response.status_code})")
                service_status[service_name] = False
        except requests.exceptions.RequestException as e:
            print(f"  ❌ {service_name} - Not accessible ({str(e)[:50]}...)")
            service_status[service_name] = False

    accessible_services = sum(1 for status in service_status.values() if status)
    total_services = len(service_status)

    print(f"\n📊 Service Status: {accessible_services}/{total_services} accessible")

    if accessible_services == 0:
        print("⚠️ No services are accessible. Make sure Docker containers are running:")
        print("   cd docker && docker-compose up -d")
        return False
    elif accessible_services < total_services:
        print("⚠️ Some services are not accessible, but we can continue with available ones.")
        return True
    else:
        print("✅ All services are accessible!")
        return True
def check_project_structure():
    """Check if the project structure is correct."""
    print("\n📁 Checking Project Structure...")
    required_paths = [
        'services/processor/src',
        'services/processor/src/models',
        'services/processor/src/pipeline',
        'services/processor/src/storage',
        'docker/docker-compose.yml'
    ]

    missing_paths = []
    found_paths = []

    for path in required_paths:
        full_path = project_root / path
        if full_path.exists():
            found_paths.append(path)
            print(f"  ✅ {path}")
        else:
            missing_paths.append(path)
            print(f"  ❌ {path}")

    if missing_paths:
        print(f"\n⚠️ Missing paths: {missing_paths}")
        print("Make sure you're running from the correct project directory.")
        return False
    else:
        print(f"✅ All {len(found_paths)} required paths found")
        return True
def check_docker_containers():
    """Check if Docker containers are running."""
    print("\n🔍 Checking Docker Containers...")
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "table {{.Names}}\t{{.Status}}"],
            capture_output=True, text=True
        )
        
        if result.returncode == 0:
            print("Running containers:")
            print(result.stdout)
            
            # Check for expected containers
            expected_containers = ['chroma', 'neo4j', 'duckdb', 'processor', 'api']
            running_containers = result.stdout.lower()
            
            found_containers = []
            for container in expected_containers:
                if container in running_containers:
                    found_containers.append(container)
            
            print(f"\n📊 Expected containers found: {len(found_containers)}/{len(expected_containers)}")
            
            if len(found_containers) >= 2:  # At least some containers running
                return True
            else:
                print("⚠️ Few or no expected containers are running.")
                return False
        else:
            print("❌ Failed to check Docker containers")
            return False
            
    except Exception as e:
        print(f"❌ Error checking Docker containers: {e}")
        return False
def test_basic_imports():
    """Test basic imports from the project."""
    print("\n📦 Testing Project Imports...")
    try:
        # Test importing core models
        from services.processor.src.models.document import Document
        print("  ✅ Document model import")
        
        # Test creating a basic document
        test_doc = Document(
            doc_id="test-123",
            file_name="test.txt",
            file_extension="txt",
            content_type=["text"],
            text_content="Test content"
        )
        print("  ✅ Document creation")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        print("  💡 Make sure you're running from the project root directory")
        return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False
def main():
    """Main validation function."""
    print("🔬 Environment Validation for Document Processing")
    print("=" * 55)
    checks = [
        ("Python Dependencies", check_python_dependencies),
        ("Project Structure", check_project_structure),
        ("Docker Containers", check_docker_containers),
        ("Docker Services", check_docker_services),
        ("Project Imports", test_basic_imports)
    ]

    results = {}

    for check_name, check_function in checks:
        try:
            results[check_name] = check_function()
        except Exception as e:
            print(f"❌ Error during {check_name}: {e}")
            results[check_name] = False
        
        time.sleep(1)  # Brief pause between checks

    # Summary
    print("\n" + "=" * 55)
    print("📊 VALIDATION SUMMARY")
    print("=" * 55)

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    print(f"Checks Passed: {passed}/{total}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")

    print("\nDetailed Results:")
    for check_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {check_name}")

    if passed == total:
        print("\n🎉 Environment validation successful! Ready to run document processing tests.")
        return True
    elif passed >= total * 0.6:  # 60% pass rate
        print("\n⚠️ Environment partially ready. Some features may not work correctly.")
        return True
    else:
        print("\n❌ Environment validation failed. Please fix the issues above before proceeding.")
        return False
if name == "main":
    success = main()
    sys.exit(0 if success else 1)