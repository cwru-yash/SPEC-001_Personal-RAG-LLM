import os
import sys
import subprocess
import logging
from pathlib import Path
from datetime import datetime

def setup_logging():
    """Setup logging for test execution."""
    logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
    logging.StreamHandler(sys.stdout),
    logging.FileHandler(f'test_processing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
    )
    return logging.getLogger(name)
def check_docker_services():
    """Check if Docker services are running."""
    logger = logging.getLogger(name)
    try:
        # Check if docker-compose services are running
        result = subprocess.run(
            ["docker-compose", "-f", "docker/docker-compose.yml", "ps"], 
            capture_output=True, text=True, cwd="."
        )
        
        if result.returncode == 0:
            logger.info("✅ Docker Compose services status:")
            print(result.stdout)
            return True
        else:
            logger.error("❌ Failed to check Docker services")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error checking Docker services: {e}")
        return False
def create_test_directories():
    """Create necessary test directories."""
    logger = logging.getLogger(name)
    directories = [
        "test_processing",
        "test_processing/sample_documents",
        "test_processing/output",
        "test_processing/logs"
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"📁 Created directory: {directory}")
def run_test_script(script_name, description):
    """Run a specific test script."""
    logger = logging.getLogger(name)
    logger.info(f"🔄 Running: {description}")
    logger.info("=" * 50)

    try:
        result = subprocess.run([sys.executable, f"test_processing/{script_name}"], 
                            capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"✅ {description} - PASSED")
            if result.stdout:
                print(result.stdout)
            return True
        else:
            logger.error(f"❌ {description} - FAILED")
            if result.stderr:
                print("Error output:")
                print(result.stderr)
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed to run {script_name}: {e}")
        return False
def main():
    """Main test execution function."""
    logger = setup_logging()
    print("🚀 Personal RAG-LLM Document Processing Tests")
    print("=" * 60)
    print(f"Started at: {datetime.now().isoformat()}")
    print()

    # Step 1: Create test directories
    logger.info("📁 Setting up test environment...")
    create_test_directories()

    # Step 2: Check Docker services
    logger.info("🐳 Checking Docker services...")
    if not check_docker_services():
        logger.error("❌ Docker services are not running. Please start them first:")
        logger.error("   cd docker && docker-compose up -d")
        return False

    # Step 3: Run test scripts in sequence
    test_scripts = [
        ("validate_environment.py", "Environment Validation"),
        ("create_test_data.py", "Test Data Creation"),
        ("test_documents.py", "Document Processing Tests")
    ]

    results = {}

    for script, description in test_scripts:
        results[description] = run_test_script(script, description)
        print()  # Add spacing between tests

    # Step 4: Generate summary report
    logger.info("📊 Test Summary")
    logger.info("=" * 30)

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    logger.info(f"Total Tests: {total}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {total - passed}")
    logger.info(f"Success Rate: {(passed/total)*100:.1f}%")

    print("\nDetailed Results:")
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")

    # Step 5: Next steps
    if passed == total:
        logger.info("\n🎉 All tests passed! Your document processing layer is working correctly.")
        logger.info("\n📋 Next Steps:")
        logger.info("1. Check test_processing/output/ for processed documents")
        logger.info("2. Review test_processing/logs/ for detailed logs")
        logger.info("3. Try processing your own documents")
        logger.info("4. Scale up to larger datasets")
    else:
        logger.info("\n⚠️ Some tests failed. Please check the error messages above.")
        logger.info("\n🔧 Troubleshooting:")
        logger.info("1. Ensure all Docker services are running")
        logger.info("2. Check Python dependencies are installed")
        logger.info("3. Verify file permissions")
        logger.info("4. Review error logs for specific issues")

    return passed == total
if name == "main":
    success = main()
    sys.exit(0 if success else 1)
