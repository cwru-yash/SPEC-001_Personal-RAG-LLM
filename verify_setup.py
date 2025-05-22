# verify_setup.py
import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.abspath('.'))

# Try importing key modules
try:
    from services.processor.src.models.document import Document
    print("✅ Document model imported successfully")
    
    # More imports to verify
    from services.processor.src.storage.duckdb import DuckDBStorage
    print("✅ DuckDB storage imported successfully")
    
    print("\n🎉 Environment setup verified successfully!")
except Exception as e:
    print(f"❌ Setup verification failed: {e}")