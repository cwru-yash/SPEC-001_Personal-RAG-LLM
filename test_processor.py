import os
import sys
import hydra
from omegaconf import DictConfig

# Add the project root to path
# sys.path.append('services/processor')
sys.path.insert(0, os.path.abspath('.'))

# Import the processor
from services.processor.src.pipeline.pdf_processor import PDFProcessor
from services.processor.src.storage.duckdb import DuckDBStorage

os.environ['DATA_DIR'] = './data'
os.environ['CACHE_DIR'] = './cache'

# @hydra.main(config_path="services/processor/conf", config_name="config")
@hydra.main(config_path="services/processor/conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Test the processor with a basic file."""
    print(f"Config: {cfg}")
    
    # Initialize storage
    db_path = os.path.join(cfg.data_dir, "metadata.db")
    storage = DuckDBStorage(db_path)
    
    # Check if storage is connected
    print(f"DuckDB connected: {storage.conn is not None}")
    
    print("Environment test successful!")

if __name__ == "__main__":
    main()
