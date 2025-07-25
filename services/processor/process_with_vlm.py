# services/processor/process_with_vlm.py
#!/usr/bin/env python3
"""
Main script to process document dataset using VLM as primary method.
Processes ZIP files containing document pairs and outputs JSON results.
"""

import os
import sys
import asyncio
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.append('src')

# Import Hydra for configuration management
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

# Import our VLM-integrated processor
from pipeline.vlm_integrated_processor import VLMIntegratedProcessor

def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'vlm_processing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )

async def main():
    """Main processing function."""
    
    # Setup argument parsing
    parser = argparse.ArgumentParser(description="Process documents using VLM as primary method")
    parser.add_argument(
        "--input-dir", 
        type=str, 
        default="/Users/yashm/Documents/GitHub/SPEC-001_Personal-RAG-LLM/SPEC-001_Personal-RAG-LLM/data/input",
        help="Input directory containing category folders with ZIP files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./vlm_processing_results",
        help="Output directory for processed results"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Process only one ZIP file from each category for testing"
    )
    parser.add_argument(
        "--category",
        type=str,
        help="Process only specific category (Documents, Emails, etc.)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 VLM Document Processing Started")
    logger.info(f"📂 Input directory: {args.input_dir}")
    logger.info(f"📁 Output directory: {args.output_dir}")
    logger.info(f"🧪 Test mode: {args.test_mode}")
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        logger.error(f"❌ Input directory does not exist: {args.input_dir}")
        return 1
    
    try:
        # Load Hydra configuration
        logger.info("⚙️ Loading configuration...")
        
        config_dir = os.path.abspath("conf")
        
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            # Load configuration with VLM enabled
            config = compose(config_name="config", overrides=["vlm.enabled=true"])
            
            logger.info("✅ Configuration loaded successfully")
            logger.info(f"   VLM enabled: {config.vlm.enabled}")
            logger.info(f"   VLM model: {config.vlm.llava.model}")
            logger.info(f"   VLM timeout: {config.vlm.base_timeout}s")
            
            # Convert to dict for processor
            config_dict = dict(config)
            
            # Initialize VLM processor
            logger.info("🤖 Initializing VLM processor...")
            processor = VLMIntegratedProcessor(config_dict)
            
            # Process the input directory
            logger.info("📄 Starting document processing...")
            
            start_time = datetime.now()
            
            if args.test_mode:
                logger.info("🧪 Running in test mode - processing limited files")
                results = await process_test_mode(processor, args.input_dir, args.output_dir, args.category)
            else:
                logger.info("🏭 Running in full processing mode")
                results = await processor.process_input_directory(args.input_dir, args.output_dir)
            
            total_time = (datetime.now() - start_time).total_seconds()
            
            # Print summary
            print_processing_summary(results, total_time)
            
            # Cleanup
            processor.cleanup()
            
            logger.info("✅ Processing completed successfully!")
            return 0
            
    except Exception as e:
        logger.error(f"❌ Processing failed: {str(e)}")
        logger.exception("Full error details:")
        return 1

async def process_test_mode(processor, input_dir: str, output_dir: str, specific_category: str = None):
    """Process in test mode - one ZIP file per category."""
    
    logger = logging.getLogger(__name__)
    logger.info("🧪 Test mode: Processing one ZIP file per category")
    
    categories = ["Documents", "Emails", "Images", "Presentations", "Spreadsheets"]
    
    if specific_category:
        categories = [specific_category]
        logger.info(f"🎯 Processing only category: {specific_category}")
    
    test_results = {}
    
    for category in categories:
        category_path = os.path.join(input_dir, category)
        
        if not os.path.exists(category_path):
            logger.warning(f"⚠️ Category directory not found: {category_path}")
            continue
        
        # Find first ZIP file in category
        zip_files = [f for f in os.listdir(category_path) if f.endswith('.zip')]
        
        if not zip_files:
            logger.warning(f"⚠️ No ZIP files found in {category}")
            continue
        
        # Process just the first ZIP file
        test_zip = zip_files[0]
        zip_path = os.path.join(category_path, test_zip)
        
        logger.info(f"🧪 Testing {category} with: {test_zip}")
        
        # Create temporary input structure for single file
        temp_category_path = os.path.join("/tmp", f"vlm_test_{category}")
        os.makedirs(temp_category_path, exist_ok=True)
        
        # Copy test file
        import shutil
        test_zip_path = os.path.join(temp_category_path, test_zip)
        shutil.copy2(zip_path, test_zip_path)
        
        # Process the single category
        category_result = await processor._process_category(
            temp_category_path, category, output_dir
        )
        
        test_results[category] = category_result
        
        # Cleanup temp directory
        shutil.rmtree(temp_category_path, ignore_errors=True)
    
    return {
        "processing_mode": "test",
        "category_results": test_results,
        "test_summary": {
            "categories_tested": len(test_results),
            "total_files_processed": sum(len(r.get("processed_pairs", [])) for r in test_results.values())
        }
    }

def print_processing_summary(results: dict, total_time: float):
    """Print a nice summary of processing results."""
    
    print("\n" + "="*60)
    print("📊 VLM DOCUMENT PROCESSING SUMMARY")
    print("="*60)
    
    # Basic stats
    if "statistics" in results:
        stats = results["statistics"]
        print(f"📦 ZIP files processed: {stats.get('zip_files_processed', 0)}")
        print(f"📄 Documents processed: {stats.get('documents_processed', 0)}")
        print(f"🤖 VLM successes: {stats.get('vlm_successes', 0)}")
        print(f"🔄 VLM fallbacks: {stats.get('vlm_fallbacks', 0)}")
        print(f"❌ Errors: {stats.get('errors', 0)}")
        print(f"📈 VLM success rate: {stats.get('vlm_success_rate', 0):.1%}")
    
    print(f"⏱️ Total processing time: {total_time/60:.1f} minutes")
    
    # Category breakdown
    if "category_results" in results:
        print(f"\n📂 CATEGORY BREAKDOWN:")
        for category, category_data in results["category_results"].items():
            success_count = category_data.get("success_count", 0)
            error_count = category_data.get("error_count", 0)
            total_count = success_count + error_count
            
            print(f"   {category}: {success_count}/{total_count} successful")
    
    # VLM performance
    if "vlm_processor_info" in results:
        vlm_info = results["vlm_processor_info"]
        print(f"\n🤖 VLM PROCESSOR INFO:")
        print(f"   Model: {vlm_info.get('model', 'unknown')}")
        print(f"   Enabled: {vlm_info.get('enabled', False)}")
        
        if "performance_metrics" in vlm_info and vlm_info["performance_metrics"]:
            perf = vlm_info["performance_metrics"]
            avg_time = perf.get("average_response_time", 0)
            if avg_time > 0:
                print(f"   Avg response time: {avg_time:.1f}s")
    
    print("="*60)
    
    # Processing advice
    if "statistics" in results:
        stats = results["statistics"]
        vlm_rate = stats.get('vlm_success_rate', 0)
        
        if vlm_rate > 0.8:
            print("🎉 Excellent VLM performance! Your setup is working great.")
        elif vlm_rate > 0.6:
            print("👍 Good VLM performance. Consider checking any fallback cases.")
        elif vlm_rate > 0.3:
            print("⚠️ Moderate VLM performance. Check timeout settings and model availability.")
        else:
            print("❌ Low VLM performance. Check Ollama service and model installation.")

def print_quick_start_guide():
    """Print quick start guide for users."""
    
    print("\n" + "="*60)
    print("🚀 VLM DOCUMENT PROCESSING - QUICK START")
    print("="*60)
    
    print("\n📋 PREREQUISITES:")
    print("1. Ollama service running: ollama serve")
    print("2. LLaVA model installed: ollama pull llava:7b")
    print("3. VLM enabled: export VLM_ENABLED=true")
    
    print("\n🎯 USAGE EXAMPLES:")
    print("# Test mode (process one file per category)")
    print("python3 process_with_vlm.py --test-mode")
    print("")
    print("# Process specific category")
    print("python3 process_with_vlm.py --category Documents --test-mode")
    print("")
    print("# Full processing")
    print("python3 process_with_vlm.py")
    print("")
    print("# Custom directories")
    print("python3 process_with_vlm.py --input-dir /path/to/input --output-dir /path/to/output")
    
    print("\n📁 OUTPUT STRUCTURE:")
    print("vlm_processing_results/")
    print("├── Documents/")
    print("│   └── frdx0256/")
    print("│       ├── frdx0256_content.json")
    print("│       ├── frdx0256_metadata.json")
    print("│       └── frdx0256_pair_analysis.json")
    print("├── Emails/")
    print("│   └── ...")
    print("└── processing_summary.json")
    
    print("="*60)

if __name__ == "__main__":
    # Check if user wants help
    if len(sys.argv) == 1 or "--help" in sys.argv or "-h" in sys.argv:
        print_quick_start_guide()
        
        if "--help" not in sys.argv and "-h" not in sys.argv:
            print("\nRun with --help for detailed argument options")
            sys.exit(0)
    
    # Set VLM enabled by default for this script
    os.environ["VLM_ENABLED"] = "true"
    
    # Run the main processing
    exit_code = asyncio.run(main())
    sys.exit(exit_code)