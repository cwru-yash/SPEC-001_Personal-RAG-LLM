# test_cpu_vlm.py
# Test script to verify CPU-optimized VLM configuration works correctly

import asyncio
import logging
import sys
import os
from datetime import datetime
from pathlib import Path

# Add the src directory to Python path so we can import our modules
sys.path.append('src')

# Import Hydra for configuration management
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

# Import our CPU-optimized VLM processor
from pipeline.vlm.cpu_optimized_processor import CPUOptimizedVLMProcessor

class VLMTester:
    """Test harness for CPU-optimized VLM processing."""
    
    def __init__(self):
        """Initialize the tester with proper configuration loading."""
        
        # Set up logging to see what's happening
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(f'vlm_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.config = None
        self.vlm_processor = None
        
    def load_configuration(self):
        """Load the Hydra configuration with our CPU-optimized VLM settings."""
        
        try:
            # Get the absolute path to the configuration directory
            config_dir = os.path.abspath("conf")
            
            # Initialize Hydra with our configuration directory
            with initialize_config_dir(config_dir=config_dir, version_base=None):
                # Compose the configuration
                # This loads the main config.yaml which includes vlm: cpu_optimized
                self.config = compose(config_name="config")
                
                self.logger.info("✅ Configuration loaded successfully")
                self.logger.info(f"   VLM enabled: {self.config.vlm.enabled}")
                self.logger.info(f"   VLM processor type: {self.config.vlm.get('processor_type', 'unknown')}")
                self.logger.info(f"   Base timeout: {self.config.vlm.base_timeout}s")
                
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Configuration loading failed: {str(e)}")
            self.logger.error("   Make sure you're running this from the processor directory")
            self.logger.error("   and that conf/vlm/cpu_optimized.yaml exists")
            return False
    
    def initialize_vlm_processor(self):
        """Initialize the VLM processor with our configuration."""
        
        try:
            # Convert OmegaConf to regular dict for our processor
            config_dict = dict(self.config)
            
            # Create the CPU-optimized VLM processor
            self.vlm_processor = CPUOptimizedVLMProcessor(config_dict)
            
            if self.vlm_processor.enabled:
                self.logger.info("✅ VLM processor initialized and enabled")
                self.logger.info(f"   Model: {self.vlm_processor.model}")
                self.logger.info(f"   API endpoint: {self.vlm_processor.api_endpoint}")
                self.logger.info(f"   Current timeout: {self.vlm_processor.get_current_timeout():.1f}s")
                return True
            else:
                self.logger.warning("⚠️ VLM processor initialized but disabled")
                self.logger.warning("   Set VLM_ENABLED=true to enable VLM processing")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ VLM processor initialization failed: {str(e)}")
            return False
    
    async def test_ollama_connectivity(self):
        """Test that Ollama is running and accessible."""
        
        self.logger.info("🔍 Testing Ollama connectivity...")
        
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.vlm_processor.api_endpoint}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        models = [m['name'] for m in data.get('models', [])]
                        self.logger.info(f"✅ Ollama is accessible")
                        self.logger.info(f"   Available models: {models}")
                        
                        # Check if our model is available
                        if self.vlm_processor.model in models:
                            self.logger.info(f"✅ Required model '{self.vlm_processor.model}' is available")
                            return True
                        else:
                            self.logger.error(f"❌ Required model '{self.vlm_processor.model}' not found")
                            self.logger.error(f"   Run: ollama pull {self.vlm_processor.model}")
                            return False
                    else:
                        self.logger.error(f"❌ Ollama responded with status {response.status}")
                        return False
                        
        except Exception as e:
            self.logger.error(f"❌ Cannot connect to Ollama: {str(e)}")
            self.logger.error("   Make sure Ollama is running: ollama serve")
            return False
    
    async def test_single_page_processing(self, pdf_path: str):
        """Test processing a single page with detailed timeout information."""
        
        self.logger.info(f"🔍 Testing single page processing with: {pdf_path}")
        
        try:
            # Check if PDF file exists
            if not os.path.exists(pdf_path):
                self.logger.error(f"❌ PDF file not found: {pdf_path}")
                return False
            
            # Record start time for total processing measurement
            start_time = datetime.now()
            
            # Process the document with our CPU-optimized processor
            self.logger.info("📄 Starting document processing...")
            self.logger.info(f"   Configured timeout: {self.vlm_processor.get_current_timeout():.1f}s")
            self.logger.info(f"   Max pages to process: {self.vlm_processor.max_pages_per_document}")
            
            # This is where the actual VLM processing happens
            result = await self.vlm_processor.process_document_with_progress(pdf_path)
            
            # Calculate total processing time
            total_time = (datetime.now() - start_time).total_seconds()
            
            # Analyze and report results
            self.logger.info(f"🎉 Processing completed in {total_time:.1f}s")
            
            # Show what happened with each page
            if result.get("pages"):
                for page_result in result["pages"]:
                    page_num = page_result["page_number"]
                    vlm_result = page_result["vlm_result"]
                    processing_time = page_result["processing_time"]
                    
                    if vlm_result["success"]:
                        self.logger.info(f"   Page {page_num + 1}: ✅ Success in {processing_time:.1f}s")
                        self.logger.info(f"     Content type: {vlm_result['content_type']}")
                        self.logger.info(f"     Confidence: {vlm_result['confidence']:.2f}")
                        if vlm_result.get('error_message'):
                            self.logger.info(f"     Note: {vlm_result['error_message']}")
                    else:
                        self.logger.warning(f"   Page {page_num + 1}: ❌ Failed in {processing_time:.1f}s")
                        self.logger.warning(f"     Error: {vlm_result.get('error_message', 'Unknown error')}")
                        if page_result.get("traditional_fallback"):
                            self.logger.info(f"     Fallback text length: {len(page_result['traditional_fallback'])} chars")
            
            # Show performance metrics
            if result.get("performance_metrics"):
                metrics = result["performance_metrics"]
                self.logger.info(f"📊 Performance Summary:")
                self.logger.info(f"   Success rate: {metrics['success_rate']:.1%}")
                self.logger.info(f"   Average page time: {metrics.get('average_page_time', 0):.1f}s")
                self.logger.info(f"   Current timeout: {metrics.get('current_timeout', 0):.1f}s")
            
            # Show what the system learned
            processor_metrics = self.vlm_processor.performance_metrics
            if processor_metrics.api_response_times:
                self.logger.info(f"🧠 System Learning:")
                self.logger.info(f"   Measurements taken: {len(processor_metrics.api_response_times)}")
                self.logger.info(f"   Average response time: {processor_metrics.average_response_time:.1f}s")
                self.logger.info(f"   95th percentile time: {processor_metrics.p95_response_time:.1f}s")
                
                # Show how timeout would adapt
                next_timeout = self.vlm_processor.get_current_timeout()
                self.logger.info(f"   Next timeout would be: {next_timeout:.1f}s")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Single page processing failed: {str(e)}")
            return False
    
    def show_timeout_explanation(self):
        """Explain how the timeout system works."""
        
        self.logger.info("\n" + "="*60)
        self.logger.info("🎓 UNDERSTANDING TIMEOUT BEHAVIOR")
        self.logger.info("="*60)
        
        self.logger.info("The timeout system works in layers:")
        self.logger.info("")
        self.logger.info("1. 🕐 INITIAL TIMEOUT")
        self.logger.info(f"   Starts at: {self.config.vlm.base_timeout}s")
        self.logger.info("   This is a conservative starting point for CPU processing")
        self.logger.info("")
        
        self.logger.info("2. 🧠 ADAPTIVE LEARNING")
        self.logger.info(f"   After each successful request, the system learns:")
        self.logger.info(f"   - Average response time for your hardware")
        self.logger.info(f"   - 95th percentile time (accounts for occasional slow responses)")
        self.logger.info(f"   - Adjusts timeout to: P95_time × {self.config.vlm.timeout_multiplier}")
        self.logger.info("")
        
        self.logger.info("3. 🛡️ SAFETY BOUNDS")
        self.logger.info(f"   Minimum timeout: {self.config.vlm.min_timeout}s (even fast responses need time)")
        self.logger.info(f"   Maximum timeout: {self.config.vlm.max_timeout}s (prevents infinite waits)")
        self.logger.info("")
        
        self.logger.info("4. 🔄 RETRY LOGIC")
        self.logger.info(f"   If timeout occurs, retry up to {self.config.vlm.max_retries} times")
        self.logger.info(f"   Wait between retries: 2^attempt seconds (max {self.config.vlm.retry_backoff_max}s)")
        self.logger.info("")
        
        self.logger.info("5. 🛟 FALLBACK BEHAVIOR")
        self.logger.info("   If all retries fail:")
        self.logger.info("   - Falls back to traditional OCR processing")
        self.logger.info("   - Still produces useful results")
        self.logger.info("   - Continues processing other pages")
        self.logger.info("")
        
        self.logger.info("This design ensures your system is both patient with CPU processing")
        self.logger.info("and responsive when things go wrong.")
        self.logger.info("="*60 + "\n")
    
    async def run_full_test(self, pdf_path: str):
        """Run the complete test suite."""
        
        self.logger.info("🚀 Starting CPU-Optimized VLM Test Suite")
        self.logger.info("="*60)
        
        # Step 1: Load configuration
        if not self.load_configuration():
            return False
        
        # Step 2: Initialize VLM processor
        if not self.initialize_vlm_processor():
            return False
        
        # Step 3: Show timeout explanation
        self.show_timeout_explanation()
        
        # Step 4: Test Ollama connectivity
        if not await self.test_ollama_connectivity():
            return False
        
        # Step 5: Test actual document processing
        if not await self.test_single_page_processing(pdf_path):
            return False
        
        self.logger.info("🎉 All tests completed successfully!")
        self.logger.info("Your CPU-optimized VLM setup is working correctly.")
        return True

# Main execution
async def main():
    """Main test function."""
    
    # Path to your test PDF
    pdf_path = "/Users/yashm/Documents/GitHub/SPEC-001_Personal-RAG-LLM/SPEC-001_Personal-RAG-LLM/data/input/Documents/frdx0256/frdx0256.pdf"
    
    # Create and run the tester
    tester = VLMTester()
    success = await tester.run_full_test(pdf_path)
    
    if success:
        print("\n✅ Test completed successfully!")
        print("You can now integrate this CPU-optimized VLM processor into your main pipeline.")
    else:
        print("\n❌ Test failed. Check the log messages above for specific issues to fix.")

if __name__ == "__main__":
    # Enable VLM for testing
    os.environ["VLM_ENABLED"] = "true"
    
    # Run the test
    asyncio.run(main())