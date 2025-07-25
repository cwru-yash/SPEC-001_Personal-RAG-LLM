# debug_config.py
# Debug script to understand what Hydra is actually loading

import os
import sys
from pathlib import Path
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

def debug_configuration():
    """Debug the Hydra configuration to see what's actually being loaded."""
    
    print("🔍 Debugging Hydra Configuration")
    print("=" * 50)
    
    # Step 1: Check if configuration files exist
    print("\n1. 📁 Checking configuration files:")
    
    config_dir = Path("conf")
    main_config = config_dir / "config.yaml"
    vlm_config = config_dir / "vlm" / "cpu_optimized.yaml"
    
    print(f"   Config directory: {config_dir.absolute()}")
    print(f"   Main config exists: {main_config.exists()} - {main_config}")
    print(f"   VLM config exists: {vlm_config.exists()} - {vlm_config}")
    
    if not config_dir.exists():
        print("❌ Configuration directory 'conf' not found!")
        print("   Make sure you're running this from the processor directory")
        return False
    
    if not main_config.exists():
        print("❌ Main config.yaml not found!")
        return False
    
    # Step 2: Show contents of configuration files
    print("\n2. 📄 Configuration file contents:")
    
    try:
        with open(main_config, 'r') as f:
            main_content = f.read()
        print(f"\n   config.yaml contents:")
        print("   " + "-" * 30)
        for line_num, line in enumerate(main_content.split('\n'), 1):
            print(f"   {line_num:2d}: {line}")
    except Exception as e:
        print(f"   ❌ Error reading config.yaml: {e}")
    
    if vlm_config.exists():
        try:
            with open(vlm_config, 'r') as f:
                vlm_content = f.read()
            print(f"\n   vlm/cpu_optimized.yaml contents (first 20 lines):")
            print("   " + "-" * 40)
            for line_num, line in enumerate(vlm_content.split('\n')[:20], 1):
                print(f"   {line_num:2d}: {line}")
            print("   ... (truncated)")
        except Exception as e:
            print(f"   ❌ Error reading vlm config: {e}")
    
    # Step 3: Try to load configuration and see what we get
    print("\n3. 🔄 Loading configuration with Hydra:")
    
    try:
        config_dir_abs = str(config_dir.absolute())
        
        with initialize_config_dir(config_dir=config_dir_abs, version_base=None):
            # Load without any overrides first
            config = compose(config_name="config")
            
            print("   ✅ Configuration loaded successfully!")
            
            # Show the structure we actually got
            print(f"\n4. 🏗️ Actual configuration structure:")
            print("   " + "-" * 35)
            
            # Convert to dict for easier inspection
            config_dict = OmegaConf.to_container(config, resolve=True)
            
            # Show top-level keys
            print(f"   Top-level keys: {list(config_dict.keys())}")
            
            # Check if vlm exists
            if 'vlm' in config_dict:
                print(f"   ✅ 'vlm' key found")
                vlm_config = config_dict['vlm']
                print(f"   VLM keys: {list(vlm_config.keys()) if isinstance(vlm_config, dict) else 'Not a dict!'}")
                
                # Check specifically for 'enabled'
                if isinstance(vlm_config, dict):
                    if 'enabled' in vlm_config:
                        print(f"   ✅ 'enabled' key found: {vlm_config['enabled']}")
                    else:
                        print(f"   ❌ 'enabled' key NOT found in vlm config")
                        print(f"   Available vlm keys: {list(vlm_config.keys())}")
                else:
                    print(f"   ❌ vlm is not a dictionary: {type(vlm_config)} = {vlm_config}")
            else:
                print(f"   ❌ 'vlm' key NOT found in configuration")
            
            # Show full config structure (truncated)
            print(f"\n5. 📋 Full configuration preview:")
            print("   " + "-" * 30)
            config_yaml = OmegaConf.to_yaml(config)
            for line_num, line in enumerate(config_yaml.split('\n')[:30], 1):
                print(f"   {line_num:2d}: {line}")
            if len(config_yaml.split('\n')) > 30:
                print("   ... (truncated)")
                
            return True
            
    except Exception as e:
        print(f"   ❌ Configuration loading failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        print(f"   Traceback:")
        for line in traceback.format_exc().split('\n'):
            print(f"      {line}")
        return False

def suggest_fixes():
    """Suggest potential fixes based on common issues."""
    
    print("\n" + "=" * 50)
    print("💡 COMMON FIXES FOR CONFIGURATION ISSUES")
    print("=" * 50)
    
    print("\n1. 🔧 Check defaults section in config.yaml:")
    print("   Make sure your config.yaml has:")
    print("   ```yaml")
    print("   defaults:")
    print("     - _self_")
    print("     - vlm: cpu_optimized")
    print("   ```")
    
    print("\n2. 🔧 Avoid conflicting vlm definitions:")
    print("   If you have both:")
    print("   - A defaults entry: `- vlm: cpu_optimized`")
    print("   - AND a vlm section in the same file")
    print("   They might conflict. Try removing one.")
    
    print("\n3. 🔧 Check file structure:")
    print("   Ensure you have:")
    print("   conf/")
    print("   ├── config.yaml")
    print("   └── vlm/")
    print("       └── cpu_optimized.yaml")
    
    print("\n4. 🔧 Environment variable override:")
    print("   The error might be from trying to override with:")
    print("   vlm:")
    print("     enabled: ${oc.env:VLM_ENABLED,false}")
    # print("     enabled: 'enabled")
    print("   Try removing this and put 'enabled' in cpu_optimized.yaml")

if __name__ == "__main__":
    print(f"🏃 Running from: {os.getcwd()}")
    print(f"🐍 Python path: {sys.path[:3]}...")  # Show first few entries
    
    success = debug_configuration()
    
    if not success:
        suggest_fixes()
    
    print("\n" + "=" * 50)