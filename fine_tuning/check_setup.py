#!/usr/bin/env python3
"""
Setup validation script for Mixtral fine-tuning
Checks training data, environment, and dependencies
"""

import json
import os
from pathlib import Path
import sys

def check_training_data():
    """Check training data access and format"""
    print("🔍 Checking training data access...")
    
    # Load environment first
    from dotenv import load_dotenv
    load_dotenv()
    
    huggingface_account = os.getenv('HUGGINGFACE_ACCOUNT')
    huggingface_api_key = os.getenv('HUGGINGFACE_API_KEY')
    
    if not huggingface_account or not huggingface_api_key:
        print("❌ HuggingFace credentials not configured")
        print("   Need HUGGINGFACE_ACCOUNT and HUGGINGFACE_API_KEY in .env")
        return False
    
    dataset_name = f"{huggingface_account}/{os.getenv('DATASET_NAME', 'expert_training_data')}"
    print(f"📊 Checking dataset: {dataset_name}")
    
    try:
        from datasets import load_dataset
        
        # Try to load the dataset
        dataset = load_dataset(dataset_name, split="train", token=huggingface_api_key)
        
        print(f"✅ Dataset accessible: {len(dataset)} examples")
        
        # Check format
        if len(dataset) > 0:
            example = dataset[0]
            if 'text' in example:
                text = example['text']
                if '[INST]' in text and '[/INST]' in text:
                    print(f"✅ Correct Mixtral format detected")
                    sample = text[:100]
                    print(f"📝 Sample: {sample}...")
                else:
                    print("⚠️  Warning: Examples may not be in Mixtral format")
            else:
                print("❌ Missing 'text' field in dataset")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error accessing dataset: {e}")
        print("   Possible issues:")
        print("   • Dataset doesn't exist yet - run upload_dataset_to_hf.py first")
        print("   • Invalid HuggingFace credentials")
        print("   • Dataset is private and token lacks access")
        return False

def check_environment():
    """Check environment variables"""
    print("\n🔧 Checking environment...")
    
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️  .env file not found - copy env_template to .env")
        return False
    
    required_vars = [
        'HUGGINGFACE_ACCOUNT',
        'HUGGINGFACE_API_KEY',
        'WANDB_API_KEY'
    ]
    
    missing = []
    for var in required_vars:
        if not os.getenv(var):
            missing.append(var)
    
    if missing:
        print(f"⚠️  Missing environment variables: {', '.join(missing)}")
        return False
    
    print("✅ Environment variables configured")
    return True

def check_dependencies():
    """Check Python dependencies"""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        'torch',
        'transformers',
        'datasets',
        'peft',
        'bitsandbytes',
        'accelerate',
        'wandb',
        'huggingface_hub'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies installed")
    return True

def check_gpu():
    """Check GPU availability"""
    print("\n🖥️  Checking GPU...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            print(f"✅ GPU available: {gpu_name}")
            print(f"📊 Memory: {memory_gb:.1f} GB")
            print(f"🔢 Device count: {gpu_count}")
            
            if memory_gb < 20:
                print("⚠️  Warning: <20GB VRAM may cause issues with Mixtral")
            
            return True
        else:
            print("⚠️  No GPU detected - training will be very slow")
            return False
    except Exception as e:
        print(f"❌ Error checking GPU: {e}")
        return False

def main():
    """Main validation function"""
    print("🚀 Mixtral Expert Model - Setup Validation")
    print("=" * 50)
    
    checks = [
        check_training_data(),
        check_environment(),
        check_dependencies(),
        check_gpu()
    ]
    
    print("\n" + "=" * 50)
    
    if all(checks):
        print("🎉 Setup validation passed!")
        print("Ready to start fine-tuning with:")
        print("  python finetune_mixtral_lambda.py  # Auto Lambda")
        print("  python run_training_simple.py      # Manual/Local")
    else:
        print("❌ Setup validation failed")
        print("Please fix the issues above before training")
        sys.exit(1)

if __name__ == "__main__":
    main() 