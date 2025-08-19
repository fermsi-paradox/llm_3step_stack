#!/usr/bin/env python3
"""
Setup script for HuggingFace dataset upload and validation
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

def check_environment():
    """Check if environment variables are set"""
    print("🔧 Checking environment variables...")
    
    # Load environment variables
    load_dotenv()
    
    required_vars = [
        'HUGGINGFACE_API_KEY',
        'HUGGINGFACE_ACCOUNT'
    ]
    
    missing = []
    for var in required_vars:
        if not os.getenv(var):
            missing.append(var)
    
    if missing:
        print(f"❌ Missing environment variables: {', '.join(missing)}")
        print("Please set these in your .env file:")
        for var in missing:
            print(f"  {var}=your_value_here")
        return False
    
    print("✅ Environment variables configured")
    print(f"   Account: {os.getenv('HUGGINGFACE_ACCOUNT')}")
    print(f"   Dataset will be: {os.getenv('HUGGINGFACE_ACCOUNT')}/{os.getenv('DATASET_NAME', 'expert_training_data')}")
    return True

def check_training_data():
    """Check if training data exists"""
    print("\n📊 Checking training data...")
    
    data_paths = [
        Path("../data_cleaning/training_data.jsonl"),
        Path("training_data.jsonl")
    ]
    
    for path in data_paths:
        if path.exists():
            line_count = sum(1 for line in open(path) if line.strip())
            print(f"✅ Found training data: {path}")
            print(f"   Examples: {line_count}")
            return True
    
    print("❌ Training data not found")
    print("Expected locations:")
    for path in data_paths:
        print(f"  {path}")
    return False

def main():
    """Main setup function"""
    print("🚀 HuggingFace Dataset Setup")
    print("=" * 50)
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Check training data
    if not check_training_data():
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("✅ Setup validation passed!")
    print("\nNext steps:")
    print("1. Run the upload script:")
    print("   python upload_dataset_to_hf.py")
    print("2. Start training:")
    print("   python finetune_mixtral_lambda.py")
    
    # Ask if user wants to upload now
    try:
        upload_now = input("\nWould you like to upload the dataset now? (y/N): ").lower().strip()
        if upload_now in ['y', 'yes']:
            print("\n🚀 Uploading dataset...")
            from upload_dataset_to_hf import create_and_upload_dataset
            create_and_upload_dataset()
        else:
            print("💡 Run 'python upload_dataset_to_hf.py' when ready to upload")
    except KeyboardInterrupt:
        print("\n👋 Setup complete. Run upload script manually when ready.")

if __name__ == "__main__":
    main() 