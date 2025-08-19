#!/usr/bin/env python3
"""
Simple training script for Mixtral Expert Model
Can be run on any GPU-enabled machine or existing Lambda instance
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Import our training functions
from lambda_training_script import main

def setup_local_environment():
    """Setup environment for local/manual training"""
    print("🚀 Starting Mixtral Expert Model fine-tuning...")
    print("📊 Model: mistralai/Mixtral-8x7B-Instruct-v0.1")
    print("🔧 Method: QLoRA with 4-bit quantization + anti-overfitting")
    print("🛡️ Features: Validation split, early stopping, layer freezing")
    print("📈 Monitoring: Weights & Biases")
    print("=" * 70)
    
    # Check for required files
    data_file = Path("../data_cleaning/training_data.jsonl")
    if not data_file.exists():
        print(f"❌ Training data not found at: {data_file}")
        print("Please ensure training_data.jsonl is in the data_cleaning directory")
        return False
    
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found")
        print("Please copy env_template to .env and fill in your credentials")
        return False
    
    print(f"✅ Training data found: {len(open(data_file).readlines())} examples")
    print("✅ Environment configured")
    return True

if __name__ == "__main__":
    if setup_local_environment():
        print("\n🎯 Starting training pipeline...")
        main()
    else:
        print("\n❌ Setup failed. Please check requirements above.")
        sys.exit(1) 