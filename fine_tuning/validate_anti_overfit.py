#!/usr/bin/env python3
"""
Anti-overfitting validation script
Verifies that all anti-overfitting measures are properly configured
"""

import json
import os
from pathlib import Path
import sys

# Configuration values from lambda_training_script.py
EXPECTED_LORA_RANK = 16
EXPECTED_LORA_ALPHA = 32
EXPECTED_LEARNING_RATE = 1.5e-5
EXPECTED_WEIGHT_DECAY = 0.08
EXPECTED_MAX_EPOCHS = 3
EXPECTED_VALIDATION_SPLIT = 0.2
EXPECTED_DROPOUT = 0.15

def check_anti_overfitting_config():
    """Check that anti-overfitting parameters are correctly set"""
    print("🛡️ Validating Anti-Overfitting Configuration...")
    
    # Read the training script to validate configuration
    script_path = Path("lambda_training_script.py")
    if not script_path.exists():
        print("❌ lambda_training_script.py not found")
        return False
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    checks = []
    
    # Check LoRA rank
    if f"LORA_RANK = {EXPECTED_LORA_RANK}" in content:
        print(f"✅ LoRA Rank: {EXPECTED_LORA_RANK} (low capacity)")
        checks.append(True)
    else:
        print(f"❌ LoRA Rank not set to {EXPECTED_LORA_RANK}")
        checks.append(False)
    
    # Check LoRA alpha
    if f"LORA_ALPHA = {EXPECTED_LORA_ALPHA}" in content:
        print(f"✅ LoRA Alpha: {EXPECTED_LORA_ALPHA} (balanced)")
        checks.append(True)
    else:
        print(f"❌ LoRA Alpha not set to {EXPECTED_LORA_ALPHA}")
        checks.append(False)
    
    # Check learning rate
    if f"LEARNING_RATE = {EXPECTED_LEARNING_RATE}" in content:
        print(f"✅ Learning Rate: {EXPECTED_LEARNING_RATE} (conservative)")
        checks.append(True)
    else:
        print(f"❌ Learning Rate not set to {EXPECTED_LEARNING_RATE}")
        checks.append(False)
    
    # Check weight decay
    if f"WEIGHT_DECAY = {EXPECTED_WEIGHT_DECAY}" in content:
        print(f"✅ Weight Decay: {EXPECTED_WEIGHT_DECAY} (strong regularization)")
        checks.append(True)
    else:
        print(f"❌ Weight Decay not set to {EXPECTED_WEIGHT_DECAY}")
        checks.append(False)
    
    # Check max epochs
    if f"MAX_EPOCHS = {EXPECTED_MAX_EPOCHS}" in content:
        print(f"✅ Max Epochs: {EXPECTED_MAX_EPOCHS} (limited)")
        checks.append(True)
    else:
        print(f"❌ Max Epochs not set to {EXPECTED_MAX_EPOCHS}")
        checks.append(False)
    
    # Check validation split
    if f"VALIDATION_SPLIT = {EXPECTED_VALIDATION_SPLIT}" in content:
        print(f"✅ Validation Split: {EXPECTED_VALIDATION_SPLIT*100}% (monitoring)")
        checks.append(True)
    else:
        print(f"❌ Validation Split not set to {EXPECTED_VALIDATION_SPLIT}")
        checks.append(False)
    
    # Check dropout
    if f"lora_dropout={EXPECTED_DROPOUT}" in content:
        print(f"✅ LoRA Dropout: {EXPECTED_DROPOUT} (noise injection)")
        checks.append(True)
    else:
        print(f"❌ LoRA Dropout not set to {EXPECTED_DROPOUT}")
        checks.append(False)
    
    # Check early stopping
    if "EarlyStoppingCallback" in content and "early_stopping_patience" in content:
        print("✅ Early Stopping: Enabled with patience")
        checks.append(True)
    else:
        print("❌ Early Stopping not properly configured")
        checks.append(False)
    
    # Check layer freezing
    if "freeze_model_layers" in content and "FREEZE_LAYERS = True" in content:
        print("✅ Layer Freezing: 50% of layers frozen")
        checks.append(True)
    else:
        print("❌ Layer Freezing not enabled")
        checks.append(False)
    
    # Check evaluation frequency
    if "eval_steps=50" in content:
        print("✅ Evaluation: Every 50 steps (frequent monitoring)")
        checks.append(True)
    else:
        print("❌ Evaluation frequency not optimized")
        checks.append(False)
    
    return all(checks)

def calculate_expected_split():
    """Calculate expected train/validation split"""
    print("\n📊 Dataset Split Analysis...")
    
    data_file = Path("training_data.jsonl")
    if not data_file.exists():
        data_file = Path("../data_cleaning/training_data.jsonl")
    
    if not data_file.exists():
        print("❌ Training data not found")
        return False
    
    # Count examples
    total_examples = 0
    with open(data_file, 'r') as f:
        for line in f:
            if line.strip():
                total_examples += 1
    
    val_examples = int(total_examples * EXPECTED_VALIDATION_SPLIT)
    train_examples = total_examples - val_examples
    
    print(f"📈 Total Examples: {total_examples}")
    print(f"🎯 Training Examples: {train_examples} ({(1-EXPECTED_VALIDATION_SPLIT)*100:.0f}%)")
    print(f"🔍 Validation Examples: {val_examples} ({EXPECTED_VALIDATION_SPLIT*100:.0f}%)")
    
    # Check if we have enough validation examples
    if val_examples < 10:
        print("⚠️  Warning: Very few validation examples (<10)")
        print("   Consider increasing dataset size or reducing validation split")
        return False
    elif val_examples < 20:
        print("⚠️  Warning: Limited validation examples (<20)")
        print("   Results may be noisy")
    else:
        print("✅ Sufficient validation examples for reliable monitoring")
    
    return True

def check_regularization_strength():
    """Assess overall regularization strength"""
    print("\n🛡️ Regularization Strength Assessment...")
    
    # Calculate effective capacity reduction
    rank_reduction = (64 - EXPECTED_LORA_RANK) / 64 * 100  # vs original
    lr_reduction = (2e-4 - EXPECTED_LEARNING_RATE) / 2e-4 * 100
    wd_increase = (EXPECTED_WEIGHT_DECAY - 0.001) / 0.001 * 100
    
    print(f"📉 LoRA Capacity Reduction: {rank_reduction:.0f}% (rank 64→16)")
    print(f"📉 Learning Rate Reduction: {lr_reduction:.0f}% (2e-4→1.5e-5)")
    print(f"📈 Weight Decay Increase: {wd_increase:.0f}% (0.001→0.08)")
    print(f"🎯 Layer Freezing: 50% of parameters frozen")
    print(f"🎯 Dropout Increase: 50% (0.1→0.15)")
    
    # Risk assessment
    risk_factors = []
    if EXPECTED_LORA_RANK < 8:
        risk_factors.append("Very low LoRA rank may cause underfitting")
    if EXPECTED_LEARNING_RATE < 1e-5:
        risk_factors.append("Very low learning rate may prevent convergence")
    if EXPECTED_WEIGHT_DECAY > 0.1:
        risk_factors.append("Very high weight decay may suppress learning")
    
    if risk_factors:
        print("\n⚠️  Potential Issues:")
        for risk in risk_factors:
            print(f"   • {risk}")
        print("   Monitor training for underfitting signs")
    else:
        print("\n✅ Regularization strength appears balanced")
    
    return len(risk_factors) == 0

def main():
    """Main validation function"""
    print("🛡️ Anti-Overfitting Configuration Validator")
    print("=" * 50)
    
    checks = [
        check_anti_overfitting_config(),
        calculate_expected_split(),
        check_regularization_strength()
    ]
    
    print("\n" + "=" * 50)
    
    if all(checks):
        print("🎉 Anti-overfitting configuration validated!")
        print("\n💡 Monitoring Tips:")
        print("  • Watch train/val loss curves in W&B")
        print("  • Early stopping will trigger if val loss plateaus")
        print("  • Look for 'Good generalization' vs 'Overfitting' messages")
        print("  • If underfitting, consider increasing LoRA rank to 32")
    else:
        print("❌ Anti-overfitting validation failed")
        print("Please fix the issues above before training")
        sys.exit(1)

if __name__ == "__main__":
    main() 