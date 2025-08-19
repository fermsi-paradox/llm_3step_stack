# Fine-tuning Module

This module fine-tunes a Mixtral 7Bx8 model on domain-specific expert data using Lambda GPU instances with 4-bit quantization, PEFT (LoRA), and Weights & Biases monitoring.

ANTI-OVERFITTING CONFIGURATION - Optimized for small datasets

## Overview

The pipeline handles:
1. Training: QLoRA fine-tuning with 4-bit quantization on Lambda GPU
2. Anti-Overfitting: Validation split, early stopping, layer freezing, regularization
3. Monitoring: Real-time metrics via Weights & Biases
4. Merging: Combines LoRA adapters with base model
5. Uploading: Pushes merged model to HuggingFace Hub

All steps run automatically on Lambda GPU instances to avoid local memory issues.

## Model Configuration

- Base Model: mistralai/Mixtral-8x7B-Instruct-v0.1
- Training Data: Your HuggingFace dataset (configurable size)
- Train/Val Split: 80% train / 20% validation (automatic split)
- Method: QLoRA with 4-bit quantization (NF4) + anti-overfitting measures
- Output: {YOUR_ACCOUNT}/EXPERT-Mixtral-8x7B-Instruct

## Anti-Overfitting Measures

### Dataset Management
- Validation Split: 20% held out for monitoring generalization
- Early Stopping: Patience of 3 evaluations (stops if no improvement)
- Random Shuffling: Ensures representative train/val split

### Model Regularization
- Layer Freezing: First 50% of layers frozen to retain base knowledge
- Low LoRA Rank: 16 (vs 64) to limit adapter capacity
- Higher Dropout: 15% LoRA dropout for noise injection
- Conservative Learning: 1.5e-5 learning rate (vs 2e-4)
- Strong Weight Decay: 0.08 (vs 0.001) for L2 regularization

### Training Control
- Limited Epochs: Maximum 3 epochs
- Frequent Evaluation: Every 50 steps to catch overfitting early
- Best Model Loading: Automatically loads checkpoint with lowest validation loss
- Loss Monitoring: Warns if train/val loss difference >10%

## Files

```
fine_tuning/
├── finetune_mixtral_lambda.py      # Main orchestration script
├── lambda_training_script.py       # Training script with anti-overfitting measures
├── run_training_simple.py          # Simple script for manual training
├── check_setup.py                  # Validation script for setup verification
├── setup_hf_dataset.py             # Dataset management
├── upload_dataset_to_hf.py         # HF upload utilities
├── validate_anti_overfit.py        # Training validation
├── env_template                    # Environment variables template
├── requirements.txt                # Python dependencies
├── training.log                    # Training logs (placeholder)
├── training_output.log             # Output logs (placeholder)
└── README.md                       # This file
```

## Setup

### 1. Environment Variables

```bash
cp env_template .env
# Edit .env with your credentials
```

### 2. Required Credentials

- Lambda Labs API key - For GPU instance management
- HuggingFace API token - For model upload/download
- HuggingFace account name - For model repository
- W&B API key - For experiment tracking (optional but recommended)

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Validate Setup

```bash
python check_setup.py
```

## Usage

### Automatic Training Pipeline (Recommended)

Run the complete pipeline (launches Lambda GPU, trains, merges, uploads):

```bash
python finetune_mixtral_lambda.py
```

This will:
1. Launch an A100 GPU instance on Lambda Labs
2. Set up the training environment
3. Fine-tune Mixtral using QLoRA + anti-overfitting measures
4. Monitor training via Weights & Biases with validation tracking
5. Apply early stopping if overfitting detected
6. Merge LoRA adapters with the base model
7. Upload the complete model to HuggingFace
8. Terminate the Lambda instance

### Manual Training

If you prefer to manage Lambda instances manually:

```bash
python run_training_simple.py
```

## Anti-Overfitting Parameters

| Parameter | Value | Purpose | Previous |
|-----------|-------|---------|----------|
| LoRA Rank | 16 | Limit adapter capacity | 64 |
| LoRA Alpha | 32 | Balanced adaptation | 128 |
| Learning Rate | 1.5e-5 | Conservative training | 2e-4 |
| Weight Decay | 0.08 | Strong L2 regularization | 0.001 |
| LoRA Dropout | 0.15 | Noise injection | 0.1 |
| Max Epochs | 3 | Prevent overtraining | No limit |
| Validation Split | 20% | Monitor generalization | None |
| Early Stopping | 3 patience | Stop on plateau | None |
| Layer Freezing | 50% | Retain base knowledge | None |
| Eval Frequency | 50 steps | Catch overfitting early | 100 |

## Expected Costs

- Training Time: 3-5 hours (with early stopping)
- Lambda A100 Cost: ~$1.10/hour
- Total Cost: $3.30-$5.50

## Monitoring & Validation

### Weights & Biases Dashboard

Track anti-overfitting metrics:
- Train vs Validation Loss (key indicator)
- Learning rate schedule
- GPU utilization
- Training speed
- Early stopping triggers
- Layer freeze status

### Key Metrics to Watch

```bash
# Good training signs:
✓ Train/Val loss decreasing together
✓ Loss difference <10%
✓ Gradual convergence

# Overfitting warning signs:
⚠ Train loss << Val loss
⚠ Val loss plateau while train loss drops
⚠ Loss difference >10%
```

### Training Logs

Monitor progress via:
```bash
# If running manually
tail -f training.log

# Look for these messages:
"Good generalization. Train/Val loss difference: 0.05"
"Potential overfitting detected! Train/Val loss difference: 0.15"
```

## Output

The fine-tuned model will be available at:
https://huggingface.co/YOUR_ACCOUNT/EXPERT-Mixtral-8x7B-Instruct

## Usage Example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "YOUR_ACCOUNT/EXPERT-Mixtral-8x7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto",
    torch_dtype="auto"
)

# Format prompt for Mixtral instruction format
prompt = "[INST] What are the key considerations for [YOUR DOMAIN QUESTION]? [/INST]"

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    **inputs, 
    max_length=500, 
    temperature=0.7,
    do_sample=True
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Technical Details

### Anti-Overfitting Configuration
```python
# LoRA with reduced capacity
LoraConfig(
    r=16,                    # Lower rank
    lora_alpha=32,           # Balanced scaling
    lora_dropout=0.15,       # Higher dropout
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"],
)

# Training with regularization
TrainingArguments(
    learning_rate=1.5e-5,              # Conservative LR
    weight_decay=0.08,                 # Strong regularization
    num_train_epochs=3,                # Limited epochs
    evaluation_strategy="steps",       # Frequent eval
    eval_steps=50,                     # Early detection
    load_best_model_at_end=True,       # Best checkpoint
    early_stopping_patience=3,         # Stop on plateau
)
```

### Layer Freezing
```python
# Freeze first 50% of transformer layers
def freeze_model_layers(model, freeze_percentage=0.5):
    for name, param in model.named_parameters():
        if 'layers.' in name:
            layer_num = int(name.split('layers.')[1].split('.')[0])
            if layer_num < total_layers * freeze_percentage:
                param.requires_grad = False
```

## Troubleshooting

### Overfitting Detection
- Symptom: Validation loss increases while training loss decreases
- Solution: Early stopping will trigger automatically
- Manual Check: Monitor W&B dashboard for diverging loss curves

### Underfitting Prevention
- Symptom: Both train/val loss plateau at high values
- Solution: Increase LoRA rank (16→32) or learning rate (1.5e-5→2e-5)
- Note: Start conservative, then increase capacity if needed

### Training Issues
- Memory Errors: Layer freezing + 4-bit quantization should prevent this
- Slow Convergence: Expected with conservative settings
- Early Stopping: May stop at 1-2 epochs if optimal

## Support

For issues or questions:
1. Check validation: Run `python check_setup.py`
2. Monitor W&B: Watch train/val loss curves for overfitting
3. Review logs: Look for "Good generalization" vs "Overfitting detected" messages
4. Adjust hyperparameters: Start conservative, increase capacity if underfitting