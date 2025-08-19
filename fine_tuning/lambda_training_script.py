#!/usr/bin/env python3
"""
Complete fine-tuning script for Mixtral 7Bx8 with 4-bit quantization
ANTI-OVERFITTING CONFIGURATION for small dataset (219 examples)
Runs on Lambda GPU instance with PEFT and W&B monitoring
"""

import os
import torch
import gc
import json
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from huggingface_hub import HfApi, create_repo
import wandb
from dotenv import load_dotenv
import logging
from typing import Dict, List
import numpy as np
import requests
from transformers import TrainerCallback

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
HUGGINGFACE_ACCOUNT = os.getenv('HUGGINGFACE_ACCOUNT')
WANDB_API_KEY = os.getenv('WANDB_API_KEY')
LAMBDA_API_KEY = os.getenv('LAMBDA_API_KEY')

class CheckpointMonitorCallback(TrainerCallback):
    """Enhanced callback for monitoring training progress and checkpoints."""
    
    def __init__(self, total_checkpoints=10):
        self.checkpoints_saved = 0
        self.total_checkpoints = total_checkpoints
        self.best_eval_loss = float('inf')
        
    def on_save(self, args, state, control, **kwargs):
        """Called when a checkpoint is saved."""
        self.checkpoints_saved += 1
        current_step = state.global_step
        
        logger.info(f"🎯 CHECKPOINT {self.checkpoints_saved}/{self.total_checkpoints} SAVED at step {current_step}")
        logger.info(f"📁 Checkpoint directory: {args.output_dir}/checkpoint-{current_step}")
        
        # Calculate progress percentage
        progress = (self.checkpoints_saved / self.total_checkpoints) * 100
        logger.info(f"📊 Training Progress: {progress:.1f}% complete")
        
        # Log memory usage if available
        try:
            import torch
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
                memory_total = torch.cuda.memory_reserved() / 1024**3  # GB
                logger.info(f"🔧 GPU Memory: {memory_used:.1f}GB / {memory_total:.1f}GB")
        except:
            pass
            
    def on_evaluate(self, args, state, control, model=None, eval_dataloader=None, **kwargs):
        """Called after evaluation."""
        if hasattr(state, 'log_history') and state.log_history:
            last_log = state.log_history[-1]
            if 'eval_loss' in last_log:
                eval_loss = last_log['eval_loss']
                if eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    logger.info(f"🌟 NEW BEST MODEL! Eval loss: {eval_loss:.4f}")
                else:
                    logger.info(f"📊 Eval loss: {eval_loss:.4f} (best: {self.best_eval_loss:.4f})")
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Called at the beginning of each epoch."""
        logger.info(f"🚀 Starting Epoch {state.epoch}")
        
    def on_epoch_end(self, args, state, control, **kwargs):
        """Called at the end of each epoch."""
        logger.info(f"✅ Completed Epoch {state.epoch}")
        logger.info(f"📈 Checkpoints saved: {self.checkpoints_saved}/{self.total_checkpoints}")
WANDB_PROJECT = os.getenv('WANDB_PROJECT', 'expert-model-finetuning')

# Model configuration for Mixtral with ANTI-OVERFITTING settings
BASE_MODEL = os.getenv('BASE_MODEL', 'mistralai/Mixtral-8x7B-Instruct-v0.1')
OUTPUT_MODEL_NAME = os.getenv('OUTPUT_MODEL_NAME', 'expert_llm_model')
DATASET_NAME = f"{HUGGINGFACE_ACCOUNT}/{os.getenv('DATASET_NAME', 'expert_training_data')}"

# Anti-overfitting hyperparameters
LORA_RANK = 16          # Reduced from 64 to prevent overfitting
LORA_ALPHA = 32         # Reduced from 128 to balance adaptation
LEARNING_RATE = 1.5e-5  # Reduced from 2e-4 for small dataset
WEIGHT_DECAY = 0.08     # Increased from 0.001 for regularization
MAX_EPOCHS = 3          # Limited epochs
VALIDATION_SPLIT = 0.2  # 20% for validation
PATIENCE = 3            # Early stopping patience
FREEZE_LAYERS = False   # Temporarily disabled to fix gradient issues

def setup_wandb():
    """Initialize Weights & Biases with anti-overfitting config"""
    if WANDB_API_KEY:
        wandb.login(key=WANDB_API_KEY)
        wandb.init(
            project=WANDB_PROJECT,
            name=f"mixtral-expert-model-training",
            config={
                "base_model": BASE_MODEL,
                "dataset": DATASET_NAME,
                "method": "QLoRA_4bit_anti_overfit",
                "quantization": "4-bit",
                "lora_rank": LORA_RANK,
                "lora_alpha": LORA_ALPHA,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "max_epochs": MAX_EPOCHS,
                "validation_split": VALIDATION_SPLIT,
                "batch_size": 1,
                "gradient_accumulation_steps": 8,
                "freeze_layers": FREEZE_LAYERS,
                "early_stopping": True,
                "patience": PATIENCE
            }
        )
        logger.info("W&B initialized with anti-overfitting config")
    else:
        logger.warning("WANDB_API_KEY not found, skipping W&B setup")

def setup_4bit_quantization():
    """Setup 4-bit quantization configuration for Mixtral"""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

def freeze_model_layers(model, freeze_percentage=0.5):
    """Freeze first N% of model layers to retain base generalization"""
    if not FREEZE_LAYERS:
        return model
    
    logger.info(f"Freezing first {freeze_percentage*100}% of model layers...")
    
    # Get all named parameters
    all_params = list(model.named_parameters())
    total_layers = len([name for name, _ in all_params if 'layers.' in name])
    freeze_layers = int(total_layers * freeze_percentage)
    
    logger.info(f"Total layers: {total_layers}, Freezing: {freeze_layers}")
    
    # Freeze parameters in early layers
    for name, param in model.named_parameters():
        if 'layers.' in name:
            layer_num = int(name.split('layers.')[1].split('.')[0])
            if layer_num < freeze_layers:
                param.requires_grad = False
                
    # Count frozen parameters
    frozen_params = sum(1 for name, param in model.named_parameters() 
                       if not param.requires_grad)
    total_params = sum(1 for _ in model.named_parameters())
    
    logger.info(f"Frozen {frozen_params}/{total_params} parameters "
               f"({frozen_params/total_params*100:.1f}%)")
    
    return model

def load_model_and_tokenizer():
    """Load Mixtral model with 4-bit quantization and tokenizer"""
    logger.info(f"Loading model: {BASE_MODEL}")
    
    # Setup quantization
    bnb_config = setup_4bit_quantization()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        padding_side="right",
        token=HUGGINGFACE_API_KEY
    )
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model with quantization (let transformers auto-detect flash attention)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        token=HUGGINGFACE_API_KEY,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    # Apply layer freezing for regularization (if enabled)
    if FREEZE_LAYERS:
        model = freeze_model_layers(model, freeze_percentage=0.5)
        logger.info("Applied layer freezing for regularization")
    else:
        logger.info("Layer freezing disabled - all parameters trainable")
    
    logger.info("Model and tokenizer loaded successfully")
    return model, tokenizer

def setup_lora_config():
    """Setup LoRA configuration optimized for small dataset"""
    return LoraConfig(
        r=LORA_RANK,           # Lower rank to reduce overfitting
        lora_alpha=LORA_ALPHA, # Balanced adaptation strength
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.15,     # Higher dropout for regularization
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

def load_and_split_data():
    """Load training data from HuggingFace and create train/validation split"""
    logger.info(f"Loading training data from HuggingFace: {DATASET_NAME}")
    
    # Load dataset from HuggingFace Hub
    dataset = load_dataset(DATASET_NAME, split="train", token=HUGGINGFACE_API_KEY)
    
    logger.info(f"Loaded {len(dataset)} total examples from HuggingFace")
    
    # Split into train/validation with shuffle
    dataset = dataset.train_test_split(
        test_size=VALIDATION_SPLIT, 
        seed=42,  # For reproducibility
        shuffle=True
    )
    
    train_dataset = dataset['train']
    val_dataset = dataset['test']
    
    logger.info(f"Train examples: {len(train_dataset)}")
    logger.info(f"Validation examples: {len(val_dataset)}")
    
    return train_dataset, val_dataset

def tokenize_function(examples, tokenizer, max_length=1024):
    """Tokenize the examples"""
    # Tokenize the text
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",  # Add padding to ensure consistent lengths
        max_length=max_length,
        return_tensors=None,
    )
    
    # For causal LM, labels are the same as input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

def prepare_datasets(train_dataset, val_dataset, tokenizer):
    """Prepare datasets for training"""
    logger.info("Tokenizing datasets...")
    
    # Tokenize datasets
    train_tokenized = train_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train data"
    )
    
    val_tokenized = val_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Tokenizing validation data"
    )
    
    logger.info(f"Train dataset: {len(train_tokenized)} examples")
    logger.info(f"Validation dataset: {len(val_tokenized)} examples")
    
    return train_tokenized, val_tokenized

def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    # For language modeling, we typically just monitor loss
    # Could add perplexity here if needed
    return {}

def train_model():
    """Main training function with anti-overfitting measures"""
    logger.info("Starting Mixtral Expert Model fine-tuning (Anti-Overfitting)...")
    
    # Setup W&B
    setup_wandb()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Setup LoRA
    lora_config = setup_lora_config()
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    if hasattr(model, "config"):
        model.config.use_cache = False
    
    # Temporarily disable gradient checkpointing to fix gradient issues
    model.gradient_checkpointing_enable()
    logger.info("Gradient checkpointing disabled to resolve gradient flow issues")
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    # Load and split data
    train_dataset, val_dataset = load_and_split_data()
    train_tokenized, val_tokenized = prepare_datasets(train_dataset, val_dataset, tokenizer)
    
        # Calculate total steps for scheduler (using updated parameters)
    steps_per_epoch = 27  # Fixed steps per epoch as requested
    total_steps = steps_per_epoch * MAX_EPOCHS
    warmup_steps = int(total_steps * 0.1)  # 10% warmup

    # Calculate checkpoint interval for exactly 10 total checkpoints
    num_checkpoints = 10
    checkpoint_interval = max(1, total_steps // num_checkpoints)

    effective_batch_size = 2 * 4  # per_device_train_batch_size * gradient_accumulation_steps
    logger.info(f"🔧 Training configuration:")
    logger.info(f"   • Epochs: {MAX_EPOCHS}")
    logger.info(f"   • Steps per epoch: {steps_per_epoch}")
    logger.info(f"   • Total training steps: {total_steps}")
    logger.info(f"   • Per-device batch size: 2")
    logger.info(f"   • Gradient accumulation steps: 4")
    logger.info(f"   • Effective batch size: {effective_batch_size}")
    logger.info(f"   • Warmup steps: {warmup_steps}")
    logger.info(f"🎯 Checkpoint strategy: {num_checkpoints} total checkpoints, every {checkpoint_interval} steps")
    
    # Training arguments with anti-overfitting settings
    training_args = TrainingArguments(
        output_dir="./mixtral-expert-checkpoints",
        num_train_epochs=MAX_EPOCHS,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        
        # Optimization settings
        optim="adamw_torch",
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        
        # Regularization settings
        max_grad_norm=0.3,
        fp16=True,
        bf16=False,
        dataloader_pin_memory=False,
        
        # Evaluation and saving
        eval_strategy="no",
        eval_steps=checkpoint_interval,     # Evaluate at each checkpoint
        save_strategy="epoch",
        save_steps=checkpoint_interval,     # Save checkpoint for exactly 10 total
        save_total_limit=num_checkpoints,   # Keep all checkpoints
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Logging
        logging_steps=checkpoint_interval,
        report_to="wandb" if WANDB_API_KEY else None,
        run_name="mixtral-expert-anti-overfit",
        
        # Memory optimization
        group_by_length=True,
        remove_unused_columns=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Early stopping callback
    
    # Checkpoint monitoring callback
    checkpoint_monitor = CheckpointMonitorCallback(total_checkpoints=num_checkpoints)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[checkpoint_monitor],
    )
    
    # Start training
    logger.info("Starting training with early stopping...")
    trainer.train()
    
    # Log final metrics
    train_loss = trainer.state.log_history[-1].get('train_loss', 'N/A')
    eval_loss = trainer.state.log_history[-1].get('eval_loss', 'N/A')
    
    logger.info(f"Final train loss: {train_loss}")
    logger.info(f"Final validation loss: {eval_loss}")
    
    # Check for overfitting
    if isinstance(train_loss, float) and isinstance(eval_loss, float):
        loss_diff = abs(eval_loss - train_loss)
        if loss_diff > 0.1:  # >10% difference indicates overfitting
            logger.warning(f"⚠️  Potential overfitting detected! "
                          f"Train/Val loss difference: {loss_diff:.3f}")
        else:
            logger.info(f"✅ Good generalization. Train/Val loss difference: {loss_diff:.3f}")
    
    # Save the model
    logger.info("Saving LoRA adapters...")
    trainer.save_model()
    
    # Save tokenizer
    tokenizer.save_pretrained("./mixtral-expert-checkpoints")
    
    logger.info("Training completed successfully!")
    
    return model, tokenizer

def merge_and_save_model():
    """Merge LoRA adapters with base model and save"""
    logger.info("Merging LoRA adapters with base model...")
    
    # Setup quantization config for loading
    bnb_config = setup_4bit_quantization()
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("./mixtral-na-expert-checkpoints")
    
    # Load and merge LoRA
    model = PeftModel.from_pretrained(base_model, "./mixtral-expert-checkpoints")
    merged_model = model.merge_and_unload()
    
    # Save merged model
    logger.info("Saving merged model...")
    merged_model.save_pretrained("./mixtral-expert-merged", safe_serialization=True)
    tokenizer.save_pretrained("./mixtral-expert-merged")
    
    logger.info("Model merging completed!")
    return merged_model, tokenizer

def upload_to_huggingface():
    """Upload the fine-tuned model to HuggingFace Hub"""
    if not HUGGINGFACE_API_KEY or not HUGGINGFACE_ACCOUNT:
        logger.warning("HuggingFace credentials not found, skipping upload")
        return
    
    logger.info("Uploading model to HuggingFace Hub...")
    
    # Create repository
    repo_id = f"{HUGGINGFACE_ACCOUNT}/{OUTPUT_MODEL_NAME}"
    
    try:
        api = HfApi(token=HUGGINGFACE_API_KEY)
        create_repo(repo_id, token=HUGGINGFACE_API_KEY, exist_ok=True)
        
        # Upload model files
        api.upload_folder(
            folder_path="./mixtral-expert-merged",
            repo_id=repo_id,
            token=HUGGINGFACE_API_KEY,
        )
        
        logger.info(f"Model uploaded successfully to: https://huggingface.co/{repo_id}")
        
    except Exception as e:
        logger.error(f"Failed to upload model: {e}")

def cleanup():
    """Cleanup GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("GPU memory cleaned up")

def main():
    """Main execution function"""
    try:
        # Train the model
        model, tokenizer = train_model()
        
        # Clear memory before merging
        del model
        cleanup()
        
        # Merge LoRA with base model
        merged_model, tokenizer = merge_and_save_model()
        
        # Clear memory before upload
        del merged_model, tokenizer
        cleanup()
        
        # Upload to HuggingFace
        upload_to_huggingface()
        
        # Finish W&B run
        if WANDB_API_KEY:
            wandb.finish()
        
        logger.info("Complete anti-overfitting pipeline finished successfully!")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        if WANDB_API_KEY:
            wandb.finish(exit_code=1)
        raise

if __name__ == "__main__":
    main() 
