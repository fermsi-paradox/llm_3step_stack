#!/usr/bin/env python3
"""
Upload expert training data to HuggingFace Hub as private dataset
"""

import json
import os
from pathlib import Path
from datasets import Dataset
from huggingface_hub import HfApi
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
HUGGINGFACE_ACCOUNT = os.getenv('HUGGINGFACE_ACCOUNT')
DATASET_NAME = os.getenv('DATASET_NAME', 'expert_training_data')
REPO_ID = f"{HUGGINGFACE_ACCOUNT}/{DATASET_NAME}"

def load_jsonl_data(file_path):
    """Load data from JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def create_and_upload_dataset():
    """Create and upload dataset to HuggingFace Hub"""
    
    # Check for required environment variables
    if not HUGGINGFACE_API_KEY or not HUGGINGFACE_ACCOUNT:
        raise ValueError("Please set HUGGINGFACE_API_KEY and HUGGINGFACE_ACCOUNT in your .env file")
    
    # Load training data
    data_file = Path("../data_cleaning/training_data.jsonl")
    if not data_file.exists():
        data_file = Path("training_data.jsonl")
    
    if not data_file.exists():
        raise FileNotFoundError("training_data.jsonl not found")
    
    print(f"Loading data from: {data_file}")
    data = load_jsonl_data(data_file)
    print(f"Loaded {len(data)} examples")
    
    # Convert to HuggingFace Dataset format
    # Each example has a 'text' field with the Mixtral instruction format
    texts = [item['text'] for item in data]
    
    # Create HuggingFace Dataset
    dataset = Dataset.from_dict({
        "text": texts
    })
    
    print(f"Created dataset with {len(dataset)} examples")
    print("Sample example:")
    print(f"  Text: {dataset[0]['text'][:100]}...")
    
    # Initialize HuggingFace API
    api = HfApi(token=HUGGINGFACE_API_KEY)
    
    # Create private repository
    print(f"Creating private dataset repository: {REPO_ID}")
    try:
        api.create_repo(
            repo_id=REPO_ID,
            repo_type="dataset",
            private=True,
            exist_ok=True
        )
        print("✅ Repository created successfully")
    except Exception as e:
        print(f"Repository creation: {e}")
    
    # Upload dataset
    print("Uploading dataset...")
    dataset.push_to_hub(
        repo_id=REPO_ID,
        token=HUGGINGFACE_API_KEY,
        private=True
    )
    
    print(f"✅ Dataset uploaded successfully!")
    print(f"🔗 Dataset URL: https://huggingface.co/datasets/{REPO_ID}")
    print(f"📊 Total examples: {len(dataset)}")
    
    # Create a README for the dataset
    readme_content = f"""# Expert Model Training Data

This dataset contains {len(dataset)} instruction-response pairs for fine-tuning language models on domain-specific expert knowledge.

## Dataset Description

- **Total Examples**: {len(dataset)}
- **Format**: Mixtral instruction format with `[INST]` and `[/INST]` tags
- **Domain**: Configurable expert domain (set via DOMAIN_NAME environment variable)
- **Use Case**: Fine-tuning Mixtral 7Bx8 for domain expertise

## Data Format

Each example contains:
- `text`: A formatted instruction-response pair in Mixtral format

Example:
```
[INST] What are the key considerations for [your domain question]? [/INST] [Expert response based on your training data]...
```

## Source

Generated from domain-specific documentation, videos, webinars, and technical resources focused on your area of expertise.

## License

Private dataset for fine-tuning specialized expert models.
"""
    
    try:
        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type="dataset",
            token=HUGGINGFACE_API_KEY
        )
        print("✅ README uploaded")
    except Exception as e:
        print(f"README upload failed: {e}")

if __name__ == "__main__":
    create_and_upload_dataset() 