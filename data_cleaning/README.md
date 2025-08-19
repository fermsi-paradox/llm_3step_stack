# Data Cleaning Module

This module processes raw data files and converts them into structured JSONL format for fine-tuning Llama-style models.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables:
   - Copy `env_template` to `.env`
   - Fill in your API keys and configuration:
     - `HUGGINGFACE_API_KEY`: Your Hugging Face API key
     - `HUGGINGFACE_ACCOUNT`: Your Hugging Face account name
     - `HUGGINGFACE_DATASET_NAME`: Name for your dataset on Hugging Face
     - `OPENAI_API_KEY`: Your OpenAI API key
     - `OPENAI_MODEL`: OpenAI model to use (default: o3)
     - `FINETUNING_TYPE`: Type of fine-tuning (default: QLoRA)

## Usage

1. Place your raw data files in the `raw_data/` directory. Supported formats:
   - PDF files (.pdf)
   - Word documents (.docx, .doc)
   - CSV files (.csv)
   - Text files (.txt, .md)
   - JSON files (.json)

2. Run the processing script:
```bash
python process_raw_data.py
```

3. The script will:
   - Extract text from all files in `raw_data/`
   - Use OpenAI's API to create structured training examples
   - Save the results as `training_data.jsonl`
   - Upload the dataset to your Hugging Face account

## Output Format

The generated JSONL file follows this structure:
```json
{"messages":[
  {"role":"system", "content":"System message defining the assistant's expertise"},
  {"role":"user", "content":"User's question or prompt"},
  {"role":"assistant", "content":"THINK: Assistant's reasoning process"},
  {"role":"assistant", "content":"Assistant's final response"}
]}
```

## Notes

- The script processes files in batches to manage API costs
- Large files are truncated to 4000 characters for API processing
- Progress is logged to the console
- The dataset is uploaded as a private repository on Hugging Face 