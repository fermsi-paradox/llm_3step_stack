import os
import json
import glob
from pathlib import Path
from typing import List, Dict, Any
import openai
from dotenv import load_dotenv
from huggingface_hub import HfApi, upload_file
import pandas as pd
import PyPDF2
import docx
import mimetypes
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration from environment
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'o3')
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
HUGGINGFACE_ACCOUNT = os.getenv('HUGGINGFACE_ACCOUNT')
HUGGINGFACE_DATASET_NAME = os.getenv('HUGGINGFACE_DATASET_NAME')
DOMAIN_NAME = os.getenv('DOMAIN_NAME', 'your-domain')
EXPERT_ROLE = os.getenv('EXPERT_ROLE', 'domain expert')

# Initialize OpenAI client
openai.api_key = OPENAI_API_KEY

# Initialize Hugging Face API
hf_api = HfApi(token=HUGGINGFACE_API_KEY)

class RawDataProcessor:
    def __init__(self, raw_data_path: str = "raw_data"):
        self.raw_data_path = Path(raw_data_path)
        self.processed_data = []
        
    def extract_text_from_pdf(self, file_path: Path) -> str:
        """Extract text from PDF files."""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            logger.error(f"Error reading PDF {file_path}: {e}")
            return ""
    
    def extract_text_from_docx(self, file_path: Path) -> str:
        """Extract text from DOCX files."""
        try:
            doc = docx.Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except Exception as e:
            logger.error(f"Error reading DOCX {file_path}: {e}")
            return ""
    
    def extract_text_from_csv(self, file_path: Path) -> str:
        """Extract text from CSV files."""
        try:
            df = pd.read_csv(file_path)
            return df.to_string()
        except Exception as e:
            logger.error(f"Error reading CSV {file_path}: {e}")
            return ""
    
    def extract_text_from_file(self, file_path: Path) -> str:
        """Extract text from various file types."""
        mime_type, _ = mimetypes.guess_type(str(file_path))
        
        if file_path.suffix.lower() == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif file_path.suffix.lower() in ['.docx', '.doc']:
            return self.extract_text_from_docx(file_path)
        elif file_path.suffix.lower() == '.csv':
            return self.extract_text_from_csv(file_path)
        elif file_path.suffix.lower() in ['.txt', '.md', '.json']:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Error reading text file {file_path}: {e}")
                return ""
        else:
            logger.warning(f"Unsupported file type: {file_path}")
            return ""
    
    def process_with_openai(self, content: str, file_name: str) -> List[Dict[str, Any]]:
        """Process content with OpenAI to create structured training data."""
        system_prompt = """# SYSTEM PROMPT PLACEHOLDER FOR DATA PROCESSING

Replace this placeholder with your custom system prompt for data processing that defines:
- The AI assistant's role in creating training data
- Guidelines for extracting question-answer pairs
- Format requirements for training examples
- Quality standards for generated content
- Specific domain focus areas

This system prompt should guide the AI in processing raw data into structured training examples."""
        
        user_prompt = f"""# USER PROMPT PLACEHOLDER FOR DATA PROCESSING

Replace this placeholder with your custom user prompt that instructs the AI on:
- How to process content from file '{file_name}'
- What type of training examples to create
- The specific format and structure required
- Domain expertise areas to focus on
- Technical detail requirements
- Quality standards for responses

The user prompt should provide clear instructions for converting raw content into structured training data.

Content to process:
{content[:8000]}

[Additional processing instructions would go here]"""

        try:
            response = openai.ChatCompletion.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=32000
            )
            
            # Parse the response with robust JSON handling
            response_content = response.choices[0].message.content
            
            # Try to extract JSON array from response
            try:
                examples = json.loads(response_content)
            except json.JSONDecodeError:
                # Try to find JSON array in the response
                import re
                json_match = re.search(r'\[.*\]', response_content, re.DOTALL)
                if json_match:
                    try:
                        examples = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        logger.error(f"Could not parse JSON from response. Response preview: {response_content[:200]}...")
                        return []
                else:
                    logger.error(f"No JSON array found in response. Response preview: {response_content[:200]}...")
                    return []
            
            # Format each example into the required structure
            formatted_examples = []
            for example in examples:
                messages = []
                
                # Add system message
                if 'system' in example:
                    messages.append({"role": "system", "content": example['system']})
                
                # Add user message
                if 'user' in example:
                    messages.append({"role": "user", "content": example['user']})
                
                # Add assistant thinking (if present)
                if 'thinking' in example:
                    messages.append({"role": "assistant", "content": f"THINK: {example['thinking']}"})
                
                # Add assistant response
                if 'assistant' in example:
                    messages.append({"role": "assistant", "content": example['assistant']})
                
                if messages:
                    formatted_examples.append({"messages": messages})
            
            return formatted_examples
            
        except Exception as e:
            logger.error(f"Error processing with OpenAI: {e}")
            return []
    
    def process_all_files(self):
        """Process all files in the raw_data directory."""
        all_files = list(self.raw_data_path.glob("**/*"))
        file_count = sum(1 for f in all_files if f.is_file())
        
        logger.info(f"Found {file_count} files to process")
        
        for file_path in tqdm(all_files, desc="Processing files"):
            if file_path.is_file():
                logger.info(f"Processing: {file_path.name}")
                
                # Extract text content
                content = self.extract_text_from_file(file_path)
                
                if content:
                    # Process with OpenAI
                    examples = self.process_with_openai(content, file_path.name)
                    self.processed_data.extend(examples)
                    logger.info(f"Generated {len(examples)} training examples from {file_path.name}")
    
    def save_jsonl(self, output_path: str = "training_data.jsonl"):
        """Save processed data as JSONL file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in self.processed_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(self.processed_data)} training examples to {output_path}")
        return output_path
    
    def upload_to_huggingface(self, file_path: str):
        """Upload the JSONL file to Hugging Face."""
        try:
            repo_id = f"{HUGGINGFACE_ACCOUNT}/{HUGGINGFACE_DATASET_NAME}"
            
            # Create repository if it doesn't exist
            try:
                hf_api.create_repo(
                    repo_id=repo_id,
                    repo_type="dataset",
                    private=True
                )
                logger.info(f"Created new dataset repository: {repo_id}")
            except Exception as e:
                logger.info(f"Repository might already exist: {e}")
            
            # Upload the file
            upload_file(
                path_or_fileobj=file_path,
                path_in_repo="train.jsonl",
                repo_id=repo_id,
                repo_type="dataset",
                token=HUGGINGFACE_API_KEY
            )
            
            logger.info(f"Successfully uploaded training data to Hugging Face: {repo_id}")
            
        except Exception as e:
            logger.error(f"Error uploading to Hugging Face: {e}")

def main():
    """Main function to process raw data and create training dataset."""
    processor = RawDataProcessor()
    
    # Process all files in raw_data directory
    processor.process_all_files()
    
    if processor.processed_data:
        # Save as JSONL
        output_file = processor.save_jsonl()
        
        # Upload to Hugging Face
        processor.upload_to_huggingface(output_file)
    else:
        logger.warning("No data was processed. Please add files to the raw_data directory.")

if __name__ == "__main__":
    main() 