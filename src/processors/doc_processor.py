import json
import os
import time
import random
from typing import Dict, List, Optional, Union
import requests
from tqdm import tqdm
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('document_processing.log'),
        logging.StreamHandler()
    ]
)

class DocumentProcessor:
    def __init__(self, api_token: str, chunk_size: int = 200):
        """Initialize the document processor with API token and settings."""
        self.api_token = api_token
        self.chunk_size = chunk_size
        self.headers = {"Authorization": f"Bearer {api_token}"}
        self.temp_dir = "temp_processing"
        self.progress_file = os.path.join(self.temp_dir, "processing_progress.json")
        
        # Translation models (from smallest to largest)
        self.translation_models = {
            'ar-en': [
                'Helsinki-NLP/opus-mt-ar-en',      # Fast, good quality
                'facebook/nllb-200-distilled-600M', # Better but slower
                'facebook/m2m100_418M'             # Fallback option
            ],
            'en-ar': [
                'Helsinki-NLP/opus-mt-en-ar',      # Fast, good quality
                'facebook/nllb-200-distilled-600M', # Better but slower
                'facebook/m2m100_418M'             # Fallback option
            ]
        }
        
        # Create temp directory if it doesn't exist
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
            
        self.load_progress()

    def load_progress(self):
        """Load saved progress from file."""
        try:
            if os.path.exists(self.progress_file):
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    self.progress = json.load(f)
                    logging.info(f"Loaded progress: {len(self.progress.get('completed_sections', []))} sections completed")
            else:
                self.progress = {'completed_sections': [], 'last_update': None}
        except Exception as e:
            logging.warning(f"Could not load progress: {str(e)}")
            self.progress = {'completed_sections': [], 'last_update': None}

    def save_progress(self, section_id: str, content: Dict):
        """Save progress for a section."""
        try:
            # Save section content
            section_file = os.path.join(self.temp_dir, f"section_{section_id}.json")
            with open(section_file, 'w', encoding='utf-8') as f:
                json.dump(content, f, ensure_ascii=False, indent=2)
            
            # Update progress tracking
            if section_id not in self.progress['completed_sections']:
                self.progress['completed_sections'].append(section_id)
            self.progress['last_update'] = datetime.now().isoformat()
            
            # Save progress file
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(self.progress, f, indent=2)
                
            logging.info(f"Progress saved for section: {section_id}")
        except Exception as e:
            logging.error(f"Could not save progress for {section_id}: {str(e)}")

    def api_call_with_retry(self, api_url: str, payload: dict, operation: str) -> Optional[dict]:
        """Make API call with smart retry logic."""
        max_retries = 5
        base_delay = 30  # Start with 30 seconds for rate limits
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    delay = min(base_delay * (2 ** attempt) + random.uniform(1, 5), 300)
                    logging.info(f"Waiting {delay:.1f}s before attempt {attempt + 1}/{max_retries}...")
                    time.sleep(delay)
                
                response = requests.post(
                    api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 429:  # Rate limit
                    logging.warning("Rate limit reached, implementing longer delay...")
                    continue
                    
                elif response.status_code == 503:  # Model loading
                    wait_time = min(response.json().get("estimated_time", 20), 30)
                    logging.info(f"Model loading, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                    
                elif response.status_code == 200:
                    return response.json()
                
                response.raise_for_status()
                
            except requests.exceptions.Timeout:
                logging.warning("Request timed out, retrying...")
                continue
            except Exception as e:
                logging.error(f"Error during {operation}: {str(e)}")
                if attempt < max_retries - 1:
                    continue
                break
        
        logging.error(f"All {max_retries} retry attempts failed for {operation}")
        return None

    def translate_text(self, text: str, is_to_english: bool = True) -> Optional[str]:
        """Translate text between Arabic and English with fallback models."""
        if not text or not text.strip():
            return text
            
        direction = "Arabic → English" if is_to_english else "English → Arabic"
        logging.info(f"Translating ({direction}): {text[:100]}...")
        
        # Try each model in sequence
        model_list = self.translation_models['ar-en'] if is_to_english else self.translation_models['en-ar']
        
        for model in model_list:
            logging.info(f"Trying model: {model}")
            api_url = f"https://api-inference.huggingface.co/models/{model}"
            
            # Split into smaller chunks
            chunks = [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]
            translated_chunks = []
            
            success = True
            for i, chunk in enumerate(chunks, 1):
                logging.info(f"Processing chunk {i}/{len(chunks)}")
                
                payload = {"inputs": chunk}
                if 'nllb' in model or 'm2m100' in model:
                    payload["parameters"] = {
                        "src_lang": "ar" if is_to_english else "en",
                        "tgt_lang": "en" if is_to_english else "ar"
                    }
                
                result = self.api_call_with_retry(
                    api_url, 
                    payload,
                    f"translation chunk {i}/{len(chunks)} with {model}"
                )
                
                if not result:
                    success = False
                    break
                    
                if isinstance(result, list) and result:
                    translated_text = result[0].get('translation_text', '')
                    if translated_text:
                        translated_chunks.append(translated_text)
                        if i < len(chunks):
                            time.sleep(5)  # Delay between chunks
                    else:
                        success = False
                        break
                else:
                    success = False
                    break
            
            if success:
                translation = ' '.join(translated_chunks)
                logging.info(f"Translation successful with {model}")
                return translation
            
            logging.warning(f"Failed with {model}, trying next model...")
            time.sleep(10)  # Wait before trying next model
        
        logging.error("All translation models failed")
        return None

    def process_json_document(self, input_file: str, output_file: str):
        """Process a JSON document containing Arabic and English content."""
        try:
            # Load input JSON
            with open(input_file, 'r', encoding='utf-8') as f:
                doc_content = json.load(f)
            
            total_sections = len(doc_content)
            logging.info(f"Processing {total_sections} sections from {input_file}")
            
            processed_content = []
            for i, section in enumerate(doc_content, 1):
                section_id = str(section.get('id', i))
                
                # Check if section was already processed
                if section_id in self.progress['completed_sections']:
                    section_file = os.path.join(self.temp_dir, f"section_{section_id}.json")
                    if os.path.exists(section_file):
                        with open(section_file, 'r', encoding='utf-8') as f:
                            processed_section = json.load(f)
                            processed_content.append(processed_section)
                            logging.info(f"Loaded cached section {section_id}")
                            continue
                
                logging.info(f"Processing section {i}/{total_sections}: {section.get('title', 'Untitled')}")
                
                processed_section = {
                    'id': section_id,
                    'title': section.get('title', ''),
                    'title_en': None,
                    'content': section.get('content', ''),
                    'content_en': None
                }
                
                # Translate title if it contains Arabic
                if any(ord(c) > 127 for c in processed_section['title']):
                    processed_section['title_en'] = self.translate_text(
                        processed_section['title'],
                        is_to_english=True
                    )
                
                # Translate content if it contains Arabic
                if any(ord(c) > 127 for c in processed_section['content']):
                    processed_section['content_en'] = self.translate_text(
                        processed_section['content'],
                        is_to_english=True
                    )
                
                processed_content.append(processed_section)
                self.save_progress(section_id, processed_section)
                
                # Save intermediate results
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(processed_content, f, ensure_ascii=False, indent=2)
                logging.info(f"Saved intermediate results ({i}/{total_sections})")
            
            logging.info(f"Processing complete! Results saved to {output_file}")
            
        except Exception as e:
            logging.error(f"Error processing document: {str(e)}")
            raise

def main():
    # Load API token from environment or .env file
    api_token = os.getenv('HUGGINGFACE_API_TOKEN')
    if not api_token:
        try:
            with open('.env', 'r') as f:
                for line in f:
                    if line.startswith('HUGGINGFACE_API_TOKEN='):
                        api_token = line.strip().split('=')[1]
                        break
        except:
            raise ValueError("HUGGINGFACE_API_TOKEN not found in environment or .env file")
    
    processor = DocumentProcessor(api_token)
    
    input_file = "doc_content.json"
    output_file = "doc_content_processed.json"
    
    processor.process_json_document(input_file, output_file)

if __name__ == "__main__":
    main()
