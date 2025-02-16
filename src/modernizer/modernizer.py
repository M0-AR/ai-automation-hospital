import os
from docx import Document
import requests
import json
from pathlib import Path
import time
import html
from typing import Optional
import random
import datetime
import shutil
import re

class RequirementsModernizer:
    def __init__(self, api_token: str, input_file: str, output_dir: str):
        """Initialize the modernizer with API key and file paths."""
        self.api_token = api_token
        self.input_file = input_file
        self.output_dir = output_dir
        self.temp_dir = "temp_progress"
        self.headers = {"Authorization": f"Bearer {api_token}"}
        self.sections_completed = set()
        
        # Create temp directory if it doesn't exist
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
            
        # Load previous progress if exists
        self.progress_file = os.path.join(self.temp_dir, "modernization_progress.json")
        self.load_progress()

    def load_progress(self):
        """Load previously saved progress if it exists."""
        try:
            if os.path.exists(self.progress_file):
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    saved_data = json.load(f)
                    self.sections_completed = set(saved_data.get('completed_sections', []))
                    print(f"📂 Loaded progress: {len(self.sections_completed)} sections completed")
        except Exception as e:
            print(f"⚠️ Warning: Could not load progress: {str(e)}")
            self.sections_completed = set()

    def save_progress(self, section_name: str, section_content: str):
        """Save the progress of a completed section."""
        try:
            # Save section content
            section_file = os.path.join(self.temp_dir, f"{section_name}.txt")
            with open(section_file, 'w', encoding='utf-8') as f:
                f.write(section_content)
            
            # Update progress tracking
            self.sections_completed.add(section_name)
            
            # Save progress metadata
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'completed_sections': list(self.sections_completed),
                    'last_update': datetime.datetime.now().isoformat()
                }, f, indent=2)
                
            print(f"💾 Progress saved for section: {section_name}")
        except Exception as e:
            print(f"⚠️ Warning: Could not save progress for {section_name}: {str(e)}")

    def combine_saved_sections(self) -> str:
        """Combine all saved sections into final document."""
        combined_content = []
        for section_name in sorted(self.sections_completed):
            try:
                section_file = os.path.join(self.temp_dir, f"{section_name}.txt")
                if os.path.exists(section_file):
                    with open(section_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        combined_content.append(content)
            except Exception as e:
                print(f"⚠️ Warning: Could not read section {section_name}: {str(e)}")
        
        return "\n\n".join(combined_content)

    def modernize_section(self, section_name: str, content: str) -> str:
        """Modernize a single section of the requirements."""
        if section_name in self.sections_completed:
            print(f"⏩ Skipping already completed section: {section_name}")
            section_file = os.path.join(self.temp_dir, f"{section_name}.txt")
            with open(section_file, 'r', encoding='utf-8') as f:
                return f.read()

        print(f"🔄 Modernizing section: {section_name}")
        
        # Using a combination of models for better results
        models = [
            "google/flan-t5-large",  # Good for text generation
            "facebook/bart-large-cnn",  # Good for restructuring
            "google/pegasus-pubmed"  # Medical domain expertise
        ]
        
        modernized_content = content
        for model in models:
            try:
                print(f"Using model: {model}")
                api_url = f"https://api-inference.huggingface.co/models/{model}"
                
                prompt = f"""Modernize this hospital system requirement section. Make it more precise and aligned with current healthcare technology standards.

Section: {section_name}
Content: {modernized_content}

Focus on:
1. Digital health records
2. Patient data security
3. Modern healthcare practices
4. System integration
5. User experience
"""
                
                payload = {
                    "inputs": prompt,
                    "parameters": {
                        "max_length": 800,
                        "min_length": 100,
                        "temperature": 0.7,
                        "top_p": 0.95,
                        "do_sample": True,
                        "num_return_sequences": 1
                    }
                }

                response = requests.post(
                    api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        if 'generated_text' in result[0]:
                            modernized_content = result[0]['generated_text']
                        else:
                            modernized_content = result[0]
                    elif isinstance(result, dict):
                        modernized_content = result.get('generated_text', modernized_content)
                    
                    print(f"✅ Successfully processed with {model}")
                    break  # Stop if we get a good result
                else:
                    print(f"⚠️ Model {model} returned status code {response.status_code}")
                    continue  # Try next model
                    
            except Exception as e:
                print(f"⚠️ Error with model {model}: {str(e)}")
                continue  # Try next model
        
        # Save progress regardless of which model succeeded
        self.save_progress(section_name, modernized_content)
        return modernized_content

    def process_section(self, section: dict) -> dict:
        """Process a single section with translation and modernization."""
        try:
            print(f"\n{'#'*80}")
            print(f"PROCESSING SECTION: {section['section']}")
            print(f"{'#'*80}\n")
            
            # Step 1: Translate Arabic to English
            print("Step 1: Translating section to English...")
            english_section = self.translate_text(section['section'], True)
            if not english_section:
                raise Exception("Failed to translate section title to English")
            
            print("\nStep 2: Translating content to English...")
            english_content = self.translate_text(section['content'], True)
            if not english_content:
                raise Exception("Failed to translate content to English")
            
            # Step 2: Modernize the English content
            print("\nStep 3: Modernizing content...")
            combined_english = f"{english_section}\n\n{english_content}"
            modernized_content = self.modernize_section(f"{section['section']}_english", combined_english)
            if not modernized_content:
                raise Exception("Failed to modernize content")
            
            # Step 3: Translate back to Arabic
            print("\nStep 4: Translating back to Arabic...")
            arabic_content = self.translate_text(modernized_content, False)
            if not arabic_content:
                raise Exception("Failed to translate modernized content to Arabic")
            
            print(f"\n{'#'*80}")
            print("SECTION PROCESSING COMPLETE")
            print(f"{'#'*80}\n")
            
            return {
                'original_section': section['section'],
                'modernized_content': arabic_content,
                'english_version': modernized_content
            }
        
        except Exception as e:
            error_msg = str(e)
            print(f"\n❌ Error processing section {section['section']}: {error_msg}")
            return {
                'original_section': section['section'],
                'modernized_content': f"Error: {error_msg}\n\nOriginal content:\n{section['content']}",
                'english_version': f"Error: {error_msg}"
            }

    def save_markdown(self, modernized_sections: list, output_dir: str):
        """Save modernized content as Markdown files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create main index file
        with open(output_path / 'README.md', 'w', encoding='utf-8') as f:
            f.write("# Damascus General Hospital Automation System 2025\n\n")
            f.write("## Modernized Requirements\n\n")
            
            for idx, section in enumerate(modernized_sections, 1):
                section_file = f'section_{idx:02d}.md'
                f.write(f"{idx}. [{section['original_section']}]({section_file})\n")
                
                # Create individual section file
                with open(output_path / section_file, 'w', encoding='utf-8') as sf:
                    sf.write(f"# {section['original_section']}\n\n")
                    sf.write("## النسخة العربية (Arabic Version)\n\n")
                    sf.write(section['modernized_content'])
                    sf.write("\n\n## English Version\n\n")
                    sf.write(section['english_version'])

    def process_document(self, input_file: str, output_dir: str):
        """Process the entire document."""
        print("Starting document modernization process...")
        
        # Read document
        doc = Document(input_file)
        sections = []
        current_section = ""
        section_content = []
        
        # Extract sections
        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if not text:
                continue
            
            if any(char.isdigit() for char in text[:2]) and ')' in text[:10]:
                if current_section and section_content:
                    sections.append({
                        'section': current_section,
                        'content': '\n'.join(section_content)
                    })
                current_section = text
                section_content = []
            else:
                section_content.append(text)
        
        # Add last section
        if current_section and section_content:
            sections.append({
                'section': current_section,
                'content': '\n'.join(section_content)
            })
        
        print(f"Found {len(sections)} sections to modernize")
        
        # Process sections
        modernized_sections = []
        for idx, section in enumerate(sections, 1):
            print(f"\nProcessing section {idx}/{len(sections)}: {section['section']}")
            modernized = self.process_section(section)
            modernized_sections.append(modernized)
            time.sleep(2)  # Rate limiting
        
        # Save results
        self.save_markdown(modernized_sections, output_dir)
        print(f"\nModernization complete! Results saved to {output_dir}")

    def api_call_with_retry(self, api_url: str, payload: dict, operation: str) -> Optional[dict]:
        """Make API call with retry logic with exponential backoff and rate limit handling."""
        last_error = None
        base_delay = 30  # Start with 30 seconds for rate limits
        max_retries = 5
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    # Exponential backoff with longer delays for rate limits
                    delay = min(base_delay * (2 ** attempt) + random.uniform(1, 5), 300)  # Max 5 minutes
                    print(f"⏳ Waiting {delay:.1f} seconds before attempt {attempt + 1}/{max_retries}...")
                    time.sleep(delay)
                
                response = requests.post(
                    api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=30  # Shorter timeout to fail fast
                )
                
                if response.status_code == 429:  # Rate limit hit
                    print(f"⚠️ Rate limit reached. Implementing longer delay...")
                    if attempt < max_retries - 1:
                        continue
                    
                elif response.status_code == 503:  # Model loading
                    estimated_time = response.json().get("estimated_time", 20)
                    print(f"⏳ Model is loading. Waiting {estimated_time} seconds...")
                    time.sleep(min(estimated_time, 30))  # Don't wait more than 30 seconds
                    if attempt < max_retries - 1:
                        continue
                        
                elif response.status_code == 200:
                    return response.json()
                    
                response.raise_for_status()
                
            except requests.exceptions.Timeout:
                print(f"⏳ Request timed out. Will retry with longer delay...")
                continue
                
            except requests.exceptions.RequestException as e:
                last_error = str(e)
                print(f"⚠️ Request failed: {last_error}")
                if "rate limit" in last_error.lower():
                    if attempt < max_retries - 1:
                        continue
                
            except Exception as e:
                last_error = str(e)
                print(f"❌ Error: {last_error}")
                if attempt < max_retries - 1:
                    continue
                break
        
        print(f"❌ All {max_retries} retry attempts failed: {last_error}")
        return None

    def translate_text(self, text: str, is_to_english: bool = True) -> Optional[str]:
        """Translate text between Arabic and English with improved rate limit handling."""
        if not text or not text.strip():
            return text
            
        # Use simpler models for translation to reduce rate limits
        models = {
            'ar-en': [
                'Helsinki-NLP/opus-mt-ar-en',  # Primary choice
                'facebook/nllb-200-distilled-600M',  # Backup choice
                'facebook/m2m100_418M'  # Last resort
            ],
            'en-ar': [
                'Helsinki-NLP/opus-mt-en-ar',  # Primary choice
                'facebook/nllb-200-distilled-600M',  # Backup choice
                'facebook/m2m100_418M'  # Last resort
            ]
        }
        
        direction = "Arabic → English" if is_to_english else "English → Arabic"
        print(f"\n{'='*80}")
        print(f"TRANSLATION ({direction})")
        print(f"Input text: {text[:100]}{'...' if len(text) > 100 else ''}")
        print(f"{'='*80}\n")
        
        # Try each model in sequence
        model_list = models['ar-en'] if is_to_english else models['en-ar']
        for model in model_list:
            api_url = f"https://api-inference.huggingface.co/models/{model}"
            
            # Split into smaller chunks to reduce load
            max_chunk_size = 200  # Reduced chunk size
            chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]
            translated_chunks = []
            
            success = True
            for i, chunk in enumerate(chunks):
                print(f"\nProcessing chunk {i + 1}/{len(chunks)} with {model}:")
                print(f"Content: {chunk[:50]}{'...' if len(chunk) > 50 else ''}")
                
                payload = {"inputs": chunk}
                if 'nllb' in model:
                    payload["parameters"] = {
                        "src_lang": "ar" if is_to_english else "en",
                        "tgt_lang": "en" if is_to_english else "ar"
                    }
                elif 'm2m100' in model:
                    payload["parameters"] = {
                        "src_lang": "ar" if is_to_english else "en",
                        "tgt_lang": "en" if is_to_english else "ar"
                    }
                
                result = self.api_call_with_retry(
                    api_url, 
                    payload,
                    f"translation chunk {i + 1}/{len(chunks)} with {model}"
                )
                
                if not result:
                    success = False
                    break
                    
                if isinstance(result, list) and len(result) > 0:
                    translated_text = result[0].get('translation_text', '')
                    if translated_text:
                        print(f"✓ Translated: {translated_text[:50]}{'...' if len(translated_text) > 50 else ''}")
                        translated_chunks.append(translated_text)
                        # Add delay between chunks to avoid rate limits
                        if i < len(chunks) - 1:
                            time.sleep(5)  # 5 second delay between chunks
                    else:
                        success = False
                        break
                else:
                    success = False
                    break
            
            if success:
                final_translation = ' '.join(translated_chunks)
                print(f"\n{'='*80}")
                print(f"FINAL TRANSLATION with {model}:")
                print(f"{final_translation[:100]}{'...' if len(final_translation) > 100 else ''}")
                print(f"{'='*80}\n")
                return final_translation
            
            print(f"⚠️ Failed with {model}, trying next model...")
            time.sleep(10)  # Wait before trying next model
        
        print("❌ All translation models failed")
        return None

def load_env_file(env_path='.env'):
    """Load environment variables from file."""
    env_vars = {}
    try:
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
        return env_vars
    except Exception as e:
        print(f"Error reading .env file: {str(e)}")
        return {}

def main():
    # Load API token from .env file
    env_vars = load_env_file()
    api_token = env_vars.get('HF_API_TOKEN')
    
    if not api_token:
        print("Please add your Hugging Face API token to the .env file")
        print("Format: HF_API_TOKEN=your_token_here")
        return

    input_file = "hospital_automation_requirements.docx"
    output_dir = "modernized_requirements_2025"
    
    modernizer = RequirementsModernizer(api_token, input_file, output_dir)
    modernizer.process_document(input_file, output_dir)

if __name__ == "__main__":
    main()
