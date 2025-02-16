import os
import json
import time
import requests
import html
from typing import Optional
import random
from datetime import datetime
import docx

class RequirementsModernizer:
    def __init__(self, api_token):
        self.api_token = api_token
        
        # Using Google's MT5 model which is good for Arabic
        self.ar_to_en_model = "google/mt5-small"
        self.en_to_ar_model = "google/mt5-small"
        self.modernization_model = "gpt2"
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }
        self.max_retries = 3  # Reduced retries but with longer delays
        self.retry_delay = 60  # Start with 60 second delay
        self.max_delay = 180   # Maximum delay between retries

    def api_call_with_retry(self, api_url: str, payload: dict, operation: str) -> Optional[dict]:
        """Make API call with retry logic with exponential backoff."""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                if attempt > 0:
                    # Fixed delay with small jitter for rate limits
                    delay = self.retry_delay + random.uniform(1, 5)
                    print(f"Rate limit hit. Waiting {delay:.1f} seconds before attempt {attempt + 1}/{self.max_retries} for {operation}...")
                    time.sleep(delay)
                
                response = requests.post(
                    api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=90  # Increased timeout
                )
                
                if response.status_code == 503:
                    print(f"Model is still loading for {operation}...")
                    continue
                elif response.status_code == 429:
                    print(f"Rate limit hit for {operation}. Will retry after delay.")
                    continue
                    
                response.raise_for_status()
                return response.json()
            
            except requests.exceptions.Timeout:
                print(f"Timeout occurred for {operation}. Will retry...")
                continue
            except requests.exceptions.RequestException as e:
                last_error = str(e)
                print(f"Request failed for {operation}: {last_error}")
                if "rate limit" in last_error.lower():
                    continue  # Will retry with delay
            except Exception as e:
                last_error = str(e)
                print(f"Unexpected error for {operation}: {last_error}")
        
        print(f"All {self.max_retries} retry attempts failed for {operation}: {last_error}")
        return None

    def translate_text(self, text: str, is_to_english: bool = True) -> Optional[str]:
        """Translate text between Arabic and English with improved error handling."""
        if not text or not text.strip():
            return text
            
        direction = "Arabic → English" if is_to_english else "English → Arabic"
        print(f"\n{'='*80}")
        print(f"TRANSLATION ({direction})")
        print(f"Input text: {text[:100]}{'...' if len(text) > 100 else ''}")
        print(f"{'='*80}\n")
            
        model = self.ar_to_en_model if is_to_english else self.en_to_ar_model
        api_url = f"https://api-inference.huggingface.co/models/{model}"
        
        # For MT5, prepare the input in the correct format
        prefix = "translate Arabic to English: " if is_to_english else "translate English to Arabic: "
        input_text = prefix + text
        
        payload = {
            "inputs": input_text,
            "parameters": {
                "max_length": 512,
                "num_return_sequences": 1
            }
        }
        
        result = self.api_call_with_retry(api_url, payload, f"translation ({'ar->en' if is_to_english else 'en->ar'})")
        
        if not result:
            print(f"❌ Failed to translate text")
            return None
            
        if isinstance(result, list) and len(result) > 0:
            translated_text = result[0].get('generated_text', '')
            if translated_text:
                print(f"✓ Successfully translated: {translated_text[:100]}{'...' if len(translated_text) > 100 else ''}")
                return translated_text
            else:
                print(f"❌ Empty translation result")
                return None
        else:
            print(f"❌ Unexpected response format")
            return None

    def modernize_content(self, content: str) -> Optional[str]:
        """Modernize content using GPT-2 with improved prompting."""
        if not content or not content.strip():
            return content
            
        print(f"\n{'='*80}")
        print("MODERNIZATION")
        print(f"Input content: {content[:200]}{'...' if len(content) > 200 else ''}")
        print(f"{'='*80}\n")
            
        api_url = f"https://api-inference.huggingface.co/models/{self.modernization_model}"
        
        prompt = f"""Modernize this hospital system requirement for 2025. Focus on:
1. Cloud infrastructure and microservices
2. AI/ML in healthcare
3. IoT and remote monitoring
4. Cybersecurity and compliance
5. Telemedicine integration

Original text:
{content}

Modernized version:"""

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_length": 800,
                "temperature": 0.7,
                "top_p": 0.9,
                "return_full_text": False
            }
        }
        
        result = self.api_call_with_retry(api_url, payload, "content modernization")
        if result and isinstance(result, list) and len(result) > 0:
            modernized = result[0].get('generated_text', '').strip()
            print(f"\n{'='*80}")
            print("MODERNIZED CONTENT:")
            print(f"{modernized[:200]}{'...' if len(modernized) > 200 else ''}")
            print(f"{'='*80}\n")
            return modernized
        return None

    def process_section(self, section: dict) -> dict:
        """Process a single section with translation and modernization."""
        try:
            # Extract section data
            section_title = section.get('title', '')
            content = section.get('content', '')
            
            # Step 1: Translate section title if it's in Arabic
            print("\nStep 1: Translating section title...")
            english_section = section_title
            if any(ord(c) > 127 for c in section_title):
                english_section = self.translate_text(section_title, is_to_english=True)
                if not english_section:
                    raise Exception("Failed to translate section title")
            
            # Step 2: Translate content if it's in Arabic
            print("\nStep 2: Translating content...")
            english_content = content
            if any(ord(c) > 127 for c in content):
                english_content = self.translate_text(content, is_to_english=True)
                if not english_content:
                    raise Exception("Failed to translate content")
            
            # Step 2: Modernize the English content
            print("\nStep 3: Modernizing content...")
            combined_english = f"{english_section}\n\n{english_content}"
            modernized_content = self.modernize_content(combined_english)
            if not modernized_content:
                raise Exception("Failed to modernize content")
            
            # Step 3: Translate back to Arabic if original was in Arabic
            print("\nStep 4: Translating back to Arabic...")
            arabic_content = None
            if any(ord(c) > 127 for c in content):
                arabic_content = self.translate_text(modernized_content, is_to_english=False)
                if not arabic_content:
                    raise Exception("Failed to translate back to Arabic")
            
            return {
                'original_section': section_title,
                'english_section': english_section if english_section != section_title else None,
                'original_content': content,
                'english_content': english_content if english_content != content else None,
                'modernized_content': modernized_content,
                'arabic_content': arabic_content,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error processing section: {str(e)}")
            return {
                'original_section': section.get('title', ''),
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def save_markdown(self, sections: list, output_dir: str):
        """Save processed sections as markdown files."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Create README.md with index
        readme_content = [
            "# Damascus General Hospital Automation System 2025",
            "\n## Modernized Requirements\n"
        ]
        
        # Process each section
        for i, section in enumerate(sections, 1):
            # Add to README
            title = section['original_section']
            filename = f"section_{i:02d}.md"
            readme_content.append(f"{i}. [{title}]({filename})")
            
            # Create section file
            section_content = []
            
            # Add original section
            section_content.extend([
                f"# {title}",
                "\n## النسخة العربية (Arabic Version)",
                section['original_content']
            ])
            
            # Add English version if available
            if section.get('english_content'):
                section_content.extend([
                    "\n## English Version",
                    section['english_content']
                ])
            
            # Add modernized version
            if section.get('modernized_content'):
                section_content.extend([
                    "\n## Modernized Version (2025)",
                    section['modernized_content']
                ])
            
            # Add Arabic translation if available
            if section.get('arabic_content'):
                section_content.extend([
                    "\n## النسخة العربية المحدثة (Updated Arabic Version)",
                    section['arabic_content']
                ])
            
            # Write section file
            with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
                f.write('\n\n'.join(section_content))
        
        # Write README
        with open(os.path.join(output_dir, 'README.md'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(readme_content))

    def process_document(self, input_file: str, output_dir: str):
        """Process an entire document."""
        print("\nStarting document modernization process...")
        
        try:
            # Read input file
            doc = docx.Document(input_file)
            sections = []
            current_section = None
            
            for para in doc.paragraphs:
                text = para.text.strip()
                if not text:
                    continue
                
                # Check if it's a heading by looking for numbered sections like "1-1)" or Arabic text
                is_heading = (
                    para.style.name.startswith('Heading') or
                    any(c.isdigit() for c in text[:4]) or  # Check first few chars for numbers
                    any(ord(c) > 127 for c in text[:10])   # Check first few chars for Arabic
                )
                
                if is_heading:
                    if current_section and current_section['content'].strip():
                        sections.append(current_section)
                    current_section = {'title': text, 'content': ''}
                elif current_section:
                    current_section['content'] += text + '\n'
                else:
                    # If we find text before any heading, create a misc section
                    current_section = {'title': 'Introduction', 'content': text + '\n'}
            
            if current_section and current_section['content'].strip():
                sections.append(current_section)
            
            # Filter out very short sections (likely false positives)
            sections = [s for s in sections if len(s['content']) > 50]
            
            print(f"Found {len(sections)} sections to modernize\n")
            
            # Process each section
            modernized_sections = []
            for i, section in enumerate(sections, 1):
                print(f"\nProcessing section {i}/{len(sections)}: {section['title']}")
                print("\n" + "="*80)
                print(f"PROCESSING SECTION: {section['title']}")
                print("="*80 + "\n")
                
                result = self.process_section(section)
                modernized_sections.append(result)
                
                # Save after each section in case of interruption
                self.save_markdown(modernized_sections, output_dir)
                print(f"✓ Progress saved ({i}/{len(sections)} sections)")
            
            print(f"\nModernization complete! Results saved to {output_dir}")
            
        except Exception as e:
            print(f"Error processing document: {str(e)}")

def load_env_file(env_path='.env'):
    """Load environment variables from file."""
    env_vars = {}
    try:
        with open(env_path, 'r') as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    env_vars[key] = value
    except:
        pass
    return env_vars

def main():
    # Load API token
    env_vars = load_env_file()
    api_token = env_vars.get('HUGGINGFACE_API_TOKEN')
    
    if not api_token:
        print("Error: HUGGINGFACE_API_TOKEN not found in .env file")
        return
    
    input_file = "hospital_automation_requirements.docx"
    output_dir = "modernized_requirements_2025"
    
    modernizer = RequirementsModernizer(api_token)
    modernizer.process_document(input_file, output_dir)

if __name__ == "__main__":
    main()
