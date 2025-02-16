import os
import csv
import time
import json
import requests
from datetime import datetime
from typing import Dict, List
from dotenv import load_dotenv

class ModelTester:
    def __init__(self, api_token: str):
        self.api_token = api_token
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }
        # List of models to test (verified working with free tier)
        self.models = [
            # Proven Working Models (Fast and Reliable)
            'google/flan-t5-small',         # 300MB - Fast and reliable
            'google/flan-t5-base',          # 892MB - Better quality
            'EleutherAI/pythia-160m',       # 160MB - Good balance
            
            # Powerful Small Models
            'microsoft/phi-1_5',            # 1.3GB - Smaller but powerful
            'Qwen/Qwen1.5-0.5B',           # 500MB - Very powerful for size
            'TinyLlama/TinyLlama-1.1B-Chat-v0.4',  # 1.1GB - Fast chat
            
            # Medical & Scientific
            'google/pegasus-pubmed',        # Medical papers
            'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
            
            # Summarization Models
            'facebook/bart-large-cnn',      # 1.6GB - Summarization
            'google/pegasus-xsum',          # 2.3GB - Great summarization
            
            # Multilingual Support
            'Helsinki-NLP/opus-mt-mul-en',  # Multi-language to English
            'facebook/mbart-large-50-many-to-many-mmt',  # Multilingual
            
            # Small but Efficient
            'MBZUAI/LaMini-Flan-T5-248M',  # 248MB - Efficient
            'Xenova/distilbert-base-uncased' # 267MB - Classification
        ]
        
        # Test cases including English and Arabic
        self.test_cases = [
            # English Medical Cases
            "Modernize hospital system requirements for patient registration",
            "Summarize the latest research on electronic health records",
            "Generate a specification for medical data integration",
            
            # Arabic Medical Cases
            "تحديث نظام تسجيل المرضى في المستشفى",  # Update hospital patient registration
            "تطوير نظام السجلات الطبية الإلكترونية",  # Develop electronic medical records
            "تحسين نظام إدارة البيانات الطبية"  # Improve medical data management
        ]

        self.results_file = 'model_test_results.csv'

    def test_model(self, model: str, test_input: str) -> Dict:
        """Test a single model with given input."""
        print(f"\n🔄 Testing model: {model}")
        print(f"📝 Input: {test_input[:100]}...")
        
        api_url = f"https://api-inference.huggingface.co/models/{model}"
        max_retries = 3
        retry_delay = 5
        timeout = 20  # Shorter timeout
        max_load_wait = 30  # Maximum time to wait for model loading
        
        # Model-specific parameters
        if 'flan-t5' in model.lower():
            payload = {
                "inputs": test_input,
                "parameters": {
                    "max_length": 512,
                    "min_length": 64,
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "do_sample": True
                }
            }
        elif 'pegasus' in model.lower() or 'bart' in model.lower():
            # Summarization models
            payload = {
                "inputs": test_input,
                "parameters": {
                    "max_length": 256,
                    "min_length": 32,
                    "length_penalty": 2.0,
                    "num_beams": 4,
                    "early_stopping": True
                }
            }
        elif 'biomedbert' in model.lower() or 'pubmedbert' in model.lower():
            # Medical models
            payload = {
                "inputs": test_input,
                "parameters": {
                    "max_length": 512,
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "do_sample": True,
                    "num_return_sequences": 1
                }
            }
        elif 'opus-mt' in model.lower() or 'mbart' in model.lower():
            # Translation models
            is_arabic = any(c in test_input for c in 'ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىيًٌٍَُِّْ')
            src_lang = 'ar' if is_arabic else 'en'
            tgt_lang = 'en' if is_arabic else 'ar'
            payload = {
                "inputs": test_input,
                "parameters": {
                    "src_lang": src_lang,
                    "tgt_lang": tgt_lang,
                    "max_length": 512
                }
            }
        else:
            # Default parameters for other models
            payload = {
                "inputs": test_input,
                "parameters": {
                    "max_length": 512,
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "do_sample": True
                }
            }

        for attempt in range(max_retries):
            try:
                # Quick check if model is available
                status_response = requests.post(
                    api_url, 
                    headers=self.headers, 
                    json={"inputs": "test"},
                    timeout=timeout
                )
                
                if status_response.status_code == 503:
                    estimated_time = status_response.json().get("estimated_time", 0)
                    if estimated_time > 0 and estimated_time <= max_load_wait:
                        print(f"⏳ Model is loading. Waiting {round(estimated_time, 2)} seconds...")
                        time.sleep(min(estimated_time, max_load_wait))
                    else:
                        print("⚠️ Model loading time too long, skipping...")
                        return {
                            'model': model,
                            'input': test_input,
                            'timestamp': datetime.now().isoformat(),
                            'response_time': None,
                            'status_code': 503,
                            'success': False,
                            'error': f'Model loading time exceeded {max_load_wait} seconds',
                            'output': None
                        }
                
                # Make the actual request
                start_time = time.time()
                response = requests.post(api_url, headers=self.headers, json=payload, timeout=timeout)
                end_time = time.time()
                
                result = {
                    'model': model,
                    'input': test_input,
                    'timestamp': datetime.now().isoformat(),
                    'response_time': round(end_time - start_time, 2),
                    'status_code': response.status_code,
                    'success': False,
                    'error': None,
                    'output': None
                }

                if response.status_code == 200:
                    response_data = response.json()
                    
                    # Handle different response formats
                    if isinstance(response_data, list):
                        output = response_data[0].get('generated_text', '')
                    elif isinstance(response_data, dict):
                        output = response_data.get('generated_text', '')
                        if not output:  # Try other common response formats
                            output = response_data.get('answer', '')
                            if not output:
                                output = str(response_data)
                    else:
                        output = str(response_data)
                    
                    if output and len(output.strip()) > 0:
                        result['success'] = True
                        result['output'] = output
                        print(f"✅ Success! Response time: {result['response_time']}s")
                        return result
                    else:
                        result['error'] = "Empty response"
                        print("❌ Empty response")
                
                elif response.status_code == 404:
                    result['error'] = "Model not found"
                    print("❌ Model not found")
                    return result
                    
                elif response.status_code == 400:
                    result['error'] = response.json().get('error', 'Bad request')
                    print(f"❌ Error: {result['error']}")
                    return result
                    
                else:
                    result['error'] = f"HTTP {response.status_code}: {response.text}"
                    print(f"❌ Error: {result['error']}")
                
                if attempt < max_retries - 1:
                    print(f"⏳ Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    continue
                    
                return result

            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    print(f"⏳ Request timed out. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    continue
                else:
                    return {
                        'model': model,
                        'input': test_input,
                        'timestamp': datetime.now().isoformat(),
                        'response_time': None,
                        'status_code': None,
                        'success': False,
                        'error': f'Timeout after {max_retries} attempts',
                        'output': None
                    }
                    
            except Exception as e:
                return {
                    'model': model,
                    'input': test_input,
                    'timestamp': datetime.now().isoformat(),
                    'response_time': None,
                    'status_code': None,
                    'success': False,
                    'error': str(e),
                    'output': None
                }

    def save_results(self, results: List[Dict]):
        """Save test results to CSV file."""
        fieldnames = ['model', 'input', 'timestamp', 'response_time', 'status_code', 
                     'success', 'error', 'output']
        
        file_exists = os.path.exists(self.results_file)
        
        with open(self.results_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerows(results)

    def run_tests(self):
        """Run tests for all models and test cases."""
        all_results = []
        
        print(f"🚀 Starting model tests with {len(self.models)} models and {len(self.test_cases)} test cases")
        
        for test_input in self.test_cases:
            print(f"\n{'='*80}")
            print(f"Testing with input: {test_input[:100]}...")
            print(f"{'='*80}")
            
            for model in self.models:
                result = self.test_model(model, test_input)
                all_results.append(result)
                
                # Save after each test to preserve results
                self.save_results([result])
                
                # Add delay between tests to avoid rate limits
                time.sleep(5)
        
        return all_results

    def analyze_results(self):
        """Analyze and print summary of test results."""
        if not os.path.exists(self.results_file):
            print("❌ No test results found")
            return
        
        results = []
        with open(self.results_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            results = list(reader)
        
        print(f"\n{'='*80}")
        print("📊 MODEL TEST RESULTS SUMMARY")
        print(f"{'='*80}")
        
        model_stats = {}
        for result in results:
            model = result['model']
            if model not in model_stats:
                model_stats[model] = {
                    'total': 0,
                    'success': 0,
                    'avg_time': [],
                    'errors': []
                }
            
            stats = model_stats[model]
            stats['total'] += 1
            if result['success'] == 'True':
                stats['success'] += 1
                if result['response_time']:
                    stats['avg_time'].append(float(result['response_time']))
            if result['error']:
                stats['errors'].append(result['error'])
        
        # Print summary
        print("\nModel Performance Summary:")
        print(f"{'Model':<50} | {'Success Rate':<12} | {'Avg Time':<10} | {'Common Errors'}")
        print("-" * 100)
        
        for model, stats in model_stats.items():
            success_rate = (stats['success'] / stats['total']) * 100
            avg_time = sum(stats['avg_time']) / len(stats['avg_time']) if stats['avg_time'] else None
            common_error = max(set(stats['errors']), key=stats['errors'].count) if stats['errors'] else 'None'
            
            print(f"{model:<50} | {success_rate:>10.1f}% | {avg_time:>8.2f}s | {common_error[:30]}")

def main():
    load_dotenv()
    api_token = os.getenv('HF_API_TOKEN')
    if not api_token:
        print("❌ Error: HF_API_TOKEN not found in .env file")
        return
    
    tester = ModelTester(api_token)
    
    print("🔍 Starting model testing...")
    results = tester.run_tests()
    
    print("\n📊 Analyzing results...")
    tester.analyze_results()
    
    print(f"\n✅ Testing complete! Results saved to: {tester.results_file}")

if __name__ == "__main__":
    main()
