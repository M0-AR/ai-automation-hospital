import PyPDF2
import json
import os
from datetime import datetime

def extract_pdf_content(pdf_path):
    """Extract text content from PDF file."""
    content = []
    try:
        with open(pdf_path, 'rb') as file:
            # Create PDF reader object
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Get number of pages
            num_pages = len(pdf_reader.pages)
            
            # Extract text from each page
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                content.append({
                    'page': page_num + 1,
                    'text': page.extract_text()
                })
            
            return content
    except Exception as e:
        print(f"Error reading PDF: {str(e)}")
        return None

def save_to_json(content, output_path):
    """Save extracted content to JSON file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(content, file, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"Error saving JSON: {str(e)}")
        return False

def main():
    # Input and output paths
    pdf_path = 'hospital_automation_requirements copy.pdf'
    output_path = 'extracted_requirements.json'
    
    # Extract content
    print("Extracting PDF content...")
    content = extract_pdf_content(pdf_path)
    
    if content:
        # Save to JSON
        print("Saving to JSON...")
        if save_to_json(content, output_path):
            print(f"Successfully saved content to {output_path}")
        else:
            print("Failed to save content")
    else:
        print("Failed to extract content from PDF")

if __name__ == "__main__":
    main()
