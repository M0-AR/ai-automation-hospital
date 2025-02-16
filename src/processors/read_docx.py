from docx import Document
import json

try:
    # Open the document
    doc = Document("hospital_automation_requirements.docx")
    
    # Extract text from paragraphs
    content = []
    for para in doc.paragraphs:
        if para.text.strip():
            content.append(para.text)
    
    # Write to a JSON file for easier processing
    with open('doc_content.json', 'w', encoding='utf-8') as f:
        json.dump(content, f, ensure_ascii=False, indent=2)
    
    print("Successfully extracted content to doc_content.json")
except Exception as e:
    print(f"Error: {str(e)}")
