import os
import json
from pathlib import Path
import re
import docx

def clean_filename(text):
    """Convert text to a valid filename."""
    # Replace invalid characters with underscore
    text = re.sub(r'[^\w\s-]', '_', text)
    # Replace spaces with dashes
    text = re.sub(r'[-\s]+', '-', text).strip('-')
    return text.lower()

def create_module_structure(sections):
    """Create a module-based structure from sections."""
    modules = {
        'core': {
            'title': 'Core System',
            'sections': []
        },
        'patient': {
            'title': 'Patient Management',
            'sections': []
        },
        'medical': {
            'title': 'Medical Records',
            'sections': []
        },
        'pharmacy': {
            'title': 'Pharmacy Management',
            'sections': []
        },
        'lab': {
            'title': 'Laboratory System',
            'sections': []
        },
        'admin': {
            'title': 'Administration',
            'sections': []
        },
        'security': {
            'title': 'Security & Access Control',
            'sections': []
        },
        'integration': {
            'title': 'System Integration',
            'sections': []
        },
        'misc': {
            'title': 'Miscellaneous',
            'sections': []
        }
    }
    
    # Keywords to help categorize sections
    keywords = {
        'core': ['system', 'platform', 'architecture', 'infrastructure', 'base', 'framework'],
        'patient': ['patient', 'appointment', 'scheduling', 'registration', 'admission'],
        'medical': ['medical', 'record', 'diagnosis', 'treatment', 'doctor', 'nurse', 'clinical'],
        'pharmacy': ['pharmacy', 'medication', 'drug', 'prescription'],
        'lab': ['lab', 'laboratory', 'test', 'specimen', 'sample'],
        'admin': ['admin', 'billing', 'invoice', 'payment', 'staff', 'employee', 'inventory'],
        'security': ['security', 'access', 'authentication', 'permission', 'role', 'audit'],
        'integration': ['integration', 'interface', 'api', 'connect', 'external']
    }
    
    for section in sections:
        title = section['title'].lower()
        content = section['content'].lower()
        assigned = False
        
        # Try to categorize based on keywords
        for module, words in keywords.items():
            if any(word in title or word in content for word in words):
                modules[module]['sections'].append(section)
                assigned = True
                break
        
        # If no category found, put in misc
        if not assigned:
            modules['misc']['sections'].append(section)
    
    return modules

def create_readme(module_name, module_data, base_dir):
    """Create a README.md file for a module."""
    module_dir = Path(base_dir) / module_name
    module_dir.mkdir(parents=True, exist_ok=True)
    
    readme_content = [
        f"# {module_data['title']}",
        "\n## Overview",
        "This module is part of the Hospital Automation System.",
        "\n## Requirements\n"
    ]
    
    for section in module_data['sections']:
        readme_content.extend([
            f"### {section['title']}",
            section['content'],
            ""
        ])
    
    # Create README.md
    with open(module_dir / 'README.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(readme_content))
    
    # Create a basic package structure
    (module_dir / 'src').mkdir(exist_ok=True)
    (module_dir / 'tests').mkdir(exist_ok=True)
    (module_dir / 'docs').mkdir(exist_ok=True)
    
    # Create an empty __init__.py
    (module_dir / 'src' / '__init__.py').touch()
    
    return module_dir

def create_main_readme(modules, base_dir):
    """Create the main README.md with links to all modules."""
    content = [
        "# Hospital Automation System",
        "\nThis system is divided into the following modules for better organization and development:\n"
    ]
    
    for module_name, module_data in modules.items():
        content.extend([
            f"## [{module_data['title']}]({module_name}/README.md)",
            f"Number of requirements: {len(module_data['sections'])}",
            ""
        ])
    
    content.extend([
        "\n## Development",
        "Each module contains:",
        "- `src/`: Source code",
        "- `tests/`: Unit tests",
        "- `docs/`: Additional documentation",
        "- `README.md`: Module-specific requirements",
        "\n## Getting Started",
        "1. Choose a module to work on",
        "2. Read its README.md for requirements",
        "3. Start development in the module's `src` directory",
        "4. Add tests in the module's `tests` directory"
    ])
    
    with open(Path(base_dir) / 'README.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(content))

def extract_sections_from_docx(file_path):
    """Extract sections from a Word document."""
    doc = docx.Document(file_path)
    sections = []
    current_section = None
    
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
            
        # Check if it's a heading
        if paragraph.style.name.startswith('Heading'):
            if current_section:
                sections.append(current_section)
            current_section = {'title': text, 'content': ''}
        elif current_section:
            current_section['content'] += text + '\n'
        else:
            # If we find text before any heading, create a misc section
            current_section = {'title': 'Introduction', 'content': text + '\n'}
    
    if current_section:
        sections.append(current_section)
    
    return sections

def main():
    # Read the requirements from Word document
    sections = extract_sections_from_docx('hospital_automation_requirements.docx')
    
    # Create modules structure
    modules = create_module_structure(sections)
    
    # Create base directory for modules
    base_dir = Path('modules')
    base_dir.mkdir(exist_ok=True)
    
    # Create module directories and READMEs
    for module_name, module_data in modules.items():
        create_readme(module_name, module_data, base_dir)
    
    # Create main README
    create_main_readme(modules, base_dir)
    
    print("✓ Requirements have been split into modules!")
    print(f"✓ Check the {base_dir} directory for the organized structure")
    print("\nModule breakdown:")
    for module_name, module_data in modules.items():
        print(f"- {module_data['title']}: {len(module_data['sections'])} requirements")

if __name__ == "__main__":
    main()
