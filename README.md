# Hospital Automation Requirements Modernizer

This project provides tools to modernize hospital automation system requirements using AI. It processes legacy documentation and updates it to modern standards while maintaining both Arabic and English versions.

## Features

- Document processing (DOCX, PDF)
- Bilingual support (Arabic-English)
- AI-powered requirement modernization
- Structured output in markdown format
- Version comparison and tracking

## Project Structure

```
hospital-automation/
├── src/               # Source code
│   ├── modernizer/    # Core modernization logic
│   └── processors/    # Document processors
├── utils/             # Utility functions
├── tests/             # Test files
└── docs/              # Documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/hospital-automation.git
cd hospital-automation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

```python
from src.modernizer import RequirementsModernizer

modernizer = RequirementsModernizer()
modernizer.process_document("path/to/document.docx")
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
