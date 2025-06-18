# AI Architecture Learning Project

A comprehensive local AI system demonstrating document Q&A capabilities using Ollama and vector-based retrieval.

## Overview

This project implements a complete AI architecture that runs locally on Apple Silicon, featuring:

- Document processing and embedding generation
- Vector-based similarity search using ChromaDB  
- Local LLM inference with Ollama
- Interactive web interface built with Streamlit

## Requirements

- MacBook with Apple Silicon (M-series chip)
- Python 3.9+
- [Ollama](https://ollama.com/)
- 16GB+ RAM recommended

## Quick Start

1. Install dependencies:
```sh
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Install Ollama:
```sh
curl -fsSL https://ollama.com/install.sh | sh
```

3. Pull the required model:
```sh
ollama pull llama2:7b
```

4. Run the application:
```sh
cd ai_architecture_project
streamlit run app.py
```

## Project Structure

```
ai_architecture_project/
├── app.py              # Main Streamlit application
├── config.py           # Configuration settings
├── requirements.txt    # Project dependencies
└── src/
    ├── document_processor.py  # Document handling
    ├── embeddings.py         # Vector embeddings
    └── llm_interface.py      # LLM integration
```

## Key Features

- **Document Processing**: Support for PDF, DOCX, and TXT files
- **Vector Search**: Efficient similarity search using ChromaDB
- **Local LLM**: Integrated with Ollama for fast inference
- **Interactive UI**: Real-time chat with source citations
- **Extensible**: Modular design for easy customization

## Learning Path

### Week 1: Foundation
- Set up local environment and dependencies
- Understand basic LLM interaction
- Implement document processing

### Week 2: Advanced Features
- Add embedding generation
- Implement vector similarity search
- Build retrieval-augmented generation

### Week 3: Optimization
- Fine-tune chunking strategies
- Optimize model performance
- Enhance user interface

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.