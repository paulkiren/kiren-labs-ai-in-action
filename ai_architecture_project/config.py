import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DOCUMENTS_DIR = PROJECT_ROOT / "documents"
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"

# Model configuration
OLLAMA_MODEL = "llama3.2:7b"  # Change to 3b if 7b is too slow
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Processing parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_RETRIEVAL_DOCS = 5

# Create directories
DOCUMENTS_DIR.mkdir(exist_ok=True)
VECTOR_DB_DIR.mkdir(exist_ok=True)