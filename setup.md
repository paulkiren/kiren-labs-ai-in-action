# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a base model
ollama pull llama2:7b

# Create a simple Python environment
python -m venv ai_project
source ai_project/bin/activate
pip install ollama langchain chromadb sentence-transformers