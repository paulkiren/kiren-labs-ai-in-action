import ollama
from typing import List, Dict

class LLMInterface:
    def __init__(self, model_name: str = "llama3.2:7b"):
        self.model_name = model_name
        self.client = ollama.Client()
    
    def generate_response(self, query: str, context_documents: List[Dict]) -> str:
        """Generate response using retrieved context"""
        
        # Prepare context from retrieved documents
        context = "\n\n".join([
            f"Source: {doc['source']}\nContent: {doc['content']}"
            for doc in context_documents
        ])
        
        # Create prompt
        prompt = f"""Based on the following context documents, please answer the question. If the answer cannot be found in the context, please say so.

Context:
{context}

Question: {query}

Answer:"""
        
        try:
            # Generate response using Ollama
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'max_tokens': 500
                }
            )
            
            return response['response']
        
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def test_connection(self) -> bool:
        """Test if Ollama connection is working"""
        try:
            models = self.client.list()
            available_models = [model['name'] for model in models['models']]
            return self.model_name in available_models
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False