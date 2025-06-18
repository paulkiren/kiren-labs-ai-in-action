from sentence_transformers import SentenceTransformer
import chromadb
from typing import List, Dict
import uuid

class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", db_path: str = "./vector_db"):
        self.embedding_model = SentenceTransformer(model_name)
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_documents(self, chunks: List[Dict]):
        """Add document chunks to vector database"""
        if not chunks:
            return
        
        # Extract content for embedding
        contents = [chunk['content'] for chunk in chunks]
        
        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self.embedding_model.encode(contents).tolist()
        
        # Prepare data for ChromaDB
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [{
            'source': chunk['source'],
            'chunk_id': chunk['chunk_id']
        } for chunk in chunks]
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=contents,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Added {len(chunks)} chunks to vector database")
    
    def search_similar(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search for similar documents"""
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        # Search in vector database
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'content': results['documents'][0][i],
                'source': results['metadatas'][0][i]['source'],
                'distance': results['distances'][0][i]
            })
        
        return formatted_results
    
    def get_collection_stats(self):
        """Get statistics about the collection"""
        return {
            'total_documents': self.collection.count(),
            'collection_name': self.collection.name
        }