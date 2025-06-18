import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from document_processor import DocumentProcessor
from embeddings import EmbeddingManager
from llm_interface import LLMInterface
from config import *

# Initialize components
@st.cache_resource
def initialize_components():
    processor = DocumentProcessor(CHUNK_SIZE, CHUNK_OVERLAP)
    embedding_manager = EmbeddingManager(EMBEDDING_MODEL, str(VECTOR_DB_DIR))
    llm = LLMInterface(OLLAMA_MODEL)
    return processor, embedding_manager, llm

def main():
    st.set_page_config(
        page_title="AI Architecture Learning Project",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 AI Architecture Learning Project")
    st.subheader("Document Q&A System with Local LLM")
    
    # Initialize components
    processor, embedding_manager, llm = initialize_components()
    
    # Sidebar for document management
    with st.sidebar:
        st.header("📚 Document Management")
        
        # Upload documents
        uploaded_files = st.file_uploader(
            "Upload documents",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'txt']
        )
        
        if uploaded_files:
            if st.button("Process Documents"):
                with st.spinner("Processing documents..."):
                    all_chunks = []
                    
                    for uploaded_file in uploaded_files:
                        # Save uploaded file temporarily
                        temp_path = DOCUMENTS_DIR / uploaded_file.name
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Process document
                        chunks = processor.process_document(temp_path)
                        all_chunks.extend(chunks)
                    
                    # Add to vector database
                    embedding_manager.add_documents(all_chunks)
                    
                    st.success(f"Processed {len(all_chunks)} chunks from {len(uploaded_files)} documents")
        
        # Display collection stats
        stats = embedding_manager.get_collection_stats()
        st.write(f"**Total chunks in database:** {stats['total_documents']}")
        
        # Test LLM connection
        if st.button("Test LLM Connection"):
            if llm.test_connection():
                st.success("✅ LLM connection successful")
            else:
                st.error("❌ LLM connection failed")
    
    # Main chat interface
    st.header("💬 Ask Questions About Your Documents")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("📖 Sources"):
                    for source in message["sources"]:
                        st.write(f"**{source['source']}** (similarity: {source['distance']:.3f})")
                        st.write(source['content'][:200] + "...")
                        st.write("---")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching and generating response..."):
                # Retrieve relevant documents
                retrieved_docs = embedding_manager.search_similar(prompt, MAX_RETRIEVAL_DOCS)
                
                if not retrieved_docs:
                    response = "I don't have any relevant documents to answer your question. Please upload some documents first."
                    sources = []
                else:
                    # Generate response using LLM
                    response = llm.generate_response(prompt, retrieved_docs)
                    sources = retrieved_docs
                
                st.markdown(response)
                
                # Show sources
                if sources:
                    with st.expander("📖 Sources"):
                        for source in sources:
                            st.write(f"**{source['source']}** (similarity: {source['distance']:.3f})")
                            st.write(source['content'][:200] + "...")
                            st.write("---")
        
        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "sources": sources if 'sources' in locals() else []
        })

if __name__ == "__main__":
    main()