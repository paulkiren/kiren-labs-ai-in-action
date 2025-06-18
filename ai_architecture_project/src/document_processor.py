import os
from pathlib import Path
from typing import List
import PyPDF2
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    
    def extract_text_from_txt(self, file_path: str) -> str:
        """Extract text from TXT file"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def process_document(self, file_path: str) -> List[str]:
        """Process a single document and return chunks"""
        file_path = Path(file_path)
        
        # Extract text based on file extension
        if file_path.suffix.lower() == '.pdf':
            text = self.extract_text_from_pdf(file_path)
        elif file_path.suffix.lower() == '.docx':
            text = self.extract_text_from_docx(file_path)
        elif file_path.suffix.lower() == '.txt':
            text = self.extract_text_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        # Split into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Add metadata to chunks
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            processed_chunks.append({
                'content': chunk,
                'source': str(file_path.name),
                'chunk_id': i
            })
        
        return processed_chunks
    
    def process_directory(self, directory_path: str) -> List[dict]:
        """Process all documents in a directory"""
        all_chunks = []
        directory_path = Path(directory_path)
        
        supported_extensions = ['.pdf', '.docx', '.txt']
        
        for file_path in directory_path.iterdir():
            if file_path.suffix.lower() in supported_extensions:
                print(f"Processing: {file_path.name}")
                try:
                    chunks = self.process_document(file_path)
                    all_chunks.extend(chunks)
                    print(f"  → Generated {len(chunks)} chunks")
                except Exception as e:
                    print(f"  → Error processing {file_path.name}: {e}")
        
        return all_chunks