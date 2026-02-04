cat > src/document_processor.py << 'EOF'
import os
import shutil
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings


class DocumentProcessor:
    """Handles PDF loading, splitting, and vector store creation"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
    
    def load_pdfs(self, file_paths: List[str]) -> List:
        """Load multiple PDF files"""
        all_documents = []
        
        for file_path in file_paths:
            try:
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                all_documents.extend(documents)
                print(f"✓ Loaded: {os.path.basename(file_path)} ({len(documents)} pages)")
            except Exception as e:
                print(f"✗ Error loading {file_path}: {str(e)}")
        
        return all_documents
    
    def split_documents(self, documents: List) -> List:
        """Split documents into chunks"""
        chunks = self.text_splitter.split_documents(documents)
        print(f"✓ Created {len(chunks)} chunks from documents")
        return chunks
    
    def create_vectorstore(self, chunks: List, persist_directory: str = "./chroma_db") -> Chroma:
        """Create or update vector store"""
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)
            print(f"✓ Cleaned old database at {persist_directory}")
        
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=persist_directory
        )
        print(f"✓ Created vector store with {len(chunks)} chunks")
        
        return vectorstore
    
    def process_files(self, file_paths: List[str], persist_directory: str = "./chroma_db"):
        """Complete pipeline: load -> split -> vectorize"""
        if not file_paths:
            raise ValueError("No files provided")
        
        print("\n📚 Processing documents...")
        documents = self.load_pdfs(file_paths)
        
        if not documents:
            raise ValueError("No documents were successfully loaded")
        
        chunks = self.split_documents(documents)
        vectorstore = self.create_vectorstore(chunks, persist_directory)
        
        print("✅ Processing complete!\n")
        return vectorstore
EOF