import os
import shutil
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma


class DocumentProcessor:
    """Handles PDF loading, splitting, and vector store creation"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key

        # Initialize OpenAI embeddings
        try:
            from langchain.embeddings import OpenAIEmbeddings
            self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        except Exception:
            self.embeddings = None
            print("⚠️ OpenAI embeddings not available — install 'openai' and 'langchain' or set `.embeddings` manually.")

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
    
    def create_vectorstore(self, chunks: List, persist_directory: str = "./chroma_db", force_recreate: bool = False) -> Chroma:
        """Create or update vector store.

        By default, if a persist directory exists we will load it instead of
        re-embedding to avoid unnecessary quota usage. Set `force_recreate=True`
        to force deletion and re-creation.
        """
        # Ensure embeddings are configured
        if not self.embeddings:
            raise RuntimeError("Embeddings provider not configured — install dependencies or set `DocumentProcessor.embeddings`.")

        if os.path.exists(persist_directory):
            if force_recreate:
                shutil.rmtree(persist_directory)
                print(f"✓ Cleaned old database at {persist_directory}")
            else:
                # Load existing vectorstore to avoid re-embedding
                try:
                    vectorstore = Chroma(persist_directory=persist_directory, embedding=self.embeddings)
                    print(f"✓ Loaded existing vector store from {persist_directory}")
                    return vectorstore
                except Exception:
                    # If loading fails, remove and recreate
                    shutil.rmtree(persist_directory)
                    print(f"⚠️ Failed to load existing DB; recreating {persist_directory}")

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