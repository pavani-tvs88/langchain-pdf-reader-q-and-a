cat > src/qa_engine.py << 'EOF'
import os
from typing import Dict, List, Tuple
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma


class QAEngine:
    """Handles question-answering logic with conversation history"""
    
    def __init__(self, api_key: str, temperature: float = 0.2):
        self.api_key = api_key
        self.temperature = temperature
        self.llm = GoogleGenerativeAI(
            model="gemini-1.5-pro",
            google_api_key=api_key,
            temperature=temperature
        )
        self.qa_chain = None
        self.chat_history = []
    
    def setup_chain(self, vectorstore: Chroma):
        """Initialize the QA chain with retriever"""
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True,
            verbose=False
        )
        print("✓ QA chain initialized")
    
    def ask(self, query: str) -> Dict[str, any]:
        """Ask a question and get answer with sources"""
        if not self.qa_chain:
            raise ValueError("QA chain not initialized. Call setup_chain() first.")
        
        if not query.strip():
            return {
                "answer": "Please enter a valid question.",
                "sources": [],
                "error": True
            }
        
        try:
            # Use the chain's __call__ for broader LangChain compatibility
            response = self.qa_chain({"query": query})
            
            answer = response.get('result', 'No answer generated')
            source_docs = response.get('source_documents', [])
            sources = self._format_sources(source_docs)
            
            self.chat_history.append((query, answer))
            
            return {
                "answer": answer,
                "sources": sources,
                "error": False
            }
        
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            return {
                "answer": error_msg,
                "sources": [],
                "error": True
            }
    
    def _format_sources(self, source_docs: List) -> str:
        """Format source documents for display"""
        if not source_docs:
            return ""
        
        sources_text = "\n\n📎 **Sources:**\n"
        seen_pages = set()
        
        for doc in source_docs:
            page = doc.metadata.get('page')
            source = doc.metadata.get('source', 'Unknown')
            
            # Normalize page info and avoid arithmetic on non-int values
            if isinstance(page, int):
                page_label = f"Page {page + 1}"
                page_key = f"{source}_{page}"
            else:
                page_label = "Page N/A"
                page_key = f"{source}_N/A"
            
            if page_key in seen_pages:
                continue
            seen_pages.add(page_key)
            
            snippet = doc.page_content[:150].replace('\n', ' ')
            sources_text += f"\n- **{page_label}** ({os.path.basename(source)}): _{snippet}..._"
        
        return sources_text
    
    def get_chat_history(self) -> List[Tuple[str, str]]:
        """Return chat history"""
        return self.chat_history
    
    def clear_history(self):
        """Clear chat history"""
        self.chat_history = []
        print("✓ Chat history cleared")
    
    def summarize_document(self, vectorstore: Chroma) -> str:
        """Generate a summary of the loaded documents"""
        try:
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            docs = retriever.get_relevant_documents("summary overview main points")
            
            combined_text = "\n\n".join([doc.page_content for doc in docs[:3]])
            
            summary_prompt = f"""Provide a concise summary of the following document excerpts. 
            Focus on the main topics, key points, and overall theme:

            {combined_text}
            
            Summary:"""
            
            # Use predict for LLM compatibility across providers
            summary = self.llm.predict(summary_prompt)
            return summary
        
        except Exception as e:
            return f"Error generating summary: {str(e)}"
EOF