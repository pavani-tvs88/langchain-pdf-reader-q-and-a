import gradio as gr
from typing import List, Tuple
import os
from datetime import datetime


class ChatInterface:
    """Gradio UI for the Q&A bot"""
    
    def __init__(self, document_processor, qa_engine):
        self.doc_processor = document_processor
        self.qa_engine = qa_engine
        self.current_files = []
        self.vectorstore = None
    
    def process_upload(self, files) -> str:
        """Process uploaded PDF files"""
        if not files:
            return "⚠️ Please upload at least one PDF file."
        
        try:
            file_paths = [file.name for file in files]
            self.current_files = file_paths
            
            self.vectorstore = self.doc_processor.process_files(file_paths)
            self.qa_engine.setup_chain(self.vectorstore)
            
            file_names = [os.path.basename(f) for f in file_paths]
            return f"✅ Successfully processed {len(files)} file(s):\n" + "\n".join([f"• {name}" for name in file_names])
        
        except Exception as e:
            return f"❌ Error processing files: {str(e)}"
    
    def answer_question(self, query: str, chat_history: List) -> Tuple[List, str]:
        """Handle question and update chat"""
        if not self.vectorstore:
            return chat_history, "⚠️ Please upload and process PDF files first."
        
        if not query.strip():
            return chat_history, "⚠️ Please enter a question."
        
        result = self.qa_engine.ask(query)
        
        if result['error']:
            response = result['answer']
        else:
            response = result['answer'] + result['sources']
        
        chat_history.append((query, response))
        
        return chat_history, ""
    
    def generate_summary(self) -> str:
        """Generate document summary"""
        if not self.vectorstore:
            return "⚠️ Please upload and process PDF files first."
        
        try:
            summary = self.qa_engine.summarize_document(self.vectorstore)
            return f"📋 **Document Summary:**\n\n{summary}"
        except Exception as e:
            return f"❌ Error generating summary: {str(e)}"
    
    def clear_chat(self) -> Tuple[List, str]:
        """Clear chat history"""
        self.qa_engine.clear_history()
        return [], "✅ Chat history cleared"
    
    def export_conversation(self, chat_history: List) -> str:
        """Export chat history to file"""
        if not chat_history:
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chat_export_{timestamp}.txt"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("PDF Q&A - CHAT EXPORT\n")
                f.write(f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")
                
                for i, (q, a) in enumerate(chat_history, 1):
                    f.write(f"Question {i}:\n{q}\n\n")
                    f.write(f"Answer {i}:\n{a}\n\n")
                    f.write("-" * 80 + "\n\n")
            
            return filename
        except Exception as e:
            print(f"Error exporting chat: {str(e)}")
            return None
    
    def create_interface(self):
        """Create and return Gradio interface"""
        
        custom_css = """
        .gradio-container {
            font-family: 'Arial', sans-serif;
        }
        .header {
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        """
        
        with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
            gr.HTML("""
                <div class="header">
                    <h1>🤖 PDF Q&A Bot</h1>
                    <p>Upload PDFs and ask questions using OpenAI</p>
                </div>
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📤 Upload Documents")
                    
                    file_upload = gr.File(
                        label="Select PDF Files",
                        file_count="multiple",
                        file_types=[".pdf"]
                    )
                    
                    upload_btn = gr.Button("📚 Process Documents", variant="primary", size="lg")
                    upload_status = gr.Textbox(label="Status", lines=3, interactive=False)
                    
                    gr.Markdown("---")
                    gr.Markdown("### 🎯 Quick Actions")
                    
                    summary_btn = gr.Button("📋 Generate Summary", size="sm")
                    clear_btn = gr.Button("🗑️ Clear Chat", size="sm")
                    export_btn = gr.Button("💾 Export Chat", size="sm")
                    
                    export_file = gr.File(label="Download Chat Export", visible=False)
                
                with gr.Column(scale=2):
                    gr.Markdown("### 💬 Ask Questions")
                    
                    chatbot = gr.Chatbot(
                        label="Conversation",
                        height=400,
                        avatar_images=None
                    )
                    
                    with gr.Row():
                        query_input = gr.Textbox(
                            label="Your Question",
                            placeholder="Ask anything about your documents...",
                            lines=2,
                            scale=4
                        )
                        ask_btn = gr.Button("Ask", variant="primary", scale=1)
            
            gr.Markdown("""
                ---
                ### 💡 Tips
                - Upload multiple PDFs at once for comprehensive searches
                - Ask follow-up questions for deeper insights
                - Use the summary feature to get a quick overview
                
                **Tech Stack:** LangChain • OpenAI • ChromaDB • Gradio
            """)
            
            upload_btn.click(
                fn=self.process_upload,
                inputs=[file_upload],
                outputs=[upload_status]
            )
            
            ask_btn.click(
                fn=self.answer_question,
                inputs=[query_input, chatbot],
                outputs=[chatbot, query_input]
            )
            
            query_input.submit(
                fn=self.answer_question,
                inputs=[query_input, chatbot],
                outputs=[chatbot, query_input]
            )
            
            summary_btn.click(
                fn=self.generate_summary,
                inputs=[],
                outputs=[upload_status]
            )
            
            clear_btn.click(
                fn=self.clear_chat,
                inputs=[],
                outputs=[chatbot, upload_status]
            )
            
            def export_and_show(chat_history):
                file_path = self.export_conversation(chat_history)
                if file_path:
                    return gr.File(value=file_path, visible=True)
                return gr.File(visible=False)
            
            export_btn.click(
                fn=export_and_show,
                inputs=[chatbot],
                outputs=[export_file]
            )
        
        return demo