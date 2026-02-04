cat > app.py << 'EOF'
import os
import sys
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from document_processor import DocumentProcessor
from qa_engine import QAEngine
from ui import ChatInterface


def main():
    """Main application entry point"""
    
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        print("❌ Error: GOOGLE_API_KEY not found in environment variables")
        print("\nCodespace secret should be automatically loaded (if you set it as a secret).")
        print("If not, create a `.env` file at the project root with:\n  GOOGLE_API_KEY=your_key")
        print("\nTo verify in the terminal: `echo $GOOGLE_API_KEY`")
        print("To set a Codespace secret: `gh codespace secret set -n GOOGLE_API_KEY -b <your_key>`")
        sys.exit(1)
    
    print("=" * 80)
    print("🚀 Starting Gemini PDF Q&A Bot...")
    print("=" * 80)
    
    print("\n📦 Initializing components...")
    doc_processor = DocumentProcessor(api_key)
    qa_engine = QAEngine(api_key, temperature=0.2)
    
    print("✓ Document Processor initialized")
    print("✓ QA Engine initialized")
    
    print("✓ Creating user interface...")
    chat_ui = ChatInterface(doc_processor, qa_engine)
    demo = chat_ui.create_interface()
    
    print("\n✅ Application ready!")
    print("=" * 80)
    print("\n🌐 Launching on http://0.0.0.0:8000")
    print("Press Ctrl+C to stop\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=8000,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
EOF