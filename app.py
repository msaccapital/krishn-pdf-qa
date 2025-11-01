from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile
import logging
from typing import Dict, Any
import json

# Import your existing KRISHN system components
from krishn_system import EnhancedPDFProcessor, HybridVectorDB, EnhancedAnswerGenerator
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Set cache directories for Hugging Face models
os.environ['TRANSFORMERS_CACHE'] = '/opt/render/.cache/huggingface'
os.environ['HF_HUB_CACHE'] = '/opt/render/.cache/huggingface'
os.environ['HF_HOME'] = '/opt/render/.cache/huggingface'

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KRISHNBackend:
    def __init__(self):
        self.system = None
        self.is_initialized = False
        self.uploaded_files = []
        
    def initialize_system(self):
        """Initialize the KRISHN system with Mistral model"""
        try:
            logger.info("🚀 Initializing KRISHN system with Mistral 7B...")
            
            # Create cache directory if it doesn't exist
            os.makedirs('/opt/render/.cache/huggingface', exist_ok=True)
            
            # Load model with optimizations
            model_name = "mistralai/Mistral-7B-Instruct-v0.2"
            
            logger.info("📥 Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir='/opt/render/.cache/huggingface'
            )
            tokenizer.pad_token = tokenizer.eos_token
            
            logger.info("📥 Loading model (this may take 10-15 minutes)...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_4bit=True,
                trust_remote_code=True,
                cache_dir='/opt/render/.cache/huggingface',
                low_cpu_mem_usage=True  # Add memory optimization
            )
            
            logger.info("🔧 Creating pipeline...")
            mistral_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            # Initialize system components
            self.pdf_processor = EnhancedPDFProcessor()
            self.vector_db = HybridVectorDB()
            self.answer_gen = EnhancedAnswerGenerator(self.vector_db, mistral_pipeline)
            
            self.is_initialized = True
            logger.info("✅ KRISHN system initialized successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize system: {e}")
            self.is_initialized = False
    
    def process_uploaded_pdfs(self, pdf_files):
        """Process uploaded PDF files"""
        try:
            if not self.is_initialized:
                return {"success": False, "error": "System not initialized"}
            
            # Save uploaded files temporarily
            temp_files = []
            for pdf_file in pdf_files:
                temp_path = os.path.join(tempfile.gettempdir(), pdf_file.filename)
                pdf_file.save(temp_path)
                temp_files.append(temp_path)
                self.uploaded_files.append(pdf_file.filename)
            
            # Process PDFs using the existing system
            final_pdf_path = self.pdf_processor.upload_and_combine_pdfs(pdf_files)
            
            if not final_pdf_path:
                return {"success": False, "error": "Failed to process PDFs"}
            
            # Extract text and create vector DB
            chunks = self.pdf_processor.extract_text_from_pdf(final_pdf_path)
            
            if not chunks:
                return {"success": False, "error": "No text extracted from PDF"}
            
            # Create vector database
            success = self.vector_db.create_index_from_chunks(
                chunks, 
                self.pdf_processor.document_boundaries
            )
            
            # Cleanup temp files
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            
            if success:
                return {
                    "success": True, 
                    "message": f"Processed {len(chunks)} chunks from {len(pdf_files)} PDF(s)",
                    "documents": self.uploaded_files
                }
            else:
                return {"success": False, "error": "Failed to create vector database"}
                
        except Exception as e:
            logger.error(f"PDF processing error: {e}")
            return {"success": False, "error": str(e)}

# Global backend instance
krishn_backend = KRISHNBackend()

@app.route('/')
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "KRISHN Enhanced PDF QA System",
        "initialized": krishn_backend.is_initialized,
        "model": "Mistral-7B-Instruct-v0.2"
    })

@app.route('/api/initialize', methods=['POST'])
def initialize_system():
    """Initialize the KRISHN system"""
    try:
        krishn_backend.initialize_system()
        return jsonify({
            "success": krishn_backend.is_initialized,
            "message": "System initialized successfully" if krishn_backend.is_initialized else "Initialization failed - check logs",
            "model_loaded": krishn_backend.is_initialized
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload_pdfs():
    """Upload and process PDF files"""
    try:
        if 'pdfs' not in request.files:
            return jsonify({"success": False, "error": "No PDF files provided"}), 400
        
        pdf_files = request.files.getlist('pdfs')
        if not pdf_files or pdf_files[0].filename == '':
            return jsonify({"success": False, "error": "No valid PDF files"}), 400
        
        # Check if files are PDFs
        for pdf_file in pdf_files:
            if not pdf_file.filename.lower().endswith('.pdf'):
                return jsonify({"success": False, "error": "All files must be PDFs"}), 400
        
        result = krishn_backend.process_uploaded_pdfs(pdf_files)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/ask', methods=['POST'])
def ask_question():
    """Ask a question about the uploaded PDFs"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400
            
        question = data.get('question', '').strip()
        search_mode = data.get('search_mode', 'hybrid')
        
        if not question:
            return jsonify({"success": False, "error": "No question provided"}), 400
        
        if not krishn_backend.is_initialized:
            return jsonify({"success": False, "error": "System not initialized"}), 400
        
        # Generate answer
        result = krishn_backend.answer_gen.generate_answer(
            question=question,
            search_mode=search_mode
        )
        
        return jsonify({
            "success": True,
            "question": question,
            "answer": result['answer'],
            "sources": result['sources'],
            "confidence": result.get('confidence', 0),
            "response_time": result.get('response_time', 0),
            "search_mode": search_mode
        })
        
    except Exception as e:
        logger.error(f"Question error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    """Get system analytics"""
    try:
        if not krishn_backend.is_initialized:
            return jsonify({"success": False, "error": "System not initialized"}), 400
        
        stats = krishn_backend.answer_gen.get_system_stats()
        return jsonify({"success": True, "analytics": stats})
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/clear', methods=['POST'])
def clear_system():
    """Clear conversation and cache"""
    try:
        if krishn_backend.is_initialized:
            krishn_backend.answer_gen.clear_cache()
            krishn_backend.answer_gen.conversation.clear_history()
        
        return jsonify({"success": True, "message": "System cleared successfully"})
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get system status"""
    return jsonify({
        "initialized": krishn_backend.is_initialized,
        "uploaded_documents": krishn_backend.uploaded_files,
        "system_ready": krishn_backend.is_initialized and len(krishn_backend.uploaded_files) > 0,
        "model": "Mistral-7B-Instruct-v0.2"
    })

if __name__ == '__main__':
    # Initialize system on startup
    print("🚀 Starting KRISHN PDF QA System with Mistral 7B...")
    print("📥 Model will load on first request (may take 10-15 minutes)...")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
