# ==================================================
# 🚀 KRISHN - ENHANCED GENERALIZED MULTI-PDF QA SYSTEM
# With Advanced Chunking, Hybrid Search & Analytics
# ==================================================

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import os
import time
import json
from typing import List, Dict, Any, Tuple
from PyPDF2 import PdfMerger, PdfReader
import re
from collections import defaultdict
import hashlib
import numpy as np

# ==================================================
# 🚀 ENHANCED PDF PROCESSOR
# ==================================================

class EnhancedPDFProcessor:
    def __init__(self):
        self.uploaded_pdfs = []
        self.combined_pdf_path = "/tmp/combined_document.pdf"
        self.document_boundaries = []
        
    def upload_and_combine_pdfs(self, pdf_files):
        """Process uploaded PDF files"""
        print("📤 Processing PDF Files...")
        
        if not pdf_files:
            print("❌ No files provided.")
            return False
        
        os.makedirs('/tmp', exist_ok=True)
        
        pdf_paths = []
        for pdf_file in pdf_files:
            file_path = f"/tmp/{pdf_file.filename}"
            pdf_file.save(file_path)
            pdf_paths.append(file_path)
            self.uploaded_pdfs.append(pdf_file.filename)
            print(f"✅ Processed: {pdf_file.filename}")
        
        if not pdf_paths:
            print("❌ No PDF files found.")
            return False
        
        # Combine PDFs if multiple files
        if len(pdf_paths) > 1:
            print(f"🔄 Combining {len(pdf_paths)} PDF files...")
            merger = PdfMerger()
            
            current_page = 0
            for pdf_file in pdf_paths:
                with open(pdf_file, 'rb') as f:
                    pdf_reader = PdfReader(f)
                    num_pages = len(pdf_reader.pages)
                    
                self.document_boundaries.append({
                    'file': os.path.basename(pdf_file),
                    'start_page': current_page + 1,
                    'end_page': current_page + num_pages,
                    'total_pages': num_pages
                })
                
                merger.append(pdf_file)
                current_page += num_pages
            
            with open(self.combined_pdf_path, 'wb') as combined_file:
                merger.write(combined_file)
            merger.close()
            
            print(f"✅ Combined {len(pdf_paths)} PDFs")
            print("📄 Document Structure:")
            for boundary in self.document_boundaries:
                print(f"   - {boundary['file']}: Pages {boundary['start_page']}-{boundary['end_page']}")
                
            final_pdf_path = self.combined_pdf_path
        else:
            final_pdf_path = pdf_paths[0]
            print(f"✅ Using: {os.path.basename(final_pdf_path)}")
        
        return final_pdf_path
    
    def enhanced_smart_chunking(self, text: str, page_num: int, source_file: str, chunk_size: int = 400, overlap: int = 50) -> List[Dict]:
        """Enhanced chunking with overlap for better context preservation"""
        if not text or len(text.strip()) < 20:
            return []
        
        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Simple paragraph-based chunking
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        
        chunks = []
        for paragraph in paragraphs:
            if len(paragraph) <= chunk_size:
                chunks.append(self._create_chunk(paragraph, page_num, source_file))
            else:
                # Split long paragraphs by sentences
                sentences = re.split(r'[.!?]+', paragraph)
                sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
                
                current_chunk = []
                current_length = 0
                
                for sentence in sentences:
                    sentence_length = len(sentence)
                    if current_length + sentence_length > chunk_size and current_chunk:
                        chunk_text = ' '.join(current_chunk)
                        chunks.append(self._create_chunk(chunk_text, page_num, source_file))
                        
                        # Keep overlap for next chunk
                        overlap_text = ' '.join(current_chunk[-2:]) if len(current_chunk) >= 2 else current_chunk[-1]
                        current_chunk = [overlap_text[-overlap:]] if len(overlap_text) > overlap else [overlap_text]
                        current_length = len(current_chunk[0])
                        
                        current_chunk.append(sentence)
                        current_length += sentence_length + 1
                    else:
                        current_chunk.append(sentence)
                        current_length += sentence_length + 1
                
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(self._create_chunk(chunk_text, page_num, source_file))
        
        # Create overlapping chunks for better context
        enhanced_chunks = []
        for i, chunk in enumerate(chunks):
            enhanced_chunks.append(chunk)
            
            # Create overlapping chunk if this isn't the first chunk
            if i > 0 and len(chunks[i-1]['text']) > overlap:
                prev_chunk = chunks[i-1]
                overlap_text = prev_chunk['text'][-overlap:]
                combined_text = overlap_text + " " + chunk['text']
                
                if len(combined_text) <= chunk_size * 1.5:  # Don't create overly long chunks
                    enhanced_chunks.append(self._create_chunk(combined_text, page_num, source_file))
        
        # Remove duplicates based on content hash
        unique_chunks = {}
        for chunk in enhanced_chunks:
            chunk_hash = hashlib.md5(chunk['text'].encode()).hexdigest()
            if chunk_hash not in unique_chunks:
                unique_chunks[chunk_hash] = chunk
        
        return list(unique_chunks.values())
    
    def _create_chunk(self, text: str, page_num: int, source_file: str) -> Dict:
        """Create a standardized chunk"""
        return {
            'text': text,
            'source': source_file,
            'page_number': page_num,
            'chunk_id': f"page_{page_num}_chunk_{hashlib.md5(text.encode()).hexdigest()[:8]}",
            'word_count': len(text.split()),
            'char_count': len(text)
        }
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        """Extract text from any PDF with enhanced chunking"""
        print("📖 Extracting text from PDF...")
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                all_chunks = []
                total_pages = len(pdf_reader.pages)
                
                for page_num in range(total_pages):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    
                    if text.strip():
                        page_chunks = self.enhanced_smart_chunking(
                            text, 
                            page_num + 1,
                            os.path.basename(pdf_path)
                        )
                        all_chunks.extend(page_chunks)
                        print(f"📄 Page {page_num + 1}: {len(page_chunks)} chunks")
                
                print(f"✅ Extracted {len(all_chunks)} chunks from {total_pages} pages")
                return all_chunks
                
        except Exception as e:
            print(f"❌ Error extracting text: {e}")
            return []

# ==================================================
# 🚀 HYBRID VECTOR DATABASE
# ==================================================

class HybridVectorDB:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
        self.document_metadata = {}
        self.search_stats = {
            'total_searches': 0, 
            'vector_searches': 0,
            'keyword_searches': 0,
            'hybrid_searches': 0,
            'avg_scores': []
        }
        
    def create_index_from_chunks(self, chunks: List[Dict], document_boundaries=None):
        """Create vector database from any chunks"""
        print("🔨 Creating hybrid vector database...")
        
        if not chunks:
            print("❌ No chunks to process")
            return False
            
        try:
            self.chunks = chunks
            
            # Add document metadata if available
            if document_boundaries:
                self._add_document_metadata(chunks, document_boundaries)
            
            # Generate embeddings
            texts = [chunk['text'] for chunk in chunks]
            print(f"📊 Generating embeddings for {len(texts)} chunks...")
            embeddings = self.model.encode(texts, normalize_embeddings=True)
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(embeddings.astype('float32'))
            
            # Build keyword index for hybrid search
            self._build_keyword_index()
            
            # Save index
            os.makedirs('/tmp', exist_ok=True)
            faiss.write_index(self.index, '/tmp/vector_db.index')
            
            metadata = {
                'chunks': chunks,
                'document_metadata': self.document_metadata,
                'keyword_index': getattr(self, 'keyword_index', {})
            }
            with open('/tmp/vector_db.index.metadata.pkl', 'wb') as f:
                pickle.dump(metadata, f)
            
            print(f"✅ Created hybrid database with {len(chunks)} chunks")
            self._print_statistics()
            return True
            
        except Exception as e:
            print(f"❌ Error creating database: {e}")
            return False

    def _build_keyword_index(self):
        """Build simple keyword index for hybrid search"""
        print("🔍 Building keyword index...")
        self.keyword_index = defaultdict(list)
        
        for idx, chunk in enumerate(self.chunks):
            words = set(re.findall(r'\b\w+\b', chunk['text'].lower()))
            for word in words:
                if len(word) > 2:  # Ignore very short words
                    self.keyword_index[word].append(idx)
    
    def _add_document_metadata(self, chunks, document_boundaries):
        """Add document tracking metadata"""
        doc_stats = {}
        
        for chunk in chunks:
            chunk_page = chunk['page_number']
            for boundary in document_boundaries:
                if boundary['start_page'] <= chunk_page <= boundary['end_page']:
                    doc_name = boundary['file']
                    chunk['document'] = doc_name
                    
                    if doc_name not in doc_stats:
                        doc_stats[doc_name] = {'chunk_count': 0, 'pages': set()}
                    
                    doc_stats[doc_name]['chunk_count'] += 1
                    doc_stats[doc_name]['pages'].add(chunk['page_number'])
                    break
        
        self.document_metadata = doc_stats
    
    def _print_statistics(self):
        """Print database statistics"""
        print("📊 Database Statistics:")
        total_words = sum(len(re.findall(r'\b\w+\b', chunk['text'].lower())) for chunk in self.chunks)
        avg_chunk_size = sum(chunk['char_count'] for chunk in self.chunks) / len(self.chunks)
        
        print(f"   📄 Total chunks: {len(self.chunks)}")
        print(f"   📝 Total words: {total_words:,}")
        print(f"   📏 Average chunk size: {avg_chunk_size:.0f} chars")
        
        for doc, stats in self.document_metadata.items():
            print(f"   📑 {doc}: {stats['chunk_count']} chunks, {len(stats['pages'])} pages")
    
    def vector_search(self, query: str, top_k: int = 10, min_confidence: float = 0.2) -> List[Dict]:
        """Pure vector search"""
        self.search_stats['vector_searches'] += 1
        
        if not self.index:
            print("❌ Index not initialized")
            return []
        
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, top_k * 2)  # Get more for filtering
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks) and score >= min_confidence:
                chunk = self.chunks[idx]
                results.append({
                    'text': chunk['text'],
                    'source': chunk['source'],
                    'page_number': chunk['page_number'],
                    'document': chunk.get('document', 'unknown'),
                    'score': float(score),
                    'chunk_id': chunk['chunk_id'],
                    'search_type': 'vector'
                })
        
        return sorted(results, key=lambda x: x['score'], reverse=True)[:top_k]
    
    def keyword_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Pure keyword search"""
        self.search_stats['keyword_searches'] += 1
        
        query_terms = set(re.findall(r'\b\w+\b', query.lower()))
        chunk_scores = defaultdict(float)
        
        for term in query_terms:
            if term in self.keyword_index:
                for chunk_idx in self.keyword_index[term]:
                    # Simple TF-like scoring
                    chunk_text = self.chunks[chunk_idx]['text'].lower()
                    term_count = chunk_text.count(term)
                    chunk_scores[chunk_idx] += term_count / len(query_terms)
        
        results = []
        for chunk_idx, score in sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]:
            if score > 0:
                chunk = self.chunks[chunk_idx]
                results.append({
                    'text': chunk['text'],
                    'source': chunk['source'],
                    'page_number': chunk['page_number'],
                    'document': chunk.get('document', 'unknown'),
                    'score': float(score),
                    'chunk_id': chunk['chunk_id'],
                    'search_type': 'keyword'
                })
        
        return results
    
    def hybrid_search(self, query: str, top_k: int = 5, alpha: float = 0.7, min_confidence: float = 0.2) -> List[Dict]:
        """Hybrid search combining vector and keyword approaches"""
        self.search_stats['hybrid_searches'] += 1
        self.search_stats['total_searches'] += 1
        
        # Get results from both methods
        vector_results = self.vector_search(query, top_k * 2, min_confidence)
        keyword_results = self.keyword_search(query, top_k * 2)
        
        # Combine results
        combined_scores = {}
        
        # Add vector results
        for result in vector_results:
            chunk_id = result['chunk_id']
            combined_scores[chunk_id] = {
                'result': result,
                'vector_score': result['score'],
                'keyword_score': 0.0,
                'combined_score': result['score'] * alpha
            }
        
        # Add keyword results
        for result in keyword_results:
            chunk_id = result['chunk_id']
            if chunk_id in combined_scores:
                # Update existing entry
                combined_scores[chunk_id]['keyword_score'] = result['score']
                combined_scores[chunk_id]['combined_score'] = (
                    alpha * combined_scores[chunk_id]['vector_score'] + 
                    (1 - alpha) * result['score']
                )
            else:
                # New entry
                combined_scores[chunk_id] = {
                    'result': result,
                    'vector_score': 0.0,
                    'keyword_score': result['score'],
                    'combined_score': result['score'] * (1 - alpha)
                }
        
        # Sort by combined score and prepare final results
        final_results = []
        for chunk_data in sorted(combined_scores.values(), key=lambda x: x['combined_score'], reverse=True)[:top_k]:
            result = chunk_data['result'].copy()
            result['score'] = chunk_data['combined_score']
            result['vector_score'] = chunk_data['vector_score']
            result['keyword_score'] = chunk_data['keyword_score']
            result['search_type'] = 'hybrid'
            final_results.append(result)
        
        # Update statistics
        if final_results:
            avg_score = sum([r['score'] for r in final_results]) / len(final_results)
            self.search_stats['avg_scores'].append(avg_score)
        
        return final_results
    
    def search(self, query: str, top_k: int = 5, mode: str = 'hybrid', **kwargs) -> List[Dict]:
        """Unified search interface"""
        if mode == 'vector':
            return self.vector_search(query, top_k, **kwargs)
        elif mode == 'keyword':
            return self.keyword_search(query, top_k)
        else:  # hybrid
            return self.hybrid_search(query, top_k, **kwargs)
    
    def get_search_stats(self) -> Dict:
        """Get detailed search statistics"""
        if not self.search_stats['avg_scores']:
            return self.search_stats
            
        avg_confidence = sum(self.search_stats['avg_scores']) / len(self.search_stats['avg_scores'])
        
        return {
            'total_searches': self.search_stats['total_searches'],
            'vector_searches': self.search_stats['vector_searches'],
            'keyword_searches': self.search_stats['keyword_searches'],
            'hybrid_searches': self.search_stats['hybrid_searches'],
            'average_confidence': avg_confidence,
            'recent_searches': len(self.search_stats['avg_scores'])
        }

# ==================================================
# 🚀 ENHANCED CONVERSATION MANAGER
# ==================================================

class EnhancedConversationManager:
    def __init__(self, max_history=10):
        self.history = []
        self.max_history = max_history
        self.topic_tracking = defaultdict(int)
    
    def add_interaction(self, question: str, answer: str, sources: List[Dict]):
        """Add conversation interaction with topic tracking"""
        interaction = {
            'question': question,
            'answer': answer,
            'sources': sources,
            'timestamp': time.time(),
            'topics': self._extract_topics(question)
        }
        
        self.history.append(interaction)
        
        # Update topic frequencies
        for topic in interaction['topics']:
            self.topic_tracking[topic] += 1
        
        if len(self.history) > self.max_history:
            removed = self.history.pop(0)
            # Update topic frequencies for removed interaction
            for topic in removed['topics']:
                self.topic_tracking[topic] = max(0, self.topic_tracking[topic] - 1)
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract simple topics from text"""
        words = re.findall(r'\b\w+\b', text.lower())
        # Filter for meaningful words (simple approach)
        stop_words = {'what', 'how', 'why', 'when', 'where', 'which', 'the', 'a', 'an', 'is', 'are', 'can', 'you'}
        topics = [word for word in words if len(word) > 3 and word not in stop_words]
        return topics[:5]  # Return top 5 topics
    
    def get_recent_context(self, num_interactions: int = 3) -> str:
        """Get recent conversation context with topic awareness"""
        recent = self.history[-num_interactions:] if self.history else []
        
        context_lines = []
        for i, item in enumerate(recent):
            context_lines.append(f"Q{i+1}: {item['question']}")
            context_lines.append(f"A{i+1}: {item['answer']}")
        
        return "\n".join(context_lines) if context_lines else ""
    
    def get_conversation_topics(self) -> List[Tuple[str, int]]:
        """Get most frequent conversation topics"""
        return sorted(self.topic_tracking.items(), key=lambda x: x[1], reverse=True)[:5]
    
    def clear_history(self):
        """Clear conversation history"""
        self.history.clear()
        self.topic_tracking.clear()
        print("🗑️ Conversation history cleared!")

# ==================================================
# 🚀 ENHANCED ANSWER GENERATOR
# ==================================================

class EnhancedAnswerGenerator:
    def __init__(self, vector_db, pipeline):
        self.vector_db = vector_db
        self.pipeline = pipeline
        self.conversation = EnhancedConversationManager()
        self.performance_stats = []
        self.response_cache = {}
    
    def generate_answer(self, question: str, use_conversation_context: bool = True, 
                       max_length: int = 300, search_mode: str = 'hybrid') -> Dict:
        """Generate answer for any question with enhanced search"""
        start_time = time.time()
        
        # Simple cache check
        cache_key = hashlib.md5(question.lower().encode()).hexdigest()
        if cache_key in self.response_cache:
            cached = self.response_cache[cache_key].copy()
            cached['response_time'] = time.time() - start_time
            cached['cached'] = True
            return cached
        
        # Search for relevant content with chosen mode
        relevant_chunks = self.vector_db.search(question, top_k=4, mode=search_mode)
        
        if not relevant_chunks:
            response_data = {
                "question": question,
                "answer": "I couldn't find relevant information in the uploaded documents to answer this question.",
                "sources": [],
                "response_time": time.time() - start_time,
                "confidence": 0.0,
                "cached": False,
                "search_mode": search_mode
            }
        else:
            # Build enhanced context
            context = self._build_enhanced_context(relevant_chunks)
            
            # Add conversation context if needed
            conversation_context = ""
            if use_conversation_context:
                conversation_context = self.conversation.get_recent_context()
            
            # Generate answer
            prompt = self._create_enhanced_prompt(question, context, conversation_context)
            answer = self._generate_answer(prompt, max_length)
            
            # Calculate confidence
            confidence = sum([chunk['score'] for chunk in relevant_chunks]) / len(relevant_chunks)
            
            response_data = {
                "question": question,
                "answer": answer,
                "sources": [{
                    'source': chunk['source'],
                    'page_number': chunk['page_number'],
                    'document': chunk.get('document', 'unknown'),
                    'confidence': chunk['score'],
                    'search_type': chunk.get('search_type', 'unknown')
                } for chunk in relevant_chunks],
                "response_time": time.time() - start_time,
                "confidence": confidence,
                "cached": False,
                "search_mode": search_mode
            }
            
            # Cache the result
            self.response_cache[cache_key] = response_data.copy()
            self.response_cache[cache_key].pop('response_time', None)
            self.response_cache[cache_key].pop('cached', None)
        
        # Update conversation and stats
        self.conversation.add_interaction(question, response_data['answer'], response_data['sources'])
        self.performance_stats.append({
            'question': question,
            'response_time': response_data['response_time'],
            'confidence': response_data.get('confidence', 0.0),
            'search_mode': search_mode,
            'timestamp': time.time()
        })
        
        return response_data
    
    def _build_enhanced_context(self, chunks):
        """Build enhanced context with search type information"""
        # Group by document and search type
        documents = defaultdict(lambda: defaultdict(list))
        for chunk in chunks:
            doc = chunk.get('document', 'unknown')
            search_type = chunk.get('search_type', 'unknown')
            documents[doc][search_type].append(chunk)
        
        context_parts = []
        for doc_name, search_types in documents.items():
            context_parts.append(f"From {doc_name}:")
            
            for search_type, doc_chunks in search_types.items():
                if search_type != 'unknown':
                    context_parts.append(f"  [{search_type.upper()} MATCHES]:")
                
                for i, chunk in enumerate(doc_chunks[:2]):
                    context_parts.append(f"  - {chunk['text']}")
            
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def _create_enhanced_prompt(self, question, context, conversation_context):
        """Create enhanced prompt for LLM"""
        base_prompt = f"""<s>[INST] Using ONLY the information provided below, answer the question accurately and concisely. 
If the answer cannot be found in the provided information, clearly state you don't know.

CONTEXT INFORMATION:
{context}

"""
        if conversation_context:
            base_prompt += f"RECENT CONVERSATION CONTEXT:\n{conversation_context}\n\n"

        base_prompt += f"""QUESTION: {question}

Provide a clear, accurate answer based only on the context above. If different sources provide conflicting information, mention this.

ANSWER: [/INST]"""
        
        return base_prompt
    
    def _generate_answer(self, prompt, max_length):
        """Generate answer using LLM"""
        try:
            response = self.pipeline(
                prompt,
                max_new_tokens=max_length,
                do_sample=False,
                temperature=0.1,
                top_p=0.9,
                return_full_text=False
            )
            return response[0]['generated_text'].strip()
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def clean_generate_answer(self, question: str, search_mode: str = 'hybrid', **kwargs) -> Dict:
        """User-friendly answer generation with search mode selection"""
        print(f"🤔 Question: {question}")
        print(f"🔍 Search Mode: {search_mode.upper()}")
        print("⏳ Generating answer...")
        
        result = self.generate_answer(question, search_mode=search_mode, **kwargs)
        
        if result.get('cached'):
            print("💾 (Cached Response)")
        
        print(f"💡 ANSWER: {result['answer']}")
        print(f"⏱️ Response Time: {result['response_time']:.2f}s")
        
        if result['sources']:
            print("📚 SOURCES:")
            for source in result['sources']:
                doc_info = f" ({source['document']})" if source.get('document') else ""
                search_type = f" [{source.get('search_type', 'unknown')}]" if source.get('search_type') else ""
                print(f"   - {source['source']} Page {source['page_number']}{doc_info}{search_type} - Confidence: {source['confidence']:.3f}")
        
        if result.get('confidence', 0) > 0:
            print(f"🎯 Confidence: {result['confidence']:.3f}")
        
        return result
    
    def compare_search_modes(self, question: str) -> Dict:
        """Compare different search modes for the same question"""
        print(f"🔍 COMPARING SEARCH MODES FOR: {question}")
        print("=" * 50)
        
        results = {}
        for mode in ['vector', 'keyword', 'hybrid']:
            print(f"\n{mode.upper()} SEARCH:")
            result = self.clean_generate_answer(question, search_mode=mode)
            results[mode] = result
        
        return results
    
    def get_system_stats(self) -> Dict:
        """Get comprehensive system statistics"""
        search_stats = self.vector_db.get_search_stats()
        
        if self.performance_stats:
            avg_response_time = sum([p['response_time'] for p in self.performance_stats]) / len(self.performance_stats)
            avg_confidence = sum([p['confidence'] for p in self.performance_stats]) / len(self.performance_stats)
            
            # Mode usage statistics
            mode_usage = defaultdict(int)
            for stat in self.performance_stats:
                mode_usage[stat.get('search_mode', 'unknown')] += 1
        else:
            avg_response_time = avg_confidence = 0
            mode_usage = {}
        
        return {
            'search_statistics': search_stats,
            'performance_metrics': {
                'total_questions_answered': len(self.performance_stats),
                'conversation_history_length': len(self.conversation.history),
                'avg_response_time': avg_response_time,
                'avg_confidence': avg_confidence,
                'cache_size': len(self.response_cache),
                'search_mode_usage': dict(mode_usage)
            },
            'conversation_analytics': {
                'top_topics': self.conversation.get_conversation_topics(),
                'recent_questions': [h['question'] for h in self.conversation.history[-3:]]
            }
        }
    
    def clear_cache(self):
        """Clear response cache"""
        self.response_cache.clear()
        print("🧹 Response cache cleared!")

# ==================================================
# 🚀 ANALYTICS MANAGER
# ==================================================

class AnalyticsManager:
    def __init__(self, answer_generator):
        self.answer_gen = answer_generator
        
    def generate_report(self):
        """Generate comprehensive analytics report"""
        stats = self.answer_gen.get_system_stats()
        
        print("\n" + "="*60)
        print("📈 KRISHN - COMPREHENSIVE ANALYTICS REPORT")
        print("="*60)
        
        # Performance Summary
        print("\n🎯 PERFORMANCE SUMMARY:")
        print(f"   Total Questions Answered: {stats['performance_metrics']['total_questions_answered']}")
        print(f"   Average Response Time: {stats['performance_metrics']['avg_response_time']:.2f}s")
        print(f"   Average Confidence: {stats['performance_metrics']['avg_confidence']:.3f}")
        print(f"   Cache Size: {stats['performance_metrics']['cache_size']} entries")
        
        # Search Statistics
        print("\n🔍 SEARCH STATISTICS:")
        search_stats = stats['search_statistics']
        print(f"   Total Searches: {search_stats['total_searches']}")
        print(f"   Vector Searches: {search_stats['vector_searches']}")
        print(f"   Keyword Searches: {search_stats['keyword_searches']}")
        print(f"   Hybrid Searches: {search_stats['hybrid_searches']}")
        
        # Search Mode Usage
        print("\n📊 SEARCH MODE USAGE:")
        for mode, count in stats['performance_metrics']['search_mode_usage'].items():
            percentage = (count / stats['performance_metrics']['total_questions_answered']) * 100
            print(f"   {mode.upper()}: {count} ({percentage:.1f}%)")
        
        # Conversation Analytics
        print("\n💬 CONVERSATION ANALYTICS:")
        print(f"   Active Topics: {len(stats['conversation_analytics']['top_topics'])}")
        if stats['conversation_analytics']['top_topics']:
            print("   Top Topics:")
            for topic, count in stats['conversation_analytics']['top_topics']:
                print(f"     - {topic}: {count} mentions")
        
        # Recent Activity
        print("\n🕒 RECENT ACTIVITY:")
        recent_questions = stats['conversation_analytics']['recent_questions']
        if recent_questions:
            for i, question in enumerate(recent_questions, 1):
                print(f"   {i}. {question[:60]}{'...' if len(question) > 60 else ''}")
        else:
            print("   No recent activity")
        
        print("\n" + "="*60)
        
        return stats