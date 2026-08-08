import os
import gradio as gr
from sentence_transformers import SentenceTransformer
from shared.config import TEXT_EMBED_MODEL, CHROMA_PERSIST_DIR
from serving.router import QueryIntentRouter
from serving.retriever import DualRetriever
from serving.reranker import FusionReranker
from serving.generator import GroqGenerator
from serving.groundedness import GroundednessVerifier

# Initialize Pipeline Engine
text_encoder = SentenceTransformer(TEXT_EMBED_MODEL)
router = QueryIntentRouter(text_encoder)
reranker = FusionReranker()
generator = GroqGenerator()
verifier = GroundednessVerifier()

# Lazy-loaded retriever
retriever = None

def get_retriever():
    global retriever
    if retriever is None and os.path.exists(CHROMA_PERSIST_DIR):
        retriever = DualRetriever(CHROMA_PERSIST_DIR, text_encoder)
    return retriever

def chat_pipeline(message, history):
    active_retriever = get_retriever()
    if not active_retriever:
        return "Vector index not found. Please run Phase 1 Ingestion first."

    # 1. Intent Routing
    route = router.route(message)
    
    # 2. Retrieval
    candidates = active_retriever.search_text(message, top_k=10)
    
    # 3. Reranking
    ranked_chunks = reranker.rerank(message, candidates, top_k=5)
    
    # 4. Generation via Groq
    raw_answer = generator.generate_answer(message, ranked_chunks)
    
    # 5. Groundedness Verification
    verification = verifier.verify(raw_answer, ranked_chunks)
    
    return verification["answer"]

# Gradio Interface Setup
demo = gr.ChatInterface(
    fn=chat_pipeline,
    title="RAGCorp — Enterprise Multi-Modal RAG Engine",
    description="Ask questions about your PDF documents. Answers include verified inline citations.",
    examples=["What was the Q3 revenue mentioned in the report?", "Summarize the key findings from the scanned document."]
)

if __name__ == "__main__":
    demo.launch()