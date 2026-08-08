import os

# Project Metadata
PROJECT_NAME = "RAGCorp"

# Hugging Face & Groq Settings
HF_DATASET_REPO_ID = os.getenv("HF_DATASET_REPO_ID", "your-username/ragcorp-indices")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"

# Paths
OUTPUT_INDEX_DIR = os.getenv("OUTPUT_INDEX_DIR", "indices_output")
CHROMA_PERSIST_DIR = os.path.join(OUTPUT_INDEX_DIR, "chroma")

# Ingestion Parameters
CHUNK_CHILD_SIZE = 128
CHUNK_PARENT_SIZE = 1024
CHUNK_OVERLAP = 32
INGESTION_BATCH_SIZE = 50

# Embedding & Reranker Models
TEXT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"