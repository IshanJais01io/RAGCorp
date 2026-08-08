import os
import glob
import json
import datetime
import fitz
import chromadb
from sentence_transformers import SentenceTransformer
from huggingface_hub import HfApi
from shared.schema import IngestionManifest
from shared.config import OUTPUT_INDEX_DIR, CHUNK_CHILD_SIZE, CHUNK_PARENT_SIZE
from ingestion.extractors import MultiModalExtractor
from ingestion.chunker import HierarchicalChunker

DRIVE_CHECKPOINT_DIR = "/content/drive/MyDrive/rag_checkpoint"
MANIFEST_PATH = os.path.join(DRIVE_CHECKPOINT_DIR, "manifest.json")

class IngestionPipeline:
    def __init__(self, hf_repo_id: str, batch_size: int = 50):
        self.batch_size = batch_size
        self.hf_repo_id = hf_repo_id
        self.extractor = MultiModalExtractor(use_gpu=True)
        self.chunker = HierarchicalChunker(child_size=CHUNK_CHILD_SIZE, parent_size=CHUNK_PARENT_SIZE)
        self.text_encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
        
        os.makedirs(OUTPUT_INDEX_DIR, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=os.path.join(OUTPUT_INDEX_DIR, "chroma"))
        self.collection = self.chroma_client.get_or_create_collection(
            name="document_chunks",
            metadata={"hnsw:space": "cosine"}
        )

    def load_manifest(self) -> IngestionManifest:
        if os.path.exists(MANIFEST_PATH):
            with open(MANIFEST_PATH, "r") as f:
                return IngestionManifest(**json.load(f))
        return IngestionManifest(last_updated=str(datetime.datetime.now()))

    def save_manifest(self, manifest: IngestionManifest):
        os.makedirs(DRIVE_CHECKPOINT_DIR, exist_ok=True)
        manifest.last_updated = str(datetime.datetime.now())
        with open(MANIFEST_PATH, "w") as f:
            json.dump(manifest.dict(), f, indent=2)

    def sync_to_huggingface(self):
        api = HfApi()
        api.upload_folder(
            folder_path=OUTPUT_INDEX_DIR,
            repo_id=self.hf_repo_id,
            repo_type="dataset",
            commit_message=f"Automated index update {datetime.datetime.now()}"
        )

    def run(self, pdf_folder_path: str):
        manifest = self.load_manifest()
        pdf_files = glob.glob(os.path.join(pdf_folder_path, "*.pdf"))
        unprocessed = [f for f in pdf_files if f not in manifest.processed_files]

        current_batch_chunks = []
        
        for idx, pdf_path in enumerate(unprocessed):
            file_name = os.path.basename(pdf_path)
            try:
                doc = fitz.open(pdf_path)
                for page_num in range(len(doc)):
                    page_type, text, _ = self.extractor.process_pdf_page(doc[page_num], page_num)
                    if not text.strip():
                        continue

                    metadata = {"source_file": file_name, "page_num": page_num + 1, "page_type": page_type}
                    chunks = self.chunker.chunk_document(text, metadata)
                    current_batch_chunks.extend(chunks)

                manifest.processed_files.append(pdf_path)

            except Exception as e:
                manifest.failed_files[file_name] = str(e)

            if (idx + 1) % self.batch_size == 0 or (idx + 1) == len(unprocessed):
                if current_batch_chunks:
                    texts = [c.child_text for c in current_batch_chunks]
                    ids = [c.child_id for c in current_batch_chunks]
                    embeddings = self.text_encoder.encode(texts, show_progress_bar=False).tolist()
                    metadatas = [{**c.metadata, "parent_id": c.parent_id, "parent_text": c.parent_text} for c in current_batch_chunks]

                    self.collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
                    manifest.total_chunks += len(current_batch_chunks)
                    current_batch_chunks = []

                self.save_manifest(manifest)
                self.sync_to_huggingface()