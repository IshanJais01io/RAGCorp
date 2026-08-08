import os
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict

class DualRetriever:
    def __init__(self, index_path: str, text_encoder: SentenceTransformer):
        self.chroma_client = chromadb.PersistentClient(path=index_path)
        self.collection = self.chroma_client.get_collection(name="document_chunks")
        self.encoder = text_encoder

    def search_text(self, query: str, top_k: int = 10) -> List[Dict]:
        query_vector = self.encoder.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_vector,
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        retrieved = []
        if results and results["ids"] and results["ids"][0]:
            for idx in range(len(results["ids"][0])):
                doc_id = results["ids"][0][idx]
                distance = results["distances"][0][idx]
                metadata = results["metadatas"][0][idx]
                
                # Convert cosine distance to similarity score
                similarity = 1.0 - distance
                
                retrieved.append({
                    "child_id": doc_id,
                    "child_text": results["documents"][0][idx],
                    "parent_text": metadata.get("parent_text", results["documents"][0][idx]),
                    "parent_id": metadata.get("parent_id", ""),
                    "score": similarity,
                    "metadata": metadata
                })
        return retrieved