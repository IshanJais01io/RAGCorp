import uuid
from typing import List
from shared.schema import DocumentChunk

class HierarchicalChunker:
    def __init__(self, child_size: int = 128, parent_size: int = 1024, overlap: int = 32):
        self.child_size = child_size
        self.parent_size = parent_size
        self.overlap = overlap

    def chunk_document(self, text: str, base_metadata: dict) -> List[DocumentChunk]:
        words = text.split()
        if not words:
            return []

        chunks: List[DocumentChunk] = []

        for p_idx in range(0, len(words), self.parent_size - self.overlap):
            parent_words = words[p_idx:p_idx + self.parent_size]
            parent_text = " ".join(parent_words)
            parent_id = f"parent_{uuid.uuid4().hex[:10]}"

            for c_idx in range(0, len(parent_words), self.child_size - self.overlap):
                child_words = parent_words[c_idx:c_idx + self.child_size]
                child_text = " ".join(child_words).strip()
                
                if not child_text:
                    continue

                child_id = f"child_{uuid.uuid4().hex[:10]}"
                chunk_meta = {**base_metadata, "parent_id": parent_id, "word_count": len(child_words)}

                chunks.append(
                    DocumentChunk(
                        child_id=child_id,
                        parent_id=parent_id,
                        child_text=child_text,
                        parent_text=parent_text,
                        metadata=chunk_meta
                    )
                )

        return chunks