from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class ProcessedPage(BaseModel):
    doc_id: str
    page_num: int
    page_type: str  # "digital" or "scanned"
    raw_text: str
    image_path: Optional[str] = None
    has_tables_or_charts: bool = False

class DocumentChunk(BaseModel):
    child_id: str
    parent_id: str
    child_text: str
    parent_text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class IngestionManifest(BaseModel):
    processed_files: List[str] = Field(default_factory=list)
    failed_files: Dict[str, str] = Field(default_factory=dict)
    total_chunks: int = 0
    last_updated: str