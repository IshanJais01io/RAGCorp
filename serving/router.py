import re
import torch
from sentence_transformers import SentenceTransformer, util

class QueryIntentRouter:
    def __init__(self, text_encoder: SentenceTransformer):
        self.encoder = text_encoder
        self.visual_keywords = re.compile(
            r"\b(chart|graph|plot|diagram|table|figure|map|image|picture|visual|trend|drawing|scan)\b",
            re.IGNORECASE
        )
        self.visual_exemplars = [
            "Show me the chart depicting sales growth",
            "What does the table say about revenue in Q3?",
            "Visual trend of population decrease over time",
            "Diagram of the hydraulic engine system",
            "Data shown in the figure on page 4"
        ]
        self.exemplar_embeds = self.encoder.encode(self.visual_exemplars, convert_to_tensor=True)

    def route(self, query: str, threshold: float = 0.55) -> str:
        if self.visual_keywords.search(query):
            return "VISUAL"
        
        query_embed = self.encoder.encode(query, convert_to_tensor=True)
        sims = util.cos_sim(query_embed, self.exemplar_embeds)
        max_sim = float(torch.max(sims))
        
        if max_sim >= threshold:
            return "VISUAL"
        return "TEXT"