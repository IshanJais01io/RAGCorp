from typing import List, Dict
from sentence_transformers import CrossEncoder
from shared.config import RERANKER_MODEL

class FusionReranker:
    def __init__(self, model_name: str = RERANKER_MODEL):
        self.reranker = CrossEncoder(model_name)

    def reciprocal_rank_fusion(self, results_list: List[List[Dict]], k: int = 60) -> List[Dict]:
        rrf_scores = {}
        doc_map = {}
        
        for results in results_list:
            for rank, doc in enumerate(results):
                doc_id = doc["child_id"]
                doc_map[doc_id] = doc
                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = 0.0
                rrf_scores[doc_id] += 1.0 / (k + rank + 1)
                
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[doc_id] for doc_id, _ in sorted_docs]

    def rerank(self, query: str, candidate_docs: List[Dict], top_k: int = 5) -> List[Dict]:
        if not candidate_docs:
            return []
        
        pairs = [[query, doc["parent_text"]] for doc in candidate_docs]
        scores = self.reranker.predict(pairs)
        
        for idx, score in enumerate(scores):
            candidate_docs[idx]["rerank_score"] = float(score)
            
        ranked = sorted(candidate_docs, key=lambda x: x["rerank_score"], reverse=True)
        return ranked[:top_k]