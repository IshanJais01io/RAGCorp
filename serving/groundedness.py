import re
from typing import List, Dict

class GroundednessVerifier:
    def verify(self, answer: str, contexts: List[Dict]) -> Dict:
        if "not found in the indexed documents" in answer.lower():
            return {"is_grounded": True, "answer": answer, "violations": []}

        context_dict = {c["child_id"]: c["parent_text"] for c in contexts}
        citation_pattern = re.compile(r"\[Chunk\s+([a-zA-Z0-9_]+)\]")
        sentences = [s.strip() for s in answer.split(".") if s.strip()]
        
        violations = []
        for sentence in sentences:
            citations = citation_pattern.findall(sentence)
            if not citations:
                violations.append({"sentence": sentence, "reason": "Missing citation tag"})
                continue
            
            clean_sentence = citation_pattern.sub("", sentence).strip().lower()
            grounded = False
            for cite_id in citations:
                if cite_id in context_dict:
                    parent_text = context_dict[cite_id].lower()
                    sentence_words = set(clean_sentence.split())
                    if not sentence_words:
                        grounded = True
                        break
                    match_ratio = sum(1 for w in sentence_words if w in parent_text) / len(sentence_words)
                    if match_ratio > 0.5:
                        grounded = True
                        break
            
            if not grounded:
                violations.append({"sentence": sentence, "reason": "Claim unsupported by cited context"})

        if violations:
            return {
                "is_grounded": False,
                "answer": answer + "\n\n⚠️ *Warning: Some statements could not be verified against the source text.*",
                "violations": violations
            }
            
        return {"is_grounded": True, "answer": answer, "violations": []}