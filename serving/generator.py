import os
from typing import List, Dict
from groq import Groq
from shared.config import GROQ_MODEL

SYSTEM_PROMPT = """You are RAGCorp's multi-modal intelligence assistant.
Answer the user's question strictly using ONLY the provided document context blocks.
Answer the user query strictly using the provided context.
Do not mention chunk IDs, metadata names, or database structures in your response text.

RULES:
1. If the answer cannot be directly derived from the context, respond strictly with:
   "The requested information is not found in the documents."
2. Do NOT make assumptions or use outside knowledge.
3. You can use visuals, tables, and code snippets if they are present in the context. If not present everytime just make your own meaingfull using the details in the context.
4. Visuals are needed after every answer, it can be of any type(choose accordingly).
"""

class GroqGenerator:
    def __init__(self, api_key: str = None):
        key = api_key or os.environ.get("GROQ_API_KEY")
        self.client = Groq(api_key=key) if key else None

    def generate_answer(self, query: str, contexts: List[Dict]) -> str:
        if not self.client:
            return "Error: Groq API key is not configured."
        if not contexts:
            return "The requested information is not found in the indexed documents."

        formatted_context = ""
        for c in contexts:
            formatted_context += f"\n--- START CHUNK ID: {c['child_id']} (Page {c['metadata'].get('page_num', 'N/A')}) ---\n"
            formatted_context += f"{c['parent_text']}\n"
            formatted_context += f"--- END CHUNK ID: {c['child_id']} ---\n"

        user_content = f"CONTEXT:\n{formatted_context}\n\nQUESTION: {query}"

        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            temperature=0.0,
            max_tokens=1024
        )
        return response.choices[0].message.content
