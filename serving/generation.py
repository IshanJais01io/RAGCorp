import os
import re
from groq import Groq

def retrieve_context(query, top_k=12):
    import chromadb
    
    possible_paths = [
        os.path.join(os.environ.get("OUTPUT_INDEX_DIR", ""), "chroma"),
        "/content/RAGCorp/indices_output/chroma",
        "/content/indices_output/chroma"
    ]
    
    chroma_path = None
    for p in possible_paths:
        if p and os.path.exists(p):
            chroma_path = p
            break
            
    if not chroma_path:
        print("⚠️ Warning: Chroma index path not found.")
        return []
        
    try:
        c_client = chromadb.PersistentClient(path=chroma_path)
        coll = c_client.get_collection("document_chunks")
        
        keywords = re.findall(r'\w+', query.lower())
        search_terms = [query]
        if any(k in keywords for k in ["summary", "report", "diagram", "chart", "figure", "table", "architecture"]):
            search_terms.append(f"{query} overview key findings visual diagram breakdown")

        chunks_dict = {}
        for st in search_terms:
            results = coll.query(query_texts=[st], n_results=top_k)
            if results and results.get("documents") and results["documents"][0]:
                for doc, meta, cid in zip(results["documents"][0], results["metadatas"][0], results["ids"][0]):
                    if cid not in chunks_dict:
                        chunks_dict[cid] = {"id": cid, "text": doc, "metadata": meta or {}}
                        
        return list(chunks_dict.values())[:top_k]
    except Exception as e:
        print(f"Retrieval error: {e}")
        return []

def generate_answer(query, retrieved_chunks):
    if not retrieved_chunks:
        return "The requested information is not found in the indexed documents.", []

    try:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            return "Error: GROQ_API_KEY environment variable is not set.", []
            
        client = Groq(api_key=api_key)
        
        # Build Context String with Metadata details
        context_blocks = []
        extracted_visuals = []
        
        for c in retrieved_chunks:
            meta = c.get("metadata", {})
            page_info = f" (Page {meta.get('page_number', 'N/A')})" if meta.get('page_number') else ""
            context_blocks.append(f"[Chunk {c['id']}{page_info}]\n{c['text']}")
            
            # Extract Multimodal Assets (Images / Diagrams / Charts)
            img_path = meta.get("image_path") or meta.get("fig_path") or meta.get("image")
            if img_path and os.path.exists(img_path):
                caption = meta.get("caption") or meta.get("image_caption") or f"Visual Document Asset"
                page = meta.get("page_number", "N/A")
                if img_path not in [v["path"] for v in extracted_visuals]:
                    extracted_visuals.append({"path": img_path, "caption": caption, "page": page})

        context_str = "\n\n---\n\n".join(context_blocks)
        
        system_prompt = (
            "You are RAGCorp's advanced AI Intelligence Assistant, built to deliver fluid, articulate, and deeply synthesized responses in the style of Google Gemini.\n\n"
            "GUIDELINES FOR RESPONSE GENERATION:\n"
            "1. **Conversational & Synthesized**: Do not simply quote or list chunks verbatim. Synthesize facts across all provided chunks into a coherent, highly articulate explanation.\n"
            "2. **Rich Markdown Structure**: Structure your response cleanly using:\n"
            "   - Markdown Headings (`### Key Concepts`, `### Analysis & Findings`)\n"
            "   - Bold highlights for core terminology\n"
            "   - Itemized Bullet Points and Comparison Tables where structured data or multiple items exist.\n"
            "3. **Inline Citations**: Every single factual claim must end with a precise citation using the exact chunk format `[Chunk chunk_id]`.\n"
            "4. **Strict Grounding**: Rely purely on the provided context. If the requested information is absent, state strictly: 'The requested information is not found in the indexed documents.'"
        )

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Document Context:\n{context_str}\n\nUser Query: {query}"}
            ],
            temperature=0.2
        )
        
        answer = response.choices[0].message.content
        
        # Append Multimodal Visual Assets Inline
        if extracted_visuals:
            answer += "\n\n### 📊 Relevant Diagrams & Visual Assets\n"
            for vis in extracted_visuals:
                answer += f"![{vis['caption']}]({vis['path']})\n*Figure: {vis['caption']} (Extracted from Page {vis['page']})*\n\n"

        return answer, [c["id"] for c in retrieved_chunks]
    except Exception as e:
        print(f"Generation error: {e}")
        return f"Error generating response: {str(e)}", []
