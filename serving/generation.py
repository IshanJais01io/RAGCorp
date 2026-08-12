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
        if any(k in keywords for k in ["image", "images", "picture", "photo", "diagram", "chart", "figure", "visual"]):
            search_terms.append(f"{query} image picture diagram chart figure visual photo illustration page")

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
        
        context_blocks = []
        extracted_visuals = []
        
        for c in retrieved_chunks:
            meta = c.get("metadata", {})
            context_blocks.append(f"{c['text']}")
            
            # Extract Image Path Metadata
            img_path = meta.get("image_path") or meta.get("fig_path") or meta.get("image") or meta.get("img_path")
            if img_path and os.path.exists(str(img_path)):
                caption = meta.get("caption") or meta.get("image_caption") or f"Visual Document Asset"
                page = meta.get("page_number", "N/A")
                if img_path not in [v["path"] for v in extracted_visuals]:
                    extracted_visuals.append({"path": str(img_path), "caption": caption, "page": page})

        context_str = "\n\n---\n\n".join(context_blocks)
        
        system_prompt = (
            "You are RAGCorp's advanced AI Intelligence Assistant, built to deliver fluid, articulate, and deeply synthesized responses in the style of Google Gemini.\n\n"
            "STRICT FORMATTING RULES:\n"
            "1. **NEVER SHOW CHUNK DETAILS**: Do NOT include chunk IDs, child IDs, metadata tags, or internal database citations (such as [Chunk ...] or [child_...]) anywhere in your response.\n"
            "2. **Conversational & Synthesized**: Synthesize facts across all provided context into a clean, comprehensive, professional Markdown response.\n"
            "3. **Rich Structure**: Use headings (`###`), bold highlights, bullet lists, and markdown tables where relevant.\n"
            "4. **Strict Grounding**: Base your answer solely on the provided context. If the answer is truly missing, reply strictly: 'The requested information is not found in the indexed documents.'"
        )

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Document Context:\n{context_str}\n\nUser Query: {query}"}
            ],
            temperature=0.2
        )
        
        raw_answer = response.choices[0].message.content
        
        # Regex Filter to strictly eliminate any leftover chunk tags
        clean_answer = re.sub(r'\[\s*Chunk\s+[^\]]+\\]', '', raw_answer, flags=re.IGNORECASE)
        clean_answer = re.sub(r'\[\s*child_[a-f0-9]+\s*\]', '', clean_answer, flags=re.IGNORECASE)
        clean_answer = re.sub(r'\[\s*Chunk\s*\]', '', clean_answer, flags=re.IGNORECASE)
        clean_answer = re.sub(r'  +', ' ', clean_answer)

        # Append Extracted Visual Assets if present
        if extracted_visuals:
            clean_answer += "\n\n### 📊 Document Images & Visual Assets\n"
            for vis in extracted_visuals:
                clean_answer += f"![{vis['caption']}]({vis['path']})\n*Figure: {vis['caption']} (Page {vis['page']})*\n\n"

        return clean_answer.strip(), [c["id"] for c in retrieved_chunks]
    except Exception as e:
        print(f"Generation error: {e}")
        return f"Error generating response: {str(e)}", []
