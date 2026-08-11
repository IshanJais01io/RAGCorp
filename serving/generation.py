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
        if "security" in keywords or "bugs" in keywords or "report" in keywords:
            search_terms.append(f"{query} code review vulnerability testing report summary")

        chunks_dict = {}
        for st in search_terms:
            results = coll.query(query_texts=[st], n_results=top_k)
            if results and results.get("documents") and results["documents"][0]:
                for doc, meta, cid in zip(results["documents"][0], results["metadatas"][0], results["ids"][0]):
                    if cid not in chunks_dict:
                        chunks_dict[cid] = {"id": cid, "text": doc, "metadata": meta}
                        
        chunks = list(chunks_dict.values())[:top_k]
        return chunks
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
        context_str = "\n\n".join([f"[{c['id']}] {c['text']}" for c in retrieved_chunks])
        
        system_prompt = (
            "You are an expert enterprise analytical assistant for RAGCorp. "
            "Answer the user query strictly using the provided document context. "
            "Every factual claim must end with an inline citation using the format [Chunk chunk_id]. "
            "If the user asks for a count, summary, or specific details, thoroughly review all provided context chunks. "
            "Do not mention internal database mechanics in your conversational text. "
            "If the context truly does not contain the answer, reply strictly with: 'The requested information is not found in the indexed documents.'"
        )

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context_str}\n\nQuery: {query}"}
            ],
            temperature=0.1
        )
        
        return response.choices[0].message.content, [c["id"] for c in retrieved_chunks]
    except Exception as e:
        print(f"Generation error: {e}")
        return f"Error generating response: {str(e)}", []
