import os
from groq import Groq

def retrieve_context(query, top_k=4):
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
        results = coll.query(query_texts=[query], n_results=top_k)
        chunks = []
        if results and results.get("documents") and results["documents"][0]:
            for doc, meta, cid in zip(results["documents"][0], results["metadatas"][0], results["ids"][0]):
                chunks.append({"id": cid, "text": doc, "metadata": meta})
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
            "You are an expert enterprise analytical assistant. "
            "Answer the user query strictly using the provided document context. "
            "Every factual claim must end with an inline citation using the format [Chunk chunk_id]. "
            "Do not mention chunk IDs, metadata, or database structures in your conversational text. "
            "If the context does not contain the answer, reply strictly with: 'The requested information is not found in the indexed documents.'"
        )

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context_str}\n\nQuery: {query}"}
            ],
            temperature=0.2
        )
        
        return response.choices[0].message.content, [c["id"] for c in retrieved_chunks]
    except Exception as e:
        print(f"Generation error: {e}")
        return f"Error generating response: {str(e)}", []
