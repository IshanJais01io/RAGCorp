import os
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def retrieve_context(query, top_k=4):
    import chromadb
    from shared.config import OUTPUT_INDEX_DIR
    
    chroma_path = os.path.join(OUTPUT_INDEX_DIR, "chroma")
    if not os.path.exists(chroma_path):
        chroma_path = "/content/RAGCorp/indices_output/chroma"
        
    c_client = chromadb.PersistentClient(path=chroma_path)
    try:
        coll = c_client.get_collection("document_chunks")
        results = coll.query(query_texts=[query], n_results=top_k)
        chunks = []
        if results and results.get("documents"):
            for doc, meta, cid in zip(results["documents"][0], results["metadatas"][0], results["ids"][0]):
                chunks.append({"id": cid, "text": doc, "metadata": meta})
        return chunks
    except Exception:
        return []

def generate_answer(query, retrieved_chunks):
    if not retrieved_chunks:
        return "The requested information is not found in the indexed documents.", []

    context_str = "\n\n".join([f"[{c['id']}] {c['text']}" for c in retrieved_chunks])
    
    system_prompt = (
        "You are an expert enterprise analytical assistant. "
        "Answer the user query strictly using the provided document context. "
        "Do not mention chunk IDs, metadata names, or database structures in your response text. "
        "If the context does not contain the answer, reply strictly with: 'The requested information is not found in the documents. Ask something else please!'"
        "Do NOT make assumptions or use outside knowledge."
        "You can use visuals, tables, and code snippets if they are present in the context. If not present everytime just make your own meaingfull using the details in the context."
        "Visuals are needed after every answer, it can be of any type, choose accordingly."
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
