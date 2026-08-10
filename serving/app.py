import os
import sys
import gradio as gr

RAGCORP_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RAGCORP_PATH not in sys.path:
    sys.path.insert(0, RAGCORP_PATH)

from serving.generation import generate_answer, retrieve_context

def predict(message, history):
    if not message or not message.strip():
        return ""

    retrieved_chunks = retrieve_context(message, top_k=4)
    if not retrieved_chunks:
        return "The requested information is not found in the indexed documents."

    answer, sources = generate_answer(message, retrieved_chunks)
    return answer

demo = gr.ChatInterface(
    fn=predict,
    title="RAGCorp — Enterprise Multi-Modal RAG Engine",
    description="Ask questions about your indexed PDF documents. Answers include verified inline citations.",
    textbox=gr.Textbox(placeholder="Ask a question about your indexed PDF documents...", container=False, scale=7),
)

if __name__ == "__main__":
    demo.launch(share=True, debug=True)
