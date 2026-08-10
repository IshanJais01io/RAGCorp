import os
import sys
import gradio as gr

RAGCORP_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RAGCORP_PATH not in sys.path:
    sys.path.insert(0, RAGCORP_PATH)

from serving.generation import generate_answer, retrieve_context

def respond(user_query, history):
    if not user_query or not user_query.strip():
        return history, ""

    if history is None:
        history = []

    retrieved_chunks = retrieve_context(user_query, top_k=4)
    if not retrieved_chunks:
        bot_response = "The requested information is not found in the indexed documents."
        history.append((user_query, bot_response))
        return history, ""

    answer, sources = generate_answer(user_query, retrieved_chunks)
    history.append((user_query, answer))
    return history, ""

with gr.Blocks(title="RAGCorp — Enterprise Multi-Modal RAG Engine") as demo:
    gr.Markdown("# RAGCorp — Enterprise Multi-Modal RAG Engine")
    
    chatbot = gr.Chatbot(elem_id="chatbot", label="RAGCorp Assistant", type="tuples")
    msg = gr.Textbox(placeholder="Ask a question about your indexed PDF documents...", container=False)
    
    with gr.Row():
        submit_btn = gr.Button("Submit", variant="primary")
        clear_btn = gr.Button("Clear Chat")

    msg.submit(respond, [msg, chatbot], [chatbot, msg])
    submit_btn.click(respond, [msg, chatbot], [chatbot, msg])
    clear_btn.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(share=True, debug=True)
