# Gradio web app for asking questions about a PDF or TXT document.

import gradio as gr

from src.answerer import Answerer
from src.chunker import chunk_pages
from src.embedder import Embedder
from src.loader import load_document
from src.retriever import Retriever

# Load both models once when the app starts, not on every question
embedder = Embedder()
answerer = Answerer()


def process_document(file_path, chunk_size, overlap):
    # Read the file, split it into chunks and build the search index
    if file_path is None:
        return None, "Please upload a PDF or TXT file first."
    try:
        pages = load_document(file_path)
        chunks = chunk_pages(pages, int(chunk_size), int(overlap))
        retriever = Retriever(embedder)
        retriever.build(chunks)
    except (ValueError, FileNotFoundError) as error:
        # Show a readable message instead of crashing the app
        return None, f"Error: {error}"
    return retriever, f"Ready: {len(pages)} page(s) split into {len(chunks)} chunks."


def ask_question(question, retriever, top_k):
    # Find the best chunks, then pull the answer out of them
    if retriever is None:
        return "Please process a document first.", "", ""
    try:
        hits = retriever.search(question, int(top_k))
        result = answerer.answer(question, hits)
    except (ValueError, RuntimeError) as error:
        return f"Error: {error}", "", ""

    if result["page"] is None:
        return result["answer"], "", ""
    source = f"Page {result['page']} | confidence {result['score']:.2f}"
    return result["answer"], source, result["context"]


with gr.Blocks(title="Document Q&A") as demo:
    gr.Markdown("# Document Q&A\nUpload a PDF or TXT file and ask questions about it.")

    # The retriever is kept per user session in this state object
    retriever_state = gr.State(None)

    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Document", file_types=[".pdf", ".txt"], type="filepath")
            chunk_size = gr.Slider(20, 300, value=40, step=10, label="Chunk size (words)")
            overlap = gr.Slider(0, 100, value=10, step=5, label="Overlap (words)")
            process_button = gr.Button("Process document")
            status = gr.Textbox(label="Status", interactive=False)

        with gr.Column():
            question = gr.Textbox(label="Your question", placeholder="Ask something about the document")
            top_k = gr.Slider(1, 5, value=5, step=1, label="Chunks to search")
            ask_button = gr.Button("Get answer", variant="primary")
            answer = gr.Textbox(label="Answer", interactive=False)
            source = gr.Textbox(label="Source", interactive=False)
            context = gr.Textbox(label="Source text", lines=6, interactive=False)

    process_button.click(
        process_document,
        inputs=[file_input, chunk_size, overlap],
        outputs=[retriever_state, status],
    )
    ask_button.click(
        ask_question,
        inputs=[question, retriever_state, top_k],
        outputs=[answer, source, context],
    )

if __name__ == "__main__":
    demo.launch()
