import gradio as gr
from langchain_google_genai import GoogleGenerativeAI
from vectorstore.manage_vectorstore import VectorStoreManager
from rag_pipeline import run_pipeline
from prompt_template import build_gemini_rag_prompt, extract_citations
from retrieval import retrieve_documents_multilingual, retrieve_documents, detect_language, is_thai_related_query
from dotenv import load_dotenv
import os

load_dotenv()

vectorstore_manager = VectorStoreManager()
print('checking vectorstore...')
if vectorstore_manager.check_chromadb_exists:
    print('vectorstore exists\n')
    print('initializing vectorstore...')
    vectorstore = vectorstore_manager.get_exist_cromadb()
else:
    print('vectorstore does not exist\n')
    print('running pipeline...')
    run_pipeline()
    print('pipeline completed\n')
    print('initializing vectorstore...')
    vectorstore = vectorstore_manager.get_exist_cromadb()
print('vectorstore initialized')

print('initializing model...')
model = GoogleGenerativeAI(model="gemini-2.0-flash-exp", api_key=os.getenv("GOOGLE_API_KEY"))
print('model initialized')

def query_rag(question: str, num_docs: int = 8, language: str = "Auto-detect", use_multilingual: bool = True, filter_pages: bool = True):
    """
    Process RAG query and return formatted response.
    
    Args:
        question: User's question
        num_docs: Number of documents to retrieve (k)
        language: Language preference
        use_multilingual: Enable multilingual retrieval
        filter_pages: Filter out bibliography/TOC pages
        
    Returns:
        answer: Generated answer
        sources: Formatted source information
        debug: Debug information
    """
    if not question.strip():
        return "Please enter a question.", "", ""
    
    try:
        lang_map = {
            "Auto-detect": "auto",
            "English": "en",
            "Thai (ไทย)": "th"
        }
        lang_code = lang_map.get(language, "auto")

        # detect language and query type
        detected_lang = detect_language(question)
        is_thai_query = is_thai_related_query(question)

        # retrieve documents with multilingual support
        if use_multilingual:
            results = retrieve_documents_multilingual(
                question,
                k=num_docs,
                adaptive_k=True,
                filter_pages=filter_pages
            )
        else:
            results = retrieve_documents(question, k=num_docs)
        
        if not results:
            return (
                "No relevant documents found. Try:\n" +
                "- Increasing number of documents (k slider)\n" +
                "- Enabling multilingual retrieval\n" +
                "- Rephrasing your question",
                "No sources retrieved",
                f"Query: {question}\nRetrieved: 0 documents"
            )
        
        # build prompt
        prompt = build_gemini_rag_prompt(question, results, language=lang_code)
        answer = model.invoke(prompt)

        # extract citation
        citations = extract_citations(answer)

        # format sources
        sources_text = "### Retrieved Sources:\n\n"
        
        # group by source file
        sources_by_file = {}
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get('source', 'unknown')
            page = doc.metadata.get('page', 'unknown')
            
            if source not in sources_by_file:
                sources_by_file[source] = []
            sources_by_file[source].append((page, doc.page_content))
        
        # display grouped by file
        for source, pages_content in sources_by_file.items():
            source_name = source.replace('dataset/', '').replace('.pdf', '')
            sources_text += f"**{source_name}**\n"
            for page, content in pages_content[:3]:  # show max 3 per file
                preview = content[:120].replace('\n', ' ').strip()
                sources_text += f"  • Page {page}: _{preview}..._\n"
            if len(pages_content) > 3:
                sources_text += f"  • _...and {len(pages_content) - 3} more pages_\n"
            sources_text += "\n"
        
        # format citations
        if citations:
            sources_text += "### Citations in Answer:\n\n"
            unique_cites = {}
            for cite in citations:
                key = f"{cite['source']}:{cite['page']}"
                unique_cites[key] = cite
            
            for i, cite in enumerate(unique_cites.values(), 1):
                cite_name = cite['source'].replace('dataset/', '').replace('.pdf', '')
                sources_text += f"{i}. **{cite_name}** (Page {cite['page']})\n"
        else:
            sources_text += "\n**Warning:** No citations found in answer. The response may not be well-grounded in sources.\n"
        
        sources = sources_text
        
        return answer, sources_text
    except Exception as e:
        error_msg = f"Error: {str(e)}\n\nPlease check:\n- API key is valid\n- Internet connection\n- Try again or adjust settings"
        return error_msg, "Error occurred", f"Exception: {str(e)}"

# Get vector store stats for display
stats = vectorstore_manager.get_vectorstore_stats()

# Create Gradio interface
with gr.Blocks(title="RAG Cyber Security Assignment", theme=gr.themes.Default()) as demo:
    
    # Display stats in a nice format
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(
                f"""
                # RAG Cyber Security Assignment

                ### Knowledge Base Stats
                
                **Documents:** {stats.get('count', 0):,} chunks
                
                **Sources:**
                - OWASP Top 10 (41 pages)
                - MITRE ATT&CK (46 pages)  
                - Thailand Standards (77 pages)
                
                **Collection:** `{stats.get('collection_name', 'N/A')}`
                """
            )
    
    gr.Markdown("---")
    
    with gr.Row():
        with gr.Column(scale=2):
            question_input = gr.Textbox(
                label="Your Question",
                lines=3
            )
            
            with gr.Row():
                num_docs_slider = gr.Slider(
                    minimum=3,
                    maximum=15,
                    value=8,
                    step=1,
                    label="Number of Documents (k)",
                    info="Higher k = more context but slower. Recommended: 8-10 for Thai content"
                )
                
            submit_btn = gr.Button("Ask Question", variant="primary", size="lg")
    
    with gr.Row():
        with gr.Column(scale=2):
            answer_output = gr.Textbox(label="Answer", lines=10)
            
        with gr.Column(scale=1):
            sources_output = gr.Textbox(label="Sources & Citations", lines=10)
    
    # Event handler
    submit_btn.click(
        fn=query_rag,
        inputs=[question_input, num_docs_slider],
        outputs=[answer_output, sources_output]
    )
    
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
