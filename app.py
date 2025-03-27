import streamlit as st
import logging
import os
import shutil
import sys
import pdfplumber
import ollama
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from typing import List, Tuple, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Streamlit page configuration
st.set_page_config(
    page_title="Ollama PDF RAG Streamlit",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)
def extract_model_names(models_info: Any) -> Tuple[str, ...]:
    """
    Extract model names from the provided models information.

    Args:
        models_info: Response from ollama.list()

    Returns:
        Tuple[str, ...]: A tuple of model names.
    """
    logger.info("Extracting model names from models_info")
    try:
        # The new response format returns a list of Model objects
        if hasattr(models_info, "models"):
            # Extract model names from the Model objects
            model_names = tuple(model.model for model in models_info.models)
        else:
            # Fallback for any other format
            model_names = tuple()
            
        logger.info(f"Extracted model names: {model_names}")
        return model_names
    except Exception as e:
        logger.error(f"Error extracting model names: {e}")
        return tuple()


@st.cache_resource
def get_embeddings():
    logger.info("Using Ollama Embeddings")
    return OllamaEmbeddings(model="nomic-embed-text")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = text_splitter.split_text(text)
    except Exception as e:
        logger.error(f"Error at get_text_chunks function: {e}")
        chunks = []
    return chunks

def get_vector_store(text_chunks):
    vector_store = None
    try:
        embeddings = get_embeddings()
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
    except Exception as e:
        logger.error(f"Error at get_vector_store function: {e}")
    return vector_store

@st.cache_resource
def get_llm(selected_model: str):
    logger.info(f"Getting LLM: {selected_model}")
    return ChatOllama(model=selected_model, temperature=0.1)

def process_question(question: str, vector_db: FAISS, selected_model: str) -> str:
    logger.info(f"Processing question: {question} using model: {selected_model}")
    llm = get_llm(selected_model)
    
    # Define the query prompt template
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""
        Original question: {question}""",
    )

    # Create retriever with LLM for multiple query retrieval
    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    )

    # Define the answer template
    template = """Answer the question as detailed as possible from the provided context only. 
    Do not generate a factual answer if the information is not available. 
    If you do not know the answer, respond with "I don't know the answer as not sufficient information is provided in the PDF."
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """

    prompt = ChatPromptTemplate.from_template(template)

    # Set up the chain with retriever and LLM
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Get the response from the chain
    response = chain.invoke(question)

    # Check if the retrieved context is relevant or not
    if "I don't know the answer" in response or not response.strip():
        return "I don't know the answer as not sufficient information is provided in the PDF."

    logger.info("Question processed and response generated")
    return response

@st.cache_data
def extract_all_pages_as_images(file_upload) -> List[Any]:
    logger.info(f"Extracting all pages as images from file: {file_upload.name}")
    pdf_pages = []
    with pdfplumber.open(file_upload) as pdf:
        pdf_pages = [page.to_image().original for page in pdf.pages]
    logger.info("PDF pages extracted as images")
    return pdf_pages

def delete_vector_db() -> None:
    logger.info("Deleting vector DB")
    st.session_state.pop("pdf_pages", None)
    st.session_state.pop("file_upload", None)
    st.session_state.pop("vector_db", None)
    if os.path.exists("faiss_index"):
        shutil.rmtree("faiss_index")
        logger.info("FAISS index deleted")
    st.success("Collection and temporary files deleted successfully.")
    logger.info("Vector DB and related session state cleared")
    st.rerun()

def main():
    st.subheader("⚙️ Ollama Chat with PDF RAG -- Adnen Chikh", divider="gray", anchor=False)
    models_info = ollama.list()
    available_models = extract_model_names(models_info)

    col1, col2 = st.columns([1.5, 2])

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "vector_db" not in st.session_state:
        st.session_state["vector_db"] = None

    if available_models:
        selected_model = col2.selectbox(
            "Pick a model available locally on your system ↓", available_models
        )

    pdf_docs = col1.file_uploader(
        "Upload your PDF Files", 
        accept_multiple_files=True
    )

    # Submit and Delete buttons side by side
    col_buttons = col1.columns([1, 1])
    
    with col_buttons[0]:
        submit_button = st.button("Submit & Process", key="submit_process")

    with col_buttons[1]:
        delete_collection = st.button("⚠️ Delete collection", type="secondary")

    if submit_button and pdf_docs:
        with st.spinner("Processing..."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            st.session_state["vector_db"] = get_vector_store(text_chunks)
            st.success("Done")

        pdf_pages = extract_all_pages_as_images(pdf_docs[0])  # Assuming single file upload
        st.session_state["pdf_pages"] = pdf_pages

        zoom_level = col1.slider(
            "Zoom Level", min_value=100, max_value=1000, value=700, step=50, key="zoom_slider_1"
        )

        with col1:
            with st.container(height=410, border=True):
                for page_image in pdf_pages:
                    st.image(page_image, width=zoom_level)

    if delete_collection:
        delete_vector_db()

    with col2:
        message_container = st.container(height=500, border=True)

        for message in st.session_state["messages"]:
            avatar = "🤖" if message["role"] == "assistant" else "😎"
            with message_container.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        if prompt := st.chat_input("Enter a prompt here..."):
            try:
                st.session_state["messages"].append({"role": "user", "content": prompt})
                message_container.chat_message("user", avatar="😎").markdown(prompt)

                with message_container.chat_message("assistant", avatar="🤖"):
                    with st.spinner(":green[processing...]"):
                        if st.session_state["vector_db"] is not None:
                            response = process_question(
                                prompt, st.session_state["vector_db"], selected_model
                            )
                            st.markdown(response)
                            st.session_state["messages"].append(
                                {"role": "assistant", "content": response}
                            )
                        else:
                            response = "Please upload and process a PDF file first."
                            st.warning(response)

            except Exception as e:
                st.error(e, icon="⚠️")
                logger.error(f"Error processing prompt: {e}")

        # Ensure PDF viewer is retained
        if st.session_state.get("pdf_pages"):
            zoom_level = col1.slider(
                "Zoom Level", min_value=100, max_value=1000, value=700, step=50, key="zoom_slider_2"
            )
            with col1:
                with st.container(height=410, border=True):
                    for page_image in st.session_state["pdf_pages"]:
                        st.image(page_image, width=zoom_level)

if __name__ == "__main__":
    main()