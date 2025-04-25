import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_astradb import AstraDBVectorStore
import os
import tempfile
from datetime import datetime

# Load settings from secrets.toml with error handling
try:
    ASTRA_DB_API_ENDPOINT = st.secrets["ASTRA_DB_API_ENDPOINT"]
    ASTRA_DB_APPLICATION_TOKEN = st.secrets["ASTRA_DB_APPLICATION_TOKEN"]
    ASTRA_DB_NAMESPACE = st.secrets.get("ASTRA_DB_NAMESPACE", None)
    OLLAMA_BASE_URL = st.secrets["OLLAMA_BASE_URL"]
except KeyError as e:
    st.error(f"Missing secret key: {e}")
    st.stop()

# Debug: Print secrets to verify (remove in production)
st.write("Secrets loaded:", {
    "ASTRA_DB_API_ENDPOINT": ASTRA_DB_API_ENDPOINT,
    "ASTRA_DB_APPLICATION_TOKEN": "****" if ASTRA_DB_APPLICATION_TOKEN else None,
    "ASTRA_DB_NAMESPACE": ASTRA_DB_NAMESPACE,
    "OLLAMA_BASE_URL": OLLAMA_BASE_URL
})

# App title
st.title("DocVectorizer for RAG with Ollama")

# Input for use case to dynamically set collection name
use_case = st.text_input("Enter use case (e.g., technical, marketing)", value="default")
collection_name = f"rag_{use_case.lower().replace(' ', '_')}"

# File uploader for PDF files
uploaded_files = st.file_uploader("Upload PDF Documents", type=["pdf"], accept_multiple_files=True)

# Text splitter settings
chunk_size = st.number_input("Chunk Size", min_value=100, max_value=2000, value=750, step=50)
chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=500, value=150, step=10)

# Define embedding dimensions constant
EMBEDDING_DIMENSION = 384

if uploaded_files:
    documents = []
    for file in uploaded_files:
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.name) as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Load PDF document
        try:
            loader = PyPDFLoader(tmp_file_path)
            docs = loader.load()
            # Add metadata to each document
            for doc in docs:
                doc.metadata.update({
                    "filename": file.name,
                    "upload_date": datetime.now().isoformat(),
                    "file_type": file.type
                })
            documents.extend(docs)
        finally:
            os.unlink(tmp_file_path)  # Delete temporary file
    
    if documents:
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(documents)
        
        # Generate embeddings with Ollama using the all-minilm model
        try:
            embeddings = OllamaEmbeddings(
                base_url=OLLAMA_BASE_URL,
                model="all-minilm:latest"  # Using all-minilm which produces 384-dimensional vectors
            )
            
            # Verify embedding dimensions
            sample_text = "This is a test document to check embedding dimensions."
            sample_embedding = embeddings.embed_query(sample_text)
            actual_dimension = len(sample_embedding)
            
            st.info(f"Embedding model dimension: {actual_dimension}")
            
            if actual_dimension != EMBEDDING_DIMENSION:
                st.warning(f"Warning: Embedding dimension ({actual_dimension}) does not match expected dimension ({EMBEDDING_DIMENSION})")
        
        except Exception as e:
            st.error(f"Failed to initialize OllamaEmbeddings: {str(e)}")
            st.stop()
        
        # Create or access vector store with detailed error handling
        try:
            vectorstore = AstraDBVectorStore(
                collection_name=collection_name,
                embedding=embeddings,
                api_endpoint=ASTRA_DB_API_ENDPOINT,
                token=ASTRA_DB_APPLICATION_TOKEN,
                namespace=ASTRA_DB_NAMESPACE,
                dimension=EMBEDDING_DIMENSION  # Explicitly set dimension to 384
            )
        except Exception as e:
            st.error(f"Failed to initialize AstraDBVectorStore: {str(e)}")
            st.stop()
        
        # Add documents to vector store with error handling
        try:
            # Process in smaller batches to avoid potential issues
            batch_size = 10
            total_chunks = len(chunks)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(0, total_chunks, batch_size):
                end_idx = min(i + batch_size, total_chunks)
                current_batch = chunks[i:end_idx]
                
                vectorstore.add_documents(current_batch)
                
                # Update progress
                progress = (end_idx / total_chunks)
                progress_bar.progress(progress)
                status_text.text(f"Processing batch {i//batch_size + 1}/{(total_chunks-1)//batch_size + 1} ({end_idx}/{total_chunks} documents)")
            
            st.success(f"Documents successfully vectorized and stored in collection {collection_name}")
            
        except Exception as e:
            st.error(f"Failed to store documents in AstraDB: {str(e)}")
            # Provide more detailed error information for debugging
            import traceback
            st.code(traceback.format_exc(), language="python")
    else:
        st.warning("No documents were processed")
else:
    st.info("Please upload PDF documents to proceed")