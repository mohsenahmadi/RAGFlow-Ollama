import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_astradb import AstraDBVectorStore
import os
import tempfile
import hashlib
from datetime import datetime
import json
import mimetypes

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

# File uploader for multiple document types
supported_formats = ["pdf", "md", "txt", "json"]
uploaded_files = st.file_uploader(f"Upload Documents ({', '.join(supported_formats)})", 
                                 type=supported_formats, 
                                 accept_multiple_files=True)

# Text splitter settings
chunk_size = st.number_input("Chunk Size", min_value=100, max_value=2000, value=750, step=50)
chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=500, value=150, step=10)

# Helper function to generate a hash for a document
def get_document_hash(content, filename):
    # Create a unique hash using the file content and name
    combined = f"{content}{filename}"
    return hashlib.md5(combined.encode()).hexdigest()

# Helper function to check if a document already exists
def document_exists(doc_hash, vectorstore):
    try:
        # Query for the document hash in the metadata
        results = vectorstore.similarity_search(
            "check_duplicate_placeholder",  # This won't affect the search since we're filtering by metadata
            k=1,
            filter={"doc_hash": doc_hash}
        )
        return len(results) > 0
    except Exception:
        # If there's an error or the collection is empty, assume document doesn't exist
        return False

# Helper function to load and process different file types
def load_document(file_path, file_name, file_type):
    docs = []
    
    try:
        if file_type.endswith('pdf'):
            loader = PyPDFLoader(file_path)
            docs = loader.load()
        
        elif file_type.endswith(('md', 'txt')):
            loader = TextLoader(file_path)
            docs = loader.load()
        
        elif file_type.endswith('json'):
            # Function to extract content from JSON
            def extract_content(data, metadata):
                # Adjust this based on your JSON structure
                # This example assumes a field called "content" or the first string field
                if isinstance(data, dict):
                    if "content" in data:
                        return data["content"]
                    # Try to find the first string value
                    for key, value in data.items():
                        if isinstance(value, str) and len(value) > 10:
                            return value
                return str(data)
            
            loader = JSONLoader(
                file_path=file_path,
                jq_schema='.', 
                content_key=None,  # We'll handle extraction in the transform function
                text_content=True,
                json_lines=False,
                content_extractor=extract_content
            )
            docs = loader.load()
        
        # Add file metadata to each document
        for doc in docs:
            # Read the file content for hashing
            with open(file_path, 'r', errors='ignore') as f:
                content = f.read()
            
            doc_hash = get_document_hash(content, file_name)
            
            doc.metadata.update({
                "filename": file_name,
                "upload_date": datetime.now().isoformat(),
                "file_type": file_type,
                "doc_hash": doc_hash  # Store hash for duplicate checking
            })
        
        return docs
    except Exception as e:
        st.error(f"Error loading {file_name}: {str(e)}")
        return []

if uploaded_files:
    # Initialize embeddings early to check dimensions and for document existence check
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
        
        # Define expected dimension
        EMBEDDING_DIMENSION = 384
        if actual_dimension != EMBEDDING_DIMENSION:
            st.warning(f"Warning: Embedding dimension ({actual_dimension}) does not match expected dimension ({EMBEDDING_DIMENSION})")
    
    except Exception as e:
        st.error(f"Failed to initialize OllamaEmbeddings: {str(e)}")
        st.stop()
    
    # Create or access vector store to check for duplicates
    try:
        vectorstore = AstraDBVectorStore(
            collection_name=collection_name,
            embedding=embeddings,
            api_endpoint=ASTRA_DB_API_ENDPOINT,
            token=ASTRA_DB_APPLICATION_TOKEN,
            namespace=ASTRA_DB_NAMESPACE
        )
    except Exception as e:
        st.error(f"Failed to initialize AstraDBVectorStore: {str(e)}")
        st.stop()
    
    documents = []
    skipped_docs = 0
    processed_docs = 0
    
    with st.spinner("Processing documents..."):
        for file in uploaded_files:
            # Save file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.name}") as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Determine file type
            file_type = mimetypes.guess_type(file.name)[0]
            if not file_type:
                # Guess by extension
                ext = os.path.splitext(file.name)[1].lower()
                if ext == '.md':
                    file_type = 'text/markdown'
                elif ext == '.json':
                    file_type = 'application/json'
                elif ext == '.txt':
                    file_type = 'text/plain'
                elif ext == '.pdf':
                    file_type = 'application/pdf'
            
            # Check for duplicate before processing
            with open(tmp_file_path, 'r', errors='ignore') as f:
                try:
                    content = f.read()
                    doc_hash = get_document_hash(content, file.name)
                    
                    # Skip if duplicate
                    if document_exists(doc_hash, vectorstore):
                        st.info(f"Skipping duplicate document: {file.name}")
                        skipped_docs += 1
                        continue
                except Exception as e:
                    # If we can't read the file as text for hashing, continue with processing
                    st.warning(f"Could not check duplication for {file.name}: {str(e)}")
            
            # Load and process document
            try:
                docs = load_document(tmp_file_path, file.name, file_type)
                if docs:
                    documents.extend(docs)
                    processed_docs += 1
            finally:
                # Clean up temporary file
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
    
    st.write(f"Documents processed: {processed_docs}, Duplicates skipped: {skipped_docs}")
    
    if documents:
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(documents)
        
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
                
                # For each document in batch, check if it's already stored using its hash
                filtered_batch = []
                for doc in current_batch:
                    if not document_exists(doc.metadata.get("doc_hash", ""), vectorstore):
                        filtered_batch.append(doc)
                
                if filtered_batch:
                    vectorstore.add_documents(filtered_batch)
                
                # Update progress
                progress = (end_idx / total_chunks)
                progress_bar.progress(progress)
                status_text.text(f"Processing batch {i//batch_size + 1}/{(total_chunks-1)//batch_size + 1} ({end_idx}/{total_chunks} chunks)")
            
            st.success(f"Documents successfully vectorized and stored in collection {collection_name}")
            
        except Exception as e:
            st.error(f"Failed to store documents in AstraDB: {str(e)}")
            # Provide more detailed error information for debugging
            import traceback
            st.code(traceback.format_exc(), language="python")
    else:
        if skipped_docs > 0:
            st.warning("All documents were duplicates, nothing new to process")
        else:
            st.warning("No documents were processed")
else:
    st.info(f"Please upload documents in the following formats: {', '.join(supported_formats)}")