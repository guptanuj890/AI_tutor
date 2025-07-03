import os
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
import streamlit as st

def create_or_load_vectorstore(chunks: list[Document]):
    """
    Creates a new FAISS vector store from document chunks or loads an existing one.
    Embeddings are generated using GoogleGenerativeAIEmbeddings.

    Args:
        chunks (list[Document]): A list of LangChain Document objects (text chunks).

    Returns:
        FAISS: The FAISS vector store.
    """
    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    faiss_index_path = "embeddings/faiss_index" # Correctly defined variable

    # Check if the FAISS index directory exists and is a directory
    if os.path.exists(faiss_index_path) and os.path.isdir(faiss_index_path):
        st.info(f"Loading existing FAISS index from {faiss_index_path}...")
        # Load the local FAISS index. allow_dangerous_deserialization is needed for loading
        # indexes saved with older versions or in certain environments.
        vectorstore = FAISS.load_local(faiss_index_path, embeddings_model, allow_dangerous_deserialization=True)
        st.success("FAISS index loaded successfully.")
    else:
        st.info("Creating new FAISS index from document chunks...")
        if not chunks:
            st.warning("No chunks provided to create vector store. This might lead to issues.")
            # It's good practice to raise an error if critical data is missing
            raise ValueError("No document chunks available to create a new vector store.")

        # Create a new FAISS index from the provided document chunks and embeddings
        vectorstore = FAISS.from_documents(chunks, embeddings_model)
        st.success("FAISS index created.")
        st.info(f"Saving FAISS index to {faiss_index_path}...")
        # Save the newly created FAISS index locally
        vectorstore.save_local(faiss_index_path)
        st.success("FAISS index saved successfully.")
    return vectorstore
