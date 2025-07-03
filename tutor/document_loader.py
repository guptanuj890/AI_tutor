import os
import zipfile
from PIL import Image
import pytesseract
import streamlit as st # Assuming Streamlit for st.info/warning/error

# Updated imports for LangChain 0.3.x
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
)
from langchain_core.documents import Document # Corrected import for Document
from langchain_text_splitters import RecursiveCharacterTextSplitter # Corrected import for TextSplitter

def extract_text_from_image(file_path):
    """
    Extracts text from an image file using Tesseract OCR.
    Requires pytesseract and Tesseract OCR to be installed.
    """
    try:
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        st.error(f"Error extracting text from image {file_path}: {e}. "
                 "Ensure Pillow and pytesseract are installed, and Tesseract OCR is in your system's PATH.")
        return ""

def load_single_file(file_path):
    """
    Loads content from a single file based on its extension.
    """
    ext = os.path.splitext(file_path)[1].lower()
    docs = []

    try:
        if ext == '.pdf':
            loader = PyPDFLoader(file_path)
            docs = loader.load()
        elif ext == '.txt':
            loader = TextLoader(file_path)
            docs = loader.load()
        elif ext == '.docx':
            loader = UnstructuredWordDocumentLoader(file_path)
            docs = loader.load()
        elif ext == '.pptx':
            loader = UnstructuredPowerPointLoader(file_path)
            docs = loader.load()
        elif ext in ['.png', '.jpg', '.jpeg']:
            text = extract_text_from_image(file_path)
            if text:
                docs = [Document(page_content=text, metadata={"source": file_path, "type": "image_ocr"})]
        else:
            st.warning(f"Unsupported file type: {file_path}. Skipping.")
            return []
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        return []
    
    st.info(f"Loaded {len(docs)} documents from {file_path}")
    return docs

def extract_zip(zip_path, extract_to='temp_extracted'):
    """
    Extracts contents of a ZIP file to a temporary directory.
    """
    os.makedirs(extract_to, exist_ok=True)
    st.info(f"Extracting ZIP file to: {extract_to}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        st.success("ZIP file extracted successfully.")
    except Exception as e:
        st.error(f"Error extracting ZIP file {zip_path}: {e}")
        return []

    extracted_files = []
    for root, _, files in os.walk(extract_to):
        for file in files:
            # Filter for supported file types
            if file.lower().endswith(('.pdf', '.docx', '.pptx', '.txt', '.png', '.jpg', '.jpeg')):
                extracted_files.append(os.path.join(root, file))
    st.info(f"Found {len(extracted_files)} supported files in ZIP.")
    return extracted_files

def load_and_split_documents(file_path):
    """
    Loads documents from a given file (or ZIP) and splits them into chunks.
    """
    all_docs = []

    if file_path.endswith('.zip'):
        extracted_files = extract_zip(file_path)
        for f in extracted_files:
            docs = load_single_file(f)
            all_docs.extend(docs)
    else:
        all_docs = load_single_file(file_path)

    if not all_docs:
        st.warning("No documents loaded to split.")
        return []

    # Ensure config is available or use defaults
    try:
        from config import Config
        chunk_size = Config.CHUNK_SIZE
        chunk_overlap = Config.CHUNK_OVERLAP
    except ImportError:
        st.warning("config.py not found or missing Config. Using default chunking parameters.")
        chunk_size = 1000
        chunk_overlap = 200

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_documents(all_docs)
    st.info(f"Split documents into {len(chunks)} chunks.")
    return chunks
