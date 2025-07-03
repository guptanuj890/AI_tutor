import os
from dotenv import load_dotenv

# Load .env variables if available
load_dotenv()

class Config:
    # API Keys
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    # Embedding model
    GEMINI_EMBEDDING_MODEL = "models/embedding-001"

    # File paths
    VECTORSTORE_PATH = "embeddings/index"
    TEMP_EXTRACT_PATH = "temp_extracted"
    DATA_PATH = "data"

    # Chunking settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
