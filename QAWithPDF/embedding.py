from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.node_parser import SentenceSplitter

from QAWithPDF.data_ingestion import load_data
from QAWithPDF.model_api import load_model

import sys
from exception import customexception
from logger import logging

def download_gemini_embedding(model,document):
    """
    Downloads and initializes a Gemini Embedding model for vector embediings

    Returns:
    - VectorStoreIndex: An index of vector embeddings for efficient similarity
    """
    try:
        gemini_embed_model = GeminiEmbedding(model_name="models/embedding-001")
        Settings.llm = model
        Settings.embed_model = gemini_embed_model
        Settings.node_parser = SentenceSplitter(chunk_size=800, chunk_overlap=20)
        index = VectorStoreIndex.from_documents(document,transformations=Settings.transformations)
        index.storage_context.persist()
        query_engine = index.as_query_engine()
        return query_engine
    
    except Exception as e:
        raise customexception(e,sys)