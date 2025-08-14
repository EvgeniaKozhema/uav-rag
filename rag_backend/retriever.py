import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_DIR = "embeddings/chroma_db"

embedding_model = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

def get_vectorstore():
    vectorstore = Chroma(
        embedding_function=embedding_model,
        persist_directory=CHROMA_DIR
    )
    return vectorstore
