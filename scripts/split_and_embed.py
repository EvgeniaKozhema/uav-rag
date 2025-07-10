import os
import shutil

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

INPUT_DIR = "data/processed"
CHROMA_DIR = "embeddings/chroma_db"

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " ", ""]
)

embedding_model = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

def load_documents():
    documents = []

    for filename in os.listdir(INPUT_DIR):
        if not filename.endswith('.txt'):
            continue

        filepath = os.path.join(INPUT_DIR, filename)

        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()

        chunks = text_splitter.split_text(text)

        for idx, chunk in enumerate(chunks):
            documents.append({
                'text': chunk,
                'metadata': {'source': filename, 'chunk_id': idx}
            })
    return documents

def build_vectorstore(documents):
    texts = [f"passage: {doc['text']}" for doc in documents]
    metadatas = [doc['metadata'] for doc in documents]

    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embedding_model,
        metadatas=metadatas,
        persist_directory=CHROMA_DIR
    )
    vectorstore.persist()

def main():
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    documents = load_documents()
    build_vectorstore(documents)

if __name__ == '__main__':
    main()
