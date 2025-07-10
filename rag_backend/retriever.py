import os
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

CHROMA_DIR = "embeddings/chroma_db"

embedding_model = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

vectorstore = Chroma(embedding_function=embedding_model, persist_directory=CHROMA_DIR)

def get_relevant_chunk(query: str, k: int=5):
    formatted_query = f"query: {query}"
    docs = vectorstore.similarity_search(query, k=k)

    results = []

    for doc in docs:
        results.append({
            'text':doc.page_content,
            'metadata':doc.metadata})

    return results

if __name__=='__main__':
    query = input('Введите ваш вопрос:')
    chunks = get_relevant_chunk(query)

    for i, chunk in enumerate(chunks, 1):
        print(f"\n--- Чанк {i} ---")
        print(f"Источник: {chunk['metadata'].get('source')} | ID: {chunk['metadata'].get('chunk_id')}")
        print(chunk["text"])