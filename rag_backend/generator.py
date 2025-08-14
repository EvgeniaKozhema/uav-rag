import ollama
from rag_backend.retriever import get_vectorstore

PROMPT_FILE = "rag_prompt.txt"
with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    BASE_PROMPT = f.read()

def generate_answer(question, k=3):
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(question)

    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = BASE_PROMPT.format(context=context, question=question)

    response = ollama.chat(
        model="gemma2:2b",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.7, "num_predict": 1000}
    )
    return response["message"]["content"]
