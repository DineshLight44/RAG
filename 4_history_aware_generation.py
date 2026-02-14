from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

# =========================
# 1. Vector DB + Embeddings
# =========================

persistent_directory = "db/chroma_db"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embeddings
)

# =========================
# 2. Local LLM (Ollama)
# =========================

llm = OllamaLLM(
    model="phi3",      # change to "llama3:8b" later
    temperature=0
)

# =========================
# 3. Chat History (plain text)
# =========================

chat_history = []  # list of (user, assistant) tuples

def format_chat_history():
    history_text = ""
    for u, a in chat_history:
        history_text += f"User: {u}\nAssistant: {a}\n"
    return history_text.strip()

# =========================
# 4. Ask Question Function
# =========================

def ask_question(user_question):
    print(f"\n--- You asked: {user_question} ---")

    # ---- STEP 1: Rewrite question (history → standalone) ----
    if chat_history:
        rewrite_prompt = f"""
Given the conversation history and the new question,
rewrite the new question so it is standalone and suitable for document search.

Conversation history:
{format_chat_history()}

New question:
{user_question}

Return ONLY the rewritten question.
"""
        search_question = llm.invoke(rewrite_prompt).strip()
        print(f"🔍 Searching for: {search_question}")
    else:
        search_question = user_question

    # ---- STEP 2: Retrieve documents ----
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 15}
    )

    docs = retriever.invoke(search_question)

    print(f"Found {len(docs)} relevant documents:")
    for i, doc in enumerate(docs, 1):
        preview = doc.page_content.split("\n")[0]
        print(f"  Doc {i}: {preview}...")

    # ---- STEP 3: Build RAG prompt ----
    context = "\n\n".join(doc.page_content for doc in docs)

    rag_prompt = f"""
You are a question-answering assistant.

Answer the question using ONLY the information in the context.
If the answer is not explicitly stated, say:
"I don't have enough information from the provided documents."

Context:
{context}

Question:
{user_question}
"""

    # ---- STEP 4: Generate answer ----
    answer = llm.invoke(rag_prompt).strip()

    # ---- STEP 5: Save conversation ----
    chat_history.append((user_question, answer))

    print(f"\n✅ Answer:\n{answer}")
    return answer

# =========================
# 5. Chat Loop
# =========================

def start_chat():
    print("Ask me questions! Type 'quit' to exit.")

    while True:
        question = input("\nYour question: ")

        if question.lower() == "quit":
            print("Goodbye!")
            break

        ask_question(question)

if __name__ == "__main__":
    start_chat()