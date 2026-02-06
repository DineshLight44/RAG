from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama   # ✅ NEW

persistent_directory = "db/chroma_db"

# 1. Embeddings (FREE, local)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Vector DB
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# 3. Retriever
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 3,
        "score_threshold": 0.2
    }
)

# 4. User query
query = "What job did Gregor have before his transformation?"

# 5. Retrieve documents
relevant_docs = retriever.invoke(query)

print("\n--- Retrieved Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"\n📄 Document {i}")
    print("-" * 40)
    print(doc.page_content[:300] + "...")

# 6. Combine retrieved text (RAG core)
context = "\n".join(doc.page_content for doc in relevant_docs)

# 7. Prompt (IMPORTANT CHANGE)
combined_input = f"""
You are a question-answering assistant.
Use ONLY the information provided in the context.
If the answer is not present, say you don't know.

Context:
{context}

Question:
{query}
"""

# 8. Local LLM (Phi-3 or LLaMA)
llm = Ollama(
    model="phi3",      # 🔁 change to "llama3:8b" later
    temperature=0
)

# 9. Generate answer
result = llm.invoke(combined_input)

print("\n--- Generated Response ---")
print(result)