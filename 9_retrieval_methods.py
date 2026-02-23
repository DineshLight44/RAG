from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Setup
persistent_directory = r"C:\Users\dkglt\rag\rag\dbv1\chroma_db"# Must match creation path

embedding_model = OllamaEmbeddings(
    model="nomic-embed-text"
)

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# Query to test
query = "What hyperparameters are important during fine-tuning?"
print(f"Query: {query}\n")

# Method 2 query
# query = "What myself ?"
# print(f"Query: {query}\n")

# ───────────────────────────────────────────────
# METHOD 1: Basic Similarity Search
# ───────────────────────────────────────────────

# print("=== METHOD 1: Similarity Search (k=3) ===")

# retriever = db.as_retriever(search_kwargs={"k": 3})
# docs = retriever.invoke(query)

# print(f"Retrieved {len(docs)} documents:\n")

# for i, doc in enumerate(docs, 1):
#     print(f"Document {i}:")
#     print(doc.page_content)
#     print()
    
# print("-" * 60)

# ──────────────────────────────────────────────────────────────────
# METHOD 2: Similarity with Score Threshold
# Only returns documents above a certain similarity score
# ──────────────────────────────────────────────────────────────────

# print("\n=== METHOD 2: Similarity with Score Threshold ===")
# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={
#         "k": 3,
#         "score_threshold": 0.5  # Only return docs with similarity >= 0.3
#     }
# )

# docs = retriever.invoke(query)
# print(f"Retrieved {len(docs)} documents (threshold: 0.3):\n")

# for i, doc in enumerate(docs, 1):
#     print(f"Document {i}:")
#     print(f"{doc.page_content}\n")

# print("-" * 60)

# # ──────────────────────────────────────────────────────────────────
# # METHOD 3: Maximum Marginal Relevance (MMR)
# # Balances relevance and diversity - avoids redundant results
# # ──────────────────────────────────────────────────────────────────

print("\n=== METHOD 3: Maximum Marginal Relevance (MMR) ===")
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,           # Final number of docs
        "fetch_k": 10,    # Initial pool to select from
        "lambda_mult": 0.5  # 0=max diversity, 1=max relevance
    }
)

docs = retriever.invoke(query)
print(f"Retrieved {len(docs)} documents (λ=0.5):\n")

for i, doc in enumerate(docs, 1):
    print(f"Document {i}:")
    print(f"{doc.page_content}\n")

print("=" * 60)
print("Done! Try different queries or parameters to see the differences.")