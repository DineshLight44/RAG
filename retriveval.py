from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

persistent_directory = "db/chroma_db"

embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# retriever = db.as_retriever(search_kwargs={"k": 5})


retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 5,
        "score_threshold": 0.2  # Only return chunks with cosine similarity ≥ 0.3
    }
)

query = "Who takes care of Gregor after his transformation?"

relevant_docs = retriever.invoke(query)

print("\n--- Retrieved Context ---")

for i, doc in enumerate(relevant_docs, 1):
    print(f"\n📄 Document {i}")
    print("-" * 40)
    print(doc.page_content[:300] + "...")