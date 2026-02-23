from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from pydantic import BaseModel
from typing import List


# Setup
persistent_directory = r"C:\Users\dkglt\rag\rag\dbv1\chroma_db"# Must match creation path

# 🔁 REPLACED OpenAIEmbeddings
embedding_model = OllamaEmbeddings(model="nomic-embed-text")

# 🔁 REPLACED ChatOpenAI
llm = ChatOllama(model="phi3", temperature=0)

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# Pydantic model for structured output
class QueryVariations(BaseModel):
    queries: List[str]

# ──────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ──────────────────────────────────────────────────────────────────

# Original query
original_query =  "What hyperparameters are important during fine-tuning?"
print(f"Original Query: {original_query}\n")

# ──────────────────────────────────────────────────────────────────
# Step 1: Generate Multiple Query Variations
# ──────────────────────────────────────────────────────────────────

import re

prompt = f"""
Generate exactly 3 alternative queries that mean the same as the original question.

Original question:
{original_query}

Rules:
- Keep meaning exactly same
- Do not introduce new technical terms
- Return only 3 queries as plain text lines
"""

response = llm.invoke(prompt)

query_variations = []

for line in response.content.split("\n"):
    line = line.strip()
    if not line:
        continue

    # Remove numbering like "1." or "1. 1."
    line = re.sub(r"^\d+\.\s*", "", line)
    line = re.sub(r"^\d+\.\s*", "", line)

    query_variations.append(line)

query_variations = query_variations[:3]
# ──────────────────────────────────────────────────────────────────
# Step 2: Search with Each Query Variation & Store Results
# ──────────────────────────────────────────────────────────────────

retriever = db.as_retriever(search_kwargs={"k": 5})
all_retrieval_results = []

for i, query in enumerate(query_variations, 1):
    print(f"\n=== RESULTS FOR QUERY {i}: {query} ===")
    
    docs = retriever.invoke(query)
    all_retrieval_results.append(docs)
    
    print(f"Retrieved {len(docs)} documents:\n")
    
    for j, doc in enumerate(docs, 1):
        print(f"Document {j}:")
        print(f"{doc.page_content[:150]}...\n")
    
    print("-" * 50)

print("\n" + "="*60)
print("Multi-Query Retrieval Complete!")