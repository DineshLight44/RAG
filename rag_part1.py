import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from pathlib import Path


def load_documents(docs_path="rag"):
    
    print(f"Loading from {docs_path}...")
    
    if not os.path.exists(docs_path):
        os.makedirs(docs_path, exist_ok=True)
        print("Created directory. Add .txt files.")
        return []
    
    documents = []
    for file_path in Path(docs_path).glob("*.txt"):
        try:
            loader = TextLoader(str(file_path), encoding="utf-8")
            doc = loader.load()[0]
            documents.append(doc)
            print(f"✅ Loaded: {file_path.name} ({len(doc.page_content)} chars)")
        except Exception as e:
            print(f"❌ Failed {file_path.name}: {e}")
    
    if not documents:
        print("No valid .txt files found.")
    
    return documents


def split_documents(documents, chunk_size=600, chunk_overlap=150):
    
    print("Splitting documents into chunks...")
    
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunks = text_splitter.split_documents(documents)

    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(chunk.page_content[:200])
    
    return chunks


def create_vector_store(chunks, persist_directory="db/chroma_db"):
    
    print("Creating embeddings using HuggingFace (FREE)...")

    # ✅ FREE local embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("--- Creating vector store ---")
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )

    print(f"Vector store saved to {persist_directory}")
    
    return vectorstore


def main():

    folder_path = input("Enter folder path: ")

    docs = load_documents(folder_path)

    print(f"\nLoaded {len(docs)} documents total.")

    chunks = split_documents(docs)
    if not chunks:
        print("No chunks created.")
        return
    create_vector_store(chunks)


if __name__ == "__main__":
    main()