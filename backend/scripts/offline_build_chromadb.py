import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

client = chromadb.PersistentClient(path="./chroma_db")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

store = Chroma(
    client=client,
    collection_name="machine-learning",
    embedding_function=embeddings,
)

store.add_texts(
    texts=["your chunk text here", "..."],
    metadatas=[{"source": "textbook.pdf", "page": 12, "chunk_id": "c1"}, ...],
    ids=["c1", "c2", ...],
)