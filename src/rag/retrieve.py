import chromadb
from chromadb.utils import embedding_functions

def load_paper_embedding_collection(index_path: str, embed_model: str):
    client = chromadb.PersistentClient(path=index_path)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embed_model)

    try:
        return client.get_collection(name="papers", embedding_function=embed_fn)
    except:
        raise RuntimeError("Chroma collection 'papers' not found. Run rag_embed.py first.")


def retrieve_top_k(query: str, index_path: str, embed_model: str, k: int = 3):
    collection = load_paper_embedding_collection(index_path, embed_model)
    results = collection.query(query_texts=[query], n_results=k)
    return [
        {"chunk": chunk, **meta}
        for chunk, meta in zip(results["documents"][0], results["metadatas"][0])
    ]

