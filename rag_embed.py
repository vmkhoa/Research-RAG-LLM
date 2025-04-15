import os
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from src.utils import load_jsonl_as_df
from tqdm import tqdm
from more_itertools import chunked

def build_chroma_index():
    input_path = 'data/processed/cleaned.jsonl'
    index_path = 'data/vector_index'
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    df = load_jsonl_as_df(input_path)
    client = chromadb.PersistentClient(path=index_path)
    collection = client.get_or_create_collection(
        name='papers',
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name)
    )
    docs, ids, metadatas = [], [], []

    print("Embedding chunks into ChromaDB...")
    for i, row in tqdm(df.iterrows(), total=len(df)):
        if isinstance(row['chunks'], list):
            for j, chunk in enumerate(row['chunks']):
                docs.append(chunk)
                ids.append(f"{i}-{j}")
                metadatas.append({
                    'title': row.get('title', ''),
                    "abstract": row.get("abstract", "")
                })

    batch_size = 1000
    for doc_batch, id_batch, meta_batch in zip(
        chunked(docs, batch_size),
        chunked(ids, batch_size),
        chunked(metadatas, batch_size)):
        collection.add(documents=doc_batch, ids=id_batch, metadatas=meta_batch)
    
    print(f"Stored {len(docs)} chunks to ChromaDB at {index_path}")

if __name__ == '__main__':
    build_chroma_index()
