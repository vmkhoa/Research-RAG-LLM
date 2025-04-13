from src.rag.retrieve import retrieve_top_k
from src.rag.generate import load_model, rag_generate

def main():
    index_path = 'data/vector_index'
    embed_model = "sentence-transformers/all-MiniLM-L6-v2"
    model_path = 'output'

    model, tokenizer = load_model(model_path)
    while True:
        query = input("\n Ask a question (or 'exit'): ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        top_chunks = retrieve_top_k(query, index_path=index_path, embed_model=embed_model, k=3)
        response = rag_generate(model, tokenizer, query, top_chunks)
        print("\n Answer:\n" + "="*40)
        print(response)


if __name__ == "__main__":
    main() 