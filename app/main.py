from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from app.rag_pipeline import (
    load_pdf,
    split_text,
    create_embeddings,
    store_in_faiss,
    retrieve_chunks,
    generate_answer_local,
    generate_answer_groq
)


def main():
    file_path = "data/sample.pdf"

    # 🔹 Step 1: Load PDF
    text = load_pdf(file_path)
    print("📄 Text length:", len(text))

    # 🔹 Step 2: Chunking
    chunks = split_text(text)
    print("📦 Total chunks:", len(chunks))

    # 🔹 Step 3: Load embedding model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # 🔹 Step 4: Create embeddings
    embeddings = create_embeddings(chunks, embedding_model)
    print("🧠 Embeddings created:", len(embeddings))

    # 🔹 Step 5: Store in FAISS
    index = store_in_faiss(embeddings)
    print("✅ FAISS index ready")

    # 🔹 Step 6: Choose mode
    mode = input("\nChoose mode (local/groq): ").strip().lower()

    if mode == "local":
        print("\n⏳ Loading local LLM...")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    else:
        print("\n⚡ Using Groq API (no local model loading)")

    print("\n🤖 System ready! Ask questions (type 'exit' to quit)\n")

    # 🔹 Step 7: Query loop
    while True:
        query = input("❓ Your question: ")

        if query.lower() == "exit":
            break

        results = retrieve_chunks(query, embedding_model, index, chunks)

        if mode == "groq":
            answer = generate_answer_groq(query, results)
        else:
            answer = generate_answer_local(query, results, tokenizer, model)

        print("\n🤖 Answer:\n")
        print(answer)
        print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    main()