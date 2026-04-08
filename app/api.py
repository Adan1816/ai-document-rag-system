from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer

from app.rag_pipeline import (
    load_pdf,
    split_text,
    create_embeddings,
    store_in_faiss,
    retrieve_chunks,
    generate_answer_local,
    generate_answer_groq
)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# 🔹 Create app
app = FastAPI(title="RAG AI API")


# 🔹 Request schema
class QueryRequest(BaseModel):
    question: str
    mode: str = "groq"


# 🔹 Global storage (for uploaded PDF)
user_data = {
    "chunks": None,
    "index": None
}


# 🔹 Global models (loaded once)
embedding_model = None
tokenizer = None
model = None


# 🔹 Startup
@app.on_event("startup")
def startup_event():
    global embedding_model, tokenizer, model

    print("🚀 Loading models...")

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    print("✅ Models ready!")


# 🔹 Root route (for browser test)
@app.get("/")
def home():
    return {"message": "RAG API is running 🚀"}


# 🔹 Upload PDF endpoint
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global user_data

    content = await file.read()

    # Save temporarily
    with open("temp.pdf", "wb") as f:
        f.write(content)

    # Process PDF
    text = load_pdf("temp.pdf")
    chunks = split_text(text)

    embeddings = create_embeddings(chunks, embedding_model)
    index = store_in_faiss(embeddings)

    # Store in memory
    user_data["chunks"] = chunks
    user_data["index"] = index

    return {
        "message": "PDF uploaded and processed successfully!",
        "chunks": len(chunks)
    }


# 🔹 Ask question endpoint
@app.post("/ask")
def ask_question(request: QueryRequest):
    query = request.question
    mode = request.mode.lower()

    # Check if PDF uploaded
    if user_data["index"] is None:
        return {"error": "Please upload a PDF first."}

    # Retrieve relevant chunks
    results = retrieve_chunks(
        query,
        embedding_model,
        user_data["index"],
        user_data["chunks"]
    )

    # Generate answer
    if mode == "groq":
        answer = generate_answer_groq(query, results)
    else:
        answer = generate_answer_local(query, results, tokenizer, model)

    return {
        "question": query,
        "mode": mode,
        "answer": answer
    }