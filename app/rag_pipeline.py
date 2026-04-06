from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import faiss
import numpy as np


# 🔹 Step 1: Load PDF
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    text = ""
    for doc in documents:
        text += doc.page_content + "\n"
    
    return text


# 🔹 Step 2: Split text
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_text(text)


# 🔹 Step 3: Create embeddings
def create_embeddings(chunks, model):
    return model.encode(chunks)


# 🔹 Step 4: Store in FAISS
def store_in_faiss(embeddings):
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))
    return index


# 🔹 Step 5: Retrieve relevant chunks
def retrieve_chunks(query, model, index, chunks, k=3):
    query_embedding = model.encode([query])
    
    distances, indices = index.search(
        np.array(query_embedding).astype("float32"), k
    )
    
    return [chunks[i] for i in indices[0]]


# 🔹 Step 6: Generate answer (IMPROVED VERSION)
def generate_answer(query, retrieved_chunks, tokenizer, model):
    # Use limited context for better quality
    context = "\n".join(retrieved_chunks[:2])

    # Query-aware instruction
    if "summarize" in query.lower():
        instruction = "Summarize the key ideas in simple language."
    elif "conclusion" in query.lower():
        instruction = "Explain the conclusion clearly in simple terms."
    else:
        instruction = "Answer clearly based on the context."

    prompt = f"""
You are a highly intelligent assistant.

{instruction}

IMPORTANT RULES:
- Do NOT copy text from the context
- Explain in your own words
- Keep the answer short and clear

Context:
{context}

Question: {query}

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.9
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)