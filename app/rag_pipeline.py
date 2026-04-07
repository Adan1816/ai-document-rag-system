from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import faiss
import numpy as np

from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()


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


# 🔹 Step 6A: Local LLM (HuggingFace)
def generate_answer_local(query, retrieved_chunks, tokenizer, model):
    context = "\n".join(retrieved_chunks[:2])

    prompt = f"""
You are an expert assistant.

Explain the answer in simple words.

IMPORTANT:
- Do NOT copy text
- Use your own words
- Be clear and concise

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
        top_p=0.9
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# 🔹 Step 6B: Groq LLM (API)
def generate_answer_groq(query, retrieved_chunks):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    context = "\n".join(retrieved_chunks[:2])

    prompt = f"""
You are an expert assistant.

Explain the answer in simple terms.

IMPORTANT:
- Do NOT copy text
- Use your own words
- Be clear and concise

Context:
{context}

Question: {query}

Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content