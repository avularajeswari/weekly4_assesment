
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import os
import json

from app.rag import VECTORSTORE_PATH
from app.classifier import classify_ticket
from app.tools import get_product_details, get_warranty_details

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

app = FastAPI(title="RAG Assistant API")

client = OpenAI(
    base_url=os.getenv("OLLAMA_BASE_URL"),
    api_key="ollama"
)

MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory=str(VECTORSTORE_PATH),
    embedding_function=embeddings,
    collection_name="product_manual"
)


class AskRequest(BaseModel):
    question: str


class ClassifyRequest(BaseModel):
    ticket: str


class ChatRequest(BaseModel):
    message: str


RAG_PROMPT = """
You are a helpful product manual assistant.

Answer ONLY using the provided context.
Do not use outside knowledge or make up information.
If the answer is not available, return exactly: I don't know.

Context:
{context}

Question:
{question}

Answer:
"""


@app.get("/")
def home():
    return {"message": "RAG Assistant API is running"}


@app.post("/ask")
def ask_question(request: AskRequest):
    retrieved_docs = vectorstore.similarity_search(
        request.question, k=3
    )

    if not retrieved_docs:
        return {"answer": "I don't know.", "sources": []}

    context = "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )

    prompt = RAG_PROMPT.format(
        context=context,
        question=request.question
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    answer = response.choices[0].message.content.strip()

    sources = sorted({
        doc.metadata["page"] + 1
        for doc in retrieved_docs
        if "page" in doc.metadata
    })

    if "i don't know" in answer.lower():
        return {"answer": "I don't know.", "sources": []}

    return {
        "answer": answer,
        "sources": [{"page": page} for page in sources]
    }


@app.post("/classify")
def classify(request: ClassifyRequest):
    category = classify_ticket(request.ticket)
    return {"category": category}


@app.post("/chat")
def chat(request: ChatRequest):

    message = request.message
    lower_message = message.lower()

    # Let the model decide which tool to use.
    decision = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """
Choose the best action for the user's message.

Return ONLY one of these exact values:
product_tool
warranty_tool
rag

Use product_tool for product specifications.
Use warranty_tool for warranty questions.
Use rag for questions about instructions, troubleshooting,
usage, setup, or other manual information.
"""
            },
            {"role": "user", "content": message}
        ],
        temperature=0
    )

    action = decision.choices[0].message.content.strip().lower()

    if "product_tool" in action:
        result = get_product_details()
        return {
            "method": "product_tool",
            "answer": result
        }

    if "warranty_tool" in action:
        result = get_warranty_details()
        return {
            "method": "warranty_tool",
            "answer": result
        }

    # Use RAG when the model chooses rag.
    retrieved_docs = vectorstore.similarity_search(
        message, k=3
    )

    if not retrieved_docs:
        return {
            "method": "rag",
            "answer": "I don't know.",
            "sources": []
        }

    context = "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )

    prompt = RAG_PROMPT.format(
        context=context,
        question=message
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    answer = response.choices[0].message.content.strip()

    if "i don't know" in answer.lower():
        return {
            "method": "rag",
            "answer": "I don't know.",
            "sources": []
        }

    sources = sorted({
        doc.metadata["page"] + 1
        for doc in retrieved_docs
        if "page" in doc.metadata
    })

    return {
        "method": "rag",
        "answer": answer,
        "sources": [{"page": page} for page in sources]
    }
