
import os
from typing import TypedDict, List

import chromadb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END


# ---------- Models ----------

class State(TypedDict):
    query: str
    intent: str
    answer: str
    sources: List[str]
    confidence: float


class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float = Field(ge=0, le=1)


# ---------- Vector Database ----------

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="zepto_policies"
)


# ---------- Load Documents ----------

DOCUMENTS_DIR = "documents"

if collection.count() == 0:
    if os.path.isdir(DOCUMENTS_DIR):
        for filename in os.listdir(DOCUMENTS_DIR):
            path = os.path.join(DOCUMENTS_DIR, filename)

            if not os.path.isfile(path):
                continue

            if not filename.lower().endswith(".txt"):
                continue

            with open(path, "r", encoding="utf-8") as file:
                content = file.read().strip()

            if content:
                collection.add(
                    ids=[filename],
                    documents=[content]
                )


# ---------- Intent Classification ----------

def classify_intent(state: State):
    query = state["query"].lower()

    keywords = [
        "delivery",
        "return",
        "refund",
        "membership",
        "tracking",
        "cancel",
        "cancellation",
        "gift card",
        "support hours",
        "damaged",
        "damage",
        "spoiled",
        "incorrect",
        "wrong item",
        "missing",
        "expired",
        "perishable",
        "grocery",
        "groceries",
        "order",
        "late",
        "payment",
        "replacement",
        "complaint"
    ]

    if any(keyword in query for keyword in keywords):
        state["intent"] = "policy_question"
    else:
        state["intent"] = "general_question"

    return state


# ---------- Retrieve Policy ----------

def retrieve_and_answer(state: State):
    if collection.count() == 0:
        state["answer"] = (
            "No policy documents are available yet. "
            "Please check the documents folder."
        )
        state["sources"] = []
        state["confidence"] = 0.0
        return state

    try:
        result = collection.query(
            query_texts=[state["query"]],
            n_results=min(3, collection.count())
        )

        documents = result.get("documents") or [[]]
        ids = result.get("ids") or [[]]

        if not documents[0]:
            state["answer"] = (
                "Sorry, I could not find a matching Zepto policy."
            )
            state["sources"] = []
            state["confidence"] = 0.0
            return state

        chunks = documents[0]
        sources = ids[0]

        state["answer"] = (
            "Based on the retrieved policy context:\n\n"
            + "\n\n".join(chunks)
        )
        state["sources"] = list(dict.fromkeys(sources))
        state["confidence"] = 1.0

    except Exception:
        state["answer"] = (
            "Sorry, I could not retrieve the policy right now. "
            "Please try again."
        )
        state["sources"] = []
        state["confidence"] = 0.0

    return state


# ---------- Out-of-Scope Answer ----------

def direct_answer(state: State):
    state["answer"] = (
        "I can currently answer questions about Zepto policies, "
        "such as delivery, returns, refunds, cancellations, "
        "missing items, and damaged products."
    )
    state["sources"] = []
    state["confidence"] = 1.0

    return state


# ---------- LangGraph Workflow ----------

graph = StateGraph(State)

graph.add_node("classify_intent", classify_intent)
graph.add_node("retrieve_and_answer", retrieve_and_answer)
graph.add_node("direct_answer", direct_answer)

graph.set_entry_point("classify_intent")

graph.add_conditional_edges(
    "classify_intent",
    lambda state: state["intent"],
    {
        "policy_question": "retrieve_and_answer",
        "general_question": "direct_answer"
    }
)

graph.add_edge("retrieve_and_answer", END)
graph.add_edge("direct_answer", END)

app_graph = graph.compile()


# ---------- FastAPI Application ----------

app = FastAPI(
    title="Zepto Support Assistant",
    description="A policy question-answering API for Zepto support.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/")
def home():
    return {
        "message": "Zepto Support Assistant is running.",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "policy_documents": collection.count()
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    query = request.query.strip()

    if not query:
        return AskResponse(
            answer="Please enter a question.",
            sources=[],
            confidence=0.0
        )

    result = app_graph.invoke({
        "query": query,
        "intent": "",
        "answer": "",
        "sources": [],
        "confidence": 0.0
    })

    return AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"]
    )
