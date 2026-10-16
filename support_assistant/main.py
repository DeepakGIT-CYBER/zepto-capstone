import os
from typing import TypedDict, List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import chromadb
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


# ---------- Vector DB ----------

embedder = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="zepto_policies"
)


# ---------- Load Documents ----------

if collection.count() == 0:
    for file in os.listdir("documents"):
        path = "documents/" + file

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        vector = embedder.encode(text).tolist()

        collection.add(
            ids=[file],
            documents=[text],
            embeddings=[vector]
        )


# ---------- Nodes ----------

def classify_intent(state: State):

    q = state["query"].lower()

    keywords = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
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
    "groceries"
]

    if any(k in q for k in keywords):
        state["intent"] = "policy_question"
    else:
        state["intent"] = "general_question"

    return state



def retrieve_and_answer(state: State):

    query_vector = embedder.encode(
        state["query"]
    ).tolist()

    result = collection.query(
        query_embeddings=[query_vector],
        n_results=3
    )

    chunk = result["documents"][0][0]
    source = result["ids"][0][0]


    state["answer"] = (
        "Based on the retrieved context: "
        + chunk
    )

    state["sources"] = [source]
    state["confidence"] = 1.0

    return state



def direct_answer(state: State):

    state["answer"] = (
        "I can only answer questions about Zepto policies right now."
    )

    state["sources"] = []
    state["confidence"] = 1.0

    return state



# ---------- Graph ----------

graph = StateGraph(State)

graph.add_node(
    "classify_intent",
    classify_intent
)

graph.add_node(
    "retrieve_and_answer",
    retrieve_and_answer
)

graph.add_node(
    "direct_answer",
    direct_answer
)


graph.set_entry_point(
    "classify_intent"
)


graph.add_conditional_edges(
    "classify_intent",
    lambda x: x["intent"],
    {
        "policy_question": "retrieve_and_answer",
        "general_question": "direct_answer"
    }
)


graph.add_edge(
    "retrieve_and_answer",
    END
)

graph.add_edge(
    "direct_answer",
    END
)


app_graph = graph.compile()


# ---------- FastAPI ----------

app = FastAPI(
    title="Zepto Support Assistant"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):

    result = app_graph.invoke(
        {
            "query": request.query,
            "intent": "",
            "answer": "",
            "sources": [],
            "confidence": 0
        }
    )

    return AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"]
    )
