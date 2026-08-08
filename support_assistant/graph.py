import os
from typing import TypedDict, List

import chromadb
from sentence_transformers import SentenceTransformer

from langgraph.graph import StateGraph, END

from models import AskResponse


# -----------------------------
# ChromaDB setup
# -----------------------------

client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = client.get_collection(
    name="zepto_policies"
)

embed_model = None
# -----------------------------
# Graph State
# -----------------------------

class GraphState(TypedDict):
    query: str
    intent: str
    answer: str
    sources: List[str]
    confidence: float


# -----------------------------
# Node 1
# Intent Classification
# -----------------------------

def classify_intent(state: GraphState):

    query = state["query"].lower()

    keywords = [
        "delivery",
        "return",
        "refund",
        "membership",
        "tracking",
        "cancel",
        "gift card",
        "support hours",
        "fee",
        "charge",
        "cost"
    ]

    if os.getenv("MOCK_LLM", "1") == "1":

        if any(word in query for word in keywords):
            intent = "policy_question"
        else:
            intent = "general_question"

    else:
        intent = "general_question"


    state["intent"] = intent

    return state


# -----------------------------
# Node 2
# Retrieval + Answer
# -----------------------------

def retrieve_and_answer(state: GraphState):

    global embed_model

    if embed_model is None:
        embed_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    query = state["query"]
    query = state["query"]

    query_embedding = embed_model.encode(
        query
    ).tolist()


    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )


    documents = results["documents"][0]
    ids = results["ids"][0]
    distances = results["distances"][0]


    print("\n===== RETRIEVAL DEBUG =====")

    for i, doc in enumerate(documents):

        print("----------------")
        print("SOURCE:", ids[i])
        print("DISTANCE:", distances[i])
        print(doc[:300])


    top_chunk = documents[0]


    answer = (
        "Based on the retrieved context: "
        + top_chunk[:300]
    )


    state["answer"] = answer
    state["sources"] = ids
    state["confidence"] = 1.0


    return state



# -----------------------------
# Node 3
# Direct Answer
# -----------------------------

def direct_answer(state: GraphState):

    state["answer"] = (
        "I can only answer questions about Zepto policies right now."
    )

    state["sources"] = []

    state["confidence"] = 1.0

    return state



# -----------------------------
# Router
# -----------------------------

def route_question(state: GraphState):

    if state["intent"] == "policy_question":

        return "retrieve_and_answer"

    return "direct_answer"



# -----------------------------
# Build LangGraph
# -----------------------------

workflow = StateGraph(GraphState)


workflow.add_node(
    "classify_intent",
    classify_intent
)


workflow.add_node(
    "retrieve_and_answer",
    retrieve_and_answer
)


workflow.add_node(
    "direct_answer",
    direct_answer
)



workflow.set_entry_point(
    "classify_intent"
)



workflow.add_conditional_edges(
    "classify_intent",
    route_question,
    {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer"
    }
)



workflow.add_edge(
    "retrieve_and_answer",
    END
)


workflow.add_edge(
    "direct_answer",
    END
)



graph = workflow.compile()



# -----------------------------
# Public Function
# -----------------------------

def ask_zepto(query: str):

    print("\nDEBUG: Starting graph")


    result = graph.invoke(
        {
            "query": query,
            "intent": "",
            "answer": "",
            "sources": [],
            "confidence": 0.0
        }
    )


    print("\nDEBUG RESULT:")
    print(result)



    response = AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"]
    )


    return response