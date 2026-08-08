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

    query = state["query"].lower().strip()

    policy_keywords = [
        "delivery",
        "deliver",
        "late",
        "delay",
        "delayed",
        "arrive",
        "arrival",
        "tracking",
        "track",
        "refund",
        "money back",
        "damaged",
        "damage",
        "wrong product",
        "missing product",
        "missing",
        "return",
        "cancel",
        "cancellation",
        "membership",
        "fee",
        "charge",
        "cost",
        "payment",
        "order",
    ]

    if any(keyword in query for keyword in policy_keywords):
        state["intent"] = "policy_question"
    else:
        state["intent"] = "general_question"

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

    query_embedding = embed_model.encode(
        query
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=1
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

    # -----------------------------
    # No result
    # -----------------------------

    if not documents:
        state["answer"] = (
            "I couldn't find relevant information in the Zepto policies."
        )

        state["sources"] = []
        state["confidence"] = 0.0

        return state

    # -----------------------------
    # Calculate confidence
    # -----------------------------

    top_distance = float(distances[0])

    confidence = 1.0 / (1.0 + top_distance)

    # -----------------------------
    # Get best matching document
    # -----------------------------

    top_chunk = documents[0]

    # -----------------------------
    # Create concise answer
    # -----------------------------

    query_lower = query.lower()

    if "delivery" in query_lower or "deliver" in query_lower:
        answer = (
            "Zepto aims to deliver most orders within 10 to 20 minutes, "
            "depending on your location. Delivery may take longer due to "
            "weather, traffic, high demand, or operational issues."
        )

    elif (
        "track" in query_lower
        or "tracking" in query_lower
    ):
        answer = (
            "You can track your Zepto order in real time using the "
            "Zepto application."
        )

    elif (
        "cancel" in query_lower
        or "cancellation" in query_lower
    ):
        answer = (
            "You can cancel your Zepto order before it is packed. "
            "Once the order has been packed or dispatched, cancellation "
            "may not be possible. If cancellation is successful, the "
            "refund will be processed to the original payment method."
        )

    elif (
        "refund" in query_lower
        or "money back" in query_lower
    ):
        answer = (
            "You may request a refund if the wrong product was delivered, "
            "the product is damaged, the product is missing, or your order "
            "was cancelled but payment was deducted. Refund requests should "
            "be submitted through the Zepto app within 48 hours of delivery. "
            "Approved refunds are credited to the original payment method "
            "within 5 to 7 business days."
        )

    elif (
        "damaged" in query_lower
        or "damage" in query_lower
    ):
        answer = (
            "If your product is damaged, you can request a refund through "
            "the Zepto app within 48 hours of delivery. If approved, the "
            "refund will be credited to your original payment method within "
            "5 to 7 business days."
        )

    elif (
        "wrong product" in query_lower
        or "missing product" in query_lower
        or "missing" in query_lower
    ):
        answer = (
            "If you received the wrong product or a product is missing, "
            "you can request a refund through the Zepto app within 48 hours "
            "of delivery. Approved refunds are credited to the original "
            "payment method within 5 to 7 business days."
        )

    elif (
        "delay" in query_lower
        or "delayed" in query_lower
        or "late" in query_lower
    ):
        answer = (
            "Zepto delivery may take longer than usual because of weather "
            "conditions, traffic, high demand, or operational issues. "
            "You can track your order in real time using the Zepto app."
        )

    else:
        answer = (
            "Based on the Zepto policy:\n\n"
            + top_chunk
        )

    state["answer"] = answer
    state["sources"] = ids
    state["confidence"] = confidence

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