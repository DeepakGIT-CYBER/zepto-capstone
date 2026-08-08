# Zepto Customer Support Assistant

## Overview

A document-grounded customer support assistant for Zepto.

The system uses Retrieval Augmented Generation (RAG) principles to answer customer questions using locally stored Zepto policy documents.

The assistant can answer questions related to:

- Delivery
- Order tracking
- Order cancellation
- Refunds
- Damaged products
- Missing products
- Wrong products
- Delivery delays

Questions outside the supported Zepto policy domain are rejected instead of generating unrelated answers.

---

## Features

- Document-grounded customer support
- Retrieval Augmented Generation (RAG)
- Local policy document storage
- Semantic search using Sentence Transformers
- Vector search using ChromaDB
- Query routing using LangGraph
- FastAPI backend
- Web-based chatbot interface
- Source document display
- Retrieval confidence score
- Out-of-scope question handling
- Concise customer-friendly responses

---

## Technologies Used

- Python
- FastAPI
- Uvicorn
- LangGraph
- ChromaDB
- Sentence Transformers
- HTML
- CSS
- JavaScript

### Embedding Model

The project uses:

`all-MiniLM-L6-v2`

from Sentence Transformers for generating semantic embeddings.

---

## RAG Pipeline Architecture

The complete pipeline is:

```text
User Question
      |
      v
Web Chat Interface
      |
      v
FastAPI /ask Endpoint
      |
      v
LangGraph Workflow
      |
      v
Intent Classification
      |
      +----------------------+----------------------+
      |                                             |
      | Policy Question                             | General Question
      v                                             v
ChromaDB Retrieval                         Out-of-Scope Response
      |
      v
Sentence Transformer
Embedding Search
      |
      v
Relevant Policy Document
      |
      v
Concise Policy Answer
      |
      v
Source + Confidence
      |
      v
Web Chat Interface