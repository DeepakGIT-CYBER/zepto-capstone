\# Zepto Customer Support Assistant



\## Overview



A document-grounded GenAI customer support assistant for Zepto.



The system uses Retrieval Augmented Generation (RAG) to answer customer questions using Zepto policy documents stored locally.



The application uses:



\- Sentence Transformers (`all-MiniLM-L6-v2`) for embeddings

\- ChromaDB for vector storage and retrieval

\- LangGraph for query routing workflow

\- FastAPI for API serving



\---



\# RAG Pipeline Architecture



The complete pipeline:



