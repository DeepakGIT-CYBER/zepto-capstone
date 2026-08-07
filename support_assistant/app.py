import os
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple

import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

APP_TITLE = "Zepto Customer Support Assistant"

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"

SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".pdf",
    ".docx",
    ".json",
}


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🛒",
    layout="wide",
)


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            color: #666666;
            margin-bottom: 1.5rem;
        }

        .source-box {
            border-left: 4px solid #888888;
            padding: 0.6rem 1rem;
            margin: 0.5rem 0;
            background-color: #f7f7f7;
            border-radius: 4px;
        }

        .warning-box {
            padding: 1rem;
            border-radius: 6px;
            background-color: #fff8e1;
            border: 1px solid #ffe082;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DOCUMENT READING
# ============================================================

def read_text_file(path: Path) -> str:
    """Read TXT/MD files."""
    return path.read_text(encoding="utf-8", errors="ignore")


def read_json_file(path: Path) -> str:
    """Read JSON and convert it into searchable text."""
    try:
        data = json.loads(
            path.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        )

        return json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        )

    except Exception:
        return ""


def read_pdf_file(path: Path) -> str:
    """Read text from a PDF."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))

        pages = []

        for page in reader.pages:
            try:
                text = page.extract_text() or ""
                pages.append(text)
            except Exception:
                continue

        return "\n".join(pages)

    except Exception as error:
        return f"[Could not read PDF: {error}]"


def read_docx_file(path: Path) -> str:
    """Read text from a DOCX document."""
    try:
        from docx import Document

        document = Document(str(path))

        paragraphs = []

        for paragraph in document.paragraphs:
            text = paragraph.text.strip()

            if text:
                paragraphs.append(text)

        return "\n".join(paragraphs)

    except Exception as error:
        return f"[Could not read DOCX: {error}]"


def read_document(path: Path) -> str:
    """Read a supported document."""
    extension = path.suffix.lower()

    if extension in {".txt", ".md"}:
        return read_text_file(path)

    if extension == ".json":
        return read_json_file(path)

    if extension == ".pdf":
        return read_pdf_file(path)

    if extension == ".docx":
        return read_docx_file(path)

    return ""


# ============================================================
# DOCUMENT CHUNKING
# ============================================================

def clean_text(text: str) -> str:
    """Clean excessive whitespace."""
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = 900,
    overlap: int = 150,
) -> List[str]:
    """
    Split long documents into overlapping chunks.

    This provides a simple retrieval layer without requiring
    a vector database.
    """

    text = clean_text(text)

    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks = []

    start = 0

    while start < len(text):
        end = min(
            start + chunk_size,
            len(text),
        )

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(
            end - overlap,
            start + 1,
        )

    return chunks


# ============================================================
# LOAD KNOWLEDGE BASE
# ============================================================

@st.cache_data(show_spinner=False)
def load_documents() -> List[Dict]:
    """
    Load documents from the documents directory.
    """

    DOCUMENTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []

    for path in sorted(DOCUMENTS_DIR.rglob("*")):

        if not path.is_file():
            continue

        if path.name.startswith("."):
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        text = read_document(path)

        if not text:
            continue

        chunks = chunk_text(text)

        for index, chunk in enumerate(chunks):

            records.append(
                {
                    "source": path.name,
                    "path": str(path),
                    "chunk_id": index + 1,
                    "text": chunk,
                }
            )

    return records


# ============================================================
# RETRIEVAL
# ============================================================

STOP_WORDS = {
    "the",
    "is",
    "are",
    "was",
    "were",
    "what",
    "when",
    "where",
    "how",
    "why",
    "can",
    "could",
    "would",
    "should",
    "do",
    "does",
    "did",
    "i",
    "me",
    "my",
    "we",
    "our",
    "you",
    "your",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "from",
    "about",
    "please",
}


def tokenize(text: str) -> List[str]:
    """Convert text into simple lowercase tokens."""

    words = re.findall(
        r"[a-zA-Z0-9₹$£€]+",
        text.lower(),
    )

    return [
        word
        for word in words
        if word not in STOP_WORDS
    ]


def retrieve_documents(
    query: str,
    documents: List[Dict],
    top_k: int = 5,
) -> List[Dict]:
    """
    Simple lexical retrieval.

    The query is compared against each document chunk
    using token overlap and phrase matching.
    """

    if not documents:
        return []

    query_clean = query.lower().strip()

    query_tokens = set(
        tokenize(query)
    )

    if not query_tokens:
        return documents[:top_k]

    scored = []

    for document in documents:

        text = document["text"].lower()

        document_tokens = set(
            tokenize(text)
        )

        overlap = len(
            query_tokens.intersection(
                document_tokens
            )
        )

        score = float(overlap)

        # Bonus if the complete query appears.
        if query_clean and query_clean in text:
            score += 10

        # Bonus for multiple matching words.
        if overlap >= 3:
            score += 2

        if score > 0:
            scored.append(
                (
                    score,
                    document,
                )
            )

    scored.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return [
        document
        for _, document in scored[:top_k]
    ]


# ============================================================
# FALLBACK ANSWER
# ============================================================

def build_fallback_answer(
    query: str,
    retrieved: List[Dict],
) -> str:
    """
    Extractive fallback when no LLM API key is available.

    This deliberately avoids inventing policy.
    """

    if not retrieved:
        return (
            "I could not find this information in the "
            "available Zepto policy documents.\n\n"
            "Please provide the relevant policy document "
            "or contact customer support."
        )

    answer_parts = [
        "I found the following relevant information "
        "in the available policy documents:"
    ]

    for document in retrieved[:3]:

        answer_parts.append(
            f"\n**Source: {document['source']}**\n"
            f"{document['text'][:700]}"
        )

    answer_parts.append(
        "\n\nThis response is based only on the "
        "retrieved document content."
    )

    return "\n".join(answer_parts)


# ============================================================
# OPENAI RESPONSE
# ============================================================

def generate_ai_answer(
    query: str,
    retrieved: List[Dict],
    api_key: str,
    model: str,
) -> str:
    """
    Generate an answer using OpenAI when an API key
    is supplied.

    The model is instructed to answer only from the
    retrieved policy context.
    """

    if not api_key:
        return build_fallback_answer(
            query,
            retrieved,
        )

    if not retrieved:
        return (
            "I could not find a relevant Zepto policy "
            "document for this question. I don't want "
            "to guess or invent a policy."
        )

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key,
        )

        context_parts = []

        for document in retrieved:

            context_parts.append(
                (
                    f"SOURCE: {document['source']}\n"
                    f"CHUNK: {document['chunk_id']}\n"
                    f"CONTENT:\n{document['text']}"
                )
            )

        context = "\n\n---\n\n".join(
            context_parts
        )

        instructions = """
You are a customer support assistant for Zepto.

Answer the customer's question using ONLY the
provided policy/document context.

Rules:

1. Do not invent policies.
2. Do not make up prices, refunds, delivery times,
   eligibility rules, or compensation.
3. If the documents do not contain enough information,
   clearly say that the available policy documents
   do not provide the answer.
4. Give a concise and helpful answer.
5. If a policy has conditions, mention the important
   conditions.
6. Do not claim that a policy exists unless it appears
   in the provided context.
7. When useful, mention the source document name.
"""

        user_input = f"""
CUSTOMER QUESTION:

{query}

POLICY DOCUMENT CONTEXT:

{context}
"""

        response = client.responses.create(
            model=model,
            instructions=instructions,
            input=user_input,
        )

        answer = response.output_text.strip()

        if not answer:
            return build_fallback_answer(
                query,
                retrieved,
            )

        return answer

    except Exception as error:

        return (
            "The AI service could not be reached.\n\n"
            f"Technical detail: {error}\n\n"
            "I can still show the relevant policy "
            "document excerpts below."
        )


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🛒 Zepto Customer Support Assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Policy-grounded customer support assistant"
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    st.write(
        "Add your Zepto policy documents to:"
    )

    st.code(
        str(DOCUMENTS_DIR),
        language="text",
    )

    api_key = st.text_input(
        "OpenAI API key (optional)",
        type="password",
        help=(
            "If supplied, the app will use an AI model "
            "to generate answers from retrieved policy text."
        ),
    )

    model = st.selectbox(
        "AI model",
        options=[
            "gpt-5-mini",
            "gpt-5",
        ],
        index=0,
    )

    top_k = st.slider(
        "Number of policy chunks to retrieve",
        min_value=1,
        max_value=8,
        value=5,
    )

    st.divider()

    if st.button(
        "🔄 Reload documents",
        use_container_width=True,
    ):
        load_documents.clear()
        st.rerun()

    if st.button(
        "🗑️ Clear chat",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.caption(
        "The assistant is designed to answer from "
        "provided policy documents and avoid "
        "inventing unsupported policies."
    )


# ============================================================
# LOAD DOCUMENTS
# ============================================================

documents = load_documents()


# ============================================================
# KNOWLEDGE BASE STATUS
# ============================================================

with st.expander(
    "📚 Knowledge Base",
    expanded=False,
):

    if documents:

        source_names = sorted(
            {
                document["source"]
                for document in documents
            }
        )

        st.success(
            f"{len(source_names)} document(s) loaded "
            f"with {len(documents)} searchable chunks."
        )

        for source in source_names:
            st.write(f"• {source}")

    else:

        st.warning(
            "No policy documents are currently loaded."
        )

        st.write(
            "Add TXT, MD, PDF, DOCX, or JSON files to:"
        )

        st.code(
            str(DOCUMENTS_DIR),
            language="text",
        )


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# CHAT INPUT
# ============================================================

query = st.chat_input(
    "Ask a question about Zepto policies..."
)


if query:

    # --------------------------------------------------------
    # USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    # --------------------------------------------------------
    # ASSISTANT RESPONSE
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching policy documents..."
        ):

            retrieved = retrieve_documents(
                query,
                documents,
                top_k=top_k,
            )

        with st.spinner(
            "Preparing answer..."
        ):

            answer = generate_ai_answer(
                query=query,
                retrieved=retrieved,
                api_key=api_key,
                model=model,
            )

        st.markdown(answer)

        # ----------------------------------------------------
        # SOURCES
        # ----------------------------------------------------

        if retrieved:

            st.markdown(
                "### 📚 Sources used"
            )

            seen_sources = set()

            for document in retrieved:

                source = document["source"]

                if source in seen_sources:
                    continue

                seen_sources.add(source)

                st.markdown(
                    '<div class="source-box">'
                    f"<strong>{source}</strong><br>"
                    f"Chunk {document['chunk_id']}"
                    "</div>",
                    unsafe_allow_html=True,
                )

        else:

            st.info(
                "No relevant policy document was found."
            )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Zepto Customer Support Assistant • "
    "Document-grounded support prototype"
)