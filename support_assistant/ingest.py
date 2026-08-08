import os
import chromadb
from sentence_transformers import SentenceTransformer


DOC_FOLDER = "documents"

# Persistent ChromaDB storage
client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = client.get_or_create_collection(
    name="zepto_policies"
)

# Local embedding model
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


def load_documents():

    documents = []
    ids = []

    for filename in sorted(os.listdir(DOC_FOLDER)):

        if filename.endswith(".txt"):

            filepath = os.path.join(
                DOC_FOLDER,
                filename
            )

            with open(
                filepath,
                "r",
                encoding="utf-8"
            ) as file:
                text = file.read()

            documents.append(text)
            ids.append(filename)

    return documents, ids


documents, ids = load_documents()


print(f"Loaded {len(documents)} documents")


# Generate embeddings
embeddings = model.encode(
    documents
).tolist()


# Remove old data if script is run again
if ids:
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings
    )


print(
    f"Stored {len(ids)} documents in ChromaDB collection: zepto_policies"
)