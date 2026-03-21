import os
import requests
import getpass
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
)
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

model = SentenceTransformer("all-MiniLM-L6-v2")

GITHUB_RAW_URL = "https://raw.githubusercontent.com/tnahddisttud/sample-doc/refs/heads/main/atliqai_hr_policies.txt"

# "path" = no server needed for demos
# Production use: QdrantClient(url="http://localhost:6333")
client = QdrantClient(url="http://localhost:6333")

COLLECTION_NAME = "chat_bot_collection5"

groq_client = Groq()
GROQ_MODEL  = "openai/gpt-oss-safeguard-20b"

SYSTEM_PROMPT = """You are a helpful HR assistant.
Answer the user's question using ONLY the context provided below.
If the context does not contain enough information, say so — do not make things up.
Always cite the section name when referencing specific information."""

def create_collection(client: QdrantClient, collection: str) -> None:
    DIM = model.get_sentence_embedding_dimension()
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(
            size=DIM,
            distance=Distance.COSINE,
        ),
    )

def load_document(url: str) -> str:
    """Fetch a plain-text file from a raw GitHub URL."""
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.text

raw_text = load_document(GITHUB_RAW_URL)
print(f"Loaded {len(raw_text):,} characters")

CHUNK_SIZE = 50

def parse_word_chunks(text: str, chunk_size: int = CHUNK_SIZE) -> list[dict]:
    # Strip markdown heading symbols and blank lines
    clean_lines = []
    for line in text.splitlines():
        line = line.strip().lstrip("#").strip()
        if line:
            clean_lines.append(line)

    # Join everything into one word list and slice
    words = " ".join(clean_lines).split()

    chunks = []
    for i in range(0, len(words), chunk_size):
        content = " ".join(words[i : i + chunk_size])
        chunks.append({
            "chunk_index": len(chunks),
            "content": content,
        })

    return chunks

chunks = parse_word_chunks(raw_text)
print(f"Total chunks: {len(chunks)}")

def build_chunk_text(chunk: dict) -> str:
    return chunk["content"]

# Extract Chunk Texts
chunk_texts = [build_chunk_text(c) for c in chunks]

print(f"Embedding {len(chunk_texts)} chunks …")
embeddings = model.encode(chunk_texts, show_progress_bar=True)

create_collection(client, COLLECTION_NAME)

# Creating Points
points = [
    PointStruct(
        id=idx,
        vector=embedding.tolist(),
        payload={
            "content": build_chunk_text(chunk),
        },
    )
    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings))
]

result = client.upsert(
    collection_name=COLLECTION_NAME,
    points=points,
    wait=True,   # Block until indexing completes before returning
)

print(f"Indexed {len(points)} points — status: {result.status}")

def retrieve(
    query: str,
    top_k: int = 5
) -> list[dict]:
    """
    Embed the query and return the top-k most similar chunks.

    Args:
        query          : User's question.
        top_k          : Number of chunks to return.
        section_filter : Optional H2 heading to restrict the search scope.
    """
    query_vector = model.encode(query).tolist()

    hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )

    return [{**hit.payload, "score": round(hit.score, 4)} for hit in hits.points]

def build_context(retrieved_chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        parts.append(f"[Source {i}]\n{chunk['content']}")
    return "\n\n---\n\n".join(parts)

def rag(query: str, top_k: int = 5):
    """
    End-to-end RAG pipeline:
      1. Retrieve relevant chunks from Qdrant
      2. Format them as a context block
      3. Send context + query to Groq and return the answer
    """
    # Step 1 — Retrieve
    chunks = retrieve(query, top_k=top_k)
    if not chunks:
        return "No relevant content found in the document."

    # Step 2 — Build context
    context = build_context(chunks)

    # Step 3 — Generate
    user_message = f"Context:\n{context}\n\nQuestion: {query}"

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.2,   # Low = factual;  High = creative
    )
    return response.choices[0].message.content, context

answer, context = rag("What are the main topics covered in this document?")
print(answer)
print(f"{250*'='}")
print(f"\n\nSOURCES:\n {context}")