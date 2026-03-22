import os

from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker import HierarchicalChunker
from hierarchical.postprocessor import ResultPostprocessor

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
)
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

SOURCE = "https://raw.githubusercontent.com/tnahddisttud/sample-doc/refs/heads/main/AtliqAI_HR_Policies.pdf"
client = QdrantClient(url="http://localhost:6333")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
COLLECTION_NAME = "tano_collection5"
groq_client = Groq()   # Reads GROQ_API_KEY from environment automatically
GROQ_MODEL  = "openai/gpt-oss-safeguard-20b"

def load_document(source: str):
    """
    Parse a PDF using Docling.
    Returns a DoclingDocument object — not a plain string.
    """
    converter = DocumentConverter()
    result = converter.convert(source)
    ResultPostprocessor(result).process()
    return result.document

doc = load_document(SOURCE)

chunker   = HierarchicalChunker()
doc_chunks = list(chunker.chunk(doc))

def convert_chunk(doc_chunk) -> dict:
    """
    Convert a Docling DocChunk into a plain dict.

    headings   → list preserved as-is
    content    → paragraph text
    chunk_text → breadcrumb + content  (what gets embedded)
    """
    headings   = doc_chunk.meta.headings or []
    content    = doc_chunk.text.strip()
    breadcrumb = " > ".join(headings)
    chunk_text = f"{breadcrumb}\n\n{content}" if breadcrumb else content

    return {
        "headings":   headings,
        "content":    content,
        "chunk_text": chunk_text,
    }

chunks = [convert_chunk(c) for c in doc_chunks]
chunk_texts = [c["chunk_text"] for c in chunks]
embeddings = embedder.encode(chunk_texts, show_progress_bar=True)

DIM = embedder.get_sentence_embedding_dimension()

client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=DIM,
        distance=Distance.COSINE,
    ),
)

# Creating Points
points = [
    PointStruct(
        id=idx,
        vector=embedding.tolist(),
        payload={
            "headings":   chunk["headings"],   # stored as a JSON array
            "content":    chunk["content"],
            "chunk_text": chunk["chunk_text"],
        },
    )
    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings))
]

result = client.upsert(
    collection_name=COLLECTION_NAME,
    points=points,
    wait=True,
)

info = client.get_collection(COLLECTION_NAME)

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
    query_vector = embedder.encode(query).tolist()

    hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )

    return [{**hit.payload, "score": round(hit.score, 4)} for hit in hits.points]

SYSTEM_PROMPT = """You are a helpful HR assistant.
Answer the user's question using ONLY the context provided below.
If the context does not contain enough information, say so — do not make things up.
Always cite the section name when referencing specific information."""

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

answer, context = rag("How many casual leaves am I entitled to?")
print(answer)
print(f"{250*'='}")
print(f"\n\nSOURCES:\n {context}")