from qdrant_client import QdrantClient, models
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

model = SentenceTransformer("all-MiniLM-L6-v2")

# Example Documents
docs = [
    "Dogs are loyal and friendly domestic animals.",                  # text_1
    "Cats are independent and curious creatures.",                    # text_2
    "The Milky Way galaxy contains over 200 billion stars."           # text_3
    ]

embedding = model.encode(docs)

query = "What animals make good pets?"
query_embedding = model.encode(query)

# Calculate "Similarity" between query and each document
for idx, doc_embedding in enumerate(embedding, start=1):
    score = model.similarity(query_embedding, doc_embedding)
    print(f"text_{idx} similarity: {score}")
