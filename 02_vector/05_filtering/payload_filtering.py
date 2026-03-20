from qdrant_client import QdrantClient, models
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

client = QdrantClient(host="localhost", port=6333)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Filtering By Exact Match
results = client.query_points(
    collection_name="tano_collection",
    query=model.encode("AI").tolist(),
    query_filter=Filter(
        must=[
            FieldCondition(
                key="category",
                match=MatchValue(value="aspiration")
            )
        ]
    ),
    limit=2
)

for r in results.points:
    print(f"Score: {r.score:.4f} | {r.payload['text'][:60]}")