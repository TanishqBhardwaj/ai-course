from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

client = QdrantClient(host="localhost", port=6333)
model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    {"id": 1, "text": "My name is Tano.",
     "category": "introduction", "role": "public"},
    {"id": 2, "text": "I like to code.",
     "category": "hobby", "role": "public"},
    {"id": 3, "text": "I want to learn AI.",
     "category": "aspiration", "role": "admin"},
]

# Creating Embeddings from text
texts    = [doc["text"] for doc in documents]
vectors  = model.encode(texts)

# Creating Points with Payload (Qdrant's Datastructure)
points = [
    models.PointStruct(
        id      = doc["id"],
        vector  = vectors[i].tolist(),
        payload = {
            "text"    : doc["text"],
            "category": doc["category"],
            "role"    : doc["role"],
        }
    )
    for i, doc in enumerate(documents)
]

# Upserting Points
operation_info = client.upsert(
    collection_name="tano_collection",
    wait=True,
    points=points
)

print(f"Status: {operation_info.status}")

# Performing Semantic Search
query     = "Who loves AI?"
query_vec = model.encode(query).tolist()

results = client.query_points(
    collection_name="tano_collection",
    query=query_vec,
    limit=2,
    score_threshold=0.35
)

for r in results.points:
    print(f"Score: {r.score:.4f} | {r.payload['text']}")

# client.close()