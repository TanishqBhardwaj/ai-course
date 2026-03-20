from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

client = QdrantClient(host="localhost", port=6333)
model = SentenceTransformer("all-MiniLM-L6-v2")

new_text   = "My name is Tano."
new_vector = model.encode(new_text).tolist()

CREATE 
client.upsert(
    collection_name="tano_collection1",
    points=[
        models.PointStruct(
            id      = 0,
            vector  = new_vector,
            payload = {"text": new_text, "category": "introduction", "role": "public"}
        )
    ]
)
print("Point 0 inserted.")

#READ
point = client.retrieve(
    collection_name="tano_collection1",
    ids=[0],
    with_payload=True,
    with_vectors=True
)

# UPDATE: Modify payload only
client.set_payload(
    collection_name="tano_collection1",
    payload={"verified": True, "updated_by": "admin"},
    points=[0]
)

# Checking
updated = client.retrieve("tano_collection1", ids=[0], with_payload=True)
print("Updated payload:", updated[0].payload)