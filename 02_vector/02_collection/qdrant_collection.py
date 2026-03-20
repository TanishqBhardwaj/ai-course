from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

client = QdrantClient(host="localhost", port=6333)
model = SentenceTransformer("all-MiniLM-L6-v2")

DIM = model.get_sentence_embedding_dimension()

# Creating a collection
def create_collection(name: str, distance: models.Distance):
    if not client.collection_exists(collection_name=name):
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=DIM,
                distance=distance
            )
        )
        print(f"Collection '{name}' created successfully.")
    else:
        print(f"Collection '{name}' already exists.")

create_collection("tano_collection1", models.Distance.COSINE)

# Inspecting a collection
info = client.get_collection("tano_collection1")
print(f"""Collection: '{info}'""")

# Listing collections
collections = client.get_collections()
print(f"Collections: {collections}")

# Deleting a collection
#client.delete_collection("my_collection")


client.close()