from qdrant_client import QdrantClient, models
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

client = QdrantClient(host="localhost", port=6333)
model = SentenceTransformer("all-MiniLM-L6-v2")

COLLECTION_NAME = "tano_collection"

def get_role_filter(user_role: str):
    """
    Returns a Qdrant Filter that restricts results to
    what the given role is allowed to see.
    """
    if user_role == "admin":
        return None
    elif user_role == "public":
        return Filter(
            must=[
                FieldCondition(key="role", match=MatchValue(value="public"))
            ]
        )
    else:
        raise ValueError(f"Unknown role: {user_role}")

def role_aware_search(query: str, user_role: str, top_k: int = 3):
    query_vec    = model.encode(query).tolist()
    role_filter  = get_role_filter(user_role)
 
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        query_filter=role_filter,
        limit=top_k
    )
 
    print(f"\n--- Results for role='{user_role}', query='{query}' ---")
    for r in results.points:
        print(f"  Score: {r.score:.4f} | role: {r.payload['role']:6} | {r.payload['text'][:55]}")
    return results

# Public user: should NOT see admin-only content
role_aware_search("Who is Tano?", user_role="public")