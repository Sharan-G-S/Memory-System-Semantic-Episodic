# redis_stm_index.py
from redis_client import get_redis

r = get_redis()

INDEX = "stm_idx"
DIM = 384

def create_index():
    """
    Create Redis search index for STM using RediSearch.
    Note: Requires Redis Stack or RediSearch module.
    """
    try:
        # Check if index exists
        r.execute_command("FT.INFO", INDEX)
        print("✅ STM index exists")
        return
    except Exception:
        pass

    try:
        # Create index with vector field
        r.execute_command(
            "FT.CREATE", INDEX,
            "ON", "HASH",
            "PREFIX", "1", "stm:",
            "SCHEMA",
            "query_vector", "VECTOR", "HNSW", "6",
            "TYPE", "FLOAT32",
            "DIM", str(DIM),
            "DISTANCE_METRIC", "COSINE",
            "created_at", "NUMERIC", "SORTABLE"
        )
        print("🚀 STM index created")
    except Exception as e:
        print(f"⚠️  Could not create Redis search index: {e}")
        print("   Redis Stack or RediSearch module required for vector search")
        print("   Install: brew install redis-stack")

if __name__ == "__main__":
    create_index()
