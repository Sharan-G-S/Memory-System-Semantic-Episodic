# redis_stm_index.py
from redis_client import get_redis

r = get_redis()

INDEX = "episodic_stm_idx"
DIM = 384

def create_index():
    """
    Create Redis search index for Episodic STM using RediSearch.
    Note: Requires Redis Stack or RediSearch module.
    """
    try:
        # Check if index exists
        r.execute_command("FT.INFO", INDEX)
        print("✅ Episodic STM index exists")
        return
    except Exception:
        pass

    try:
        # Create index with vector field for episodic namespace (using FLAT for simpler hybrid search)
        r.execute_command(
            "FT.CREATE", INDEX,
            "ON", "HASH",
            "PREFIX", "1", "episodic:stm:",
            "SCHEMA",
            "query_vector", "VECTOR", "FLAT", "6",
            "TYPE", "FLOAT32",
            "DIM", str(DIM),
            "DISTANCE_METRIC", "COSINE",
            "created_at", "NUMERIC", "SORTABLE"
        )
        print("🚀 Episodic STM index created")
    except Exception as e:
        print(f"⚠️  Could not create Redis search index: {e}")
        print("   Redis Stack or RediSearch module required for vector search")
        print("   Install: brew install redis-stack")

if __name__ == "__main__":
    create_index()
