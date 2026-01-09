# redis_common_client.py
"""
Unified Redis client for both Episodic and Semantic memory
Uses Redis Cloud with proper namespacing for different data types
"""
import os
from redis import Redis
from dotenv import load_dotenv

load_dotenv()

def get_redis_common():
    """
    Get unified Redis connection for both episodic and semantic memory
    Uses Redis Cloud with namespaced keys:
    - episodic:stm:{user_id}:* - Short-term memory (recent chats)
    - semantic:persona:{user_id} - User personas
    - semantic:knowledge:{user_id}:* - Knowledge base searches
    """
    host = os.getenv("REDIS_HOST", "localhost")
    port = os.getenv("REDIS_PORT", "6379")
    password = os.getenv("REDIS_PASSWORD", "")
    db = os.getenv("REDIS_DB", "0")

    if not host or not port:
        raise RuntimeError("❌ Missing Redis env vars (REDIS_HOST, REDIS_PORT)")

    # Configure Redis connection
    kwargs = {
        "host": host,
        "port": int(port),
        "db": int(db),
        "decode_responses": False,  # REQUIRED for binary vectors
        "socket_connect_timeout": 5,
        "socket_timeout": 5,
        "retry_on_timeout": True
    }
    
    # Add password if provided
    if password:
        kwargs["password"] = password

    return Redis(**kwargs)


# Singleton instance
_redis_instance = None

def get_redis():
    """Get or create Redis singleton instance"""
    global _redis_instance
    if _redis_instance is None:
        _redis_instance = get_redis_common()
    return _redis_instance
