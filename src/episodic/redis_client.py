# redis_client.py
import os
from redis import Redis
from dotenv import load_dotenv

load_dotenv()

def get_redis():
    """Get Redis connection with proper configuration"""
    host = os.getenv("REDIS_HOST", "localhost")
    port = os.getenv("REDIS_PORT", "6379")
    password = os.getenv("REDIS_PASSWORD", "")

    if not host or not port:
        raise RuntimeError("❌ Missing Redis env vars (REDIS_HOST, REDIS_PORT)")

    # Only include password if it's not empty
    kwargs = {
        "host": host,
        "port": int(port),
        "decode_responses": False  # REQUIRED for binary vectors
    }
    
    if password:
        kwargs["username"] = "default"
        kwargs["password"] = password

    return Redis(**kwargs)
