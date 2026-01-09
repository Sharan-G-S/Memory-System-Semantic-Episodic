# redis_client.py
"""
Redis client for Episodic Memory using unified Redis instance
Data is namespaced with 'episodic:' prefix
"""
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from services.redis_common_client import get_redis
except ImportError:
    # Fallback for direct imports
    from redis import Redis
    from dotenv import load_dotenv
    load_dotenv()
    
    def get_redis():
        """Fallback Redis connection"""
        host = os.getenv("REDIS_HOST", "localhost")
        port = os.getenv("REDIS_PORT", "6379")
        password = os.getenv("REDIS_PASSWORD", "")
        db = os.getenv("REDIS_DB", "0")
        
        kwargs = {
            "host": host,
            "port": int(port),
            "db": int(db),
            "decode_responses": False
        }
        
        if password:
            kwargs["password"] = password
        
        return Redis(**kwargs)

