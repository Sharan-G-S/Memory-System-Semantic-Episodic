# Redis Integration - Actual Redis Server

## ✅ Successfully Implemented

The temporary memory cache now uses **actual Redis** instead of Python's deque.

## What Changed

### Before (deque)
```python
from collections import deque
self.temp_memory = deque(maxlen=15)
```

### After (Redis)
```python
import redis
self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
```

## Installation & Setup

### 1. Install Redis Server
```bash
# macOS
brew install redis

# Start Redis
brew services start redis

# Verify running
redis-cli ping  # Should return PONG
```

### 2. Install Python Client
```bash
pip install redis
```

### 3. Configure (Optional)
Add to `.env` file:
```
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
```

## Features

### ✅ Persistent Storage
- Data survives app restarts (Redis persists to disk)
- Configurable TTL (default: 24 hours)

### ✅ Distributed Support
- Can be shared across multiple app instances
- Remote Redis server support

### ✅ Better Performance
- Dedicated in-memory database
- Optimized for cache operations
- Built-in data structures

### ✅ Automatic Cleanup
- LTRIM keeps only last 15 messages
- EXPIRE sets 24-hour TTL
- Memory management handled by Redis

## How It Works

### Storage Structure
```
Key: temp_memory:{user_id}:messages
Type: Redis List
Max Size: 15 messages
TTL: 24 hours
```

### Operations

**Load Messages:**
```python
cache_key = f"temp_memory:{user_id}:messages"
messages = redis_client.lrange(cache_key, 0, -1)
```

**Add Message:**
```python
# Add to end
redis_client.rpush(cache_key, json.dumps(message))

# Keep only last 15
redis_client.ltrim(cache_key, -15, -1)

# Refresh TTL
redis_client.expire(cache_key, 86400)
```

**Search:**
```python
temp_messages = get_temp_memory()
results = [msg for msg in temp_messages if query in msg['content']]
```

## Benefits Over Deque

| Feature | Deque | Redis |
|---------|-------|-------|
| **Persistence** | ❌ Lost on restart | ✅ Survives restarts |
| **Distributed** | ❌ Single process | ✅ Multiple processes |
| **TTL Support** | ❌ Manual cleanup | ✅ Automatic expiry |
| **Remote Access** | ❌ Local only | ✅ Network accessible |
| **Memory Management** | ⚠️ Python heap | ✅ Redis memory limits |
| **Performance** | ✅ Very fast | ✅ Very fast |

## Testing

Run the test:
```bash
python3 -c "
from interactive_memory_app import InteractiveMemorySystem
app = InteractiveMemorySystem()
print('Redis status:', 'Connected ✅' if app.redis_client else 'Disconnected ❌')
"
```

## Fallback Mode

If Redis is not available, the app gracefully degrades:
```
⚠️  Redis not available - temporary cache disabled
```

The app will still work but without the temporary cache feature.

## Monitoring Redis

### Check cache contents:
```bash
redis-cli

# List all keys
KEYS temp_memory:*

# View specific user's cache
LRANGE temp_memory:team_lead_001:messages 0 -1

# Check TTL
TTL temp_memory:team_lead_001:messages
```

### Monitor operations:
```bash
redis-cli MONITOR
```

## Configuration Options

### Custom Redis Settings
Edit `.env`:
```
# Use remote Redis
REDIS_HOST=redis.example.com
REDIS_PORT=6379

# Use different database
REDIS_DB=1

# Use password (if needed)
REDIS_PASSWORD=your_password
```

Update code to use password:
```python
self.redis_client = redis.Redis(
    host=redis_host,
    port=redis_port,
    db=redis_db,
    password=os.getenv('REDIS_PASSWORD'),
    decode_responses=True
)
```

## Production Recommendations

1. **Enable Persistence:** Configure Redis with AOF or RDB
2. **Set Memory Limits:** Configure maxmemory in redis.conf
3. **Use Redis Sentinel:** For high availability
4. **Monitor Performance:** Use Redis INFO commands
5. **Secure Access:** Use passwords and firewall rules

## Troubleshooting

### Redis not connecting
```bash
# Check if Redis is running
redis-cli ping

# Start Redis
brew services start redis

# Check logs
brew services info redis
```

### Cache not updating
```bash
# Flush specific key
redis-cli DEL temp_memory:team_lead_001:messages

# Restart app to reload from database
```

### Memory issues
```bash
# Check memory usage
redis-cli INFO memory

# Clear all cache (careful!)
redis-cli FLUSHDB
```

## Summary

✅ **Fully functional Redis integration**
- 15-message cache per user
- 24-hour TTL
- Automatic cleanup
- Graceful fallback if Redis unavailable
- Production-ready

🚀 **Ready to use!** Just run the app - Redis works automatically.
