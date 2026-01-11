#!/usr/bin/env python3
"""
Test script to verify Redis and PostgreSQL connections
"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_postgresql():
    """Test PostgreSQL connection"""
    print("Testing PostgreSQL connection...")
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
        
        # Get connection details
        dbname = os.getenv("PG_DB") or os.getenv("DB_NAME", "semantic_memory")
        user = os.getenv("PG_USER") or os.getenv("DB_USER", "postgres")
        password = os.getenv("PG_PASSWORD") or os.getenv("DB_PASSWORD", "")
        host = os.getenv("PG_HOST") or os.getenv("DB_HOST", "localhost")
        port = os.getenv("PG_PORT") or os.getenv("DB_PORT", "5432")
        
        print(f"  Host: {host}")
        print(f"  Port: {port}")
        print(f"  Database: {dbname}")
        print(f"  User: {user}")
        
        # Try to connect
        conn = psycopg2.connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port,
            cursor_factory=RealDictCursor,
            connect_timeout=5
        )
        
        # Test query
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        
        print(f"  ✅ PostgreSQL connected successfully!")
        print(f"  Version: {version['version'][:50]}...")
        
        # Check for tables
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        tables = cursor.fetchall()
        
        if tables:
            print(f"  Tables found: {len(tables)}")
            for table in tables[:5]:
                print(f"    - {table['table_name']}")
            if len(tables) > 5:
                print(f"    ... and {len(tables) - 5} more")
        else:
            print("  ⚠️  No tables found in database")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"  ❌ PostgreSQL connection failed: {e}")
        return False


def test_redis():
    """Test Redis connection"""
    print("\nTesting Redis connection...")
    try:
        import redis
        
        # Get connection details
        host = os.getenv("REDIS_HOST", "localhost")
        port = os.getenv("REDIS_PORT", "6379")
        password = os.getenv("REDIS_PASSWORD", "")
        db = os.getenv("REDIS_DB", "0")
        
        print(f"  Host: {host}")
        print(f"  Port: {port}")
        print(f"  DB: {db}")
        
        # Try to connect
        kwargs = {
            "host": host,
            "port": int(port),
            "db": int(db),
            "decode_responses": True,
            "socket_connect_timeout": 5,
            "socket_timeout": 5
        }
        
        if password:
            kwargs["password"] = password
        
        client = redis.Redis(**kwargs)
        
        # Test connection
        client.ping()
        
        # Get info
        info = client.info()
        print(f"  ✅ Redis connected successfully!")
        print(f"  Version: {info['redis_version']}")
        print(f"  Used memory: {info['used_memory_human']}")
        
        # Check for keys
        keys = client.keys('*')
        print(f"  Total keys: {len(keys)}")
        
        # Show sample keys by prefix
        prefixes = {}
        for key in keys[:100]:  # Sample first 100 keys
            prefix = key.split(':')[0] if ':' in key else 'no-prefix'
            prefixes[prefix] = prefixes.get(prefix, 0) + 1
        
        if prefixes:
            print("  Key prefixes:")
            for prefix, count in sorted(prefixes.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"    - {prefix}: {count} keys")
        
        client.close()
        return True
        
    except Exception as e:
        print(f"  ❌ Redis connection failed: {e}")
        return False


def main():
    """Main test function"""
    print("=" * 60)
    print("Testing Database Connections")
    print("=" * 60)
    
    pg_ok = test_postgresql()
    redis_ok = test_redis()
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"PostgreSQL: {'✅ Connected' if pg_ok else '❌ Failed'}")
    print(f"Redis:      {'✅ Connected' if redis_ok else '❌ Failed'}")
    
    if pg_ok and redis_ok:
        print("\n🎉 All connections successful!")
        return 0
    else:
        print("\n⚠️  Some connections failed. Check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
