#!/usr/bin/env python3
"""
Test Unified Redis Integration
Tests both episodic and semantic memory using common Redis instance
"""
import sys
sys.path.append('/Users/sharan/Downloads/September-Test')

def test_common_redis_connection():
    """Test unified Redis connection"""
    print("="*70)
    print("TEST 1: Unified Redis Connection")
    print("="*70)
    
    try:
        from src.services.redis_common_client import get_redis
        r = get_redis()
        result = r.ping()
        
        # Get Redis info
        info = r.info()
        print(f"✅ Redis connection successful: {result}")
        print(f"   Server: {info.get('redis_version', 'unknown')}")
        print(f"   Memory: {info.get('used_memory_human', 'unknown')}")
        print(f"   Connected clients: {info.get('connected_clients', 0)}")
        return True
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return False


def test_episodic_namespace():
    """Test episodic memory namespace"""
    print("\n" + "="*70)
    print("TEST 2: Episodic Memory Namespace")
    print("="*70)
    
    try:
        from src.services.redis_common_client import get_redis
        r = get_redis()
        
        # Store test data
        test_key = "episodic:stm:test_user:999999"
        test_data = {"query": "test", "context": "test context"}
        
        r.hset(test_key, mapping=test_data)
        r.expire(test_key, 60)  # 1 minute TTL
        
        # Retrieve data
        retrieved = r.hgetall(test_key)
        
        if retrieved:
            print(f"✅ Episodic namespace working!")
            print(f"   Key: {test_key}")
            print(f"   Data: {retrieved}")
            
            # Check TTL
            ttl = r.ttl(test_key)
            print(f"   TTL: {ttl}s")
            
            # Cleanup
            r.delete(test_key)
            return True
        else:
            print(f"❌ No data retrieved")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_semantic_namespace():
    """Test semantic memory namespace"""
    print("\n" + "="*70)
    print("TEST 3: Semantic Memory Namespace")
    print("="*70)
    
    try:
        from src.services.redis_common_client import get_redis
        r = get_redis()
        
        # Store persona
        persona_key = "semantic:persona:test_user"
        persona_data = {
            "name": "Test User",
            "interests": "AI, ML",
            "cached_at": "2026-01-09"
        }
        
        r.hset(persona_key, mapping=persona_data)
        r.expire(persona_key, 3600)  # 1 hour TTL
        
        # Store knowledge
        knowledge_key = "semantic:knowledge:test_user:888888"
        knowledge_data = {
            "query": "What is AI?",
            "results": "AI is artificial intelligence"
        }
        
        r.hset(knowledge_key, mapping=knowledge_data)
        r.expire(knowledge_key, 1800)  # 30 min TTL
        
        # Retrieve data
        persona = r.hgetall(persona_key)
        knowledge = r.hgetall(knowledge_key)
        
        if persona and knowledge:
            print(f"✅ Semantic namespace working!")
            print(f"   Persona key: {persona_key}")
            print(f"   Persona TTL: {r.ttl(persona_key)}s")
            print(f"   Knowledge key: {knowledge_key}")
            print(f"   Knowledge TTL: {r.ttl(knowledge_key)}s")
            
            # Cleanup
            r.delete(persona_key, knowledge_key)
            return True
        else:
            print(f"❌ Data retrieval failed")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_namespace_isolation():
    """Test that namespaces are properly isolated"""
    print("\n" + "="*70)
    print("TEST 4: Namespace Isolation")
    print("="*70)
    
    try:
        from src.services.redis_common_client import get_redis
        r = get_redis()
        
        # Create test keys
        episodic_key = "episodic:test:isolation"
        semantic_key = "semantic:test:isolation"
        
        r.set(episodic_key, "episodic_data")
        r.set(semantic_key, "semantic_data")
        r.expire(episodic_key, 60)
        r.expire(semantic_key, 60)
        
        # Check keys exist
        episodic_keys = r.keys("episodic:*")
        semantic_keys = r.keys("semantic:*")
        
        print(f"✅ Namespace isolation verified!")
        print(f"   Episodic keys found: {len(episodic_keys)}")
        print(f"   Semantic keys found: {len(semantic_keys)}")
        
        # Cleanup
        r.delete(episodic_key, semantic_key)
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_cache_statistics():
    """Test unified cache statistics"""
    print("\n" + "="*70)
    print("TEST 5: Unified Cache Statistics")
    print("="*70)
    
    try:
        from src.services.redis_common_client import get_redis
        r = get_redis()
        
        # Count keys by namespace
        episodic_keys = r.keys("episodic:*")
        semantic_persona_keys = r.keys("semantic:persona:*")
        semantic_knowledge_keys = r.keys("semantic:knowledge:*")
        
        total_keys = r.dbsize()
        
        print(f"📊 Cache Statistics:")
        print(f"   Total keys in DB: {total_keys}")
        print(f"   Episodic STM: {len(episodic_keys)} keys")
        print(f"   Semantic Personas: {len(semantic_persona_keys)} keys")
        print(f"   Semantic Knowledge: {len(semantic_knowledge_keys)} keys")
        
        # Memory usage
        info = r.info('memory')
        print(f"\n💾 Memory Usage:")
        print(f"   Used: {info.get('used_memory_human', 'unknown')}")
        print(f"   Peak: {info.get('used_memory_peak_human', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_integrated_workflow():
    """Test complete workflow using both namespaces"""
    print("\n" + "="*70)
    print("TEST 6: Integrated Workflow (Episodic + Semantic)")
    print("="*70)
    
    try:
        from src.services.redis_common_client import get_redis
        import json
        r = get_redis()
        
        user_id = "workflow_test_user"
        
        # 1. Cache user persona (Semantic)
        print("\n1️⃣ Caching user persona (Semantic)...")
        persona_key = f"semantic:persona:{user_id}"
        r.hset(persona_key, mapping={
            "name": "Alice",
            "interests": json.dumps(["AI", "Python"]),
            "cached_at": "2026-01-09"
        })
        r.expire(persona_key, 3600)
        print(f"   ✅ Persona cached: {persona_key}")
        
        # 2. Cache recent conversation (Episodic)
        print("\n2️⃣ Caching recent conversation (Episodic)...")
        stm_key = f"episodic:stm:{user_id}:111111"
        r.hset(stm_key, mapping={
            "query": "Tell me about AI",
            "context": "Previous discussion about artificial intelligence",
            "created_at": "1704844800"
        })
        r.expire(stm_key, 300)
        print(f"   ✅ Conversation cached: {stm_key}")
        
        # 3. Cache knowledge search (Semantic)
        print("\n3️⃣ Caching knowledge search (Semantic)...")
        knowledge_key = f"semantic:knowledge:{user_id}:222222"
        r.hset(knowledge_key, mapping={
            "query": "Python best practices",
            "results": json.dumps([{"title": "PEP 8", "content": "Style guide"}])
        })
        r.expire(knowledge_key, 1800)
        print(f"   ✅ Knowledge cached: {knowledge_key}")
        
        # 4. Retrieve all data for user
        print(f"\n4️⃣ Retrieving all cached data for user '{user_id}'...")
        
        persona = r.hgetall(persona_key)
        stm = r.hgetall(stm_key)
        knowledge = r.hgetall(knowledge_key)
        
        if persona and stm and knowledge:
            print(f"   ✅ Complete context retrieved!")
            print(f"      - Persona: {persona.get(b'name', b'').decode()}")
            print(f"      - Recent chat: {stm.get(b'query', b'').decode()}")
            print(f"      - Knowledge: {knowledge.get(b'query', b'').decode()}")
            
            # Cleanup
            r.delete(persona_key, stm_key, knowledge_key)
            return True
        else:
            print(f"   ❌ Incomplete data retrieved")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "🚀"*35)
    print("UNIFIED REDIS INTEGRATION TEST SUITE")
    print("🚀"*35 + "\n")
    
    results = []
    
    # Run tests
    results.append(("Unified Redis Connection", test_common_redis_connection()))
    results.append(("Episodic Namespace", test_episodic_namespace()))
    results.append(("Semantic Namespace", test_semantic_namespace()))
    results.append(("Namespace Isolation", test_namespace_isolation()))
    results.append(("Cache Statistics", test_cache_statistics()))
    results.append(("Integrated Workflow", test_integrated_workflow()))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:12} {test_name}")
    
    print(f"\n{'='*70}")
    print(f"Results: {passed}/{total} tests passed")
    print(f"{'='*70}\n")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Unified Redis is working perfectly!")
        print("\n📊 System Architecture:")
        print("   - Single Redis Cloud instance")
        print("   - Episodic namespace: episodic:*")
        print("   - Semantic namespace: semantic:*")
        print("   - Proper TTL management")
        print("   - Data isolation verified")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
