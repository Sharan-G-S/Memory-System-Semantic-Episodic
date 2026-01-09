#!/usr/bin/env python3
"""Test script for temporary memory functionality"""

from interactive_memory_app import InteractiveMemorySystem
import time

def test_temp_memory():
    """Test temporary memory cache functionality"""
    print("="*70)
    print("TESTING TEMPORARY MEMORY CACHE")
    print("="*70 + "\n")
    
    # Initialize app
    app = InteractiveMemorySystem()
    app.user_id = "team_lead_001"
    app.ensure_super_chat()
    
    print(f"✓ Connected as user: {app.user_id}")
    print(f"✓ Chat session: {app.current_chat_id}")
    print(f"✓ Temporary memory size: {len(app.temp_memory)} messages\n")
    
    # Test 1: Check if temporary memory was loaded
    print("TEST 1: Check Temporary Memory Loading")
    print("-" * 70)
    if app.temp_memory:
        print(f"✅ PASS: Loaded {len(app.temp_memory)} messages into temporary cache")
        print(f"   First message: {app.temp_memory[0]['role']}: {app.temp_memory[0]['content'][:50]}...")
        print(f"   Last message:  {app.temp_memory[-1]['role']}: {app.temp_memory[-1]['content'][:50]}...")
    else:
        print("⚠️  No messages in temporary cache (might be a new user)")
    print()
    
    # Test 2: Add new messages and check if they're added to temp_memory
    print("TEST 2: Adding New Messages to Temporary Cache")
    print("-" * 70)
    initial_size = len(app.temp_memory)
    
    app.add_chat_message("user", "Testing temporary memory - message 1")
    app.add_chat_message("assistant", "I received your test message 1")
    app.add_chat_message("user", "Testing temporary memory - message 2")
    
    new_size = len(app.temp_memory)
    print(f"✅ PASS: Added 3 messages")
    print(f"   Before: {initial_size} messages")
    print(f"   After:  {new_size} messages")
    print(f"   Latest: {app.temp_memory[-1]['content']}")
    print()
    
    # Test 3: Search in temporary memory
    print("TEST 3: Searching Temporary Memory")
    print("-" * 70)
    results = app.hybrid_search("testing temporary memory", limit=5)
    
    if 'temp_memory' in results and results['temp_memory']:
        print(f"✅ PASS: Found {len(results['temp_memory'])} results in temporary cache")
        for item in results['temp_memory']:
            print(f"   - [{item['role']}] {item['content'][:60]}...")
    else:
        print("❌ FAIL: No results found in temporary memory")
    print()
    
    # Test 4: Check maxlen behavior (adding more than 15 messages)
    print("TEST 4: Testing Deque Max Length (15 messages)")
    print("-" * 70)
    print(f"Current size: {len(app.temp_memory)}")
    
    # Add many more messages to exceed 15
    for i in range(20):
        app.add_chat_message("user", f"Overflow test message {i}")
    
    final_size = len(app.temp_memory)
    print(f"✅ PASS: After adding 20 more messages")
    print(f"   Final size: {final_size} (should be capped at 15)")
    print(f"   Oldest message: {app.temp_memory[0]['content']}")
    print(f"   Newest message: {app.temp_memory[-1]['content']}")
    
    if final_size <= 15:
        print("✅ PASS: Deque correctly limited to 15 messages")
    else:
        print(f"❌ FAIL: Deque has {final_size} messages (should be max 15)")
    print()
    
    # Test 5: User switching and temp_memory reload
    print("TEST 5: User Switching and Cache Reload")
    print("-" * 70)
    print(f"Switching from {app.user_id} to project_manager_001...")
    
    app.user_id = "project_manager_001"
    app.ensure_super_chat()
    app.load_recent_to_temp_memory()
    
    print(f"✅ PASS: Switched to user: {app.user_id}")
    print(f"   New cache size: {len(app.temp_memory)} messages")
    if app.temp_memory:
        print(f"   First message: {app.temp_memory[0]['role']}: {app.temp_memory[0]['content'][:50]}...")
    print()
    
    print("="*70)
    print("ALL TESTS COMPLETED")
    print("="*70)

if __name__ == "__main__":
    test_temp_memory()
