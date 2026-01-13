#!/usr/bin/env python3
"""
Demo: RAG-Enhanced Model Selection
Shows how the system learns from past performance and makes intelligent model choices
"""
import os
import sys
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from services.model_selector import ModelSelector


def demo_basic_selection():
    """Demo 1: Basic model selection without RAG"""
    print("=" * 70)
    print("DEMO 1: Basic Model Selection (No RAG)")
    print("=" * 70)
    
    # Class method - backward compatible
    model, reason = ModelSelector.select_model("chat", verbose=True)
    print(f"\n✓ Selected: {model}")
    print(f"✓ Reason: {reason}\n")
    
    model, reason = ModelSelector.select_model("summarization", verbose=True)
    print(f"\n✓ Selected: {model}")
    print(f"✓ Reason: {reason}\n")


def demo_rag_selection_with_db():
    """Demo 2: RAG-enhanced selection with database tracking"""
    print("\n" + "=" * 70)
    print("DEMO 2: RAG-Enhanced Model Selection (With Performance Tracking)")
    print("=" * 70)
    
    # Simulate DB connection (in real usage, pass actual connection)
    print("\n📊 Simulating model performance history...\n")
    
    # For demo purposes, we'll show the API without actual DB
    print("selector = ModelSelector(db_connection=conn, redis_client=redis)")
    print("\n# Select model with RAG insights")
    print("model, reason, insights = selector.select_model_with_rag(")
    print("    task_type='chat',")
    print("    query_context='What is the company vacation policy?',")
    print("    user_id='hr_manager_001',")
    print("    verbose=True")
    print(")")
    
    print("\n📈 Expected RAG insights:")
    print("   • Historical performance: 150 past uses")
    print("   • Best performer: llama-3.3-70b-versatile (92% success rate)")
    print("   • User preference: llama-3.3-70b-versatile (used 45 times)")
    print("   • Similar contexts found: 12 related queries")
    print("   • Decision: Use llama-3.3-70b-versatile (RAG-optimized)\n")


def demo_performance_logging():
    """Demo 3: Logging performance for future RAG decisions"""
    print("\n" + "=" * 70)
    print("DEMO 3: Performance Logging for RAG Learning")
    print("=" * 70)
    
    print("\n💾 After each model use, log performance:\n")
    
    print("selector.log_performance(")
    print("    user_id='project_manager_001',")
    print("    task_type='chat',")
    print("    model_name='llama-3.3-70b-versatile',")
    print("    query_context='Show me team performance metrics',")
    print("    response_quality=0.95,  # Quality score 0-1")
    print("    latency_ms=1200,        # Response time")
    print("    token_count=450,        # Tokens used")
    print("    success=True,           # Success flag")
    print("    feedback_score=5        # User feedback 1-5")
    print(")")
    
    print("\n✅ Performance logged to database")
    print("   → Future selections will learn from this data")
    print("   → Better models get prioritized based on actual performance\n")


def demo_rag_workflow():
    """Demo 4: Complete RAG workflow"""
    print("\n" + "=" * 70)
    print("DEMO 4: Complete RAG Model Selection Workflow")
    print("=" * 70)
    
    print("\n🔄 RAG Selection Process:\n")
    
    steps = [
        ("1️⃣  RETRIEVE", [
            "Query database for historical performance",
            "Find models used for similar tasks",
            "Get user-specific preferences",
            "Check Redis cache for recent decisions"
        ]),
        ("2️⃣  ANALYZE", [
            "Calculate average success rates",
            "Compare response quality scores",
            "Consider latency and token costs",
            "Weight user feedback"
        ]),
        ("3️⃣  DECIDE", [
            "Start with default model from registry",
            "Override if RAG shows better performer",
            "Require >85% success + >80% quality to override",
            "Cache decision for similar future queries"
        ]),
        ("4️⃣  LOG", [
            "Track model used and task type",
            "Record response time and quality",
            "Store user feedback if available",
            "Build knowledge for next selection"
        ])
    ]
    
    for step_name, details in steps:
        print(f"{step_name}")
        for detail in details:
            print(f"   • {detail}")
        print()


def demo_benefits():
    """Demo 5: Benefits of RAG-based model selection"""
    print("=" * 70)
    print("DEMO 5: Benefits of RAG-Enhanced Model Selection")
    print("=" * 70)
    
    benefits = [
        ("🎯 Personalized Selection", "Learns user preferences and optimizes per user"),
        ("📊 Performance-Based", "Routes to models with proven success rates"),
        ("⚡ Context-Aware", "Uses similar past queries to inform decisions"),
        ("💰 Cost Optimization", "Balances quality with token usage"),
        ("🔄 Continuous Learning", "Gets better with every use"),
        ("⚡ Fast Decisions", "Redis caching for instant similar queries"),
        ("📈 Data-Driven", "Actual metrics instead of assumptions"),
        ("🎭 Task-Adaptive", "Different models excel at different tasks")
    ]
    
    print()
    for emoji_title, description in benefits:
        print(f"{emoji_title}")
        print(f"   {description}\n")


def demo_comparison():
    """Demo 6: Traditional vs RAG-based selection"""
    print("=" * 70)
    print("DEMO 6: Traditional vs RAG-Enhanced Selection")
    print("=" * 70)
    
    print("\n📋 TRADITIONAL MODEL SELECTION:")
    print("   • Static rules: Always use same model for task type")
    print("   • No learning: Doesn't improve over time")
    print("   • Generic: Same choice for all users")
    print("   • Ignores context: Doesn't consider query similarity")
    print("   • No feedback loop: Can't adapt to failures\n")
    
    print("🎯 RAG-ENHANCED MODEL SELECTION:")
    print("   • Dynamic routing: Best model based on actual data")
    print("   • Continuous learning: Improves with every use")
    print("   • Personalized: Adapts to user preferences")
    print("   • Context-aware: Uses similar past queries")
    print("   • Adaptive: Learns from successes and failures")
    print("   • Cached: Fast repeat decisions\n")
    
    print("📊 Example Improvement:")
    print("   Before RAG: 75% success rate, 2.5s avg latency")
    print("   After RAG:  92% success rate, 1.8s avg latency")
    print("   Improvement: +17% success, -28% latency\n")


def main():
    """Run all demos"""
    print("\n" + "=" * 70)
    print("🤖 RAG-ENHANCED MODEL SELECTION DEMO")
    print("Intelligent Model Routing with Historical Performance Learning")
    print("=" * 70)
    
    demo_basic_selection()
    demo_rag_selection_with_db()
    demo_performance_logging()
    demo_rag_workflow()
    demo_benefits()
    demo_comparison()
    
    print("=" * 70)
    print("✅ Demo Complete!")
    print("=" * 70)
    print("\n💡 To use in your code:")
    print("\n   from services.model_selector import ModelSelector")
    print("\n   # Initialize with DB and Redis")
    print("   selector = ModelSelector(db_connection=conn, redis_client=redis)")
    print("\n   # Use RAG-enhanced selection")
    print("   model, reason, insights = selector.select_model_with_rag(")
    print("       task_type='chat',")
    print("       query_context=user_message,")
    print("       user_id=current_user,")
    print("       verbose=True")
    print("   )")
    print("\n   # Log performance after use")
    print("   selector.log_performance(")
    print("       user_id=current_user,")
    print("       task_type='chat',")
    print("       model_name=model,")
    print("       query_context=user_message,")
    print("       response_quality=0.9,")
    print("       latency_ms=response_time,")
    print("       token_count=tokens_used,")
    print("       success=True")
    print("   )")
    print()


if __name__ == "__main__":
    main()
