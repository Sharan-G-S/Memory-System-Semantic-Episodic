#!/usr/bin/env python3
"""
Test Context Optimization System
Demonstrates deduplication, entropy filtering, compression, and re-ranking
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.services.context_optimizer import ContextOptimizer, SummarizationOptimizer
import numpy as np


def test_basic_optimization():
    """Test basic optimization pipeline"""
    print("=" * 70)
    print("TEST 1: Basic Context Optimization")
    print("=" * 70)
    
    # Sample contexts with duplicates and low-quality content
    contexts = [
        {
            "content": "The user likes to play basketball on weekends. They enjoy outdoor sports and physical activities.",
            "score": 0.9
        },
        {
            "content": "The user likes to play basketball on weekends. They enjoy outdoor sports and physical activities.",
            "score": 0.85  # Exact duplicate
        },
        {
            "content": "User enjoys basketball and outdoor activities during weekends.",
            "score": 0.88  # Similar content
        },
        {
            "content": "Hello.",  # Low entropy
            "score": 0.5
        },
        {
            "content": "The user's favorite programming language is Python. They use it for data science and machine learning projects.",
            "score": 0.7
        },
        {
            "content": "Python is the user's preferred language. They work on ML and data science.",
            "score": 0.72  # Similar to above
        },
        {
            "content": "aaaaaaaaaaa",  # Very low entropy
            "score": 0.3
        },
        {
            "content": "The user works as a software engineer at a tech company in San Francisco.",
            "score": 0.65
        }
    ]
    
    query = "What does the user do for work and hobbies?"
    
    optimizer = ContextOptimizer(
        similarity_threshold=0.85,
        entropy_threshold=0.4,
        min_info_content=15,
        max_context_tokens=500,
        rerank_threshold=0.5,
        max_iterations=2
    )
    
    optimized, stats = optimizer.optimize(contexts, query)
    
    print(f"\n📊 Original Contexts: {stats['original_count']}")
    print(f"   Estimated tokens: {stats['original_tokens']}")
    print(f"\n🗑️  Removed:")
    print(f"   Exact duplicates: {stats['duplicates_removed']}")
    print(f"   Low entropy: {stats['low_entropy_removed']}")
    print(f"\n✨ Processing:")
    print(f"   Compressed: {stats['compressed_count']}")
    print(f"   Re-ranking iterations: {stats['iterations']}")
    print(f"\n✅ Final Result: {stats['final_count']} contexts")
    print(f"   Estimated tokens: {stats['final_tokens']}")
    print(f"   Token reduction: {stats['reduction_percentage']:.1f}%")
    
    print(f"\n📝 Optimized Contexts:")
    for i, ctx in enumerate(optimized, 1):
        relevance = ctx.get('relevance_score', 0)
        compressed = " [COMPRESSED]" if ctx.get('compressed') else ""
        truncated = " [TRUNCATED]" if ctx.get('truncated') else ""
        print(f"\n   {i}. Relevance: {relevance:.3f}{compressed}{truncated}")
        print(f"      {ctx['content'][:100]}...")


def test_summarization():
    """Test aggressive summarization"""
    print("\n\n" + "=" * 70)
    print("TEST 2: Aggressive Summarization")
    print("=" * 70)
    
    contexts = [
        {
            "content": "The user is a software engineer working at Google in Mountain View. They specialize in machine learning and artificial intelligence."
        },
        {
            "content": "The user graduated from Stanford University with a degree in Computer Science. They completed their studies in 2020."
        },
        {
            "content": "User's hobbies include playing basketball, hiking, and photography. They go hiking every weekend in the Bay Area mountains."
        },
        {
            "content": "The user is proficient in Python, Java, and C++. They primarily use Python for machine learning projects."
        },
        {
            "content": "User has published 3 research papers on neural networks. They presented at NeurIPS conference last year."
        }
    ]
    
    query = "Tell me about the user's background and interests"
    
    summarizer = SummarizationOptimizer(compression_ratio=0.4)
    summary = summarizer.summarize_contexts(contexts, query)
    
    original_length = sum(len(ctx['content']) for ctx in contexts)
    summary_length = len(summary.get('content', ''))
    
    print(f"\n📊 Original: {len(contexts)} contexts, {original_length} characters")
    print(f"✨ Summary: 1 context, {summary_length} characters")
    print(f"📉 Compression: {summary.get('compression_ratio', 0):.1%}")
    print(f"\n📝 Summarized Content:")
    print(f"   {summary.get('content', '')}")


def test_reranking_iterations():
    """Test re-ranking with multiple iterations"""
    print("\n\n" + "=" * 70)
    print("TEST 3: Re-ranking with Threshold Iterations")
    print("=" * 70)
    
    contexts = [
        {"content": "Machine learning algorithms require large datasets for training.", "score": 0.4},
        {"content": "The user prefers Python for data science work.", "score": 0.8},
        {"content": "Deep learning uses neural networks with multiple layers.", "score": 0.5},
        {"content": "The user has experience with TensorFlow and PyTorch frameworks.", "score": 0.9},
        {"content": "Random content about weather.", "score": 0.2},
        {"content": "User attended a Python conference last month.", "score": 0.75},
    ]
    
    query = "What programming experience does the user have?"
    
    optimizer = ContextOptimizer(
        similarity_threshold=0.9,  # Less aggressive dedup
        entropy_threshold=0.3,
        rerank_threshold=0.65,     # Higher threshold to trigger iterations
        max_iterations=3
    )
    
    optimized, stats = optimizer.optimize(contexts, query)
    
    print(f"\n📊 Re-ranking Process:")
    print(f"   Initial contexts: {stats['original_count']}")
    print(f"   Iterations performed: {stats['iterations']}")
    print(f"   Final contexts: {stats['final_count']}")
    
    print(f"\n🎯 Final Ranked Results:")
    for i, ctx in enumerate(optimized, 1):
        relevance = ctx.get('relevance_score', 0)
        print(f"   {i}. [Relevance: {relevance:.3f}] {ctx['content'][:80]}...")


def test_token_limit_enforcement():
    """Test maximum token limit enforcement"""
    print("\n\n" + "=" * 70)
    print("TEST 4: Token Limit Enforcement")
    print("=" * 70)
    
    # Create long contexts
    contexts = []
    for i in range(20):
        content = f"Context {i}: " + " ".join([
            "This is a detailed piece of information about the user's preferences and activities."
            for _ in range(10)
        ])
        contexts.append({"content": content, "score": 0.8 - (i * 0.01)})
    
    query = "User preferences"
    
    # Set low token limit
    optimizer = ContextOptimizer(
        max_context_tokens=500,  # Very limited
        similarity_threshold=0.9,
        entropy_threshold=0.2
    )
    
    optimized, stats = optimizer.optimize(contexts, query)
    
    print(f"\n📊 Token Management:")
    print(f"   Original contexts: {stats['original_count']}")
    print(f"   Original tokens: ~{stats['original_tokens']}")
    print(f"   Token limit: 500")
    print(f"   Final contexts: {stats['final_count']}")
    print(f"   Final tokens: ~{stats['final_tokens']}")
    print(f"   Tokens saved: ~{stats['original_tokens'] - stats['final_tokens']}")
    
    truncated_count = sum(1 for ctx in optimized if ctx.get('truncated'))
    print(f"\n✂️  Truncated contexts: {truncated_count}")


def test_with_embeddings():
    """Test optimization with actual embeddings"""
    print("\n\n" + "=" * 70)
    print("TEST 5: Optimization with Vector Embeddings")
    print("=" * 70)
    
    contexts = [
        {"content": "The weather is sunny today.", "score": 0.8},
        {"content": "It's a bright and sunny day.", "score": 0.75},  # Very similar
        {"content": "The user likes pizza.", "score": 0.7},
        {"content": "Pizza is the user's favorite food.", "score": 0.72},  # Similar
        {"content": "Python is great for ML.", "score": 0.65},
    ]
    
    # Simulate embeddings (in reality, these would come from an embedding model)
    embeddings = [
        np.random.rand(100) for _ in contexts
    ]
    
    # Make similar contexts have similar embeddings
    embeddings[1] = embeddings[0] + np.random.rand(100) * 0.1  # Very similar
    embeddings[3] = embeddings[2] + np.random.rand(100) * 0.15  # Similar
    
    # Normalize
    for i in range(len(embeddings)):
        embeddings[i] = embeddings[i] / np.linalg.norm(embeddings[i])
    
    query = "What does the user like?"
    
    optimizer = ContextOptimizer(
        similarity_threshold=0.85,
        entropy_threshold=0.3
    )
    
    optimized, stats = optimizer.optimize(contexts, query, embeddings)
    
    print(f"\n📊 Embedding-based Deduplication:")
    print(f"   Original: {stats['original_count']} contexts")
    print(f"   Duplicates removed: {stats['duplicates_removed']}")
    print(f"   Final: {stats['final_count']} contexts")
    
    print(f"\n📝 Unique Contexts Retained:")
    for i, ctx in enumerate(optimized, 1):
        print(f"   {i}. {ctx['content']}")


def main():
    """Run all tests"""
    print("\n🚀 Context Optimization System Tests\n")
    
    test_basic_optimization()
    test_summarization()
    test_reranking_iterations()
    test_token_limit_enforcement()
    test_with_embeddings()
    
    print("\n\n✅ All tests completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
