# context_builder.py
from .hybrid_retriever import HybridRetriever
from .redis_stm import search_stm, store_stm
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.context_optimizer import ContextOptimizer, SummarizationOptimizer


# Initialize optimizers with configurable thresholds
context_optimizer = ContextOptimizer(
    similarity_threshold=0.85,      # Remove content >85% similar
    entropy_threshold=0.3,          # Filter low-information content
    min_info_content=10,            # Minimum meaningful content length
    max_context_tokens=4000,        # Maximum tokens in context
    rerank_threshold=0.6,           # Minimum relevance score
    max_iterations=2                # Re-ranking iterations
)

summarization_optimizer = SummarizationOptimizer(
    compression_ratio=0.3           # Compress to 30% of original
)


def build_context(user_id, user_input, deepdive_id=None, enable_optimization=True):
    """
    Build context with optional optimization for memory and token efficiency
    
    Args:
        user_id: User identifier
        user_input: User's query/input
        deepdive_id: Optional deepdive identifier
        enable_optimization: Enable context optimization (default: True)
    
    Returns:
        Optimized context with metadata about optimization performed
    """
    # 1️⃣ STM semantic cache
    stm = search_stm(user_id, user_input)
    if stm:
        return stm

    # 2️⃣ Episodic memory
    retriever = HybridRetriever()
    retriever.load(user_id, deepdive_id)
    results = retriever.search(user_input)

    context = []
    for r in results:
        ep = r["episode"]
        text = "\n".join(
            f"{m['role']}: {m['content']}"
            for m in ep["messages"]
        )
        context.append({
            "role": "system",
            "content": f"PAST EPISODE:\n{text}",
            "score": r.get("total_score", 0),
            "episode_id": ep.get("id")
        })

    # 3️⃣ Apply optimization if enabled and we have contexts
    if enable_optimization and context:
        optimized_context, stats = context_optimizer.optimize(
            contexts=context,
            query=user_input
        )
        
        # Log optimization stats
        print(f"🎯 Context Optimization Stats:")
        print(f"   Original: {stats['original_count']} items, ~{stats['original_tokens']} tokens")
        print(f"   Duplicates removed: {stats['duplicates_removed']}")
        print(f"   Low entropy removed: {stats['low_entropy_removed']}")
        print(f"   Compressed: {stats['compressed_count']}")
        print(f"   Re-ranking iterations: {stats['iterations']}")
        print(f"   Final: {stats['final_count']} items, ~{stats['final_tokens']} tokens")
        print(f"   Reduction: {stats['reduction_percentage']:.1f}%")
        
        context = optimized_context
    
    if context:
        store_stm(user_id, user_input, context)

    return context


def build_context_with_summarization(user_id, user_input, deepdive_id=None):
    """
    Build context with aggressive summarization for maximum compression
    Use when context window is very limited
    
    Returns:
        Heavily summarized context
    """
    # Get initial context
    retriever = HybridRetriever()
    retriever.load(user_id, deepdive_id)
    results = retriever.search(user_input, k=10)  # Get more results for summarization
    
    contexts = []
    for r in results:
        ep = r["episode"]
        text = "\n".join(
            f"{m['role']}: {m['content']}"
            for m in ep["messages"]
        )
        contexts.append({
            "content": f"PAST EPISODE:\n{text}",
            "score": r.get("total_score", 0)
        })
    
    if not contexts:
        return []
    
    # Apply optimization first
    optimized_contexts, stats = context_optimizer.optimize(
        contexts=contexts,
        query=user_input
    )
    
    # Then apply aggressive summarization
    summary_context = summarization_optimizer.summarize_contexts(
        optimized_contexts,
        user_input
    )
    
    print(f"📝 Summarization Stats:")
    print(f"   Original: {len(contexts)} episodes")
    print(f"   After optimization: {len(optimized_contexts)} episodes")
    print(f"   Final: 1 summarized context")
    print(f"   Compression: {summary_context.get('compression_ratio', 0):.1%}")
    
    return [{
        "role": "system",
        "content": summary_context.get('content', '')
    }]
