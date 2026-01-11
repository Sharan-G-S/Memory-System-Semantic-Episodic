# Context Optimization System

## Overview

The Context Optimization System is designed to optimize memory retrieval for:
- **Memory Optimization**: Reduce redundant and duplicate information
- **Context Window Management**: Fit more meaningful content within token limits
- **Token Consumption**: Reduce API costs and improve response speed

## Architecture

The optimization pipeline consists of 5 stages:

### 1. Deduplication
- **Exact Duplicate Removal**: Hash-based identification of identical content
- **Similarity-Based Removal**: Cosine similarity to detect near-duplicates
- **Configurable Threshold**: Adjust sensitivity (default: 0.85)

### 2. Entropy Filtering
- **Information Entropy Calculation**: Measures content information density
- **Low-Quality Filtering**: Removes repetitive or meaningless content
- **Minimum Content Length**: Filters very short, uninformative snippets

### 3. Compression & Consolidation
- **Redundant Phrase Removal**: Eliminates repeated information
- **Query-Focused Extraction**: Keeps only relevant sentences
- **Text Cleaning**: Removes excess whitespace and formatting

### 4. Re-ranking with Verification
- **Relevance Scoring**: Calculates query-context relevance
- **Threshold-Based Iteration**: Re-processes if quality drops below threshold
- **Iterative Refinement**: Up to N iterations for quality improvement

### 5. Token Limit Enforcement
- **Hard Limit**: Ensures context fits within maximum tokens
- **Smart Truncation**: Breaks at sentence boundaries when possible
- **Priority Preservation**: Keeps highest-scoring contexts

## Usage

### Basic Usage

```python
from src.services.context_optimizer import ContextOptimizer

optimizer = ContextOptimizer(
    similarity_threshold=0.85,
    entropy_threshold=0.3,
    max_context_tokens=4000
)

contexts = [
    {"content": "User likes Python programming", "score": 0.9},
    {"content": "User likes Python programming", "score": 0.85},  # Duplicate
    {"content": "User works on ML projects", "score": 0.8}
]

optimized, stats = optimizer.optimize(contexts, query="What does user like?")

print(f"Reduced from {stats['original_tokens']} to {stats['final_tokens']} tokens")
print(f"Token reduction: {stats['reduction_percentage']:.1f}%")
```

### With Optimization Profiles

```python
from src.config.optimization_config import get_optimization_profile

# Get pre-configured profile
config = get_optimization_profile("aggressive")

optimizer = ContextOptimizer(**config)
```

### In Interactive Memory App

Run with different optimization profiles:

```bash
# Balanced (default)
python interactive_memory_app.py

# Conservative - minimal optimization
python interactive_memory_app.py --optimization conservative

# Aggressive - maximum token reduction
python interactive_memory_app.py --optimization aggressive

# Quality - prioritize content quality
python interactive_memory_app.py --optimization quality

# Disabled - no optimization
python interactive_memory_app.py --no-optimization
```

## Optimization Profiles

### Conservative
- **Use Case**: When quality is paramount
- **Settings**:
  - Similarity threshold: 0.95 (only exact duplicates)
  - Entropy threshold: 0.2 (keep most content)
  - Max context tokens: 6000
  - Compression ratio: 0.5 (light compression)

### Balanced (Default)
- **Use Case**: General purpose, good quality/efficiency balance
- **Settings**:
  - Similarity threshold: 0.85
  - Entropy threshold: 0.3
  - Max context tokens: 4000
  - Compression ratio: 0.3

### Aggressive
- **Use Case**: Minimize token consumption, limited context windows
- **Settings**:
  - Similarity threshold: 0.75 (remove similar content)
  - Entropy threshold: 0.5 (strict filtering)
  - Max context tokens: 2000
  - Compression ratio: 0.2 (heavy compression)

### Quality
- **Use Case**: Prioritize information quality and relevance
- **Settings**:
  - Similarity threshold: 0.90
  - Entropy threshold: 0.25
  - Max context tokens: 5000
  - Re-ranking threshold: 0.7 (high quality bar)
  - Max iterations: 3

## Configuration

Edit `src/config/optimization_config.py` to customize:

```python
# Deduplication
SIMILARITY_THRESHOLD = 0.85

# Entropy filtering
ENTROPY_THRESHOLD = 0.3
MIN_INFO_CONTENT = 10

# Token limits
MAX_CONTEXT_TOKENS = 4000

# Re-ranking
RERANK_THRESHOLD = 0.6
MAX_ITERATIONS = 2

# Compression
COMPRESSION_RATIO = 0.3
```

## Advanced Features

### Summarization

For maximum compression when dealing with many redundant contexts:

```python
from src.services.context_optimizer import SummarizationOptimizer

summarizer = SummarizationOptimizer(compression_ratio=0.3)

summary = summarizer.summarize_contexts(contexts, query)
# Returns single consolidated summary context
```

### Model-Specific Configuration

Automatically adjust settings based on LLM:

```python
from src.config.optimization_config import get_config_for_model

config = get_config_for_model("gpt-3.5-turbo")
optimizer = ContextOptimizer(**config)
```

Supported models:
- `gpt-4`: 8000 token context, balanced optimization
- `gpt-3.5-turbo`: 4000 token context, aggressive optimization
- `claude-3`: 8000 token context, balanced optimization
- `llama-3-70b`: 3000 token context, aggressive optimization
- `groq`: 4000 token context, aggressive optimization

### Custom Embedding Integration

For better duplicate detection with actual embeddings:

```python
# Generate embeddings for contexts
embeddings = [embedding_model.encode(ctx['content']) for ctx in contexts]

# Pass to optimizer
optimized, stats = optimizer.optimize(contexts, query, embeddings)
```

## Performance Metrics

The optimizer tracks and reports:

- `original_count`: Number of input contexts
- `original_tokens`: Estimated input token count
- `duplicates_removed`: Exact and similar duplicates
- `low_entropy_removed`: Low-information content filtered
- `compressed_count`: Contexts compressed
- `iterations`: Re-ranking iterations performed
- `final_count`: Output contexts
- `final_tokens`: Estimated output token count
- `reduction_percentage`: Overall token reduction

## Testing

Run comprehensive tests:

```bash
python test_context_optimization.py
```

Tests cover:
1. Basic optimization pipeline
2. Aggressive summarization
3. Re-ranking with iterations
4. Token limit enforcement
5. Embedding-based deduplication

## Integration with Context Builder

The optimization is automatically integrated in `src/episodic/context_builder.py`:

```python
def build_context(user_id, user_input, deepdive_id=None, enable_optimization=True):
    # ... retrieve contexts ...
    
    if enable_optimization:
        optimized_context, stats = context_optimizer.optimize(
            contexts=context,
            query=user_input
        )
        
        print(f"🎯 Context Optimization Stats:")
        print(f"   Reduction: {stats['reduction_percentage']:.1f}%")
        
        context = optimized_context
    
    return context
```

## Best Practices

### When to Use Each Profile

1. **Conservative**:
   - Legal/medical applications requiring exact information
   - Initial testing phase
   - When accuracy is more important than cost

2. **Balanced**:
   - General chatbot applications
   - Customer support systems
   - Most production use cases

3. **Aggressive**:
   - High-volume applications with cost concerns
   - Models with small context windows
   - Real-time applications requiring fast responses

4. **Quality**:
   - Research applications
   - Complex reasoning tasks
   - When context quality directly impacts results

### Tuning Tips

1. **High Duplicate Rate**: Lower `similarity_threshold` (0.75-0.80)
2. **Too Much Filtering**: Lower `entropy_threshold` (0.2-0.25)
3. **Exceeding Token Limits**: Lower `max_context_tokens` or increase compression
4. **Poor Relevance**: Increase `rerank_threshold` or `max_iterations`
5. **Slow Performance**: Reduce `max_iterations` or disable summarization

## Performance Impact

### Token Reduction

Typical reduction rates by profile:
- Conservative: 10-20%
- Balanced: 30-50%
- Aggressive: 50-70%
- Quality: 20-35%

### Speed Impact

- Deduplication: ~5ms per 100 contexts
- Entropy filtering: ~2ms per 100 contexts
- Re-ranking: ~10ms per iteration
- Compression: ~8ms per 100 contexts

Total overhead: 20-50ms for typical workloads (negligible compared to LLM inference)

### Cost Savings

Example for GPT-4 ($0.03/1K input tokens):
- Original: 5000 tokens → $0.15 per request
- Optimized (50% reduction): 2500 tokens → $0.075 per request
- **Savings: 50% cost reduction**

For 10,000 requests/month: **$750/month saved**

## Troubleshooting

### Issue: Too much content removed

**Solution**: Use conservative profile or adjust thresholds
```python
optimizer = ContextOptimizer(
    similarity_threshold=0.95,  # Only remove near-exact duplicates
    entropy_threshold=0.2        # Keep more content
)
```

### Issue: Still exceeding token limits

**Solution**: Enable summarization or use aggressive compression
```python
from src.episodic.context_builder import build_context_with_summarization

context = build_context_with_summarization(user_id, query)
```

### Issue: Low relevance scores

**Solution**: Increase re-ranking iterations
```python
optimizer = ContextOptimizer(
    rerank_threshold=0.7,
    max_iterations=3
)
```

## Future Enhancements

Planned features:
- [ ] GPU-accelerated embedding similarity
- [ ] Semantic clustering for better grouping
- [ ] Adaptive threshold tuning based on feedback
- [ ] Multi-level caching for optimization results
- [ ] Integration with vector databases
- [ ] Real-time optimization metrics dashboard

## References

- Entropy calculation: Shannon entropy for information content
- Cosine similarity: Standard vector similarity metric
- RRF (Reciprocal Rank Fusion): For hybrid search scoring
- Extractive summarization: Sentence importance ranking
