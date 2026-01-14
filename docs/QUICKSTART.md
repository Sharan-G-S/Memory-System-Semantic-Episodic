# Context Optimization - Quick Reference

## 🚀 Quick Start

```bash
# Run with default balanced optimization
python3 interactive_memory_app.py

# Run with aggressive optimization (max token savings)
python3 interactive_memory_app.py --optimization aggressive

# Run with quality optimization (max quality)
python3 interactive_memory_app.py --optimization quality

# Disable optimization
python3 interactive_memory_app.py --no-optimization
```

## 📊 Optimization Pipeline

```
Input Contexts
    ↓
1. Deduplication (Remove exact + similar duplicates)
    ↓
2. Entropy Filtering (Remove low-information content)
    ↓
3. Compression (Consolidate redundant information)
    ↓
4. Re-ranking (Verify relevance, iterate if needed)
    ↓
5. Token Limit (Enforce maximum token count)
    ↓
Optimized Contexts (30-70% smaller)
```

## ⚙️ Profiles at a Glance

| Profile | Token Reduction | Quality | Use Case |
|---------|----------------|---------|----------|
| **Conservative** | 10-20% | Highest | Legal, Medical, Critical |
| **Balanced** | 30-50% | High | General purpose (default) |
| **Aggressive** | 50-70% | Good | Cost-sensitive, High-volume |
| **Quality** | 20-35% | Highest | Research, Complex reasoning |

## 🎯 Key Parameters

```python
# Critical parameters for tuning
SIMILARITY_THRESHOLD = 0.85   # 0.75=aggressive, 0.95=conservative
ENTROPY_THRESHOLD = 0.3       # Higher = stricter filtering
MAX_CONTEXT_TOKENS = 4000     # Adjust for your model
RERANK_THRESHOLD = 0.6        # Higher = better quality
MAX_ITERATIONS = 2            # More = better but slower
```

## 📈 What Gets Optimized

✅ **Removed:**
- Exact duplicate content
- Near-duplicate content (>85% similar)
- Low-entropy text (repetitive, uninformative)
- Irrelevant content (low query relevance)

✅ **Compressed:**
- Redundant phrases
- Excess whitespace
- Non-essential sentences

✅ **Preserved:**
- High-relevance content
- High-entropy information
- Unique insights
- User preferences

## 💡 Common Scenarios

### Scenario 1: Exceeding Token Limits
**Problem**: Context still too large for model
**Solution**:
```bash
python3 interactive_memory_app.py --optimization aggressive
```
Or edit `src/config/optimization_config.py`:
```python
MAX_CONTEXT_TOKENS = 2000  # Reduce limit
COMPRESSION_RATIO = 0.2     # More aggressive
```

### Scenario 2: Important Content Being Filtered
**Problem**: System removing needed information
**Solution**:
```bash
python3 interactive_memory_app.py --optimization conservative
```
Or adjust thresholds:
```python
SIMILARITY_THRESHOLD = 0.95  # Only exact duplicates
ENTROPY_THRESHOLD = 0.2      # Keep more content
```

### Scenario 3: Poor Response Quality
**Problem**: LLM responses lack context
**Solution**:
```bash
python3 interactive_memory_app.py --optimization quality
```
Or increase iterations:
```python
RERANK_THRESHOLD = 0.7
MAX_ITERATIONS = 3
```

## 📝 Programmatic Usage

```python
from src.services.context_optimizer import ContextOptimizer
from src.config.optimization_config import get_optimization_profile

# Use profile
config = get_optimization_profile("balanced")
optimizer = ContextOptimizer(**config)

# Or customize
optimizer = ContextOptimizer(
    similarity_threshold=0.85,
    entropy_threshold=0.3,
    max_context_tokens=4000,
    rerank_threshold=0.6,
    max_iterations=2
)

# Optimize
contexts = [{"content": "text", "score": 0.8}, ...]
optimized, stats = optimizer.optimize(contexts, query="user query")

print(f"Saved {stats['reduction_percentage']:.1f}% tokens")
```

## 🔍 Understanding Statistics

When you see optimization output:
```
🎯 Context Optimization Stats:
   Original: 15 items, ~3500 tokens
   Duplicates removed: 5
   Low entropy removed: 2
   Compressed: 4
   Re-ranking iterations: 2
   Final: 8 items, ~1750 tokens
   Reduction: 50.0%
```

**This means:**
- Started with 15 contexts (~3500 tokens)
- Removed 5 duplicates
- Filtered 2 low-quality items
- Compressed 4 items
- Ran 2 re-ranking iterations
- Ended with 8 contexts (~1750 tokens)
- **Saved 50% tokens** → Lower cost, faster response

## 💰 Cost Savings

### Example: GPT-4 ($0.03/1K input tokens)

| Scenario | Tokens | Cost/Request | Monthly Cost (10K requests) |
|----------|--------|--------------|----------------------------|
| No optimization | 5000 | $0.15 | $1,500 |
| Balanced (50%) | 2500 | $0.075 | $750 |
| Aggressive (70%) | 1500 | $0.045 | $450 |

**Savings with Balanced**: $750/month
**Savings with Aggressive**: $1,050/month

## 🧪 Testing

```bash
# Run all optimization tests
python3 test_context_optimization.py

# Test specific functionality
python3 -c "
from src.services.context_optimizer import ContextOptimizer
optimizer = ContextOptimizer()
contexts = [{'content': 'test', 'score': 0.8}]
result, stats = optimizer.optimize(contexts, 'query')
print(f'Works! Reduction: {stats[\"reduction_percentage\"]:.1f}%')
"
```

## 🐛 Troubleshooting

**Import Error**: 
```bash
# Make sure you're in the project directory
cd /Users/sharan/Documents/September-Test
python3 interactive_memory_app.py
```

**No Optimization Happening**:
- Check that `--no-optimization` is not set
- Verify profile is not "off"
- Ensure contexts have content

**Too Much Filtering**:
- Switch to conservative profile
- Lower entropy threshold
- Lower similarity threshold

## 📚 Full Documentation

See [CONTEXT_OPTIMIZATION_GUIDE.md](./CONTEXT_OPTIMIZATION_GUIDE.md) for:
- Detailed architecture
- Advanced features
- Performance benchmarks
- Integration examples
- Tuning guidelines

## 🎓 Best Practices

1. **Start with Balanced**: Good default for most cases
2. **Monitor Stats**: Watch token reduction percentage
3. **Adjust Gradually**: Make small threshold changes
4. **Test Quality**: Verify LLM response quality
5. **Measure Cost**: Track actual token usage
6. **Profile Specific**: Use different profiles per use case

## 🔗 Related Files

- `src/services/context_optimizer.py` - Core optimization logic
- `src/config/optimization_config.py` - Configuration & profiles
- `src/episodic/context_builder.py` - Integration point
- `test_context_optimization.py` - Test suite
- `docs/CONTEXT_OPTIMIZATION_GUIDE.md` - Full documentation
