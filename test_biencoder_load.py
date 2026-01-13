#!/usr/bin/env python3
"""Test biencoder loading"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("=" * 70)
print("TESTING BI-ENCODER RE-RANKER LOADING")
print("=" * 70)

# Test import
try:
    from services.biencoder_reranker import BiEncoderReranker, get_recommended_config
    print("✓ Bi-Encoder Re-Ranking module loaded")
    BIENCODER_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    print(f"⚠️  Bi-Encoder Re-Ranking not available: {e}")
    BIENCODER_AVAILABLE = False

if BIENCODER_AVAILABLE:
    # Test configuration
    config = get_recommended_config("fast")
    print(f"\n🎯 Configuration:")
    print(f"   ├─ Model: {config['model_name']}")
    print(f"   ├─ Batch Size: {config['batch_size']}")
    print(f"   ├─ Score Threshold: {config['score_threshold']}")
    print(f"   └─ Description: {config['description']}")
    
    # Test initialization
    print(f"\n🤖 Initializing Bi-Encoder...")
    try:
        biencoder = BiEncoderReranker(
            model_name=config['model_name'],
            batch_size=config['batch_size']
        )
        print(f"✅ Bi-Encoder Re-Ranking: ENABLED")
        print(f"   └─ Model loaded successfully: {config['model_name']}")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
else:
    print(f"\n❌ Cannot test - module not available")

print("\n" + "=" * 70)
