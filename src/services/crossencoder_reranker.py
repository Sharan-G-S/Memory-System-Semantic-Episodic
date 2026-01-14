"""
Cross-Encoder Ranking Service
Stage 2 of Context Optimization Pipeline: Ranking

Uses cross-encoder models for accurate semantic ranking of retrieved documents.
Cross-encoders jointly encode query and document pairs for more accurate relevance scoring
compared to bi-encoders (which encode separately).

Purpose: Rank all retrieved candidates by semantic relevance to the query
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# Optional dependencies
try:
    from sentence_transformers import CrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    CrossEncoder = None


class CrossEncoderRanker:
    """
    Cross-Encoder Ranking for Context Optimization Pipeline - Stage 2
    
    Purpose: Rank retrieved documents by semantic relevance to query
    
    Features:
    - High-accuracy semantic ranking (examines query-document pairs jointly)
    - Direct relevance scoring for each candidate
    - Score-based filtering and threshold support
    - Batch processing for efficiency
    - Detailed ranking metrics and visualization
    
    Trade-off: More accurate than bi-encoders but computationally slower
    Best for: Final ranking stage where accuracy is critical
    """
    
    def __init__(
        self, 
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 32
    ):
        """
        Initialize cross-encoder ranking service
        
        Args:
            model_name: Cross-encoder model name (trained for passage ranking)
            batch_size: Batch size for scoring query-document pairs
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers not available. "
                "Install with: pip install sentence-transformers"
            )
        
        self.model_name = model_name
        self.batch_size = batch_size
        
        print(f"🤖 Loading cross-encoder ranking model: {model_name}")
        self.model = CrossEncoder(model_name)
        print(f"✅ Cross-encoder ranking model loaded successfully")
        
        self.documents: List[str] = []
    
    def build_index(self, documents: List[str]) -> None:
        """
        Store documents for ranking
        
        Note: Cross-encoders don't use pre-computed embeddings like bi-encoders.
        Each query-document pair is scored on-the-fly for maximum accuracy.
        
        Args:
            documents: List of candidate documents to rank
        """
        self.documents = documents
        print(f"📝 Prepared {len(documents)} documents for ranking")
    
    def rank(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: Optional[float] = None,
        show_progress: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Rank documents by semantic relevance to query using cross-encoder
        
        This is Stage 2 of the context optimization pipeline.
        Returns documents sorted by relevance score (highest first).
        
        Args:
            query: User query to rank documents against
            top_k: Number of top-ranked documents to return
            score_threshold: Minimum relevance score (optional, filters low-quality matches)
            show_progress: Show progress bar during scoring
            
        Returns:
            List of ranked documents with relevance scores and ranks
            Format: [{'index': int, 'document': str, 'score': float, 'rank': int}, ...]
        """
        if not self.documents:
            return []
        
        # Create query-document pairs for joint encoding
        pairs = [[query, doc] for doc in self.documents]
        
        # Score all pairs using cross-encoder
        print(f"🔍 Ranking {len(pairs)} documents using cross-encoder...")
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=show_progress
        )
        
        # Build ranked results with filtering
        results = []
        for idx, score in enumerate(scores):
            # Apply threshold if specified
            if score_threshold is None or score >= score_threshold:
                results.append({
                    'index': idx,
                    'document': self.documents[idx],
                    'score': float(score),
                    'rank': 0  # Assigned after sorting
                })
        
        # Sort by relevance score (descending: highest relevance first)
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Assign final ranks (1 = most relevant)
        for rank, result in enumerate(results[:top_k], 1):
            result['rank'] = rank
        
        return results[:top_k]
    
    def batch_rank(
        self,
        queries: List[str],
        top_k: int = 10,
        score_threshold: Optional[float] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        Rank documents for multiple queries (batch processing)
        
        Args:
            queries: List of queries to process
            top_k: Number of top results per query
            score_threshold: Minimum relevance score
            
        Returns:
            List of ranked results for each query
        """
        return [
            self.rank(query, top_k, score_threshold)
            for query in queries
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get ranking statistics and configuration"""
        return {
            'model': self.model_name,
            'num_documents': len(self.documents),
            'batch_size': self.batch_size,
            'method': 'cross-encoder',
            'stage': 'Stage 2: Ranking'
        }


# ===== Cross-Encoder Models for Ranking =====
# These models are specifically trained for passage/document ranking tasks
CROSSENCODER_RANKING_MODELS = {
    # Fast models - Recommended for real-time applications
    "fast": {
        "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "description": "Fast and accurate ranking model, trained on MS MARCO passage ranking",
        "speed": "fast",
        "accuracy": "high",
        "use_case": "Real-time ranking, low latency requirements"
    },
    "fast-multilingual": {
        "model": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        "description": "Multilingual ranking model, supports 100+ languages",
        "speed": "fast",
        "accuracy": "high",
        "use_case": "Multilingual applications, international content"
    },
    
    # Balanced models - Good trade-off between speed and accuracy
    "balanced": {
        "model": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "description": "Balanced speed/accuracy with 12 transformer layers",
        "speed": "medium",
        "accuracy": "very high",
        "use_case": "Production systems requiring high accuracy"
    },
    
    # Best accuracy models - For critical ranking tasks
    "best": {
        "model": "cross-encoder/ms-marco-electra-base",
        "description": "Highest accuracy using ELECTRA architecture",
        "speed": "slow",
        "accuracy": "excellent",
        "use_case": "Offline processing, quality-critical applications"
    }
}


def create_crossencoder_ranker(
    profile: str = "fast",
    custom_model: Optional[str] = None,
    batch_size: int = 32
) -> CrossEncoderRanker:
    """
    Factory function to create cross-encoder ranking service with preset profiles
    
    Args:
        profile: Preset profile - 'fast', 'balanced', 'best', or 'fast-multilingual'
        custom_model: Custom model name (overrides profile if provided)
        batch_size: Batch size for scoring pairs
        
    Returns:
        Configured CrossEncoderRanker instance ready for ranking
        
    Example:
        >>> ranker = create_crossencoder_ranker(profile="fast")
        >>> ranker.build_index(documents)
        >>> ranked = ranker.rank(query="what is machine learning?", top_k=10)
    """
    if custom_model:
        model_name = custom_model
    elif profile in CROSSENCODER_RANKING_MODELS:
        model_name = CROSSENCODER_RANKING_MODELS[profile]["model"]
    else:
        raise ValueError(
            f"Unknown profile: {profile}. "
            f"Choose from: {list(CROSSENCODER_RANKING_MODELS.keys())}"
        )
    
    return CrossEncoderRanker(
        model_name=model_name,
        batch_size=batch_size
    )

