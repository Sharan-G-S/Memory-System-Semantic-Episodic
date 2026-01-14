"""
Model Selection Service with RAG
Automatically selects the best LLM model based on task requirements
Uses RAG to learn from past performance and user context
Inspired by intelligent model routing for optimal performance and cost
"""
from typing import Dict, Tuple, Optional, List
import json
import hashlib
from datetime import datetime


class ModelSelector:
    """
    Intelligent model selection for different tasks with RAG support
    Routes to optimal models based on task characteristics and historical performance
    """
    
    def __init__(self, db_connection=None, redis_client=None):
        """
        Initialize model selector with optional database connection for RAG
        
        Args:
            db_connection: PostgreSQL connection for storing model performance
            redis_client: Redis client for caching model selection decisions
        """
        self.db_connection = db_connection
        self.redis_client = redis_client
        self._setup_performance_tracking()
    
    def _setup_performance_tracking(self):
        """Create table for tracking model performance if DB available"""
        if not self.db_connection:
            return
        
        try:
            cur = self.db_connection.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS model_performance_log (
                    id SERIAL PRIMARY KEY,
                    user_id VARCHAR(255),
                    task_type VARCHAR(100),
                    model_name VARCHAR(255),
                    query_context TEXT,
                    response_quality FLOAT,
                    latency_ms INTEGER,
                    token_count INTEGER,
                    success BOOLEAN,
                    feedback_score INTEGER,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Create indexes for fast retrieval
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_model_perf_task_model 
                ON model_performance_log(task_type, model_name)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_model_perf_user 
                ON model_performance_log(user_id, task_type)
            """)
            
            self.db_connection.commit()
            cur.close()
        except Exception as e:
            print(f"⚠️  Could not setup performance tracking: {e}")
    
    # Model registry with task-specific optimizations
    MODEL_REGISTRY = {
        "chat": {
            "model": "llama-3.3-70b-versatile",
            "reason": "Conversational AI - Best for natural dialogue and context understanding",
            "use_case": "General chat, Q&A, contextual conversations"
        },
        "optimization": {
            "model": "llama-3.1-8b-instant",
            "reason": "Fast inference - Optimal for real-time content optimization and filtering",
            "use_case": "Content deduplication, entropy filtering, compression"
        },
        "summarization": {
            "model": "llama-3.1-70b-versatile",
            "reason": "Compression & summarization - Excellent at condensing information accurately",
            "use_case": "Episode creation, context summarization, content consolidation"
        },
        "analysis": {
            "model": "mixtral-8x7b-32768",
            "reason": "Deep analysis - Superior reasoning for complex queries and data interpretation",
            "use_case": "Memory retrieval, context analysis, complex reasoning"
        },
        "code": {
            "model": "llama-3.1-70b-versatile",
            "reason": "Code generation - Specialized for technical tasks and structured output",
            "use_case": "Code generation, technical documentation, structured data"
        },
        "embedding": {
            "model": "llama-3.1-8b-instant",
            "reason": "Fast embedding generation - Efficient for vector operations",
            "use_case": "Semantic search, similarity calculations, clustering"
        },
        "classification": {
            "model": "llama-3.1-8b-instant",
            "reason": "Quick classification - Fast and accurate for categorization tasks",
            "use_case": "Content categorization, intent detection, routing"
        },
        "long_context": {
            "model": "mixtral-8x7b-32768",
            "reason": "Large context window (32K) - Handles extensive historical data",
            "use_case": "Long conversation history, large document processing"
        }
    }
    
    @classmethod
    def select_model(cls, task_type: str, verbose: bool = False) -> Tuple[str, str]:
        """
        Select the best model for a given task (class method for backward compatibility)
        
        Args:
            task_type: Type of task (chat, optimization, summarization, etc.)
            verbose: Whether to print selection details
            
        Returns:
            Tuple of (model_name, reason)
        """
        config = cls.MODEL_REGISTRY.get(task_type, cls.MODEL_REGISTRY["chat"])
        model_name = config["model"]
        reason = config["reason"]
        
        if verbose:
            print(f"\n🤖 Model Selection: {model_name}")
            print(f"   ├─ Task: {task_type}")
            print(f"   ├─ Reason: {reason}")
            print(f"   └─ Use Case: {config['use_case']}")
        
        return model_name, reason
    
    def select_model_with_rag(
        self, 
        task_type: str, 
        query_context: Optional[str] = None,
        user_id: Optional[str] = None,
        verbose: bool = False
    ) -> Tuple[str, str, Dict]:
        """
        Select the best model using RAG - retrieve past performance and adapt
        
        Args:
            task_type: Type of task
            query_context: Context of the current query for similarity matching
            user_id: User ID for personalized model selection
            verbose: Whether to print selection details
            
        Returns:
            Tuple of (model_name, reason, rag_insights)
        """
        # Get default model from registry
        default_config = self.MODEL_REGISTRY.get(task_type, self.MODEL_REGISTRY["chat"])
        
        # Check Redis cache for recent decision
        cache_key = None
        if self.redis_client and query_context:
            context_hash = hashlib.md5(f"{task_type}:{query_context[:100]}".encode()).hexdigest()
            cache_key = f"model_selection:{user_id or 'default'}:{context_hash}"
            
            try:
                cached = self.redis_client.get(cache_key)
                if cached:
                    cached_data = json.loads(cached)
                    if verbose:
                        print(f"\n🎯 Model Selection (Cached): {cached_data['model']}")
                        print(f"   └─ Cache hit - using previous decision")
                    return cached_data['model'], cached_data['reason'], cached_data.get('insights', {})
            except Exception:
                pass
        
        # Retrieve historical performance using RAG
        rag_insights = self._retrieve_performance_insights(task_type, query_context, user_id)
        
        # Make intelligent decision based on insights
        selected_model = default_config["model"]
        reason = default_config["reason"]
        
        if rag_insights and rag_insights.get('recommendations'):
            # Use RAG insights to potentially override default
            best_performer = rag_insights['recommendations'][0]
            
            # Override if alternative model has significantly better performance
            if best_performer['avg_success_rate'] > 0.85 and best_performer['avg_quality'] > 0.8:
                selected_model = best_performer['model_name']
                reason = f"RAG-optimized: {best_performer.get('use_case', 'task optimization')} - Best for proven performance (Success: {best_performer['avg_success_rate']:.1%}, Quality: {best_performer['avg_quality']:.1%})"
        
        if verbose:
            print(f"\n🤖 RAG-Enhanced Model Selection: {selected_model}")
            print(f"   ├─ Task: {task_type}")
            print(f"   ├─ Reason: {reason}")
            if rag_insights and rag_insights.get('total_history'):
                print(f"   ├─ Historical data: {rag_insights['total_history']} past uses")
                if rag_insights.get('user_preference'):
                    print(f"   ├─ User preference: {rag_insights['user_preference']}")
            print(f"   └─ Use Case: {default_config['use_case']}")
        
        # Cache the decision
        if self.redis_client and cache_key:
            try:
                cache_data = {
                    'model': selected_model,
                    'reason': reason,
                    'insights': rag_insights,
                    'timestamp': datetime.now().isoformat()
                }
                self.redis_client.setex(cache_key, 3600, json.dumps(cache_data))  # 1 hour TTL
            except Exception:
                pass
        
        return selected_model, reason, rag_insights
    
    def _retrieve_performance_insights(
        self, 
        task_type: str, 
        query_context: Optional[str], 
        user_id: Optional[str]
    ) -> Dict:
        """
        Retrieve historical performance insights using RAG
        
        Returns:
            Dictionary with performance metrics and recommendations
        """
        if not self.db_connection:
            return {}
        
        try:
            cur = self.db_connection.cursor()
            
            # Get overall performance by model for this task
            cur.execute("""
                SELECT 
                    model_name,
                    COUNT(*) as total_uses,
                    AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as avg_success_rate,
                    AVG(response_quality) as avg_quality,
                    AVG(latency_ms) as avg_latency,
                    AVG(feedback_score) as avg_feedback
                FROM model_performance_log
                WHERE task_type = %s
                GROUP BY model_name
                ORDER BY avg_success_rate DESC, avg_quality DESC
                LIMIT 5
            """, (task_type,))
            
            overall_performance = cur.fetchall()
            
            # Get user-specific preferences if user_id provided
            user_preference = None
            if user_id:
                cur.execute("""
                    SELECT 
                        model_name,
                        COUNT(*) as uses,
                        AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as success_rate
                    FROM model_performance_log
                    WHERE task_type = %s AND user_id = %s
                    GROUP BY model_name
                    ORDER BY uses DESC, success_rate DESC
                    LIMIT 1
                """, (task_type, user_id))
                
                user_pref = cur.fetchone()
                if user_pref:
                    user_preference = {
                        'model': user_pref['model_name'],
                        'uses': user_pref['uses'],
                        'success_rate': float(user_pref['success_rate'])
                    }
            
            # Get context-similar queries if context provided
            similar_contexts = []
            if query_context and len(query_context) > 20:
                cur.execute("""
                    SELECT 
                        model_name,
                        response_quality,
                        success,
                        query_context
                    FROM model_performance_log
                    WHERE task_type = %s
                      AND query_context ILIKE %s
                    ORDER BY created_at DESC
                    LIMIT 10
                """, (task_type, f"%{query_context[:50]}%"))
                
                similar_contexts = [dict(row) for row in cur.fetchall()]
            
            cur.close()
            
            # Build recommendations
            recommendations = []
            for perf in overall_performance:
                # Find the matching model config from registry
                model_config = None
                for task, config in self.MODEL_REGISTRY.items():
                    if config['model'] == perf['model_name']:
                        model_config = config
                        break
                
                # Fallback to task-based config if no exact match
                if not model_config:
                    model_config = self.MODEL_REGISTRY.get(task_type, self.MODEL_REGISTRY["chat"])
                
                recommendations.append({
                    'model_name': perf['model_name'],
                    'total_uses': perf['total_uses'],
                    'avg_success_rate': float(perf['avg_success_rate'] or 0),
                    'avg_quality': float(perf['avg_quality'] or 0),
                    'avg_latency_ms': int(perf['avg_latency'] or 0),
                    'avg_feedback': float(perf['avg_feedback'] or 0),
                    'reason': model_config['reason'],
                    'use_case': model_config['use_case']
                })
            
            return {
                'recommendations': recommendations,
                'user_preference': user_preference,
                'similar_contexts': similar_contexts,
                'total_history': sum(r['total_uses'] for r in recommendations)
            }
            
        except Exception as e:
            print(f"⚠️  RAG retrieval failed: {e}")
            return {}
    
    def log_performance(
        self,
        user_id: str,
        task_type: str,
        model_name: str,
        query_context: str,
        response_quality: float = 0.0,
        latency_ms: int = 0,
        token_count: int = 0,
        success: bool = True,
        feedback_score: Optional[int] = None
    ):
        """
        Log model performance for future RAG-based selection
        
        Args:
            user_id: User identifier
            task_type: Type of task performed
            model_name: Model that was used
            query_context: Context/query that was processed
            response_quality: Quality score (0-1)
            latency_ms: Response time in milliseconds
            token_count: Number of tokens used
            success: Whether the operation succeeded
            feedback_score: Optional user feedback (1-5)
        """
        if not self.db_connection:
            return
        
        try:
            cur = self.db_connection.cursor()
            cur.execute("""
                INSERT INTO model_performance_log 
                (user_id, task_type, model_name, query_context, response_quality, 
                 latency_ms, token_count, success, feedback_score)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                user_id, task_type, model_name, query_context[:500],  # Limit context length
                response_quality, latency_ms, token_count, success, feedback_score
            ))
            self.db_connection.commit()
            cur.close()
        except Exception as e:
            print(f"⚠️  Failed to log performance: {e}")
    
    @classmethod
    def get_all_models(cls) -> Dict:
        """Get all available models in registry"""
        return cls.MODEL_REGISTRY
    
    @classmethod
    def get_model_info(cls, task_type: str) -> Dict:
        """Get detailed information about model for a task"""
        return cls.MODEL_REGISTRY.get(task_type, cls.MODEL_REGISTRY["chat"])


def select_model_for_task(task_type: str, verbose: bool = False) -> Tuple[str, str]:
    """
    Convenience function for model selection
    
    Args:
        task_type: Type of task
        verbose: Whether to print details
        
    Returns:
        Tuple of (model_name, reason)
    """
    return ModelSelector.select_model(task_type, verbose)
