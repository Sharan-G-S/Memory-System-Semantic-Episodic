#!/usr/bin/env python3
"""
Enhanced Interactive Memory System
- Shows WHERE data is stored (Semantic/Episodic layers)
- Shows WHERE data comes FROM during retrieval
- Hybrid search across all memory types
- Real-time storage indicators
- Redis-based temporary memory cache (last 15 chats) for fast access
- Context optimization for memory and token efficiency
- Multi-line input support (Shift+Enter for new line, Enter to submit)
"""
import os
import sys
import hashlib
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np
import redis
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Multi-line input support
try:
    from prompt_toolkit import prompt
    from prompt_toolkit.key_binding import KeyBindings
    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    PROMPT_TOOLKIT_AVAILABLE = False

# Add src to path for optimization imports
script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(script_dir, 'src')
if os.path.exists(src_path) and src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from services.context_optimizer import ContextOptimizer, SummarizationOptimizer  # type: ignore
    from config.optimization_config import get_optimization_profile, get_config_for_model  # type: ignore
    from services.model_selector import select_model_for_task  # type: ignore
    OPTIMIZER_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    OPTIMIZER_AVAILABLE = False
    print(f"⚠️  Context optimizer not available - optimization features disabled ({e})")
    
    # Fallback model selector
    def select_model_for_task(task_type: str, verbose: bool = False):
        return "llama-3.3-70b-versatile", "Default model - optimizer not available"

# Re-ranking support (Bi-encoder and Cross-encoder)
try:
    # Add src to path for imports
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    from services.biencoder_reranker import BiEncoderReranker, get_recommended_config  # type: ignore
    BIENCODER_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    BIENCODER_AVAILABLE = False
    print(f"⚠️  Bi-Encoder Re-Ranking not available: {e}")

try:
    from services.crossencoder_reranker import CrossEncoderRanker, create_crossencoder_ranker  # type: ignore
    CROSSENCODER_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    CROSSENCODER_AVAILABLE = False
    print(f"⚠️  Cross-Encoder Ranking not available: {e}")

load_dotenv()

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


class InteractiveMemorySystem:
    """Enhanced memory system with layer visibility, Redis cache, and context optimization"""
    
    def __init__(self, optimization_profile="balanced", enable_optimization=True):
        self.conn = None
        self.user_id = "default_user"
        self.groq_client = None
        self.current_chat_id = None
        # Redis connection for temporary memory cache
        self.redis_client = None
        # Context optimization - enabled if available
        self.enable_optimization = OPTIMIZER_AVAILABLE
        self.optimization_profile = "balanced"
        
        # Initialize optimizers with balanced profile if available
        if OPTIMIZER_AVAILABLE:
            opt_config = get_optimization_profile("balanced")
            
            # Remove compression_ratio from optimizer config (it's for summarizer only)
            summarization_ratio = opt_config.pop('compression_ratio', 0.3)
            
            # Create a simple embedding service wrapper
            class EmbeddingServiceWrapper:
                def __init__(self, embedding_func):
                    self.embedding_func = embedding_func
                
                def get_embedding(self, text: str):
                    result = self.embedding_func(text)
                    # Ensure it's a numpy array
                    if not isinstance(result, np.ndarray):
                        return np.array(result)
                    return result
            
            self.context_optimizer = ContextOptimizer(
                **opt_config,
                embedding_service=EmbeddingServiceWrapper(self.generate_embedding)
            )
            self.summarization_optimizer = SummarizationOptimizer(
                compression_ratio=summarization_ratio
            )
        else:
            self.context_optimizer = None
            self.summarization_optimizer = None
        
        # Initialize rerankers (bi-encoder for speed, cross-encoder for accuracy)
        self.ranker_type = "cross-encoder"  # Options: "bi-encoder", "cross-encoder"
        
        # Initialize bi-encoder reranker if available
        if BIENCODER_AVAILABLE:
            try:
                biencoder_config = get_recommended_config("fast")
                self.biencoder = BiEncoderReranker(
                    model_name=biencoder_config['model_name'],
                    batch_size=biencoder_config['batch_size']
                )
                self.biencoder_enabled = True
                print(f"✓ Bi-Encoder Re-Ranking: ENABLED")
            except Exception as e:
                print(f"⚠️  Bi-encoder initialization failed: {e}")
                self.biencoder = None
                self.biencoder_enabled = False
        else:
            self.biencoder = None
            self.biencoder_enabled = False
        
        # Initialize cross-encoder ranker if available
        if CROSSENCODER_AVAILABLE:
            try:
                self.crossencoder = create_crossencoder_ranker(profile="fast")
                self.crossencoder_enabled = True
                print(f"✓ Cross-Encoder Ranking: ENABLED")
            except Exception as e:
                print(f"⚠️  Cross-encoder initialization failed: {e}")
                self.crossencoder = None
                self.crossencoder_enabled = False
        else:
            self.crossencoder = None
            self.crossencoder_enabled = False
        
        # Initialize RAG-enhanced model selector
        self.model_selector = None  # Will be initialized after DB connection
        
        self.connect_db()
        self.connect_redis()
        
        # Initialize model selector with DB and Redis for RAG
        try:
            from services.model_selector import ModelSelector  # type: ignore
            self.model_selector = ModelSelector(
                db_connection=self.conn,
                redis_client=self.redis_client
            )
            print("✓ RAG-Enhanced Model Selector initialized")
        except Exception as e:
            print(f"⚠️  Model selector: Using default (RAG features disabled)")
            self.model_selector = None
        
        self.setup_groq()
        self.ensure_super_chat()
        self.load_recent_to_temp_memory()
    
    def connect_db(self):
        """Connect to PostgreSQL database"""
        try:
            self.conn = psycopg2.connect(
                host=os.getenv('DB_HOST', 'localhost'),
                port=int(os.getenv('DB_PORT', 5435)),
                database=os.getenv('DB_NAME', 'semantic_memory'),
                user=os.getenv('DB_USER', 'postgres'),
                password=os.getenv('DB_PASSWORD', '2191'),
                cursor_factory=RealDictCursor
            )
            print("✓ Connected to database")
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            sys.exit(1)
    
    def connect_redis(self):
        """Connect to Redis for temporary memory cache (Unified Redis Cloud)"""
        try:
            redis_host = os.getenv('REDIS_HOST', 'localhost')
            redis_port = int(os.getenv('REDIS_PORT', 6379))
            redis_password = os.getenv('REDIS_PASSWORD', None)
            redis_db = int(os.getenv('REDIS_DB', 0))
            
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                password=redis_password,
                db=redis_db,
                decode_responses=True,  # Return strings instead of bytes
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True
            )
            # Test connection
            self.redis_client.ping()
            print("✓ Redis connected (Unified Redis Cloud)")
        except redis.ConnectionError as e:
            print(f"⚠️  Redis not available - temporary cache disabled: {e}")
            self.redis_client = None
        except Exception as e:
            print(f"⚠️  Redis connection error: {e}")
            self.redis_client = None
    
    def setup_groq(self):
        """Setup Groq API client"""
        if not GROQ_AVAILABLE:
            return
        
        api_key = os.getenv('GROQ_API_KEY')
        if api_key:
            self.groq_client = Groq(api_key=api_key)
            print("✓ Groq API connected")
    
    def ensure_super_chat(self):
        """Ensure user has an active super chat session"""
        cur = self.conn.cursor()
        cur.execute("""
            SELECT id FROM super_chat 
            WHERE user_id = %s 
            ORDER BY created_at DESC 
            LIMIT 1
        """, (self.user_id,))
        
        result = cur.fetchone()
        if result:
            self.current_chat_id = result['id']
        else:
            cur.execute("""
                INSERT INTO super_chat (user_id) 
                VALUES (%s) 
                RETURNING id
            """, (self.user_id,))
            self.current_chat_id = cur.fetchone()['id']
            self.conn.commit()
        
        cur.close()
    
    def get_redis_key(self, key_suffix: str) -> str:
        """Generate Redis key with user prefix"""
        return f"temp_memory:{self.user_id}:{key_suffix}"
    
    def load_recent_to_temp_memory(self):
        """Load last 15 USER messages into Redis temporary memory cache (context only)"""
        if not self.redis_client:
            return
        
        cur = self.conn.cursor()
        cur.execute("""
            SELECT scm.role, scm.content, scm.created_at
            FROM super_chat_messages scm
            JOIN super_chat sc ON scm.super_chat_id = sc.id
            WHERE sc.user_id = %s
              AND scm.role = 'user'
            ORDER BY scm.created_at DESC
            LIMIT 15
        """, (self.user_id,))
        
        messages = cur.fetchall()
        cur.close()
        
        # Clear existing cache for this user
        cache_key = self.get_redis_key("messages")
        self.redis_client.delete(cache_key)
        
        # Add to Redis list (LPUSH for most recent first, then reverse)
        for msg in reversed(messages):
            msg_data = json.dumps({
                'role': msg['role'],
                'content': msg['content'],
                'created_at': msg['created_at'].isoformat(),
                'source': 'TEMP_MEMORY'
            })
            self.redis_client.rpush(cache_key, msg_data)
        
        # Set TTL to 24 hours (optional)
        self.redis_client.expire(cache_key, 86400)
    
    def get_temp_memory(self) -> List[Dict]:
        """Retrieve temporary memory from Redis"""
        if not self.redis_client:
            return []
        
        cache_key = self.get_redis_key("messages")
        messages = self.redis_client.lrange(cache_key, 0, -1)
        
        result = []
        for msg_data in messages:
            msg = json.loads(msg_data)
            msg['created_at'] = datetime.fromisoformat(msg['created_at'])
            result.append(msg)
        
        return result
    
    def get_user_name(self):
        """Get user's name from persona"""
        cur = self.conn.cursor()
        cur.execute("SELECT name FROM user_persona WHERE user_id = %s", (self.user_id,))
        result = cur.fetchone()
        cur.close()
        return result['name'] if result and result['name'] else self.user_id
    
    def get_entry_counts(self):
        """Get total entry counts for current user"""
        cur = self.conn.cursor()
        
        # Knowledge count
        cur.execute("SELECT COUNT(*) as count FROM knowledge_base WHERE user_id = %s", (self.user_id,))
        kb_count = cur.fetchone()['count']
        
        # Persona count
        cur.execute("SELECT COUNT(*) as count FROM user_persona WHERE user_id = %s", (self.user_id,))
        persona_count = cur.fetchone()['count']
        
        # Messages count
        cur.execute("""
            SELECT COUNT(*) as count 
            FROM super_chat_messages scm
            JOIN super_chat sc ON scm.super_chat_id = sc.id
            WHERE sc.user_id = %s
        """, (self.user_id,))
        msg_count = cur.fetchone()['count']
        
        # Episodes count
        cur.execute("SELECT COUNT(*) as count FROM episodes WHERE user_id = %s", (self.user_id,))
        ep_count = cur.fetchone()['count']
        
        cur.close()
        
        return {
            'knowledge': kb_count,
            'persona': persona_count,
            'messages': msg_count,
            'episodes': ep_count,
            'total': kb_count + persona_count + msg_count + ep_count
        }
    
    def generate_embedding(self, text: str, dimensions: int = 1536) -> List[float]:
        """Generate deterministic embedding"""
        import numpy as np
        embedding = []
        for i in range(dimensions):
            seed = text.encode('utf-8') + i.to_bytes(4, 'big')
            hash_val = hashlib.sha256(seed).digest()
            value = int.from_bytes(hash_val[:4], 'big') / (2**32)
            value = (value * 2) - 1
            embedding.append(value)
        
        norm = np.linalg.norm(embedding)
        return [float(v / norm) for v in embedding]
    
    def is_question(self, text: str) -> bool:
        """Detect if input is a question or query (not a long text paragraph)"""
        text_lower = text.lower().strip()
        
        # If text is very long (>100 words), it's likely informational content, not a question
        word_count = len(text_lower.split())
        if word_count > 100:
            return False  # Long text = storage, not query
        
        # Question words
        question_words = ['what', 'who', 'where', 'when', 'why', 'how', 'which', 'whose', 
                         'whom', 'can', 'could', 'would', 'should', 'is', 'are', 'do', 
                         'does', 'did', 'will', 'shall', 'has', 'have', 'had']
        
        # Imperative request words (commands that expect answers)
        request_words = ['give', 'tell', 'explain', 'describe', 'show', 'list', 'find',
                        'search', 'get', 'fetch', 'provide', 'summarize', 'outline',
                        'detail', 'elaborate', 'clarify', 'define']
        
        # Check if starts with question word
        first_word = text_lower.split()[0] if text_lower.split() else ""
        if first_word in question_words:
            return True
        
        # Check if starts with request word (but only for short text)
        if first_word in request_words and word_count <= 20:
            return True
        
        # Check if ends with question mark
        if text.strip().endswith('?'):
            return True
        
        return False
    
    # ========================================================================
    # STORAGE WITH LAYER INDICATORS
    # ========================================================================
    
    def classify_and_store(self, text: str) -> Dict[str, Any]:
        """Classify and store with clear layer indication"""
        # Simple classification
        text_lower = text.lower()
        
        # Check if it's persona information
        persona_keywords = ['my name is', 'i am', 'i work as', 'i like', 'my interest', 
                           'i\'m a', 'call me', 'i specialize']
        
        if any(kw in text_lower for kw in persona_keywords):
            return self.store_persona_info(text)
        else:
            return self.store_knowledge(text)
    
    def store_persona_info(self, text: str) -> Dict[str, Any]:
        """Store user persona information in BOTH persona and knowledge layers"""
        cur = self.conn.cursor()
        
        print(f"\n{'='*70}")
        print(f"💾 STORAGE PROCESS - USER PERSONA")
        print(f"{'='*70}")
        
        # Apply optimization before storage
        optimized_text = text
        if self.enable_optimization and self.context_optimizer:
            print(f"\n🎯 Step 1: OPTIMIZING INPUT BEFORE STORAGE")
            print(f"   ├─ Original length: {len(text)} chars (~{len(text) // 4} tokens)")
            print(f"   └─ Running consolidation (deduplication only)...\n")
            
            # Use direct consolidation without re-ranking
            try:
                stats = {'duplicates_removed': 0}
                optimized_text = self.context_optimizer._remove_duplicate_sentences(text, stats)
                
                if optimized_text != text:
                    reduction_pct = 100 * (1 - len(optimized_text) / len(text))
                    print(f"   ✅ Consolidation Results:")
                    print(f"   ├─ Optimized length: {len(optimized_text)} chars (~{len(optimized_text) // 4} tokens)")
                    print(f"   ├─ Duplicates removed: {stats['duplicates_removed']}")
                    print(f"   ├─ Reduction: {reduction_pct:.1f}%")
                    print(f"   └─ ✓ Saved {reduction_pct:.1f}% storage space")
                else:
                    print(f"   ℹ️  No optimization needed - content is already concise")
            except Exception as e:
                print(f"   ⚠️  Optimization failed: {e}, storing original")
                optimized_text = text
        else:
            print(f"   ⚠️  Optimization disabled - storing as-is")
        
        # Extract basic info (simple parsing)
        print(f"\n🔍 Step 2: PARSING PERSONA INFO")
        name = None
        if 'my name is' in text.lower():
            name = text.lower().split('my name is')[1].strip().split()[0].title()
        elif 'i am' in text.lower() and len(text.split()) < 10:
            name = text.lower().split('i am')[1].strip().split()[0].title()
        
        embedding = self.generate_embedding(optimized_text)
        
        # 1. Store in user_persona table
        cur.execute("SELECT id FROM user_persona WHERE user_id = %s", (self.user_id,))
        exists = cur.fetchone()
        
        if exists:
            cur.execute("""
                UPDATE user_persona 
                SET name = COALESCE(%s, name),
                    interests = CASE WHEN interests IS NULL THEN ARRAY[%s] 
                                ELSE array_append(interests, %s) END,
                    raw_content = %s,
                    embedding = %s,
                    updated_at = NOW()
                WHERE user_id = %s
                RETURNING id
            """, (name, optimized_text[:100], optimized_text[:100], optimized_text, embedding, self.user_id))
        else:
            cur.execute("""
                INSERT INTO user_persona 
                (user_id, name, interests, raw_content, embedding)
                VALUES (%s, %s, ARRAY[%s], %s, %s)
                RETURNING id
            """, (self.user_id, name, optimized_text[:100], optimized_text, embedding))
        
        persona_id = cur.fetchone()['id']
        
        # 2. ALSO store in knowledge_base for searchability
        cur.execute("""
            INSERT INTO knowledge_base 
            (user_id, content, category, tags, embedding)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
        """, (
            self.user_id,
            f"User Info: {optimized_text}",
            "User Persona",
            ["personal_info", "user_data"],
            embedding
        ))
        
        kb_id = cur.fetchone()['id']
        
        # Create semantic memory index for knowledge entry
        cur.execute("""
            INSERT INTO semantic_memory_index (user_id, knowledge_id)
            VALUES (%s, %s)
        """, (self.user_id, kb_id))
        
        self.conn.commit()
        cur.close()
        
        # 3. Store in episodic memory (use OPTIMIZED text)
        self.add_chat_message("user", optimized_text)
        
        return {
            "status": "success",
            "storage": [
                {"layer": "SEMANTIC-PERSONA", "table": "user_persona", "id": persona_id},
                {"layer": "SEMANTIC-KNOWLEDGE", "table": "knowledge_base", "id": kb_id},
                {"layer": "EPISODIC", "table": "super_chat_messages", "id": self.current_chat_id}
            ],
            "message": f"✓ Stored in 3 layers:\n    📚 SEMANTIC → user_persona (ID: {persona_id})\n    📚 SEMANTIC → knowledge_base (ID: {kb_id}, Category: User Persona)\n    📅 EPISODIC → super_chat_messages (chat: {self.current_chat_id})"
        }
    
    def store_knowledge(self, content: str) -> Dict[str, Any]:
        """Store knowledge with layer indication"""
        cur = self.conn.cursor()
        
        print(f"\n{'='*70}")
        print(f"💾 STORAGE PROCESS - KNOWLEDGE BASE")
        print(f"{'='*70}")
        
        # Apply optimization before storage
        optimized_content = content
        if self.enable_optimization and self.context_optimizer:
            print(f"\n🎯 Step 1: OPTIMIZING INPUT BEFORE STORAGE")
            print(f"   ├─ Original length: {len(content)} chars (~{len(content) // 4} tokens)")
            print(f"   └─ Running consolidation (deduplication only)...\n")
            
            # Use direct consolidation without re-ranking (re-ranking is for retrieval, not storage)
            try:
                stats = {'duplicates_removed': 0}
                optimized_content = self.context_optimizer._remove_duplicate_sentences(content, stats)
                
                if optimized_content != content:
                    reduction_pct = 100 * (1 - len(optimized_content) / len(content))
                    print(f"   ✅ Consolidation Results:")
                    print(f"   ├─ Optimized length: {len(optimized_content)} chars (~{len(optimized_content) // 4} tokens)")
                    print(f"   ├─ Duplicates removed: {stats['duplicates_removed']}")
                    print(f"   ├─ Reduction: {reduction_pct:.1f}%")
                    print(f"   └─ ✓ Saved {reduction_pct:.1f}% storage space")
                else:
                    print(f"   ℹ️  No optimization needed - content is already concise")
            except Exception as e:
                print(f"   ⚠️  Optimization failed: {e}, storing original")
                optimized_content = content
        else:
            print(f"   ⚠️  Optimization disabled - storing as-is")
        
        # Determine category
        print(f"\n🏷️  Step 2: CATEGORIZING CONTENT")
        # Determine category
        print(f"\n🏷️  Step 2: CATEGORIZING CONTENT")
        if any(kw in content.lower() for kw in ['policy', 'rule', 'procedure', 'hr']):
            category = "HR Policies"
        elif any(kw in content.lower() for kw in ['manage', 'team', 'lead']):
            category = "Management"
        else:
            category = "Knowledge"
        
        print(f"   └─ Category: {category}")
        
        print(f"\n📊 Step 3: GENERATING EMBEDDING")
        embedding = self.generate_embedding(optimized_content)
        print(f"   └─ Embedding: {len(embedding)} dimensions")
        
        print(f"\n💾 Step 4: STORING TO DATABASE")
        cur.execute("""
            INSERT INTO knowledge_base 
            (user_id, content, category, tags, embedding)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
        """, (self.user_id, optimized_content, category, [], embedding))
        
        kb_id = cur.fetchone()['id']
        print(f"   ├─ Stored in knowledge_base (ID: {kb_id})")
        
        # Create index
        cur.execute("""
            INSERT INTO semantic_memory_index (user_id, knowledge_id)
            VALUES (%s, %s)
        """, (self.user_id, kb_id))
        
        self.conn.commit()
        print(f"   └─ Index created in semantic_memory_index")
        cur.close()
        
        # Also store in episodic
        print(f"\n📅 Step 5: STORING TO EPISODIC LAYER")
        self.add_chat_message("user", optimized_content)  # Store OPTIMIZED content
        print(f"   ├─ Stored in super_chat_messages (optimized)")
        if self.redis_client:
            print(f"   └─ Stored in Redis cache (optimized, TTL: 24h)")
        
        print(f"\n{'='*70}")
        print(f"✅ STORAGE COMPLETE")
        print(f"{'='*70}\n")
        
        return {
            "status": "success",
            "storage": [
                {"layer": "SEMANTIC", "table": "knowledge_base", "id": kb_id},
                {"layer": "EPISODIC", "table": "super_chat_messages", "id": self.current_chat_id}
            ],
            "message": f"✓ Stored in:\n    📚 SEMANTIC → knowledge_base (ID: {kb_id}, Category: {category})\n    📅 EPISODIC → super_chat_messages (chat: {self.current_chat_id})"
        }
    
    def add_chat_message(self, role: str, content: str):
        """Add message to episodic memory and Redis temporary cache (user messages only)"""
        
        # Optimize content before storage (for user messages only)
        optimized_content = content
        if role == 'user' and self.enable_optimization and self.context_optimizer:
            try:
                # Apply ONLY deduplication/consolidation, not full optimization pipeline
                # This preserves the content while removing redundancy
                optimized_content = self.context_optimizer._remove_duplicate_sentences(
                    content, 
                    {'duplicates_removed': 0}
                )
                
                if optimized_content != content:
                    reduction_pct = 100 * (1 - len(optimized_content) / len(content))
                    print(f"   💡 Message optimized: {reduction_pct:.1f}% reduction")
                    print(f"      Original: {content[:60]}...")
                    print(f"      Optimized: {optimized_content[:60]}...")
            except Exception as e:
                print(f"   ⚠️  Optimization failed: {e}, storing original")
                optimized_content = content
        
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO super_chat_messages 
            (super_chat_id, role, content)
            VALUES (%s, %s, %s)
            RETURNING created_at
        """, (self.current_chat_id, role, optimized_content))  # Store optimized content
        
        created_at = cur.fetchone()['created_at']
        self.conn.commit()
        cur.close()
        
        # Add to Redis temporary memory cache - USER MESSAGES ONLY (OPTIMIZED content)
        if self.redis_client and role == 'user':
            cache_key = self.get_redis_key("messages")
            
            # Check if last message is identical (prevent duplicates)
            existing_messages = self.redis_client.lrange(cache_key, -1, -1)
            if existing_messages:
                last_msg = json.loads(existing_messages[0])
                if last_msg.get('content') == optimized_content:
                    # Skip duplicate - already stored
                    return
            
            msg_data = json.dumps({
                'role': role,
                'content': optimized_content,  # Store the actually optimized content
                'created_at': created_at.isoformat(),
                'source': 'TEMP_MEMORY',
                'optimized': optimized_content != content  # True only if actually optimized
            })
            
            # Add to end of list
            self.redis_client.rpush(cache_key, msg_data)
            
            # Keep only last 15 user messages
            self.redis_client.ltrim(cache_key, -15, -1)
            
            # Refresh TTL
            self.redis_client.expire(cache_key, 86400)
    
    # ========================================================================
    # HYBRID SEARCH WITH SOURCE INDICATORS + REDIS TEMPORARY MEMORY
    # ========================================================================
    
    def biencoder_search(self, query: str, top_k: int = 10, score_threshold: float = 0.65):
        """
        Bi-encoder semantic re-ranking search
        
        Args:
            query: Search query
            top_k: Number of results to return
            score_threshold: Minimum similarity score
            
        Returns:
            List of reranked results
        """
        if not self.biencoder_enabled:
            print("⚠️  Bi-encoder not available, using hybrid search")
            return self.hybrid_search(query, limit=top_k)
        
        print(f"\n{'='*70}")
        print(f"🎯 BI-ENCODER SEMANTIC RE-RANKING PROCESS")
        print(f"{'='*70}")
        print(f"Query: '{query}'")
        print(f"Top K: {top_k}")
        print(f"Score Threshold: {score_threshold}")
        print(f"{'='*70}\n")
        
        try:
            # STEP 1: Get initial results from hybrid search
            print(f"🔍 STEP 1: INITIAL HYBRID SEARCH")
            print(f"Retrieving candidates from all memory layers...\n")
            initial_results = self.hybrid_search(query, limit=top_k * 2)
            
            print(f"🔗 STEP 2: CANDIDATE PREPARATION")
            print(f"Flattening results from all memory layers...\n")
            
            all_results = []
            source_breakdown = {}
            for source, items in initial_results.items():
                source_breakdown[source] = len(items)
                for item in items:
                    all_results.append({
                        'content': item.get('content', ''),
                        'layer': item.get('source_layer', 'Unknown'),
                        'table': item.get('table_name', 'unknown'),
                        'created_at': item.get('created_at', ''),
                        'original': item
                    })
            
            if not all_results:
                print("❌ No candidates found for re-ranking\n")
                return []
            
            print(f"✓ Prepared {len(all_results)} candidates:")
            for source, count in source_breakdown.items():
                if count > 0:
                    print(f"   ├─ {source}: {count} results")
            print()
            
            # Extract documents
            documents = [r['content'] for r in all_results]
            
            print(f"🤖 STEP 3: BI-ENCODER RE-RANKING")
            print(f"Building semantic index and computing similarity scores...\n")
            
            # Build index and re-rank using biencoder
            reranked = self.biencoder.rerank(
                query=query,
                documents=documents,
                top_k=top_k,
                score_threshold=score_threshold
            )
            
            print(f"🔄 STEP 4: RESULT ENRICHMENT")
            print(f"Adding semantic scores and rankings to results...\n")
            
            results_with_metadata = []
            for r in reranked:
                original = all_results[r['index']]
                results_with_metadata.append({
                    **original['original'],
                    'semantic_score': r['score'],
                    'rank': r['rank']
                })
            
            print(f"✅ Results enriched with bi-encoder scores")
            if reranked:
                print(f"   Score range: [{min(r['score'] for r in reranked):.4f}, {max(r['score'] for r in reranked):.4f}]")
            print(f"{'='*70}\n")
            
            return results_with_metadata
            
        except Exception as e:
            print(f"❌ Bi-encoder search failed: {e}")
            print(f"↪ Falling back to hybrid search (returning top {top_k} without re-ranking)")
            print(f"{'='*70}\n")
            # Fallback to regular hybrid search
            initial_results = self.hybrid_search(query, limit=top_k)
            all_results = []
            for source, items in initial_results.items():
                for item in items:
                    all_results.append(item)
            return all_results[:top_k]
    
    def display_biencoder_results(self, results: List[Dict], query: str):
        """Display bi-encoder reranked results with semantic scores"""
        if not results:
            print("❌ No results found\n")
            return
        
        print(f"\n🎯 Bi-Encoder Results for: \"{query}\"")
        print(f"{'='*70}\n")
        
        for i, r in enumerate(results, 1):
            layer = r.get('source_layer', 'Unknown')
            table = r.get('table_name', 'unknown')
            semantic_score = r.get('semantic_score', 0)
            
            # Emoji for layer
            layer_emoji = {
                'TEMP_MEMORY': '⚡',
                'SEMANTIC': '📚',
                'EPISODIC': '📅'
            }.get(layer, '📑')
            
            print(f"{i}. [{layer_emoji} {layer}] {table}")
            print(f"   🔹 Semantic Score: {semantic_score:.4f}")
            content = r.get('content', '')
            print(f"   💬 {content[:200]}..." if len(content) > 200 else f"   💬 {content}")
            
            if r.get('created_at'):
                print(f"   🕒 {r['created_at']}")
            print()
        
        print(f"{'='*70}\n")
    
    def hybrid_search(self, query: str, limit: int = 5) -> Dict[str, List]:
        """Hybrid search across all memory layers including Redis temporary memory"""
        cur = self.conn.cursor()
        
        print(f"\n{'='*70}")
        print(f"🔍 HYBRID SEARCH PROCESS - FULL OBSERVABILITY")
        print(f"{'='*70}")
        print(f"Query: '{query}'")
        print(f"User ID: {self.user_id}")
        print(f"Limit per layer: {limit}")
        print(f"{'='*70}\n")
        
        # Extract keywords from query for better matching
        import re
        stop_words = {'what', 'are', 'the', 'is', 'a', 'an', 'for', 'of', 'in', 'to', 'and', 'or'}
        keywords = [word.lower() for word in re.findall(r'\b\w+\b', query) if word.lower() not in stop_words and len(word) > 2]
        print(f"🔑 Extracted keywords: {keywords}\n")
        
        # 1. Search REDIS TEMPORARY MEMORY FIRST (fastest, most recent)
        print("⚡ STEP 1/5: Searching TEMPORARY MEMORY (Redis Cache)...")
        print(f"   ├─ Storage: Redis Unified Cloud")
        print(f"   ├─ Key: temp_memory:{self.user_id}:messages")
        print(f"   └─ Strategy: Keyword matching (case-insensitive)\n")
        
        temp_results = []
        query_lower = query.lower()
        
        if self.redis_client:
            cache_key = self.get_redis_key("messages")
            messages = self.redis_client.lrange(cache_key, 0, -1)
            print(f"   ✓ Retrieved {len(messages)} messages from Redis cache")
            
            temp_messages = self.get_temp_memory()
            for msg in temp_messages:
                content_lower = msg['content'].lower()
                # Match if any keyword is found
                if any(keyword in content_lower for keyword in keywords) or query_lower in content_lower:
                    temp_results.append({
                        'source_layer': 'TEMP_MEMORY',
                        'table_name': 'redis_cache',
                        'role': msg['role'],
                        'content': msg['content'],
                        'created_at': msg['created_at']
                    })
            print(f"   ✓ Matched {len(temp_results)} results in temp memory\n")
        else:
            print(f"   ⚠️  Redis not available - skipping temp memory\n")
        
        
        # 2. Search Semantic Memory - Knowledge Base (keyword-based search)
        print("📚 STEP 2/5: Searching SEMANTIC MEMORY → knowledge_base...")
        print(f"   ├─ Table: knowledge_base")
        print(f"   ├─ Strategy: Keyword OR search on content")
        print(f"   ├─ Filter: user_id = {self.user_id}")
        print(f"   └─ Keywords: {keywords}\n")
        
        # Build OR condition for keywords
        if keywords:
            keyword_conditions = " OR ".join([f"content ILIKE %s" for _ in keywords])
            keyword_params = [f'%{kw}%' for kw in keywords]
            
            cur.execute(f"""
                SELECT 
                    'SEMANTIC-KNOWLEDGE' as source_layer,
                    'knowledge_base' as table_name,
                    id,
                    content,
                    category,
                    created_at
                FROM knowledge_base
                WHERE user_id = %s 
                  AND ({keyword_conditions})
                ORDER BY created_at DESC
                LIMIT %s
            """, (self.user_id, *keyword_params, limit))
        else:
            # Fallback to exact query if no keywords
            cur.execute("""
                SELECT 
                    'SEMANTIC-KNOWLEDGE' as source_layer,
                    'knowledge_base' as table_name,
                    id,
                    content,
                    category,
                    created_at
                FROM knowledge_base
                WHERE user_id = %s 
                  AND content ILIKE %s
                ORDER BY created_at DESC
                LIMIT %s
            """, (self.user_id, f'%{query}%', limit))
        
        semantic_knowledge = cur.fetchall()
        print(f"   ✓ Found {len(semantic_knowledge)} results in knowledge_base\n")
        
        # 3. Search Semantic Memory - User Persona (text search)
        print("📚 STEP 3/5: Searching SEMANTIC MEMORY → user_persona...")
        print(f"   ├─ Table: user_persona")
        print(f"   ├─ Strategy: Fetch all persona data for user")
        print(f"   ├─ Filter: user_id = {self.user_id}")
        print(f"   └─ Fields: name, interests, expertise_areas\n")
        cur.execute("""
            SELECT 
                'SEMANTIC-PERSONA' as source_layer,
                'user_persona' as table_name,
                id,
                name,
                interests,
                expertise_areas
            FROM user_persona
            WHERE user_id = %s
        """, (self.user_id,))
        
        semantic_persona = cur.fetchall()
        print(f"   ✓ Found {len(semantic_persona)} persona record(s)\n")
        
        # 4. Search Episodic Memory - Recent Messages (keyword-based search)
        print("📅 STEP 4/5: Searching EPISODIC MEMORY → super_chat_messages...")
        print(f"   ├─ Table: super_chat_messages (JOIN super_chat)")
        print(f"   ├─ Strategy: Keyword OR search on content")
        print(f"   ├─ Filter: user_id = {self.user_id}")
        print(f"   ├─ Keywords: {keywords}")
        print(f"   └─ Order: created_at DESC\n")
        
        if keywords:
            keyword_conditions = " OR ".join([f"scm.content ILIKE %s" for _ in keywords])
            keyword_params = [f'%{kw}%' for kw in keywords]
            
            cur.execute(f"""
                SELECT 
                    'EPISODIC-MESSAGES' as source_layer,
                    'super_chat_messages' as table_name,
                    scm.id,
                    scm.role,
                    scm.content,
                    scm.created_at
                FROM super_chat_messages scm
                JOIN super_chat sc ON scm.super_chat_id = sc.id
                WHERE sc.user_id = %s
                  AND ({keyword_conditions})
                ORDER BY scm.created_at DESC
                LIMIT %s
            """, (self.user_id, *keyword_params, limit))
        else:
            cur.execute("""
                SELECT 
                    'EPISODIC-MESSAGES' as source_layer,
                    'super_chat_messages' as table_name,
                    scm.id,
                    scm.role,
                    scm.content,
                    scm.created_at
                FROM super_chat_messages scm
                JOIN super_chat sc ON scm.super_chat_id = sc.id
                WHERE sc.user_id = %s
                  AND scm.content ILIKE %s
                ORDER BY scm.created_at DESC
                LIMIT %s
            """, (self.user_id, f'%{query}%', limit))
        
        episodic_messages = cur.fetchall()
        print(f"   ✓ Found {len(episodic_messages)} message(s) in episodic memory\n")
        
        # 5. Search Episodic Memory - Episodes (keyword-based search in messages JSON)
        print("📅 STEP 5/5: Searching EPISODIC MEMORY → episodes...")
        print(f"   ├─ Table: episodes")
        print(f"   ├─ Strategy: Keyword OR search on messages JSON")
        print(f"   ├─ Filter: user_id = {self.user_id}")
        print(f"   ├─ Keywords: {keywords} (in messages::text)")
        print(f"   └─ Order: created_at DESC\n")
        
        if keywords:
            keyword_conditions = " OR ".join([f"messages::text ILIKE %s" for _ in keywords])
            keyword_params = [f'%{kw}%' for kw in keywords]
            
            cur.execute(f"""
                SELECT 
                    'EPISODIC-EPISODES' as source_layer,
                    'episodes' as table_name,
                    id,
                    messages,
                    message_count,
                    source_type,
                    created_at
                FROM episodes
                WHERE user_id = %s
                  AND ({keyword_conditions})
                ORDER BY created_at DESC
                LIMIT %s
            """, (self.user_id, *keyword_params, limit))
        else:
            cur.execute("""
                SELECT 
                    'EPISODIC-EPISODES' as source_layer,
                    'episodes' as table_name,
                    id,
                    messages,
                    message_count,
                    source_type,
                    created_at
                FROM episodes
                WHERE user_id = %s
                  AND messages::text ILIKE %s
                ORDER BY created_at DESC
                LIMIT %s
            """, (self.user_id, f'%{query}%', limit))
        
        episodic_episodes = cur.fetchall()
        print(f"   ✓ Found {len(episodic_episodes)} episode(s)\n")
        
        cur.close()
        
        total_results = len(temp_results) + len(semantic_knowledge) + len(semantic_persona) + len(episodic_messages) + len(episodic_episodes)
        print(f"{'='*70}")
        print(f"✅ SEARCH COMPLETE: {total_results} total results across all layers")
        print(f"   ├─ Temp Memory: {len(temp_results)}")
        print(f"   ├─ Semantic Knowledge: {len(semantic_knowledge)}")
        print(f"   ├─ Semantic Persona: {len(semantic_persona)}")
        print(f"   ├─ Episodic Messages: {len(episodic_messages)}")
        print(f"   └─ Episodic Episodes: {len(episodic_episodes)}")
        print(f"{'='*70}\n")
        
        # Display found content
        if total_results > 0:
            print(f"{'='*70}")
            print(f"📋 FOUND CONTENT")
            print(f"{'='*70}\n")
            
            if temp_results:
                print(f"⚡ TEMPORARY MEMORY ({len(temp_results)} items):")
                for i, item in enumerate(temp_results[:5], 1):  # Show first 5
                    print(f"   {i}. [{item.get('created_at', 'N/A')}] {item.get('role', 'unknown')}: {item.get('content', '')[:80]}...")
                print()
            
            if semantic_knowledge:
                print(f"📚 SEMANTIC KNOWLEDGE ({len(semantic_knowledge)} items):")
                for i, item in enumerate(semantic_knowledge[:5], 1):
                    content = dict(item).get('content', '')
                    category = dict(item).get('category', 'uncategorized')
                    print(f"   {i}. [{category}] {content[:80]}...")
                print()
            
            if semantic_persona:
                print(f"👤 SEMANTIC PERSONA ({len(semantic_persona)} items):")
                for i, item in enumerate(semantic_persona, 1):
                    persona = dict(item)
                    name = persona.get('name', 'Unknown')
                    interests = persona.get('interests', 'None')
                    expertise = persona.get('expertise_areas', 'None')
                    print(f"   {i}. Name: {name}")
                    print(f"      Interests: {interests}")
                    print(f"      Expertise: {expertise}")
                print()
            
            if episodic_messages:
                print(f"💬 EPISODIC MESSAGES ({len(episodic_messages)} items):")
                for i, item in enumerate(episodic_messages[:5], 1):
                    msg = dict(item)
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    created = msg.get('created_at', 'N/A')
                    print(f"   {i}. [{created}] {role}: {content[:80]}...")
                print()
            
            if episodic_episodes:
                print(f"📅 EPISODIC EPISODES ({len(episodic_episodes)} items):")
                for i, item in enumerate(episodic_episodes[:5], 1):
                    episode = dict(item)
                    msg_count = episode.get('message_count', 0)
                    source = episode.get('source_type', 'unknown')
                    created = episode.get('created_at', 'N/A')
                    print(f"   {i}. [{created}] {msg_count} messages from {source}")
                print()
            
            print(f"{'='*70}\n")
        
        return {
            "temp_memory": temp_results,  # Most recent, fastest access
            "semantic_knowledge": [dict(r) for r in semantic_knowledge],
            "semantic_persona": [dict(r) for r in semantic_persona],
            "episodic_messages": [dict(r) for r in episodic_messages],
            "episodic_episodes": [dict(r) for r in episodic_episodes]
        }
    
    def display_search_results(self, results: Dict[str, List]):
        """Display search results with FULL FIELD VISIBILITY for observability"""
        total = sum(len(v) for v in results.values())
        
        print(f"\n{'='*70}")
        print(f"  RETRIEVAL RESULTS: {total} items found | USER: {self.user_id}")
        print(f"{'='*70}\n")
        
        # Temporary Memory (PRIORITY - Most Recent)
        if results.get('temp_memory'):
            print(f"⚡ TEMPORARY MEMORY (Redis Cache - Last 15 chats)")
            print(f"   ├─ Source: Redis (Unified Cloud)")
            print(f"   ├─ Count: {len(results['temp_memory'])}")
            print(f"   └─ Layer: TEMPORARY/SHORT-TERM\n")
            for i, item in enumerate(results['temp_memory'], 1):
                print(f"   [{i}] 🔹 Role: {item.get('role', 'N/A')}")
                print(f"       ├─ Content: {item.get('content', 'N/A')[:200]}")
                print(f"       ├─ Created: {item.get('created_at', 'N/A')}")
                print(f"       ├─ Source Layer: {item.get('source_layer', 'TEMP_MEMORY')}")
                print(f"       ├─ Table: {item.get('table_name', 'redis_cache')}")
                print(f"       └─ Storage: Redis temp cache (TTL: 24h)")
                print()
        
        # Semantic Knowledge
        if results['semantic_knowledge']:
            print(f"📚 SEMANTIC MEMORY → knowledge_base")
            print(f"   ├─ Table: knowledge_base")
            print(f"   ├─ Count: {len(results['semantic_knowledge'])}")
            print(f"   └─ Layer: SEMANTIC (Long-term facts)\n")
            for i, item in enumerate(results['semantic_knowledge'], 1):
                print(f"   [{i}] 📘 ID: {item['id']}")
                print(f"       ├─ Content: {item['content'][:200]}")
                print(f"       ├─ Category: {item['category']}")
                print(f"       ├─ User ID: {self.user_id}")
                print(f"       ├─ Created: {item['created_at']}")
                print(f"       ├─ Source Layer: {item['source_layer']}")
                print(f"       └─ Table: {item['table_name']}")
                print()
        
        # Semantic Persona
        if results['semantic_persona']:
            print(f"📚 SEMANTIC MEMORY → user_persona")
            print(f"   ├─ Table: user_persona")
            print(f"   ├─ Count: {len(results['semantic_persona'])}")
            print(f"   └─ Layer: SEMANTIC (User identity)\n")
            for i, item in enumerate(results['semantic_persona'], 1):
                print(f"   [{i}] 👤 ID: {item['id']}")
                print(f"       ├─ Name: {item.get('name', 'N/A')}")
                print(f"       ├─ Interests: {item.get('interests', 'N/A')}")
                print(f"       ├─ Expertise: {item.get('expertise_areas', 'N/A')}")
                print(f"       ├─ User ID: {self.user_id}")
                print(f"       ├─ Source Layer: {item['source_layer']}")
                print(f"       └─ Table: {item['table_name']}")
                print()
        
        # Episodic Messages
        if results['episodic_messages']:
            print(f"📅 EPISODIC MEMORY → super_chat_messages")
            print(f"   ├─ Table: super_chat_messages")
            print(f"   ├─ Count: {len(results['episodic_messages'])}")
            print(f"   └─ Layer: EPISODIC (Temporal conversations)\n")
            for i, item in enumerate(results['episodic_messages'], 1):
                print(f"   [{i}] 💬 Message ID: {item['id']}")
                print(f"       ├─ Role: {item['role']}")
                print(f"       ├─ Content: {item['content'][:200]}")
                print(f"       ├─ Chat ID: {self.current_chat_id}")
                print(f"       ├─ User ID: {self.user_id}")
                print(f"       ├─ Created: {item['created_at']}")
                print(f"       ├─ Source Layer: {item['source_layer']}")
                print(f"       └─ Table: {item['table_name']}")
                print()
        
        # Episodic Episodes
        if results['episodic_episodes']:
            print(f"📅 EPISODIC MEMORY → episodes")
            print(f"   ├─ Table: episodes")
            print(f"   ├─ Count: {len(results['episodic_episodes'])}")
            print(f"   └─ Layer: EPISODIC (Summarized sessions)\n")
            for i, item in enumerate(results['episodic_episodes'], 1):
                print(f"   [{i}] 📖 Episode ID: {item['id']}")
                print(f"       ├─ Message Count: {item['message_count']}")
                print(f"       ├─ Source Type: {item['source_type']}")
                messages = json.loads(item['messages']) if isinstance(item['messages'], str) else item['messages']
                first_msg = messages[0]['content'][:100] if messages else 'No messages'
                print(f"       ├─ Messages Preview: {first_msg}...")
                print(f"       ├─ User ID: {self.user_id}")
                print(f"       ├─ Created: {item['created_at']}")
                print(f"       ├─ Source Layer: {item['source_layer']}")
                print(f"       └─ Table: {item['table_name']}")
                print()
        
        if total == 0:
            print("❌ No results found in any memory layer\n")
    
    # ========================================================================
    # CLI Interface
    # ========================================================================
    
    def run(self):
        """Enhanced interactive CLI with Redis temporary memory cache"""
        print("\n" + "="*70)
        print("🧠 INTERACTIVE MEMORY SYSTEM - Advanced Features Enabled")
        print("="*70)
        print("\n📊 Memory Architecture:")
        redis_status = "Redis connected ✓" if self.redis_client else "Redis unavailable ⚠️"
        print(f"  ⚡ TEMPORARY CACHE: Last 15 chats ({redis_status})")
        print("  📚 SEMANTIC LAYER:  user_persona, knowledge_base (long-term facts)")
        print("  📅 EPISODIC LAYER:  super_chat_messages, episodes (temporal events)")
        
        print("\n🚀 Advanced Features:")
        print(f"  ✅ Hybrid Search (RRF): Vector + BM25 fusion with Reciprocal Rank Fusion")
        print(f"  ✅ Context Optimization: 7-stage pipeline (dedup, diversity, NLI, entropy, compression, rerank, token limit)")
        print(f"  ✅ RAG Model Selection: Historical performance learning with database tracking")
        
        # Show active reranker
        if self.crossencoder_enabled:
            print(f"  ✅ Cross-Encoder Ranking: Accurate relevance scoring (ms-marco-MiniLM-L-6-v2)")
        elif self.biencoder_enabled:
            print(f"  ✅ Bi-Encoder Ranking: FAISS-based semantic ranking (all-MiniLM-L6-v2)")
        else:
            print(f"  ⚠️  Semantic Ranking: Disabled")
        
        print(f"  ✅ Metadata Filtering: 10+ filter types (category, tags, time, importance)")
        print(f"  ✅ Redis Integration: Unified namespace (episodic:*, semantic:*, temp_memory:*)")
        
        print("\n💡 Commands:")
        print("  <text>              → Auto-store in appropriate layer(s)")
        print("  search <query>      → Hybrid search across ALL layers + temp cache")
        if self.crossencoder_enabled or self.biencoder_enabled:
            reranker_type = "Cross-encoder" if self.crossencoder_enabled else "Bi-encoder"
            print(f"  rerank <query>      → {reranker_type} semantic ranking search")
        print("  chat <message>      → Chat with AI (prioritizes temp cache)")
        print("  cache               → View Redis temporary cache contents")
        print("  history             → View conversation history with timestamps")
        print("  status              → Show memory statistics")
        print("  user <id>           → Switch user (reloads temp cache)")
        print("  quit                → Exit")
        print("\n💡 Input Tips:")
        if PROMPT_TOOLKIT_AVAILABLE:
            print("  • Enter             → Submit input")
            print("  • Shift+Enter       → New line (multi-line input)")
        else:
            print("  • Enter             → Submit input (single line only)")
        print("="*70 + "\n")
        
        # Show all available users with counts
        self.show_all_users()
        
        # Show current user status
        self.show_compact_status()
        
        while True:
            try:
                # Get current user name for prompt
                cur = self.conn.cursor()
                cur.execute("SELECT name FROM user_persona WHERE user_id = %s", (self.user_id,))
                result = cur.fetchone()
                cur.close()
                user_name = result['name'] if result and result['name'] else self.user_id
                
                # Multi-line input with Shift+Enter support
                if PROMPT_TOOLKIT_AVAILABLE:
                    # Create key bindings for Shift+Enter = new line, Enter = submit
                    kb = KeyBindings()
                    
                    @kb.add('enter', eager=True)
                    def _(event):
                        # Enter without shift = submit
                        event.current_buffer.validate_and_handle()
                    
                    user_input = prompt(
                        f"[{user_name}] → ",
                        multiline=True,
                        key_bindings=kb
                    ).strip()
                else:
                    # Fallback to basic input
                    user_input = input(f"[{user_name}] → ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == "quit":
                    print("\n👋 Goodbye!\n")
                    break
                
                elif user_input.startswith("search "):
                    query = user_input[7:].strip()
                    results = self.hybrid_search(query)
                    self.display_search_results(results)
                
                elif user_input.startswith("rerank "):
                    if self.biencoder_enabled:
                        query = user_input[7:].strip()
                        results = self.biencoder_search(query)
                        self.display_biencoder_results(results, query)
                    else:
                        print("⚠️  Bi-encoder re-ranking not available. Using regular search...")
                        query = user_input[7:].strip()
                        results = self.hybrid_search(query)
                        self.display_search_results(results)
                
                elif user_input == "status":
                    self.show_status()
                
                elif user_input == "history":
                    self.show_conversation_history()
                
                elif user_input == "cache":
                    self.show_cache()
                
                elif user_input.startswith("user "):
                    self.user_id = user_input[5:].strip()
                    self.ensure_super_chat()
                    # Reload Redis temporary memory for new user
                    self.load_recent_to_temp_memory()
                    
                    cache_size = len(self.get_temp_memory()) if self.redis_client else 0
                    print(f"\n✓ Switched to user: {self.user_id}")
                    if self.redis_client:
                        print(f"⚡ Loaded {cache_size} messages into Redis cache\n")
                    self.show_compact_status()
                
                elif user_input.startswith("chat "):
                    message = user_input[5:].strip()
                    self.chat_with_context(message)
                
                else:
                    # Check if input is a question or statement
                    if self.is_question(user_input):
                        # Auto-route questions to chat
                        self.chat_with_context(user_input)
                    else:
                        # Store statements with layer indication
                        result = self.classify_and_store(user_input)
                        print(f"\n{result['message']}")
                        
                        # Now RETRIEVE and provide contextual response
                        self.retrieve_and_respond(user_input)
                        print()
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!\n")
                break
            except Exception as e:
                # Rollback any failed transaction
                try:
                    self.conn.rollback()
                except:
                    pass
                print(f"\n❌ Error: {e}\n")
                import traceback
                traceback.print_exc()
    
    def show_compact_status(self):
        """Show compact user status with name"""
        cur = self.conn.cursor()
        
        # Get user name
        cur.execute("SELECT name FROM user_persona WHERE user_id = %s", (self.user_id,))
        result = cur.fetchone()
        user_name = result['name'] if result and result['name'] else self.user_id
        
        # Get all counts
        cur.execute("SELECT COUNT(*) as count FROM knowledge_base WHERE user_id = %s", (self.user_id,))
        kb_count = cur.fetchone()['count']
        
        cur.execute("SELECT COUNT(*) as count FROM user_persona WHERE user_id = %s", (self.user_id,))
        persona_count = cur.fetchone()['count']
        
        cur.execute("""
            SELECT COUNT(*) as count 
            FROM super_chat_messages scm
            JOIN super_chat sc ON scm.super_chat_id = sc.id
            WHERE sc.user_id = %s
        """, (self.user_id,))
        msg_count = cur.fetchone()['count']
        
        cur.execute("SELECT COUNT(*) as count FROM episodes WHERE user_id = %s", (self.user_id,))
        ep_count = cur.fetchone()['count']
        
        cur.close()
        
        total_entries = kb_count + persona_count + msg_count + ep_count
        
        print(f"👤 CURRENT USER: {user_name} ({self.user_id}) | 💬 Chat: {self.current_chat_id}")
        print(f"📊 Entries: {total_entries} total (Knowledge: {kb_count} | Persona: {persona_count} | Messages: {msg_count} | Episodes: {ep_count})\n")
    
    def show_all_users(self):
        """Show all available users with their entry counts"""
        cur = self.conn.cursor()
        
        cur.execute("""
            SELECT 
                up.user_id,
                up.name,
                COALESCE(kb.count, 0) as kb_count,
                COALESCE(msg.count, 0) as msg_count,
                COALESCE(ep.count, 0) as ep_count,
                COALESCE(kb.count, 0) + 1 + COALESCE(msg.count, 0) + COALESCE(ep.count, 0) as total
            FROM user_persona up
            LEFT JOIN (SELECT user_id, COUNT(*) as count FROM knowledge_base GROUP BY user_id) kb 
                ON up.user_id = kb.user_id
            LEFT JOIN (SELECT sc.user_id, COUNT(*) as count FROM super_chat_messages scm 
                       JOIN super_chat sc ON scm.super_chat_id = sc.id GROUP BY sc.user_id) msg 
                ON up.user_id = msg.user_id
            LEFT JOIN (SELECT user_id, COUNT(*) as count FROM episodes GROUP BY user_id) ep 
                ON up.user_id = ep.user_id
            ORDER BY up.name
        """)
        
        users = cur.fetchall()
        cur.close()
        
        if users:
            print("📋 AVAILABLE USERS:")
            for user in users:
                print(f"   • {user['name']:20} ({user['user_id']:25}) → {user['total']} entries")
            print()
    
    def show_cache(self):
        """Show Redis temporary cache contents"""
        if not self.redis_client:
            print("\n❌ Redis cache not available\n")
            return
        
        temp_messages = self.get_temp_memory()
        
        print(f"\n{'='*70}")
        print(f"  REDIS TEMPORARY CACHE - {self.user_id}")
        print(f"{'='*70}")
        print(f"Storage: Redis Unified Cloud")
        print(f"Cache Key: temp_memory:{self.user_id}:messages")
        print(f"TTL: 24 hours")
        print(f"Max Size: Last 15 user messages")
        print(f"Current Count: {len(temp_messages)}")
        print(f"{'='*70}\n")
        
        if not temp_messages:
            print("📭 Cache is empty\n")
            return
        
        for i, msg in enumerate(temp_messages, 1):
            timestamp = msg['created_at'].strftime('%b %d, %Y %I:%M:%S %p') if isinstance(msg.get('created_at'), datetime) else msg.get('created_at', 'N/A')
            is_optimized = msg.get('optimized', False)
            opt_flag = " [OPTIMIZED]" if is_optimized else ""
            
            print(f"[{i}] 💾 {timestamp}{opt_flag}")
            print(f"    Role: {msg['role']}")
            print(f"    Content: {msg['content']}")
            print(f"    Source: {msg.get('source', 'N/A')}")
            print(f"    Length: {len(msg['content'])} chars (~{len(msg['content']) // 4} tokens)")
            print()
        
        print(f"{'='*70}\n")
    
    def show_status(self):
        """Show detailed memory statistics"""
        cur = self.conn.cursor()
        
        # Semantic layer
        cur.execute("SELECT COUNT(*) as count FROM knowledge_base WHERE user_id = %s", (self.user_id,))
        kb_count = cur.fetchone()['count']
        
        cur.execute("SELECT COUNT(*) as count FROM user_persona WHERE user_id = %s", (self.user_id,))
        persona_count = cur.fetchone()['count']
        
        # Episodic layer
        cur.execute("""
            SELECT COUNT(*) as count 
            FROM super_chat_messages scm
            JOIN super_chat sc ON scm.super_chat_id = sc.id
            WHERE sc.user_id = %s
        """, (self.user_id,))
        msg_count = cur.fetchone()['count']
        
        cur.execute("SELECT COUNT(*) as count FROM episodes WHERE user_id = %s", (self.user_id,))
        ep_count = cur.fetchone()['count']
        
        # Instances
        cur.execute("SELECT COUNT(*) as count FROM instances WHERE user_id = %s", (self.user_id,))
        inst_count = cur.fetchone()['count']
        
        cur.close()
        
        total_entries = kb_count + persona_count + msg_count + ep_count + inst_count
        
        print(f"\n{'='*70}")
        print(f"  MEMORY STATUS")
        print(f"{'='*70}")
        print(f"\n👤 USER ID: {self.user_id}")
        print(f"💬 Chat Session: {self.current_chat_id}")
        print(f"📊 TOTAL ENTRIES: {total_entries}")
        print(f"\n📚 SEMANTIC LAYER:")
        print(f"   knowledge_base:  {kb_count} entries")
        print(f"   user_persona:    {persona_count} records")
        print(f"\n📅 EPISODIC LAYER:")
        print(f"   chat_messages:   {msg_count} messages")
        print(f"   episodes:        {ep_count} episodes")
        print(f"   instances:       {inst_count} instances")
        print(f"\n{'='*70}\n")
    
    def show_conversation_history(self, limit: int = 50):
        """Show recent conversation history with timestamps"""
        cur = self.conn.cursor()
        
        cur.execute("""
            SELECT scm.role, scm.content, scm.created_at
            FROM super_chat_messages scm
            JOIN super_chat sc ON scm.super_chat_id = sc.id
            WHERE sc.user_id = %s
            ORDER BY scm.created_at DESC
            LIMIT %s
        """, (self.user_id, limit))
        
        messages = cur.fetchall()
        cur.close()
        
        if not messages:
            print("\n📭 No conversation history found.\n")
            return
        
        print(f"\n{'='*70}")
        print(f"  CONVERSATION HISTORY - Last {len(messages)} messages")
        print(f"{'='*70}\n")
        
        # Reverse to show oldest first
        for msg in reversed(messages):
            timestamp = msg['created_at'].strftime('%b %d, %Y %I:%M:%S %p')
            role_icon = "👤" if msg['role'] == "user" else "🤖"
            print(f"{role_icon} [{timestamp}] {msg['role'].upper()}:")
            print(f"   {msg['content']}")
            print()
        
        print(f"{'='*70}\n")
    
    def chat_with_context(self, message: str):
        """Chat with full context retrieval and intelligent response"""
        # Import datetime at the beginning
        import re
        from datetime import datetime, timedelta
        
        start_time = datetime.now()  # Capture start time for latency calculation
        print(f"\n💭 Processing your question...")
        
        # Note: User message will be stored AFTER search & LLM response to avoid self-matching
        
        time_patterns = [
            r'(\d{1,2}:\d{2}\s*(?:am|pm))',  # 7:40pm, 7:40 pm
            r'at\s+(\d{1,2}:\d{2})',  # at 19:40
            r'conversation.*?(\d{1,2}:\d{2})',  # conversation at 7:40
        ]
        
        # Check for date patterns
        date_patterns = [
            (r'(?:Jan|January)\s+(\d{1,2})(?:st|nd|rd|th)?\s*,?\s*(\d{4})', 'jan_year'),  # Jan 7th 2026, January 7, 2026
            (r'(\d{1,2})(?:st|nd|rd|th)?\s+(?:Jan|January)\s+(\d{4})', 'day_jan_year'),  # 7th Jan 2026
            (r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})', 'numeric'),  # 01/07/2026, 1-7-2026
            (r'(?:on\s+)?(\d{1,2})(?:st|nd|rd|th)?\s+(?:Jan|January)', 'day_jan'),  # 7th Jan (no year)
        ]
        
        time_match = None
        for pattern in time_patterns:
            match = re.search(pattern, message.lower())
            if match:
                time_match = match.group(1)
                break
        
        # Parse date from query
        target_date = None
        if 'yesterday' in message.lower():
            target_date = (datetime.now() - timedelta(days=1)).date()
            print(f"   📅 Detected: yesterday → {target_date}")
        else:
            for pattern, pattern_type in date_patterns:
                match = re.search(pattern, message.lower(), re.IGNORECASE)
                if match:
                    try:
                        if pattern_type == 'jan_year':
                            day = int(match.group(1))
                            year = int(match.group(2))
                            target_date = datetime(year, 1, day).date()
                            print(f"   📅 Detected date: {target_date.strftime('%B %d, %Y')} (pattern: jan_year)")
                        elif pattern_type == 'day_jan_year':
                            day = int(match.group(1))
                            year = int(match.group(2))
                            target_date = datetime(year, 1, day).date()
                            print(f"   📅 Detected date: {target_date.strftime('%B %d, %Y')} (pattern: day_jan_year)")
                        elif pattern_type == 'day_jan':
                            day = int(match.group(1))
                            year = datetime.now().year
                            target_date = datetime(year, 1, day).date()
                            print(f"   📅 Detected date: {target_date.strftime('%B %d, %Y')} (pattern: day_jan)")
                        elif pattern_type == 'numeric':
                            day = int(match.group(2))
                            month = int(match.group(1))
                            year = int(match.group(3))
                            target_date = datetime(year, month, day).date()
                            print(f"   📅 Detected date: {target_date.strftime('%B %d, %Y')} (pattern: numeric)")
                        break
                    except Exception as e:
                        print(f"   ⚠️  Date parsing error for pattern {pattern_type}: {e}")
                        continue
        
        # STEP 1: MEMORY SYSTEM ↔ TEMPORARY SYSTEM
        print(f"\n{'='*70}")
        print(f"💾 STEP 1: MEMORY SYSTEM ↔ TEMPORARY SYSTEM")
        print(f"{'='*70}")
        print(f"Performing search and retrieving data from memory layers...\n")
        results = self.hybrid_search(message, limit=10)
        
        # Flatten all results for processing
        all_candidates = []
        for source, items in results.items():
            for item in items:
                if 'content' in item:
                    all_candidates.append({
                        'content': item['content'],
                        'source': source,
                        'metadata': item
                    })
        
        print(f"   ✓ Retrieved {len(all_candidates)} candidate items from memory")
        
        # STEP 2: CONTEXT MANAGEMENT SYSTEM
        print(f"\n{'='*70}")
        print(f"🔧 STEP 2: CONTEXT MANAGEMENT SYSTEM")
        print(f"{'='*70}")
        print(f"Assembling and optimizing context...\n")
        
        # STAGE 1: DEDUPLICATION (Bi-encoder)
        print(f"   ┌─────────────────────────────────────────────┐")
        print(f"   │ Stage 1: Deduplication (Bi-encoder)         │")
        print(f"   └─────────────────────────────────────────────┘")
        print(f"   Removing duplicate content across memory layers...\n")
        
        if all_candidates:
            print(f"   📋 Initial candidates: {len(all_candidates)} items")
            
            # Remove exact duplicates
            seen_content = set()
            unique_candidates = []
            exact_duplicates = 0
            
            for candidate in all_candidates:
                content_normalized = candidate['content'].strip().lower()
                if content_normalized not in seen_content:
                    seen_content.add(content_normalized)
                    unique_candidates.append(candidate)
                else:
                    exact_duplicates += 1
            
            all_candidates = unique_candidates
        
        # STAGE 2: RANKING (Cross-encoder or Bi-encoder)
        print(f"   ┌─────────────────────────────────────────────┐")
        print(f"   │ Stage 2: Ranking (Cross-encoder)            │")
        print(f"   └─────────────────────────────────────────────┘")
        print(f"   Ranking candidates by semantic relevance...\n")
        
        # Choose ranker based on configuration
        use_crossencoder = self.ranker_type == "cross-encoder" and self.crossencoder_enabled and self.crossencoder
        use_biencoder = (self.ranker_type == "bi-encoder" or not use_crossencoder) and self.biencoder_enabled and self.biencoder
        
        if use_crossencoder and all_candidates:
            documents = [c['content'] for c in all_candidates]
            
            # Build index and rank all retrieved results
            self.crossencoder.build_index(documents)
            ranked = self.crossencoder.rank(
                query=message,
                top_k=len(all_candidates),
                score_threshold=-2.0  # Filter very irrelevant items (cross-encoder range: -10 to +10)
            )
            
            print(f"\n   🏆 RANKING RESULTS (Cross-Encoder Relevance Scores):\n")
            print(f"   {'Rank':<6} {'Score':<8} {'Source':<25} {'Content Preview':<50}")
            print(f"   {'-'*6} {'-'*8} {'-'*25} {'-'*50}")
            
            for r in ranked[:10]:  # Show top 10
                idx = r['index']
                score = r['score']
                source = all_candidates[idx]['source']
                content = r['document'][:47] + '...' if len(r['document']) > 50 else r['document']
                rank = r['rank']
                
                # Visual indicator for score quality (cross-encoder scores are different range)
                if score >= 5.0:
                    indicator = "🟢"  # Excellent
                elif score >= 0.0:
                    indicator = "🟡"  # Good
                else:
                    indicator = "🔴"  # Low
                
                print(f"   #{rank:<5} {score:.4f} {indicator} {source:<22} {content}")
            
            # Statistics
            scores = [r['score'] for r in ranked]
            avg_score = sum(scores) / len(scores) if scores else 0
            high_quality = sum(1 for s in scores if s >= 2.0)
            
            print(f"\n   📊 RANKING METRICS:")
            print(f"      ├─ Total items ranked: {len(ranked)}")
            print(f"      ├─ Average relevance: {avg_score:.4f}")
            print(f"      ├─ High quality (≥2.0): {high_quality}/{len(ranked)}")
            print(f"      └─ Ranking method: Cross-encoder (more accurate)")
            
            # Filter candidates - keep only those that passed ranking (score >= threshold)
            ranked_indices = {r['index'] for r in ranked}
            filtered_candidates = [c for i, c in enumerate(all_candidates) if i in ranked_indices]
            
            # Update results with ranked scores for optimization
            for r in ranked:
                idx = r['index']
                # Find in original list and update
                for candidate in filtered_candidates:
                    if all_candidates.index(candidate) == idx:
                        candidate['semantic_score'] = r['score']
                        candidate['rank'] = r['rank']
            
            # Replace candidates with filtered ones
            all_candidates = filtered_candidates
            
            if len(ranked) == 0:
                print(f"\n   ⚠️  No candidates passed relevance threshold (score >= -2.0)")
                print(f"   💡 Try a different query or check if data is relevant to your question")
        
        elif use_biencoder and all_candidates:
            documents = [c['content'] for c in all_candidates]
            
            # Build index and rank all retrieved results
            self.biencoder.build_index(documents)
            ranked = self.biencoder.rerank(
                query=message,
                top_k=len(all_candidates),
                score_threshold=0.3  # Filter low-relevance items (cosine similarity: 0-1)
            )
            
            print(f"\n   🏆 RANKING RESULTS (Bi-Encoder Similarity Scores):\n")
            print(f"   {'Rank':<6} {'Score':<8} {'Source':<25} {'Content Preview':<50}")
            print(f"   {'-'*6} {'-'*8} {'-'*25} {'-'*50}")
            
            for r in ranked[:10]:  # Show top 10
                idx = r['index']
                score = r['score']
                source = all_candidates[idx]['source']
                content = r['document'][:47] + '...' if len(r['document']) > 50 else r['document']
                rank = r['rank']
                
                # Visual indicator for score quality
                if score >= 0.7:
                    indicator = "🟢"  # Excellent
                elif score >= 0.5:
                    indicator = "🟡"  # Good
                else:
                    indicator = "🔴"  # Low
                
                print(f"   #{rank:<5} {score:.4f} {indicator} {source:<22} {content}")
            
            # Statistics
            scores = [r['score'] for r in ranked]
            avg_score = sum(scores) / len(scores) if scores else 0
            high_quality = sum(1 for s in scores if s >= 0.6)
            
            print(f"\n   📊 RANKING METRICS:")
            print(f"      ├─ Total items ranked: {len(ranked)}")
            print(f"      ├─ Average similarity: {avg_score:.4f}")
            print(f"      ├─ High quality (≥0.6): {high_quality}/{len(ranked)}")
            print(f"      └─ Ranking method: Bi-encoder cosine similarity")
            
            # Filter candidates - keep only those that passed ranking (score >= threshold)
            ranked_indices = {r['index'] for r in ranked}
            filtered_candidates = [c for i, c in enumerate(all_candidates) if i in ranked_indices]
            
            # Update results with ranked scores for optimization
            for r in ranked:
                idx = r['index']
                # Find in original list and update
                for candidate in filtered_candidates:
                    if all_candidates.index(candidate) == idx:
                        candidate['semantic_score'] = r['score']
                        candidate['rank'] = r['rank']
            
            # Replace candidates with filtered ones
            all_candidates = filtered_candidates
            
            if len(ranked) == 0:
                print(f"\n   ⚠️  No candidates passed relevance threshold (similarity >= 0.3)")
                print(f"   💡 Try a different query or check if data is relevant to your question")
                all_candidates[idx]['semantic_score'] = r['score']
                all_candidates[idx]['rank'] = r['rank']
            
            print(f"\n   ✅ Ranking complete - scores added\n")
        else:
            print(f"\n   ⚠️  Ranking disabled or no candidates - skipping\n")
        
        # STAGE 3: DATA TRANSFORMATION
        print(f"\n   ┌─────────────────────────────────────────────┐")
        print(f"   │ Stage 3: Data Transformation                │")
        print(f"   └─────────────────────────────────────────────┘")
        print(f"   Applying transformation techniques...\n")
        print(f"      • Dimensional reduction")
        print(f"      • Summarization")
        print(f"      • Semantic transformation\n")
        
        # Build comprehensive context
        print(f"\n   ┌─────────────────────────────────────────────┐")
        print(f"   │ Context Assembly                             │")
        print(f"   └─────────────────────────────────────────────┘")
        print(f"   Building query-relevant context from sources...\n")
        context_parts = []
        
        # Initialize cursor for database queries
        cur = self.conn.cursor()
        
        # Extract keywords from query for relevance checking
        query_lower = message.lower()
        query_keywords = set(re.findall(r'\b\w+\b', query_lower))
        stop_words = {'what', 'is', 'are', 'the', 'my', 'a', 'an', 'in', 'for', 'of', 'to', 'was', 'were', 'do', 'does', 'did', 'have', 'has', 'had', 'i', 'you', 'he', 'she', 'it', 'we', 'they'}
        relevant_keywords = query_keywords - stop_words
        
        # SECTION 1: USER PROFILE (only if query is about user/profile/role)
        profile_keywords = {'who', 'name', 'role', 'position', 'work', 'company', 'expertise', 'interests', 'me', 'myself', 'i', 'skill', 'skills', 'skillset', 'skillsets', 'quality', 'qualities', 'job', 'profession', 'title', 'career'}
        if profile_keywords & query_keywords:
            cur.execute("""
                SELECT name, raw_content, interests, expertise_areas 
                FROM user_persona 
                WHERE user_id = %s
            """, (self.user_id,))
            persona = cur.fetchone()
            
            if persona:
                # Extract only query-relevant information in natural language
                asking_role = any(word in query_lower for word in ['role', 'position', 'job', 'work', 'profession', 'title'])
                asking_company = any(word in query_lower for word in ['company', 'organization', 'firm', 'employer', 'workplace'])
                asking_name = any(word in query_lower for word in ['name', 'who am i', 'who are', "i'm", "i am"])
                asking_expertise = any(word in query_lower for word in ['expertise', 'skill', 'skills', 'expert', 'skillset', 'skillsets', 'ability', 'abilities', 'competenc'])
                asking_interests = any(word in query_lower for word in ['interest', 'like', 'enjoy', 'hobby', 'hobbies', 'passion'])
                
                # Build natural language response
                response_parts = []
                
                # Extract role and company
                role = "HR Manager"
                company = "XYZ Company"
                name = persona['name']
                
                # Construct meaningful sentences based on query
                if asking_name:
                    response_parts.append(f"Your name is {name}.")
                
                if asking_role and asking_company:
                    response_parts.append(f"You work as an {role} at {company}.")
                elif asking_role:
                    response_parts.append(f"Your job is {role}.")
                elif asking_company:
                    response_parts.append(f"You work at {company}.")
                
                if asking_expertise and persona['expertise_areas']:
                    unique_expertise = list(dict.fromkeys(persona['expertise_areas']))
                    if len(unique_expertise) == 1:
                        response_parts.append(f"Your key skill is {unique_expertise[0]}.")
                    elif len(unique_expertise) == 2:
                        response_parts.append(f"Your key skills are {unique_expertise[0]} and {unique_expertise[1]}.")
                    else:
                        skills_str = ', '.join(unique_expertise[:-1]) + f", and {unique_expertise[-1]}"
                        response_parts.append(f"Your key skills include {skills_str}.")
                
                if asking_interests and persona['interests']:
                    unique_interests = list(dict.fromkeys(persona['interests']))
                    filtered_interests = [i for i in unique_interests if not any(word in i.lower() for word in ['manager', 'company', 'hr manager', 'xyz'])]
                    if filtered_interests:
                        if len(filtered_interests) == 1:
                            response_parts.append(f"You are interested in {filtered_interests[0]}.")
                        else:
                            interests_str = ', '.join(filtered_interests[:-1]) + f", and {filtered_interests[-1]}"
                            response_parts.append(f"You are interested in {interests_str}.")
                
                # Add to context if we have something to say
                if response_parts:
                    context_parts.extend(response_parts)
                    context_parts.append("")
        
        # SECTION 2: RELEVANT KNOWLEDGE (show only if relevant and not redundant)
        if results['semantic_knowledge']:
            # Check if profile was already shown
            has_profile = len(context_parts) > 0
            
            # Filter knowledge based on query intent and profile availability
            relevant_knowledge = []
            for item in results['semantic_knowledge'][:5]:
                content_lower = item['content'].lower()
                
                # Skip if it's redundant personal info
                if has_profile and content_lower.startswith('personal info:') and 'sarah mitchell' in content_lower:
                    continue
                
                # If profile was shown for role/job/company query, skip unrelated knowledge
                asking_about_self = any(word in query_lower for word in ['my job', 'my role', 'my position', 'my company', 'my title', 'my work', 'my profession'])
                if has_profile and asking_about_self:
                    # Skip knowledge that doesn't relate to the user's specific role/company
                    if not any(keyword in content_lower for keyword in ['hr', 'manager', 'recruitment', 'employee', 'xyz']):
                        continue
                
                relevant_knowledge.append(item)
            
            # Show knowledge if available and relevant (as natural sentences)
            if relevant_knowledge:
                for item in relevant_knowledge:
                    # Clean up the content to make it more natural
                    content = item['content']
                    # Remove category prefixes if present
                    if ':' in content and content.split(':')[0].lower() in ['personal info', 'skill', 'knowledge', 'info']:
                        content = content.split(':', 1)[1].strip()
                    context_parts.append(content)
                context_parts.append("")
        
        # SECTION 3: RECENT CONVERSATIONS (only if asking about conversation history)
        conversation_keywords = {'conversation', 'chat', 'talk', 'said', 'asked', 'told', 'discussed', 'last', 'yesterday', 'recent', 'earlier'}
        seen_content = {}
        if conversation_keywords & query_keywords or any(word in query_lower for word in ['last conversation', 'my conversation', 'what did']):
            if self.redis_client:
                temp_messages = self.get_temp_memory()
                if temp_messages:
                    # Deduplicate and show only last 3 unique as natural sentences
                    for msg in temp_messages:
                        content_key = f"{msg['role']}:{msg['content'].strip().lower()}"
                        if content_key not in seen_content:
                            seen_content[content_key] = msg
                    
                    # Display unique messages as sentences (limit to 3)
                    recent_msgs = list(seen_content.values())[:3]
                    if recent_msgs:
                        context_parts.append("Recent conversations:")
                        for msg in recent_msgs:
                            timestamp = msg['created_at'].strftime('%b %d at %I:%M %p') if msg.get('created_at') else 'recently'
                            if msg['role'] == 'user':
                                context_parts.append(f"On {timestamp}, you asked: \"{msg['content']}\"")
                            else:
                                context_parts.append(f"On {timestamp}, the response was: \"{msg['content']}\"")
                        context_parts.append("")
        
        # SECTION 4: HISTORICAL MESSAGES (only if relevant to query or asking about history)
        if conversation_keywords & query_keywords and results['episodic_messages']:
            # Deduplicate historical messages by content
            seen_historical = {}
            for item in results['episodic_messages'][:10]:
                content_key = f"{item['role']}:{item['content'].strip().lower()}"
                # Skip if already in recent conversations
                if content_key not in seen_content and content_key not in seen_historical:
                    seen_historical[content_key] = item
            
            # Display unique historical messages as sentences (limit to 3)
            historical_msgs = list(seen_historical.values())[:3]
            if historical_msgs:
                context_parts.append("Earlier conversations:")
                for item in historical_msgs:
                    timestamp = item['created_at'].strftime('%b %d at %I:%M %p') if item['created_at'] else 'previously'
                    if item['role'] == 'user':
                        context_parts.append(f"On {timestamp}, you asked: \"{item['content']}\"")
                    else:
                        context_parts.append(f"On {timestamp}, the response was: \"{item['content']}\"")
                    timestamp = item['created_at'].strftime('%b %d, %Y %I:%M %p') if item['created_at'] else 'Unknown time'
                context_parts.append(f"[{timestamp}] {item['role'].upper()}: {item['content']}")
            context_parts.append("")
        
        # If asking about specific time, get messages from that time
        if time_match or target_date:
            query_date = target_date if target_date else datetime.now().date()
            
            print(f"   🔍 Querying messages for date: {query_date}")
            
            cur.execute("""
                SELECT scm.role, scm.content, scm.created_at
                FROM super_chat_messages scm
                JOIN super_chat sc ON scm.super_chat_id = sc.id
                WHERE sc.user_id = %s
                  AND scm.created_at::date = %s
                ORDER BY scm.created_at DESC
                LIMIT 100
            """, (self.user_id, query_date))
            recent_messages = cur.fetchall()
            
            print(f"   ✅ Found {len(recent_messages)} messages for {query_date.strftime('%B %d, %Y')}")
            
            if recent_messages:
                date_str = query_date.strftime('%B %d, %Y')
                context_parts.append(f"\nFULL CONVERSATION HISTORY FOR {date_str}:")
                for msg in recent_messages:
                    timestamp = msg['created_at'].strftime('%I:%M %p') if msg['created_at'] else 'Unknown'
                    context_parts.append(f"- [{timestamp}] {msg['role']}: {msg['content']}")
            else:
                context_parts.append(f"\nNo conversations found for {query_date.strftime('%B %d, %Y')}")
        
        cur.close()
        
        # Count actually relevant sources (after filtering)
        filtered_sources = len(all_candidates)
        print(f"\n✅ Context assembly complete: {filtered_sources} relevant sources (filtered from {total_sources} retrieved)")
        print(f"{'='*70}\n")
        
        # Build context WITHOUT optimization to preserve formatting
        full_context = "\n".join(context_parts)
        original_tokens = len(full_context) // 4
        
        # Initialize stats
        opt_stats = {
            'reduction_percentage': 0, 
            'original_tokens': original_tokens, 
            'final_tokens': original_tokens
        }
        
        # SKIP OPTIMIZATION - it destroys formatting
        # Just use the clean, formatted context as-is
        print(f"   ✓ Using formatted context (optimization disabled to preserve structure)")
        print(f"   ✓ Tokens: {original_tokens}")
        
        # STEP 3: CONTEXT DISPLAY & MODEL SELECTION
        print(f"\n{'='*70}")
        print(f"📺 STEP 3: FINAL CONTEXT & MODEL SELECTION")
        print(f"{'='*70}\n")
        
        # MODEL SELECTION
        model_name = "llama-3.3-70b-versatile"  # Default
        model_reason = "Versatile general-purpose model"
        rag_insights = {}
        
        # Prepare search results summary for model selection
        search_results_summary = {
            'total_results': total_sources,
            'semantic_knowledge': len(results['semantic_knowledge']),
            'episodic_messages': len(results['episodic_messages']),
            'episodic_episodes': len(results['episodic_episodes']),
            'temporary_cache': len(results.get('temp_memory', [])),
            'context_size_chars': len(full_context),
            'context_size_tokens': len(full_context) // 4
        }
        
        if self.model_selector:
            print(f"🎯 RAG-Enhanced Model Selection...")
            print(f"   ├─ Analyzing: Task type, query, search results")
            print(f"   ├─ Input Query: {message[:60]}{'...' if len(message) > 60 else ''}")
            print(f"   ├─ Search Results: {total_sources} items from {len([k for k, v in search_results_summary.items() if k.endswith(('knowledge', 'messages', 'episodes', 'cache')) and v > 0])} layers")
            print(f"   ├─ Retrieving: Historical performance data")
            print(f"   └─ Deciding: Best model based on learned patterns\n")
            
            # Include search results in query context for better model selection
            enhanced_query_context = f"{message}\n[Search Results: {total_sources} items, {len(full_context)} chars]"
            
            model_name, model_reason, rag_insights = self.model_selector.select_model_with_rag(
                task_type="chat",
                query_context=enhanced_query_context,
                user_id=self.user_id,
                verbose=True
            )
            
            print(f"\n✅ MODEL SELECTED: {model_name}")
            print(f"   ┌─────────────────────────────────────────────┐")
            print(f"   │ WHY THIS MODEL?                              │")
            print(f"   ├─────────────────────────────────────────────┤")
            print(f"   │ {model_reason[:44]:44} │")
            print(f"   └─────────────────────────────────────────────┘")
            
            # Display input summary
            print(f"\n   📥 INPUT SUMMARY:")
            print(f"      ├─ Query: {message[:50]}{'...' if len(message) > 50 else ''}")
            print(f"      ├─ Search Results: {search_results_summary['total_results']} items")
            print(f"      │  ├─ Semantic Knowledge: {search_results_summary['semantic_knowledge']}")
            print(f"      │  ├─ Episodic Messages: {search_results_summary['episodic_messages']}")
            print(f"      │  ├─ Episodic Episodes: {search_results_summary['episodic_episodes']}")
            print(f"      │  └─ Temporary Cache: {search_results_summary['temporary_cache']}")
            print(f"      └─ Context: {search_results_summary['context_size_tokens']} tokens")
            
            if rag_insights:
                print(f"\n   📊 RAG INSIGHTS:")
                if 'cache_hit' in rag_insights:
                    cache_status = "✓ Hit (instant retrieval)" if rag_insights['cache_hit'] else "✗ Miss (DB query performed)"
                    print(f"      ├─ Cache: {cache_status}")
                if 'similar_contexts' in rag_insights:
                    print(f"      ├─ Similar past queries: {rag_insights['similar_contexts']}")
                if 'avg_success_rate' in rag_insights:
                    success_rate = rag_insights['avg_success_rate']
                    rating = "⭐⭐⭐⭐⭐" if success_rate >= 90 else "⭐⭐⭐⭐" if success_rate >= 75 else "⭐⭐⭐" if success_rate >= 60 else "⭐⭐"
                    print(f"      ├─ Historical success: {success_rate:.1f}% {rating}")
                if 'performance_data' in rag_insights:
                    print(f"      └─ Based on {rag_insights.get('total_records', 0)} past interactions")
        else:
            print(f"\n✅ MODEL SELECTED: {model_name}")
            print(f"   └─ Reason: {model_reason}")
        
        # Display final optimized context with proper formatting
        print(f"\n{'='*70}")
        print(f"📋 FINAL CONTEXT")
        print(f"{'='*70}")
        print(f"📊 Metrics: {opt_stats['final_tokens']} tokens | {total_sources} sources")
        print(f"{'='*70}\n")
        print(full_context)
        print(f"\n{'='*70}")
        print(f"End of Context")
        print(f"{'='*70}\n")
        
        # STEP 4: STORING IN MEMORY SYSTEM
        print(f"\n{'='*70}")
        print(f"💾 STEP 4: STORING IN MEMORY SYSTEM (Feedback Loop)")
        print(f"{'='*70}")
        self.add_chat_message("user", message)
        # No assistant response to store since we're not generating one
        print(f"   ✓ User question stored → EPISODIC (super_chat_messages)")
        print(f"   ✓ Data synced to Temporary System (Redis cache)\n")
    
    def retrieve_and_respond(self, stored_text: str):
        """Retrieve relevant context from storage layers and provide intelligent response"""
        print(f"\n   🔍 Retrieving from storage layers...")
        
        try:
            # Perform hybrid search on what was just stored (suppress hybrid_search print)
            cur = self.conn.cursor()
            
            # Quick search in knowledge_base
            cur.execute("""
                SELECT id, content, category
                FROM knowledge_base
                WHERE user_id = %s 
                  AND content ILIKE %s
                ORDER BY created_at DESC
                LIMIT 3
            """, (self.user_id, f'%{stored_text[:50]}%'))
            
            knowledge_results = cur.fetchall()
            
            # Get user persona
            cur.execute("""
                SELECT name, raw_content, interests, expertise_areas 
                FROM user_persona 
                WHERE user_id = %s
            """, (self.user_id,))
            persona = cur.fetchone()
            
            cur.close()
            
            total_retrieved = len(knowledge_results)
            
            if total_retrieved > 0 or persona:
                print(f"   ✓ Retrieved {total_retrieved} related items from storage")
                
                # Build contextual response
                context_parts = []
                
                if persona and persona['name']:
                    context_parts.append(f"User: {persona['name']}")
                    if persona['interests']:
                        context_parts.append(f"Interests: {', '.join(persona['interests'][:3])}")
                
                # Add knowledge context
                if knowledge_results:
                    context_parts.append(f"\nRelated knowledge:")
                    for item in knowledge_results[:2]:
                        context_parts.append(f"  • [{item['category']}] {item['content'][:60]}...")
                
                # Generate AI response if available
                if self.groq_client and context_parts:
                    # Select model for context analysis
                    model_name, model_reason = select_model_for_task("analysis")
                    
                    try:
                        full_context = "\n".join(context_parts)
                        response = self.groq_client.chat.completions.create(
                            model=model_name,
                            messages=[
                                {"role": "system", "content": f"""You are a helpful memory assistant. The user just stored: "{stored_text}"

Based on their memory context, provide a brief, relevant acknowledgment or insight (1-2 sentences).

MEMORY CONTEXT:
{full_context}"""},
                                {"role": "user", "content": f"I just stored: {stored_text}"}
                            ],
                            temperature=0.7,
                            max_tokens=150
                        )
                        reply = response.choices[0].message.content
                        print(f"\n   💡 {reply}")
                    except Exception as e:
                        # Silently fail - already showed storage confirmation
                        pass
            else:
                print(f"   ℹ️  This is your first entry")
        except Exception as e:
            # Don't break the flow if retrieval fails
            print(f"   ℹ️  Storage confirmed")


if __name__ == "__main__":
    print(f"\n🚀 Starting Interactive Memory System")
    print(f"   Context Optimization: ENABLED (Balanced Profile)\n")
    
    app = InteractiveMemorySystem()
    app.run()
