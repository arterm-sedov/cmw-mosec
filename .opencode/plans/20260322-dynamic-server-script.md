# Plan: Create Dynamic v2 Server - No Script Generation

## Objective
Combine server_manager logic (start/stop/health check/PID) with workers (embedding/reranker/guard) into a coherent v2 that:
- **NO script generation** - workers directly in package
- **Dynamic config at runtime** - fetches from ModelRegistry + env vars
- **Follows Mosec best practices** - like examples/embedding/server.py

## Design Principles (from AGENTS.md)

- **Lean**: Minimal code, no overengineering
- **DRY**: Reuse existing ModelRegistry and settings loading
- **Non-breaking**: v1 endpoints unchanged
- **12-Factor**: Config in env vars, stateless processes
- **Error Handling**: Use logger, try/except around process ops

## Code Requirements

- Type hints required
- Google docstring convention
- Line length: 100
- ruff for linting
- snake_case for functions/variables, PascalCase for classes
- Include Apache 2.0 license header (per Mosec examples)

---

## MOSEC BEST PRACTICES (validated against ~/mosec/examples/)

### Pattern from embedding/server.py
```python
class Embedding(Worker):
    def __init__(self):
        self.model_name = os.getenv("EMB_MODEL", DEFAULT_MODEL)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
        self.model = transformers.AutoModel.from_pretrained(self.model_name)

if __name__ == "__main__":
    server = Server()
    emb = Runtime(Embedding)
    server.register_runtime({"/v1/embeddings": [emb]})
    server.run()
```

### Key Mosec patterns:
1. `os.getenv("MODEL_NAME", default)` for runtime config
2. `Runtime(Worker)` wraps worker for dynamic batching
3. `server.register_runtime({endpoint: [Runtime(...)]})` registers routes
4. Workers inherit `Worker` class
5. Simple `__init__` loads model, `forward()` handles inference
6. Optional: `max_batch_size=N` for dynamic batching performance

---

## FILES TO CREATE

### 1. `cmw_mosec/v2/__init__.py`
```python
"""Dynamic v2 server - no script generation, runtime config."""

from .dynamic_server import run_server

__all__ = ["run_server"]
```

### 2. `cmw_mosec/v2/workers.py`
Apache 2.0 header + worker classes

**EmbeddingWorkerV2(Worker)**:
```python
class EmbeddingWorkerV2(Worker):
    def __init__(self):
        from cmw_mosec.server_config import ModelRegistry, load_server_settings
        
        settings = load_server_settings()
        registry = ModelRegistry()
        
        # Follow Mosec pattern: env var with fallback
        model_slug = os.getenv("ACTIVE_EMBEDDING_MODEL") or settings.active_embedding_model
        config = registry.get_embedding_config(model_slug.lower())
        
        self.model_name = config.model_id
        self.pooling = config.pooling
        self.dimensions = config.dimensions
        self.max_length = config.max_length
        self.model_class = config.model_class or "AutoModel"
        
        # ... rest mirrors examples/embedding/server.py pattern
```

**RerankerWorkerV2(Worker)**: Uses `os.getenv("ACTIVE_RERANKER_MODEL")`

**ScoreWorkerV2(RerankerWorkerV2)**: Extends, vLLM format

**RerankWorkerV2(RerankerWorkerV2)**: Extends, Cohere format

**GuardWorkerV2(Worker)**: Uses `os.getenv("ACTIVE_GUARD_MODEL")`

### 3. `cmw_mosec/v2/dynamic_server.py`
```python
"""Dynamic v2 server - follows Mosec examples pattern."""

from mosec import Server, Runtime
from .workers import (
    EmbeddingWorkerV2,
    ScoreWorkerV2,
    RerankWorkerV2,
    GuardWorkerV2,
)

def run_server():
    """Start v2 server with runtime-configured workers."""
    server = Server()
    
    routes = {
        "/v2/embeddings": [Runtime(EmbeddingWorkerV2)],
        "/v2/score": [Runtime(ScoreWorkerV2)],
        "/v2/rerank": [Runtime(RerankWorkerV2)],
        "/v2/moderate": [Runtime(GuardWorkerV2)],
    }
    
    server.register_runtime(routes)
    server.run()

if __name__ == "__main__":
    run_server()
```

---

## FILES TO MODIFY

### `cmw_mosec/server_manager.py`

Add `start_v2()` method:
- Same validation as `start()`
- NO script generation
- Pass env vars: `ACTIVE_EMBEDDING_MODEL`, `ACTIVE_RERANKER_MODEL`, `ACTIVE_GUARD_MODEL`
- Spawn: `python -m cmw_mosec.v2.dynamic_server`
- Same PID management, health check, graceful shutdown

```python
def start_v2(self, ...):
    # Same validation logic as start()
    env["ACTIVE_EMBEDDING_MODEL"] = emb_model
    env["ACTIVE_RERANKER_MODEL"] = rer_model
    env["ACTIVE_GUARD_MODEL"] = guard_m
    
    cmd = [sys.executable, "-m", "cmw_mosec.v2.dynamic_server"]
    # ... rest mirrors start()
```

### `cmw_mosec/cli.py`

Add `serve-v2` command (mirror `serve()` but calls `start_v2()`)

---

## COMPLETE FEATURE MAPPING

| Feature | v1 | v2 |
|---------|-----|-----|
| Config | Baked in script | Runtime via env + registry |
| Script generation | Yes | **NO** |
| Model validation | Yes | Yes |
| PID management | Yes | Yes |
| HF_TOKEN pass | Yes | Yes |
| Graceful shutdown | Yes | Yes |
| Health check | Yes | Yes |
| Workers in package | No | **YES** |
| Mosec pattern | Custom | **Standard** |
| Runnable directly | No | **YES** (`python -m cmw_mosec.v2` |

---

## Implementation Order

1. Create `cmw_mosec/v2/__init__.py`
2. Create `cmw_mosec/v2/workers.py` - with Apache header, dynamic `__init__`
3. Create `cmw_mosec/v2/dynamic_server.py` - Mosec standard pattern
4. Test direct run: `python -m cmw_mosec.v2` (needs env vars set)
5. Test import: `python -c "from cmw_mosec.v2 import run_server"`
6. Add `start_v2()` to server_manager.py
7. Add `serve-v2` CLI command
8. Full integration test

---

## Verification Checklist

1. `ruff check cmw_mosec/v2/`
2. Direct run: `ACTIVE_EMBEDDING_MODEL=ai-forever/FRIDA python -m cmw_mosec.v2`
3. `curl http://localhost:<port>/v2/embeddings`
4. `cmw-mosec serve-v2 --embedding ai-forever/FRIDA`
5. Compare with v1: same results
6. Test all model types
7. v1 still works
8. Tests pass: pytest
