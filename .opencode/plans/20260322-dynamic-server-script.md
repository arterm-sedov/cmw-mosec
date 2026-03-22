# Plan: Create Dynamic v2 Server - No Script Generation

## Objective
Create v2 with **identical behavior** to current cmw-mosec but:
- **NO script generation** - workers in package
- **v2 endpoints** (`/v2/*`)
- **Dynamic config at runtime**
- **Follows Mosec best practices**

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

---

## MOSEC BEST PRACTICES

```python
class Embedding(Worker):
    def __init__(self):
        self.model_name = os.getenv("EMB_MODEL", DEFAULT_MODEL)

if __name__ == "__main__":
    server = Server()
    server.register_runtime({"/v1/embeddings": [Runtime(Embedding)]})
    server.run()
```

---

## FILES TO CREATE

### 1. `cmw_mosec/v2/__init__.py`
```python
"""Dynamic v2 server - no script generation."""

from .dynamic_server import run_server

__all__ = ["run_server"]
```

### 2. `cmw_mosec/v2/workers.py`

Worker classes with **graceful handling when not configured**.

**EmbeddingWorkerV2(Worker)**:
```python
class EmbeddingWorkerV2(Worker):
    def __init__(self):
        from cmw_mosec.server_config import ModelRegistry
        
        model_slug = os.getenv("ACTIVE_EMBEDDING_MODEL", "")
        if not model_slug:
            return  # Worker not configured, skip loading
        
        registry = ModelRegistry()
        config = registry.get_embedding_config(model_slug.lower())
        
        self.model_name = config.model_id
        self.pooling = config.pooling
        self.dimensions = config.dimensions
        self.max_length = config.max_length
        self.model_class = config.model_class or "AutoModel"
        
        # Load model (same as current EmbeddingWorker)
```

**Key**: Worker checks env var first. If empty/not set, `__init__` returns early without loading model.

**RerankerWorkerV2(Worker)**: Checks `os.getenv("ACTIVE_RERANKER_MODEL")`

**ScoreWorkerV2(RerankerWorkerV2)**: Returns vLLM format

**RerankWorkerV2(RerankerWorkerV2)**: Returns Cohere format

**GuardWorkerV2(Worker)**: Checks `os.getenv("ACTIVE_GUARD_MODEL")`

### 3. `cmw_mosec/v2/dynamic_server.py`
```python
"""Dynamic v2 server - Mosec standard pattern."""

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
    
    routes = {}
    
    # Conditionally register based on env vars (matches v1 behavior)
    if os.getenv("ACTIVE_EMBEDDING_MODEL"):
        routes["/v2/embeddings"] = [Runtime(EmbeddingWorkerV2)]
    
    if os.getenv("ACTIVE_RERANKER_MODEL"):
        routes["/v2/score"] = [Runtime(ScoreWorkerV2)]
        routes["/v2/rerank"] = [Runtime(RerankWorkerV2)]
    
    if os.getenv("ACTIVE_GUARD_MODEL"):
        routes["/v2/moderate"] = [Runtime(GuardWorkerV2)]
    
    server.register_runtime(routes)
    server.run()

if __name__ == "__main__":
    run_server()
```

---

## FILES TO MODIFY

### `cmw_mosec/server_manager.py`

Add `start_v2()` - **exact same as `start()`** but:
- NO script generation
- Pass env vars: `ACTIVE_EMBEDDING_MODEL`, `ACTIVE_RERANKER_MODEL`, `ACTIVE_GUARD_MODEL`
- Spawn: `python -m cmw_mosec.v2.dynamic_server --port X`
- Save PID with `"version": "v2"`

```python
def start_v2(
    self,
    embedding_model: str | None = None,
    reranker_model: str | None = None,
    guard_model: str | None = None,
    background: bool = True,
) -> tuple[bool, list[str]]:
    """Start v2 server with dynamic config."""
    # 1. Idempotent check - EXACT same as start()
    if self.is_running():
        logger.info("Server already running")
        return True, []

    # 2. Load settings - EXACT same as start()
    try:
        settings = load_server_settings()
    except Exception as e:
        logger.error(f"Failed to load settings: {e}")
        return False, ["settings"]

    # 3. Model validation - EXACT same as start()
    registry = ModelRegistry()
    emb_model = embedding_model
    rer_model = reranker_model
    guard_m = guard_model
    failed_models = []
    
    if emb_model:
        try:
            registry.get_config(emb_model)
        except ValueError as e:
            logger.error(f"Embedding model error: {e}")
            failed_models.append(f"embedding: {emb_model}")
            emb_model = None
            
    # ... same for reranker and guard ...

    # 4. Env vars (empty string if not configured - workers check these)
    env = os.environ.copy()
    env["ACTIVE_EMBEDDING_MODEL"] = emb_model or ""
    env["ACTIVE_RERANKER_MODEL"] = rer_model or ""
    env["ACTIVE_GUARD_MODEL"] = guard_m or ""
    
    if settings.hf_token:
        env["HF_TOKEN"] = settings.hf_token

    # 5. Spawn v2 (NO script generation)
    cmd = [sys.executable, "-m", "cmw_mosec.v2.dynamic_server", "--port",
           str(settings.server_port)]

    # 6. Process management - EXACT same as start()
    if background:
        process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, 
                                    stderr=subprocess.DEVNULL,
                                    start_new_session=True, env=env)
    else:
        process = subprocess.Popen(cmd, env=env)

    # 7. Save PID with version
    loaded_models = {
        "embedding": emb_model,
        "reranker": rer_model,
        "guard": guard_m,
        "version": "v2",
    }
    _save_server_pid(process.pid, settings.server_port, loaded_models)

    # 8. Health check - EXACT same as start()
    if background:
        for _ in range(60):
            if _check_server_health(settings.server_port):
                return True, failed_models
            time.sleep(1)
            if process.poll() is not None:
                _remove_server_pid()
                return False, failed_models
        return True, failed_models
    else:
        return process.wait() == 0, failed_models
```

### `cmw_mosec/cli.py`

Add `serve-v2` command - **mirror `serve()`** but calls `start_v2()`

---

## BEHAVIOR COMPARISON

| Behavior | v1 | v2 |
|----------|-----|-----|
| `serve --embedding X` | Only `/v1/embeddings` | Only `/v2/embeddings` |
| `serve --embedding X --reranker Y` | `/v1/embeddings`, `/v1/score`, `/v1/rerank` | `/v2/embeddings`, `/v2/score`, `/v2/rerank` |
| `serve --embedding X --reranker Y --guard Z` | All endpoints | All endpoints |
| Model validation | ✓ | ✓ |
| PID management | ✓ | ✓ |
| Health check | ✓ | ✓ |
| Graceful shutdown | ✓ | ✓ |

---

## Implementation Order

1. Create `cmw_mosec/v2/__init__.py`
2. Create `cmw_mosec/v2/workers.py`:
   - Copy worker methods from `~/.cmw-mosec/scripts/mosec_server.py`
   - Add graceful handling: `if not os.getenv("ACTIVE_*"): return`
   - Replace hardcoded constants with `self.*` from config
3. Create `cmw_mosec/v2/dynamic_server.py`:
   - Conditional route registration based on env vars
4. Add `start_v2()` to server_manager.py (mirror `start()`)
5. Add `serve-v2` CLI command
6. Test all combinations

---

## Verification Checklist

1. `ruff check cmw_mosec/v2/`
2. `ruff check cmw_mosec/server_manager.py`
3. `cmw-mosec serve-v2 --embedding ai-forever/FRIDA` → only `/v2/embeddings`
4. `cmw-mosec serve-v2 --reranker DiTy/cross-encoder-russian-msmarco` → only `/v2/score`, `/v2/rerank`
5. `cmw-mosec serve-v2 --embedding X --reranker Y --guard Z` → all v2 endpoints
6. Compare v2 responses with v1 responses: **identical**
7. v1 still works: `cmw-mosec serve` unchanged
8. Tests pass: `pytest`
