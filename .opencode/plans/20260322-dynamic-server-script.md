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

---

## MOSEC BEST PRACTICES (validated against ~/mosec/examples/)

### Pattern from embedding/server.py
```python
class Embedding(Worker):
    def __init__(self):
        self.model_name = os.getenv("EMB_MODEL", DEFAULT_MODEL)

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

---

## FILES TO CREATE

### 1. `cmw_mosec/v2/__init__.py`
```python
"""Dynamic v2 server - no script generation, runtime config."""

from .dynamic_server import run_server

__all__ = ["run_server"]
```

### 2. `cmw_mosec/v2/workers.py`
Worker classes with dynamic config lookup.

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
"""Dynamic v2 server - follows Mosec examples pattern.

Port is handled by Mosec CLI automatically:
- CLI: python -m cmw_mosec.v2.dynamic_server --port 8000
- Env: MOSEC_PORT=8000 python -m cmw_mosec.v2.dynamic_server
"""

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
    server.run()  # Mosec parses --port / MOSEC_PORT automatically

if __name__ == "__main__":
    run_server()
```

---

## FILES TO MODIFY

### `cmw_mosec/server_manager.py`

Add `start_v2()` method - **exact same structure as `start()`** but:
- NO script generation
- Pass env vars: `ACTIVE_EMBEDDING_MODEL`, `ACTIVE_RERANKER_MODEL`, `ACTIVE_GUARD_MODEL`
- Spawn: `python -m cmw_mosec.v2.dynamic_server --port X`
- Save PID with `"version": "v2"` for tracking

```python
def start_v2(
    self,
    embedding_model: str | None = None,
    reranker_model: str | None = None,
    guard_model: str | None = None,
    background: bool = True,
) -> tuple[bool, list[str]]:
    """Start v2 server with dynamic config."""
    # 1. Idempotent check - same as start()
    if self.is_running():
        logger.info("Server already running")
        return True, []

    # 2. Load settings - same as start()
    try:
        settings = load_server_settings()
    except Exception as e:
        logger.error(f"Failed to load settings: {e}")
        return False, ["settings"]

    # 3. Model validation - SAME as start()
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
            
    # ... same validation for reranker and guard ...

    if not emb_model and not rer_model and not guard_m:
        logger.error("No valid models to load")
        return False, failed_models

    # 4. Spawn v2 (NO script generation)
    env = os.environ.copy()
    env["ACTIVE_EMBEDDING_MODEL"] = emb_model or ""
    env["ACTIVE_RERANKER_MODEL"] = rer_model or ""
    env["ACTIVE_GUARD_MODEL"] = guard_m or ""
    
    if settings.hf_token:
        env["HF_TOKEN"] = settings.hf_token

    cmd = [sys.executable, "-m", "cmw_mosec.v2.dynamic_server", "--port",
           str(settings.server_port)]

    # 5. Process management - SAME as start()
    if background:
        process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, 
                                    stderr=subprocess.DEVNULL,
                                    start_new_session=True, env=env)
    else:
        process = subprocess.Popen(cmd, env=env)

    # 6. Save PID with version tracking
    loaded_models = {
        "embedding": emb_model,
        "reranker": rer_model,
        "guard": guard_m,
        "version": "v2",  # KEY: tracks which version started
    }
    _save_server_pid(process.pid, settings.server_port, loaded_models)

    # 7. Health check - SAME as start()
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

Add `serve-v2` command (mirror `serve()` but calls `start_v2()`)

---

## ROBUSTNESS COMPARISON

| Feature | v1 | v2 | Status |
|---------|-----|-----|--------|
| Idempotent start | ✓ | ✓ | Must match |
| Settings loading | ✓ | ✓ | Must match |
| Model validation | ✓ | ✓ | Must match |
| Env vars pass | HF_TOKEN | HF_TOKEN + ACTIVE_* | Must match |
| Process spawn | ✓ | ✓ | Must match |
| PID save | ✓ | ✓ + version | Enhanced |
| Health check | ✓ | ✓ | Must match |
| Background mode | ✓ | ✓ | Must match |
| Foreground mode | ✓ | ✓ | Must match |
| Error handling | ✓ | ✓ | Must match |
| Graceful shutdown | ✓ | ✓ | Must match |

---

## Implementation Order

1. Create `cmw_mosec/v2/__init__.py`
2. Create `cmw_mosec/v2/workers.py`:
   - Copy worker methods from `~/.cmw-mosec/scripts/mosec_server.py`
   - Replace hardcoded constants (`EMBEDDING_MODEL`, `MAX_LENGTH`, etc.) with `self.*` from config lookup
   - Add dynamic `__init__` that reads env vars + ModelRegistry
3. Create `cmw_mosec/v2/dynamic_server.py` - Mosec standard pattern
4. Test direct run: `python -m cmw_mosec.v2.dynamic_server --port 8000`
5. Add `start_v2()` to server_manager.py (exact mirror of `start()`)
6. Add `serve-v2` CLI command
7. Full integration test

---

## Verification Checklist

1. `ruff check cmw_mosec/v2/`
2. Direct run: `python -m cmw_mosec.v2.dynamic_server --port 8000` (needs env vars set externally)
3. Health check: `curl http://localhost:8000/metrics`
4. `cmw-mosec serve-v2 --embedding ai-forever/FRIDA`
5. Compare v2 `/v2/embeddings` with v1 `/v1/embeddings`: identical results
6. Test all model types: FRIDA, Qwen3, DiTy, BAAI, Guard
7. v1 still works: `cmw-mosec serve` unchanged
8. Tests pass: `pytest`
9. `ruff check cmw_mosec/server_manager.py`
