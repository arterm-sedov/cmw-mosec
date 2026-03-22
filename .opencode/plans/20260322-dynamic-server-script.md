# Plan: Create Dynamic v2 Server Script

## Objective
Create a static v2 server that fetches model configurations at runtime using the same sources as cmw-mosec parent.

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
- Comments: explain why, not what

## Files TO CREATE

### 1. `cmw_mosec/v2/__init__.py`
```python
from .dynamic_server import run_server

__all__ = ["run_server"]
```

### 2. `cmw_mosec/v2/workers.py`
Create these classes with dynamic `__init__`:

**EmbeddingWorkerV2(Worker)**:
- `__init__`: Call `load_server_settings().active_embedding_model`, then `ModelRegistry().get_embedding_config()`
- Copy all methods from generated EmbeddingWorker but use `self.max_length` instead of `MAX_LENGTH`

**RerankerWorkerV2(Worker)**:
- `__init__`: Call `load_server_settings().active_reranker_model`, then `ModelRegistry().get_reranker_config()`
- Copy `_compute_scores` from generated RerankerWorker

**ScoreWorkerV2(RerankerWorkerV2)**:
- Copy `forward()` from generated ScoreWorker

**RerankWorkerV2(RerankerWorkerV2)**:
- Copy `forward()` from generated RerankWorker

**GuardWorkerV2(Worker)**:
- `__init__`: Call `load_server_settings().active_guard_model`, then `ModelRegistry().get_guard_config()`
- Copy all methods from generated GuardWorker but use `self.max_new_tokens`, `self.max_length`

### 3. `cmw_mosec/v2/dynamic_server.py`
```python
from mosec import Server, Runtime
from .workers import EmbeddingWorkerV2, ScoreWorkerV2, RerankWorkerV2, GuardWorkerV2

def run_server():
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

## FILES TO MODIFY

### `cmw_mosec/server_manager.py`

Add new method in `MosecServerManager` class:
```python
def start_v2(self, background: bool = True) -> tuple[bool, list[str]]:
    """Start v2 server with dynamic config."""
    # Read current settings
    settings = load_server_settings()
    
    # Get port from settings.server_port
    port = settings.server_port
    
    # Spawn: python -m cmw_mosec.v2.dynamic_server
    script_path = self._script_dir / "mosec_server_v2.py"
    script_path.write_text(f"""
import sys
sys.path.insert(0, '{PROJECT_ROOT}')
from cmw_mosec.v2.dynamic_server import run_server
run_server()
""")
    
    # subprocess.Popen similar to existing start()
```

## EXACT COPY-TRANSFORM PATTERN

1. Take worker code from `~/.cmw-mosec/scripts/mosec_server.py`
2. In each `__init__`:
   - DELETE: `self.model_name = RERANKER_MODEL`
   - ADD: Config lookup from settings + registry
3. In methods:
   - REPLACE: `MAX_LENGTH` → `self.max_length`
   - REPLACE: `DTYPE` → `self.dtype`
4. Keep all business logic identical (pooling, MRL truncation, scoring methods)

## Verification Checklist

1. Tests pass: pytest
2. Lint passes: ruff check
3. Shared logic (DRY): Reuse ModelRegistry patterns
4. Configs in YAML: Already done
5. Scores identical across endpoints: v1 and v2 produce same results
6. All models tested: Test with FRIDA, Qwen3, DiTy, BAAI, Guard
7. CLI commands work: Both v1 and v2 commands functional
8. Other endpoints unchanged: v1 continues working
9. README updated: Document v2 usage

## Verification Steps

1. `ruff check cmw_mosec/v2/`
2. `python -c "from cmw_mosec.v2 import run_server; print('import ok')"`
3. Test: `cmw-mosec start-v2`
4. `curl http://localhost:<port>/v2/embeddings`
5. Compare responses with v1 endpoints

## Implementation Order

1. Create `cmw_mosec/v2/__init__.py`
2. Create `cmw_mosec/v2/workers.py` - EmbeddingWorkerV2 first
3. Create `cmw_mosec/v2/dynamic_server.py`
4. Test import: `python -c "from cmw_mosec.v2 import run_server"`
5. Add `start_v2()` to server_manager.py
6. Add CLI command
7. Full integration test
