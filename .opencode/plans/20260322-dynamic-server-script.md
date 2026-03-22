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

---

## CURRENT IMPLEMENTATION ANALYSIS

### server_manager.py start() does:
1. Check if running → return early (idempotent)
2. Load settings via `load_server_settings()`
3. Validate models via `registry.get_config()`
4. Generate server script via `_generate_server_script()`
5. Write script to `~/.cmw-mosec/scripts/mosec_server.py`
6. Spawn subprocess: `python script.py --port X`
7. Pass `HF_TOKEN` via env
8. Wait for health check (60 attempts × 1 sec)
9. Save PID info with loaded models
10. Return success/failure + failed models list

### server_manager.py stop() does:
1. Load PID info
2. Send SIGTERM
3. Wait 10 sec for graceful shutdown
4. Fallback to SIGKILL
5. Remove PID file

### CLI serve() does:
1. Load active models from .env
2. Accept --embedding/--reranker/--guard overrides
3. Call manager.start()
4. Display endpoints

---

## PLAN: Files TO CREATE

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

---

## PLAN: Files TO MODIFY

### `cmw_mosec/server_manager.py`

Add `start_v2()` method that mirrors `start()` but:
1. **Idempotent check**: `if self.is_running(): return True, []`
2. **Model validation**: Call `registry.get_config()` for each model
3. **Write launcher script**: Creates wrapper that imports `cmw_mosec.v2.dynamic_server`
4. **Spawn subprocess**: `subprocess.Popen()` with same options as `start()`
5. **Pass env vars**: `HF_TOKEN` if set
6. **Health check**: Wait 60 sec for `/v2/embeddings` or health endpoint
7. **PID management**: Save PID with loaded models to `~/.cmw-mosec/pid.json`

```python
def start_v2(
    self,
    embedding_model: str | None = None,
    reranker_model: str | None = None,
    guard_model: str | None = None,
    background: bool = True,
) -> tuple[bool, list[str]]:
    """Start v2 server with dynamic config.
    
    Mirrors start() but uses cmw_mosec.v2.dynamic_server
    instead of generated script.
    """
    # 1. Check if running (idempotent)
    if self.is_running():
        logger.info("Server already running")
        return True, []
    
    # 2. Load settings
    settings = load_server_settings()
    
    # 3. Validate models
    registry = ModelRegistry()
    # ... validation logic same as start() ...
    
    # 4. Write launcher script
    script_path = self._script_dir / "mosec_server_v2.py"
    script_path.write_text(f"""
import sys
sys.path.insert(0, '{PROJECT_ROOT}')
from cmw_mosec.v2.dynamic_server import run_server
run_server()
""")
    
    # 5. Spawn subprocess
    cmd = [sys.executable, str(script_path), "--port", str(settings.server_port)]
    env = os.environ.copy()
    if settings.hf_token:
        env["HF_TOKEN"] = settings.hf_token
    
    process = subprocess.Popen(cmd, ..., env=env)
    
    # 6. Save PID
    _save_server_pid(process.pid, settings.server_port, {
        "embedding": emb_model,
        "reranker": rer_model,
        "guard": guard_m,
        "version": "v2",  # Track which version started
    })
    
    # 7. Health check
    for _ in range(60):
        if _check_server_health_v2(settings.server_port):
            return True, failed_models
        time.sleep(1)
```

### `cmw_mosec/cli.py`

Add `serve-v2` command:
```python
@cli.command()
@click.option("--foreground", "-f", is_flag=True)
@click.option("--embedding", help="Embedding model")
@click.option("--reranker", help="Reranker model")
@click.option("--guard", help="Guard model")
def serve_v2(foreground, embedding, reranker, guard):
    """Start v2 server with dynamic config."""
    manager = MosecServerManager()
    # Same logic as serve() but calls start_v2()
    # Display /v2/* endpoints
```

---

## EXACT COPY-TRANSFORM PATTERN

1. Take worker code from `~/.cmw-mosec/scripts/mosec_server.py`
2. In each `__init__`:
   - DELETE: `self.model_name = RERANKER_MODEL`
   - ADD: Config lookup from settings + registry
3. In methods:
   - REPLACE: `MAX_LENGTH` → `self.max_length`
   - REPLACE: `DTYPE` → `self.dtype`
4. Keep all business logic identical (pooling, MRL truncation, scoring methods)

---

## COMPLETE FEATURE MAPPING

| Feature | v1 (current) | v2 (plan) |
|---------|--------------|-----------|
| Config source | Baked at generation | Runtime lookup |
| Script | Generated | Static in v2/ |
| Model validation | Yes | Yes |
| Health check | Yes | Yes (via /v2/) |
| PID management | Yes | Yes |
| HF_TOKEN pass | Yes | Yes |
| Graceful shutdown | Yes (SIGTERM→KILL) | Yes (same) |
| Idempotent start | Yes | Yes |
| Endpoints | /v1/* | /v2/* |

---

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
3. Test: `cmw-mosec serve-v2 --embedding ai-forever/FRIDA`
4. `curl http://localhost:<port>/v2/embeddings`
5. `curl http://localhost:<port>/v1/embeddings` (confirm v1 still works)
6. Compare responses with v1 endpoints

## Implementation Order

1. Create `cmw_mosec/v2/__init__.py`
2. Create `cmw_mosec/v2/workers.py` - EmbeddingWorkerV2 first
3. Create `cmw_mosec/v2/dynamic_server.py`
4. Test import: `python -c "from cmw_mosec.v2 import run_server"`
5. Add `start_v2()` to server_manager.py (mirror start())
6. Add `serve-v2` CLI command (mirror serve())
7. Full integration test
