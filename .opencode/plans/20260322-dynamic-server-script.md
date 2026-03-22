# Plan: Create Dynamic v2 Server Script

## Objective
Create a static v2 server that fetches model configurations at runtime using the same sources as cmw-mosec parent, rather than having them baked in during generation.

## Core Insight
Create a new directory `cmw_mosec/v2/` with static Python files that:
1. Fetch configuration at runtime from cmw-mosec's sources (ModelRegistry, settings, env vars)
2. Serve at `/v2/` endpoints
3. Follow existing cmw-mosec patterns exactly

## Architecture

```
cmw_mosec/v2/
├── __init__.py           # Package exports
├── workers.py           # Worker classes with dynamic config lookup
└── dynamic_server.py    # Mosec server setup with /v2/ endpoints
```

## Implementation Steps

### 1. Create cmw_mosec/v2/__init__.py
Export workers and server components.

### 2. Create cmw_mosec/v2/workers.py
Worker classes that fetch config dynamically:

- **EmbeddingWorkerV2**: Uses `registry.get_embedding_config()`, `settings.active_embedding_model`
- **RerankerWorkerV2**: Uses `registry.get_reranker_config()`, `settings.active_reranker_model`
- **ScoreWorkerV2**: Extends RerankerWorkerV2, vLLM format
- **RerankWorkerV2**: Extends RerankerWorkerV2, Cohere format
- **GuardWorkerV2**: Uses `registry.get_guard_config()`, `settings.active_guard_model`

Each worker `__init__` fetches config like:
```python
from cmw_mosec.server_config import ModelRegistry, load_server_settings

settings = load_server_settings()
registry = ModelRegistry()
config = registry.get_embedding_config(settings.active_embedding_model.lower())
```

### 3. Create cmw_mosec/v2/dynamic_server.py
Mosec server that registers `/v2/` endpoints:
- `/v2/embeddings` → EmbeddingWorkerV2
- `/v2/score` → ScoreWorkerV2 (vLLM format)
- `/v2/rerank` → RerankWorkerV2 (Cohere format)
- `/v2/moderate` → GuardWorkerV2

### 4. Update server_manager.py
Add method to launch v2 server:
- Read current settings (`load_server_settings()`)
- Set environment variables for active models
- Spawn v2 server via subprocess
- Use same PID management as v1

### 5. Update CLI (optional)
Add `cmw-mosec start-v2` command or flag.

## Configuration Flow

```
cmw-mosec start → server_manager.py → v2/dynamic_server.py → v2/workers.py
                      ↓                                       ↓
              load_server_settings()              registry.get_*_config()
                      ↓
              active_embedding_model etc.
```

## Key Design Principles (from AGENTS.md)

1. **Lean**: Minimal code, no overengineering
2. **DRY**: Reuse existing ModelRegistry and settings loading
3. **Non-breaking**: v1 endpoints unchanged
4. **12-Factor**: Config in env vars, stateless processes
5. **Error Handling**: Use logger, try/except around process ops

## Code Requirements

- Type hints required
- Google docstring convention
- Line length: 100
- ruff for linting
- snake_case for functions/variables, PascalCase for classes

## Verification Checklist

1. **Tests pass**: pytest
2. **Lint passes**: ruff check
3. **Shared logic (DRY)**: Reuse ModelRegistry patterns
4. **Configs in YAML**: Already done
5. **Scores identical across endpoints**: v1 and v2 produce same results
6. **All models tested**: Test with FRIDA, Qwen3, DiTy, BAAI, Guard
7. **CLI commands work**: Both v1 and v2 commands functional
8. **Other endpoints unchanged**: v1 continues working
9. **README updated**: Document v2 usage

## Validation Steps

1. Verify v2 workers fetch config correctly
2. Test v2 endpoints respond correctly with different models
3. Confirm v1 endpoints still work (backward compatibility)
4. Verify config changes picked up without regeneration
5. Test start/stop lifecycle
6. Health check via HTTP
7. Multiple start calls (idempotent)
