# Plan: Dynamic Server Implementation

## Objective
Replace the generated script approach with a dynamic server that fetches model configurations at runtime, while maintaining the same API contract (/v1/* endpoints).

## Key Insight
- API contract stays the same (v1 endpoints)
- Implementation changes from script generation to dynamic config
- Existing tests validate the new implementation

## Design Principles (from AGENTS.md)

- **Lean**: Minimal code, no overengineering
- **DRY**: Reuse existing ModelRegistry and settings loading
- **Non-breaking**: Same API contract (/v1/* endpoints)
- **12-Factor**: Config in env vars, stateless processes
- **Error Handling**: Use logger, try/except around process ops

## Code Requirements

- Type hints required
- Google docstring convention
- Line length: 100
- ruff for linting
- snake_case for functions/variables, PascalCase for classes

---

## Files Created

### `cmw_mosec/v2/__init__.py`
```python
from .dynamic_server import run_server
__all__ = ["run_server"]
```

### `cmw_mosec/v2/workers.py`
Worker classes with dynamic config:
- EmbeddingWorkerV2
- RerankerWorkerV2
- ScoreWorkerV2
- RerankWorkerV2
- GuardWorkerV2

Each worker reads config from:
1. Environment variables (ACTIVE_EMBEDDING_MODEL, etc.)
2. ModelRegistry

### `cmw_mosec/v2/dynamic_server.py`
Standard Mosec server pattern:
- Reads ACTIVE_* env vars
- Conditionally registers routes
- Uses /v1/* endpoints (same API contract)

## Files Modified

### `cmw_mosec/server_manager.py`
- Removed `_generate_server_script()` (dead code)
- Removed `start_v2()` (merged into `start()`)
- `start()` now uses dynamic server with env vars
- Removed unused `ServerSettings` import

### `cmw_mosec/cli.py`
- Removed `serve_v2` command (no longer needed)
- `serve` command works as before

## Implementation Summary

1. Created v2 workers with dynamic config lookup
2. Created v2 dynamic_server following Mosec patterns
3. Modified start() to use v2 dynamic server
4. Removed script generation code
5. Removed start_v2() and serve_v2()
6. Serves at /v1/* endpoints (same API contract)

## Verification

1. `ruff check cmw_mosec/` - All checks passed
2. Import tests passed
3. Existing tests will validate new implementation
4. Same API contract maintained
