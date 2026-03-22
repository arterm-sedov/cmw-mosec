# Plan: Create Dynamic v2 Server Script

## Objective
Create a static v2 server script that fetches model configurations at runtime using the same sources as cmw-mosec parent, rather than having them baked in during generation.

## Core Insight
Instead of modifying the script generator, we will:
1. Create a new directory `cmw_mosec/v2/`
2. Copy the generated script structure as a static file
3. Modify worker `__init__` methods to fetch config dynamically from cmw-mosec's sources
4. Serve at `/v2/` endpoints

## Approach

### 1. Create Directory Structure
```
cmw_mosec/v2/
├── __init__.py
├── dynamic_server.py  # The dynamic v2 server script
└── workers.py          # Worker classes that fetch config dynamically
```

### 2. Take Existing Generated Script as Base
Use the generated script at `~/.cmw-mosec/scripts/mosec_server.py` as the template, but modify it to:
- Fetch configuration at runtime instead of using hardcoded constants
- Use the same patterns as cmw-mosec parent (`ModelRegistry`, `load_server_settings`)

### 3. Modify Worker Classes to Use Dynamic Config
Replace hardcoded constants with runtime lookup:

```python
# Instead of:
RERANKER_MODEL = "DiTy/cross-encoder-russian-msmarco"
MAX_LENGTH = 512

# Use:
class RerankerWorker(Worker):
    def __init__(self):
        from cmw_mosec.server_config import ModelRegistry, load_server_settings
        
        settings = load_server_settings()
        registry = ModelRegistry()
        
        # Get reranker model from settings (set by cmw-mosec) or environment
        reranker_model = getattr(settings, 'active_reranker_model', None) or os.getenv("ACTIVE_RERANKER_MODEL")
        
        config = registry.get_reranker_config(reranker_model.lower())
        
        self.model_name = config.model_id
        self.max_length = config.max_length
        self.reranker_type = config.reranker_type
        # ... etc
```

### 4. Update Endpoints to /v2/
Change from:
- `/v1/embeddings` → `/v2/embeddings`
- `/v1/score` → `/v2/score`
- `/v1/rerank` → `/v2/rerank`
- `/v1/moderate` → `/v2/moderate`

### 5. Update server_manager.py
Add a new method to launch the v2 server:
- Instead of generating a script, instantiate workers directly from `cmw_mosec.v2.workers`
- Register routes with `/v2/` endpoints
- Launch via `subprocess.Popen()` similar to current approach

## Key Differences from v1

| Aspect | v1 (Current) | v2 (New) |
|--------|-------------|----------|
| Config source | Baked in at generation | Fetched at runtime |
| Script | Generated on-demand | Static file in v2/ |
| Flexibility | Needs regeneration for config change | Works with any model combo |
| Endpoints | /v1/* | /v2/* |

## Benefits

1. **No generator modification**: Works with existing `_generate_server_script()`
2. **True dynamic config**: Workers fetch from ModelRegistry at init time
3. **Same patterns as parent**: Reuses cmw-mosec's own config loading
4. **Backward compatible**: v1 endpoints unchanged
5. **Easy to maintain**: Static file, easy to edit/debug

## Validation
1. Test v2 server starts successfully
2. Verify all v2 endpoints respond correctly
3. Confirm model-specific behaviors work
4. Ensure v1 endpoints still work
