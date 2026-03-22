# Plan: Modify Existing Script Generator for Dynamic Configuration

## Objective
Modify the existing `_generate_server_script()` function in server_manager.py to generate a server script where:
1. Model configurations are fetched at runtime by worker classes (not baked in during generation)
2. Workers use the same configuration sources as the parent cmw-mosec process
3. Endpoints are served at `/v2/` to avoid conflicts with existing v1 endpoints

## Core Insight
Instead of baking configuration constants into the generated script during generation time, we will modify worker `__init__` methods to fetch current configuration from the same sources as the parent cmw-mosec process (ModelRegistry, environment, settings) at worker initialization time.

## Transformation Plan

### 1. Modify _generate_server_script() Function
Update the existing function to generate workers that perform runtime configuration lookup instead of using hardcoded constants.

### 2. Replace Bake-Time Constants with Runtime Lookup
Instead of generating lines like:
```python
EMBEDDING_MODEL = "{embedding_model}"
DTYPE = "{embed_dtype}"
```
We will generate worker `__init__` methods that fetch these values at runtime using:
```python
from cmw_mosec.server_config import ModelRegistry, load_server_settings
import os

settings = load_server_settings()
registry = ModelRegistry()
embedding_model = getattr(settings, 'active_embedding_model', None) or os.getenv("ACTIVE_EMBEDDING_MODEL")
config = registry.get_embedding_config(embedding_model.lower())
```

### 3. Modify EmbeddingWorker.__init__()
Replace hardcoded constant assignments with runtime lookup:
```python
def __init__(self):
    from cmw_mosec.server_config import ModelRegistry, load_server_settings
    import os
    
    settings = load_server_settings()
    registry = ModelRegistry()
    
    # Get embedding model from settings (set by cmw-mosec) or environment
    embedding_model = getattr(settings, 'active_embedding_model', None) or os.getenv("ACTIVE_EMBEDDING_MODEL")
    if not embedding_model:
        # Fallback for direct execution
        embedding_model = os.getenv("EMBEDDING_MODEL", "ai-forever/FRIDA")
    
    config = registry.get_embedding_config(embedding_model.lower())
    
    self.model_name = config.model_id
    self.pooling = config.pooling
    self.dimensions = config.dimensions
    self.max_length = config.max_length
    self.model_class = config.model_class or "AutoModel"  # sensible default

    # LLM-based embedders (Qwen3) need left padding for last_token pooling
    # Encoder-based (FRIDA) use default right padding
    if self.pooling == "last_token":
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side='left'
        )
    else:
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)

    # Load model class from config (e.g., T5EncoderModel for FRIDA, AutoModel for others)
    if self.model_class == "T5EncoderModel":
        self.model = transformers.T5EncoderModel.from_pretrained(self.model_name)
    else:
        self.model = transformers.AutoModel.from_pretrained(self.model_name)

    self.device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    self.model = self.model.to(self.device)
    self.model.eval()
    if settings.dtype == "float16" and self.device != "cpu":
        self.model = self.model.half()
    elif settings.dtype == "int8":
        self.model = self.model.quantized = True
```

### 4. Modify RerankerWorker.__init__()
Apply similar dynamic lookup for reranker configuration using `registry.get_reranker_config()`.

### 5. Modify GuardWorker.__init__()
Apply similar dynamic lookup for guard configuration using `registry.get_guard_config()`.

### 6. Update Endpoint Registration
Change endpoint paths from `/v1/` to `/v2/`:
```python
# Register embedding endpoint
if "EmbeddingWorker" in globals():
    routes["/v2/embeddings"] = [Runtime(EmbeddingWorker)]

# Register reranker endpoints with separate workers for each format
# ScoreWorker: /v2/score -> vLLM format {data: [...]}}
# RerankWorker: /v2/rerank -> Cohere format {results: [...]}}
if "ScoreWorker" in globals():
    routes["/v2/score"] = [Runtime(ScoreWorker)]
if "RerankWorker" in globals():
    routes["/v2/rerank"] = [Runtime(RerankWorker)]

# Register guard endpoint
if "GuardWorker" in globals():
    routes["/v2/moderate"] = [Runtime(GuardWorker)]
```

## Key Benefits

1. **Eliminates regeneration need for config changes**: Same generated script works for different model combinations
2. **True dynamic configuration**: Workers fetch current configuration at initialization time
3. **Leverages existing cmw-mosec infrastructure**: Uses same ModelRegistry and settings loading as parent process
4. **Backward compatibility**: v1 endpoints remain unchanged and functional
5. **Clean upgrade path**: Users can migrate to v2 endpoints when ready without breaking existing functionality

## Validation Approach
1. Verify modified `_generate_server_script()` produces valid dynamic script
2. Test that generated script starts successfully
3. Confirm all v2 endpoints respond correctly with different model combinations
4. Validate model-specific behaviors (pooling methods, MRL truncation, etc.)
5. Ensure error handling works for invalid configurations
6. Verify v1 endpoints continue to work as before (backward compatibility)
7. Test that configuration changes are picked up without script regeneration

This approach modifies the existing script generator to produce runtime-configurable workers, achieving the goal of dynamic configuration fetching while maintaining simplicity and backward compatibility.