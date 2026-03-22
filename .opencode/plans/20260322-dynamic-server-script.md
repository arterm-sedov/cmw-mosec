# Plan: Dynamic ServerScript from Generated Base

## Objective
Transform the existing generated server script to fetch model configurations at runtime (using same sources as cmw-mosec) instead of baking them in during generation, while serving at `/v2/` endpoints.

## Core Insight
Take the output of `_generate_server_script()` and modify it to:
1. Remove hardcoded configuration constants
2. Replace worker initialization with dynamic config lookup
3. Update endpoint paths from `/v1/` to `/v2/`

## Transformation Plan

### 1. Remove Bake-Time Constants Section
**Delete these blocks** from generated script:
```python
# Embedding section (lines ~157-162)
EMBEDDING_MODEL = "{embedding_model}"
DTYPE = "{embed_dtype}"
POOLING = "{pooling_method}"
MODEL_CLASS = "{model_class}"
DIMENSIONS = {embed_dimensions}
MAX_LENGTH = {embed_max_length}

# Reranker section (lines ~352-357)
RERANKER_MODEL = "{reranker_model}"
MAX_LENGTH = {reranker_max_length}
RERANKER_TYPE = "{reranker_type}"
SCORING_METHOD = {scoring_method_str}
SCORING_TOKENS = {scoring_tokens_str}
INFERENCE_BATCH_SIZE = {inference_batch_size}

# Guard section (lines ~527-530)
GUARD_MODEL = "{guard_model}"
DTYPE = "{settings.dtype}"
MAX_NEW_TOKENS = {guard_max_new_tokens}
MAX_LENGTH = {guard_max_length}
```

### 2. Modify EmbeddingWorker.__init__()
**Replace** the current `__init__` method (lines ~166-194) with:
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

### 3. Modify RerankerWorker.__init__()
**Replace** the current `__init__` method (lines ~363-398) with similar dynamic lookup for reranker config.

### 4. Modify GuardWorker.__init__()
**Replace** the current `__init__` method (lines ~549-561) with similar dynamic lookup for guard config.

### 5. Update Endpoint Registration
**Change** these lines (around 683-696):
```python
# Register embedding endpoint
if "EmbeddingWorker" in globals():
    routes["/v1/embeddings"] = [Runtime(EmbeddingWorker)]

# Register reranker endpoints with separate workers for each format
# ScoreWorker: /v1/score -> vLLM format {data: [...]}}
# RerankWorker: /v1/rerank -> Cohere format {results: [...]}}
if "ScoreWorker" in globals():
    routes["/v1/score"] = [Runtime(ScoreWorker)]
if "RerankWorker" in globals():
    routes["/v1/rerank"] = [Runtime(RerankWorker)]

# Register guard endpoint
if "GuardWorker" in globals():
    routes["/v1/moderate"] = [Runtime(GuardWorker)]
```

**To**:
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

## Key Implementation Notes

### Configuration Source Alignment
The dynamic lookup uses:
- `load_server_settings()` → same as cmw-mosec parent
- `ModelRegistry()` → same source as cmw-mosec parent
- Environment variables as fallback for direct execution

### Error Handling Considerations
Should add try/except blocks around config lookups with meaningful error messages, matching cmw-mosec's existing error handling patterns.

### Performance Impact
- Configuration lookup occurs once per worker initialization (negligible)
- Eliminates script regeneration overhead
- Maintains identical inference performance

### Backward Compatibility
- v1 endpoints remain unchanged (existing functionality preserved)
- v2 endpoints provide new dynamic configuration capability
- Clean upgrade path for users

## Validation Approach
1. Verify generated v2 script starts successfully
2. Test all v2 endpoints respond correctly
3. Confirm model-specific behaviors (pooling methods, MRL truncation, etc.)
4. Validate with different model combinations
5. Ensure error handling works for invalid configurations