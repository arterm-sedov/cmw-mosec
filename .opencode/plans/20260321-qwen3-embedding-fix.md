# Qwen3 Embedding Fix Session
**Date:** 2026-03-21
**Status:** Investigation Complete, Fix Needed

## Summary
Investigated why Qwen3-Embedding models fail in cmw-mosec while working perfectly with transformers directly. Root cause identified.

## Problem
- Qwen3-Embedding endpoints return "inference internal error"
- Direct transformers testing works perfectly
- Reranker endpoints (Qwen3-Reranker) work correctly
- FRIDA embedding works correctly

## Root Cause
**EmbeddingWorker doesn't set `padding_side='left'` for tokenizer.**

Qwen3-Embedding models are **LLM-based causal models** requiring:
- Last-token pooling (config already has `pooling: last_token`)
- **Left padding** (tokenizer needs `padding_side='left'`)

Current code (line 151):
```python
self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
```

**Missing:** `padding_side='left'`

## Technical Details

### Architecture Comparison

| Model | Type | Pooling | Padding | Works? |
|-------|------|---------|---------|--------|
| FRIDA | Encoder (T5) | CLS | right (default) | ✅ |
| Qwen3-Embedding | Causal LM | last_token | **left REQUIRED** | ❌ |
| DiTy | Cross-encoder | - | - | ✅ |
| Qwen3-Reranker | Causal LM | - | left | ✅ |

### Why Left Padding Matters
- Last-token pooling gets the final meaningful token
- Without left padding, last token is a padding token
- Results in incorrect embeddings or errors

### RerankerWorker Already Fixed
Line 316 in `server_manager.py`:
```python
self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side='left')
```

## Fix Required

**File:** `cmw_mosec/server_manager.py`

**Location:** EmbeddingWorker `__init__` method (around line 151)

**Change:**
```python
# Before
self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)

# After
self.tokenizer = transformers.AutoTokenizer.from_pretrained(
    self.model_name,
    padding_side='left'
)
```

## Model Types in Config (config/models.yaml)

### Encoder-based (default padding)
- FRIDA: `model_class: T5EncoderModel`, `pooling: cls`

### LLM-based (need left padding)
- Qwen3-Embedding: `pooling: last_token` (causal LM)
- All need `padding_side='left'`

## Testing Done
1. Direct transformers test - ✅ Works
2. Last-token pooling test - ✅ Works
3. cmw-mosec embedding endpoint - ❌ Fails
4. cmw-mosec reranker endpoints - ✅ Work
5. FRIDA embedding - ✅ Works

## Next Steps
1. Apply fix to `server_manager.py`
2. Test Qwen3-Embedding endpoint
3. Verify FRIDA still works
4. Update AGENTS.md with lessons learned

## Related Files
- `cmw_mosec/server_manager.py` - EmbeddingWorker class
- `config/models.yaml` - Model configurations
- `.env` - Active model settings

## Configuration Needed
Models with `pooling: last_token` **must** use `padding_side='left'`:
- Qwen/Qwen3-Embedding-0.6B
- Qwen/Qwen3-Embedding-4B
- Qwen/Qwen3-Embedding-8B

## Cross-Reference with cmw-rag
cmw-rag uses OpenRouter for Qwen3-Embedding with instruction formatting.
Client-side formatting: `Instruct: {task}\nQuery: {query}`
Server (cmw-mosec) accepts pre-formatted text, applies pooling.