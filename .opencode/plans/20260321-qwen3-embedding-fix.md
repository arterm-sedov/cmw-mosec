# Qwen3 Embedding Fix Session
**Date:** 2026-03-21
**Status:** ✅ COMPLETE

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

## Fix Applied
**Commit:** `0aabd76`

Changed `cmw_mosec/server_manager.py` line 148-160:
```python
# LLM-based embedders (Qwen3) need left padding for last_token pooling
# Encoder-based (FRIDA) use default right padding
if self.pooling == "last_token":
    self.tokenizer = transformers.AutoTokenizer.from_pretrained(
        self.model_name,
        padding_side='left'
    )
else:
    self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
```

## Testing Results
- Qwen/Qwen3-Embedding-0.6B ✅ Works (dim 1024)
- Qwen/Qwen3-Embedding-4B ✅ Works (dim 2560)
- Both tested with instruction formatting

## Root Cause Analysis
The issue was in the `EmbeddingWorker.__init__()` method. Qwen3-Embedding models are based on a **causal language model architecture** (decoder-only), which means:
- They need to use the **last token** for pooling (not CLS or mean pooling)
- They require **left padding** so the last token position is meaningful
- Without left padding, the "last token" is just padding, causing incorrect embeddings

This is similar to how `RerankerWorker` already handles LLM-based rerankers (Qwen3-Reranker uses left padding via line 316).

## Lessons Learned
1. **Different embedding architectures need different tokenization:**
   - Encoder-based (FRIDA/T5): right padding + CLS pooling
   - LLM-based (Qwen3): left padding + last_token pooling

2. **Check HuggingFace model card for pooling requirements**
   - Qwen3-Embedding docs explicitly mention `padding_side='left'`

3. **Model type determines tokenization, not just pooling method**

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