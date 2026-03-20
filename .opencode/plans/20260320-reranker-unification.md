# Reranker Unification Plan - March 20, 2026

## Executive Summary

Unify reranker architecture by moving instruction/formatting responsibility to client (cmw-rag) while server (cmw-mosec) handles inference and context windows. Match existing embedding pattern. Enable multi-provider support (mosec, vLLM, OpenRouter).

## Problem Statement

**Current State:**
- **Embeddings**: Client formats → Server accepts raw text → Vectors ✅ Correct
- **Rerankers (Qwen3)**: Server formats ChatML → Client sends separate fields ❌ Inconsistent
- **Rerankers (DiTy/BGE)**: Client sends raw query → Server scores ✅ Works (DO NOT TOUCH)

**Gap:** `InfinityReranker` in cmw-rag embeds instruction in query string incorrectly.

## Model Card Reference (Qwen3-Reranker)

```python
# HuggingFace format
system_prompt = "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
user_content = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"
suffix_tokens = "<|im_end|>\n<|im_start|>assistant\n\n\n\n\n"
```

## Architecture

### Server Responsibilities (cmw-mosec)
- Context window enforcement (`max_length` defaults in `config/models.yaml`)
- Tokenization and inference
- Static model parts: `suffix_tokens`, `scoring_tokens`
- `/v1/rerank` endpoint for ALL rerankers (unchanged)

### Client Responsibilities (cmw-rag)
- Format user content per model config
- Handle instruction overrides at call time
- Route to appropriate endpoint based on model type
- Override context window if needed

### What Does NOT Change
- ❌ Cross-encoder handling (DiTy/BGE) - works perfectly
- ❌ Embedding endpoint - untouched
- ❌ Guardian endpoint - untouched
- ❌ `/v1/rerank` endpoint - kept for all models

## Configuration Schema

### Server Config (cmw-mosec `config/models.yaml`)
```yaml
Qwen/Qwen3-Reranker-0.6B:
  type: reranker
  reranker_type: causal_lm
  context_window: 32768
  system_prompt: "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
  suffix_tokens: "<|im_end|>\n<|im_start|>assistant\n\n\n\n\n"  # Tied to tokenizer, server-side only
  scoring_tokens: {true: "yes", false: "no"}
  default_instruction: "Given a web search query, retrieve relevant passages that answer the query"

DiTy/cross-encoder-russian-msmarco:
  type: reranker
  reranker_type: cross_encoder
  context_window: 512  # Native scoring, no formatting
```

### Client Config (cmw-rag `models.yaml`)
```yaml
Qwen/Qwen3-Reranker-0.6B:
  type: reranker
  reranker_type: causal_lm
  system_prompt: "Judge whether the Document meets the requirements..."
  user_template: "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"
  default_instruction: "Given a web search query, retrieve relevant passages that answer the query"
  context_window_override: null
  endpoint: "/v1/rerank"

DiTy/cross-encoder-russian-msmarco:
  type: reranker
  reranker_type: cross_encoder
  endpoint: "/v1/rerank"  # No formatting needed
```

## Implementation Plan (cmw-rag Only)

### Phase 1: Configuration
1. Add `system_prompt`, `user_template`, `default_instruction` to `models.yaml` schema
2. Add `context_window_override` field for request-time overrides
3. Keep `endpoint` field for model-specific routing

### Phase 2: Unified Reranker
```python
class UnifiedReranker:
    """Unified reranker supporting both causal LM and cross-encoder types."""
    
    def rerank(
        self,
        query: str,
        docs: list[str],
        instruction: str | None = None,
        max_length: int | None = None,
    ) -> list[float]:
        config = self.config
        
        if config.reranker_type == "cross_encoder":
            # DiTy/BGE: Pass through unchanged
            return self._post({
                "query": query,
                "docs": docs,
                "max_length": max_length or config.context_window_override,
            })
        
        # Qwen3: Format using config
        return self._post({
            "query": query,
            "docs": docs,
            "instruction": instruction or config.default_instruction,
            "max_length": max_length,
        })
```

### Phase 3: Deprecation
1. Mark `InfinityReranker` as deprecated
2. Update test harness to use `UnifiedReranker`
3. Remove `InfinityReranker` after validation

## Testing Strategy

### Unit Tests
- [ ] Configuration parsing for `system_prompt`, `user_template`, `default_instruction`
- [ ] `UnifiedReranker` formatting logic for Qwen3
- [ ] Passthrough logic for cross-encoders

### Integration Tests
- [ ] DiTy/BGE regression: Verify unchanged behavior
- [ ] Qwen3 formatting: Verify correct message structure
- [ ] Context window override: Verify client can override

### Regression Tests
- [ ] Existing cmw-mosec endpoints continue working
- [ ] Test harness produces same scores as before

## Design Principles

### DRY (Don't Repeat Yourself)
- Single source of truth: `models.yaml` for formatting config
- Reuse embedding pattern for rerankers

### Lean & Minimal
- Changes only in cmw-rag
- No changes to cmw-mosec server
- No breaking changes to existing clients

### Pythonic & Abstract
- `UnifiedReranker` uses strategy pattern based on `reranker_type`
- Configuration-driven behavior, not hardcoded conditionals

### Non-Breaking
- Cross-encoders continue working unchanged
- Existing `/v1/rerank` endpoint preserved
- Backward compatible configuration schema

## Files to Modify (cmw-rag Only)

| File | Changes |
|------|---------|
| `rag_engine/config/models.yaml` | Add `system_prompt`, `user_template`, `default_instruction` |
| `rag_engine/config/schemas.py` | Update `ServerRerankerConfig` schema |
| `rag_engine/retrieval/reranker.py` | Implement `UnifiedReranker` |

## Files to Reference (cmw-mosec - No Changes)

| File | Purpose |
|------|---------|
| `config/models.yaml` | Server-side defaults reference |
| `cmw_mosec/server_manager.py` | Current Qwen3 implementation reference |

## Success Criteria

1. ✅ DiTy/BGE continue working without any changes
2. ✅ Embedding and Guardian endpoints untouched
3. ✅ UnifiedReranker correctly formats Qwen3 requests
4. ✅ Works with mosec, vLLM, OpenRouter using same format
5. ✅ Test harness validates against model card examples
6. ✅ Zero breaking changes to cmw-mosec

## Timeline

| Week | Deliverable |
|------|-------------|
| 1 | Configuration schema updates |
| 2 | UnifiedReranker implementation |
| 3 | Testing and validation |
| 4 | Deprecation and cleanup |

---

**Key Decision:** `suffix_tokens` in server config only (tied to tokenizer). Client only needs `system_prompt` + `user_template` for message content formatting.

**Invariant:** Cross-encoder (DiTy/BGE) behavior unchanged - zero modifications to that code path.