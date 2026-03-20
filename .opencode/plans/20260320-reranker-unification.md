# Reranker Unification Plan - March 20, 2026

## Executive Summary

Unify reranker architecture: client (cmw-rag) formats user content, server (cmw-mosec) applies model-specific tokens and handles inference. Match existing embedding pattern. Enable multi-provider support (mosec, vLLM, OpenRouter).

## Problem Statement

**Current State:**
- **Embeddings**: Client formats → Server accepts raw text → Vectors ✅ Correct
- **Rerankers (Qwen3)**: Server constructs user content + applies prefix/suffix ❌ Should be client-side
- **Rerankers (DiTy/BGE)**: Client sends raw query → Server scores ✅ Works (DO NOT TOUCH)

**Gap:** Qwen3 formatting should move to client-side to match embedding pattern and enable vLLM/OpenRouter support.

## Model Card Reference (Qwen3-Reranker)

```python
# HuggingFace format - client constructs this
user_content = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

# Server applies (tied to tokenizer)
prefix = "<|im_start|>system\nJudge whether the Document...\n<|im_end|>\n<|im_start|>user\n"
suffix = "<|im_end|>\n<|im_start|>assistant\n\n\n\n\n"
```

## Architecture

### Server Responsibilities (cmw-mosec) - Changes Required
- Accept `user_content` field from client (for Qwen3/causal LM rerankers)
- Apply `prefix_tokens + user_content + suffix_tokens` during tokenization
- Keep YES/NO token scoring logic
- Context window enforcement
- **Cross-encoders (DiTy/BGE)**: Keep unchanged, accept `{query, docs}`

### Client Responsibilities (cmw-rag)
- Format `user_content` per model config
- Send to server with appropriate fields:
  - Qwen3: `{user_content, max_length}`
  - DiTy/BGE: `{query, docs, max_length}`
- Handle instruction overrides at call time

### What Does NOT Change
- ❌ DiTy/BGE handling - works perfectly, no changes
- ❌ Embedding endpoint - untouched
- ❌ Guardian endpoint - untouched

## API Contract

### Qwen3/Causal LM Rerankers
```json
// Request
{
  "user_content": "<Instruct>: Given a web search query...\n<Query>: What is...\n<Document>: Paris is...",
  "max_length": 8192  // Optional override
}

// Response
{
  "scores": [0.95, 0.32, 0.12]
}
```

### DiTy/BGE Cross-Encoders (Unchanged)
```json
// Request
{
  "query": "What is the capital of France?",
  "docs": ["Paris is the capital...", "France is a country..."],
  "max_length": 512  // Optional override
}

// Response
{
  "scores": [0.95, 0.12]
}
```

## Configuration Schema

### Server Config (cmw-mosec `config/models.yaml`)
```yaml
Qwen/Qwen3-Reranker-0.6B:
  type: reranker
  reranker_type: causal_lm
  context_window: 32768
  # Server-side only (tied to tokenizer)
  system_prompt: "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
  prefix_tokens: "<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n"
  suffix_tokens: "<|im_end|>\n<|im_start|>assistant\n\n\n\n\n"
  scoring_tokens: {true: "yes", false: "no"}
  default_instruction: "Given a web search query, retrieve relevant passages that answer the query"

DiTy/cross-encoder-russian-msmarco:
  type: reranker
  reranker_type: cross_encoder
  context_window: 512
  # Native scoring, no prefix/suffix needed
```

### Client Config (cmw-rag `models.yaml`)
```yaml
Qwen/Qwen3-Reranker-0.6B:
  type: reranker
  reranker_type: causal_lm
  # Client-side formatting
  user_template: "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"
  default_instruction: "Given a web search query, retrieve relevant passages that answer the query"
  context_window: 32768
  endpoint: "/v1/rerank"

DiTy/cross-encoder-russian-msmarco:
  type: reranker
  reranker_type: cross_encoder
  context_window: 512
  endpoint: "/v1/rerank"
  # No formatting needed
```

## Implementation Plan

### Phase 1: cmw-mosec Changes

**File: `cmw_mosec/server_manager.py`**

1. **Update RerankerWorker API:**
```python
def forward(self, data: dict[str, Any]) -> dict[str, Any]:
    if self.is_qwen3:
        # NEW: Accept pre-formatted user_content from client
        user_content = data.get("user_content")
        max_length = data.get("max_length") or self.max_length

        # Apply prefix/suffix tokens
        inputs = self.tokenizer(
            [user_content], padding=False, truncation='longest_first',
            return_attention_mask=False, max_length=max_length
        )
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.prefix_tokens + ele + self.suffix_tokens
        # ... rest of scoring logic
    else:
        # Cross-encoder: UNCHANGED
        query = data["query"]
        docs = data.get("docs") or data.get("documents")
        # ... existing logic
```

2. **Remove Qwen3-specific formatting from server:**
   - Remove lines 356-360 (instruction handling and formatting)
   - Keep prefix/suffix tokens and scoring logic

### Phase 2: cmw-rag Changes

**File: `rag_engine/retrieval/reranker.py`**

```python
class UnifiedReranker:
    def rerank(self, query: str, docs: list[str], instruction: str | None = None) -> list[float]:
        config = self.config

        if config.reranker_type == "cross_encoder":
            # DiTy/BGE: Pass through unchanged
            return self._post({
                "query": query,
                "docs": docs,
            })

        # Qwen3: Format user_content from config
        user_content = config.user_template.format(
            instruction=instruction or config.default_instruction,
            query=query,
            doc=doc  # Per-document
        )
        return self._post({
            "user_content": user_content,
        })
```

### Phase 3: Testing

1. **Unit tests:** format_user_content() produces correct output
2. **Integration tests:** cmw-rag → cmw-mosec Qwen3 produces same scores
3. **Regression tests:** DiTy/BGE unchanged behavior

## Files to Modify

### cmw-mosec
| File | Changes |
|------|---------|
| `cmw_mosec/server_manager.py` | Accept `user_content`, remove client formatting logic |
| `config/models.yaml` | Add `default_instruction` for documentation |

### cmw-rag
| File | Changes |
|------|---------|
| `rag_engine/config/models.yaml` | Add `user_template`, `default_instruction` |
| `rag_engine/config/schemas.py` | Update schema |
| `rag_engine/retrieval/reranker.py` | Implement client-side formatting |

## Design Principles

### DRY (Don't Repeat Yourself)
- `user_template` in client config (single source for formatting)
- `prefix_tokens`/`suffix_tokens` in server config (single source for token-level)

### Lean & Minimal
- Server removes Qwen3-specific formatting code
- Client adds simple template formatting
- DiTy/BGE code path unchanged

### Pythonic & Abstract
- Reranker types: `causal_lm` vs `cross_encoder`
- Template-based formatting, not hardcoded conditionals

### Non-Breaking
- DiTy/BGE continue working unchanged
- New `user_content` field added for Qwen3
- Old `{query, docs, instruction}` API removed for Qwen3 (breaking, but acceptable since endpoint created days ago)

## Success Criteria

1. ✅ DiTy/BGE unchanged behavior
2. ✅ Client formats user_content per model config
3. ✅ Server applies prefix/suffix tokens for Qwen3
4. ✅ Works with mosec (user_content field)
5. ✅ Works with vLLM/OpenRouter (messages array)
6. ✅ Test harness validates against model card examples

## timeline

| Week | Deliverable |
|------|-------------|
| 1 | cmw-mosec: Accept `user_content`, remove Qwen3 formatting |
| 2 | cmw-rag: Implement `UnifiedReranker`, add config |
| 3 | Integration testing, regression validation |
| 4 | Documentation, cleanup |

---

**Key Decision:** Qwen3 formatting moves to client. Server only applies `prefix_tokens + user_content + suffix_tokens` and scores.

**Invariant:** DiTy/BGE cross-encoder path unchanged.