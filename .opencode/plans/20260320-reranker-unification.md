# Reranker Unification Plan - March 20, 2026

## Executive Summary

Unify reranker architecture: client (cmw-rag) formats user message content per model card,
server (cmw-mosec) applies tokenizer-level parts (prefix/suffix tokens) and handles inference.
Match existing embedding pattern. Enable multi-provider support (mosec, vLLM, OpenRouter).

## Problem Statement

**Current state:**
- **Embeddings**: Client formats → Server accepts raw text → Vectors ✅
- **Rerankers (Qwen3)**: Server constructs user content + applies prefix/suffix ❌ Should be client-side
- **Rerankers (DiTy/BGE)**: Client sends raw query → Server scores ✅ DO NOT TOUCH

**Gap:** Qwen3 formatting should move to client-side to match embedding pattern and enable
vLLM/OpenRouter support.

## Model Card Reference (Qwen3-Reranker)

Source: https://huggingface.co/Qwen/Qwen3-Reranker-0.6B

### Transformers Usage (canonical format)
```python
# Client constructs this (one per query-doc pair):
user_content = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

# Server applies these (tied to tokenizer, static per model):
prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

# Tokenization flow:
#   tokenize(user_content) → prefix_tokens + content_tokens + suffix_tokens
#   pad → inference → logits[:, -1, :] → softmax(yes, no) → score
```

### vLLM Usage (chat messages format)
```python
# Client constructs messages array:
messages = [
    {"role": "system", "content": "Judge whether the Document..."},
    {"role": "user", "content": f"<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}"}
]
# Note: vLLM example uses double newlines (\n\n) between fields
# Transformers example uses single newlines (\n)
# Both work; single newline is canonical per format_instruction()
```

### BUG in current cmw-mosec
Current `server_manager.py:318` has suffix WITHOUT `<think>` tags:
```python
self.suffix = "<|im_end|>\\n<|im_start|>assistant\\n\\n\\n\\n\\n"
```
Model card specifies:
```python
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
```
**Must fix suffix to match model card.**

## Architecture

### Server Responsibilities (cmw-mosec) - Changes Required
- Accept `pairs` field: list of pre-formatted user content strings (one per query-doc pair)
- Apply `prefix_tokens + content_tokens + suffix_tokens` during tokenization
- YES/NO token scoring logic (unchanged)
- Context window enforcement via `max_length` (unchanged)
- Suffix/prefix from config (move from hardcoded to `config/models.yaml`)
- **Cross-encoders (DiTy/BGE)**: Keep existing `{query, docs}` API unchanged

### Client Responsibilities (cmw-rag)
- Format each query-doc pair into user content using `user_template` from config
- Send list of formatted pairs to server
- Handle instruction overrides at call time

### What Does NOT Change
- DiTy/BGE cross-encoder handling
- Embedding endpoint (`/v1/embeddings`)
- Guardian endpoint (`/v1/moderate`)
- `/v1/rerank` endpoint path (kept for all rerankers)

## API Contract

### Qwen3/Causal LM Rerankers (NEW)
```json
{
  "pairs": [
    "<Instruct>: Given a web search query...\n<Query>: What is...\n<Document>: Paris is...",
    "<Instruct>: Given a web search query...\n<Query>: What is...\n<Document>: London is..."
  ],
  "max_length": 8192
}

// Response (unchanged)
{"scores": [0.95, 0.12]}
```

### DiTy/BGE Cross-Encoders (UNCHANGED)
```json
{
  "query": "What is the capital of France?",
  "docs": ["Paris is the capital...", "France is a country..."],
  "max_length": 512
}

// Response (unchanged)
{"scores": [0.95, 0.12]}
```

## Configuration Schema

### Server Config (cmw-mosec `config/models.yaml`)

Current fields kept: `model_id`, `reranker_type`, `dtype`, `batch_size`, `memory_gb`, `workers`, `max_length`.

Fields to add for causal_lm models:
```yaml
reranker_models:
  Qwen/Qwen3-Reranker-0.6B:
    model_id: Qwen/Qwen3-Reranker-0.6B
    reranker_type: causal_lm
    dtype: float16
    batch_size: 32
    memory_gb: 2.0
    workers: 1
    max_length: 32768
    default_instruction: "Given a web search query, retrieve relevant passages that answer the query"
    # NEW: Server-side static parts (tied to tokenizer, from model card)
    prefix: "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
    suffix: "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    scoring_tokens:
      true: "yes"
      false: "no"

  # Cross-encoders: NO CHANGES
  DiTy/cross-encoder-russian-msmarco:
    model_id: DiTy/cross-encoder-russian-msmarco
    reranker_type: cross_encoder
    dtype: float16
    batch_size: 32
    memory_gb: 2.0
    workers: 1
    max_length: 512
```

### Client Config (cmw-rag `models.yaml`)

Fields to add for Qwen3:
```yaml
Qwen/Qwen3-Reranker-0.6B:
  type: reranker
  dimensions: 1
  description: "Lightweight Qwen3 reranker"
  default_instruction: "Given a web search query, retrieve relevant passages that answer the query"
  # NEW: Client-side formatting template
  user_template: "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"
  provider_formats:
    direct:
      supported: false
    mosec: {}

# Cross-encoders: NO CHANGES
DiTy/cross-encoder-russian-msmarco:
  type: reranker
  dimensions: 1
  description: "Russian-optimized cross-encoder reranker"
  provider_formats:
    direct:
      batch_size: 16
      device: auto
```

## Implementation Plan

### Phase 1: cmw-mosec Changes

**File: `config/models.yaml`**
- Add `prefix`, `suffix`, `scoring_tokens` to Qwen3 reranker configs
- Fix suffix to match model card (add `<think>\n\n</think>` tags)

**File: `cmw_mosec/server_manager.py`**

1. Read `prefix`, `suffix`, `scoring_tokens` from config at script generation time
2. Update `RerankerWorker.__init__()`: Use config values instead of hardcoded strings
3. Update `RerankerWorker.forward()`:
```python
def forward(self, data: dict[str, Any]) -> dict[str, Any]:
    effective_max_length = data.get("max_length") or self.max_length

    if self.is_qwen3:
        # NEW: Accept pre-formatted pairs from client
        pairs = data.get("pairs", [])

        # Tokenize user content, apply prefix/suffix tokens
        inputs = self.tokenizer(
            pairs, padding=False, truncation='longest_first',
            return_attention_mask=False,
            max_length=effective_max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.prefix_tokens + ele + self.suffix_tokens
        inputs = self.tokenizer.pad(
            inputs, padding=True, return_tensors="pt",
            max_length=effective_max_length
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Scoring logic (unchanged)
        with torch.no_grad():
            batch_scores = self.model(**inputs).logits[:, -1, :]
            true_vector = batch_scores[:, self.token_true_id]
            false_vector = batch_scores[:, self.token_false_id]
            batch_scores = torch.stack([false_vector, true_vector], dim=1)
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
            scores = batch_scores[:, 1].exp().tolist()
        return {"scores": scores}
    else:
        # Cross-encoder: UNCHANGED
        query = data["query"]
        docs = data.get("docs") or data.get("documents")
        # ... existing logic unchanged
```

4. Remove: instruction handling (lines 347-354), user content formatting (lines 356-360)

**Also fix:** `max_length` calculation. Current code uses `max_length=effective_max_length`
for tokenization but the model card uses `max_length - len(prefix_tokens) - len(suffix_tokens)`.
This is already correct in the proposed code above.

### Phase 2: cmw-rag Changes

**File: `rag_engine/config/models.yaml`**
- Add `user_template` field to Qwen3 reranker configs

**File: `rag_engine/config/schemas.py`**
- Add `user_template: str | None = None` to `ServerRerankerConfig`

**File: `rag_engine/retrieval/reranker.py`**
- Update `InfinityReranker.rerank()`:

```python
def rerank(self, query, candidates, top_k, metadata_boost_weights=None, instruction=None):
    documents = [
        doc.page_content if hasattr(doc, "page_content") else str(doc)
        for doc, _ in candidates
    ]

    if self.config.user_template:
        # Qwen3: Format each pair client-side
        task = instruction or self.config.default_instruction or ""
        pairs = [
            self.config.user_template.format(
                instruction=task, query=query, doc=doc
            )
            for doc in documents
        ]
        response = self._post({"pairs": pairs})
    else:
        # DiTy/BGE: Pass through unchanged
        response = self._post({"query": query, "documents": documents, "top_k": top_k})

    scores = response["scores"]
    # ... existing metadata boost and sort logic unchanged
```

### Phase 3: Testing

**cmw-mosec tests:**
- [ ] Qwen3 reranker accepts `pairs` field and returns scores
- [ ] Suffix matches model card format (includes `<think>` tags)
- [ ] prefix/suffix from config, not hardcoded
- [ ] DiTy/BGE regression: unchanged behavior with `{query, docs}`
- [ ] max_length calculation accounts for prefix/suffix token overhead

**cmw-rag tests:**
- [ ] `user_template.format()` produces correct output per model card
- [ ] `InfinityReranker` sends `pairs` for Qwen3
- [ ] `InfinityReranker` sends `{query, documents}` for DiTy/BGE (unchanged)
- [ ] Instruction override works at call time

**Integration tests:**
- [ ] cmw-rag → cmw-mosec Qwen3: end-to-end scoring matches model card examples
- [ ] cmw-rag → cmw-mosec DiTy: unchanged behavior

## Files to Modify

### cmw-mosec
| File | Changes |
|------|---------|
| `cmw_mosec/server_manager.py` | Accept `pairs`, remove formatting, read prefix/suffix from config, fix suffix |
| `config/models.yaml` | Add `prefix`, `suffix`, `scoring_tokens` to Qwen3 configs |
| `tests/` | Update tests for new API |

### cmw-rag
| File | Changes |
|------|---------|
| `rag_engine/config/models.yaml` | Add `user_template` to Qwen3 reranker configs |
| `rag_engine/config/schemas.py` | Add `user_template` field to `ServerRerankerConfig` |
| `rag_engine/retrieval/reranker.py` | Format pairs client-side for Qwen3 |

## Design Principles

- **DRY**: `user_template` in client config, `prefix`/`suffix` in server config
- **Lean**: Remove server-side formatting, add simple template format on client
- **Non-Breaking**: DiTy/BGE path completely unchanged
- **Pythonic**: Template-based formatting, strategy pattern via `reranker_type`

## Errata Found During Validation

1. **Suffix bug**: Current cmw-mosec suffix is missing `<think>\n\n</think>` tags from model card
2. **max_length**: Current tokenization uses `max_length=effective_max_length` but should be
   `max_length=effective_max_length - len(prefix_tokens) - len(suffix_tokens)` per model card
3. **Newlines**: Transformers example uses `\n`, vLLM example uses `\n\n` between fields.
   Canonical is single `\n` per `format_instruction()` function in model card

---

**Key Decision:** Client formats `pairs` list (user content per doc). Server applies
`prefix_tokens + content + suffix_tokens` and scores.

**Invariant:** DiTy/BGE cross-encoder path unchanged.