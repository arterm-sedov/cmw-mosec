# Reranker Unification Plan - March 21, 2026

## Executive Summary

Move ALL abstraction to cmw-rag. The client adapter owns formatting (including prefix/suffix),
document iteration, instruction handling, and score extraction. Servers (cmw-mosec, vLLM,
OpenRouter) are fully agnostic: load model, accept pre-formatted input, return raw scores.
Test harness acts as a client with its own dynamic config.

## Working Models (DO NOT BREAK)

- **FRIDA** embedder via cmw-mosec `/v1/embeddings` ✅
- **DiTy** cross-encoder reranker via cmw-mosec `/v1/rerank` ✅
- **Qwen3Guard** via cmw-mosec `/v1/moderate` ✅

## Model Card Analysis

Sources:
- https://huggingface.co/DiTy/cross-encoder-russian-msmarco
- https://huggingface.co/BAAI/bge-reranker-v2-m3
- https://huggingface.co/BAAI/bge-reranker-v2-gemma
- https://huggingface.co/Qwen/Qwen3-Reranker-0.6B
- https://docs.vllm.ai/en/v0.10.0/examples/offline_inference/qwen3_reranker.html

### Three Reranker Architectures

| Model | Type | Model Class | Client Formats | Scoring |
|-------|------|-------------|----------------|---------|
| DiTy | `cross_encoder` | `SequenceClassification` | `[query, doc]` pairs | `logits.view(-1,)` |
| BGE-m3 | `cross_encoder` | `SequenceClassification` | `[query, doc]` pairs | `logits.view(-1,)` |
| BGE-Gemma | `llm_reranker` | `CausalLM` | `bos + A:{q}\nB:{d}\n{prompt}` | `logits[:,-1, Yes]` |
| Qwen3 | `llm_reranker` | `CausalLM` | `prefix + user_content + suffix` | `softmax(yes, no)` |

### Critical Finding: Client Iterates Over Documents

Models do NOT batch internally. In every model card, the **client** loops over documents
and builds query-document pairs:

```python
# DiTy: client builds pairs
reranker_model.predict([[query, doc] for doc in documents])

# Qwen3 Transformers: client builds pairs
pairs = [format_instruction(task, query, doc) for query, doc in zip(queries, documents)]

# Qwen3 vLLM: client builds pairs
pairs = list(zip(queries, documents))
```

### Critical Finding: Prefix/Suffix Are Client-Side

From vLLM official Qwen3 reranker example (score API):
```python
# CLIENT constructs prefix and suffix:
prefix = '<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n'
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
query_template = "{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
document_template = "<Document>: {doc}{suffix}"

# CLIENT formats everything:
queries = [query_template.format(prefix=prefix, instruction=instruction, query=query)
           for query in queries]
documents = [document_template.format(doc=doc, suffix=suffix) for doc in documents]

# Server just scores:
outputs = llm.score(queries, documents)
```

**vLLM score API applies NO template.** Server receives fully formatted strings.
This means cmw-mosec can be equally agnostic.

## Architecture

### Principle: Everything Client-Side, Server Is Agnostic

```
cmw-rag (smart adapter)              Server (agnostic inference)
─────────────────────────             ───────────────────────────
Reads model card config               Loads model at startup
Applies prefix/suffix                  Accepts pre-formatted input
Formats user content                   Tokenizes and truncates
Iterates over documents                Runs inference
Inserts dynamic instruction            Returns raw scores
Extracts/normalizes scores
```

### What Server Does

1. Load model (from config: model_id, dtype, device)
2. Accept pre-formatted pairs from client
3. Tokenize with max_length truncation
4. Run inference
5. Return raw scores

Server does NOT know about:
- Prefix/suffix tokens
- Instructions
- Document formatting
- Score normalization (softmax vs raw)

### What Client Does

1. Read model config (user_template, prefix, suffix, scoring_tokens, scoring_method)
2. For each document: format pair using template with prefix + user content + suffix
3. Send formatted pairs to server
4. Extract scores from response
5. Apply scoring normalization if needed (softmax vs raw logit vs sigmoid)

## API Contract

### Cross-Encoders (DiTy, BGE-m3) - UNCHANGED
```json
{"query": "...", "docs": ["...", "..."], "max_length": 512}
→ {"scores": [0.88, 0.001]}
```
Server handles [query, doc] pairing internally (existing behavior, works, don't touch).

### LLM Rerankers (Qwen3, BGE-Gemma) - NEW
```json
{"pair": "<|im_start|>system\n...<Instruct>: ...\n<Query>: ...\n<Document>: ...<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", "max_length": 8192}
→ {"score": 0.95}
```
**One pair per request.** Client sends a single fully formatted string.
Server tokenizes, runs inference, returns one score.

Client iterates over documents, sends N requests, collects scores.
Mosec dynamic batching handles efficiency transparently (multiple concurrent
requests get batched into a single GPU forward pass).

This matches vLLM's `llm.score(query, document)` pattern where each
query-document is an independent scoring unit.

## Configuration

### Server Config (cmw-mosec `config/models.yaml`)

Server config is minimal - just model loading and inference params:
```yaml
reranker_models:
  DiTy/cross-encoder-russian-msmarco:
    model_id: DiTy/cross-encoder-russian-msmarco
    reranker_type: cross_encoder
    max_length: 512

  BAAI/bge-reranker-v2-m3:
    model_id: BAAI/bge-reranker-v2-m3
    reranker_type: cross_encoder
    max_length: 8192

  BAAI/bge-reranker-v2-gemma:
    model_id: BAAI/bge-reranker-v2-gemma
    reranker_type: llm_reranker
    max_length: 1024
    scoring_tokens: {true: "Yes"}
    scoring_method: raw_logit

  Qwen/Qwen3-Reranker-0.6B:
    model_id: Qwen/Qwen3-Reranker-0.6B
    reranker_type: llm_reranker
    max_length: 32768
    scoring_tokens: {true: "yes", false: "no"}
    scoring_method: softmax
```

No prefix, suffix, user_template, or instruction in server config.
Server is agnostic.

### Client Config (cmw-rag `models.yaml`)

Client config owns ALL formatting knowledge from model cards:
```yaml
Qwen/Qwen3-Reranker-0.6B:
  type: reranker
  reranker_type: llm_reranker
  # All from HuggingFace model card
  prefix: "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
  suffix: "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
  user_template: "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"
  default_instruction: "Given a web search query, retrieve relevant passages that answer the query"
  scoring_method: softmax  # For client-side score extraction if needed
  provider_formats:
    mosec: {}
    vllm: {}

BAAI/bge-reranker-v2-gemma:
  type: reranker
  reranker_type: llm_reranker
  # All from HuggingFace model card
  prefix: ""  # bos_token handled by tokenizer
  suffix: ""
  user_template: "A: {query}\nB: {doc}"
  prompt_suffix: "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."
  scoring_method: raw_logit
  provider_formats:
    mosec: {}

DiTy/cross-encoder-russian-msmarco:
  type: reranker
  reranker_type: cross_encoder
  # No formatting - raw [query, doc] pairs
  provider_formats:
    direct:
      batch_size: 16
      device: auto
    mosec: {}
```

### Test Harness Config (cmw-mosec `tests/test_rerankers.yaml`)

Dynamic parts for testing - NOT in server config:
```yaml
test_cases:
  qwen3:
    instructions:
      - "Given a web search query, retrieve relevant passages that answer the query"
      - "Find technical documentation about machine learning"
    prefix: "<|im_start|>system\nJudge whether..."
    suffix: "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    user_template: "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"
    queries:
      - query: "What is the capital of France?"
        docs: ["Paris is the capital...", "London is the capital..."]
        expected_order: [0, 1]
  dity:
    queries:
      - query: "машина"
        docs: ["Автомобиль для перевозки грузов", "Куриное блюдо"]
        expected_order: [0, 1]
```

## Implementation Plan

### Phase 1: cmw-mosec Server Simplification

**`cmw_mosec/server_manager.py`:**
1. Add `pair` field support for `llm_reranker` type (single string, one query-doc pair)
2. When `pair` is provided: tokenize, run inference, return single score
3. Remove: prefix/suffix construction, instruction handling, user content formatting,
   document iteration loop
4. Keep: tokenization, max_length truncation, scoring logic (from config)
5. Cross-encoder path: UNCHANGED (accepts `{query, docs}`, iterates server-side)
6. Mosec dynamic batching handles concurrent single-pair requests efficiently

**`config/models.yaml`:**
- Remove `default_instruction` from Qwen3 configs (client-side concern)
- Add `scoring_tokens` and `scoring_method` (server needs these for inference)

**`tests/test_rerankers.yaml`:**
- Create separate test config with formatting templates and test data
- Test harness formats like a client

### Phase 2: cmw-rag Client Enhancement

**`rag_engine/config/models.yaml`:**
- Add `prefix`, `suffix`, `user_template`, `default_instruction` to Qwen3 config
- Add `user_template`, `prompt_suffix` to BGE-Gemma config

**`rag_engine/config/schemas.py`:**
- Add fields to `ServerRerankerConfig`: `prefix`, `suffix`, `user_template`,
  `default_instruction`, `prompt_suffix`, `scoring_method`

**`rag_engine/retrieval/reranker.py`:**
```python
class InfinityReranker(HTTPClientMixin):
    def rerank(self, query, candidates, top_k, instruction=None, **kwargs):
        documents = [doc.page_content if hasattr(doc, "page_content") else str(doc)
                     for doc, _ in candidates]

        if self.config.user_template:
            # LLM reranker: format and score each pair individually
            task = instruction or self.config.default_instruction or ""
            prefix = self.config.prefix or ""
            suffix = self.config.suffix or ""
            prompt_suffix = self.config.prompt_suffix or ""

            scores = []
            for doc in documents:
                content = self.config.user_template.format(
                    instruction=task, query=query, doc=doc
                )
                pair = f"{prefix}{content}{prompt_suffix}{suffix}"
                response = self._post({"pair": pair})
                scores.append(response["score"])
        else:
            # Cross-encoder: pass through unchanged (server iterates)
            response = self._post({"query": query, "documents": documents, "top_k": top_k})
            scores = response["scores"]

        # ... existing metadata boost and sort logic
```

Note: For performance, the client loop can be parallelized with `concurrent.futures`
or `asyncio`. Mosec batches concurrent requests into single GPU forward passes
automatically.

### Phase 3: Testing

- [ ] DiTy regression: `{query, docs}` path unchanged
- [ ] Qwen3 via mosec: client formats with prefix/suffix, server returns scores
- [ ] Test harness: formats from test yaml, validates scores
- [ ] Compare scores with model card examples

## Errata in Current cmw-mosec Implementation

1. **Suffix bug:** Missing `<think>\n\n</think>` tags per model card
2. **max_length:** Should be `max_length - len(prefix_tokens) - len(suffix_tokens)` for
   tokenization, but with client-side prefix/suffix this becomes simpler: just `max_length`
3. **Server formats:** Server currently builds `<Instruct>:...\n<Query>:...\n<Document>:...`
   and applies prefix/suffix. All of this moves to client.

## Design Principles

- **Smart client, dumb server**: All model knowledge in client config and adapter
- **DRY**: Model card details in client yaml only, server yaml is minimal
- **Lean**: Server removes formatting code, becomes simpler
- **Non-breaking**: DiTy/BGE-m3/FRIDA/Guardian unchanged
- **Pythonic**: Strategy pattern via `reranker_type` + template formatting
- **TDD**: Test harness as client with its own config

---

**Key Decision:** Prefix, suffix, instruction, user template - ALL client-side.
Server is agnostic inference. Matches vLLM score API pattern exactly.

**Invariant:** FRIDA, DiTy, Qwen3Guard continue working unchanged.

**Supersedes:** `.opencode/plans/20260320-reranker-unification.md`