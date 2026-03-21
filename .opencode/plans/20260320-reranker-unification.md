# Reranker Unification Plan - March 21, 2026

## Executive Summary

Move abstraction to cmw-rag. The client (cmw-rag) owns all non-static formatting and scoring
logic. Inference servers (cmw-mosec, vLLM, OpenRouter) remain agnostic: they load models,
apply static defaults from config, and serve inference. Dynamic instructions are empty
server-side. Test harness acts as a client and stores dynamic parts in its own yamls.

## Working Models (DO NOT BREAK)

- **FRIDA** embedder via cmw-mosec `/v1/embeddings` ✅
- **DiTy** cross-encoder reranker via cmw-mosec `/v1/rerank` ✅
- **Qwen3Guard** via cmw-mosec `/v1/moderate` ✅

## Problem Statement

Qwen3-Reranker was recently added to cmw-mosec with server-side formatting. This is
inconsistent with the embedding pattern (where client formats) and prevents cmw-rag from
using vLLM or OpenRouter for the same model. The abstraction should live in the client.

## Model Card Analysis

### Three Reranker Architectures

Source model cards:
- https://huggingface.co/DiTy/cross-encoder-russian-msmarco
- https://huggingface.co/BAAI/bge-reranker-v2-m3
- https://huggingface.co/BAAI/bge-reranker-v2-gemma
- https://huggingface.co/Qwen/Qwen3-Reranker-0.6B

| Model | Type | Model Class | Input Format | Scoring | Dynamic Parts |
|-------|------|-------------|--------------|---------|---------------|
| DiTy | `cross_encoder` | `AutoModelForSequenceClassification` | `[query, doc]` pairs | `logits.view(-1,)` | None |
| BGE-m3 | `cross_encoder` | `AutoModelForSequenceClassification` | `[query, doc]` pairs | `logits.view(-1,)` | None |
| BGE-Gemma | `llm_reranker` | `AutoModelForCausalLM` | `A: {q}\nB: {d}\n{prompt}` | `logits[:,-1,yes_loc]` | `prompt` (static default) |
| Qwen3 | `llm_reranker` | `AutoModelForCausalLM` | ChatML + `<Instruct>:\n<Query>:\n<Document>:` | `softmax(yes,no)` | `instruction` (dynamic) |

### Key Differences Between LLM Rerankers

**BGE-Gemma** (from model card):
```python
prompt = "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."
# Input: bos_token + "A: {query}" + "\n" + "B: {passage}" + "\n" + prompt
# Score: logits[:, -1, yes_loc]  (single Yes token logit, raw)
```

**Qwen3** (from model card):
```python
prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
user_content = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"
# Input: prefix_tokens + tokenize(user_content) + suffix_tokens
# Score: softmax(yes_logit, no_logit) → probability
```

**These are fundamentally different:** different token layout, different scoring math,
different dynamic parts. The client adapter must handle each.

## Architecture

### Principle: Smart Client, Agnostic Server

```
cmw-rag (smart client)
├── knows model features from models.yaml
├── formats user message content per model card
├── handles instruction overrides per query
├── extracts scores from server responses
└── adapts to server type (mosec, vLLM, OpenRouter)

Server (agnostic inference)
├── loads model at startup
├── applies static parts from server config (prefix, suffix, scoring tokens)
├── accepts pre-formatted content from client
├── handles tokenization, truncation, batching
└── returns raw inference output (scores or logprobs)
```

### Server-Side: What's Static (Baked at Startup)

| Part | DiTy/BGE-m3 | BGE-Gemma | Qwen3 |
|------|-------------|-----------|-------|
| Model class | `SequenceClassification` | `CausalLM` | `CausalLM` |
| Prefix tokens | N/A | `bos_token` | ChatML system + user start |
| Suffix tokens | N/A | `\n` + prompt tokens | ChatML user end + assistant |
| Scoring tokens | N/A | `Yes` token ID | `yes`/`no` token IDs |
| Scoring method | `logits.view(-1,)` | `logits[:,-1,yes_loc]` | `softmax(yes,no)` |
| `max_length` | 512 | 1024 | 32768 |

### Client-Side: What's Dynamic (Per Request)

| Part | DiTy/BGE-m3 | BGE-Gemma | Qwen3 |
|------|-------------|-----------|-------|
| User content | `[query, doc]` pairs (raw) | `A: {q}\nB: {d}` | `<Instruct>:\n<Query>:\n<Document>:` |
| Instruction | N/A | N/A (prompt is static) | Dynamic per request |

### Test Harness

Server-side test harness acts as a proper client:
- Stores test instructions and queries in **separate test yaml** (not in server config)
- Formats content the same way cmw-rag would
- Validates against model card examples

## API Contract

### Cross-Encoders (DiTy, BGE-m3) - UNCHANGED
```json
{"query": "...", "docs": ["...", "..."], "max_length": 512}
→ {"scores": [0.88, 0.001]}
```

### LLM Rerankers (Qwen3, BGE-Gemma) - NEW
```json
{"pairs": ["<Instruct>: ...\n<Query>: ...\n<Document>: ...", "..."], "max_length": 8192}
→ {"scores": [0.95, 0.12]}
```

Client formats each pair. Server applies static prefix/suffix and scores.

## Configuration

### Server Config (cmw-mosec `config/models.yaml`)
```yaml
reranker_models:
  DiTy/cross-encoder-russian-msmarco:
    model_id: DiTy/cross-encoder-russian-msmarco
    reranker_type: cross_encoder  # AutoModelForSequenceClassification
    max_length: 512

  BAAI/bge-reranker-v2-m3:
    model_id: BAAI/bge-reranker-v2-m3
    reranker_type: cross_encoder
    max_length: 8192

  BAAI/bge-reranker-v2-gemma:
    model_id: BAAI/bge-reranker-v2-gemma
    reranker_type: llm_reranker  # AutoModelForCausalLM
    max_length: 1024
    # Static server-side parts from model card
    prompt: "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."
    scoring_tokens: {true: "Yes"}  # Single token, raw logit
    scoring_method: raw_logit  # logits[:, -1, yes_loc]

  Qwen/Qwen3-Reranker-0.6B:
    model_id: Qwen/Qwen3-Reranker-0.6B
    reranker_type: llm_reranker
    max_length: 32768
    # Static server-side parts from model card
    prefix: "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
    suffix: "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    scoring_tokens: {true: "yes", false: "no"}
    scoring_method: softmax  # softmax(yes, no)
```

### Client Config (cmw-rag `models.yaml`)
```yaml
Qwen/Qwen3-Reranker-0.6B:
  type: reranker
  user_template: "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"
  default_instruction: "Given a web search query, retrieve relevant passages that answer the query"
  provider_formats:
    mosec: {}

BAAI/bge-reranker-v2-gemma:
  type: reranker
  user_template: "A: {query}\nB: {doc}"
  # No dynamic instruction - prompt is static server-side
  provider_formats:
    mosec: {}

DiTy/cross-encoder-russian-msmarco:
  type: reranker
  # No user_template - raw [query, doc] pairs
  provider_formats:
    direct:
      batch_size: 16
      device: auto
```

### Test Harness Config (cmw-mosec `tests/test_models.yaml`)
```yaml
# Dynamic parts for testing - NOT in server config
test_instructions:
  qwen3_reranker:
    - "Given a web search query, retrieve relevant passages that answer the query"
    - "Find documents about artificial intelligence"
test_queries:
  - query: "What is the capital of France?"
    docs: ["Paris is the capital...", "London is the capital..."]
```

## Implementation Plan

### Phase 1: cmw-mosec Server Changes

**`config/models.yaml`:**
- Add `prefix`, `suffix`, `scoring_tokens`, `scoring_method` to Qwen3 configs
- Add `prompt`, `scoring_tokens`, `scoring_method` for BGE-Gemma (future)
- Fix suffix: add `<think>\n\n</think>` tags per model card

**`cmw_mosec/server_manager.py`:**
1. Read `prefix`, `suffix`, `scoring_tokens`, `scoring_method` from config
2. Accept `pairs` field for `llm_reranker` type (pre-formatted user content)
3. Remove instruction handling and user content formatting from server
4. Keep: tokenization, prefix/suffix token application, scoring logic
5. Cross-encoder path: UNCHANGED

**`tests/test_models.yaml`:**
- Create separate test config with dynamic instructions and test queries

### Phase 2: cmw-rag Client Changes

**`rag_engine/config/models.yaml`:**
- Add `user_template` to Qwen3 and BGE-Gemma reranker configs

**`rag_engine/config/schemas.py`:**
- Add `user_template: str | None` to `ServerRerankerConfig`

**`rag_engine/retrieval/reranker.py`:**
- Update `InfinityReranker.rerank()`:
  - If `user_template` exists: format pairs client-side, send `{pairs}`
  - If no `user_template`: send `{query, documents}` (cross-encoder, unchanged)

### Phase 3: Testing

- [ ] DiTy regression: unchanged behavior
- [ ] Qwen3: client formats pairs, server applies prefix/suffix, scores match
- [ ] BGE-Gemma: client formats pairs, server applies prompt/scoring (future)
- [ ] Test harness uses test yaml, not server config, for dynamic parts

## Errata in Current Implementation

1. **Suffix bug:** Missing `<think>\n\n</think>` tags per model card
2. **max_length:** Should subtract prefix/suffix token lengths before tokenization
3. **Scoring method:** BGE-Gemma uses raw `Yes` logit, Qwen3 uses `softmax(yes,no)` - these are distinct

## Design Principles

- **Smart client, agnostic server**: Client knows models, server serves inference
- **DRY**: `user_template` in client, static parts in server, test parts in test yaml
- **Lean**: Minimal changes, no overengineering
- **Non-breaking**: DiTy/BGE-m3/FRIDA/Guardian unchanged
- **Pythonic**: Strategy pattern via `reranker_type`, template-based formatting

---

**Key Decision:** Abstraction lives in cmw-rag. Server is agnostic. Dynamic instructions
are empty server-side. Test harness provides dynamic parts from its own yaml.

**Invariant:** FRIDA, DiTy, Qwen3Guard continue working unchanged.