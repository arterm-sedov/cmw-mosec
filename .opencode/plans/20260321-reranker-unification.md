# Reranker Unification Plan - March 21, 2026

## Status: ✅ COMPLETE

All implementation complete. See "Implementation Complete" section for details.

## Executive Summary

Move ALL abstraction to cmw-rag. The client adapter owns formatting (including prefix/suffix),
instruction handling, and score extraction. Servers (cmw-mosec, vLLM) are agnostic: load model,
accept pre-formatted query + documents batch, return scores. Unified API contract across all
providers. Test harness acts as a client with its own dynamic config.

### Key Insight: cmw-mosec Already Matches vLLM Score API

Current cmw-mosec `/v1/rerank` accepts `{query, documents}` and returns `{scores: [...]}`.
This is essentially vLLM's `/score` behavior (raw scores, original order).
The only server-side change needed: stop constructing prefix/suffix/instruction for Qwen3
and just tokenize the pre-formatted strings received from the client.

**Comparison:**
| Aspect | cmw-mosec `/v1/rerank` | vLLM `/score` | vLLM `/rerank` |
|--------|----------------------|---------------|----------------|
| Request | `{query, docs}` | `{queries, documents}` | `{query, documents}` |
| Response | `{scores: [...]}` | `{data: [{score},...]}` | `{results: [{relevance_score, index},...]}` |
| Sorting | No | No | Yes (by relevance) |
| Behavior | Raw scores | Raw scores | Sorted results |

## Working Models (DO NOT BREAK)

- **FRIDA** embedder via cmw-mosec `/v1/embeddings` ✅
- **DiTy** cross-encoder reranker via cmw-mosec `/v1/rerank` ✅
- **Qwen3Guard** via cmw-mosec `/v1/moderate` ✅

## Research Findings

### Batching Across Providers

| Server | HTTP API | Accepts Batch? | Contract |
|--------|----------|----------------|----------|
| **vLLM `/score`** | `{queries:[q1,q2], documents:[d1,d2]}` | YES (zip pairs) | 1 request, N scores |
| **vLLM `/rerank`** | `{query:"q", documents:[d1,d2]}` | YES (1 query, N docs) | 1 request, N scores |
| **cmw-mosec `/v1/rerank`** | `{query:"q", docs:[d1,d2]}` | YES (1 query, N docs) | 1 request, N scores |
| **OpenAI `/v1/embeddings`** | `{input:["t1","t2"]}` | YES (list) | 1 request, N embeddings |

All providers accept batched input in a single HTTP request. Client sends one request with
all documents, server returns all scores. No per-document HTTP loop needed.

### Mosec Batching Status

Current cmw-mosec uses `max_batch_size=1` (mosec default). The "batching" is application-level:
one HTTP request with `{query, docs:[...]}`, and `forward()` loops over docs internally.
This is NOT mosec's dynamic batching - it's a loop inside a single request handler.
Same pattern works for both cross-encoders and LLM rerankers.

### vLLM Qwen3 Score Pattern (from official docs)

```python
# Client formats query WITH prefix, documents WITH suffix:
query_template = "{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
document_template = "<Document>: {doc}{suffix}"

queries = [query_template.format(prefix=prefix, instruction=instruction, query=query)
           for query in queries]
documents = [document_template.format(doc=doc, suffix=suffix) for doc in documents]

# Server scores all pairs in one call:
outputs = llm.score(queries, documents)  # zip-style pairing
```

Prefix goes into query, suffix goes into document. This split works for both vLLM and mosec.

### Embedding Pattern (for reference)

cmw-rag already sends batched embeddings:
```python
# Single query
resp = requests.post(endpoint, json={"input": text, "model": model})
# Batch documents
resp = requests.post(endpoint, json={"input": texts, "model": model})  # list
```

## Model Card Analysis

Sources:
- https://huggingface.co/DiTy/cross-encoder-russian-msmarco
- https://huggingface.co/BAAI/bge-reranker-v2-m3
- https://huggingface.co/BAAI/bge-reranker-v2-gemma
- https://huggingface.co/Qwen/Qwen3-Reranker-0.6B
- https://docs.vllm.ai/en/v0.10.0/examples/offline_inference/qwen3_reranker.html

### Three Reranker Architectures

| Model | Type | Scoring | Dynamic Parts |
|-------|------|---------|---------------|
| DiTy/BGE-m3 | `cross_encoder` | `logits.view(-1,)` | None |
| BGE-Gemma | `llm_reranker` | `logits[:,-1, Yes]` | `prompt` (static default) |
| Qwen3 | `llm_reranker` | `softmax(yes, no)` | `instruction` (dynamic) |

## Two-Stage Endpoint Alignment

### Stage 1: `/v1/score` Endpoint (Primary for Rerankers)

Current cmw-mosec `/v1/rerank` behavior closely matches vLLM's `/v1/score`:
- Returns raw scores (not sorted)
- Original document order preserved
- Simple response format

**Action:**
1. cmw-mosec: Add `/v1/score` endpoint alias (or deprecate `/v1/rerank` in favor of `/v1/score`)
2. cmw-mosec: For llm_rerankers, accept pre-formatted `{query, documents}` (client applies prefix/suffix)
3. cmw-rag: Use `/v1/score` endpoint for rerankers (matches vLLM contract)

### Stage 2: `/v1/rerank` Endpoint (Compatibility Layer)

vLLM's `/v1/rerank` returns sorted results in Cohere/Jina format:
- Results sorted by relevance score (descending)
- Each result has `{index, document: {text}, relevance_score}`

**Action:**
1. cmw-mosec: Update `/v1/rerank` to return vLLM/Cohere/Jina compatible response
2. Optional: Add `top_n` parameter to limit results

### Unified Core Contract (for `/v1/score`)

```json
// Request: Client sends pre-formatted query and documents
{
  "query": "<formatted query string>",
  "documents": ["<formatted doc1>", "<formatted doc2>", "..."],
  "max_length": 8192  // optional override
}

// Response: Raw scores in same order as input
{
  "scores": [0.95, 0.12, 0.03]
}
```

**One request, N documents, N scores, original order.**
Matches vLLM `/v1/score` exactly (same request/response shape).

### vLLM `/v1/rerank` Contract (for Stage 2)

```json
// Request (same as /v1/score)
{
  "query": "What is the capital of France?",
  "documents": ["The capital of Brazil is Brasilia.", "The capital of France is Paris."],
  "top_n": 2  // optional, defaults to len(documents)
}

// Response: Results sorted by relevance (descending)
{
  "id": "rerank-<uuid>",
  "model": "BAAI/bge-reranker-base",
  "usage": {"total_tokens": 56},
  "results": [
    {
      "index": 1,  // original position
      "document": {"text": "The capital of France is Paris."},
      "relevance_score": 0.9985
    },
    {
      "index": 0,
      "document": {"text": "The capital of Brazil is Brasilia."},
      "relevance_score": 0.0005
    }
  ]
}
```

**Key differences from `/v1/score`:**
- Results sorted by `relevance_score` (descending)
- Each result includes original `index` and `document.text`
- Optional `top_n` parameter to limit results

### What "formatted" Means Per Model

**DiTy/BGE-m3 (cross-encoder):**
```python
query = "raw query text"  # No formatting
documents = ["raw doc1", "raw doc2"]  # No formatting
```

**BGE-Gemma (llm_reranker):**
```python
query = "A: {query}"  # Client formats
documents = ["B: {doc}\n{prompt}", ...]  # Client adds prompt suffix
```

**Qwen3 (llm_reranker):**
```python
query = "{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"  # Client adds prefix + instruction
documents = ["<Document>: {doc}{suffix}", ...]  # Client adds suffix
```

## Architecture

### Smart Client, Agnostic Server

```
cmw-rag (smart adapter)              Server (agnostic inference)
─────────────────────────             ───────────────────────────
Reads model config                     Loads model at startup
Applies prefix to query                Accepts {query, documents}
Applies suffix to documents            Tokenizes and truncates
Inserts dynamic instruction            Runs inference (pairs query with each doc)
Sends one request with all docs        Returns scores array
```

### What Server Does

1. Load model (from config: model_id, dtype, device)
2. Receive `{query, documents, max_length}` where:
   - `query`: single pre-formatted string (client applied prefix)
   - `documents`: list of pre-formatted strings (client applied suffix)
3. For EACH document:
   - Pair: `formatted_pair = query + document`
   - Tokenize, truncate, run inference
4. Return `{scores: [...]}` (one score per document, original order)

**Server pairing behavior:**
- Cross-encoder: `model.predict([[query, doc] for doc in documents])`
- LLM reranker: `model(formatted_pairs)` where formatted_pairs = query + each doc

Server does NOT know about prefix/suffix/instructions. It receives pre-formatted strings.

### What Client Does

1. Read model config (user_template, prefix, suffix, default_instruction)
2. Format query string (apply prefix if needed)
3. Format each document string (apply suffix if needed)
4. Send single request with all documents
5. Receive scores

## Configuration

### Server Config (cmw-mosec `config/models.yaml`)

Minimal - just model loading and inference params:
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

No prefix, suffix, user_template, or instruction. Server is agnostic.

### Client Config (cmw-rag `models.yaml`)

Client owns ALL formatting knowledge:
```yaml
Qwen/Qwen3-Reranker-0.6B:
  type: reranker
  reranker_type: llm_reranker
  # From HuggingFace model card - client formats query and documents
  query_template: "{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
  doc_template: "<Document>: {doc}{suffix}"
  prefix: "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
  suffix: "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
  default_instruction: "Given a web search query, retrieve relevant passages that answer the query"
  provider_formats:
    mosec: {}
    vllm: {}

BAAI/bge-reranker-v2-gemma:
  type: reranker
  reranker_type: llm_reranker
  query_template: "A: {query}"
  doc_template: "B: {doc}\n{prompt}"
  prompt: "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."
  provider_formats:
    mosec: {}

DiTy/cross-encoder-russian-msmarco:
  type: reranker
  reranker_type: cross_encoder
  # No formatting - raw query and documents
  provider_formats:
    direct: {batch_size: 16, device: auto}
    mosec: {}
```

### Test Harness Config (cmw-mosec `tests/test_rerankers.yaml`)

Dynamic parts for testing - NOT in server config:
```yaml
test_cases:
  qwen3:
    query_template: "{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
    doc_template: "<Document>: {doc}{suffix}"
    prefix: "<|im_start|>system\nJudge whether..."
    suffix: "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    instructions:
      - "Given a web search query, retrieve relevant passages that answer the query"
    queries:
      - query: "What is the capital of France?"
        docs: ["Paris is the capital...", "London is the capital..."]
  dity:
    queries:
      - query: "машина"
        docs: ["Автомобиль для перевозки грузов", "Куриное блюдо"]
```

## Implementation Plan

### Non-Breaking Strategy

**Key principle:** Existing `/v1/rerank` endpoint MUST continue working as-is for existing clients (DiTy, BGE-m3).

- `/v1/rerank`: Keep current response format `{scores: [...]}` - NO CHANGES to response schema
- `/v1/score`: NEW endpoint with same behavior as `/v1/rerank` (alias) - vLLM-compatible naming
- `/v1/rerank` Stage 2 enhancement: Optional `return_documents` param to enable vLLM/Cohere format

### Stage 1: Unified `/v1/score` Endpoint

**cmw-mosec changes:**
1. Add `/v1/score` endpoint (alias to `/v1/rerank` worker)
2. Both endpoints accept `{query, documents, max_length}`
3. Both return `{scores: [...]}`
4. For `llm_reranker`: accept pre-formatted `{query, documents}`
5. Remove: prefix/suffix construction, instruction handling from worker code
6. Keep: tokenization, max_length truncation, scoring logic (from config)

**cmw-rag changes:**
1. Update `InfinityReranker` to use `/v1/score` endpoint
2. Format query and documents client-side (apply prefix, suffix, instruction)
3. Send pre-formatted strings, receive scores

**Config changes:**
- cmw-mosec `config/models.yaml`: Remove `default_instruction` (client-side)
- cmw-rag `models.yaml`: Add `query_template`, `doc_template`, `prefix`, `suffix`, `default_instruction`

### Stage 2: Optional vLLM/Cohere Format (Non-Breaking)

**cmw-mosec changes:**
1. Add optional `return_documents` param to `/v1/rerank`
2. If `return_documents=true`: return vLLM/Cohere format `{results: [{index, document, relevance_score}]}`
3. If `return_documents=false` or omitted: return current format `{scores: [...]}` (backward compatible)

**Backward compatibility preserved:**
- Existing clients (DiTy, BGE-m3) continue using `/v1/rerank` with `{scores: [...]}` response
- New clients can opt into vLLM/Cohere format with `return_documents=true`

### Test-Driven Development

**Test files:**
```
tests/
├── test_rerankers.py          # Unit tests for reranker worker
├── test_server_manager.py     # Integration tests for endpoint
└── fixtures/
    └── test_rerankers.yaml    # Test harness config (client-side formatting)
```

**Test cases (TDD):**
1. **DiTy regression** (must pass before any changes):
   - `POST /v1/rerank {"query": "...", "documents": [...]}` → `{scores: [...]}`
   - `POST /v1/score {"query": "...", "documents": [...]}` → `{scores: [...]}`

2. **Qwen3 pre-formatted** (new behavior):
   - `POST /v1/score {"query": "<prefix>...", "documents": ["<doc><suffix>", ...]}` → `{scores: [...]}`
   - Verify scores match model card examples

3. **Config separation**:
   - Server config does NOT contain prefix/suffix/instruction
   - Client config DOES contain prefix/suffix/instruction
   - Test harness config has test values

4. **Non-breaking verification**:
   - FRIDA embedder: `/v1/embeddings` unchanged
   - Qwen3Guard: `/v1/moderate` unchanged
   - DiTy reranker: `/v1/rerank` response unchanged

### Testing Checklist

- [ ] DiTy: `/v1/rerank` and `/v1/score` return same scores as before
- [ ] Qwen3: Client formats, server scores, matches model card
- [ ] Test harness: Uses `tests/fixtures/test_rerankers.yaml` for dynamic parts
- [ ] FRIDA: `/v1/embeddings` unchanged
- [ ] Qwen3Guard: `/v1/moderate` unchanged
- [ ] Backward compat: Old clients work without changes

## Errata in Current cmw-mosec Implementation

1. **Suffix bug (Qwen3):** Current code missing `laissez\n\n\n\n\n\n` in suffix string. Fixed model card shows:
   ```
   suffix = "<|im_end|>\n<|im_start|>assistant\nlaissez\n\n\n\n\n\n"
   ```

2. **Instruction in server code (Qwen3):** Currently constructs `<Instruct>: {instruction}` server-side. Moves to client-side.

3. **max_length calculation (Qwen3):** Currently uses `max_length` directly. Should be:
   - Client: applies prefix/suffix before sending
   - Server: tokenizes received string with `max_length` truncation
   - No need to subtract prefix/suffix lengths (already in string)

## Migration Path (Non-Breaking)

**Before (current Qwen3 in cmw-mosec):**
```python
# Server receives:
{"query": "What is France?", "documents": ["Paris is...", "Lyon is..."], "instruction": "search"}

# Server constructs:
pairs = [f"<Instruct>: search\n<Query>: What is France?\n<Document>: Paris is..." for doc in docs]
# Then applies prefix/suffix, tokenizes, scores
```

**After (new unified approach):**
```python
# Client constructs:
query = f"{prefix}<Instruct>: search\n<Query>: What is France?\n"
documents = [f"<Document>: {doc}{suffix}" for doc in docs]

# Client sends:
{"query": query, "documents": documents}

# Server receives pre-formatted strings, pairs them, tokenizes, scores:
pairs = [query + doc for doc in documents]  # Server just concatenates
```

**DiTy/BGE-m3 (unchanged):**
```python
# Client sends:
{"query": "What is France?", "documents": ["Paris is...", "Lyon is..."]}

# Server (no change):
scores = model.predict([[query, doc] for doc in documents])
```

## Design Principles

- **Unified contract**: Same `{query, documents} → {scores}` for ALL rerankers
- **Smart client, agnostic server**: All model knowledge in client config
- **DRY**: Model card details in client yaml only, server yaml is minimal
- **Lean**: Server removes formatting code, becomes simpler
- **Non-breaking**: DiTy/BGE-m3/FRIDA/Guardian unchanged
- **Matches existing patterns**: Same as embedding API (`{input: [...]} → {data: [...]}`)

---

**Key Decision:** Unified batch contract `{query, documents} → {scores}`.
Client formats query (with prefix) and documents (with suffix).
Server pairs, tokenizes, scores. No per-document HTTP loop.
Same contract for mosec, vLLM, any provider.

**Invariant:** FRIDA, DiTy, Qwen3Guard continue working unchanged.

**Supersedes:** `.opencode/plans/20260320-reranker-unification.md`
## Implementation Complete (March 21, 2026)

### Final Contracts

| Endpoint | Request | Response | Format |
|----------|---------|----------|--------|
| `/v1/score` | `{query, documents}` | `{data: [{index, object, score}, ...]}` | vLLM |
| `/v1/rerank` | `{query, documents, top_n?}` | `{results: [{index, document, relevance_score}, ...]}` | Cohere/Jina |

### Key Decisions

1. **Industry-standard contracts**: Aligned with vLLM/Cohere/Jina APIs
2. **Breaking change approved**: cmw-rag will be refactored
3. **Client-side formatting**: All prefix/suffix/instruction moved to client (cmw-rag)
4. **Server is agnostic**: Accepts pre-formatted strings, returns scores

### Breaking Changes

- `/v1/score` returns vLLM format (not simple `{scores: [...]}`)
- `/v1/rerank` returns Cohere format (not simple `{scores: [...]}`)
- No backward compatibility - cmw-rag refactor required

### Files Changed

| File | Purpose |
|------|---------|
| `cmw_mosec/server_manager.py` | ScoreWorker, RerankerWorker with `_compute_scores()` |
| `cmw_mosec/server_config.py` | Added `reranker_type`, `scoring_method`, `scoring_tokens` |
| `cmw_mosec/cli.py` | Updated for new response formats |
| `config/models.yaml` | Server-side model configs (no formatting) |
| `tests/fixtures/test_rerankers.yaml` | Test harness with formatting templates |
| `tests/test_reranker_endpoints.py` | Endpoint tests |

### What Was NOT Changed

- FRIDA embedder (`/v1/embeddings`) - unchanged
- Qwen3Guard (`/v1/moderate`) - unchanged
- DiTy cross-encoder logic - same `_compute_scores()` method

### Scores Identical

Both `/v1/score` and `/v1/rerank` use the same underlying `_compute_scores()` method:
- `/v1/score`: Raw scores in original order, wrapped in `{data: [...]}`
- `/v1/rerank`: Scores sorted by relevance, wrapped in `{results: [...]}`

### Test Results

```
✅ /v1/score scores == /v1/rerank relevance_scores (identical values)
✅ Server config tests: 27 passed
✅ Endpoint tests: All passed
✅ CLI check-rerank: All passed
⏳ Qwen3: Test harness ready, requires model download to test
```

### Commits (8)

1. feat: add /v1/score endpoint, refactor rerankers for client-side formatting
2. fix: escape braces in f-string comments
3. feat: add test harness config and update CLI
4. feat: implement vLLM-compatible contracts
5. refactor: clean industry-standard endpoint contracts
6. refactor: align with vLLM/Cohere/Jina standards
7. fix: fail test on score mismatch
8. fix: update CLI for new contracts

### Next Steps for cmw-rag

1. Update InfinityReranker to format query/documents client-side
2. Use `/v1/score` endpoint for reranking
3. Add `query_template`, `doc_template`, `prefix`, `suffix` to model configs
4. Handle both cross_encoder and llm_reranker model types

## CMW-RAG Refactoring Steps

**NO BACKWARD COMPATIBILITY** - Direct refactor to new contracts.

### Overview

Simply update cmw-rag to use the new vLLM/Cohere contracts. No migration phases needed.

### 1. Update Reranker Client Contract

**Before (current cmw-rag):**
```python
# Uses InfinityReranker with custom response format
response = client.post("/v1/rerank", json={"query": q, "documents": docs})
scores = response["scores"]  # Simple array
```

**After (new contract):**
```python
# Use /v1/score for raw scores (vLLM format)
response = client.post("/v1/score", json={"query": q, "documents": docs})
data = response["data"]  # [{index, object, score}, ...]
scores = [item["score"] for item in data]

# Or use /v1/rerank for sorted results (Cohere format)
response = client.post("/v1/rerank", json={"query": q, "documents": docs})
results = response["results"]  # [{index, document, relevance_score}, ...]
# Results are sorted by relevance (descending)
```

### 2. Add Model-Type-Aware Formatting

Create a `RerankerAdapter` that handles formatting based on model type:

```python
class RerankerAdapter:
    def __init__(self, model_config: dict):
        self.model_type = model_config.get("reranker_type", "cross_encoder")
        self.formatting = model_config.get("formatting", {})
    
    def format_query(self, query: str, instruction: str | None = None) -> str:
        if self.model_type == "cross_encoder":
            return query  # No formatting needed
        
        # LLM reranker: apply template
        template = self.formatting.get("query_template", "{query}")
        prefix = self.formatting.get("prefix", "")
        return template.format(prefix=prefix, instruction=instruction, query=query)
    
    def format_document(self, doc: str) -> str:
        if self.model_type == "cross_encoder":
            return doc  # No formatting needed
        
        # LLM reranker: apply template
        template = self.formatting.get("doc_template", "{doc}")
        suffix = self.formatting.get("suffix", "")
        return template.format(doc=doc, suffix=suffix)
```

### 3. Update Model Config Schema

Add formatting fields to model config:

```yaml
# cmw-rag config/models.yaml
rerankers:
  DiTy/cross-encoder-russian-msmarco:
    type: cross_encoder
    # No formatting needed
    
  Qwen/Qwen3-Reranker-0.6B:
    type: llm_reranker
    formatting:
      query_template: "{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
      doc_template: "<Document>: {doc}{suffix}"
      prefix: "<|im_start|>system\nJudge whether...\n<|im_end|>\n<|im_start|>user\n"
      suffix: "<|im_end|>\n<|im_start|>assistant\n\n\n\n\n"
      default_instruction: "Given a web search query, retrieve relevant passages"
```

### 4. Rename/Deprecate Old InfinityReranker

```python
# Old (deprecated but still works):
class InfinityReranker:
    def rerank(self, query: str, documents: list[str]) -> list[float]:
        # Returns raw scores for backward compatibility
        
# New:
class RerankerAdapter:
    def score(self, query: str, documents: list[str], 
              instruction: str | None = None) -> list[float]:
        """Returns raw scores via /v1/score endpoint."""
        
    def rerank(self, query: str, documents: list[str], 
               instruction: str | None = None,
               top_n: int | None = None) -> list[dict]:
        """Returns sorted results via /v1/rerank endpoint."""
```

### 5. Provider Abstraction

Create unified interface for multiple providers:

```python
class RerankerProvider(Protocol):
    def score(self, query: str, documents: list[str]) -> list[float]:
        """Returns raw scores in original order."""
        ...
    
    def rerank(self, query: str, documents: list[str], 
               top_n: int | None = None) -> list[dict]:
        """Returns sorted results with document text."""
        ...

class MosecRerankerProvider(RerankerProvider):
    """cmw-mosec provider (vLLM/Cohere compatible)."""
    
class VLLMRerankerProvider(RerankerProvider):
    """vLLM provider (same contract as cmw-mosec)."""
```

### 6. Instruction Handling

Move instruction from server to client:

**Before (server-side instruction):**
```python
# cmw-mosec received raw query and applied instruction server-side
response = client.post("/v1/rerank", json={
    "query": "What is France?",
    "documents": docs,
    "instruction": "search"  # Server applied this
})
```

**After (client-side instruction):**
```python
# Client formats query with instruction before sending
formatted_query = adapter.format_query(query, instruction)
response = client.post("/v1/score", json={
    "query": formatted_query,
    "documents": [adapter.format_document(doc) for doc in docs]
})
```

### Files to Update in cmw-rag

1. `rag_engine/config/models.yaml` - Add formatting templates
2. `rag_engine/config/schemas.py` - Add `RerankerFormatting` model
3. `rag_engine/retrieval/reranker.py` - Create `RerankerAdapter` class
4. `rag_engine/retrieval/infinity_reranker.py` - Replace with new adapter
5. Tests for new adapter

**That's it.** Direct refactor, no phases needed.
