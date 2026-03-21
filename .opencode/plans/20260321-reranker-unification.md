# Reranker Unification Plan - March 21, 2026

## Executive Summary

Move ALL abstraction to cmw-rag. The client adapter owns formatting (including prefix/suffix),
instruction handling, and score extraction. Servers (cmw-mosec, vLLM) are agnostic: load model,
accept pre-formatted query + documents batch, return scores. Unified API contract across all
providers. Test harness acts as a client with its own dynamic config.

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

## Unified API Contract

### For ALL Rerankers (cross-encoder AND llm_reranker)

```json
// Request: Client sends pre-formatted query and documents
{
  "query": "<formatted query string>",
  "documents": ["<formatted doc1>", "<formatted doc2>", "..."],
  "max_length": 8192
}

// Response: Server returns scores in same order
{
  "scores": [0.95, 0.12, 0.03]
}
```

**One request, N documents, N scores. Same contract for mosec, vLLM, any server.**

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
2. Receive `{query, documents, max_length}`
3. Pair query with each document
4. Tokenize, truncate, run inference
5. Return `{scores: [...]}`

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

### Phase 1: cmw-mosec Server Simplification

**`cmw_mosec/server_manager.py`:**
1. For `llm_reranker` type: accept `{query, documents}` where query and documents are
   pre-formatted strings (client already applied prefix/suffix/instruction)
2. Remove: prefix/suffix construction, instruction handling, user content formatting
3. Keep: tokenization, max_length truncation, scoring logic (from config)
4. Server pairs query with each document, tokenizes each pair, runs inference
5. Cross-encoder path: UNCHANGED (same `{query, docs}` contract)

**`config/models.yaml`:**
- Remove `default_instruction` from Qwen3 configs (client-side concern)
- Add `scoring_tokens` and `scoring_method` (server needs these for inference)

**`tests/test_rerankers.yaml`:**
- Create separate test config with formatting templates and test data
- Test harness formats like a client

### Phase 2: cmw-rag Client Enhancement

**`rag_engine/config/models.yaml`:**
- Add `query_template`, `doc_template`, `prefix`, `suffix`, `default_instruction`
  to Qwen3 and BGE-Gemma configs

**`rag_engine/config/schemas.py`:**
- Add fields to `ServerRerankerConfig`: `query_template`, `doc_template`,
  `prefix`, `suffix`, `prompt`, `default_instruction`

**`rag_engine/retrieval/reranker.py`:**
```python
class InfinityReranker(HTTPClientMixin):
    def rerank(self, query, candidates, top_k, instruction=None, **kwargs):
        documents = [
            doc.page_content if hasattr(doc, "page_content") else str(doc)
            for doc, _ in candidates
        ]

        if self.config.query_template:
            # LLM reranker: format query and documents client-side
            task = instruction or self.config.default_instruction or ""
            prefix = self.config.prefix or ""
            suffix = self.config.suffix or ""
            prompt = self.config.prompt or ""

            formatted_query = self.config.query_template.format(
                prefix=prefix, instruction=task, query=query
            )
            formatted_docs = [
                self.config.doc_template.format(doc=doc, suffix=suffix, prompt=prompt)
                for doc in documents
            ]
            response = self._post({
                "query": formatted_query,
                "documents": formatted_docs,
            })
        else:
            # Cross-encoder: pass through unchanged
            response = self._post({
                "query": query,
                "documents": documents,
                "top_k": top_k,
            })

        scores = response["scores"]
        # ... existing metadata boost and sort logic
```

### Phase 3: Testing

- [ ] DiTy regression: `{query, documents}` path unchanged
- [ ] Qwen3: client formats query+docs, server scores, results match model card
- [ ] Test harness: formats from test yaml, validates scores
- [ ] Compare scores with current implementation

## Errata in Current cmw-mosec Implementation

1. **Suffix bug:** Missing `<think>\n\n</think>` tags per model card
2. **max_length:** Should be `max_length - prefix_len - suffix_len` for tokenization
   (with client-side prefix/suffix, server just needs raw `max_length`)

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