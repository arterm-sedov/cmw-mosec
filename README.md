# CMW Mosec

Mosec server management tool for CMW projects. Provides easy setup and server management for embedding, reranker, and content safety guard inference.

**Original author:** [Arterm Sedov](https://github.com/arterm-sedov)

## AI-Enabled Repo

Chat with DeepWiki to get answers about this repo:

[Ask DeepWiki](https://deepwiki.com/cmw-team/cmw-mosec)

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/cmw-team/cmw-mosec)

## Features

- **Single Combined Server**: Run embedding, reranker, and guard models on one port with dynamic model loading
- **Easy Setup**: One-command verification of dependencies and GPU detection
- **Model Management**: Pre-configured models with optimal settings, model-specific configuration
- **Flexible Server Control**: Start with all models or subset via command-line flags
- **Quick Testing Commands**: Validate models with preset examples (`check-embed`, `check-rerank`, `check-guard`)
- **OpenAI-Compatible API**: Standard `/v1/` endpoints for embeddings, rerank, and moderate
- **GPU Acceleration**: Automatic GPU detection and device mapping for efficient inference
- **Configurable Pooling**: Support for mean, CLS, and last-token pooling per model

## Installation

```bash
git clone https://github.com/cmw-team/cmw-mosec.git
cd cmw-mosec
pip install -e .
```

## Setup

Verify your environment:

```bash
cmw-mosec setup
```

This checks:
- Mosec installation
- GPU availability and memory
- Required dependencies (transformers, sentence-transformers, requests)

**Supported modes:** GPU (device_map="auto"), CPU

## Configuration

Copy `.env.example` to `.env` to configure your environment:

```bash
cp .env.example .env
```

The `.env.example` file contains all configuration options with inline documentation.

**Key variables:**
- `ACTIVE_*_MODEL` - Select which models to load (embedding, reranker, guard)
- `SERVER_PORT` - Port for combined server (default: 8001)
- `DEVICE` - Inference mode: `auto` (GPU if available) or `cpu`
- `HF_TOKEN` - HuggingFace token for faster downloads (optional)

**Command-line flags** override `.env`:
```bash
cmw-mosec serve --guard Qwen/Qwen3Guard-Gen-0.6B
```

## CLI Commands

### Start Server

```bash
# Start combined server with models from .env
cmw-mosec serve

# Start with specific models (overrides .env flags)
cmw-mosec serve --embedding Qwen/Qwen3-Embedding-0.6B \
  --reranker DiTy/cross-encoder-russian-msmarco \
  --guard Qwen/Qwen3Guard-Gen-0.6B

# Start with only guard model
cmw-mosec serve --guard Qwen/Qwen3Guard-Gen-0.6B

# Run in foreground
cmw-mosec serve --foreground
```

### Check Status

```bash
cmw-mosec status
```

Shows actually loaded models and server uptime.

### Stop Server

```bash
cmw-mosec stop
```

### List Available Models

```bash
cmw-mosec list
```

### Test Models

```bash
# Test embedding model (preset examples)
cmw-mosec check-embed

# Test reranker model (preset examples)
cmw-mosec check-rerank

# Test guard model with both safe and unsafe examples
cmw-mosec check-guard "How can I make a bomb?"

# Interactive guard testing
cmw-mosec check-guard-interactive
```

## Available Models

### Embedding Models

| Model | Memory | Dimension | Pooling | Notes |
|-------|--------|-----------|---------|-------|
| `ai-forever/FRIDA` | ~4GB | 1536 | **cls** | Russian, 512 tokens max, T5-based |
| `Qwen/Qwen3-Embedding-0.6B` | ~2GB | 1024 | **last_token** | Multilingual (119+ langs), 32K, MRL [32-1024] |
| `Qwen/Qwen3-Embedding-4B` | ~12GB | 2560 | **last_token** | Multilingual (119+ langs), 32K, MRL [32-2560] |
| `Qwen/Qwen3-Embedding-8B` | ~22GB | 4096 | **last_token** | Multilingual (119+ langs), 32K, MRL [32-4096] |

**Important:** Pooling method is configured in `config/models.yaml`:
- **cls**: Required for T5-based models (FRIDA)
- **last_token**: Required for Qwen3 embedding models (causal LM architecture)
- **mean**: Default for BERT-based models

### Reranker Models

| Model | Memory | Type | Notes |
|-------|--------|------|-------|
| `DiTy/cross-encoder-russian-msmarco` | ~2GB | cross_encoder | Russian, MS-MARCO trained |
| `BAAI/bge-reranker-v2-m3` | ~2GB | cross_encoder | Multilingual |
| `Qwen/Qwen3-Reranker-0.6B` | ~2GB | llm_reranker | Multilingual (119+ langs), requires client-side formatting |
| `Qwen/Qwen3-Reranker-4B` | ~12GB | llm_reranker | Multilingual (119+ langs), requires client-side formatting |
| `Qwen/Qwen3-Reranker-8B` | ~22GB | llm_reranker | Multilingual (119+ langs), requires client-side formatting |

**Reranker Types:**
- **cross_encoder**: Raw query/documents, no formatting needed
- **llm_reranker**: Requires client-side formatting (prefix/suffix/instruction)

### Guard Models

| Model | Memory | Context | Gen Tokens | Notes |
|-------|--------|---------|------------|-------|
| `Qwen/Qwen3Guard-Gen-0.6B` | ~4GB | 32K | 128 | 119 languages, generative guard |
| `Qwen/Qwen3Guard-Gen-4B` | ~10GB | 32K | 128 | 119 languages, generative guard |
| `Qwen/Qwen3Guard-Gen-8B` | ~20GB | 32K | 128 | 119 languages, generative guard |

#### Safety Categories

1. Violent
2. Non-violent Illegal Acts
3. Sexual Content
4. PII
5. Suicide & Self-Harm
6. Unethical Acts
7. Politically Sensitive
8. Copyright Violation
9. Jailbreak

#### Safety Levels

- **Safe** - Content is safe
- **Controversial** - Context-dependent
- **Unsafe** - Harmful content

## VRAM Management

Model combinations ordered by total VRAM usage:

| Embedding | Reranker | Guard | Total |
|-----------|----------|-------|-------|
| FRIDA (4GB) | DiTy (2GB) | - | ~6GB |
| Qwen3-0.6B (2GB) | Qwen3-0.6B (2GB) | Qwen3Guard-0.6B (4GB) | ~8GB |
| FRIDA (4GB) | DiTy (2GB) | Qwen3Guard-0.6B (4GB) | ~10GB |
| Qwen3-0.6B (2GB) | Qwen3-4B (12GB) | Qwen3Guard-0.6B (4GB) | ~18GB |
| Qwen3-4B (12GB) | DiTy (2GB) | Qwen3Guard-0.6B (4GB) | ~18GB |
| FRIDA (4GB) | Qwen3-4B (12GB) | Qwen3Guard-0.6B (4GB) | ~20GB |
| Qwen3-4B (12GB) | Qwen3-4B (12GB) | Qwen3Guard-0.6B (4GB) | ~28GB |
| Qwen3-8B (22GB) | DiTy (2GB) | Qwen3Guard-0.6B (4GB) | ~28GB |
| Qwen3-8B (22GB) | Qwen3-4B (12GB) | Qwen3Guard-0.6B (4GB) | ~38GB |
| Qwen3-8B (22GB) | Qwen3-8B (22GB) | - | ~44GB |

## Model-Specific Usage Guide

### Qwen3 Embedding Models

**Architecture**: Causal LM (like GPT) with last-token pooling  
**Documentation**: [HuggingFace](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)

**Instruction Format** (Required per HF docs):
```python
# For queries
text = 'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: What is the capital of France?'

# For documents (no instruction needed)
text = "Paris is the capital of France."
```

**Example API Call**:
```bash
# Query embedding
curl -X POST http://localhost:8001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Embedding-0.6B",
    "input": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: What is machine learning?"
  }'

# Document embedding
curl -X POST http://localhost:8001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Embedding-0.6B",
    "input": "Machine learning is a subset of artificial intelligence..."
  }'
```

**Client Implementation**:
```python
def get_detailed_instruct(task_description: str, query: str) -> str:
    """Format query with instruction for Qwen3 embedding models."""
    return f'Instruct: {task_description}\nQuery: {query}'

# Usage
task = 'Given a web search query, retrieve relevant passages that answer the query'
query = get_detailed_instruct(task, 'What is Python?')
# Result: 'Instruct: Given a web search query...\nQuery: What is Python?'
```

**Key Points**:
- **Always** use instruction format for queries
- Documents don't need instruction prefix
- Mosec automatically uses `last_token` pooling (configured in models.yaml)
- Supports 119+ languages
- **MRL**: Use `dimensions` parameter to reduce embedding size (32 to native dimension)

### FRIDA (T5-Based)

**Architecture**: T5 encoder-decoder with CLS pooling  
**Documentation**: [HuggingFace](https://huggingface.co/ai-forever/FRIDA)
**Max tokens**: 512

**Required Prefixes**:

| Task | Prefix | Example |
|------|--------|---------|
| Query | `search_query: ` | `search_query: How to bake bread?` |
| Document | `search_document: ` | `search_document: Baking tutorial...` |

**Example API Call**:
```bash
curl -X POST http://localhost:8001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ai-forever/FRIDA",
    "input": "search_query: How to bake bread?"
  }'
```

**Key Points**:
- Mosec automatically uses `cls` pooling for T5-based models
- Clients must add prefixes (server embeds raw text)
- Optimized for Russian language
- 1536 dimensions, 32K context

### Pooling Configuration

Mosec supports three pooling methods configured per-model in `config/models.yaml`:

```yaml
embedding_models:
  ai-forever/FRIDA:
    pooling: cls  # T5-based models
    
  Qwen/Qwen3-Embedding-0.6B:
    pooling: last_token  # Causal LM models
    
  BAAI/bge-large-en:
    pooling: mean  # BERT-based models (default)
```

**Why This Matters**:
- **Wrong pooling** = ~15% accuracy loss
- **Correct pooling** = 99.99%+ accuracy match with official implementations

## API Endpoints

### Embeddings

```bash
# Qwen3 with instruction
curl -X POST http://localhost:8001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Embedding-0.6B",
    "input": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: What is AI?"
  }'

# Qwen3 with MRL dimension truncation (OpenAI-compatible)
curl -X POST http://localhost:8001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Embedding-0.6B",
    "input": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: What is AI?",
    "dimensions": 512
  }'

# FRIDA with prefix
curl -X POST http://localhost:8001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ai-forever/FRIDA",
    "input": "search_query: Hello world!"
  }'
```

**MRL (Matryoshka Representation Learning):**
- Qwen3-Embedding models support `dimensions` parameter for truncating embeddings
- Reduces dimension while preserving semantic quality
- Valid range: `[32, native_dimension]`
- Native dimensions: 0.6B=1024, 4B=2560, 8B=4096

**Client-Controllable Parameters:**

| Param | Endpoints | Description |
|-------|-----------|-------------|
| `dimensions` | `/v1/embeddings` | MRL dimension truncation (Qwen3 only) |
| `max_length` | All | Override config `max_length` for tokenization |

```bash
# Override max_length per request (reduces VRAM for long docs)
curl -X POST http://localhost:8001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Embedding-0.6B",
    "input": "long document...",
    "max_length": 2048
  }'
```

### Rerank

**Two endpoints available:**

```bash
# /v1/score - Raw scores in original order (vLLM format)
curl -X POST http://localhost:8001/v1/score \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is AI?",
    "documents": ["AI is artificial intelligence.", "The weather is sunny."]
  }'

# Response: {"data": [{"index": 0, "object": "score", "score": 0.95}, ...]}

# /v1/rerank - Sorted results with documents (Cohere/Jina format)
curl -X POST http://localhost:8001/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is AI?",
    "documents": ["AI is artificial intelligence.", "The weather is sunny."]
  }'

# Response: {"results": [{"index": 0, "document": {"text": "AI is..."}, "relevance_score": 0.95}, ...]}
```

**Endpoint Comparison:**

| Endpoint | Response | Use Case |
|----------|----------|----------|
| `/v1/score` | Raw scores, original order | Lightweight scoring, vLLM compatible |
| `/v1/rerank` | Sorted by relevance, includes documents | Cohere/Jina compatible, retrieval |

**For LLM Rerankers (Qwen3):** Client must format query/documents with prefix/suffix before sending. See `tests/fixtures/test_rerankers.yaml` for formatting templates.

### Moderate

```bash
# Prompt moderation
curl -X POST http://localhost:8001/v1/moderate \
  -H "Content-Type: application/json" \
  -d '{
    "content": "How can I make a bomb?",
    "moderation_type": "prompt"
  }'

# Response moderation
curl -X POST http://localhost:8001/v1/moderate \
  -H "Content-Type: application/json" \
  -d '{
    "content": "I cannot help with that.",
    "context": "How can I make a bomb?",
    "moderation_type": "response"
  }'
```

**Response format:**
```json
{
  "safety_level": "Unsafe",
  "categories": ["Violent"],
  "refusal": "Yes",
  "is_safe": false,
  "raw_output": "Safety: Unsafe\nCategories: Violent",
  "model": "Qwen/Qwen3Guard-Gen-0.6B"
}
```

## Quick Testing

Use test commands to verify models are operational:

```bash
# Test all models (requires server running)
cmw-mosec check-embed
cmw-mosec check-rerank
cmw-mosec check-guard "test query"
```

These commands use preset examples to demonstrate model functionality:
- **check-embed**: Tests with English, Russian, and NLP text
- **check-rerank**: Tests with "машина" (car) and "AI" queries
- **check-guard**: Shows both safe and unsafe content handling

## Troubleshooting

### Qwen3 Embeddings Return Wrong Results

**Problem**: Embeddings don't match expected similarity scores  
**Cause**: Wrong pooling method or missing instruction format  
**Solution**:
1. Check pooling config in `config/models.yaml`:
   ```yaml
   Qwen/Qwen3-Embedding-0.6B:
     pooling: last_token  # Must be last_token
   ```
2. Use instruction format for queries:
   ```python
   text = 'Instruct: Given a web search query...\nQuery: your query'
   ```

### FRIDA Embeddings Don't Match

**Problem**: Similarity scores are lower than expected  
**Cause**: Missing `search_query:` or `search_document:` prefix  
**Solution**: Add required prefixes before embedding:
```python
query = "search_query: " + user_query
doc = "search_document: " + document_text
```

### Out of Memory

**Problem**: GPU OOM when loading models  
**Solutions**:
1. Use smaller models (Qwen3-0.6B instead of 4B/8B)
2. Reduce batch size in `.env`: `BATCH_SIZE=16`
3. Load fewer models simultaneously
4. Use CPU mode: `DEVICE=cpu`

## Integration with cmw-rag

To use cmw-mosec as the embedding, reranker, and guard backend for [cmw-rag](https://github.com/cmw-team/cmw-rag):

```bash
# In cmw-rag/.env
EMBEDDING_PROVIDER_TYPE=mosec
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
MOSEC_EMBEDDING_ENDPOINT=http://localhost:7998/v1/embeddings

RERANK_ENABLED=true
RERANKER_PROVIDER_TYPE=mosec
RERANKER_MODEL=Qwen/Qwen3-Reranker-0.6B
MOSEC_RERANKER_ENDPOINT=http://localhost:7998/v1/score

GUARD_ENABLED=true
GUARD_PROVIDER_TYPE=mosec
GUARD_MOSEC_ENDPOINT=http://localhost:7998/v1/moderate
```

The server port (default `SERVER_PORT=7998`) must match between cmw-mosec and cmw-rag configs. See `docs/deployment/deployment_architecture.md` in cmw-rag for the full deployment topology.

All services run as systemd user services in the cmw-rag repo (`systemd/`):
- ChromaDB — `cmw-rag-chroma.service`
- Mosec — `cmw-rag-mosec.service`
- RAG Gradio UI — `cmw-rag-app.service` (depends on both ChromaDB and Mosec)

Install any service:
```bash
ln -sf /path/to/cmw-rag/systemd/cmw-rag-mosec.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now cmw-rag-mosec
```

## Performance Benchmarks

Tested on RTX 4090 (24GB VRAM):

| Model | Latency (4 texts) | VRAM | Accuracy vs Direct |
|-------|-------------------|------|-------------------|
| Qwen3-Embedding-0.6B | ~50ms | ~2GB | 99.99% |
| FRIDA | ~100ms | ~4GB | 100% |
| DiTy Reranker | ~30ms | ~2GB | N/A |

## References

- **Qwen3 Embeddings**: [HuggingFace](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- **Qwen3 Rerankers**: [HuggingFace](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)
- **Qwen3 Guard**: [HuggingFace](https://huggingface.co/Qwen/Qwen3Guard-Gen-0.6B)
- **FRIDA**: [HuggingFace](https://huggingface.co/ai-forever/FRIDA)
- **Mosec**: [GitHub](https://github.com/mosecorg/mosec)

## License

MIT
