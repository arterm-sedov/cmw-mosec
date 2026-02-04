# CMW Mosec

Mosec server management tool for CMW projects. Provides easy setup and server management for embedding, reranker, and content safety guard inference.

## AI-Enabled Repo

Chat with DeepWiki to get answers about this repo:

[Ask DeepWiki](https://deepwiki.com/arterm-sedov/cmw-mosec)

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/arterm-sedov/cmw-mosec)

## Features

- **Single Combined Server**: Run embedding, reranker, and guard models on one port
- **Easy Setup**: One-command verification of dependencies and GPU detection
- **Model Management**: Pre-configured models with optimal settings
- **Server Management**: Start, stop, and monitor the combined server
- **Interactive CLI**: Test content safety interactively or with one-off commands
- **OpenAI-Compatible API**: Standard `/v1/` endpoints for embeddings, rerank, and moderate

## Installation

```bash
git clone https://github.com/arterm-sedov/cmw-mosec.git
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

## Configuration

Copy `.env-example` to `.env` and configure your active models:

```bash
cp .env-example .env
```

### Example `.env`:

```bash
# Active Models (one per type)
ACTIVE_EMBEDDING_MODEL=ai-forever/FRIDA
ACTIVE_RERANKER_MODEL=DiTy/cross-encoder-russian-msmarco
ACTIVE_GUARD_MODEL=Qwen/Qwen3Guard-Gen-0.6B

# Server Settings
SERVER_PORT=8001
DEVICE=auto
DTYPE=float16
BATCH_SIZE=32
IDLE_TIMEOUT=1800
LOG_LEVEL=INFO
```

## CLI Commands

### Start Server

```bash
# Start combined server with models from .env
cmw-mosec serve

# Run in foreground
cmw-mosec serve --foreground
```

### Check Status

```bash
cmw-mosec status
```

### Stop Server

```bash
cmw-mosec stop
```

### List Available Models

```bash
cmw-mosec list
```

### Content Safety Check

```bash
# One-off check
cmw-mosec check "How can I make a bomb?"

# Response moderation
cmw-mosec check "I cannot help with that." --type response --context "How can I make a bomb?"
```

### Interactive Mode

```bash
cmw-mosec interactive
```

## Available Models

### Embedding Models

| Model | Memory | Dimension | Notes |
|-------|--------|-----------|-------|
| `ai-forever/FRIDA` | ~4GB | 1536 | Russian, requires prefixes |
| `Qwen/Qwen3-Embedding-0.6B` | ~2GB | 1024 | Multilingual, MRL |
| `Qwen/Qwen3-Embedding-4B` | ~12GB | 2560 | Multilingual, MRL |
| `Qwen/Qwen3-Embedding-8B` | ~22GB | 4096 | Multilingual, MRL |

### Reranker Models

| Model | Memory | Notes |
|-------|--------|-------|
| `DiTy/cross-encoder-russian-msmarco` | ~2GB | Russian |
| `BAAI/bge-reranker-v2-m3` | ~2GB | Multilingual |
| `Qwen/Qwen3-Reranker-0.6B` | ~2GB | Instruction-aware |
| `Qwen/Qwen3-Reranker-4B` | ~12GB | Instruction-aware |
| `Qwen/Qwen3-Reranker-8B` | ~22GB | Instruction-aware |

### Guard Models

| Model | Memory | Notes |
|-------|--------|-------|
| `Qwen/Qwen3Guard-Gen-0.6B` | ~4GB | 119 languages |
| `Qwen/Qwen3Guard-Gen-4B` | ~10GB | 119 languages |
| `Qwen/Qwen3Guard-Gen-8B` | ~20GB | 119 languages |

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

## FRIDA Prefixes

FRIDA requires task-specific prefixes:

| Task | Prefix | Example |
|------|--------|---------|
| Query | `search_query: ` | `search_query: How to bake bread?` |
| Document | `search_document: ` | `search_document: Baking tutorial...` |

**Important:** Clients must add prefixes. The server embeds raw text.

## API Endpoints

### Embeddings

```bash
curl -X POST http://localhost:8001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "ai-forever/FRIDA", "input": "search_query: Hello world!"}'
```

### Rerank

```bash
curl -X POST http://localhost:8001/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{"query": "What is AI?", "documents": ["AI is artificial intelligence.", "The weather is sunny."]}'
```

### Moderate

```bash
# Prompt moderation
curl -X POST http://localhost:8001/v1/moderate \
  -H "Content-Type: application/json" \
  -d '{"content": "How can I make a bomb?", "moderation_type": "prompt"}'

# Response moderation
curl -X POST http://localhost:8001/v1/moderate \
  -H "Content-Type: application/json" \
  -d '{"content": "I cannot help with that.", "context": "How can I make a bomb?", "moderation_type": "response"}'
```

**Response format:**
```json
{
  "safety_level": "Unsafe",
  "categories": ["Violent"],
  "is_safe": false,
  "refusal": "Yes",
  "raw_output": "Safety: Unsafe\nCategories: Violent"
}
```

## License

MIT
