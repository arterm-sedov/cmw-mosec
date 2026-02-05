# CMW Mosec

Mosec server management tool for CMW projects. Provides easy setup and server management for embedding, reranker, and content safety guard inference.

## AI-Enabled Repo

Chat with DeepWiki to get answers about this repo:

[Ask DeepWiki](https://deepwiki.com/arterm-sedov/cmw-mosec)

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/arterm-sedov/cmw-mosec)

## Features

- **Single Combined Server**: Run embedding, reranker, and guard models on one port with dynamic model loading
- **Easy Setup**: One-command verification of dependencies and GPU detection
- **Model Management**: Pre-configured models with optimal settings, model-specific configuration
- **Flexible Server Control**: Start with all models or subset via command-line flags
- **Quick Testing Commands**: Validate models with preset examples (`check-embed`, `check-rerank`, `check-guard`)
- **OpenAI-Compatible API**: Standard `/v1/` endpoints for embeddings, rerank, and moderate
- **GPU Acceleration**: Automatic GPU detection and device mapping for efficient inference

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

| Model | Memory | Dimension | Notes |
|-------|--------|-----------|-------|
| `ai-forever/FRIDA` | ~4GB | 1536 | Russian, 32K context, requires prefixes |
| `Qwen/Qwen3-Embedding-0.6B` | ~2GB | 1024 | Multilingual (119+ langs), 32K, MRL |
| `Qwen/Qwen3-Embedding-4B` | ~12GB | 2560 | Multilingual (119+ langs), 32K, MRL |
| `Qwen/Qwen3-Embedding-8B` | ~22GB | 4096 | Multilingual (119+ langs), 32K, MRL |

### Reranker Models

| Model | Memory | Notes |
|-------|--------|-------|
| `DiTy/cross-encoder-russian-msmarco` | ~2GB | Russian, MS-MARCO trained |
| `BAAI/bge-reranker-v2-m3` | ~2GB | Multilingual |
| `Qwen/Qwen3-Reranker-0.6B` | ~2GB | Multilingual (119+ langs), instruction-aware |
| `Qwen/Qwen3-Reranker-4B` | ~12GB | Multilingual (119+ langs), instruction-aware |
| `Qwen/Qwen3-Reranker-8B` | ~22GB | Multilingual (119+ langs), instruction-aware |

### Guard Models

| Model | Memory | Max Tokens | Notes |
|-------|--------|------------|-------|
| `Qwen/Qwen3Guard-Gen-0.6B` | ~4GB | 128 | 119 languages, generative guard |
| `Qwen/Qwen3Guard-Gen-4B` | ~10GB | 128 | 119 languages, generative guard |
| `Qwen/Qwen3Guard-Gen-8B` | ~20GB | 128 | 119 languages, generative guard |

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

## Model-Specific Notes

### FRIDA Prefixes

FRIDA requires task-specific prefixes (not needed for Qwen3 models):

| Task | Prefix | Example |
|------|--------|---------|
| Query | `search_query: ` | `search_query: How to bake bread?` |
| Document | `search_document: ` | `search_document: Baking tutorial...` |

**Important:** Clients must add prefixes for FRIDA. The server embeds raw text.

### Guard Model Output

Guard models categorize content into:
- **Safety Levels**: Safe, Controversial, Unsafe
- **Categories**: Violent, Non-violent Illegal Acts, Sexual Content, PII, Suicide & Self-Harm, Unethical Acts, Politically Sensitive, Copyright Violation, Jailbreak

The model outputs structured JSON with `safety_level`, `categories`, `is_safe`, and `raw_output`.

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

## License

MIT
