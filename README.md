# CMW Mosec

Combined Mosec server for embedding, reranker, and content safety guard inference.

## Architecture

Single combined server with up to 3 models loaded simultaneously:
- **1 Embedding model** → `/v1/embeddings` endpoint
- **1 Reranker model** → `/v1/rerank` endpoint
- **1 Guard model** → `/v1/moderate` endpoint

All configured via `.env` file.

## Installation

```bash
git clone https://github.com/arterm-sedov/cmw-mosec.git
cd cmw-mosec
pip install -e .
```

## Configuration

Copy `.env-example` to `.env` and configure:

```bash
cp .env-example .env
# Edit .env with your active models
```

### Example `.env`:

```bash
# Active Models (one per type)
ACTIVE_EMBEDDING_MODEL=ai-forever/FRIDA
ACTIVE_RERANKER_MODEL=DiTy/cross-encoder-russian-msmarco
ACTIVE_GUARD_MODEL=Qwen/Qwen3Guard-Gen-0.6B

# Server Settings
SERVER_PORT=8000
DEVICE=auto
DTYPE=float16
BATCH_SIZE=32
IDLE_TIMEOUT=1800
LOG_LEVEL=INFO
```

## Usage

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

### Interactive Safety Check

```bash
# Interactive mode
cmw-mosec interactive

# One-off check
cmw-mosec check "How can I make a bomb?"
```

## Available Models

### Embedding Models

| Model | Memory | Description |
|-------|--------|-------------|
| `ai-forever/FRIDA` | ~4GB | Russian-optimized |
| `Qwen/Qwen3-Embedding-0.6B` | ~2GB | Multilingual, 1024 dim |
| `Qwen/Qwen3-Embedding-4B` | ~12GB | Multilingual, 2560 dim |
| `Qwen/Qwen3-Embedding-8B` | ~22GB | Multilingual, 4096 dim |

### Reranker Models

| Model | Memory | Description |
|-------|--------|-------------|
| `DiTy/cross-encoder-russian-msmarco` | ~2GB | Russian-optimized |
| `BAAI/bge-reranker-v2-m3` | ~2GB | Multilingual |
| `Qwen/Qwen3-Reranker-0.6B` | ~2GB | Instruction-aware |
| `Qwen/Qwen3-Reranker-4B` | ~12GB | Instruction-aware |
| `Qwen/Qwen3-Reranker-8B` | ~22GB | Instruction-aware |

### Guard Models (Content Safety)

| Model | Memory | Description |
|-------|--------|-------------|
| `Qwen/Qwen3Guard-Gen-0.6B` | ~4GB | 119 languages |
| `Qwen/Qwen3Guard-Gen-4B` | ~10GB | 119 languages |
| `Qwen/Qwen3Guard-Gen-8B` | ~20GB | 119 languages |

#### Guard Safety Categories

1. Violent
2. Non-violent Illegal Acts
3. Sexual Content
4. PII
5. Suicide & Self-Harm
6. Unethical Acts
7. Politically Sensitive
8. Copyright Violation
9. Jailbreak

#### Guard Safety Levels

- **Safe** - Content is safe
- **Controversial** - Context-dependent
- **Unsafe** - Harmful content

## API Endpoints

### Embeddings

```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "ai-forever/FRIDA", "input": "Hello world!"}'
```

### Rerank

```bash
curl -X POST http://localhost:8000/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{"query": "What is AI?", "docs": ["AI is artificial intelligence.", "The weather is sunny."]}'
```

### Moderate

```bash
# Prompt moderation
curl -X POST http://localhost:8000/v1/moderate \
  -H "Content-Type: application/json" \
  -d '{"content": "How can I make a bomb?", "moderation_type": "prompt"}'

# Response moderation
curl -X POST http://localhost:8000/v1/moderate \
  -H "Content-Type: application/json" \
  -d '{"content": "I cannot help with that.", "context": "How can I make a bomb?", "moderation_type": "response"}'
```

Response format:
```json
{
  "safety_level": "Unsafe",
  "categories": ["Violent"],
  "is_safe": false,
  "raw_output": "Safety: Unsafe\nCategories: Violent"
}
```

## VRAM Management

With 48GB VRAM shared with other processes, recommended combinations:

| Embedding | Reranker | Guard | Total |
|-----------|-----------|-------|-------|
| Qwen3-4B (12GB) | Qwen3-4B (12GB) | Qwen3Guard-0.6B (4GB) | ~28GB |
| Qwen3-8B (22GB) | DiTy (2GB) | Qwen3Guard-0.6B (4GB) | ~28GB |
| FRIDA (4GB) | DiTy (2GB) | Qwen3Guard-4B (10GB) | ~16GB |

## License

MIT
