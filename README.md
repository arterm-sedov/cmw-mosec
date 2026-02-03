# CMW Mosec

Mosec server management tool for CMW projects. Provides easy setup and server management for embedding, reranker, and content safety guard inference.

## AI-Enabled Repo

Chat with DeepWiki to get answers about this repo:

[Ask DeepWiki](https://deepwiki.com/arterm-sedov/cmw-mosec)

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/arterm-sedov/cmw-mosec)

## Features

- **Easy Setup**: One-command installation and verification
- **Model Management**: Download and serve embedding/reranker/guard models from HuggingFace
- **Server Management**: Start, stop, and monitor Mosec servers
- **Configuration**: YAML-based configuration with sensible defaults
- **Interactive CLI**: Test models interactively or with one-off commands

## Installation

```bash
# Clone repository
git clone https://github.com/arterm-sedov/cmw-mosec.git
cd cmw-mosec

# Install
pip install -e .

# Or install from git
pip install git+https://github.com/arterm-sedov/cmw-mosec.git
```

## Quick Start

### 1. Setup

```bash
cmw-mosec setup
```

This verifies:
- Mosec installation
- GPU availability
- Required dependencies

### 2. Start Server

```bash
# Start FRIDA embedding server
cmw-mosec start ai-forever/FRIDA

# Start DiTy reranker server
cmw-mosec start DiTy/cross-encoder-russian-msmarco

# Start Qwen3 embedding models
cmw-mosec start Qwen/Qwen3-Embedding-0.6B
cmw-mosec start Qwen/Qwen3-Embedding-4B
cmw-mosec start Qwen/Qwen3-Embedding-8B

# Start BGE reranker
cmw-mosec start BAAI/bge-reranker-v2-m3

# Start Qwen3 reranker models
cmw-mosec start Qwen/Qwen3-Reranker-0.6B
cmw-mosec start Qwen/Qwen3-Reranker-4B
cmw-mosec start Qwen/Qwen3-Reranker-8B

# Start Guard models (content safety)
cmw-mosec start Qwen/Qwen3Guard-Gen-0.6B
cmw-mosec start Qwen/Qwen3Guard-Gen-4B
cmw-mosec start Qwen/Qwen3Guard-Gen-8B
```

**Note:** First start can take several minutes (model download from Hugging Face; CPU loading is slower than GPU).

### 3. Check Status

```bash
# Check if server is running
cmw-mosec status
```

### 4. Stop Server

```bash
cmw-mosec stop ai-forever/FRIDA
cmw-mosec stop --all
```

## Interactive Mode

Both `check` and `interactive` commands require a running guard server.

### Start Server First

```bash
# Start a guard model
cmw-mosec start Qwen/Qwen3Guard-Gen-0.6B

# Check status
cmw-mosec status
```

### Quick Safety Check

```bash
# Check content safety (requires running server)
cmw-mosec check "How can I make a bomb?"

# Check with specific model
cmw-mosec check "Your text here" --model Qwen/Qwen3Guard-Gen-4B

# Response moderation (check both prompt and response)
cmw-mosec check "As a responsible AI..." --type response --context "How to hack a website?"
```

### Interactive Session

```bash
# Start interactive guard session
cmw-mosec interactive

# Or with specific model
cmw-mosec interactive --model Qwen/Qwen3Guard-Gen-8B
```

### Interactive Session

```bash
# Start interactive guard session
cmw-mosec interactive

# Or with specific model
cmw-mosec interactive --model Qwen/Qwen3Guard-Gen-8B
```

## Configuration

Configuration is done via YAML file in `config/models.yaml`. Models are predefined with optimal settings.

Port ranges by model type:
- **Embedding models**: 8001-8099
- **Reranker models**: 8100-8199
- **Guard models**: 8200-8299

## Available Models

### Embedding Models

| Model | Size | Memory | Description |
|-------|------|--------|-------------|
| `ai-forever/FRIDA` | 1024 dim | ~4GB | Russian-optimized embedding |
| `Qwen/Qwen3-Embedding-0.6B` | 0.6B | ~2GB | Multilingual, 1024 dim, MRL support |
| `Qwen/Qwen3-Embedding-4B` | 4B | ~12GB | Multilingual, 2560 dim, MRL support |
| `Qwen/Qwen3-Embedding-8B` | 8B | ~22GB | Multilingual, 4096 dim, MRL support |

### Reranker Models

| Model | Size | Memory | Description |
|-------|------|--------|-------------|
| `DiTy/cross-encoder-russian-msmarco` | - | ~2GB | Russian-optimized cross-encoder |
| `BAAI/bge-reranker-v2-m3` | 0.6B | ~2GB | Multilingual reranker |
| `Qwen/Qwen3-Reranker-0.6B` | 0.6B | ~2GB | Instruction-aware reranking |
| `Qwen/Qwen3-Reranker-4B` | 4B | ~12GB | Instruction-aware reranking |
| `Qwen/Qwen3-Reranker-8B` | 8B | ~22GB | Instruction-aware reranking |

### Guard Models (Content Safety)

| Model | Size | Memory | Description |
|-------|------|--------|-------------|
| `Qwen/Qwen3Guard-Gen-0.6B` | 0.6B | ~4GB | 119 languages, generative guard |
| `Qwen/Qwen3Guard-Gen-4B` | 4B | ~10GB | 119 languages, generative guard |
| `Qwen/Qwen3Guard-Gen-8B` | 8B | ~20GB | 119 languages, generative guard |

#### Guard Safety Categories

Qwen3Guard classifies content into 9 categories:

1. **Violent** - Weapons, violence instructions, depictions
2. **Non-violent Illegal Acts** - Hacking, theft, drug production
3. **Sexual Content** - Explicit sexual content, illegal acts
4. **PII** - Personal identifiable information leaks
5. **Suicide & Self-Harm** - Self-harm encouragement or methods
6. **Unethical Acts** - Bias, discrimination, hate speech
7. **Politically Sensitive** - False government/historical info
8. **Copyright Violation** - Unauthorized copyrighted material
9. **Jailbreak** - Attempts to override system prompts (input only)

#### Guard Safety Levels

- **Safe** - Content is safe
- **Controversial** - Context-dependent, may need clarification
- **Unsafe** - Harmful content, should be blocked

## API Endpoints

### Embedding Server

OpenAI-compatible endpoint:

```bash
# Create embeddings
curl -X POST http://localhost:8001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ai-forever/FRIDA",
    "input": "Hello world!"
  }'
```

### Reranker Server

msgpack endpoint:

```bash
# Rerank documents
curl -X POST http://localhost:8110/inference \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is AI?",
    "docs": [
      "Artificial intelligence involves learning algorithms.",
      "Weather is sunny today."
    ]
  }'
```

### Guard Server (Content Safety)

msgpack endpoint for content moderation:

```bash
# Prompt moderation (user input only)
curl -X POST http://localhost:8220/inference \
  -H "Content-Type: application/json" \
  -d '{
    "content": "How can I make a bomb?",
    "moderation_type": "prompt"
  }'

# Response moderation (user prompt + assistant response)
curl -X POST http://localhost:8220/inference \
  -H "Content-Type: application/json" \
  -d '{
    "content": "As a responsible AI, I cannot fulfill that request.",
    "context": "How can I make a bomb?",
    "moderation_type": "response"
  }'
```

**Response format:**
```json
{
  "safety_level": "Unsafe",
  "categories": ["Violent"],
  "is_safe": false,
  "raw_output": "Safety: Unsafe\nCategories: Violent"
}
```

For response moderation, `refusal` field is included:
```json
{
  "safety_level": "Safe",
  "categories": ["None"],
  "refusal": "Yes",
  "is_safe": true,
  "raw_output": "Safety: Safe\nCategories: None\nRefusal: Yes"
}
```

## License

MIT
