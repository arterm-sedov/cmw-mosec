# CMW Mosec

Mosec server management tool for CMW projects. Provides easy setup and server management for embedding and reranking inference servers.

## AI-Enabled Repo

Chat with DeepWiki to get answers about this repo:

[Ask DeepWiki](https://deepwiki.com/arterm-sedov/cmw-mosec)

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/arterm-sedov/cmw-mosec)

## Features

- **Easy Setup**: One-command installation and verification
- **Model Management**: Download and serve embedding/reranker models from HuggingFace
- **Server Management**: Start, stop, and monitor Mosec servers
- **Configuration**: YAML-based configuration with sensible defaults

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

## Configuration

Configuration is done via YAML file in `config/models.yaml`. Models are predefined with optimal settings.

## Available Models

### Embedding Models
- `ai-forever/FRIDA` - 1024 dim, Russian optimized, ~4GB
- `Qwen/Qwen3-Embedding-0.6B` - 1024 dim, multilingual, ~2GB
- `Qwen/Qwen3-Embedding-4B` - 2560 dim, multilingual, ~12GB
- `Qwen/Qwen3-Embedding-8B` - 4096 dim, multilingual, ~22GB

### Reranker Models
- `DiTy/cross-encoder-russian-msmarco` - Russian optimized, ~2GB
- `BAAI/bge-reranker-v2-m3` - Multilingual, ~2GB
- `Qwen/Qwen3-Reranker-0.6B` - Multilingual, ~2GB
- `Qwen/Qwen3-Reranker-4B` - Multilingual, ~12GB
- `Qwen/Qwen3-Reranker-8B` - Multilingual, ~22GB

## API Endpoints

### Embedding Server (ai-forever/FRIDA)

The embedding server provides OpenAI-compatible endpoints:

```bash
# Create embeddings
curl -X POST http://localhost:8001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ai-forever/FRIDA",
    "input": "Hello world!"
  }'
```

### Reranker Server (DiTy/cross-encoder-russian-msmarco)

```bash
# Rerank documents
curl -X POST http://localhost:8010/inference \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is AI?",
    "docs": [
      "Artificial intelligence involves learning algorithms.",
      "Weather is sunny today."
    ]
  }'
```

## License

MIT
