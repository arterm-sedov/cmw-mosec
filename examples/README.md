# CMW Mosec Examples

This directory contains practical examples for using CMW Mosec with different embedding models.

## Available Examples

### 1. Qwen3 Embedding Examples (`qwen3_embedding_examples.py`)

Demonstrates proper usage of Qwen3 embedding models (Qwen3-Embedding-0.6B/4B/8B).

**Key Concepts:**
- Instruction format for queries (required per HF docs)
- Document format (no instruction needed)
- Multilingual support (119+ languages)
- Performance comparison: wrong vs right format

**Run:**
```bash
# Start server with Qwen3 embedding
cmw-mosec serve --embedding Qwen/Qwen3-Embedding-0.6B

# Run examples
python examples/qwen3_embedding_examples.py
```

**What You'll Learn:**
- How to format queries with instructions
- Why instruction format matters (~15% accuracy improvement)
- Cross-lingual retrieval capabilities
- Batch processing for efficiency

### 2. FRIDA Embedding Examples (`frida_embedding_examples.py`)

Demonstrates proper usage of FRIDA (T5-based Russian embedding model).

**Key Concepts:**
- `search_query:` prefix for queries
- `search_document:` prefix for documents
- Russian language optimization
- Impact of prefixes on accuracy

**Run:**
```bash
# Start server with FRIDA
cmw-mosec serve --embedding ai-forever/FRIDA

# Run examples
python examples/frida_embedding_examples.py
```

**What You'll Learn:**
- How to use task-specific prefixes
- Why prefixes are required for T5 models
- Russian language capabilities
- Cross-lingual retrieval with FRIDA

## Quick Reference

### Qwen3 Instruction Format

```python
def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

# Usage
task = 'Given a web search query, retrieve relevant passages that answer the query'
query = get_detailed_instruct(task, 'What is Python?')
# Result: 'Instruct: Given a web search query...\nQuery: What is Python?'
```

### FRIDA Prefix Format

```python
# Query
query = "search_query: " + user_query

# Document
doc = "search_document: " + document_text
```

### API Call Examples

**Qwen3:**
```bash
curl -X POST http://localhost:8001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Embedding-0.6B",
    "input": "Instruct: Given a web search query...\nQuery: What is AI?"
  }'
```

**FRIDA:**
```bash
curl -X POST http://localhost:8001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ai-forever/FRIDA",
    "input": "search_query: What is AI?"
  }'
```

## Requirements

- Mosec server running with appropriate model
- Python 3.8+
- `requests` and `numpy` packages

## Model Selection Guide

| Use Case | Recommended Model | Why |
|----------|------------------|-----|
| Multilingual (119+ langs) | Qwen3-Embedding-0.6B | Best multilingual performance |
| Russian-focused | FRIDA | Optimized for Russian |
| High accuracy | Qwen3-Embedding-4B/8B | Larger models = better quality |
| Low VRAM | Qwen3-Embedding-0.6B | Only ~2GB |

## Common Mistakes

### Qwen3

❌ **Wrong:** Sending raw query without instruction
```python
query = "What is Python?"  # Missing instruction!
```

✅ **Correct:** Using instruction format
```python
query = "Instruct: Given a web search query...\nQuery: What is Python?"
```

### FRIDA

❌ **Wrong:** Sending text without prefixes
```python
query = "Что такое Python?"  # Missing prefix!
```

✅ **Correct:** Using search_query: prefix
```python
query = "search_query: Что такое Python?"
```

## Troubleshooting

### "Cannot connect to Mosec server"

**Solution:** Start the server first:
```bash
cmw-mosec serve --embedding Qwen/Qwen3-Embedding-0.6B
```

### "Low similarity scores"

**Qwen3:** Make sure you're using instruction format for queries

**FRIDA:** Make sure you're using `search_query:` and `search_document:` prefixes

### "Model not found"

**Solution:** Check available models:
```bash
cmw-mosec list
```

## References

- [Qwen3-Embedding-0.6B on HuggingFace](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- [FRIDA on HuggingFace](https://huggingface.co/ai-forever/FRIDA)
- [CMW Mosec README](../README.md)
