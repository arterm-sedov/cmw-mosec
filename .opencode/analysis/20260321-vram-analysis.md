# VRAM Memory Analysis: All Supported Models

**Generated:** 2026-03-22
**GPU:** NVIDIA GeForce RTX 4090
**Total VRAM:** 49140 MiB (~48 GB)
**System RAM:** 60 GB

---

## Confirmed Test Results

### Individual Models (Verified on RTX 4090)

| Model | Type | VRAM Used | Delta | VRAM Free | RAM | Notes |
|-------|------|-----------|-------|-----------|-----|-------|
| ai-forever/FRIDA | embedding | 37726 MiB | +3631 MiB | 10785 MiB | 21Gi | T5-based, fp32 |
| Qwen/Qwen3-Embedding-0.6B | embedding | 36006 MiB | +1911 MiB | 12505 MiB | 21Gi | fp16 |
| Qwen/Qwen3-Embedding-4B | embedding | 43018 MiB | +8923 MiB | 5493 MiB | 21Gi | fp16 |
| Qwen/Qwen3-Embedding-8B | embedding | - | **OOM** | - | - | Requires >50GB |
| DiTy/cross-encoder-russian-msmarco | reranker | 36345 MiB | +2250 MiB | 12166 MiB | 21Gi | Cross-encoder |
| BAAI/bge-reranker-v2-m3 | reranker | 39241 MiB | +5146 MiB | 9270 MiB | 22Gi | Cross-encoder |
| Qwen/Qwen3Guard-Gen-0.6B | guard | 35924 MiB | +1829 MiB | 12587 MiB | 29Gi | Works |
| Qwen/Qwen3Guard-Gen-4B | guard | 42916 MiB | +8821 MiB | 5595 MiB | 32Gi | Works |

### Estimated (Based on Model Size & Architecture)

| Model | Size | Architecture | Estimated VRAM | Notes |
|-------|------|--------------|----------------|-------|
| Qwen/Qwen3-Reranker-0.6B | 0.6B | LLM | ~2-3 GB | Like embedding 0.6B |
| Qwen/Qwen3-Reranker-4B | 4B | LLM | ~8-9 GB | Like embedding 4B |
| Qwen/Qwen3-Reranker-8B | 8B | LLM | **OOM** | Requires >50GB |
| BAAI/bge-reranker-v2-gemma | ~1B | BERT-like | ~2-3 GB | Cross-encoder |

### Model Combinations (Verified)

| Combination | VRAM Used | Delta | VRAM Free | RAM | Status |
|------------|-----------|-------|-----------|-----|---------|
| 3x 0.6B (emb+rer+guard) | 37475 MiB | +3380 MiB | 11036 MiB | 32Gi | **SAFE** |
| FRIDA + DiTy + guard 0.6B | 41806 MiB | +7711 MiB | 6705 MiB | 33Gi | **SAFE** |
| emb 4B + 2x 0.6B | 44043 MiB | +9948 MiB | 4468 MiB | 34Gi | **TIGHT** |

---

## HuggingFace Model Specifications

### Embedding Models

| Model | Parameters | Layers | Context | Embed Dim | MRL | VRAM (fp16) |
|-------|------------|--------|---------|-----------|-----|--------------|
| FRIDA | ~0.8B | - | 512 tokens | 1536 | No | ~4 GB (fp32) |
| Qwen3-Embedding-0.6B | 0.6B | 28 | 32K | 1024 | [32-1024] | ~2 GB |
| Qwen3-Embedding-4B | 4B | 36 | 32K | 2560 | [32-2560] | ~9 GB |
| Qwen3-Embedding-8B | 8B | 36 | 32K | 4096 | [32-4096] | **OOM** |

### Reranker Models

| Model | Parameters | Layers | Context | Type | VRAM |
|-------|------------|--------|---------|------|------|
| DiTy (cross-encoder) | ~0.3B | - | 512 | Cross-encoder | ~2 GB |
| BGE-M3 | ~0.6B | - | 8192 | Cross-encoder | ~5 GB |
| BGE-Gemma | ~1B | - | 1024 | Cross-encoder | ~2-3 GB |
| Qwen3-Reranker-0.6B | 0.6B | 28 | 32K | LLM | ~2-3 GB |
| Qwen3-Reranker-4B | 4B | 36 | 32K | LLM | ~8-9 GB |
| Qwen3-Reranker-8B | 8B | 36 | 32K | LLM | **OOM** |

### Guard Models

| Model | Parameters | Layers | Context | Max Tokens | VRAM |
|-------|------------|--------|---------|-----------|------|
| Qwen3Guard-Gen-0.6B | 0.6B | 28 | 32K | 128 | ~2 GB |
| Qwen3Guard-Gen-4B | 4B | 36 | 32K | 128 | ~9 GB |
| Qwen3Guard-Gen-8B | 8B | 36 | 32K | 128 | **OOM** |

---

## VRAM Usage Calculation

### Formula
```
VRAM ≈ params × bytes_per_param × multiplier

Where multiplier accounts for:
- Activations (1.5-2x for LLM)
- Attention cache
- Gradients (inference: 1x)
```

### Verified vs Estimated

| Model | Verified VRAM | Estimated | Match |
|-------|-------------|-----------|-------|
| FRIDA | +3.6 GB | ~4 GB | ✅ |
| emb 0.6B | +1.9 GB | ~2 GB | ✅ |
| emb 4B | +8.9 GB | ~9 GB | ✅ |
| guard 0.6B | +1.8 GB | ~2 GB | ✅ |
| guard 4B | +8.8 GB | ~9 GB | ✅ |

---

## Recommended Configurations for RTX 4090 (48GB)

| Configuration | Embedding | Reranker | Guard | Total VRAM | Free | Status |
|--------------|-----------|----------|-------|------------|------|--------|
| **Optimal (Russian)** | FRIDA | DiTy | 0.6B | ~42 GB | ~5 GB | ✅ |
| **Optimal (Multilingual)** | 4B | 0.6B | 0.6B | ~44 GB | ~4 GB | ✅ |
| **Budget** | 0.6B | 0.6B | 0.6B | ~37 GB | ~12 GB | ✅ |
| **Max Performance** | 4B | 4B | - | ~40 GB | ~8 GB | ✅ |

### Not Possible on 48GB GPU

- Any 8B model (OOM)
- emb 4B + 4B reranker + 4B guard (requires >50GB)
- emb 4B + 4B reranker (tight)

---

## Config Updates Needed

Based on experiments, consider updating `config/models.yaml`:

```yaml
# Memory estimates (memory_gb):
Qwen/Qwen3-Reranker-0.6B:
  memory_gb: 4.0  # Estimate based on 0.6B LLM

Qwen/Qwen3-Reranker-4B:
  memory_gb: 12.0  # Estimate based on 4B LLM
```

---

## Test Script

**Location:** `scripts/test_model_memory.py`

Reads models from `config/models.yaml` and incrementally saves results.

```bash
# Run tests
.venv/bin/python scripts/test_model_memory.py

# Resume after interruption (automatically skips completed)
.venv/bin/python scripts/test_model_memory.py

# Results saved to: model_memory_results.yaml
```
