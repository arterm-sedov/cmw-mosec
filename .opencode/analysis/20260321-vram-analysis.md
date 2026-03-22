# VRAM Memory Analysis: All Supported Models

**Generated:** 2026-03-22
**GPU:** NVIDIA GeForce RTX 4090
**Total VRAM:** 49140 MiB (~48 GB)
**System RAM:** 60 GB

---

## Confirmed Test Results

### Individual Models (Verified on RTX 4090)

| Model | Type | VRAM Used | Delta | Workers | RAM | Notes |
|-------|------|-----------|-------|---------|-----|-------|
| ai-forever/FRIDA | embedding | 37726 MiB | +3631 MiB | 1 | 21Gi | T5-based, fp32 |
| Qwen/Qwen3-Embedding-0.6B | embedding | 36006 MiB | +1911 MiB | 1 | 21Gi | fp16 |
| Qwen/Qwen3-Embedding-4B | embedding | 43018 MiB | +8923 MiB | 1 | 21Gi | fp16 |
| Qwen/Qwen3-Embedding-8B | embedding | - | **OOM** | - | - | Requires >50GB |
| DiTy/cross-encoder-russian-msmarco | reranker | 36345 MiB | +2250 MiB | 1 | 21Gi | Cross-encoder |
| BAAI/bge-reranker-v2-m3 | reranker | 39241 MiB | +5146 MiB | 1 | 22Gi | Cross-encoder |
| Qwen/Qwen3-Reranker-0.6B | reranker | 35644 MiB | +1549 MiB | 1 | - | **LLM, single worker** |
| Qwen/Qwen3-Reranker-4B | reranker | 42178 MiB | +8083 MiB | 1 | - | **LLM, single worker** |
| Qwen/Qwen3-Reranker-8B | reranker | - | **OOM** | - | - | Requires >50GB |
| Qwen/Qwen3Guard-Gen-0.6B | guard | 35924 MiB | +1829 MiB | 1 | 29Gi | Works |
| Qwen/Qwen3Guard-Gen-4B | guard | 42916 MiB | +8821 MiB | 1 | 32Gi | Works |

### Model Combinations (Verified)

| Combination | VRAM Used | Delta | Workers | Status | Notes |
|------------|-----------|-------|---------|--------|-------|
| 3x 0.6B (emb+rer+guard) | 39025 MiB | +4930 MiB | 3 | **ERROR** | Guard+emb OOM together |
| emb 0.6B + rer 0.6B | 37555 MiB | +3460 MiB | 2 | ✅ SAFE | Both work |
| emb 4B + rer 0.6B | 44567 MiB | +10472 MiB | 2 | ✅ SAFE | Both work |
| FRIDA + DiTy + guard 0.6B | 41806 MiB | +7711 MiB | 3 | **TIGHT** | ~4.5GB free |

---

## HuggingFace Model Specifications

### Embedding Models

| Model | Parameters | Layers | Context | Embed Dim | MRL | Stored | Expected VRAM |
|-------|------------|--------|---------|-----------|-----|--------|---------------|
| FRIDA | ~0.8B | - | 512 tokens | 1536 | No | fp32 | ~4 GB |
| Qwen3-Embedding-0.6B | 0.6B | 28 | 32K | 1024 | [32-1024] | fp16 | ~2 GB |
| Qwen3-Embedding-4B | 4B | 36 | 32K | 2560 | [32-2560] | fp16 | ~9 GB |
| Qwen3-Embedding-8B | 8B | 36 | 32K | 4096 | [32-4096] | bf16 | ~18 GB |

### Reranker Models

| Model | Parameters | Layers | Context | Type | Stored | Expected VRAM |
|-------|------------|--------|---------|------|--------|---------------|
| DiTy (cross-encoder) | ~0.3B | - | 512 | Cross-encoder | fp16 | ~2 GB |
| BGE-M3 | ~0.6B | - | 8192 | Cross-encoder | fp16 | ~5 GB |
| BGE-Gemma | ~1B | - | 1024 | Cross-encoder | fp16 | ~2-3 GB |
| Qwen3-Reranker-0.6B | 0.6B | 28 | 32K | LLM | bf16 | ~1.5 GB |
| Qwen3-Reranker-4B | 4B | 36 | 32K | LLM | bf16 | ~8 GB |
| Qwen3-Reranker-8B | 8B | 36 | 32K | LLM | bf16 | ~16 GB |

### Guard Models

| Model | Parameters | Layers | Context | Max Tokens | Stored | Expected VRAM |
|-------|------------|--------|---------|-----------|--------|---------------|
| Qwen3Guard-Gen-0.6B | 0.6B | 28 | 32K | 128 | bf16 | ~1.8 GB |
| Qwen3Guard-Gen-4B | 4B | 36 | 32K | 128 | bf16 | ~9 GB |
| Qwen3Guard-Gen-8B | 8B | 36 | 32K | 128 | bf16 | ~18 GB |

---

## VRAM Usage Summary

### All Models - Verified on RTX 4090 (48GB)

| Model | Delta | Workers | Status |
|-------|-------|---------|--------|
| **Embeddings** | | | |
| Qwen3-Embedding-0.6B | +1.9 GB | 1 | ✅ |
| Qwen3-Embedding-4B | +8.9 GB | 1 | ✅ |
| Qwen3-Embedding-8B | - | - | **OOM** |
| **Rerankers (Cross-Encoder)** | | | |
| DiTy cross-encoder | +2.2 GB | 1 | ✅ |
| BGE-reranker-v2-m3 | +5.1 GB | 1 | ✅ |
| **Rerankers (LLM)** | | | |
| Qwen3-Reranker-0.6B | +1.5 GB | 1 | ✅ |
| Qwen3-Reranker-4B | +8.1 GB | 1 | ✅ |
| Qwen3-Reranker-8B | - | - | **OOM** |
| **Guards** | | | |
| Qwen3Guard-Gen-0.6B | +1.8 GB | 1 | ✅ |
| Qwen3Guard-Gen-4B | +8.8 GB | 1 | ✅ |
| Qwen3Guard-Gen-8B | - | - | **OOM** |

### Combinations Tested

| Combination | Delta | Status |
|------------|-------|--------|
| emb 0.6B + rer 0.6B | +3.5 GB | ✅ SAFE |
| emb 4B + rer 0.6B | +10.5 GB | ✅ SAFE |
| emb + rer + guard (all 0.6B) | +4.9 GB | ⚠️ ERROR (guard conflicts with emb) |
| FRIDA + DiTy + guard 0.6B | +7.7 GB | ✅ TIGHT (~4.5GB free) |

---

## 8B Model VRAM Estimates

Based on HuggingFace model cards (stored as bf16):

| Model | Parameters | Expected VRAM | Notes |
|-------|------------|--------------|-------|
| Qwen3-Embedding-8B | 8B | ~16 GB | 4 shard files |
| Qwen3-Reranker-8B | 8B | ~16 GB | Finetuned from Qwen3-8B-Base |
| Qwen3Guard-Gen-8B | 8B | ~18 GB | Higher due to generation |

---

## OOM and Memory Fragmentation

**OOM Thresholds Vary:**
- Raw model VRAM + activations + overhead = actual usage
- VRAM fragmentation from previous allocations can cause OOM even when total fits
- PyTorch/CUDA memory allocator may not release fragmented memory quickly

**Mitigation:**
- Load models in consistent order (largest first)
- Avoid loading/unloading models rapidly
- Use `torch.cuda.empty_cache()` between model switches

---

## Single Worker Fix (2026-03-22)

**Problem:** Reranker endpoints used 2 workers for the same model (~2x VRAM)

**Fix:** Unified RerankerWorker.forward() handles both `/v1/score` and `/v1/rerank` formats.
Single Runtime instance shared between both routes.

**Results:**
- Qwen3-Reranker-0.6B: **1549 MiB** (was ~3GB with 2 workers)
- Qwen3-Reranker-4B: **8083 MiB** (was ~16GB with 2 workers)

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
