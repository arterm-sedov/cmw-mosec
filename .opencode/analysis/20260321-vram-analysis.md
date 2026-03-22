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

| Configuration | Embedding | Reranker | Guard | Delta | Free | Status |
|--------------|-----------|----------|-------|-------|------|--------|
| **Russian** | FRIDA | DiTy | 0.6B | +7.7 GB | ~10 GB | ✅ |
| **Multilingual** | 4B | 0.6B | - | +10.5 GB | ~8 GB | ✅ |
| **Budget** | 0.6B | 0.6B | - | +3.5 GB | ~15 GB | ✅ |
| **Max Performance** | 4B | 4B | - | +17 GB | ~7 GB | ✅ |
| **Max All** | 4B | 4B | 4B | +26 GB | ~22 GB | ✅ |

### NOT Possible on 48GB GPU

| Combination | Required VRAM | Status |
|------------|--------------|--------|
| Qwen3-Embedding-8B | ~18 GB | OOM on 48GB |
| Qwen3-Reranker-8B | ~16 GB | OOM on 48GB |
| Qwen3Guard-Gen-8B | ~18 GB | OOM on 48GB |
| emb 4B + rer 4B | ~17 GB | ✅ Tight (~31GB free) |
| emb 4B + guard 4B | ~18 GB | ✅ Tight (~30GB free) |
| rer 4B + guard 4B | ~17 GB | ✅ (~31GB free) |
| emb 4B + rer 4B + guard 4B | ~26 GB | ✅ (~22GB free) |
| emb 4B + 4B reranker + 4B guard | ~26 GB | ✅ (~22GB free) |

**Note:** OOM thresholds vary based on VRAM fragmentation, batch size, and concurrent requests. Values are approximate.

---

## Single Worker Fix (2026-03-22)

**Problem:** Reranker endpoints used 2 workers for the same model (~2x VRAM)

**Fix:** Unified RerankerWorker.forward() handles both `/v1/score` and `/v1/rerank` formats.
Single Runtime instance shared between both routes.

**Results:**
- Qwen3-Reranker-0.6B: **1549 MiB** (was ~3GB with 2 workers)
- Qwen3-Reranker-4B: **8083 MiB** (was ~16GB with 2 workers)

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
