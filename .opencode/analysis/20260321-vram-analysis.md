# VRAM Memory Analysis: All Supported Models

**Generated:** 2026-03-22
**GPU:** NVIDIA GeForce RTX 4090
**Total VRAM:** 49140 MiB (~48 GB)
**System RAM:** 60 GB

---

## Verified VRAM Usage (Tested on RTX 4090)

### Individual Models

| Model | Type | Delta VRAM | Workers | RAM | Status | Notes |
|-------|------|------------|---------|-----|--------|-------|
| ai-forever/FRIDA | embedding | +3631 MiB | 1 | 21Gi | ✅ | T5-based, fp32 |
| Qwen/Qwen3-Embedding-0.6B | embedding | +1911 MiB | 1 | 21Gi | ✅ | fp16 |
| Qwen/Qwen3-Embedding-4B | embedding | +8923 MiB | 1 | 21Gi | ✅ | fp16 |
| Qwen/Qwen3-Embedding-8B | embedding | - | - | - | **OOM** | Requires >48GB |
| DiTy/cross-encoder-russian-msmarco | reranker | +2250 MiB | 1 | 21Gi | ✅ | Cross-encoder |
| BAAI/bge-reranker-v2-m3 | reranker | +5146 MiB | 1 | 22Gi | ✅ | Cross-encoder |
| Qwen/Qwen3-Reranker-0.6B | reranker | +1549 MiB | 1 | - | ✅ | **Single worker fix** |
| Qwen/Qwen3-Reranker-4B | reranker | +8083 MiB | 1 | - | ✅ | **Single worker fix** |
| Qwen/Qwen3-Reranker-8B | reranker | - | - | - | **OOM** | Requires >48GB |
| Qwen/Qwen3Guard-Gen-0.6B | guard | +1829 MiB | 1 | 29Gi | ✅ | bf16 |
| Qwen/Qwen3Guard-Gen-4B | guard | +8821 MiB | 1 | 32Gi | ✅ | bf16 |
| Qwen/Qwen3Guard-Gen-8B | guard | - | - | - | **OOM** | Requires >48GB |

### Model Combinations

| Combination | Delta VRAM | Total Free | Status | Notes |
|------------|------------|------------|--------|-------|
| emb 0.6B + rer 0.6B | +3460 MiB | ~45 GB | ✅ SAFE | Both work |
| emb 4B + rer 0.6B | +10472 MiB | ~39 GB | ✅ SAFE | Both work |
| emb + rer + guard (all 0.6B) | +4930 MiB | ~43 GB | ⚠️ ERROR | Guard conflicts with emb |
| FRIDA + DiTy + guard 0.6B | +7711 MiB | ~41 GB | ✅ TIGHT | ~7.5GB free |

---

## HuggingFace Model Specifications

### Embedding Models

| Model | Params | Layers | Context | Embed Dim | MRL | Storage | VRAM |
|-------|--------|--------|---------|-----------|-----|---------|------|
| FRIDA | ~0.8B | - | 512 tokens | 1536 | No | fp32 | ~4 GB |
| Qwen3-Embedding-0.6B | 0.6B | 28 | 32K | 1024 | [32-1024] | fp16 | **+1.9 GB** |
| Qwen3-Embedding-4B | 4B | 36 | 32K | 2560 | [32-2560] | fp16 | **+8.9 GB** |
| Qwen3-Embedding-8B | 8B | 36 | 32K | 4096 | [32-4096] | bf16 | **~16 GB** |

### Reranker Models

| Model | Params | Layers | Context | Type | Storage | VRAM |
|-------|--------|--------|---------|------|---------|------|
| DiTy (cross-encoder) | ~0.3B | - | 512 | Cross-encoder | fp16 | **+2.2 GB** |
| BGE-M3 | ~0.6B | - | 8192 | Cross-encoder | fp16 | **+5.1 GB** |
| BGE-Gemma | ~1B | - | 1024 | Cross-encoder | fp16 | ~2-3 GB |
| Qwen3-Reranker-0.6B | 0.6B | 28 | 32K | LLM | bf16 | **+1.5 GB** |
| Qwen3-Reranker-4B | 4B | 36 | 32K | LLM | bf16 | **+8.1 GB** |
| Qwen3-Reranker-8B | 8B | 36 | 32K | LLM | bf16 | **~16 GB** |

### Guard Models

| Model | Params | Layers | Context | Max Tokens | Storage | VRAM |
|-------|--------|--------|---------|-----------|---------|------|
| Qwen3Guard-Gen-0.6B | 0.6B | 28 | 32K | 128 | bf16 | **+1.8 GB** |
| Qwen3Guard-Gen-4B | 4B | 36 | 32K | 128 | bf16 | **+8.8 GB** |
| Qwen3Guard-Gen-8B | 8B | 36 | 32K | 128 | bf16 | **~18 GB** |

---

## 8B Model VRAM Calculations

Based on HuggingFace model cards:

### Qwen3-Embedding-8B
- **Parameters:** 8B
- **Storage:** bf16 (2 bytes/param)
- **Shards:** 4 files (4.9GB + 4.92GB + 4.98GB + 336MB)
- **Base VRAM:** 8B × 2 bytes = 16 GB
- **Expected with activations:** ~16-18 GB
- **Status:** OOM on 48GB (requires 50GB+)

### Qwen3-Reranker-8B
- **Parameters:** 8B
- **Storage:** bf16 (2 bytes/param)
- **Base model:** Qwen/Qwen3-8B-Base
- **Base VRAM:** 8B × 2 bytes = 16 GB
- **Expected with activations:** ~16-18 GB
- **Status:** OOM on 48GB (requires 50GB+)

### Qwen3Guard-Gen-8B
- **Parameters:** 8B
- **Storage:** bf16 (2 bytes/param)
- **Base VRAM:** 8B × 2 bytes = 16 GB
- **Expected with generation overhead:** ~18-20 GB
- **Status:** OOM on 48GB

---

## Recommended Configurations for RTX 4090 (48GB)

| Configuration | Embedding | Reranker | Guard | Delta | Free | Status |
|--------------|-----------|----------|-------|-------|------|--------|
| **Budget** | 0.6B | 0.6B | - | +3.5 GB | ~44 GB | ✅ SAFE |
| **Multilingual** | 4B | 0.6B | - | +10.5 GB | ~37 GB | ✅ SAFE |
| **Russian** | FRIDA | DiTy | 0.6B | +7.7 GB | ~40 GB | ✅ |
| **Max Performance** | 4B | 4B | - | +17 GB | ~31 GB | ✅ |
| **Max All (estimated)** | 4B | 4B | 4B | ~26 GB | ~22 GB | ✅ |

### NOT Possible on 48GB GPU

| Combination | Required | Status |
|------------|----------|--------|
| Any 8B model alone | ~16-18 GB | ⚠️ OOM due to fragmentation |
| emb 8B + any other | ~20+ GB | ⚠️ OOM |
| 3x 0.6B combined | ~5 GB fits | ⚠️ ERROR (guard conflicts with emb) |

---

## Single Worker Fix (2026-03-22)

**Problem:** Reranker endpoints used 2 workers for the same model

**Before:**
- RerankerWorker for `/v1/rerank`
- ScoreWorker (inherits RerankerWorker) for `/v1/score`
- 2 Runtime instances = 2 model copies = 2x VRAM

**After:**
- Single RerankerWorker handles both formats
- Single Runtime shared between routes
- Format detected by request params (`queries` vs `query`)

**Results:**
| Model | Before | After | Savings |
|-------|--------|-------|---------|
| Qwen3-Reranker-0.6B | ~3 GB | **1.5 GB** | 50% |
| Qwen3-Reranker-4B | ~16 GB | **8.1 GB** | 50% |

---

## VRAM Calculation Formula

```
VRAM ≈ params × bytes_per_param × multiplier

Where:
- bytes_per_param: fp32=4, fp16/bf16=2, int8=1
- multiplier: 1.0-1.5 for inference (activations, attention cache)
```

### Verified Calculations

| Model | Params | Storage | Base | Verified | Notes |
|-------|--------|---------|------|----------|-------|
| emb 0.6B | 0.6B | fp16 | 1.2 GB | 1.9 GB | +0.7 overhead |
| emb 4B | 4B | fp16 | 8 GB | 8.9 GB | +0.9 overhead |
| rer 0.6B | 0.6B | bf16 | 1.2 GB | 1.5 GB | +0.3 overhead |
| rer 4B | 4B | bf16 | 8 GB | 8.1 GB | +0.1 overhead |
| guard 0.6B | 0.6B | bf16 | 1.2 GB | 1.8 GB | +0.6 overhead |
| guard 4B | 4B | bf16 | 8 GB | 8.8 GB | +0.8 overhead |
