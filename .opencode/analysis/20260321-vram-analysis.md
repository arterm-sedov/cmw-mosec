# VRAM Memory Analysis: All Supported Models

**Generated:** 2026-03-22
**GPU:** NVIDIA GeForce RTX 4090
**Total VRAM:** 49140 MiB (~48 GB)
**System RAM:** 60 GB

---

## Complete Test Results

### Individual Models

| Model | Type | VRAM Used | Delta | VRAM Free | RAM | Notes |
|-------|------|-----------|-------|-----------|-----|-------|
| ai-forever/FRIDA | embedding | 37726 MiB | +3631 MiB | 10785 MiB | 21Gi | T5-based, fp32 |
| Qwen/Qwen3-Embedding-0.6B | embedding | 36006 MiB | +1911 MiB | 12505 MiB | 21Gi | fp16 |
| Qwen/Qwen3-Embedding-4B | embedding | 43018 MiB | +8923 MiB | 5493 MiB | 21Gi | fp16 |
| Qwen/Qwen3-Embedding-8B | embedding | - | **OOM** | - | - | Requires >50GB |
| DiTy/cross-encoder-russian-msmarco | reranker | 36345 MiB | +2250 MiB | 12166 MiB | 21Gi | Cross-encoder |
| BAAI/bge-reranker-v2-m3 | reranker | 39241 MiB | +5146 MiB | 9270 MiB | 22Gi | Cross-encoder |
| BAAI/bge-reranker-v2-gemma | reranker | - | **FAILED** | - | - | Model not found |
| Qwen/Qwen3-Reranker-0.6B | reranker | 44467 MiB | +10372 MiB | 4044 MiB | 37Gi | Works |
| Qwen/Qwen3-Reranker-4B | reranker | 44467 MiB | +10372 MiB | 4044 MiB | 34Gi | Works |
| Qwen/Qwen3-Reranker-8B | reranker | - | **OOM** | - | - | Requires >50GB |
| Qwen/Qwen3Guard-Gen-0.6B | guard | 35924 MiB | +1829 MiB | 12587 MiB | 29Gi | Works |
| Qwen/Qwen3Guard-Gen-4B | guard | 42916 MiB | +8821 MiB | 5595 MiB | 32Gi | Works |
| Qwen/Qwen3Guard-Gen-8B | guard | - | **OOM** | - | - | Requires >50GB |

### Model Combinations

| Combination | VRAM Used | Delta | VRAM Free | RAM | Notes |
|-------------|-----------|-------|-----------|-----|-------|
| 3x 0.6B (emb+rer+guard) | 37475 MiB | +3380 MiB | 11036 MiB | 32Gi | **SAFE** |
| FRIDA + DiTy + guard 0.6B | 41806 MiB | +7711 MiB | 6705 MiB | 33Gi | **SAFE** |
| emb 4B + 2x 0.6B | 44043 MiB | +9948 MiB | 4468 MiB | 34Gi | **TIGHT** |
| emb 0.6B + 2x 4B | - | **OOM** | - | - | Requires >50GB |
| emb 4B + 2x 4B | - | **OOM** | - | - | Requires >50GB |

---

## VRAM Usage Summary by Category

| Category | 0.6B | 4B | 8B |
|----------|-------|-----|-----|
| **Embedding** | +2 GB | +9 GB | OOM |
| **Reranker (LLM)** | +10 GB | +10 GB | OOM |
| **Reranker (Cross)** | +2-5 GB | N/A | N/A |
| **Guard** | +2 GB | +9 GB | OOM |

---

## Recommended Configurations

### For RTX 4090 (48GB VRAM)

| Configuration | Embedding | Reranker | Guard | Total VRAM | Free |
|--------------|-----------|----------|-------|------------|------|
| **Optimal (Russian)** | FRIDA | DiTy | 0.6B | ~42 GB | ~5 GB |
| **Optimal (Multilingual)** | 4B | 0.6B | 0.6B | ~44 GB | ~5 GB |
| **Budget** | 0.6B | 0.6B | 0.6B | ~37 GB | ~12 GB |

### Not Possible on 48GB GPU

- Any 8B model
- emb 4B + 4B reranker
- emb 4B + 4B guard
- emb 0.6B + 4B reranker + 4B guard

---

## Key Findings

1. **8B models require >50GB VRAM** - All 8B variants (embedding, reranker, guard) OOM on 48GB GPU
2. **Qwen3-Reranker 0.6B/4B uses ~10GB** - More than expected (LLM with generation overhead)
3. **Cross-encoder rerankers are lightweight** - Only 2-5GB
4. **FRIDA (fp32) uses 4GB** - Despite being 0.8B params
5. **BGE-Gemma model not found** - May need to be downloaded separately

---

## Test Script

**Location:** `scripts/test_model_memory.py`

Reads models from `config/models.yaml` and incrementally saves results to `model_memory_results.yaml`.

```bash
# Run tests
.venv/bin/python scripts/test_model_memory.py

# Resume after interruption (automatically skips completed)
.venv/bin/python scripts/test_model_memory.py
```
