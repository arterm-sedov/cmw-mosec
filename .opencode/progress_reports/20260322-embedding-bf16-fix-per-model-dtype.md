# Embedding BF16 Fix & Per-Model DType - March 22, 2026

## Problem

`cmw-rag` `build_index.py` failed with 500 Internal Server Error on `/v1/embeddings` endpoint when using `Qwen/Qwen3-Embedding-0.6B`. The error was:

```
TypeError: Got unsupported ScalarType BFloat16
```

## Root Cause

`EmbeddingWorkerV2.forward()` called `embeddings.numpy()` directly on bf16 tensors. PyTorch's `.numpy()` does not support bf16 dtype — requires conversion to float32 first.

## Changes

### `cmw_mosec/v2/workers.py`

1. **Fix bf16→numpy conversion** (line 180):
   ```python
   # Before
   embeddings = embeddings.numpy()
   # After
   embeddings = embeddings.float().numpy()
   ```

2. **Per-model dtype for EmbeddingWorkerV2** (line 91):
   ```python
   # Before
   self.dtype = os.getenv("DTYPE", "float32")
   # After
   self.dtype = config.dtype
   ```

3. **Per-model dtype for GuardWorkerV2** (line 373):
   ```python
   # Before
   self.dtype = os.getenv("DTYPE", "float32")
   # After
   self.dtype = config.dtype
   ```

### `cmw_mosec/server_config.py`

Removed unused `ServerSettings` fields and their validators:
- `dtype` — workers now read from per-model config (`config/models.yaml`)
- `batch_size` — per-model only (in models.yaml)
- `idle_timeout` — never referenced
- `log_level` — never referenced

`load_server_settings()` now only requires `SERVER_PORT` and `HF_TOKEN`.

### `config/models.yaml`

Reduced `Qwen/Qwen3-Embedding-0.6B` batch_size from 16 to 8. After OOM, worker enters bad state and subsequent requests fail with 500 even for single items. Smaller batches reduce OOM risk with ~13GB free VRAM.

## Verification

All embedding models that fit in ~13GB free VRAM tested successfully:

| Model | Status | Dims |
|-------|--------|------|
| `ai-forever/FRIDA` | ✓ | 1536 |
| `Qwen/Qwen3-Embedding-0.6B` | ✓ | 1024 |
| `Qwen/Qwen3-Embedding-4B` | ✓ | 2560 |
| `Qwen/Qwen3-Embedding-8B` | skip (22GB VRAM) | — |

## Design Notes

- `max_length` in config is a tokenizer truncation cap, not pre-allocated memory
- Dtype is now purely per-model from `config/models.yaml` (e.g. FRIDA=float32, Qwen=float16, Guard=bf16)
- The global `DTYPE` env var is no longer used by workers
