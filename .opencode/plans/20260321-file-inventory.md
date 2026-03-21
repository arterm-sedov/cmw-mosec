# File Inventory: March 18-21, 2026

**Generated:** 2026-03-21
**Period:** 2026-03-18 to 2026-03-21
**Status:** VERIFIED - Integration tests passed

This inventory lists all files changed in cmw-mosec during the reranker unification and Qwen3 embedding fixes.

---

## Summary

| Category | Files Changed |
|----------|---------------|
| Core Code | 5 |
| Config | 1 |
| Tests | 2 |
| Documentation | 3 |
| Plans/Reports | 6 |

---

## Core Code Changes

### `cmw_mosec/server_manager.py`
**Commits:** 15+
**Changes:**
- Unified reranker endpoints (`/v1/score`, `/v1/rerank`)
- Client-side formatting for LLM rerankers
- Config-driven `max_length` for all model types
- MRL dimension truncation for Qwen3-Embedding
- Removed hardcoded defaults, all params from config
- Left padding for LLM-based embedders (Qwen3)

### `cmw_mosec/server_config.py`
**Commits:** 1
**Changes:**
- Added `reranker_type` config field
- Added reranker model configuration support

### `cmw_mosec/cli.py`
**Commits:** 3
**Changes:**
- Added `/v1/score` endpoint support
- Updated `check-rerank` for new endpoint contracts
- Added embedding/rerank test commands

### `config/models.yaml`
**Commits:** 5+
**Changes:**
- Added `dimensions`, `max_length` for Qwen3-Embedding models
- Added `max_length` for FRIDA (512 tokens)
- Added `max_length` for Qwen3-Guard models
- Added `reranker_type`, `scoring_method`, `scoring_tokens` for rerankers
- Added `default_instruction` for Qwen3 rerankers
- Added `model_class: T5EncoderModel` for FRIDA

### `pyproject.toml`
**Commits:** 1
**Changes:**
- Integrated reranker tests into pytest

---

## Test Changes

### `tests/test_reranker_endpoints.py`
**Commits:** 5+
**Changes:**
- New test file for reranker endpoint contracts
- Tests for `/v1/score` and `/v1/rerank`
- Score comparison between endpoints
- Cross-encoder and LLM reranker tests

### `tests/fixtures/test_rerankers.yaml`
**Commits:** 3
**Changes:**
- Test harness configuration for rerankers
- Query/document test cases

---

## Documentation Changes

### `README.md`
**Commits:** 4+
**Changes:**
- Added MRL `dimensions` parameter documentation
- Added client-controllable parameters table
- Updated embedding models table (correct FRIDA context)
- Added guard model context/gen tokens columns
- Updated reranker endpoints documentation

### `AGENTS.md`
**Commits:** 2
**Changes:**
- Restored original structure
- Added design principles section

### `docs/` (empty)
**Commits:** 1
**Changes:**
- Created empty docs directory (later removed)

---

## Plans and Reports

### `.opencode/plans/`

| File | Date | Description |
|------|------|-------------|
| `20260320-reranker-unification.md` | Mar 20 | Initial reranker unification plan |
| `20260321-reranker-unification.md` | Mar 21 | Final reranker unification plan |
| `20260321-qwen3-embedding-fix.md` | Mar 21 | Qwen3 embedding fixes (padding, MRL) |
| `fix_duplicate_workers.md` | Mar 19 | Fixed duplicate route workers |
| `mosec_reranker_implementation.md` | Feb 3 | Original reranker implementation |
| `qwen3_reranker_implementation.md` | Feb 3 | Original Qwen3 reranker docs |
| `qwen3guard_implementation.md` | Feb 3 | Original Qwen3Guard docs |
| `session_reranker_refactor_20260321.md` | Mar 21 | Session notes |

### `.opencode/progress_reports/`

| File | Date | Description |
|------|------|-------------|
| `20260319-final-summary.md` | Mar 19 | Reranker final implementation summary |
| `20260319-implementation-fixes.md` | Mar 19 | Implementation fixes details |
| `20260319-reranker-comparison-analysis.md` | Mar 19 | Reranker model comparison |
| `20260319-reranker-models-comparison.md` | Mar 19 | Detailed model comparison |

---

## Key Features Implemented

### 1. Unified Reranker Architecture
- `/v1/score`: vLLM-compatible, returns raw scores
- `/v1/rerank`: Cohere/Jina-compatible, returns sorted results
- Client-side formatting for all reranker types
- Config-driven `reranker_type`, `scoring_method`, `scoring_tokens`

### 2. Qwen3-Embedding Support
- MRL dimension truncation (32 to native)
- Left padding for last_token pooling
- Config-driven `dimensions` and `max_length`
- OpenAI-compatible `dimensions` parameter

### 3. Config-Driven Architecture
- No hardcoded defaults in code
- All parameters from `models.yaml`
- Client-controllable parameters via API

### 4. Model Corrections
- FRIDA: 512 tokens (not 32K)
- Qwen3Guard: `max_length: 32768` in config
- All models: `dtype`, `pooling`, `max_length`, `dimensions` configurable

---

## Related Changes in cmw-rag

| Commit | Description |
|--------|-------------|
| `28bccb3` | Send dimensions parameter to embedding server |
| `2a675d0` | Send dimensions to all endpoints (local and remote) |
| `7691b4b` | Update report with completed dimensions fix |
| `4142f11` | Add model sizing and API parameters analysis |

---

## File Inventory by Date

### March 21, 2026
```
.opencode/plans/20260321-qwen3-embedding-fix.md (updated)
.opencode/plans/20260321-reranker-unification.md (final)
README.md (multiple updates)
cmw_mosec/server_manager.py (multiple updates)
config/models.yaml (multiple updates)
```

### March 20, 2026
```
.opencode/plans/20260320-reranker-unification.md
cmw_mosec/server_manager.py
cmw_mosec/server_config.py
config/models.yaml
```

### March 19, 2026
```
.opencode/progress_reports/20260319-final-summary.md
.opencode/progress_reports/20260319-implementation-fixes.md
.opencode/progress_reports/20260319-reranker-comparison-analysis.md
.opencode/progress_reports/20260319-reranker-models-comparison.md
.opencode/plans/fix_duplicate_workers.md
cmw_mosec/server_manager.py
config/models.yaml
tests/test_reranker_endpoints.py
tests/fixtures/test_rerankers.yaml
```

### March 18, 2026
```
cmw_mosec/server_manager.py
cmw_mosec/server_config.py
config/models.yaml
tests/test_reranker_endpoints.py
pyproject.toml
```

---

## Integration Verification

### Test Results (2026-03-21)

**CMW-MOSEC Tests:**
```
tests/test_server_config.py: 27 passed in 0.09s
```

**CMW-RAG Tests:**
```
rag_engine/tests/test_retrieval_embedder.py: 2 skipped (requires model download)
```

**Integration Tests:**

| Test | Provider | Status |
|------|----------|--------|
| Dimensions parameter | mosec local | ✓ Pass |
| MRL truncation (512) | mosec local | ✓ Pass |
| Native dimension (1024) | mosec local | ✓ Pass |
| Dimension mismatch (4096 > 1024) | mosec local | ✓ Pass (400 error) |
| CMW-RAG embedder | openrouter | ✓ Pass |

**Key Verifications:**
1. `dimensions` parameter sent from cmw-rag to all endpoints ✓
2. Mosec validates dimensions against model max ✓
3. Mosec truncates to requested dimension ✓
4. CMW-RAG config dimensions match server response ✓

---

## Related Changes in CMW-RAG

| Commit | Description |
|--------|-------------|
| `28bccb3` | Send dimensions parameter to embedding server |
| `2a675d0` | Send dimensions to all endpoints (local and remote) |
| `7691b4b` | Update report with completed dimensions fix |
| `4142f11` | Add model sizing and API parameters analysis |