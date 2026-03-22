# File Inventory: March 18-22, 2026

**Generated:** 2026-03-22
**Updated:** 2026-03-22

This inventory lists all files changed in cmw-mosec.

---

## Core Code Changes

| File | Changes |
|------|---------|
| `cmw_mosec/server_manager.py` | Separate ScoreWorker/RerankWorker, cross-encoder max_length fix |
| `cmw_mosec/server_config.py` | Added reranker_type config field |
| `cmw_mosec/cli.py` | Added /v1/score endpoint support |
| `config/models.yaml` | Added dimensions, max_length, reranker config |

## Tests

| File | Description |
|------|-------------|
| `tests/test_reranker_endpoints.py` | Reranker endpoint contracts, vLLM `queries` param |
| `tests/test_cpu_integration.py` | Server integration tests |
| `tests/test_config.yaml` | Test model configuration |
| `tests/fixtures/test_rankers.yaml` | Test harness configuration |

## Configuration

| File | Description |
|------|-------------|
| `.env` | Active models, server settings |
| `config/models.yaml` | All model specifications |

## Documentation

| File | Description |
|------|-------------|
| `README.md` | API documentation, model tables |
| `AGENTS.md` | Agent guidance |

## Plans and Reports

| File | Date | Description |
|------|------|-------------|
| `.opencode/plans/20260320-reranker-unification.md` | Mar 20 | Reranker unification plan |
| `.opencode/plans/20260321-reranker-unification.md` | Mar 22 | Final reranker plan (separate workers, vLLM `queries`) |
| `.opencode/plans/20260321-qwen3-embedding-fix.md` | Mar 21 | Embedding fixes |
| `.opencode/analysis/20260321-vram-analysis.md` | Mar 22 | VRAM memory analysis |
| `.opencode/progress_reports/20260319-*.md` | Mar 19 | Implementation reports |
| `.opencode/progress_reports/20260322-embedding-bf16-fix-per-model-dtype.md` | Mar 22 | BF16 numpy fix, per-model dtype |
