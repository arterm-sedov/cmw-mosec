# File Inventory: March 18-22, 2026

**Generated:** 2026-03-22
**Period:** 2026-03-18 to 2026-03-22

This inventory lists all files changed in cmw-mosec.

---

## Core Code Changes

| File | Changes |
|------|---------|
| `cmw_mosec/server_manager.py` | Unified reranker endpoints, single worker fix |
| `cmw_mosec/server_config.py` | Added reranker_type config field |
| `cmw_mosec/cli.py` | Added /v1/score endpoint support |
| `config/models.yaml` | Added dimensions, max_length, reranker config |

## Tests

| File | Description |
|------|-------------|
| `tests/test_reranker_endpoints.py` | Reranker endpoint contracts |
| `tests/test_cpu_integration.py` | Combined server integration tests |
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
| `.opencode/plans/20260321-reranker-unification.md` | Mar 21 | Final reranker plan |
| `.opencode/plans/20260321-qwen3-embedding-fix.md` | Mar 21 | Embedding fixes |
| `.opencode/analysis/20260321-vram-analysis.md` | Mar 22 | VRAM memory analysis |
| `.opencode/progress_reports/20260319-*.md` | Mar 19 | Implementation reports |

---

## Analysis Reports

See `.opencode/analysis/20260321-vram-analysis.md` for complete VRAM findings.
