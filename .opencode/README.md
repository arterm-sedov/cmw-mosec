# CMW-Mosec Project Index

**Generated:** 2026-03-22

All plans, reports, analysis, code, tests, and configuration files in cmw-mosec.

---

## Plans

| File | Date | Description |
|------|------|-------------|
| <plans/20260322-dynamic-server-script.md> | Mar 22 | Replace script generation with dynamic runtime server configuration |
| <plans/20260321-reranker-unification.md> | Mar 21 | Final reranker plan: separate workers, vLLM `queries` param (COMPLETE, supersedes Mar 20) |
| <plans/20260321-qwen3-embedding-fix.md> | Mar 21 | Qwen3-Embedding inference error investigation and fix (COMPLETE) |
| <plans/20260320-reranker-unification.md> | Mar 20 | Earlier reranker unification plan (superseded by Mar 21 version) |
| <plans/session_reranker_refactor_20260321.md> | Mar 18–21 | Session log for reranker endpoint refactoring and testing |
| <plans/fix_duplicate_workers.md> | Mar 19 | Fix duplicate GPU workers reducing memory usage by ~50% |
| <plans/qwen3_reranker_implementation.md> | Feb 3 | Qwen3-Reranker-0.6B MOSEC server: CausalLM reranker, 100+ languages, 32K context |
| <plans/qwen3guard_implementation.md> | Feb 3 | Qwen3Guard-Gen-0.6B content safety moderation with three-tier severity classification |
| <plans/mosec_reranker_implementation.md> | Feb 3 | Production MOSEC reranker for DiTy Russian cross-encoder with OpenVINO acceleration |

## Progress Reports

| File | Date | Description |
|------|------|-------------|
| <progress_reports/20260322-embedding-bf16-fix-per-model-dtype.md> | Mar 22 | Fix BFloat16 error in Qwen3-Embedding, add per-model dtype support |
| <progress_reports/20260319-final-summary.md> | Mar 19 | Summary: Qwen3-Reranker support, configurable max_length, instruction handling |
| <progress_reports/20260319-implementation-fixes.md> | Mar 19 | Fix Qwen3-Reranker inference errors and padding token issues |
| <progress_reports/20260319-reranker-models-comparison.md> | Mar 19 | Comparison: DiTy, Qwen3-Reranker-0.6B, BGE-reranker-v2-m3 |
| <progress_reports/20260319-reranker-comparison-analysis.md> | Mar 19 | Detailed comparative analysis with test datasets |

## Analysis

| File | Date | Description |
|------|------|-------------|
| <analysis/20260321-vram-analysis.md> | Mar 22 | VRAM memory analysis for all supported models on RTX 4090 (48GB) |

---

## Core Code (`cmw_mosec/`)

| File | Description |
|------|-------------|
| <../cmw_mosec/__init__.py> | Package exports: `ModelRegistry`, `MosecModelConfig`, `MosecServerManager` |
| <../cmw_mosec/server_config.py> | Pydantic schemas, model registry, YAML config loading, `.env` settings |
| <../cmw_mosec/server_manager.py> | Process management: start, stop, health checks, PID files |
| <../cmw_mosec/cli.py> | Click CLI: `start`, `stop`, `status`, `models`, `check-rerank` |

### v2 Dynamic Server (`cmw_mosec/v2/`)

| File | Description |
|------|-------------|
| <../cmw_mosec/v2/__init__.py> | Exports `run_server` for dynamic v2 server |
| <../cmw_mosec/v2/dynamic_server.py> | Dynamic server entry point (Mosec standard pattern, no script generation) |
| <../cmw_mosec/v2/workers.py> | Worker classes: `EmbeddingWorkerV2`, `RerankWorkerV2`, `GuardWorkerV2`, `ScoreWorkerV2` |

## Tests (`tests/`)

| File | Description |
|------|-------------|
| <../tests/test_reranker_endpoints.py> | Reranker endpoint contracts: `/v1/score` (vLLM) and `/v1/rerank` (Cohere) |
| <../tests/test_cpu_integration.py> | Integration tests with real inference (embedding, reranker, guard) |
| <../tests/test_server_config.py> | Unit tests for config, model registry, and guard parsing |
| <../tests/test_reranker_only.py> | Standalone test for DiTy cross-encoder reranker |
| <../tests/test_config.yaml> | Test model configuration (models for combined/guard server fixtures) |
| <../tests/fixtures/test_rerankers.yaml> | Test harness config: test cases, formatting templates, endpoint contracts |

## Configuration

| File | Description |
|------|-------------|
| <../config/models.yaml> | All model specs: embedding, reranker, guard (dimensions, max_length, pooling, dtype) |
| <../.env-example> | Example environment config with active model list and server settings |
| <../.env.example> | Alternative example environment config |
| <../pyproject.toml> | Project metadata, dependencies, build config |
| <../model_memory_results.yaml> | GPU memory benchmark results for all models (RTX 4090, 2026-03-22) |

## Scripts & Examples

| File | Description |
|------|-------------|
| <../scripts/test_model_memory.py> | Measure VRAM/RAM usage for all supported models |
| <../scripts/test_configs/model_memory_tests.yaml> | Model combination test scenarios for memory benchmarking |
| <../examples/frida_embedding_examples.py> | FRIDA (T5-based) embedding usage examples with query/document prefixes |
| <../examples/qwen3_embedding_examples.py> | Qwen3 embedding examples with instruction format |
| <../examples/README.md> | Examples directory documentation |

## Root Documentation

| File | Description |
|------|-------------|
| <../README.md> | Project documentation: features, API, model tables, CLI reference |
| <../AGENTS.md> | AI agent guide: architecture, conventions, testing practices |
| <../UPDATE_SUMMARY.md> | Qwen3 Embedding support update summary (2026-02-20) |
| <../session-ses_397c.md> | Session log: MOSec server status check (2026-02-16) |
