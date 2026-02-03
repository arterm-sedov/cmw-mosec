# Agent Guide for cmw-mosec

This document provides guidance for AI agents working on the cmw-mosec project. Rule set for opencode: keep solutions lean; do not overengineer.

## Git & Commits

- **Do NOT create or push commits automatically.** The user reviews all commits first. You may suggest commit messages or stage files only when explicitly asked.
- If generating a commit message: keep it concise, structured, and strictly relevant to the changes. Do not add, stage, or push.

## Project Overview

cmw-mosec is a CLI tool for managing Mosec embedding and reranker servers. It provides:
- Server lifecycle management (start, stop, status)
- Pre-configured model definitions
- Process management with PID files
- Health checking

## Architecture

```
cmw_mosec/
├── __init__.py          # Package exports
├── cli.py              # Click CLI commands
├── server_config.py    # Pydantic schemas and model definitions
└── server_manager.py   # Process management
```

## Key Components

### MosecModelConfig (Pydantic)
Defines model configurations including:
- model_id: HuggingFace model identifier
- port: Server port (unique per model)
- memory_gb: Estimated VRAM usage
- dtype: Data type (float16, float32, int8)
- batch_size: Dynamic batching size
- workers: Number of Mosec workers

### MosecServerManager
Manages Mosec server processes:
- start(): Launch server in background/foreground
- stop(): Graceful shutdown with fallback to force kill
- get_status(): Check if server is running and responding
- list_running(): List all servers with PID files

### CLI Commands
- setup: Verify dependencies
- start <model>: Start server for model
- stop <model>: Stop server
- status: Show running servers
- list: Show available models

## Dependencies

Core:
- click: CLI framework
- pydantic: Data validation
- requests: HTTP health checks

External (user-installed):
- mosec: The server framework
- torch: For GPU detection
- transformers: For model loading
- sentence-transformers: For reranker models
- llmspec: OpenAI-compatible API schemas

## Environment Setup

- **Linux (native):** `source .venv/bin/activate`
- **Install Dependencies:** `pip install -e .`

## Build, Lint & Test

### Testing
- **Run all tests:** `pytest`
- **Run specific test file:** `pytest tests/test_server_config.py`
- **Run with coverage:** `pytest --cov=cmw_mosec --cov-report=term-missing`

### Linting
- **Lint modified files:** `ruff check <modified_file>`
- **Auto-fix issues:** `ruff check --fix <modified_file>`

## Test Practices

Following industry best practices from Google Test Primer and IBM Unit Testing Guidelines:

### Test Behavior, Not Implementation

**Core Principle:** Tests should validate what code **does**, not **how** it does it.

**BAD - Testing implementation details:**
```python
def test_server_port():
    config = get_model_config("ai-forever/FRIDA")
    assert config.port == 8001  # Fragile!
```

**GOOD - Testing behavior:**
```python
def test_embedding_config_valid():
    config = get_model_config("ai-forever/FRIDA")
    assert config.port > 7000  # Valid range
    assert config.port < 65535  # Valid range
    assert config.model_type == "embedding"
```

### Key Guidelines

1. **Test Outcomes, Not Mechanisms**
   - Test that a feature works correctly
   - Don't test internal function calls or implementation paths
   - Example: Test that config returns valid port range, not specific port

2. **Avoid Hardcoded Values**
   - Don't assert on specific ports, paths, or internal states
   - Assert on functional requirements and valid patterns
   - Example: Assert port is in valid range (7000-65535), not specific value

3. **Test Behavior Contracts**
   - Define what the function should do (inputs → outputs)
   - Test the contract, not the implementation
   - Example: "Given a model slug, return valid config" not "call registry.get()"

4. **Use Mocks Judiciously**
   - Mock external dependencies (HTTP APIs, file system)
   - Don't mock internal implementation details

5. **Test Real Scenarios**
   - Test user-facing behavior
   - Test edge cases and error handling
   - Example: Test "unknown model raises error", not "ValueError raised"

## Error Handling

- Use try/except around process operations
- Log errors with logger, not print
- Return True/False from manager methods
- CLI catches exceptions and exits with code 1

## Platform Notes

- Windows: SIGKILL not available, use SIGTERM
- Linux/macOS: Full signal support
- PID files stored in ~/.cmw-mosec/

## Development

- Activate the project venv before running Python or tests

## Testing

Test scenarios:
1. Start/stop embedding server (ai-forever/FRIDA)
2. Start/stop reranker server (DiTy/cross-encoder-russian-msmarco)
3. Health check via HTTP
4. Multiple start calls (idempotent)
5. Stop non-running server
6. List running servers

## Agent Behavior

- **Planning:** Plan your course of action before implementing.
- **Verification:** Run `ruff check <modified_file>` after changes. Run relevant tests. Reanalyze changes for introduced issues.
- **Linting:** Only lint files that were modified, not the entire codebase. Be critical about Ruff reports; implement only necessary changes.
- **Secrets:** Never hardcode secrets. Use environment variables.
- **No breakage:** Never break existing code.

## Code Style

- Follow Google docstring convention. Type hints required. Line length: 100. Use ruff for linting.
- **Naming:** `snake_case` for variables/functions, `PascalCase` for classes, `UPPER_CASE` for constants.
- **Imports:** At top of file; ruff handles sorting.
- **Comments:** Explain why, not what. Do not delete existing comments or logging; update if needed.
- **Error handling:** Avoid unnecessary try/except. Catch only when necessary and meaningful. Prefer robust, explicit logic over hardcoded fallbacks.

## 12-Factor App Principles

Following twelve-factor methodology for the repository as a whole:

- **Codebase:** One codebase tracked in revision control, many deploys.
- **Dependencies:** Declare all dependencies explicitly in `pyproject.toml`.
- **Config:** Store all environment-specific config in env vars. Use `.env` files for local dev.
- **Backing Services:** Treat model caches, PID directories as attached resources.
- **Build, Release, Run:** Separate build (pip install) and run (cmw-mosec start) stages.
- **Processes:** Execute as stateless processes. PID files track state.
- **Port Binding:** Export services via port binding. Port specified in config.
- **Disposability:** Maximize robustness with fast startup and graceful shutdown.
- **Dev/Prod Parity:** Keep development and production similar.
- **Logs:** Treat logs as event streams. Use logging module.
- **Admin Processes:** Run admin tasks (status, list) as one-off processes.
