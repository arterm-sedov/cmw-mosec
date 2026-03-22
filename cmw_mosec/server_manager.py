"""Process management for Mosec servers.

Single combined server with dynamic model loading/unloading.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests

from .server_config import (
    ModelRegistry,
    ServerStatus,
    load_active_models,
    load_server_settings,
)

logger = logging.getLogger(__name__)

PID_DIR = Path.home() / ".cmw-mosec"
SERVER_PID_FILE = PID_DIR / "server.pid"


def _ensure_pid_dir() -> None:
    """Ensure PID directory exists."""
    PID_DIR.mkdir(parents=True, exist_ok=True)


def _save_server_pid(pid: int, port: int, models: dict[str, str | None] | None = None) -> None:
    """Save server PID info."""
    _ensure_pid_dir()
    data = {
        "pid": pid,
        "port": port,
        "started_at": time.time(),
        "models": models or {},
    }
    SERVER_PID_FILE.write_text(json.dumps(data))


def _load_server_pid() -> dict[str, Any] | None:
    """Load server PID info."""
    if not SERVER_PID_FILE.exists():
        return None
    try:
        return json.loads(SERVER_PID_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _remove_server_pid() -> None:
    """Remove server PID file."""
    if SERVER_PID_FILE.exists():
        SERVER_PID_FILE.unlink()


def _is_process_running(pid: int) -> bool:
    """Check if a process is running."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _check_server_health(port: int, timeout: float = 2.0) -> bool:
    """Check if Mosec server is responding (using metrics endpoint)."""
    try:
        response = requests.get(f"http://localhost:{port}/metrics", timeout=timeout)
        return response.status_code == 200
    except requests.RequestException:
        return False


class MosecServerManager:
    """Manages the combined Mosec server with dynamic model loading."""

    def __init__(self):
        self.pid_dir = PID_DIR

    def get_status(self) -> ServerStatus:
        """Get status of the combined server."""
        pid_info = _load_server_pid()
        if not pid_info:
            return ServerStatus(
                model_key="combined",
                model_id="combined",
                model_type="combined",
                port=0,
                device="unknown",
                pid=None,
                is_running=False,
                uptime_seconds=None,
            )

        pid = pid_info.get("pid")
        port = pid_info.get("port", 0)

        is_running = False
        uptime = None

        if pid and _is_process_running(pid) and _check_server_health(port):
            is_running = True
            if "started_at" in pid_info:
                uptime = time.time() - pid_info["started_at"]

        try:
            settings = load_server_settings()
            device = settings.device
        except Exception:
            device = "unknown"

        return ServerStatus(
            model_key="combined",
            model_id="combined",
            model_type="combined",
            port=port,
            device=device,
            pid=pid,
            is_running=is_running,
            uptime_seconds=uptime,
        )

    def is_running(self) -> bool:
        """Check if server is running."""
        status = self.get_status()
        return status.is_running

    def start(
        self,
        embedding_model: str | None = None,
        reranker_model: str | None = None,
        guard_model: str | None = None,
        background: bool = True,
    ) -> tuple[bool, list[str]]:
        """Start the combined server.

        Args:
            embedding_model: Embedding model slug to load
            reranker_model: Reranker model slug to load
            guard_model: Guard model slug to load
            background: Whether to run in background

        Returns:
            Tuple of (success, list of failed models)
        """
        if self.is_running():
            logger.info("Server already running")
            return True, []

        try:
            settings = load_server_settings()
        except Exception as e:
            logger.error(f"Failed to load settings: {e}")
            return False, ["settings"]

        # cli.py passes None when a model is not specified
        # If None, don't load that model (no fallback to .env)
        emb_model = embedding_model
        rer_model = reranker_model
        guard_m = guard_model

        failed_models = []

        # Validate models exist
        from .server_config import ModelRegistry

        registry = ModelRegistry()

        if emb_model:
            try:
                registry.get_config(emb_model)
            except ValueError as e:
                logger.error(f"Embedding model error: {e}")
                failed_models.append(f"embedding: {emb_model}")
                emb_model = None

        if rer_model:
            try:
                registry.get_config(rer_model)
            except ValueError as e:
                logger.error(f"Reranker model error: {e}")
                failed_models.append(f"reranker: {rer_model}")
                rer_model = None

        if guard_m:
            try:
                registry.get_config(guard_m)
            except ValueError as e:
                logger.error(f"Guard model error: {e}")
                failed_models.append(f"guard: {guard_m}")
                guard_m = None

        if not emb_model and not rer_model and not guard_m:
            logger.error("No valid models to load")
            return False, failed_models

        logger.info(
            f"Starting server with models: emb={emb_model}, rer={rer_model}, guard={guard_m}"
        )

        env = os.environ.copy()
        env["ACTIVE_EMBEDDING_MODEL"] = emb_model or ""
        env["ACTIVE_RERANKER_MODEL"] = rer_model or ""
        env["ACTIVE_GUARD_MODEL"] = guard_m or ""

        if settings.hf_token:
            env["HF_TOKEN"] = settings.hf_token

        cmd = [
            sys.executable,
            "-m",
            "cmw_mosec.v2.dynamic_server",
            "--port",
            str(settings.server_port),
        ]

        try:
            if background:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                    env=env,
                )
            else:
                process = subprocess.Popen(cmd, env=env)

            loaded_models = {
                "embedding": emb_model,
                "reranker": rer_model,
                "guard": guard_m,
            }
            _save_server_pid(process.pid, settings.server_port, loaded_models)

            if background:
                logger.info(f"Waiting for server on port {settings.server_port}...")
                for _ in range(60):
                    if _check_server_health(settings.server_port):
                        logger.info("Server is ready!")
                        return True, failed_models
                    time.sleep(1)
                    if process.poll() is not None:
                        logger.error(f"Server process exited with code {process.returncode}")
                        _remove_server_pid()
                        return False, failed_models

                logger.warning("Server may still be starting...")
                return True, failed_models
            else:
                process.wait()
                return process.returncode == 0, failed_models

        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            return False, failed_models

    def stop(self) -> bool:
        """Stop the server."""
        pid_info = _load_server_pid()
        if not pid_info:
            logger.info("No server running")
            return True

        pid = pid_info.get("pid")
        if not pid:
            _remove_server_pid()
            return True

        if not _is_process_running(pid):
            logger.info("Server not running")
            _remove_server_pid()
            return True

        logger.info(f"Stopping server (PID {pid})...")

        try:
            os.kill(pid, signal.SIGTERM)

            for _ in range(10):
                if not _is_process_running(pid):
                    logger.info("Server stopped gracefully")
                    _remove_server_pid()
                    return True
                time.sleep(1)

            logger.warning(f"Force killing server (PID {pid})...")
            kill_signal = signal.SIGTERM if sys.platform == "win32" else signal.SIGKILL
            with contextlib.suppress(OSError, ProcessLookupError):
                os.kill(pid, kill_signal)
            time.sleep(1)

            _remove_server_pid()
            return True

        except (OSError, ProcessLookupError) as e:
            logger.warning(f"Error stopping process: {e}")
            _remove_server_pid()
            return True

    def load_model(self, model_slug: str) -> bool:
        """Load a model into the running server.

        Note: This requires hot-reload support in Mosec which may not be available.
        For now, this is a placeholder - full implementation would require
        Mosec hot-reload or model swapping functionality.

        Args:
            model_slug: Model to load

        Returns:
            True if successful
        """
        logger.info(f"Model loading requested: {model_slug}")
        logger.warning("Dynamic model loading requires Mosec hot-reload support")

        try:
            ModelRegistry().get_config(model_slug)
            logger.info(f"Model {model_slug} config loaded (server restart required to apply)")
            return True
        except ValueError as e:
            logger.error(f"Unknown model: {e}")
            return False

    def unload_model(self, model_slug: str) -> bool:
        """Unload a model from the running server.

        Note: See load_model() - this is a placeholder.

        Args:
            model_slug: Model to unload

        Returns:
            True if successful
        """
        logger.info(f"Model unloading requested: {model_slug}")
        logger.warning("Dynamic model unloading requires Mosec hot-reload support")
        return True

    def list_loaded_models(self) -> dict[str, str | None]:
        """List currently loaded models.

        Returns:
            Dict with embedding, reranker, guard -> model slug or None
        """
        pid_info = _load_server_pid()
        if pid_info and "models" in pid_info:
            return pid_info["models"]

        # Fallback to .env for compatibility
        active = load_active_models()
        return {
            "embedding": active["embedding"],
            "reranker": active["reranker"],
            "guard": active["guard"],
        }

    def stop_all(self) -> bool:
        """Stop all servers (just stops the combined server)."""
        return self.stop()
