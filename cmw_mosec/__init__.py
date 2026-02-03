"""CMW Mosec - Server management for embedding/reranker inference using Mosec."""

from __future__ import annotations

from .server_config import (
    ModelRegistry,
    MosecModelConfig,
    ServerStatus,
    get_model_config,
    list_available_models,
)
from .server_manager import MosecServerManager

__version__ = "0.1.0"
__all__ = [
    "MosecModelConfig",
    "ServerStatus",
    "MosecServerManager",
    "get_model_config",
    "list_available_models",
    "ModelRegistry",
]
