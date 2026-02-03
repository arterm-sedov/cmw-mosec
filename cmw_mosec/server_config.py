"""Server configuration management for Mosec.

Supports case-insensitive model slug lookup with canonical normalization.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class MosecModelConfig(BaseModel):
    """Server-side configuration for Mosec models."""

    model_id: str = Field(description="HuggingFace model ID")
    model_type: Literal["embedding", "reranker"] = Field(description="Model type")
    port: int = Field(description="Server port (must be unique per model)")
    device: str = Field(default="auto", description="Device (auto/cpu/cuda)")
    dtype: Literal["float16", "float32", "int8"] = Field(default="float16")
    batch_size: int = Field(default=32, description="Dynamic batching size")
    memory_gb: float = Field(description="Estimated VRAM usage in GB")
    workers: int = Field(default=1, description="Number of Mosec workers")

    @field_validator("port", mode="before")
    @classmethod
    def validate_port_range(cls, v: int) -> int:
        if not 7000 <= v <= 65535:
            raise ValueError("Port must be between 7000-65535")
        return v

    @field_validator("workers", mode="before")
    @classmethod
    def validate_workers_count(cls, v: int) -> int:
        if v < 1:
            raise ValueError("Workers must be at least 1")
        return v


class ServerStatus(BaseModel):
    """Status of a running Mosec server."""

    model_key: str = Field(description="Model identifier")
    model_id: str = Field(description="HuggingFace model ID")
    port: int = Field(description="Server port")
    device: str = Field(description="Device (auto/cpu/cuda)")
    pid: int | None = Field(None, description="Process ID")
    is_running: bool = Field(False, description="Whether server is responding")
    uptime_seconds: float | None = Field(None, description="Server uptime")


class ModelRegistry:
    """Registry for model metadata loaded from YAML.

    Supports case-insensitive model slug lookup with canonical normalization.
    """

    _instance = None
    _embeddings: dict[str, dict[str, Any]] = {}
    _rerankers: dict[str, dict[str, Any]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_registry()
        return cls._instance

    def _load_registry(self) -> None:
        """Load model registry from YAML file."""
        config_path = Path(__file__).parent.parent / "config" / "models.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"Model registry not found: {config_path}")

        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        for model_slug, model_data in data.get("embedding_models", {}).items():
            normalized = model_slug.lower()
            self._embeddings[normalized] = {
                "canonical_slug": model_slug,
                "model_type": "embedding",
                **model_data,
            }

        for model_slug, model_data in data.get("reranker_models", {}).items():
            normalized = model_slug.lower()
            self._rerankers[normalized] = {
                "canonical_slug": model_slug,
                "model_type": "reranker",
                **model_data,
            }

        logger.info(
            f"Loaded {len(self._embeddings)} embedding models and {len(self._rerankers)} reranker models"
        )

    def _normalize_slug(self, model_slug: str) -> str:
        """Normalize model slug to lowercase for case-insensitive lookup."""
        return model_slug.lower().strip()

    def _to_config(self, data: dict[str, Any]) -> MosecModelConfig:
        """Build MosecModelConfig from registry dict (exclude canonical_slug)."""
        return MosecModelConfig(
            **{k: v for k, v in data.items() if k != "canonical_slug"}
        )

    def get_embedding_config(self, model_slug: str) -> MosecModelConfig:
        """Get configuration for an embedding model (case-insensitive)."""
        normalized = self._normalize_slug(model_slug)
        if normalized not in self._embeddings:
            available = [m["canonical_slug"] for m in self._embeddings.values()]
            raise ValueError(f"Unknown embedding model: {model_slug}. Available: {available}")
        return self._to_config(self._embeddings[normalized])

    def get_reranker_config(self, model_slug: str) -> MosecModelConfig:
        """Get configuration for a reranker model (case-insensitive)."""
        normalized = self._normalize_slug(model_slug)
        if normalized not in self._rerankers:
            available = [m["canonical_slug"] for m in self._rerankers.values()]
            raise ValueError(f"Unknown reranker model: {model_slug}. Available: {available}")
        return self._to_config(self._rerankers[normalized])

    def get_config(self, model_slug: str) -> MosecModelConfig:
        """Get configuration for any model (case-insensitive)."""
        normalized = self._normalize_slug(model_slug)
        if normalized in self._embeddings:
            return self._to_config(self._embeddings[normalized])
        if normalized in self._rerankers:
            return self._to_config(self._rerankers[normalized])
        available = [m["canonical_slug"] for m in self._embeddings.values()] + [
            m["canonical_slug"] for m in self._rerankers.values()
        ]
        raise ValueError(f"Unknown model: {model_slug}. Available: {available}")

    def get_model_type(self, model_slug: str) -> Literal["embedding", "reranker"]:
        """Get model type for a model slug."""
        normalized = self._normalize_slug(model_slug)
        if normalized in self._embeddings:
            return "embedding"
        if normalized in self._rerankers:
            return "reranker"
        raise ValueError(f"Unknown model: {model_slug}")

    def list_embeddings(self) -> list[str]:
        """List all available embedding models."""
        return [m["canonical_slug"] for m in self._embeddings.values()]

    def list_rerankers(self) -> list[str]:
        """List all available reranker models."""
        return [m["canonical_slug"] for m in self._rerankers.values()]

    def list_all(self) -> dict[str, list[str]]:
        """List all available models by type."""
        return {
            "embedding": self.list_embeddings(),
            "reranker": self.list_rerankers(),
        }


def get_model_config(model_slug: str) -> MosecModelConfig:
    """Get configuration for a model (case-insensitive)."""
    return ModelRegistry().get_config(model_slug)


def list_available_models() -> dict[str, list[str]]:
    """List all available models by type."""
    return ModelRegistry().list_all()
