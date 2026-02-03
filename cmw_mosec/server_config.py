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
    model_type: Literal["embedding", "reranker", "guard"] = Field(description="Model type")
    port: int = Field(description="Server port (must be unique per model)")
    device: str = Field(default="auto", description="Device (auto/cpu/cuda)")
    dtype: Literal["float16", "float32", "bf16", "int8"] = Field(default="float16")
    batch_size: int = Field(default=32, description="Dynamic batching size")
    memory_gb: float = Field(description="Estimated VRAM usage in GB")
    workers: int = Field(default=1, description="Number of Mosec workers")
    max_new_tokens: int | None = Field(default=None, description="Max new tokens (guards only)")
    transformers_min_version: str | None = Field(default=None, description="Min transformers version")
    description: str | None = Field(default=None, description="Model description")

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

    @field_validator("max_new_tokens", mode="before")
    @classmethod
    def validate_max_new_tokens(cls, v: int | None) -> int | None:
        if v is not None and v < 1:
            raise ValueError("max_new_tokens must be at least 1")
        return v


class ServerStatus(BaseModel):
    """Status of a running Mosec server."""

    model_key: str = Field(description="Model identifier")
    model_id: str = Field(description="HuggingFace model ID")
    model_type: str = Field(description="Model type")
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
    _guards: dict[str, dict[str, Any]] = {}

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

        for model_slug, model_data in data.get("guard_models", {}).items():
            normalized = model_slug.lower()
            self._guards[normalized] = {
                "canonical_slug": model_slug,
                "model_type": "guard",
                **model_data,
            }

        logger.info(
            f"Loaded {len(self._embeddings)} embedding models, "
            f"{len(self._rerankers)} reranker models, "
            f"and {len(self._guards)} guard models"
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

    def get_guard_config(self, model_slug: str) -> MosecModelConfig:
        """Get configuration for a guard model (case-insensitive)."""
        normalized = self._normalize_slug(model_slug)
        if normalized not in self._guards:
            available = [m["canonical_slug"] for m in self._guards.values()]
            raise ValueError(f"Unknown guard model: {model_slug}. Available: {available}")
        return self._to_config(self._guards[normalized])

    def get_config(self, model_slug: str) -> MosecModelConfig:
        """Get configuration for any model (case-insensitive)."""
        normalized = self._normalize_slug(model_slug)
        if normalized in self._embeddings:
            return self._to_config(self._embeddings[normalized])
        if normalized in self._rerankers:
            return self._to_config(self._rerankers[normalized])
        if normalized in self._guards:
            return self._to_config(self._guards[normalized])
        available = (
            [m["canonical_slug"] for m in self._embeddings.values()] +
            [m["canonical_slug"] for m in self._rerankers.values()] +
            [m["canonical_slug"] for m in self._guards.values()]
        )
        raise ValueError(f"Unknown model: {model_slug}. Available: {available}")

    def get_model_type(self, model_slug: str) -> Literal["embedding", "reranker", "guard"]:
        """Get model type for a model slug."""
        normalized = self._normalize_slug(model_slug)
        if normalized in self._embeddings:
            return "embedding"
        if normalized in self._rerankers:
            return "reranker"
        if normalized in self._guards:
            return "guard"
        raise ValueError(f"Unknown model: {model_slug}")

    def list_embeddings(self) -> list[str]:
        """List all available embedding models."""
        return [m["canonical_slug"] for m in self._embeddings.values()]

    def list_rerankers(self) -> list[str]:
        """List all available reranker models."""
        return [m["canonical_slug"] for m in self._rerankers.values()]

    def list_guards(self) -> list[str]:
        """List all available guard models."""
        return [m["canonical_slug"] for m in self._guards.values()]

    def list_all(self) -> dict[str, list[str]]:
        """List all available models by type."""
        return {
            "embedding": self.list_embeddings(),
            "reranker": self.list_rerankers(),
            "guard": self.list_guards(),
        }

    def list_by_type(self, model_type: str) -> list[str]:
        """List models of a specific type."""
        if model_type == "embedding":
            return self.list_embeddings()
        elif model_type == "reranker":
            return self.list_rerankers()
        elif model_type == "guard":
            return self.list_guards()
        else:
            raise ValueError(f"Unknown model type: {model_type}")


def get_model_config(model_slug: str) -> MosecModelConfig:
    """Get configuration for a model (case-insensitive)."""
    return ModelRegistry().get_config(model_slug)


def list_available_models() -> dict[str, list[str]]:
    """List all available models by type."""
    return ModelRegistry().list_all()


# Guard-specific constants

GUARD_SAFETY_LEVELS = ["Safe", "Controversial", "Unsafe"]

GUARD_CATEGORIES = [
    "Violent",
    "Non-violent Illegal Acts",
    "Sexual Content or Sexual Acts",
    "PII",
    "Suicide & Self-Harm",
    "Unethical Acts",
    "Politically Sensitive Topics",
    "Copyright Violation",
    "Jailbreak",
]

GUARD_CATEGORY_PATTERNS = [
    r"Violent",
    r"Non-violent Illegal Acts",
    r"Sexual Content or Sexual Acts",
    r"PII",
    r"Suicide & Self-Harm",
    r"Unethical Acts",
    r"Politically Sensitive Topics",
    r"Copyright Violation",
    r"Jailbreak",
]


def parse_guard_output(output: str) -> dict[str, Any]:
    """Parse guard model output into structured result.

    Args:
        output: Raw model output

    Returns:
        Dict with safety_level, categories, refusal (optional)
    """
    import re

    result = {
        "safety_level": "Unknown",
        "categories": [],
        "refusal": None,
        "raw_output": output,
    }

    # Extract safety level
    safety_match = re.search(
        r"Safety:\s*(Safe|Controversial|Unsafe)",
        output,
        re.IGNORECASE,
    )
    if safety_match:
        result["safety_level"] = safety_match.group(1).capitalize()

    # Extract categories
    category_pattern = "|".join(re.escape(cat) for cat in GUARD_CATEGORIES)
    category_matches = re.findall(f"({category_pattern})", output, re.IGNORECASE)
    if category_matches:
        # Normalize category names
        normalized = []
        for match in category_matches:
            for cat in GUARD_CATEGORIES:
                if cat.lower() == match.lower():
                    normalized.append(cat)
                    break
            else:
                normalized.append(match)
        result["categories"] = normalized
    else:
        result["categories"] = ["None"]

    # Extract refusal (for response moderation)
    refusal_match = re.search(r"Refusal:\s*(Yes|No)", output, re.IGNORECASE)
    if refusal_match:
        result["refusal"] = refusal_match.group(1).capitalize()

    return result
