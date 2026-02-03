"""Tests for cmw-mosec."""

from __future__ import annotations

import pytest

from cmw_mosec.server_config import (
    ModelRegistry,
    get_model_config,
    list_available_models,
)


def test_get_model_config_frida():
    """Test getting FRIDA config (case-insensitive slug)."""
    config = get_model_config("ai-forever/FRIDA")
    assert config.model_id == "ai-forever/FRIDA"
    assert config.model_type == "embedding"
    assert config.port == 8001
    config_lower = get_model_config("ai-forever/frida")
    assert config_lower.model_id == "ai-forever/FRIDA"


def test_get_model_config_dity():
    """Test getting DiTy reranker config (case-insensitive slug)."""
    config = get_model_config("DiTy/cross-encoder-russian-msmarco")
    assert config.model_id == "DiTy/cross-encoder-russian-msmarco"
    assert config.model_type == "reranker"
    assert config.port == 8010
    config_lower = get_model_config("dity/cross-encoder-russian-msmarco")
    assert config_lower.model_id == "DiTy/cross-encoder-russian-msmarco"


def test_get_model_config_qwen3_embedding():
    """Test getting Qwen3 embedding configs."""
    config = get_model_config("Qwen/Qwen3-Embedding-0.6B")
    assert config.model_id == "Qwen/Qwen3-Embedding-0.6B"
    assert config.model_type == "embedding"
    assert config.port == 8002
    assert config.memory_gb == 2.0

    config_4b = get_model_config("Qwen/Qwen3-Embedding-4B")
    assert config_4b.port == 8003
    assert config_4b.memory_gb == 12.0

    config_8b = get_model_config("Qwen/Qwen3-Embedding-8B")
    assert config_8b.port == 8004
    assert config_8b.memory_gb == 22.0


def test_get_model_config_bge_reranker():
    """Test getting BGE reranker config."""
    config = get_model_config("BAAI/bge-reranker-v2-m3")
    assert config.model_id == "BAAI/bge-reranker-v2-m3"
    assert config.model_type == "reranker"
    assert config.port == 8011
    assert config.memory_gb == 2.0


def test_get_model_config_qwen3_reranker():
    """Test getting Qwen3 reranker configs."""
    config = get_model_config("Qwen/Qwen3-Reranker-0.6B")
    assert config.model_id == "Qwen/Qwen3-Reranker-0.6B"
    assert config.model_type == "reranker"
    assert config.port == 8012
    assert config.memory_gb == 2.0

    config_4b = get_model_config("Qwen/Qwen3-Reranker-4B")
    assert config_4b.port == 8013
    assert config_4b.memory_gb == 12.0

    config_8b = get_model_config("Qwen/Qwen3-Reranker-8B")
    assert config_8b.port == 8014
    assert config_8b.memory_gb == 22.0


def test_get_model_config_unknown():
    """Test getting unknown model raises error."""
    with pytest.raises(ValueError, match="Unknown model"):
        get_model_config("unknown-model")


def test_list_available_models():
    """Test listing available models."""
    models = list_available_models()
    assert "embedding" in models
    assert "reranker" in models
    assert "ai-forever/FRIDA" in models["embedding"]
    assert "DiTy/cross-encoder-russian-msmarco" in models["reranker"]


def test_port_validation():
    """Test port range validation."""
    from pydantic import ValidationError

    from cmw_mosec.server_config import MosecModelConfig

    config = MosecModelConfig(
        model_id="test/model",
        model_type="embedding",
        port=8000,
        memory_gb=4.0,
    )
    assert config.port == 8000

    with pytest.raises(ValidationError):
        MosecModelConfig(
            model_id="test/model",
            model_type="embedding",
            port=1000,
            memory_gb=4.0,
        )


def test_workers_validation():
    """Test workers validation."""
    from pydantic import ValidationError

    from cmw_mosec.server_config import MosecModelConfig

    config = MosecModelConfig(
        model_id="test/model",
        model_type="embedding",
        port=8000,
        memory_gb=4.0,
        workers=4,
    )
    assert config.workers == 4

    with pytest.raises(ValidationError):
        MosecModelConfig(
            model_id="test/model",
            model_type="embedding",
            port=8000,
            memory_gb=4.0,
            workers=0,
        )


def test_all_embedding_models():
    """Test all embedding models have required fields."""
    registry = ModelRegistry()
    for slug in registry.list_embeddings():
        config = registry.get_embedding_config(slug)
        assert config.model_id
        assert config.model_type == "embedding"
        assert config.port > 0
        assert config.memory_gb > 0
        assert config.workers >= 1


def test_all_reranker_models():
    """Test all reranker models have required fields."""
    registry = ModelRegistry()
    for slug in registry.list_rerankers():
        config = registry.get_reranker_config(slug)
        assert config.model_id
        assert config.model_type == "reranker"
        assert config.port > 0
        assert config.memory_gb > 0
        assert config.workers >= 1


def test_case_insensitive_lookup():
    """Test case-insensitive model slug lookup."""
    registry = ModelRegistry()

    upper = registry.get_config("AI-FOREVER/FRIDA")
    lower = registry.get_config("ai-forever/frida")
    mixed = registry.get_config("Ai-Forever/Frida")

    assert upper.model_id == lower.model_id == mixed.model_id == "ai-forever/FRIDA"
    assert upper.port == lower.port == mixed.port == 8001
