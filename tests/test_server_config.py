"""Tests for cmw-mosec configuration and guards."""

from __future__ import annotations

import pytest

from cmw_mosec.server_config import (
    GUARD_CATEGORIES,
    GUARD_SAFETY_LEVELS,
    ModelRegistry,
    get_model_config,
    list_available_models,
    parse_guard_output,
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
    assert config.port == 8110
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
    assert config.port == 8111
    assert config.memory_gb == 2.0


def test_get_model_config_qwen3_reranker():
    """Test getting Qwen3 reranker configs."""
    config = get_model_config("Qwen/Qwen3-Reranker-0.6B")
    assert config.model_id == "Qwen/Qwen3-Reranker-0.6B"
    assert config.model_type == "reranker"
    assert config.port == 8112
    assert config.memory_gb == 2.0

    config_4b = get_model_config("Qwen/Qwen3-Reranker-4B")
    assert config_4b.port == 8113
    assert config_4b.memory_gb == 12.0

    config_8b = get_model_config("Qwen/Qwen3-Reranker-8B")
    assert config_8b.port == 8114
    assert config_8b.memory_gb == 22.0


def test_get_model_config_guard_0_6b():
    """Test getting Qwen3Guard-Gen-0.6B config."""
    config = get_model_config("Qwen/Qwen3Guard-Gen-0.6B")
    assert config.model_id == "Qwen/Qwen3Guard-Gen-0.6B"
    assert config.model_type == "guard"
    assert config.port == 8220
    assert config.memory_gb == 4.0
    assert config.max_new_tokens == 128
    assert config.dtype == "bf16"


def test_get_model_config_guard_4b():
    """Test getting Qwen3Guard-Gen-4B config."""
    config = get_model_config("Qwen/Qwen3Guard-Gen-4B")
    assert config.model_id == "Qwen/Qwen3Guard-Gen-4B"
    assert config.model_type == "guard"
    assert config.port == 8221
    assert config.memory_gb == 10.0
    assert config.max_new_tokens == 128


def test_get_model_config_guard_8b():
    """Test getting Qwen3Guard-Gen-8B config."""
    config = get_model_config("Qwen/Qwen3Guard-Gen-8B")
    assert config.model_id == "Qwen/Qwen3Guard-Gen-8B"
    assert config.model_type == "guard"
    assert config.port == 8222
    assert config.memory_gb == 20.0
    assert config.max_new_tokens == 128


def test_get_model_config_unknown():
    """Test getting unknown model raises error."""
    with pytest.raises(ValueError, match="Unknown model"):
        get_model_config("unknown-model")


def test_list_available_models():
    """Test listing available models."""
    models = list_available_models()
    assert "embedding" in models
    assert "reranker" in models
    assert "guard" in models
    assert "ai-forever/FRIDA" in models["embedding"]
    assert "DiTy/cross-encoder-russian-msmarco" in models["reranker"]
    assert "Qwen/Qwen3Guard-Gen-0.6B" in models["guard"]


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


def test_all_guard_models():
    """Test all guard models have required fields."""
    registry = ModelRegistry()
    for slug in registry.list_guards():
        config = registry.get_guard_config(slug)
        assert config.model_id
        assert config.model_type == "guard"
        assert config.port > 0
        assert config.memory_gb > 0
        assert config.workers >= 1
        assert config.max_new_tokens is not None
        assert config.max_new_tokens > 0


def test_case_insensitive_lookup():
    """Test case-insensitive model slug lookup."""
    registry = ModelRegistry()

    upper = registry.get_config("AI-FOREVER/FRIDA")
    lower = registry.get_config("ai-forever/frida")
    mixed = registry.get_config("Ai-Forever/Frida")

    assert upper.model_id == lower.model_id == mixed.model_id == "ai-forever/FRIDA"
    assert upper.port == lower.port == mixed.port == 8001


def test_model_type_detection():
    """Test model type detection."""
    registry = ModelRegistry()

    assert registry.get_model_type("ai-forever/FRIDA") == "embedding"
    assert registry.get_model_type("DiTy/cross-encoder-russian-msmarco") == "reranker"
    assert registry.get_model_type("Qwen/Qwen3Guard-Gen-0.6B") == "guard"


class TestGuardOutputParsing:
    """Tests for guard output parsing."""

    def test_parse_safe_output(self):
        """Test parsing safe output."""
        output = "Safety: Safe\nCategories: None"
        result = parse_guard_output(output)
        assert result["safety_level"] == "Safe"
        assert result["categories"] == ["None"]
        assert result["refusal"] is None

    def test_parse_unsafe_violent(self):
        """Test parsing unsafe violent output."""
        output = "Safety: Unsafe\nCategories: Violent"
        result = parse_guard_output(output)
        assert result["safety_level"] == "Unsafe"
        assert "Violent" in result["categories"]

    def test_parse_unsafe_multiple_categories(self):
        """Test parsing output with multiple categories."""
        output = "Safety: Unsafe\nCategories: Violent, Non-violent Illegal Acts"
        result = parse_guard_output(output)
        assert result["safety_level"] == "Unsafe"
        assert "Violent" in result["categories"]
        assert "Non-violent Illegal Acts" in result["categories"]

    def test_parse_controversial(self):
        """Test parsing controversial output."""
        output = "Safety: Controversial\nCategories: Politically Sensitive Topics"
        result = parse_guard_output(output)
        assert result["safety_level"] == "Controversial"
        assert "Politically Sensitive Topics" in result["categories"]

    def test_parse_with_refusal(self):
        """Test parsing response moderation output with refusal."""
        output = "Safety: Safe\nCategories: None\nRefusal: Yes"
        result = parse_guard_output(output)
        assert result["safety_level"] == "Safe"
        assert result["refusal"] == "Yes"

    def test_parse_refusal_no(self):
        """Test parsing output with Refusal: No."""
        output = "Safety: Safe\nCategories: None\nRefusal: No"
        result = parse_guard_output(output)
        assert result["refusal"] == "No"

    def test_parse_all_categories(self):
        """Test that all guard categories can be parsed."""
        for category in GUARD_CATEGORIES:
            output = f"Safety: Unsafe\nCategories: {category}"
            result = parse_guard_output(output)
            assert category in result["categories"], f"Failed to parse category: {category}"

    def test_parse_preserves_raw_output(self):
        """Test that raw output is preserved."""
        output = "Safety: Unsafe\nCategories: Violent\nRefusal: Yes"
        result = parse_guard_output(output)
        assert result["raw_output"] == output


class TestGuardConstants:
    """Tests for guard model constants."""

    def test_safety_levels(self):
        """Test safety level constants."""
        assert "Safe" in GUARD_SAFETY_LEVELS
        assert "Controversial" in GUARD_SAFETY_LEVELS
        assert "Unsafe" in GUARD_SAFETY_LEVELS
        assert len(GUARD_SAFETY_LEVELS) == 3

    def test_guard_categories(self):
        """Test guard category constants."""
        expected = [
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
        assert expected == GUARD_CATEGORIES
        assert len(GUARD_CATEGORIES) == 9

    def test_categories_include_none(self):
        """Test that 'None' category is handled."""
        output = "Safety: Safe\nCategories: None"
        result = parse_guard_output(output)
        assert "None" in result["categories"]
