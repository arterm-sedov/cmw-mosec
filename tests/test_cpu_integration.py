"""Integration tests for cmw-mosec with real inference.

These tests start actual Mosec servers and test real embedding/reranking/guard.
They use small models and device='auto' to work on both CPU and GPU.
"""

from __future__ import annotations

import contextlib
import time

import pytest
import requests

from cmw_mosec.server_config import MosecModelConfig
from cmw_mosec.server_manager import (
    MosecServerManager,
    _check_server_health,
    _remove_pid_file,
)

TEST_EMBEDDING_CONFIG = MosecModelConfig(
    model_id="sentence-transformers/all-MiniLM-L6-v2",
    model_type="embedding",
    port=9001,
    device="auto",
    dtype="float32",
    batch_size=8,
    memory_gb=0.5,
    workers=1,
)

TEST_RERANKER_CONFIG = MosecModelConfig(
    model_id="cross-encoder/ms-marco-MiniLM-L-2-v2",
    model_type="reranker",
    port=9002,
    device="auto",
    dtype="float32",
    batch_size=8,
    memory_gb=0.5,
    workers=1,
)


def cleanup_test_servers():
    """Stop any running test servers and clean up PID files."""
    manager = MosecServerManager()

    for model_key in ["test-embedding", "test-reranker"]:
        with contextlib.suppress(Exception):
            manager.stop(model_key)
        _remove_pid_file(model_key)


@pytest.fixture(scope="module", autouse=True)
def setup_test_module():
    """Clean up before and after test module."""
    cleanup_test_servers()
    yield
    cleanup_test_servers()


@pytest.fixture
def manager():
    """Create a fresh server manager."""
    return MosecServerManager()


@pytest.fixture
def embedding_server(manager):
    """Start an embedding server for testing."""
    model_key = "test-embedding"

    manager.stop(model_key)
    _remove_pid_file(model_key)

    success = manager.start(model_key, TEST_EMBEDDING_CONFIG, background=True)
    if not success:
        pytest.skip(
            "Failed to start embedding server (mosec may not be installed or dependencies missing)"
        )

    for _ in range(60):
        if _check_server_health(TEST_EMBEDDING_CONFIG.port, timeout=2.0):
            break
        time.sleep(1)
    else:
        manager.stop(model_key)
        pytest.skip("Embedding server failed to become healthy within timeout")

    yield TEST_EMBEDDING_CONFIG.port

    manager.stop(model_key)


@pytest.fixture
def reranker_server(manager):
    """Start a reranker server for testing."""
    model_key = "test-reranker"

    manager.stop(model_key)
    _remove_pid_file(model_key)

    success = manager.start(model_key, TEST_RERANKER_CONFIG, background=True)
    if not success:
        pytest.skip(
            "Failed to start reranker server (mosec may not be installed or dependencies missing)"
        )

    for _ in range(60):
        if _check_server_health(TEST_RERANKER_CONFIG.port, timeout=2.0):
            break
        time.sleep(1)
    else:
        manager.stop(model_key)
        pytest.skip("Reranker server failed to become healthy within timeout")

    yield TEST_RERANKER_CONFIG.port

    manager.stop(model_key)


class TestServerLifecycle:
    """Test server start/stop lifecycle."""

    def test_start_embedding_server(self, manager):
        """Test starting an embedding server."""
        model_key = "test-lifecycle-embedding"
        config = MosecModelConfig(
            model_id="sentence-transformers/all-MiniLM-L6-v2",
            model_type="embedding",
            port=9003,
            device="auto",
            dtype="float32",
            batch_size=8,
            memory_gb=0.5,
            workers=1,
        )

        manager.stop(model_key)
        _remove_pid_file(model_key)

        try:
            success = manager.start(model_key, config, background=True)
            if not success:
                pytest.skip("mosec not installed or dependencies missing")

            for _ in range(60):
                if _check_server_health(config.port, timeout=2.0):
                    break
                time.sleep(1)
            else:
                pytest.fail("Server failed to become healthy")

            status = manager.get_status(model_key, config)
            assert status.is_running is True
            assert status.port == config.port
            assert status.pid is not None

        finally:
            manager.stop(model_key)
            _remove_pid_file(model_key)

    def test_server_health_check(self, embedding_server):
        """Test health check endpoint."""
        port = embedding_server

        response = requests.get(f"http://localhost:{port}/health", timeout=5.0)
        assert response.status_code == 200


class TestEmbeddingAPI:
    """Test real embedding API calls."""

    def test_single_text_embedding(self, embedding_server):
        """Test embedding a single text."""
        port = embedding_server

        response = requests.post(
            f"http://localhost:{port}/v1/embeddings",
            json={
                "input": "This is a test sentence for embedding.",
                "model": "sentence-transformers/all-MiniLM-L6-v2",
            },
            timeout=10.0,
        )

        assert response.status_code == 200
        data = response.json()

        assert "data" in data
        assert len(data["data"]) == 1
        assert "embedding" in data["data"][0]

        embedding = data["data"][0]["embedding"]
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, (int, float)) for x in embedding)

    def test_batch_text_embedding(self, embedding_server):
        """Test embedding multiple texts at once."""
        port = embedding_server

        texts = [
            "First test sentence.",
            "Second test sentence.",
            "Third test sentence.",
        ]

        response = requests.post(
            f"http://localhost:{port}/v1/embeddings",
            json={
                "input": texts,
                "model": "sentence-transformers/all-MiniLM-L6-v2",
            },
            timeout=15.0,
        )

        assert response.status_code == 200
        data = response.json()

        assert len(data["data"]) == len(texts)

        dimensions = [len(item["embedding"]) for item in data["data"]]
        assert all(d == dimensions[0] for d in dimensions)

    def test_embedding_similarity(self, embedding_server):
        """Test that similar texts have higher similarity."""
        port = embedding_server

        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "A fast brown fox leaps over a sleepy dog.",
            "Machine learning is a subset of artificial intelligence.",
        ]

        response = requests.post(
            f"http://localhost:{port}/v1/embeddings",
            json={
                "input": texts,
                "model": "sentence-transformers/all-MiniLM-L6-v2",
            },
            timeout=15.0,
        )

        assert response.status_code == 200
        data = response.json()

        embeddings = [item["embedding"] for item in data["data"]]

        def cosine_similarity(a, b):
            import math

            dot = sum(x * y for x, y in zip(a, b, strict=True))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            return dot / (norm_a * norm_b)

        sim_01 = cosine_similarity(embeddings[0], embeddings[1])
        sim_02 = cosine_similarity(embeddings[0], embeddings[2])

        assert sim_01 > sim_02, f"Similar texts should have higher similarity: {sim_01} vs {sim_02}"


class TestRerankingAPI:
    """Test real reranking API calls."""

    def test_basic_reranking(self, reranker_server):
        """Test basic reranking functionality."""
        port = reranker_server

        query = "What is machine learning?"
        documents = [
            "Machine learning is a method of data analysis.",
            "The weather is sunny today.",
            "Deep learning is a subset of machine learning.",
        ]

        response = requests.post(
            f"http://localhost:{port}/inference",
            json={
                "query": query,
                "docs": documents,
            },
            timeout=15.0,
        )

        assert response.status_code == 200
        data = response.json()

        assert "scores" in data
        assert len(data["scores"]) == len(documents)

        for score in data["scores"]:
            assert isinstance(score, (int, float))

    def test_reranking_relevance_ordering(self, reranker_server):
        """Test that reranking orders documents by relevance."""
        port = reranker_server

        query = "artificial intelligence"
        documents = [
            "The capital of France is Paris.",
            "AI and machine learning are transforming technology.",
            "Python is a programming language.",
            "Artificial intelligence enables machines to learn.",
        ]

        response = requests.post(
            f"http://localhost:{port}/inference",
            json={
                "query": query,
                "docs": documents,
            },
            timeout=15.0,
        )

        assert response.status_code == 200
        data = response.json()

        scores = data["scores"]

        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        top_indices = sorted_indices[:2]
        ai_related_indices = [1, 3]

        assert any(i in ai_related_indices for i in top_indices), (
            "Top reranked results should include AI-related documents"
        )


class TestServerManagement:
    """Test server management functionality."""

    def test_list_running_servers(self, manager, embedding_server):
        """Test listing running servers."""
        running = manager.list_running()

        test_servers = [s for s in running if s.model_key == "test-embedding"]
        assert len(test_servers) > 0

        status = test_servers[0]
        assert status.is_running is True
        assert status.port == TEST_EMBEDDING_CONFIG.port

    def test_server_status_reporting(self, manager, embedding_server):
        """Test that server status is correctly reported."""
        status = manager.get_status("test-embedding", TEST_EMBEDDING_CONFIG)

        assert status.model_key == "test-embedding"
        assert status.model_id == TEST_EMBEDDING_CONFIG.model_id
        assert status.model_type == "embedding"
        assert status.port == TEST_EMBEDDING_CONFIG.port
        assert status.is_running is True
        assert status.pid is not None
        assert status.uptime_seconds is not None
        assert status.uptime_seconds >= 0

    def test_stop_server(self, manager):
        """Test stopping a server."""
        model_key = "test-stop-server"
        config = MosecModelConfig(
            model_id="sentence-transformers/all-MiniLM-L6-v2",
            model_type="embedding",
            port=9005,
            device="auto",
            dtype="float32",
            batch_size=8,
            memory_gb=0.5,
            workers=1,
        )

        manager.stop(model_key)
        _remove_pid_file(model_key)

        try:
            success = manager.start(model_key, config, background=True)
            if not success:
                pytest.skip(f"Failed to start server on port {config.port}")

            for _ in range(60):
                if _check_server_health(config.port, timeout=2.0):
                    break
                time.sleep(1)
            else:
                pytest.fail("Server failed to start")

            assert _check_server_health(config.port) is True

            assert manager.stop(model_key) is True

            time.sleep(2)

            assert _check_server_health(config.port) is False

        finally:
            manager.stop(model_key)
            _remove_pid_file(model_key)


class TestGuardCategories:
    """Test guard model categories based on HuggingFace docs.

    Tests cover all 9 safety categories and 3 severity levels from Qwen3Guard:
    - Violent
    - Non-violent Illegal Acts
    - Sexual Content or Sexual Acts
    - PII
    - Suicide & Self-Harm
    - Unethical Acts
    - Politically Sensitive Topics
    - Copyright Violation
    - Jailbreak (input only)
    """

    # Test categories (simplified - real tests would require actual guard server)
    def test_guard_categories_constant(self):
        """Test that all guard categories are defined."""
        from cmw_mosec.server_config import GUARD_CATEGORIES

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

    def test_guard_safety_levels_constant(self):
        """Test that all safety levels are defined."""
        from cmw_mosec.server_config import GUARD_SAFETY_LEVELS

        assert "Safe" in GUARD_SAFETY_LEVELS
        assert "Controversial" in GUARD_SAFETY_LEVELS
        assert "Unsafe" in GUARD_SAFETY_LEVELS
        assert len(GUARD_SAFETY_LEVELS) == 3

    def test_guard_config_has_max_new_tokens(self):
        """Test that guard configs have max_new_tokens configured."""
        from cmw_mosec.server_config import ModelRegistry

        registry = ModelRegistry()
        for slug in registry.list_guards():
            config = registry.get_guard_config(slug)
            assert config.max_new_tokens is not None
            assert config.max_new_tokens == 128, f"Guard {slug} should have max_new_tokens=128"

    def test_guard_config_has_transformers_version(self):
        """Test that guard configs have transformers version requirement."""
        from cmw_mosec.server_config import ModelRegistry

        registry = ModelRegistry()
        for slug in registry.list_guards():
            config = registry.get_guard_config(slug)
            assert config.transformers_min_version is not None
            assert config.transformers_min_version == "4.51.0", (
                f"Guard {slug} should require transformers>=4.51.0"
            )


class TestGuardOutputParsing:
    """Test guard output parsing for all categories."""

    def test_parse_violent(self):
        """Test parsing Violent category output."""
        from cmw_mosec.server_config import parse_guard_output

        output = "Safety: Unsafe\nCategories: Violent"
        result = parse_guard_output(output)
        assert "Violent" in result["categories"]
        assert result["safety_level"] == "Unsafe"

    def test_parse_nonviolent_illegal(self):
        """Test parsing Non-violent Illegal Acts category."""
        from cmw_mosec.server_config import parse_guard_output

        output = "Safety: Unsafe\nCategories: Non-violent Illegal Acts"
        result = parse_guard_output(output)
        assert "Non-violent Illegal Acts" in result["categories"]

    def test_parse_sexual_content(self):
        """Test parsing Sexual Content category."""
        from cmw_mosec.server_config import parse_guard_output

        output = "Safety: Unsafe\nCategories: Sexual Content or Sexual Acts"
        result = parse_guard_output(output)
        assert "Sexual Content or Sexual Acts" in result["categories"]

    def test_parse_pii(self):
        """Test parsing PII category."""
        from cmw_mosec.server_config import parse_guard_output

        output = "Safety: Unsafe\nCategories: PII"
        result = parse_guard_output(output)
        assert "PII" in result["categories"]

    def test_parse_suicide_self_harm(self):
        """Test parsing Suicide & Self-Harm category."""
        from cmw_mosec.server_config import parse_guard_output

        output = "Safety: Unsafe\nCategories: Suicide & Self-Harm"
        result = parse_guard_output(output)
        assert "Suicide & Self-Harm" in result["categories"]

    def test_parse_unethical_acts(self):
        """Test parsing Unethical Acts category."""
        from cmw_mosec.server_config import parse_guard_output

        output = "Safety: Controversial\nCategories: Unethical Acts"
        result = parse_guard_output(output)
        assert "Unethical Acts" in result["categories"]
        assert result["safety_level"] == "Controversial"

    def test_parse_politically_sensitive(self):
        """Test parsing Politically Sensitive Topics category."""
        from cmw_mosec.server_config import parse_guard_output

        output = "Safety: Controversial\nCategories: Politically Sensitive Topics"
        result = parse_guard_output(output)
        assert "Politically Sensitive Topics" in result["categories"]

    def test_parse_copyright(self):
        """Test parsing Copyright Violation category."""
        from cmw_mosec.server_config import parse_guard_output

        output = "Safety: Unsafe\nCategories: Copyright Violation"
        result = parse_guard_output(output)
        assert "Copyright Violation" in result["categories"]

    def test_parse_jailbreak(self):
        """Test parsing Jailbreak category (input only)."""
        from cmw_mosec.server_config import parse_guard_output

        output = "Safety: Unsafe\nCategories: Jailbreak"
        result = parse_guard_output(output)
        assert "Jailbreak" in result["categories"]

    def test_parse_safe_none_categories(self):
        """Test that Safe outputs have 'None' category."""
        from cmw_mosec.server_config import parse_guard_output

        output = "Safety: Safe\nCategories: None"
        result = parse_guard_output(output)
        assert result["safety_level"] == "Safe"
        assert "None" in result["categories"]

    def test_parse_refusal_yes(self):
        """Test parsing Refusal: Yes (for response moderation)."""
        from cmw_mosec.server_config import parse_guard_output

        output = "Safety: Safe\nCategories: None\nRefusal: Yes"
        result = parse_guard_output(output)
        assert result["refusal"] == "Yes"

    def test_parse_refusal_no(self):
        """Test parsing Refusal: No."""
        from cmw_mosec.server_config import parse_guard_output

        output = "Safety: Safe\nCategories: None\nRefusal: No"
        result = parse_guard_output(output)
        assert result["refusal"] == "No"

    def test_parse_multiple_categories(self):
        """Test parsing multiple categories at once."""
        from cmw_mosec.server_config import parse_guard_output

        output = "Safety: Unsafe\nCategories: Violent, Non-violent Illegal Acts"
        result = parse_guard_output(output)
        assert "Violent" in result["categories"]
        assert "Non-violent Illegal Acts" in result["categories"]

    def test_preserve_raw_output(self):
        """Test that raw output is preserved."""
        from cmw_mosec.server_config import parse_guard_output

        output = "Safety: Unsafe\nCategories: Violent"
        result = parse_guard_output(output)
        assert result["raw_output"] == output
