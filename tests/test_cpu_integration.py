"""Integration tests for cmw-mosec with real inference.

These tests start actual Mosec servers and test real embedding/reranking.
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

    max_retries = 60
    for _ in range(max_retries):
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

    max_retries = 60
    for _ in range(max_retries):
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
