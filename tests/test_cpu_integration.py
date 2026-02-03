"""Integration tests for cmw-mosec with real inference.

These tests start the combined Mosec server and test real embedding/reranking/guard.
They use small models and device='auto' to work on both CPU and GPU.
"""

from __future__ import annotations

import contextlib
import time

import pytest
import requests

from cmw_mosec.server_config import (
    ModelRegistry,
    load_active_models,
    load_server_settings,
)
from cmw_mosec.server_manager import (
    MosecServerManager,
    _check_server_health,
    _remove_server_pid,
)


def get_active_config(model_slug: str):
    """Get the MosecModelConfig for an active model."""
    registry = ModelRegistry()
    model_type = registry.get_model_type(model_slug)
    if model_type == "embedding":
        return registry.get_embedding_config(model_slug)
    elif model_type == "reranker":
        return registry.get_reranker_config(model_slug)
    elif model_type == "guard":
        return registry.get_guard_config(model_slug)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


SETTINGS = load_server_settings()
ACTIVE_MODELS = load_active_models()

TEST_PORT = SETTINGS.server_port
TEST_EMBEDDER_SLUG = ACTIVE_MODELS["embedding"]
TEST_RERANKER_SLUG = ACTIVE_MODELS["reranker"]
TEST_GUARD_SLUG = ACTIVE_MODELS["guard"]

TEST_EMBEDDER_CONFIG = get_active_config(TEST_EMBEDDER_SLUG)
TEST_RERANKER_CONFIG = get_active_config(TEST_RERANKER_SLUG)
TEST_GUARD_CONFIG = get_active_config(TEST_GUARD_SLUG)


def cleanup_test_servers():
    """Stop any running test servers and clean up PID files."""
    manager = MosecServerManager()
    with contextlib.suppress(Exception):
        manager.stop()


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
def combined_server(manager):
    """Start the combined server for testing."""
    manager.stop()
    _remove_server_pid()

    success, failed = manager.start(
        embedding_model=TEST_EMBEDDER_SLUG,
        reranker_model=TEST_RERANKER_SLUG,
        guard_model=TEST_GUARD_SLUG,
        background=True,
    )
    if not success:
        pytest.skip(
            f"Failed to start combined server (mosec may not be installed or dependencies missing). Failed models: {failed}"
        )

    for _ in range(120):
        if _check_server_health(TEST_PORT, timeout=2.0):
            break
        time.sleep(1)
    else:
        manager.stop()
        pytest.skip("Combined server failed to become healthy within timeout")

    yield TEST_PORT

    manager.stop()


class TestServerLifecycle:
    """Test server start/stop lifecycle."""

    def test_start_combined_server(self, manager):
        """Test starting the combined server."""
        manager.stop()
        _remove_server_pid()

        try:
            success, failed = manager.start(
                embedding_model=TEST_EMBEDDER_SLUG,
                reranker_model=TEST_RERANKER_SLUG,
                guard_model=TEST_GUARD_SLUG,
                background=True,
            )
            if not success:
                pytest.skip("mosec not installed or dependencies missing")

            for _ in range(120):
                if _check_server_health(TEST_PORT, timeout=2.0):
                    break
                time.sleep(1)
            else:
                pytest.fail("Combined server failed to become healthy")

            status = manager.get_status()
            assert status.is_running is True
            assert status.port == TEST_PORT
            assert status.pid is not None

        finally:
            manager.stop()
            _remove_server_pid()

    def test_server_health_check(self, combined_server):
        """Test health check endpoint."""
        port = combined_server

        response = requests.get(f"http://localhost:{port}/health", timeout=5.0)
        assert response.status_code == 200


class TestEmbeddingAPI:
    """Test real embedding API calls."""

    def test_single_text_embedding(self, combined_server):
        """Test embedding a single text."""
        port = combined_server
        model_id = TEST_EMBEDDER_CONFIG.model_id

        response = requests.post(
            f"http://localhost:{port}/v1/embeddings",
            json={
                "input": "This is a test sentence for embedding.",
                "model": model_id,
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

    def test_batch_text_embedding(self, combined_server):
        """Test embedding multiple texts at once."""
        port = combined_server
        model_id = TEST_EMBEDDER_CONFIG.model_id

        texts = [
            "First test sentence.",
            "Second test sentence.",
            "Third test sentence.",
        ]

        response = requests.post(
            f"http://localhost:{port}/v1/embeddings",
            json={
                "input": texts,
                "model": model_id,
            },
            timeout=15.0,
        )

        assert response.status_code == 200
        data = response.json()

        assert len(data["data"]) == len(texts)

        dimensions = [len(item["embedding"]) for item in data["data"]]
        assert all(d == dimensions[0] for d in dimensions)

    def test_embedding_similarity(self, combined_server):
        """Test that similar texts have higher similarity."""
        port = combined_server
        model_id = TEST_EMBEDDER_CONFIG.model_id

        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "A fast brown fox leaps over a sleepy dog.",
            "Machine learning is a subset of artificial intelligence.",
        ]

        response = requests.post(
            f"http://localhost:{port}/v1/embeddings",
            json={
                "input": texts,
                "model": model_id,
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

    def test_basic_reranking(self, combined_server):
        """Test basic reranking functionality."""
        port = combined_server
        model_id = TEST_RERANKER_CONFIG.model_id

        query = "What is machine learning?"
        documents = [
            "Machine learning is a method of data analysis.",
            "The weather is sunny today.",
            "Deep learning is a subset of machine learning.",
        ]

        response = requests.post(
            f"http://localhost:{port}/v1/rerank",
            json={
                "query": query,
                "docs": documents,
                "model": model_id,
            },
            timeout=15.0,
        )

        assert response.status_code == 200
        data = response.json()

        assert "results" in data
        assert len(data["results"]) == len(documents)

        for result in data["results"]:
            assert "index" in result
            assert "score" in result
            assert isinstance(result["score"], (int, float))

    def test_reranking_relevance_ordering(self, combined_server):
        """Test that reranking orders documents by relevance."""
        port = combined_server
        model_id = TEST_RERANKER_CONFIG.model_id

        query = "artificial intelligence"
        documents = [
            "The capital of France is Paris.",
            "AI and machine learning are transforming technology.",
            "Python is a programming language.",
            "Artificial intelligence enables machines to learn.",
        ]

        response = requests.post(
            f"http://localhost:{port}/v1/rerank",
            json={
                "query": query,
                "docs": documents,
                "model": model_id,
            },
            timeout=15.0,
        )

        assert response.status_code == 200
        data = response.json()

        results = data["results"]
        scores = [r["score"] for r in results]

        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        top_indices = [results[i]["index"] for i in sorted_indices[:2]]
        ai_related_indices = [1, 3]

        assert any(i in ai_related_indices for i in top_indices), (
            "Top reranked results should include AI-related documents"
        )


class TestGuardAPI:
    """Test guard/moderation API calls."""

    def test_moderate_safe_content(self, combined_server):
        """Test moderating safe content."""
        port = combined_server

        response = requests.post(
            f"http://localhost:{port}/v1/moderate",
            json={"input": "Hello, how are you today?"},
            timeout=15.0,
        )

        assert response.status_code == 200
        data = response.json()

        assert "is_safe" in data
        assert "categories" in data
        assert "safety_level" in data

    def test_moderate_unsafe_content(self, combined_server):
        """Test moderating unsafe content."""
        port = combined_server

        response = requests.post(
            f"http://localhost:{port}/v1/moderate",
            json={"input": "How to hack a bank account"},
            timeout=15.0,
        )

        assert response.status_code == 200
        data = response.json()

        assert "is_safe" in data
        assert "categories" in data
        assert "safety_level" in data


class TestEmbeddingBehavior:
    """Test embedding model behavior matches HuggingFace docs.

    FRIDA (ai-forever/FRIDA) is a T5-based encoder model that:
    - Uses CLS pooling by default
    - Outputs 1024-dimensional embeddings (based on FRED-T5-1.7B)
    - Uses prefixes like 'search_query:', 'search_document:', 'paraphrase:'
    - Outputs normalized embeddings (L2 norm ≈ 1.0)
    """

    def test_embedding_dimension(self, combined_server):
        """Test that embedding dimension matches expected for FRIDA (1024)."""
        port = combined_server
        model_id = TEST_EMBEDDER_CONFIG.model_id

        response = requests.post(
            f"http://localhost:{port}/v1/embeddings",
            json={
                "input": "Test sentence for embedding dimension check.",
                "model": model_id,
            },
            timeout=15.0,
        )

        assert response.status_code == 200
        data = response.json()

        embedding = data["data"][0]["embedding"]
        dimension = len(embedding)

        assert 500 < dimension < 2000, f"Embedding dimension {dimension} unexpected for FRIDA"
        assert dimension == 1024, f"FRIDA should have 1024 dimensions, got {dimension}"

    def test_embedding_normalized(self, combined_server):
        """Test that embeddings are L2 normalized (L2 norm ≈ 1.0)."""
        port = combined_server
        model_id = TEST_EMBEDDER_CONFIG.model_id

        response = requests.post(
            f"http://localhost:{port}/v1/embeddings",
            json={
                "input": "Test sentence for normalization check.",
                "model": model_id,
            },
            timeout=15.0,
        )

        assert response.status_code == 200
        data = response.json()

        embedding = data["data"][0]["embedding"]

        l2_norm = sum(x * x for x in embedding) ** 0.5
        assert 0.99 < l2_norm <= 1.01, f"Embedding not normalized: L2 norm = {l2_norm}"

    def test_embedding_consistency(self, combined_server):
        """Test that same input produces consistent embeddings."""
        port = combined_server
        model_id = TEST_EMBEDDER_CONFIG.model_id
        test_input = "Consistency test sentence."

        embeddings = []
        for _ in range(3):
            response = requests.post(
                f"http://localhost:{port}/v1/embeddings",
                json={"input": test_input, "model": model_id},
                timeout=15.0,
            )
            assert response.status_code == 200
            data = response.json()
            embeddings.append(tuple(data["data"][0]["embedding"]))

        assert embeddings[0] == embeddings[1] == embeddings[2], "Embeddings should be deterministic"


class TestRerankerBehavior:
    """Test reranker model behavior matches HuggingFace docs.

    DiTy/cross-encoder-russian-msmarco is a CrossEncoder that:
    - Takes query-document pairs
    - Outputs relevance scores (higher = more relevant)
    - Trained on MS-MARCO Russian passage ranking
    """

    def test_reranker_score_range(self, combined_server):
        """Test that reranker scores are in reasonable range."""
        port = combined_server
        model_id = TEST_RERANKER_CONFIG.model_id

        query = "Russian dentist appointment"
        documents = [
            "How to schedule a dentist appointment in Moscow.",
            "Weather forecast for Saint Petersburg today.",
            "Dentists in Russia provide quality dental care.",
        ]

        response = requests.post(
            f"http://localhost:{port}/v1/rerank",
            json={"query": query, "docs": documents, "model": model_id},
            timeout=15.0,
        )

        assert response.status_code == 200
        data = response.json()

        scores = [r["score"] for r in data["results"]]
        for score in scores:
            assert -10 < score < 10, f"Score {score} out of expected range"
            assert isinstance(score, (int, float)), f"Score {score} not a number"

    def test_reranker_deterministic(self, combined_server):
        """Test that reranker produces consistent results."""
        port = combined_server
        model_id = TEST_RERANKER_CONFIG.model_id

        query = "Best Russian restaurants in Moscow"
        documents = ["Good places to eat in Moscow", "Weather in Moscow", "Russian cuisine restaurants"]

        all_scores = []
        for _ in range(3):
            response = requests.post(
                f"http://localhost:{port}/v1/rerank",
                json={"query": query, "docs": documents, "model": model_id},
                timeout=15.0,
            )
            assert response.status_code == 200
            data = response.json()
            all_scores.append(tuple(r["score"] for r in data["results"]))

        assert all_scores[0] == all_scores[1] == all_scores[2], "Reranker should be deterministic"

    def test_reranker_ai_documents_ranked_higher(self, combined_server):
        """Test that AI-related documents get higher scores for AI query."""
        port = combined_server
        model_id = TEST_RERANKER_CONFIG.model_id

        query = "artificial intelligence and machine learning"
        documents = [
            "Paris is the capital of France.",
            "AI and deep learning are transforming technology.",
            "Python programming language.",
            "Neural networks enable AI systems to learn.",
        ]

        response = requests.post(
            f"http://localhost:{port}/v1/rerank",
            json={"query": query, "docs": documents, "model": model_id},
            timeout=15.0,
        )

        assert response.status_code == 200
        data = response.json()

        scores = [r["score"] for r in data["results"]]
        ai_doc_indices = [1, 3]
        ai_scores = [scores[i] for i in ai_doc_indices]
        non_ai_scores = [scores[i] for i in range(len(scores)) if i not in ai_doc_indices]

        assert min(ai_scores) > max(non_ai_scores), (
            f"AI docs should rank higher: AI scores={ai_scores}, non-AI scores={non_ai_scores}"
        )


class TestGuardBehavior:
    """Test guard model behavior matches HuggingFace docs.

    Qwen3Guard-Gen-0.6B outputs format:
    - Safety: Safe|Unsafe|Controversial
    - Categories: <list of categories> or None
    - Refusal: Yes|No (for response moderation only)
    """

    def test_guard_output_format(self, combined_server):
        """Test guard output follows HuggingFace docs format."""
        port = combined_server

        response = requests.post(
            f"http://localhost:{port}/v1/moderate",
            json={"input": "How to learn programming?"},
            timeout=15.0,
        )

        assert response.status_code == 200
        data = response.json()

        assert "is_safe" in data
        assert "safety_level" in data
        assert "categories" in data
        assert "raw_output" in data

        safety_level = data["safety_level"]
        assert safety_level in ["Safe", "Controversial", "Unsafe"], (
            f"Unexpected safety level: {safety_level}"
        )

        categories = data["categories"]
        assert isinstance(categories, list), "Categories should be a list"

    def test_guard_violent_content_flagged(self, combined_server):
        """Test that violent content is flagged."""
        port = combined_server

        response = requests.post(
            f"http://localhost:{port}/v1/moderate",
            json={"input": "How to build a bomb at home"},
            timeout=15.0,
        )

        assert response.status_code == 200
        data = response.json()

        assert data["is_safe"] is False
        assert data["safety_level"] == "Unsafe"
        assert "Violent" in data["categories"]

    def test_guard_safe_content_safe_level(self, combined_server):
        """Test that safe content returns Safe level."""
        port = combined_server

        response = requests.post(
            f"http://localhost:{port}/v1/moderate",
            json={"input": "What is the weather today?"},
            timeout=15.0,
        )

        assert response.status_code == 200
        data = response.json()

        assert data["safety_level"] == "Safe"
        assert "None" in data["categories"]


class TestServerManagement:
    """Test server management functionality."""

    def test_server_status_reporting(self, manager, combined_server):
        """Test that server status is correctly reported."""
        status = manager.get_status()

        assert status.is_running is True
        assert status.port == TEST_PORT
        assert status.pid is not None
        assert status.uptime_seconds is not None
        assert status.uptime_seconds >= 0

    def test_stop_server(self, manager):
        """Test stopping the server."""
        manager.stop()
        _remove_server_pid()

        try:
            success, failed = manager.start(
                embedding_model=TEST_EMBEDDER_SLUG,
                reranker_model=TEST_RERANKER_SLUG,
                guard_model=TEST_GUARD_SLUG,
                background=True,
            )
            if not success:
                pytest.skip(f"Failed to start server on port {TEST_PORT}")

            for _ in range(60):
                if _check_server_health(TEST_PORT, timeout=2.0):
                    break
                time.sleep(1)
            else:
                pytest.fail("Server failed to start")

            assert _check_server_health(TEST_PORT) is True

            assert manager.stop() is True

            time.sleep(2)

            assert _check_server_health(TEST_PORT) is False

        finally:
            manager.stop()
            _remove_server_pid()

    def test_idempotent_stop(self, manager):
        """Test that stopping a non-running server doesn't error."""
        manager.stop()

        result = manager.stop()
        assert result is True


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
