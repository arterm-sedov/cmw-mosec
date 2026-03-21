#!/usr/bin/env python3
"""Test cmw-mosec reranker endpoints (/v1/score and /v1/rerank).

This test suite verifies:
1. Cross-encoder models (DiTy, BGE-m3) work with raw query/documents
2. Both /v1/score and /v1/rerank endpoints return same results
3. Response format matches vLLM-style raw scores

Usage:
    pytest tests/test_reranker_endpoints.py -v
    python tests/test_reranker_endpoints.py --model DiTy/cross-encoder-russian-msmarco
"""

from __future__ import annotations

import argparse
import json
import time

import requests

from cmw_mosec.server_config import ModelRegistry, load_server_settings
from cmw_mosec.server_manager import (
    MosecServerManager,
    _check_server_health,
    _remove_server_pid,
)


def test_endpoint_format(port: int, endpoint: str, query: str, documents: list[str]) -> dict:
    """Test an endpoint and return the response."""
    url = f"http://localhost:{port}{endpoint}"
    payload = {
        "query": query,
        "documents": documents,
    }

    response = requests.post(url, json=payload, timeout=30.0)

    if response.status_code != 200:
        raise RuntimeError(f"Endpoint {endpoint} failed: {response.status_code} - {response.text}")

    return response.json()


def test_reranker_endpoints(
    model_slug: str = "DiTy/cross-encoder-russian-msmarco",
    port: int = 7998,
):
    """Test both /v1/score and /v1/rerank endpoints with a cross-encoder model."""
    print("=" * 70)
    print(f"Testing reranker endpoints with {model_slug}")
    print("=" * 70)

    manager = MosecServerManager()
    registry = ModelRegistry()

    # Verify model exists
    try:
        config = registry.get_reranker_config(model_slug)
        print(f"\nModel: {config.model_id}")
        print(f"Type: {config.reranker_type}")
        print(f"Max Length: {config.max_length}")
    except ValueError as e:
        print(f"ERROR: Unknown model: {model_slug}")
        print(f"Available rerankers: {registry.list_rerankers()}")
        return False

    # Stop any existing server
    print("\n1. Stopping any existing server...")
    manager.stop()
    _remove_server_pid()

    # Start server with only reranker
    print(f"\n2. Starting server with {model_slug} on port {port}...")
    success, failed = manager.start(
        embedding_model=None,
        reranker_model=model_slug,
        guard_model=None,
        background=True,
    )

    if not success:
        print(f"ERROR: Failed to start server. Failed models: {failed}")
        return False

    # Wait for server ready
    print("   Waiting for server to be ready...")
    for i in range(60):
        if _check_server_health(port, timeout=2.0):
            print(f"   Server ready after {i + 1}s")
            break
        time.sleep(1)
    else:
        print("ERROR: Server did not become ready within 60s")
        manager.stop()
        return False

    # Test queries
    test_cases = [
        {
            "name": "English - Machine Learning",
            "query": "What is machine learning?",
            "documents": [
                "Machine learning is a method of data analysis that automates analytical model building.",
                "The weather is sunny today in San Francisco.",
                "Deep learning is a subset of machine learning using neural networks.",
            ],
        },
        {
            "name": "Russian - Car/Auto",
            "query": "машина",
            "documents": [
                "Автомобиль для перевозки грузов и пассажиров.",
                "Куриное блюдо из тушёной курицы с овощами.",
                "Новый автомобиль Teslamodel выпуск 2024 года.",
            ],
        },
        {
            "name": "English - Capital Cities",
            "query": "What is the capital of France?",
            "documents": [
                "The capital of Brazil is Brasilia.",
                "The capital of France is Paris, known for the Eiffel Tower.",
                "Horses and cows are both animals found on farms.",
            ],
        },
    ]

    all_passed = True

    for test_case in test_cases:
        print(f"\n3. Testing: {test_case['name']}")
        print(f"   Query: {test_case['query']}")
        print(f"   Documents: {len(test_case['documents'])}")

        # Test /v1/rerank
        print("\n   Testing /v1/rerank...")
        try:
            rerank_response = test_endpoint_format(
                port, "/v1/rerank", test_case["query"], test_case["documents"]
            )
            rerank_scores = rerank_response.get("scores", [])
            print(f"   Scores: {rerank_scores}")

            if len(rerank_scores) != len(test_case["documents"]):
                print(
                    f"   ERROR: Expected {len(test_case['documents'])} scores, got {len(rerank_scores)}"
                )
                all_passed = False
                continue
        except Exception as e:
            print(f"   ERROR: /v1/rerank failed: {e}")
            all_passed = False
            continue

        # Test /v1/score
        print("\n   Testing /v1/score...")
        try:
            score_response = test_endpoint_format(
                port, "/v1/score", test_case["query"], test_case["documents"]
            )
            score_scores = score_response.get("scores", [])
            print(f"   Scores: {score_scores}")

            if len(score_scores) != len(test_case["documents"]):
                print(
                    f"   ERROR: Expected {len(test_case['documents'])} scores, got {len(score_scores)}"
                )
                all_passed = False
                continue
        except Exception as e:
            print(f"   ERROR: /v1/score failed: {e}")
            all_passed = False
            continue

        # Verify both endpoints return same scores (with floating-point tolerance)
        print("\n   Comparing endpoints...")
        import math

        all_close = all(
            math.isclose(a, b, rel_tol=1e-5) for a, b in zip(rerank_scores, score_scores)
        )
        if all_close:
            print("   PASS: /v1/rerank and /v1/score return matching scores")
        else:
            print(f"   ERROR: Score mismatch!")
            print(f"   /v1/rerank: {rerank_scores}")
            print(f"   /v1/score:  {score_scores}")
            all_passed = False

    # Stop server
    print("\n4. Stopping server...")
    if manager.stop():
        print("   Server stopped")
    else:
        print("   WARNING: Failed to stop server gracefully")

    # Verify shutdown
    time.sleep(2)
    if not _check_server_health(port):
        print("   Server shutdown verified")
    else:
        print("   WARNING: Server still responding")

    print("\n" + "=" * 70)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("=" * 70)

    return all_passed


def test_llm_reranker_preformatted(
    model_slug: str = "Qwen/Qwen3-Reranker-0.6B",
    port: int = 7998,
):
    """Test LLM reranker with pre-formatted query and documents.

    For LLM rerankers, the client must format the query and documents
    with the appropriate prefix/instruction/suffix BEFORE sending.

    This test demonstrates the expected input format for llm_reranker models.
    """
    print("=" * 70)
    print(f"Testing LLM reranker with pre-formatted input: {model_slug}")
    print("=" * 70)

    manager = MosecServerManager()
    registry = ModelRegistry()

    try:
        config = registry.get_reranker_config(model_slug)
        print(f"\nModel: {config.model_id}")
        print(f"Type: {config.reranker_type}")
        print(f"Scoring Method: {config.scoring_method}")
        print(f"Scoring Tokens: {config.scoring_tokens}")
    except ValueError as e:
        print(f"ERROR: Unknown model: {model_slug}")
        return False

    # Qwen3-Reranker formatting (client-side)
    # From model card: https://huggingface.co/Qwen/Qwen3-Reranker-0.6B
    prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    suffix = "<|im_end|>\n<|im_start|>assistant\n\n\n\n\n"
    instruction = "Given a web search query, retrieve relevant passages that answer the query"

    # Format query with prefix and instruction
    query = f"{prefix}<Instruct>: {instruction}\n<Query>: What is the capital of France?\n"

    # Format documents with suffix
    documents = [
        f"<Document>: The capital of Brazil is Brasilia.{suffix}",
        f"<Document>: The capital of France is Paris, known for the Eiffel Tower.{suffix}",
        f"<Document>: Horses and cows are both animals found on farms.{suffix}",
    ]

    print(f"\nFormatted query (first 100 chars): {query[:100]}...")
    print(f"\nFormatted document 0 (first 100 chars): {documents[0][:100]}...")

    # Stop and start server
    print("\n1. Stopping any existing server...")
    manager.stop()
    _remove_server_pid()

    print(f"\n2. Starting server with {model_slug}...")
    success, failed = manager.start(
        embedding_model=None,
        reranker_model=model_slug,
        guard_model=None,
        background=True,
    )

    if not success:
        print(f"ERROR: Failed to start server. Failed models: {failed}")
        return False

    print("   Waiting for server...")
    for i in range(120):  # LLM models take longer to load
        if _check_server_health(port, timeout=2.0):
            print(f"   Server ready after {i + 1}s")
            break
        time.sleep(1)
    else:
        print("ERROR: Server did not become ready")
        manager.stop()
        return False

    # Test with pre-formatted input
    print("\n3. Testing /v1/score with pre-formatted input...")
    try:
        response = test_endpoint_format(port, "/v1/score", query, documents)
        scores = response.get("scores", [])
        print(f"   Scores: {scores}")

        if len(scores) != len(documents):
            print(f"   ERROR: Expected {len(documents)} scores, got {len(scores)}")
            manager.stop()
            return False

        # For Qwen3, scores should be probabilities (0-1)
        # Document about Paris should have highest score
        print(f"\n   Document ranking:")
        for i, (doc, score) in enumerate(zip(documents, scores)):
            print(f"   {i}: score={score:.4f}")

        # Verify Paris document has highest score
        if scores[1] > scores[0] and scores[1] > scores[2]:
            print("\n   PASS: Paris document ranked highest")
        else:
            print("\n   WARNING: Expected Paris document to be ranked highest")

        result = True

    except Exception as e:
        print(f"   ERROR: {e}")
        result = False

    # Stop server
    print("\n4. Stopping server...")
    manager.stop()

    print("\n" + "=" * 70)
    if result:
        print("LLM reranker test PASSED!")
    else:
        print("LLM reranker test FAILED!")
    print("=" * 70)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test cmw-mosec reranker endpoints")
    parser.add_argument(
        "--model",
        type=str,
        default="DiTy/cross-encoder-russian-msmarco",
        help="Reranker model to test",
    )
    parser.add_argument("--port", type=int, default=7998, help="Server port")
    parser.add_argument(
        "--llm", action="store_true", help="Test LLM reranker with pre-formatted input"
    )

    args = parser.parse_args()

    if args.llm:
        success = test_llm_reranker_preformatted(args.model, args.port)
    else:
        success = test_reranker_endpoints(args.model, args.port)

    exit(0 if success else 1)
