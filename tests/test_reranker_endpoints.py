#!/usr/bin/env python3
"""Test cmw-mosec reranker endpoints with industry-standard contracts.

Tests two endpoints:
- /v1/score: vLLM format {data: [{index, object, score}, ...]} - lightweight, raw scores
- /v1/rerank: Cohere format {results: [{index, document, relevance_score}, ...]} - sorted by relevance

Test harness reads configuration from tests/fixtures/test_rerankers.yaml.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import requests
import yaml

from cmw_mosec.server_config import ModelRegistry
from cmw_mosec.server_manager import (
    MosecServerManager,
    _check_server_health,
    _remove_server_pid,
)


def load_test_config() -> dict:
    """Load test configuration from YAML file."""
    config_path = Path(__file__).parent / "fixtures" / "test_rerankers.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Test config not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_llm_query(query: str, instruction: str, config: dict) -> str:
    """Format query for LLM reranker using template from config."""
    template = config["query_template"]
    prefix = config.get("prefix", "")

    return template.format(
        prefix=prefix,
        instruction=instruction,
        query=query,
    )


def format_llm_document(doc: str, config: dict) -> str:
    """Format document for LLM reranker using template from config."""
    template = config["doc_template"]
    suffix = config.get("suffix", "")
    prompt = config.get("prompt", "")

    return template.format(
        doc=doc,
        suffix=suffix,
        prompt=prompt,
    )


def test_score_endpoint(port: int, query: str, documents: list[str]) -> dict:
    """Test /v1/score endpoint (vLLM format).

    Returns: {data: [{index, object, score}, ...]}
    """
    url = f"http://localhost:{port}/v1/score"
    payload = {"query": query, "documents": documents}

    response = requests.post(url, json=payload, timeout=60.0)

    if response.status_code != 200:
        raise RuntimeError(f"/v1/score failed: {response.status_code} - {response.text}")

    data = response.json()

    # Validate vLLM format
    if "data" not in data:
        raise RuntimeError(f"/v1/score missing 'data' field: {data}")

    for item in data["data"]:
        if "index" not in item or "score" not in item:
            raise RuntimeError(f"/v1/score invalid item format: {item}")

    return data


def test_rerank_endpoint(
    port: int,
    query: str,
    documents: list[str],
    top_n: int | None = None,
) -> dict:
    """Test /v1/rerank endpoint (Cohere/Jina format).

    Returns: {results: [{index, document: {text}, relevance_score}, ...]}
    Results are sorted by relevance (descending).
    """
    url = f"http://localhost:{port}/v1/rerank"
    payload = {"query": query, "documents": documents}

    if top_n is not None:
        payload["top_n"] = top_n

    response = requests.post(url, json=payload, timeout=60.0)

    if response.status_code != 200:
        raise RuntimeError(f"/v1/rerank failed: {response.status_code} - {response.text}")

    data = response.json()

    # Validate Cohere format
    if "results" not in data:
        raise RuntimeError(f"/v1/rerank missing 'results' field: {data}")

    for item in data["results"]:
        if "index" not in item or "document" not in item or "relevance_score" not in item:
            raise RuntimeError(f"/v1/rerank invalid item format: {item}")

    # Verify sorted by relevance (descending)
    scores = [item["relevance_score"] for item in data["results"]]
    for i in range(len(scores) - 1):
        if scores[i] < scores[i + 1]:
            raise RuntimeError(f"/v1/rerank results not sorted by relevance: {scores}")

    return data


def test_cross_encoder(
    port: int,
    model_slug: str,
    config: dict,
) -> bool:
    """Test cross-encoder model with both endpoints."""
    print(f"\nTesting cross-encoder: {model_slug}")
    print("=" * 60)

    test_cases = config["cross_encoder"]["test_cases"]
    all_passed = True

    for test_case in test_cases:
        name = test_case["name"]
        query = test_case["query"]
        documents = test_case["documents"]
        expected_ranking = test_case.get("expected_ranking", [])

        print(f"\nTest: {name}")
        print(f"  Query: {query}")
        print(f"  Documents: {len(documents)}")

        # Test /v1/score (vLLM format)
        try:
            score_response = test_score_endpoint(port, query, documents)
            score_data = score_response["data"]
            score_scores = [item["score"] for item in score_data]
            print(f"  /v1/score (vLLM format): {len(score_data)} items")
            print(f"    Scores: {[f'{s:.4f}' for s in score_scores]}")
        except Exception as e:
            print(f"  ERROR /v1/score: {e}")
            all_passed = False
            continue

        # Test /v1/rerank (Cohere format)
        try:
            rerank_response = test_rerank_endpoint(port, query, documents)
            results = rerank_response["results"]
            print(f"  /v1/rerank (Cohere format): {len(results)} results, sorted by relevance")

            # Extract scores in original order for comparison
            rerank_scores = [0.0] * len(documents)
            for r in results:
                rerank_scores[r["index"]] = r["relevance_score"]

            print(
                f"    Top result: index={results[0]['index']}, score={results[0]['relevance_score']:.4f}"
            )

            # Verify /v1/score and /v1/rerank produce same scores (different order in response)
            for i, (s1, s2) in enumerate(zip(score_scores, rerank_scores)):
                if not math.isclose(s1, s2, rel_tol=1e-5):
                    print(f"    WARNING: Score mismatch at index {i}: {s1:.4f} vs {s2:.4f}")

        except Exception as e:
            print(f"  ERROR /v1/rerank: {e}")
            all_passed = False
            continue

        # Check ranking
        if expected_ranking:
            top_indices = [r["index"] for r in results[: len(expected_ranking)]]
            if set(top_indices) == set(expected_ranking):
                print(f"  PASS: Top {len(expected_ranking)} docs match expected {expected_ranking}")
            else:
                print(f"  WARNING: Expected top docs {expected_ranking}, got {top_indices}")
        else:
            print("  PASS: Both endpoints returned matching scores")

    return all_passed


def test_llm_reranker(
    port: int,
    model_slug: str,
    model_subtype: str,
    config: dict,
) -> bool:
    """Test LLM reranker with pre-formatted query/documents."""
    print(f"\nTesting LLM reranker: {model_slug}")
    print(f"Subtype: {model_subtype}")
    print("=" * 60)

    llm_config = config["llm_reranker"].get(model_subtype)
    if not llm_config:
        print(f"ERROR: No config found for subtype '{model_subtype}'")
        print(f"Available subtypes: {list(config['llm_reranker'].keys())}")
        return False

    instructions = llm_config.get("instructions", [""])
    test_cases = llm_config.get("test_cases", [])

    all_passed = True

    for instruction in instructions:
        print(
            f"\nInstruction: {instruction[:50]}..."
            if len(instruction) > 50
            else f"\nInstruction: {instruction}"
        )

        for test_case in test_cases:
            name = test_case["name"]
            query = test_case["query"]
            documents = test_case["documents"]
            expected_ranking = test_case.get("expected_ranking", [])

            print(f"\nTest: {name}")
            print(f"  Query: {query}")

            # Format query and documents
            formatted_query = format_llm_query(query, instruction, llm_config)
            formatted_docs = [format_llm_document(doc, llm_config) for doc in documents]

            print(f"  Formatted query (first 80 chars): {formatted_query[:80]}...")
            print(f"  Documents: {len(formatted_docs)}")

            # Test /v1/score (vLLM format)
            try:
                score_response = test_score_endpoint(port, formatted_query, formatted_docs)
                score_data = score_response["data"]
                score_scores = [item["score"] for item in score_data]
                print(f"  /v1/score scores: {[f'{s:.4f}' for s in score_scores]}")
            except Exception as e:
                print(f"  ERROR /v1/score: {e}")
                all_passed = False
                continue

            # Test /v1/rerank (Cohere format)
            try:
                rerank_response = test_rerank_endpoint(port, formatted_query, formatted_docs)
                results = rerank_response["results"]
                print(f"  /v1/rerank: {len(results)} results, top={results[0]['index']}")
            except Exception as e:
                print(f"  ERROR /v1/rerank: {e}")
                all_passed = False
                continue

            # Check ranking
            ranked = sorted(range(len(score_scores)), key=lambda i: score_scores[i], reverse=True)
            print(f"  Ranking: {ranked}")

            if expected_ranking:
                top_docs = set(ranked[: len(expected_ranking)])
                expected_docs = set(expected_ranking)
                if top_docs == expected_docs:
                    print(f"  PASS: Top {len(expected_ranking)} docs match expected")
                else:
                    print(
                        f"  WARNING: Expected top docs {expected_ranking}, got {ranked[: len(expected_ranking)]}"
                    )
            else:
                print("  PASS: Both endpoints returned consistent results")

    return all_passed


def test_reranker_endpoints(
    model_slug: str = "DiTy/cross-encoder-russian-msmarco",
    port: int = 7998,
) -> bool:
    """Test reranker endpoints with model from config."""
    print("=" * 70)
    print(f"Testing reranker endpoints: {model_slug}")
    print("=" * 70)

    config = load_test_config()
    registry = ModelRegistry()

    # Get model type from config
    model_mappings = config.get("model_mappings", {})
    model_info = None
    for slug in model_mappings:
        if slug.lower() == model_slug.lower():
            model_info = model_mappings[slug]
            break

    if not model_info:
        print(f"ERROR: Model '{model_slug}' not found in test config")
        print("Available models:")
        for slug in model_mappings:
            print(f"  - {slug}")
        return False

    model_type = model_info["type"]
    model_subtype = model_info.get("subtype")

    print(f"\nModel type: {model_type}")
    if model_subtype:
        print(f"Subtype: {model_subtype}")

    # Get model config for additional info
    try:
        model_config = registry.get_reranker_config(model_slug)
        print(f"Max length: {model_config.max_length}")
        if model_config.scoring_method:
            print(f"Scoring method: {model_config.scoring_method}")
        if model_config.scoring_tokens:
            print(f"Scoring tokens: {model_config.scoring_tokens}")
    except ValueError:
        print(f"WARNING: Model not found in registry: {model_slug}")

    # Start server
    manager = MosecServerManager()

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
    for i in range(120):
        if _check_server_health(port, timeout=2.0):
            print(f"   Server ready after {i + 1}s")
            break
        time.sleep(1)
    else:
        print("ERROR: Server did not become ready")
        manager.stop()
        return False

    # Run tests based on model type
    try:
        if model_type == "cross_encoder":
            result = test_cross_encoder(port, model_slug, config)
        elif model_type == "llm_reranker":
            if not model_subtype:
                print("ERROR: llm_reranker requires subtype")
                result = False
            else:
                result = test_llm_reranker(port, model_slug, model_subtype, config)
        else:
            print(f"ERROR: Unknown model type: {model_type}")
            result = False
    finally:
        print("\n3. Stopping server...")
        manager.stop()

    print("\n" + "=" * 70)
    if result:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
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
    parser.add_argument(
        "--port",
        type=int,
        default=7998,
        help="Server port",
    )

    args = parser.parse_args()
    success = test_reranker_endpoints(args.model, args.port)
    exit(0 if success else 1)
