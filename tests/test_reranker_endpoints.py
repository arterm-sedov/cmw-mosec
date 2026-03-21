#!/usr/bin/env python3
"""Test cmw-mosec reranker endpoints with vLLM-compatible contracts.

Tests two endpoints:
- /v1/score: Returns vLLM format {data: [{index, object, score}, ...]}
- /v1/rerank: Returns Cohere/Jina format {results: [{index, document, relevance_score}, ...]}
                          OR simple format {scores: [...]}

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
    payload = {
        "query": query,
        "documents": documents,
        "response_format": "vllm_score",
    }

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
    return_documents: bool = False,
    top_n: int | None = None,
) -> dict:
    """Test /v1/rerank endpoint.

    Args:
        port: Server port
        query: Query string
        documents: List of documents
        return_documents: If True, returns Cohere/Jina format with results sorted by relevance
        top_n: Ifreturn_documents=True, limit to top N results

    Returns:
        If return_documents=False: {scores: [...]}(simple format)
        If return_documents=True: {results: [{index, document, relevance_score}, ...]}
    """
    url = f"http://localhost:{port}/v1/rerank"
    payload = {
        "query": query,
        "documents": documents,
        "return_documents": return_documents,
    }

    if top_n is not None:
        payload["top_n"] = top_n

    response = requests.post(url, json=payload, timeout=60.0)

    if response.status_code != 200:
        raise RuntimeError(f"/v1/rerank failed: {response.status_code} - {response.text}")

    return response.json()


def test_cross_encoder(
    port: int,
    model_slug: str,
    config: dict,
) -> bool:
    """Test cross-encoder model with both endpoint formats."""
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
            # Convert to simple list for comparison
            score_scores = [item["score"] for item in score_data]
            print(f"  /v1/score (vLLM format): {len(score_data)} items")
            print(f"    Scores: {[f'{s:.4f}' for s in score_scores]}")
        except Exception as e:
            print(f"  ERROR /v1/score: {e}")
            all_passed = False
            continue

        # Test /v1/rerank (simple format)
        try:
            rerank_response = test_rerank_endpoint(port, query, documents, return_documents=False)
            rerank_scores = rerank_response["scores"]
            print(f"  /v1/rerank (simple format): {[f'{s:.4f}' for s in rerank_scores]}")
        except Exception as e:
            print(f"  ERROR /v1/rerank: {e}")
            all_passed = False
            continue

        # Test /v1/rerank (Cohere/Jina format)
        try:
            rerank_cohere = test_rerank_endpoint(port, query, documents, return_documents=True)
            results = rerank_cohere["results"]
            # Verify results are sorted by relevance (descending)
            for i in range(len(results) - 1):
                if results[i]["relevance_score"] < results[i + 1]["relevance_score"]:
                    print("  ERROR: Results not sorted by relevance")
                    all_passed = False
                    continue
            print(f"  /v1/rerank (Cohere format): {len(results)} results, sorted by relevance")
            print(
                f"    Top result: index={results[0]['index']}, score={results[0]['relevance_score']:.4f}"
            )
        except Exception as e:
            print(f"  ERROR /v1/rerank (Cohere): {e}")
            all_passed = False
            continue

        # Verify all formats produce same scores (ignoring order)
        simple_scores = sorted(score_scores, reverse=True)
        rerank_scores_sorted = sorted(rerank_scores, reverse=True)
        cohere_scores = sorted([r["relevance_score"] for r in results], reverse=True)

        if not all(
            math.isclose(a, b, rel_tol=1e-5)
            for a, b in zip(simple_scores, rerank_scores_sorted, strict=False)
        ):
            print("  ERROR: /v1/score and /v1/rerank produce different scores")
            all_passed = False
            continue

        if not all(
            math.isclose(a, b, rel_tol=1e-5)
            for a, b in zip(simple_scores, cohere_scores, strict=False)
        ):
            print("  ERROR: /v1/score and /v1/rerank (Cohere) produce different scores")
            all_passed = False
            continue

        # Check ranking
        if expected_ranking:
            ranked = sorted(range(len(rerank_scores)), key=lambda i: rerank_scores[i], reverse=True)
            top_docs = set(ranked[: len(expected_ranking)])
            expected_docs = set(expected_ranking)
            if top_docs == expected_docs:
                print(f"  PASS: Top {len(expected_ranking)} docs match expected {expected_ranking}")
            else:
                print(
                    f"  WARNING: Expected top docs {expected_ranking}, got {ranked[: len(expected_ranking)]}"
                )
        else:
            print("  PASS: All formats produce consistent scores")

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

            # Test /v1/rerank (simple and Cohere formats)
            try:
                rerank_response = test_rerank_endpoint(
                    port, formatted_query, formatted_docs, return_documents=False
                )
                rerank_scores = rerank_response["scores"]

                rerank_cohere = test_rerank_endpoint(
                    port, formatted_query, formatted_docs, return_documents=True
                )
                cohere_results = rerank_cohere["results"]

                if len(score_scores) != len(documents):
                    print(f"  ERROR: Expected {len(documents)} scores, got {len(score_scores)}")
                    all_passed = False
                    continue

                print(f"  /v1/rerank scores: {[f'{s:.4f}' for s in rerank_scores]}")
                print(
                    f"  /v1/rerank (Cohere): {len(cohere_results)} results, top={cohere_results[0]['index']}"
                )

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
                print("  PASS: All formats produce consistent results")

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
