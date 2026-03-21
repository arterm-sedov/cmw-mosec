#!/usr/bin/env python3
"""Test cmw-mosec reranker endpoints (/v1/score and /v1/rerank).

Test harness reads configuration from tests/fixtures/test_rerankers.yaml.
Tests both cross-encoder and LLM reranker models.

Usage:
    pytest tests/test_reranker_endpoints.py -v
    python tests/test_reranker_endpoints.py --model DiTy/cross-encoder-russian-msmarco
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


def format_llm_query(
    query: str,
    instruction: str,
    config: dict,
) -> str:
    """Format query for LLM reranker using template from config."""
    template = config["query_template"]
    prefix = config.get("prefix", "")

    return template.format(
        prefix=prefix,
        instruction=instruction,
        query=query,
    )


def format_llm_document(
    doc: str,
    config: dict,
) -> str:
    """Format document for LLM reranker using template from config."""
    template = config["doc_template"]
    suffix = config.get("suffix", "")
    prompt = config.get("prompt", "")

    return template.format(
        doc=doc,
        suffix=suffix,
        prompt=prompt,
    )


def test_endpoint(
    port: int,
    endpoint: str,
    query: str,
    documents: list[str],
) -> dict:
    """Test an endpoint and return the response."""
    url = f"http://localhost:{port}{endpoint}"
    payload = {
        "query": query,
        "documents": documents,
    }

    response = requests.post(url, json=payload, timeout=60.0)

    if response.status_code != 200:
        raise RuntimeError(f"Endpoint {endpoint} failed: {response.status_code} - {response.text}")

    return response.json()


def test_cross_encoder(
    port: int,
    model_slug: str,
    config: dict,
) -> bool:
    """Test cross-encoder model with raw query/documents."""
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

        # Test /v1/rerank
        try:
            rerank_response = test_endpoint(port, "/v1/rerank", query, documents)
            rerank_scores = rerank_response.get("scores", [])
        except Exception as e:
            print(f"  ERROR /v1/rerank: {e}")
            all_passed = False
            continue

        # Test /v1/score
        try:
            score_response = test_endpoint(port, "/v1/score", query, documents)
            score_scores = score_response.get("scores", [])
        except Exception as e:
            print(f"  ERROR /v1/score: {e}")
            all_passed = False
            continue

        # Compare endpoints
        if len(rerank_scores) != len(documents) or len(score_scores) != len(documents):
            print(f"  ERROR: Expected {len(documents)} scores")
            all_passed = False
            continue

        print(f"  /v1/rerank scores: {[f'{s:.4f}' for s in rerank_scores]}")
        print(f"  /v1/score scores:  {[f'{s:.4f}' for s in score_scores]}")

        # Check endpoints match
        all_close = all(
            math.isclose(a, b, rel_tol=1e-5)
            for a, b in zip(rerank_scores, score_scores, strict=False)
        )
        if not all_close:
            print("  ERROR: Endpoints returned different scores")
            all_passed = False
            continue

        # Check ranking
        ranked = sorted(range(len(rerank_scores)), key=lambda i: rerank_scores[i], reverse=True)
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

    # Get formatting config for this model subtype
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

            # Test /v1/score
            try:
                score_response = test_endpoint(port, "/v1/score", formatted_query, formatted_docs)
                score_scores = score_response.get("scores", [])
            except Exception as e:
                print(f"  ERROR /v1/score: {e}")
                all_passed = False
                continue

            if len(score_scores) != len(formatted_docs):
                print(f"  ERROR: Expected {len(formatted_docs)} scores, got {len(score_scores)}")
                all_passed = False
                continue

            print(f"  /v1/score scores: {[f'{s:.4f}' for s in score_scores]}")

            # Test /v1/rerank with same formatted input
            try:
                rerank_response = test_endpoint(port, "/v1/rerank", formatted_query, formatted_docs)
                rerank_scores = rerank_response.get("scores", [])
            except Exception as e:
                print(f"  ERROR /v1/rerank: {e}")
                all_passed = False
                continue

            # Check endpoints match
            all_close = all(
                math.isclose(a, b, rel_tol=1e-5)
                for a, b in zip(rerank_scores, score_scores, strict=False)
            )
            if not all_close:
                print("  ERROR: Endpoints returned different scores")
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
                print("  PASS: Both endpoints returned matching scores")

    return all_passed


def test_reranker_endpoints(
    model_slug: str = "DiTy/cross-encoder-russian-msmarco",
    port: int = 7998,
) -> bool:
    """Test reranker endpoints with model from config."""
    print("=" * 70)
    print(f"Testing reranker endpoints: {model_slug}")
    print("=" * 70)

    # Load config
    config = load_test_config()
    registry = ModelRegistry()

    # Get model type from config
    model_mappings = config.get("model_mappings", {})
    model_info = model_mappings.get(model_slug.lower(), {})

    if not model_info:
        # Try case-insensitive lookup
        for key in model_mappings:
            if key.lower() == model_slug.lower():
                model_info = model_mappings[key]
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
