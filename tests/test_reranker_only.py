#!/usr/bin/env python3
"""Test cmw-mosec with single reranker model (DiTy).

This script tests starting the server with only the DiTy/cross-encoder-russian-msmarco
reranker model and verifies basic functionality.
"""

from __future__ import annotations

import time

import requests

from cmw_mosec.server_config import ModelRegistry, load_server_settings
from cmw_mosec.server_manager import (
    MosecServerManager,
    _check_server_health,
    _remove_server_pid,
)





def test_reranker_only():
    """Test server with only reranker model."""
    print("=" * 60)
    print("Testing cmw-mosec with DiTy reranker only")
    print("=" * 60)

    settings = load_server_settings()
    port = settings.server_port

    manager = MosecServerManager()

    # Clean up any existing server
    print("\n1. Cleaning up any existing server...")
    manager.stop()
    _remove_server_pid()

    # Get reranker config
    print("\n2. Loading reranker configuration...")
    registry = ModelRegistry()
    reranker_slug = "DiTy/cross-encoder-russian-msmarco"
    config = registry.get_reranker_config(reranker_slug)
    print(f"   Model: {config.model_id}")
    print(f"   Type: {config.model_type}")
    print(f"   Device: {config.device}")
    print(f"   Batch size: {config.batch_size}")

    # Start server with only reranker
    print(f"\n3. Starting server with only reranker on port {port}...")
    success, failed = manager.start(
        embedding_model=None,
        reranker_model=reranker_slug,
        guard_model=None,
        background=True,
    )

    if not success:
        print(f"   FAILED: Could not start server. Failed models: {failed}")
        return False

    print("   Server process started, waiting for readiness...")

    # Wait for server to be ready (using metrics endpoint)
    for i in range(60):
        if _check_server_health(port, timeout=2.0):
            print(f"   Server is ready after {i + 1} seconds!")
            break
        time.sleep(1)
    else:
        print("   FAILED: Server did not become ready within timeout")
        manager.stop()
        return False

    # Test metrics endpoint
    print("\n4. Testing metrics endpoint...")
    try:
        response = requests.get(f"http://localhost:{port}/metrics", timeout=5.0)
        if response.status_code == 200:
            print("   Metrics check: PASSED")
        else:
            print(f"   Metrics check: FAILED (status {response.status_code})")
            manager.stop()
            return False
    except Exception as e:
        print(f"   Metrics check: FAILED ({e})")
        manager.stop()
        return False

    # Test reranking
    print("\n5. Testing reranking API...")
    query = "What is machine learning?"
    documents = [
        "Machine learning is a method of data analysis.",
        "The weather is sunny today.",
        "Deep learning is a subset of machine learning.",
    ]

    try:
        response = requests.post(
            f"http://localhost:{port}/v1/rerank",
            json={
                "query": query,
                "documents": documents,
                "model": config.model_id,
                "top_k": 3,
            },
            timeout=15.0,
        )

        if response.status_code == 200:
            data = response.json()
            scores = data.get("scores", [])
            print(f"   Query: {query}")
            print(f"   Documents: {len(documents)}")
            print(f"   Scores: {scores}")

            if len(scores) == len(documents):
                print("   Reranking: PASSED")

                # Verify AI-related docs get higher scores
                ai_scores = [scores[0], scores[2]]  # docs 0 and 2 are AI-related
                non_ai_score = scores[1]  # doc 1 is about weather

                if min(ai_scores) > non_ai_score:
                    print("   Relevance ordering: PASSED (AI docs ranked higher)")
                else:
                    print("   Relevance ordering: WARNING (AI docs not clearly ranked higher)")
            else:
                print(f"   Reranking: FAILED (expected {len(documents)} scores, got {len(scores)})")
                manager.stop()
                return False
        else:
            print(f"   Reranking: FAILED (status {response.status_code})")
            print(f"   Response: {response.text}")
            manager.stop()
            return False
    except Exception as e:
        print(f"   Reranking: FAILED ({e})")
        manager.stop()
        return False

    # Test server status
    print("\n6. Testing server status...")
    status = manager.get_status()
    print(f"   Running: {status.is_running}")
    print(f"   Port: {status.port}")
    print(f"   PID: {status.pid}")
    uptime_str = f"{status.uptime_seconds:.1f}s" if status.uptime_seconds is not None else "N/A"
    print(f"   Uptime: {uptime_str}")

    # Stop server
    print("\n7. Stopping server...")
    if manager.stop():
        print("   Server stopped: PASSED")
    else:
        print("   Server stop: FAILED")
        return False

    # Verify server is down
    time.sleep(2)
    if not _check_server_health(port):
        print("   Server shutdown verified: PASSED")
    else:
        print("   Server shutdown: FAILED (still responding)")
        return False

    print("\n" + "=" * 60)
    print("All tests PASSED!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    import sys

    success = test_reranker_only()
    sys.exit(0 if success else 1)
