#!/usr/bin/env python3
"""
Qwen3 Embedding Examples

This script demonstrates proper usage of Qwen3 embedding models with Mosec.

Key points:
1. Use instruction format for queries (required per HF docs)
2. Documents don't need instruction prefix
3. Mosec handles last-token pooling automatically

References:
- https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
- https://huggingface.co/Qwen/Qwen3-Embedding-0.6B#vllm-usage
"""

import numpy as np
import requests


def get_detailed_instruct(task_description: str, query: str) -> str:
    """
    Format query with instruction for Qwen3 embedding models.

    Per HuggingFace docs, queries must use this format:
    'Instruct: {task}\nQuery: {query}'

    Args:
        task_description: Task description (e.g., "Given a web search query...")
        query: The actual query text

    Returns:
        Formatted text with instruction
    """
    return f"Instruct: {task_description}\nQuery: {query}"


def embed_with_mosec(
    text: str, model: str = "Qwen/Qwen3-Embedding-0.6B", base_url: str = "http://localhost:8001"
) -> np.ndarray:
    """
    Get embeddings from Mosec server.

    Args:
        text: Text to embed (with instruction for queries)
        model: Model ID
        base_url: Mosec server URL

    Returns:
        Embedding vector as numpy array
    """
    response = requests.post(
        f"{base_url}/v1/embeddings", json={"model": model, "input": text}, timeout=30
    )
    response.raise_for_status()
    result = response.json()
    return np.array(result["data"][0]["embedding"])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def example_1_basic_query_document():
    """Example 1: Basic query-document retrieval."""
    print("=" * 70)
    print("Example 1: Basic Query-Document Retrieval")
    print("=" * 70)

    # Task description (from HF docs)
    task = "Given a web search query, retrieve relevant passages that answer the query"

    # Format queries WITH instruction
    queries = [
        get_detailed_instruct(task, "What is the capital of France?"),
        get_detailed_instruct(task, "How does photosynthesis work?"),
    ]

    # Documents WITHOUT instruction
    documents = [
        "Paris is the capital and most populous city of France.",
        "Photosynthesis is the process by which plants convert light energy into chemical energy.",
        "The Eiffel Tower is a wrought-iron lattice tower in Paris.",
    ]

    print("\nQueries (with instruction):")
    for i, q in enumerate(queries, 1):
        print(f"  {i}. {q[:60]}...")

    print("\nDocuments (no instruction):")
    for i, d in enumerate(documents, 1):
        print(f"  {i}. {d[:60]}...")

    # Get embeddings
    print("\nGetting embeddings from Mosec...")
    query_embeddings = [embed_with_mosec(q) for q in queries]
    doc_embeddings = [embed_with_mosec(d) for d in documents]

    # Compute similarity matrix
    print("\nSimilarity Matrix:")
    print("                 Doc1    Doc2    Doc3")
    for i, q_emb in enumerate(query_embeddings):
        scores = [cosine_similarity(q_emb, d_emb) for d_emb in doc_embeddings]
        query_short = f"Query {i + 1}"
        print(f"{query_short:15} {scores[0]:.4f}  {scores[1]:.4f}  {scores[2]:.4f}")

    print("\nExpected: Query 1 matches Doc 1, Query 2 matches Doc 2")


def example_2_multilingual():
    """Example 2: Multilingual support (119+ languages)."""
    print("\n" + "=" * 70)
    print("Example 2: Multilingual Support")
    print("=" * 70)

    task = "Given a web search query, retrieve relevant passages that answer the query"

    # Same query in different languages
    queries = {
        "English": get_detailed_instruct(task, "machine learning applications"),
        "Russian": get_detailed_instruct(task, "применение машинного обучения"),
        "Chinese": get_detailed_instruct(task, "机器学习应用"),
        "French": get_detailed_instruct(task, "applications apprentissage automatique"),
    }

    documents = [
        "Machine learning is used in recommendation systems, image recognition, and NLP.",
        "Deep learning is a subset of machine learning based on artificial neural networks.",
    ]

    print("\nQueries in different languages:")
    for lang, query in queries.items():
        print(f"  {lang:12}: {query.split('Query: ')[1][:40]}...")

    # Get embeddings
    print("\nComputing cross-lingual similarities...")
    query_embeddings = {lang: embed_with_mosec(q) for lang, q in queries.items()}
    doc_embeddings = [embed_with_mosec(d) for d in documents]

    # All queries should match Doc 1 (about machine learning applications)
    print("\nSimilarity to Doc 1 (machine learning applications):")
    for lang, q_emb in query_embeddings.items():
        sim = cosine_similarity(q_emb, doc_embeddings[0])
        print(f"  {lang:12}: {sim:.4f}")

    print("\nAll languages should show high similarity (>0.7) to Doc 1")


def example_3_wrong_vs_right_format():
    """Example 3: Demonstrate importance of instruction format."""
    print("\n" + "=" * 70)
    print("Example 3: Wrong vs Right Format")
    print("=" * 70)

    task = "Given a web search query, retrieve relevant passages that answer the query"

    # Wrong format (no instruction)
    query_wrong = "What is Python programming language?"

    # Right format (with instruction)
    query_right = get_detailed_instruct(task, "What is Python programming language?")

    document = "Python is a high-level programming language known for its readability."

    print(f"\nQuery (WRONG - no instruction):\n  {query_wrong}")
    print(f"\nQuery (RIGHT - with instruction):\n  {query_right[:70]}...")
    print(f"\nDocument:\n  {document}")

    # Get embeddings
    emb_wrong = embed_with_mosec(query_wrong)
    emb_right = embed_with_mosec(query_right)
    emb_doc = embed_with_mosec(document)

    # Compare similarities
    sim_wrong = cosine_similarity(emb_wrong, emb_doc)
    sim_right = cosine_similarity(emb_right, emb_doc)

    print(f"\nSimilarity (WRONG format): {sim_wrong:.4f}")
    print(f"Similarity (RIGHT format): {sim_right:.4f}")
    print(
        f"\nDifference: {sim_right - sim_wrong:.4f} ({(sim_right / sim_wrong - 1) * 100:.1f}% better)"
    )


def example_4_batch_processing():
    """Example 4: Efficient batch processing."""
    print("\n" + "=" * 70)
    print("Example 4: Batch Processing")
    print("=" * 70)

    task = "Given a web search query, retrieve relevant passages that answer the query"

    # Multiple queries
    queries = [
        get_detailed_instruct(task, "What is artificial intelligence?"),
        get_detailed_instruct(task, "How do neural networks work?"),
        get_detailed_instruct(task, "What is deep learning?"),
        get_detailed_instruct(task, "Explain natural language processing"),
        get_detailed_instruct(task, "What is computer vision?"),
    ]

    print(f"\nProcessing {len(queries)} queries...")

    import time

    start = time.time()
    embeddings = [embed_with_mosec(q) for q in queries]
    elapsed = time.time() - start

    print(f"Total time: {elapsed:.2f}s")
    print(f"Average per query: {elapsed / len(queries) * 1000:.1f}ms")
    print(f"Embedding dimension: {embeddings[0].shape[0]}")


if __name__ == "__main__":
    print("Qwen3 Embedding Examples")
    print(
        "Make sure Mosec server is running: cmw-mosec serve --embedding Qwen/Qwen3-Embedding-0.6B"
    )
    print()

    # Run examples
    try:
        example_1_basic_query_document()
        example_2_multilingual()
        example_3_wrong_vs_right_format()
        example_4_batch_processing()

        print("\n" + "=" * 70)
        print("All examples completed successfully!")
        print("=" * 70)

    except requests.exceptions.ConnectionError:
        print("\nERROR: Cannot connect to Mosec server.")
        print("Start the server first:")
        print("  cmw-mosec serve --embedding Qwen/Qwen3-Embedding-0.6B")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
