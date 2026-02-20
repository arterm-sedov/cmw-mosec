#!/usr/bin/env python3
"""
FRIDA Embedding Examples

This script demonstrates proper usage of FRIDA (T5-based) embedding model with Mosec.

Key points:
1. Use search_query: prefix for queries
2. Use search_document: prefix for documents
3. Mosec handles CLS pooling automatically for T5 models

References:
- https://huggingface.co/ai-forever/FRIDA
"""

import numpy as np
import requests


def embed_with_mosec(
    text: str, model: str = "ai-forever/FRIDA", base_url: str = "http://localhost:8001"
) -> np.ndarray:
    """Get embeddings from Mosec server."""
    response = requests.post(
        f"{base_url}/v1/embeddings", json={"model": model, "input": text}, timeout=30
    )
    response.raise_for_status()
    result = response.json()
    return np.array(result["data"][0]["embedding"])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def example_1_basic_usage():
    """Example 1: Basic FRIDA usage with prefixes."""
    print("=" * 70)
    print("Example 1: Basic FRIDA Usage")
    print("=" * 70)

    # Queries with search_query: prefix
    queries = [
        "search_query: Как приготовить борщ?",  # Russian
        "search_query: How to cook borscht?",  # English
    ]

    # Documents with search_document: prefix
    documents = [
        "search_document: Борщ - это традиционный русский суп из свеклы.",
        "search_document: Паста - это итальянское блюдо из муки и яиц.",
        "search_document: Borscht is a traditional Russian beet soup.",
    ]

    print("\nQueries (with search_query: prefix):")
    for i, q in enumerate(queries, 1):
        print(f"  {i}. {q}")

    print("\nDocuments (with search_document: prefix):")
    for i, d in enumerate(documents, 1):
        print(f"  {i}. {d}")

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

    print("\nExpected: Both queries match Doc 1 and Doc 3 (about borscht)")


def example_2_without_prefixes():
    """Example 2: Demonstrate importance of prefixes."""
    print("\n" + "=" * 70)
    print("Example 2: With vs Without Prefixes")
    print("=" * 70)

    # With prefixes (correct)
    query_with = "search_query: машинное обучение"
    doc_with = "search_document: Машинное обучение - это область ИИ."

    # Without prefixes (wrong)
    query_without = "машинное обучение"
    doc_without = "Машинное обучение - это область ИИ."

    print(f"\nWith prefixes (CORRECT):")
    print(f"  Query: {query_with}")
    print(f"  Doc: {doc_with}")

    print(f"\nWithout prefixes (WRONG):")
    print(f"  Query: {query_without}")
    print(f"  Doc: {doc_without}")

    # Get embeddings
    emb_query_with = embed_with_mosec(query_with)
    emb_doc_with = embed_with_mosec(doc_with)
    emb_query_without = embed_with_mosec(query_without)
    emb_doc_without = embed_with_mosec(doc_without)

    # Compare
    sim_with = cosine_similarity(emb_query_with, emb_doc_with)
    sim_without = cosine_similarity(emb_query_without, emb_doc_without)

    print(f"\nSimilarity (with prefixes):    {sim_with:.4f}")
    print(f"Similarity (without prefixes): {sim_without:.4f}")
    print(f"\nPrefixes improve similarity by: {sim_with - sim_without:.4f}")


def example_3_multilingual_russian():
    """Example 3: Russian language optimization."""
    print("\n" + "=" * 70)
    print("Example 3: Russian Language (FRIDA is optimized for Russian)")
    print("=" * 70)

    queries = [
        "search_query: искусственный интеллект",
        "search_query: машинное обучение",
        "search_query: глубокое обучение",
    ]

    documents = [
        "search_document: Искусственный интеллект (ИИ) - это способность машин имитировать человеческий интеллект.",
        "search_document: Машинное обучение - это подраздел ИИ, изучающий алгоритмы обучения.",
        "search_document: Глубокое обучение использует нейронные сети с множеством слоев.",
    ]

    print("\nQueries:")
    for i, q in enumerate(queries, 1):
        print(f"  {i}. {q}")

    # Get embeddings
    query_embeddings = [embed_with_mosec(q) for q in queries]
    doc_embeddings = [embed_with_mosec(d) for d in documents]

    # Compute similarity matrix
    print("\nSimilarity Matrix:")
    print("                 Doc1    Doc2    Doc3")
    for i, q_emb in enumerate(query_embeddings):
        scores = [cosine_similarity(q_emb, d_emb) for d_emb in doc_embeddings]
        query_short = f"Query {i + 1}"
        print(f"{query_short:15} {scores[0]:.4f}  {scores[1]:.4f}  {scores[2]:.4f}")

    print("\nExpected: Diagonal should have highest values (query i matches doc i)")


if __name__ == "__main__":
    print("FRIDA (T5-Based) Embedding Examples")
    print("Make sure Mosec server is running: cmw-mosec serve --embedding ai-forever/FRIDA")
    print()

    try:
        example_1_basic_usage()
        example_2_without_prefixes()
        example_3_multilingual_russian()

        print("\n" + "=" * 70)
        print("All examples completed successfully!")
        print("=" * 70)

    except requests.exceptions.ConnectionError:
        print("\nERROR: Cannot connect to Mosec server.")
        print("Start the server first:")
        print("  cmw-mosec serve --embedding ai-forever/FRIDA")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
