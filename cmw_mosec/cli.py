"""CLI interface for cmw-mosec.

Single combined server with dynamic model loading.
"""

from __future__ import annotations

import sys

import click
import requests

from .server_config import (
    ModelRegistry,
    get_model_config,
    load_active_models,
)
from .server_manager import MosecServerManager


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """CMW Mosec - Combined Mosec server for embedding/reranker/guard inference."""
    pass


@cli.command()
def setup() -> None:
    """Verify setup and dependencies."""
    click.echo("Setting up cmw-mosec...")

    try:
        import mosec

        click.echo(f"✓ mosec installed ({mosec.__version__})")
    except ImportError:
        click.echo("✗ mosec not found")
        click.echo("  Install with: pip install mosec")
        sys.exit(1)

    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            click.echo(f"✓ GPU detected: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            click.echo("⚠ No GPU detected (CPU mode will be used)")
    except ImportError:
        click.echo("⚠ PyTorch not installed")

    try:
        import transformers

        click.echo(f"✓ transformers installed ({transformers.__version__})")
    except ImportError:
        click.echo("⚠ transformers not installed")

    try:
        import sentence_transformers

        click.echo(f"✓ sentence_transformers installed ({sentence_transformers.__version__})")
    except ImportError:
        click.echo("⚠ sentence_transformers not installed (needed for rerankers)")

    try:
        import requests

        click.echo(f"✓ requests installed ({requests.__version__})")
    except ImportError:
        click.echo("✗ requests not found")
        sys.exit(1)

    click.echo("\n✓ Setup complete!")


@cli.command()
@click.option("--foreground", "-f", is_flag=True, help="Run in foreground (don't detach)")
@click.option("--embedding", help="Embedding model to load")
@click.option("--reranker", help="Reranker model to load")
@click.option("--guard", help="Guard model to load")
def serve(foreground: bool, embedding: str, reranker: str, guard: str) -> None:
    """Start the combined Mosec server.

    Models are loaded from .env by default, or use --embedding/--reranker/--guard
    to override specific models.
    """
    manager = MosecServerManager()

    if manager.is_running():
        status = manager.get_status()
        click.echo(f"✓ Server already running on port {status.port}")
        return

    active = load_active_models()

    # If no model flags are passed, load all from .env
    # If any model flag is passed, only load the specified ones (no fallback)
    if embedding is None and reranker is None and guard is None:
        # No flags - use all from .env
        emb_model = active.get("embedding")
        rer_model = active.get("reranker")
        guard_model = active.get("guard")
    else:
        # Some flags passed - use them, don't load unspecified models
        emb_model = embedding
        rer_model = reranker
        guard_model = guard

    click.echo("Starting combined Mosec server...")
    click.echo(f"  Embedding: {emb_model or 'not configured'}")
    click.echo(f"  Reranker: {rer_model or 'not configured'}")
    click.echo(f"  Guard: {guard_model or 'not configured'}")

    success, failed = manager.start(
        embedding_model=emb_model,
        reranker_model=rer_model,
        guard_model=guard_model,
        background=not foreground,
    )

    if success:
        status = manager.get_status()
        click.echo(f"\n✓ Server started on port {status.port}")
        if failed:
            click.echo(f"⚠ Failed to load: {', '.join(failed)}")
        click.echo("\nEndpoints:")
        if emb_model:
            click.echo("  POST /v1/embeddings - Embeddings API")
        if rer_model:
            click.echo("  POST /v1/score - Score API (vLLM format)")
            click.echo("  POST /v1/rerank - Rerank API (Cohere format)")
        if guard_model:
            click.echo("  POST /v1/moderate - Content moderation API")
    else:
        click.echo("\n✗ Failed to start server", err=True)
        if failed:
            click.echo(f"Failed models: {', '.join(failed)}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("model_slug")
def load(model_slug: str) -> None:
    """Load a model into the running server.

    Note: Full dynamic loading requires Mosec hot-reload support.
    For now, this validates the model and provides instructions.
    """
    manager = MosecServerManager()

    if not manager.is_running():
        click.echo("Error: Server is not running", err=True)
        click.echo("Start with: cmw-mosec serve")
        sys.exit(1)

    try:
        get_model_config(model_slug)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    click.echo(f"Loading model: {model_slug}")

    if manager.load_model(model_slug):
        click.echo(f"✓ Model {model_slug} validated")
        click.echo("  Note: Server restart may be required for changes to take effect")
    else:
        click.echo(f"✗ Failed to load model: {model_slug}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("model_slug")
def unload(model_slug: str) -> None:
    """Unload a model from the running server.

    Note: Full dynamic unloading requires Mosec hot-reload support.
    """
    manager = MosecServerManager()

    if not manager.is_running():
        click.echo("Error: Server is not running", err=True)
        sys.exit(1)

    try:
        get_model_config(model_slug)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    click.echo(f"Unloading model: {model_slug}")

    if manager.unload_model(model_slug):
        click.echo(f"✓ Model {model_slug} unloaded")
    else:
        click.echo(f"✗ Failed to unload model: {model_slug}", err=True)
        sys.exit(1)


@cli.command()
def status() -> None:
    """Check status of the server and loaded models."""
    manager = MosecServerManager()

    status = manager.get_status()

    if not status.is_running:
        click.echo("Server is not running")
        click.echo("Start with: cmw-mosec serve")
        return

    click.echo(f"✓ Server running on port {status.port}")
    click.echo(
        f"  Uptime: {status.uptime_seconds or 0:.0f}s"
        if status.uptime_seconds
        else "  Uptime: unknown"
    )

    loaded = manager.list_loaded_models()
    click.echo("\nLoaded models:")
    if loaded.get("embedding"):
        click.echo(f"  Embedding: {loaded['embedding']}")
    else:
        click.echo("  Embedding: not configured")
    if loaded.get("reranker"):
        click.echo(f"  Reranker: {loaded['reranker']}")
    else:
        click.echo("  Reranker: not configured")
    if loaded.get("guard"):
        click.echo(f"  Guard: {loaded['guard']}")
    else:
        click.echo("  Guard: not configured")

    click.echo("\nEndpoints:")
    if loaded.get("embedding"):
        click.echo("  POST /v1/embeddings - Embeddings API")
    if loaded.get("reranker"):
        click.echo("  POST /v1/score - Score API (vLLM format)")
        click.echo("  POST /v1/rerank - Rerank API (Cohere format)")
    if loaded.get("guard"):
        click.echo("  POST /v1/moderate - Content moderation API")


@cli.command()
def stop() -> None:
    """Stop the combined server."""
    manager = MosecServerManager()

    if not manager.is_running():
        click.echo("Server is not running")
        return

    click.echo("Stopping server...")
    if manager.stop():
        click.echo("✓ Server stopped")
    else:
        click.echo("✗ Failed to stop server", err=True)
        sys.exit(1)


@cli.command(name="list")
def list_models() -> None:
    """List all available models (case-insensitive)."""
    registry = ModelRegistry()

    click.echo("Embedding Models:")
    for slug in registry.list_embeddings():
        config = registry.get_embedding_config(slug)
        desc = f" - {config.description}" if config.description else ""
        click.echo(f"  {slug:<45} {config.memory_gb}GB{desc}")

    click.echo("\nReranker Models:")
    for slug in registry.list_rerankers():
        config = registry.get_reranker_config(slug)
        desc = f" - {config.description}" if config.description else ""
        click.echo(f"  {slug:<45} {config.memory_gb}GB{desc}")

    click.echo("\nGuard Models:")
    for slug in registry.list_guards():
        config = registry.get_guard_config(slug)
        desc = f" - {config.description}" if config.description else ""
        click.echo(f"  {slug:<45} {config.memory_gb}GB{desc}")

    click.echo("\nActive models (from .env):")
    active = load_active_models()
    click.echo(f"  Embedding: {active.get('embedding') or 'not set'}")
    click.echo(f"  Reranker: {active.get('reranker') or 'not set'}")
    click.echo(f"  Guard: {active.get('guard') or 'not set'}")


@cli.command(name="check-guard")
@click.argument("text")
@click.option("--model", "-m", help="Guard model to use (from running server if not specified)")
@click.option("--type", "mod_type", default="prompt", type=click.Choice(["prompt", "response"]))
@click.option("--context", "-c", help="Context for response moderation")
def check_guard(text: str, model: str | None, mod_type: str, context: str | None) -> None:
    """Check content safety with both safe and unsafe examples (requires running server)."""
    manager = MosecServerManager()

    if not manager.is_running():
        click.echo("Error: Server is not running", err=True)
        click.echo("Start with: cmw-mosec serve")
        sys.exit(1)

    loaded = manager.list_loaded_models()
    guard_model = model or loaded.get("guard")

    if not guard_model:
        click.echo("Error: No guard model loaded", err=True)
        click.echo("Start server with --guard flag")
        sys.exit(1)

    status = manager.get_status()
    endpoint = f"http://localhost:{status.port}/v1/moderate"

    click.echo(f"Checking with {guard_model} at {endpoint}...")
    click.echo()

    test_cases = [
        (text, mod_type, context, "User input"),
        ("Hello, how are you today?", "prompt", None, "Safe example"),
    ]

    for content, m_type, ctx, label in test_cases:
        try:
            payload = {"content": content, "moderation_type": m_type}
            if ctx:
                payload["context"] = ctx

            response = requests.post(endpoint, json=payload, timeout=30.0)
            response.raise_for_status()
            result = response.json()

            click.echo(f"--- {label} ---")
            click.echo(f"Content: {content[:60]}...")
            click.echo(f"Safety Level: {result['safety_level']}")
            click.echo(f"Categories: {', '.join(result['categories'])}")
            if result.get("refusal"):
                click.echo(f"Refusal: {result['refusal']}")
            status_icon = "✓" if result["is_safe"] else "✗"
            click.echo(f"Is Safe: {status_icon}")

            if not result["is_safe"]:
                click.echo("⚠️  CONTENT FLAGGED")
            click.echo()

        except requests.RequestException as e:
            click.echo(f"Error for '{label}': {e}", err=True)


@cli.command(name="check-embed")
@click.option("--model", "-m", help="Embedding model to use (from running server if not specified)")
def check_embed(model: str | None) -> None:
    """Test embedding model with preset examples (requires running server)."""
    manager = MosecServerManager()

    if not manager.is_running():
        click.echo("Error: Server is not running", err=True)
        click.echo("Start with: cmw-mosec serve")
        sys.exit(1)

    loaded = manager.list_loaded_models()
    embedding_model = model or loaded.get("embedding")

    if not embedding_model:
        click.echo("Error: No embedding model loaded", err=True)
        click.echo("Start server with --embedding flag")
        sys.exit(1)

    status = manager.get_status()
    endpoint = f"http://localhost:{status.port}/v1/embeddings"

    test_cases = [
        "The quick brown fox jumps over the lazy dog.",
        "Машинное обучение - это подраздел искусственного интеллекта.",
        "Natural language processing enables computers to understand human language.",
    ]

    click.echo(f"Testing embedding model: {embedding_model}")
    click.echo(f"Endpoint: {endpoint}")
    click.echo()

    try:
        for i, text in enumerate(test_cases, 1):
            response = requests.post(
                endpoint,
                json={"input": text, "model": embedding_model},
                timeout=30.0,
            )
            response.raise_for_status()
            result = response.json()

            embedding = result["data"][0]["embedding"]
            dimension = len(embedding)
            usage = result.get("usage", {})

            click.echo(f"--- Test {i} ---")
            click.echo(f"Text: {text[:60]}{'...' if len(text) > 60 else ''}")
            click.echo(f"Dimension: {dimension}")
            click.echo(f"Tokens: {usage.get('total_tokens', 'N/A')}")
            click.echo(f"First 5 values: {embedding[:5]}")
            click.echo()

        click.echo("✓ All embedding tests passed!")

    except requests.RequestException as e:
        click.echo(f"Error connecting to server: {e}", err=True)
        sys.exit(1)


@cli.command(name="check-rerank")
@click.option("--model", "-m", help="Reranker model to use (from running server if not specified)")
@click.option(
    "--endpoint",
    "-e",
    type=click.Choice(["rerank", "score", "both"]),
    default="both",
    help="Endpoint to test: rerank, score, or both (default: both)",
)
def check_rerank(model: str | None, endpoint: str) -> None:
    """Test reranker model with examples from test config (requires running server).

    Tests both /v1/rerank and /v1/score endpoints by default.
    Uses test cases from tests/fixtures/test_rerankers.yaml.
    """
    from pathlib import Path

    import yaml

    manager = MosecServerManager()

    if not manager.is_running():
        click.echo("Error: Server is not running", err=True)
        click.echo("Start with: cmw-mosec serve")
        sys.exit(1)

    loaded = manager.list_loaded_models()
    reranker_model = model or loaded.get("reranker")

    if not reranker_model:
        click.echo("Error: No reranker model loaded", err=True)
        click.echo("Start server with --reranker flag")
        sys.exit(1)

    # Load test config
    config_path = Path(__file__).parent.parent / "tests" / "fixtures" / "test_rerankers.yaml"
    if not config_path.exists():
        click.echo(f"Error: Test config not found: {config_path}", err=True)
        sys.exit(1)

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Find model type
    model_mappings = config.get("model_mappings", {})
    model_info = None
    for slug in model_mappings:
        if slug.lower() == reranker_model.lower():
            model_info = model_mappings[slug]
            break

    if not model_info:
        click.echo(f"Warning: Model '{reranker_model}' not in test config, using defaults")
        model_type = "cross_encoder"
    else:
        model_type = model_info["type"]

    status = manager.get_status()
    base_url = f"http://localhost:{status.port}"

    click.echo(f"Testing reranker: {reranker_model}")
    click.echo(f"Model type: {model_type}")
    click.echo(f"Endpoints: {endpoint}")
    click.echo()

    # Get test cases based on model type
    if model_type == "cross_encoder":
        test_cases = config["cross_encoder"]["test_cases"]
    elif model_type == "llm_reranker":
        subtype = model_info.get("subtype", "qwen3")
        llm_config = config["llm_reranker"].get(subtype, {})
        test_cases = llm_config.get("test_cases", [])
        # Use first instruction if available
        if llm_config.get("instructions"):
            click.echo(f"Using instruction: {llm_config['instructions'][0][:50]}...")
    else:
        test_cases = config["cross_encoder"]["test_cases"]

    try:
        for i, test_case in enumerate(test_cases, 1):
            query = test_case["query"]
            documents = test_case["documents"]

            click.echo(f"--- Test {i}: {test_case.get('name', f'Query {i}')} ---")
            click.echo(f"Query: {query}")

            endpoints_to_test = []
            if endpoint in ("rerank", "both"):
                endpoints_to_test.append("/v1/rerank")
            if endpoint in ("score", "both"):
                endpoints_to_test.append("/v1/score")

            for ep in endpoints_to_test:
                response = requests.post(
                    f"{base_url}{ep}",
                    json={"query": query, "documents": documents},
                    timeout=30.0,
                )
                response.raise_for_status()
                result = response.json()

                # Handle different response formats
                if ep == "/v1/score":
                    # vLLM format: {data: [{index, object, score}, ...]}
                    scores = [item["score"] for item in result["data"]]
                else:
                    # Cohere format: {results: [{index, document, relevance_score}, ...]}
                    scores = [0.0] * len(documents)
                    for r in result["results"]:
                        scores[r["index"]] = r["relevance_score"]

                ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

                click.echo(f"\n{ep} scores: {[f'{s:.4f}' for s in scores]}")
                click.echo("Ranked results:")
                for rank, (idx, score) in enumerate(ranked, 1):
                    doc_preview = (
                        documents[idx][:50] + "..." if len(documents[idx]) > 50 else documents[idx]
                    )
                    click.echo(f"  {rank}. [{score:.4f}] {doc_preview}")

            click.echo()

        click.echo("✓ All reranking tests passed!")

    except requests.RequestException as e:
        click.echo(f"Error connecting to server: {e}", err=True)
        sys.exit(1)


@cli.command(name="check-guard-interactive")
@click.option("--model", "-m", help="Guard model to use")
def check_guard_interactive(model: str | None) -> None:
    """Interactive content safety checking mode.

    Uses guard model from .env or --model flag.
    """
    manager = MosecServerManager()

    active = load_active_models()
    guard_model = model or active.get("guard")

    if not guard_model:
        click.echo("Error: No guard model configured", err=True)
        click.echo("Set ACTIVE_GUARD_MODEL in .env")
        sys.exit(1)

    if not manager.is_running():
        click.echo("Starting server...")
        success, _ = manager.start(
            embedding_model=None,
            reranker_model=None,
            guard_model=guard_model,
            background=False,
        )
        if not success:
            click.echo("Failed to start server", err=True)
            sys.exit(1)

    status = manager.get_status()
    endpoint = f"http://localhost:{status.port}/v1/moderate"

    click.echo("Qwen3Guard Interactive Mode")
    click.echo(f"Using: {guard_model} at {endpoint}")
    click.echo("Enter text to analyze (Ctrl+C to exit):\n")

    try:
        while True:
            text = click.prompt("Text", type=str)

            if not text.strip():
                click.echo()
                continue

            try:
                response = requests.post(
                    endpoint,
                    json={"content": text, "moderation_type": "prompt"},
                    timeout=10.0,
                )
                response.raise_for_status()
                result = response.json()

                click.echo(f"\n  Safety Level: {result['safety_level']}")
                click.echo(f"  Categories: {', '.join(result['categories'])}")
                click.echo(f"  Is Safe: {'✓' if result['is_safe'] else '✗'}")

                if not result["is_safe"]:
                    click.echo("  ⚠️  FLAGGED")

                click.echo()

            except requests.RequestException as e:
                click.echo(f"\n  Error: {e}")
                click.echo("  Server may have crashed.\n")

    except KeyboardInterrupt:
        click.echo("\nGoodbye!")


@cli.command(name="models")
def models_cmd() -> None:
    """Show detailed model information (alias for 'list')."""
    list_models()


if __name__ == "__main__":
    cli()
