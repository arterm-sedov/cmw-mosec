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
def serve(foreground: bool) -> None:
    """Start the combined Mosec server with models from .env.

    Models are loaded from ACTIVE_EMBEDDING_MODEL, ACTIVE_RERANKER_MODEL,
    and ACTIVE_GUARD_MODEL environment variables.
    """
    manager = MosecServerManager()

    if manager.is_running():
        status = manager.get_status()
        click.echo(f"✓ Server already running on port {status.port}")
        return

    active = load_active_models()

    click.echo("Starting combined Mosec server...")
    click.echo(f"  Embedding: {active['embedding'] or 'not configured'}")
    click.echo(f"  Reranker: {active['reranker'] or 'not configured'}")
    click.echo(f"  Guard: {active['guard'] or 'not configured'}")

    success, failed = manager.start(
        embedding_model=active.get("embedding"),
        reranker_model=active.get("reranker"),
        guard_model=active.get("guard"),
        background=not foreground,
    )

    if success:
        status = manager.get_status()
        click.echo(f"\n✓ Server started on port {status.port}")
        if failed:
            click.echo(f"⚠ Failed to load: {', '.join(failed)}")
        click.echo("\nEndpoints:")
        if active.get("embedding"):
            click.echo("  POST /v1/embeddings - Embeddings API")
        if active.get("reranker"):
            click.echo("  POST /v1/rerank - Reranking API")
        if active.get("guard"):
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
    click.echo(f"  Uptime: {status.uptime_seconds or 0:.0f}s" if status.uptime_seconds else "  Uptime: unknown")

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


@cli.command()
@click.argument("text")
@click.option("--model", "-m", help="Guard model to use (from .env if not specified)")
@click.option("--type", "mod_type", default="prompt", type=click.Choice(["prompt", "response"]))
@click.option("--context", "-c", help="Context for response moderation")
def check(text: str, model: str | None, mod_type: str, context: str | None) -> None:
    """Check content safety (requires running server)."""
    manager = MosecServerManager()

    if not manager.is_running():
        click.echo("Error: Server is not running", err=True)
        click.echo("Start with: cmw-mosec serve")
        sys.exit(1)

    active = load_active_models()
    guard_model = model or active.get("guard")

    if not guard_model:
        click.echo("Error: No guard model configured", err=True)
        click.echo("Set ACTIVE_GUARD_MODEL in .env")
        sys.exit(1)

    status = manager.get_status()
    endpoint = f"http://localhost:{status.port}/v1/moderate"

    click.echo(f"Checking with {guard_model} at {endpoint}...")
    click.echo(f"Type: {mod_type}")
    if mod_type == "response" and context:
        click.echo(f"Context: {context[:50]}...")
    click.echo()

    try:
        response = requests.post(
            endpoint,
            json={"content": text, "context": context, "moderation_type": mod_type},
            timeout=30.0,
        )
        response.raise_for_status()
        result = response.json()

        click.echo(f"Safety Level: {result['safety_level']}")
        click.echo(f"Categories: {', '.join(result['categories'])}")
        if result.get("refusal"):
            click.echo(f"Refusal: {result['refusal']}")
        click.echo(f"Is Safe: {'✓' if result['is_safe'] else '✗'}")

        if not result["is_safe"]:
            click.echo("\n⚠️  CONTENT FLAGGED")

    except requests.RequestException as e:
        click.echo(f"Error connecting to server: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--model", "-m", help="Guard model to use (from .env if not specified)")
def interactive(model: str | None) -> None:
    """Interactive content safety checking mode."""
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
            embedding_model=active.get("embedding"),
            reranker_model=active.get("reranker"),
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
