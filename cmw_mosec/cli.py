"""CLI interface for cmw-mosec."""

from __future__ import annotations

import sys

import click

from .server_config import (
    ModelRegistry,
    get_model_config,
    list_available_models,
)
from .server_manager import MosecServerManager


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """CMW Mosec - Mosec server management for embedding/reranker inference."""
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
@click.argument("model_key")
@click.option("--foreground", "-f", is_flag=True, help="Run in foreground (don't detach)")
@click.option(
    "--device",
    type=click.Choice(["auto", "cpu", "cuda"]),
    default=None,
    help="Device to use (overrides config)",
)
def start(model_key: str, foreground: bool, device: str | None) -> None:
    """Start a Mosec server for a model."""
    try:
        config = get_model_config(model_key)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        available = list_available_models()
        click.echo("\nAvailable models:")
        click.echo(f"  Embedding: {', '.join(available['embedding'])}")
        click.echo(f"  Reranker: {', '.join(available['reranker'])}")
        sys.exit(1)

    if device is not None:
        config = config.model_copy(update={"device": device})

    manager = MosecServerManager()
    status = manager.get_status(model_key, config)

    if status.is_running:
        click.echo(f"✓ Server '{model_key}' is already running on port {config.port}")
        return

    click.echo(f"Starting Mosec server for '{model_key}'...")
    click.echo(f"  Model: {config.model_id}")
    click.echo(f"  Port: {config.port}")
    click.echo(f"  Workers: {config.workers}")
    click.echo(f"  Estimated memory: {config.memory_gb} GB")

    success = manager.start(model_key, config, background=not foreground)

    if success:
        if foreground:
            click.echo("\n✓ Server stopped")
        else:
            click.echo(f"✓ Server started on port {config.port}")
            click.echo("  Use 'cmw-mosec status' to check health")
    else:
        click.echo("✗ Failed to start server", err=True)
        sys.exit(1)


@cli.command()
@click.argument("model_key", required=False)
@click.option("--all", "stop_all", is_flag=True, help="Stop all running servers")
def stop(model_key: str | None, stop_all: bool) -> None:
    """Stop a Mosec server."""
    manager = MosecServerManager()

    if stop_all:
        running = manager.list_running()
        if not running:
            click.echo("No servers are running")
            return

        click.echo(f"Stopping {len(running)} server(s)...")
        for status in running:
            click.echo(f"  Stopping '{status.model_key}'...")
            manager.stop(status.model_key)
        click.echo("✓ All servers stopped")
        return

    if not model_key:
        click.echo("Error: Specify model key or use --all", err=True)
        sys.exit(1)

    try:
        config = get_model_config(model_key)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    status = manager.get_status(model_key, config)
    if not status.pid:
        click.echo(f"Server '{model_key}' is not running")
        return

    click.echo(f"Stopping server '{model_key}'...")
    if manager.stop(model_key):
        click.echo("✓ Server stopped")
    else:
        click.echo("✗ Failed to stop server", err=True)
        sys.exit(1)


@cli.command()
def status() -> None:
    """Check status of all servers."""
    manager = MosecServerManager()
    running = manager.list_running()

    if not running:
        click.echo("No servers are running")
        return

    click.echo(f"{'Model':<40} {'Type':<10} {'Device':<8} {'Port':<8} {'Status':<12}")
    click.echo("-" * 85)

    registry = ModelRegistry()

    for s in running:
        status_str = "✓ running" if s.is_running else "✗ not responding"
        uptime_str = ""
        if s.uptime_seconds:
            minutes = int(s.uptime_seconds // 60)
            hours = minutes // 60
            uptime_str = f"{hours}h {minutes % 60}m" if hours > 0 else f"{minutes}m"

        try:
            model_type = registry.get_model_type(s.model_key)
        except ValueError:
            model_type = "unknown"

        click.echo(f"{s.model_key:<40} {model_type:<10} {s.device:<8} {s.port:<8} {status_str:<12} {uptime_str}")


@cli.command(name="list")
def list_models() -> None:
    """List all available models (case-insensitive)."""
    registry = ModelRegistry()

    click.echo("Embedding Models:")
    for slug in registry.list_embeddings():
        config = registry.get_embedding_config(slug)
        click.echo(f"  {slug:<40} {config.model_id:<40} {config.memory_gb} GB")

    click.echo("\nReranker Models:")
    for slug in registry.list_rerankers():
        config = registry.get_reranker_config(slug)
        click.echo(f"  {slug:<40} {config.model_id:<40} {config.memory_gb} GB")

    click.echo("\nUsage:")
    click.echo("  cmw-mosec start ai-forever/FRIDA")
    click.echo("  cmw-mosec start DiTy/cross-encoder-russian-msmarco")


if __name__ == "__main__":
    cli()
