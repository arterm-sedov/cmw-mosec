"""CLI interface for cmw-mosec."""

from __future__ import annotations

import sys

import click
import requests

from .server_config import (
    ModelRegistry,
    get_model_config,
    list_available_models,
)
from .server_manager import MosecServerManager


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """CMW Mosec - Mosec server management for embedding/reranker/guard inference."""
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
        for model_type, models in available.items():
            click.echo(f"  {model_type.capitalize()}: {', '.join(models)}")
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
    click.echo(f"  Type: {config.model_type}")
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

    click.echo(f"{'Model':<45} {'Type':<10} {'Device':<8} {'Port':<8} {'Status':<12} {'Uptime'}")
    click.echo("-" * 100)

    for s in running:
        status_str = "✓ running" if s.is_running else "✗ not responding"
        uptime_str = ""
        if s.uptime_seconds:
            minutes = int(s.uptime_seconds // 60)
            hours = minutes // 60
            uptime_str = f"{hours}h {minutes % 60}m" if hours > 0 else f"{minutes}m"

        click.echo(
            f"{s.model_key:<45} {s.model_type:<10} {s.device:<8} {s.port:<8} {status_str:<12} {uptime_str}"
        )


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

    click.echo("\nUsage:")
    click.echo("  cmw-mosec start ai-forever/FRIDA")
    click.echo("  cmw-mosec start DiTy/cross-encoder-russian-msmarco")
    click.echo("  cmw-mosec start Qwen/Qwen3Guard-Gen-0.6B")


@cli.command()
@click.argument("text")
@click.option("--model", "-m", default="Qwen/Qwen3Guard-Gen-0.6B", help="Guard model to use")
@click.option("--type", "mod_type", default="prompt", type=click.Choice(["prompt", "response"]))
@click.option("--context", "-c", help="Context for response moderation")
@click.option("--endpoint", "-e", default=None, help="Guard server endpoint (auto-detect if not provided)")
def check(text: str, model: str, mod_type: str, context: str | None, endpoint: str | None) -> None:
    """Check content safety (one-off command).

    Use this to test guard models without starting a server.

    Examples:
        cmw-mosec check "How can I make a bomb?"
        cmw-mosec check "As a responsible AI..." --type response --context "How can I make a bomb?"
    """
    try:
        config = get_model_config(model)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if config.model_type != "guard":
        click.echo(f"Error: '{model}' is not a guard model", err=True)
        sys.exit(1)

    if endpoint is None:
        endpoint = f"http://localhost:{config.port}"

    click.echo(f"Checking with {model} at {endpoint}...")
    click.echo(f"Type: {mod_type}")
    if mod_type == "response" and context:
        click.echo(f"Context: {context[:50]}...")
    click.echo()

    try:
        response = requests.post(
            f"{endpoint}/inference",
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
            click.echo(f"Reason: Safety level is {result['safety_level']}")

    except requests.RequestException as e:
        click.echo(f"Error connecting to guard server: {e}", err=True)
        click.echo("Make sure the guard server is running with:")
        click.echo(f"  cmw-mosec start {model}")
        sys.exit(1)


@cli.command()
@click.option("--model", "-m", default="Qwen/Qwen3Guard-Gen-0.6B", help="Guard model to use")
@click.option("--endpoint", "-e", default=None, help="Guard server endpoint (auto-detect if not provided)")
@click.option("--auto-start/--no-auto-start", default=True, help="Auto-start server if not running")
def interactive(model: str, endpoint: str | None, auto_start: bool) -> None:
    """Interactive content safety checking mode.

    Starts a guard server if needed and enters interactive session.
    """
    try:
        config = get_model_config(model)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if config.model_type != "guard":
        click.echo(f"Error: '{model}' is not a guard model", err=True)
        sys.exit(1)

    if endpoint is None:
        endpoint = f"http://localhost:{config.port}"

    manager = MosecServerManager()
    status = manager.get_status(model, config)

    server_started = False
    if not status.is_running:
        if auto_start:
            click.echo(f"Starting {model} server...")
            success = manager.start(model, config, background=True)
            if not success:
                click.echo(f"Error: Failed to start server", err=True)
                sys.exit(1)
            server_started = True
            click.echo(f"Waiting for server on port {config.port}...")
            for _ in range(60):
                time.sleep(1)
                try:
                    response = requests.get(f"{endpoint}/health", timeout=2.0)
                    if response.status_code == 200:
                        break
                except requests.RequestException:
                    pass
            else:
                click.echo(f"Error: Server failed to start", err=True)
                sys.exit(1)
        else:
            click.echo(f"Error: Server not running. Start with: cmw-mosec start {model}", err=True)
            sys.exit(1)

    click.echo(f"Qwen3Guard Interactive Mode")
    click.echo(f"Using: {model} at {endpoint}")
    if server_started:
        click.echo("Server started automatically.")
    click.echo("Enter text to analyze (Ctrl+C to exit):\n")

    try:
        while True:
            text = click.prompt("Text", type=str)

            if not text.strip():
                click.echo()
                continue

            try:
                response = requests.post(
                    f"{endpoint}/inference",
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
                click.echo(f"  Server may have crashed. Try: cmw-mosec start {model}\n")

    except KeyboardInterrupt:
        click.echo("\nGoodbye!")


@cli.command()
def models() -> None:
    """Show detailed model information (alias for 'list')."""
    list_models()


if __name__ == "__main__":
    cli()
