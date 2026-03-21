#!/usr/bin/env python3
"""Test VRAM/RAM for all supported models."""

import subprocess
import time
import yaml
from pathlib import Path

GPU_TOTAL_MIB = 49140
RESULTS_FILE = "model_memory_results.yaml"


def get_gpu_info():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.free,name", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    )
    parts = [p.strip() for p in result.stdout.strip().split(",")]
    return {"used": int(parts[0]), "free": int(parts[1]), "name": parts[2]}


def get_ram_used():
    result = subprocess.run(["free", "-h"], capture_output=True, text=True)
    for line in result.stdout.split("\n"):
        if line.startswith("Mem:"):
            return line.split()[2]
    return "N/A"


def stop_server():
    subprocess.run([".venv/bin/cmw-mosec", "stop"], capture_output=True)
    time.sleep(3)


def start_and_measure(name, emb=None, rer=None, guard=None, wait=60):
    print(f"\n--- {name} ---", flush=True)
    stop_server()
    time.sleep(2)

    baseline = get_gpu_info()
    baseline_ram = get_ram_used()
    print(f"  Baseline: VRAM={baseline['used']} MiB, RAM={baseline_ram}", flush=True)

    cmd = [".venv/bin/cmw-mosec", "serve"]
    if emb:
        cmd.extend(["--embedding", emb])
    if rer:
        cmd.extend(["--reranker", rer])
    if guard:
        cmd.extend(["--guard", guard])

    subprocess.run(cmd, capture_output=True)
    print(f"  Waiting {wait}s...", flush=True)
    time.sleep(wait)

    after = get_gpu_info()
    after_ram = get_ram_used()
    delta = after["used"] - baseline["used"]

    print(
        f"  Result: VRAM={after['used']} MiB (+{delta}), RAM={after_ram}, Free={after['free']} MiB",
        flush=True,
    )

    return {
        "name": name,
        "embedding": emb,
        "reranker": rer,
        "guard": guard,
        "baseline_vram_mib": baseline["used"],
        "vram_used_mib": after["used"],
        "vram_free_mib": after["free"],
        "delta_vram_mib": delta,
        "ram_used": after_ram,
    }


def load_results():
    if Path(RESULTS_FILE).exists():
        with open(RESULTS_FILE) as f:
            return yaml.safe_load(f)
    return {"results": []}


def save_results(results):
    output = {
        "test_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpu_name": "NVIDIA GeForce RTX 4090",
        "gpu_total_mib": GPU_TOTAL_MIB,
        "results": results,
    }
    with open(RESULTS_FILE, "w") as f:
        yaml.dump(output, f, default_flow_style=False)


def load_models_from_yaml():
    with open("config/models.yaml") as f:
        config = yaml.safe_load(f)
    models = []
    for name, data in config.get("embedding_models", {}).items():
        models.append({"type": "embedding", "name": name, "model_id": data.get("model_id", name)})
    for name, data in config.get("reranker_models", {}).items():
        models.append({"type": "reranker", "name": name, "model_id": data.get("model_id", name)})
    for name, data in config.get("guard_models", {}).items():
        models.append({"type": "guard", "name": name, "model_id": data.get("model_id", name)})
    return models


def main():
    models = load_models_from_yaml()
    print(f"Found {len(models)} models in config/models.yaml", flush=True)

    all_scenarios = []

    for m in models:
        model_id = m["model_id"]
        short_name = model_id.split("/")[-1].replace("-", "_")
        all_scenarios.append(
            {
                "name": f"{m['type'][:3]}_{short_name}",
                m["type"]: model_id,
            }
        )

    combinations = [
        {
            "name": "3x_0.6b",
            "embedding": "Qwen/Qwen3-Embedding-0.6B",
            "reranker": "Qwen/Qwen3-Reranker-0.6B",
            "guard": "Qwen/Qwen3Guard-Gen-0.6B",
        },
        {
            "name": "emb_frida_ditty_0.6b",
            "embedding": "ai-forever/FRIDA",
            "reranker": "DiTy/cross-encoder-russian-msmarco",
            "guard": "Qwen/Qwen3Guard-Gen-0.6B",
        },
        {
            "name": "emb_4b_2x_0.6b",
            "embedding": "Qwen/Qwen3-Embedding-4B",
            "reranker": "Qwen/Qwen3-Reranker-0.6B",
            "guard": "Qwen/Qwen3Guard-Gen-0.6B",
        },
        {
            "name": "emb_0.6b_2x_4b",
            "embedding": "Qwen/Qwen3-Embedding-0.6B",
            "reranker": "Qwen/Qwen3-Reranker-4B",
            "guard": "Qwen/Qwen3Guard-Gen-4B",
        },
        {
            "name": "emb_4b_2x_4b",
            "embedding": "Qwen/Qwen3-Embedding-4B",
            "reranker": "Qwen/Qwen3-Reranker-4B",
            "guard": "Qwen/Qwen3Guard-Gen-4B",
        },
    ]
    all_scenarios.extend(combinations)

    existing = load_results()
    completed = {r["name"] for r in existing.get("results", [])}
    print(f"Already completed: {len(completed)}", flush=True)

    results = list(existing.get("results", []))

    for scenario in all_scenarios:
        name = scenario["name"]
        if name in completed:
            print(f"[{len(results) + 1}/{len(all_scenarios)}] SKIP {name}", flush=True)
            continue

        print(f"[{len(results) + 1}/{len(all_scenarios)}] TEST {name}", flush=True)
        result = start_and_measure(
            name=name,
            emb=scenario.get("embedding"),
            rer=scenario.get("reranker"),
            guard=scenario.get("guard"),
        )
        results.append(result)
        save_results(results)
        print(
            f"  -> VRAM={result['vram_used_mib']} MiB (+{result['delta_vram_mib']}), Free={result['vram_free_mib']} MiB",
            flush=True,
        )

    stop_server()

    print("\n" + "=" * 80, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 80, flush=True)
    print(f"{'Name':<30} {'VRAM':>10} {'Delta':>8} {'Free':>8} {'RAM':>8}", flush=True)
    print("-" * 80, flush=True)
    for r in results:
        print(
            f"{r['name']:<30} {r['vram_used_mib']:>10} {r['delta_vram_mib']:>+8} {r['vram_free_mib']:>8} {r['ram_used']:>8}",
            flush=True,
        )

    print(f"\nResults saved to: {RESULTS_FILE}", flush=True)


if __name__ == "__main__":
    main()
