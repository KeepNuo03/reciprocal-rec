"""Measure sequential HTTP recommendation latency for a seeded sample of users."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import httpx
import numpy as np


def benchmark(client, users):
    elapsed = {"http": [], "retrieval": [], "features": [], "ranking": []}
    for user in users:
        started = perf_counter()
        response = client.get(f"/recommend/{user}")
        elapsed["http"].append(1000 * (perf_counter() - started))
        response.raise_for_status()
        body = response.json()
        if body["fallback"] or len(body["recommendations"]) != 10:
            raise ValueError(f"Benchmark requires 10 model recommendations for known user {user}")
        for stage in ("retrieval", "features", "ranking"):
            elapsed[stage].append(body["timings_ms"][stage])
    return {
        stage: {"p50": float(np.percentile(values, 50)), "p99": float(np.percentile(values, 99))}
        for stage, values in elapsed.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--embeddings", type=Path, default=Path("artifacts/retrieval/embeddings.npz"))
    parser.add_argument("--output", type=Path, default=Path("results/serving.json"))
    parser.add_argument("--requests", type=int, default=1000)
    args = parser.parse_args()
    with np.load(args.embeddings) as embeddings:
        ids = embeddings["user_ids"]
    if not 1 <= args.requests <= len(ids):
        parser.error("--requests must be between 1 and the number of known users")
    users = np.random.default_rng(42).choice(ids, args.requests, replace=False)
    with httpx.Client(base_url=args.url, timeout=10, trust_env=False) as client:
        client.get("/health").raise_for_status()
        benchmark(client, users[:20])
        report = {"requests": args.requests, "warmup_requests": min(20, args.requests), "seed": 42,
                  "latency_ms": benchmark(client, users)}
    output = json.dumps(report, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
