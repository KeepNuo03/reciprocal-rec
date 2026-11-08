"""Reproduce the offline pipeline from an authorized archive to evaluation reports."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def reproduce(archive, config, output, device, threads):
    if output.exists():
        raise FileExistsError(f"Choose a fresh output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".reciprec-run-", dir=output.parent) as temporary:
        run = Path(temporary) / "run"
        raw, prepared = run / "data/raw", run / "data/prepared"
        retrieval, ranking = run / "artifacts/retrieval", run / "artifacts/ranking"
        results = run / "results"
        snapshot = run / "configs/default.yaml"
        snapshot.parent.mkdir(parents=True)
        shutil.copyfile(config, snapshot)

        def step(module, *arguments):
            print(f"Running {module}", flush=True)
            subprocess.run([sys.executable, "-m", module, *map(str, arguments)], check=True)

        step("reciprec.data.download", "--archive", archive, "--output", raw)
        step("reciprec.data.prepare", "--raw", raw, "--output", prepared, "--config", snapshot)
        step("reciprec.eval.popularity", "--prepared", prepared, "--output", results / "popularity.json")
        common = ("--prepared", prepared, "--config", snapshot, "--device", device, "--threads", threads)
        step("reciprec.retrieval.train", *common, "--artifacts", retrieval, "--report", results / "retrieval.json")
        step("reciprec.ranking.ranker", *common, "--retrieval", retrieval, "--artifacts", ranking,
             "--report", results / "ranking.json")
        step("reciprec.eval.exposure", "--prepared", prepared, "--retrieval", retrieval,
             "--ranker", ranking / "ranker.txt", "--config", snapshot, "--output", results)
        run.rename(output)
    print(f"Reports written to {output / 'results'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/run"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1 or not args.archive.is_file() or not args.config.is_file():
        parser.error("Provide an existing archive and config, and a positive thread count")
    reproduce(args.archive, args.config, args.output.resolve(), args.device, args.threads)


if __name__ == "__main__":
    main()
