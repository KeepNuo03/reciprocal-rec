"""Exercise the archive-to-report CLI and serve its synthetic model artifacts."""

import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import yaml
from fastapi.testclient import TestClient

from reciprec.serving.app import create_app


def main():
    with TemporaryDirectory(prefix="reciprec-smoke-") as temp:
        root = Path(temp)
        archive, config, output = root / "fixture.zip", root / "config.yaml", root / "run"
        users = range(1, 31)
        with ZipFile(archive, "w") as fixture:
            fixture.writestr("gender.dat", "".join(f"{user},{'F' if user % 2 else 'M'}\n" for user in users))
            fixture.writestr("ratings.dat", "".join(
                f"{src},{dst},{10 if (src + dst) % 3 else 5}\n"
                for src in users for dst in users if src != dst
            ))
        settings = yaml.safe_load(Path("configs/default.yaml").read_text(encoding="utf-8"))
        settings["retrieval"].update(epochs=1, batch_size=64, dimension=8, k=10, validation_queries=10)
        settings["ranking"].update(rounds=2, patience=1)
        config.write_text(yaml.safe_dump(settings), encoding="utf-8")
        subprocess.run([
            sys.executable, "-m", "reciprec.run", "--archive", str(archive), "--config", str(config),
            "--output", str(output), "--threads", "2",
        ], check=True)
        for name in ("popularity", "retrieval", "ranking", "exposure"):
            report = output / "results" / f"{name}.json"
            assert json.loads(report.read_text(encoding="utf-8")), name
        assert (output / "results/exposure_tradeoff.png").stat().st_size > 0
        os.environ.update(
            RECIPREC_PREPARED=str(output / "data/prepared"),
            RECIPREC_ARTIFACTS=str(output / "artifacts"),
            RECIPREC_CONFIG=str(output / "configs/default.yaml"),
        )
        with np.load(output / "artifacts/retrieval/embeddings.npz") as embeddings:
            ids = embeddings["user_ids"]
        with TestClient(create_app()) as client:
            assert client.get("/health").json() == {"status": "ok"}
            responses = [client.get(f"/recommend/{user}?k=1").json() for user in ids]
            assert all(response["fallback"] is False for response in responses)
            assert any(len(response["recommendations"]) == 1 for response in responses)


if __name__ == "__main__":
    main()
