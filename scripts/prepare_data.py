"""M1 driver: build the datamodule, print stats, and persist manifest/partition.

Run from the repo root:  python scripts/prepare_data.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import build_datamodule  # noqa: E402
from src.utils.config import load_config, get  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def main() -> int:
    cfg = load_config(REPO_ROOT / "configs" / "data.yaml")
    set_seed(get(cfg, "seed", 42))
    log = get_logger("prepare_data")

    dm = build_datamodule(cfg)
    stats, manifest, partition = dm["stats"], dm["manifest"], dm["partition"]

    n_by_split = manifest["split"].value_counts().to_dict()
    log.info("Manifest: %d images / %d patients | splits=%s",
             len(manifest), manifest["patient_id"].nunique(), n_by_split)
    log.info("Clients=%d | patient_disjoint=%s | patients_covered=%s",
             len(stats["clients"]), stats["patient_disjoint"], stats["patients_covered"])
    for cid, s in stats["clients"].items():
        log.info("  client %d: %d images, %d patients, labels=%s",
                 cid, s["n_images"], s["n_patients"], s["label_counts"])

    out = Path(REPO_ROOT / get(cfg, "data.root", "data"))
    out.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out / "manifest.csv", index=False)
    with open(out / "partition.json", "w") as f:
        json.dump({"client_to_patients": partition["client_to_patients"],
                   "client_to_images": partition["client_to_images"],
                   "stats": stats}, f, indent=2)
    log.info("Saved manifest.csv and partition.json to %s/", out)

    ok = stats["patient_disjoint"] and stats["patients_covered"]
    print("M1 DATA PIPELINE OK" if ok else "M1 DATA PIPELINE FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
