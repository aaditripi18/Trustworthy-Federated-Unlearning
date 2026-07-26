"""B2 — checkpoint verification utility (READ-ONLY).

Confirms that every model checkpoint produced by the canonical retraining is
valid: it loads, rebuilds the correct architecture, loads the state_dict with no
missing/unexpected keys, and switches to eval mode. It NEVER writes, modifies, or
deletes any checkpoint or experiment output.

Each checkpoint payload (written by run_single._save_ckpt) is self-describing:
    {"state_dict", "seed", "model", "config"}
so the architecture is rebuilt from the checkpoint's own saved config — no YAML
needed and no risk of config drift.

Usage (after a run or smoke run):
    python scripts/verify_checkpoints.py                         # experiments/full_experiment
    python scripts/verify_checkpoints.py experiments/full_experiment_smoke
Exit code 0 => all checkpoints valid; 1 => any missing or invalid.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.logging import get_logger  # noqa: E402

FIXED = ["original.pt", "oracle.pt", "plain.pt"]   # + any epu_lambda*.pt present


def _verify_one(path: Path, build_model, torch, log) -> bool:
    """Load + rebuild + load_state_dict + eval() for one checkpoint. READ-ONLY."""
    try:
        ckpt = torch.load(path, map_location="cpu")   # read-only load
    except Exception as e:                              # noqa: BLE001
        log.error("[FAIL] %s : cannot load (%s)", path.name, e)
        return False
    if "state_dict" not in ckpt or "config" not in ckpt:
        log.error("[FAIL] %s : payload missing 'state_dict'/'config' keys", path.name)
        return False
    cfg = ckpt["config"]
    try:
        model = build_model(cfg["model"], cfg["data"]["image_size"], cfg["data"]["num_labels"])
    except Exception as e:                              # noqa: BLE001
        log.error("[FAIL] %s : could not rebuild architecture (%s)", path.name, e)
        return False
    result = model.load_state_dict(ckpt["state_dict"], strict=False)  # report, don't raise
    if result.missing_keys or result.unexpected_keys:
        log.error("[FAIL] %s : missing=%d unexpected=%d (first missing: %s / first unexpected: %s)",
                  path.name, len(result.missing_keys), len(result.unexpected_keys),
                  result.missing_keys[:1], result.unexpected_keys[:1])
        return False
    model.eval()
    if model.training:                                 # eval() must set training=False
        log.error("[FAIL] %s : model did not switch to eval mode", path.name)
        return False
    n = sum(p.numel() for p in model.parameters())
    log.info("[ OK ] %s : rebuilt %s, %d params, keys match, eval mode OK (seed=%s)",
             path.name, cfg["model"]["name"], n, ckpt.get("seed"))
    return True


def main() -> int:
    log = get_logger("verify_checkpoints")
    try:
        import torch
        from src.models.vit import build_model
    except ImportError:
        log.warning("torch/timm not installed; verification is a no-op here. SKIPPED.")
        print("CHECKPOINT VERIFY SKIPPED (torch/timm unavailable)")
        return 0

    root = REPO_ROOT / (sys.argv[1] if len(sys.argv) > 1 else "experiments/full_experiment")
    seed_dirs = sorted(root.glob("seed_*"))
    if not seed_dirs:
        log.error("no seed_* directories under %s", root)
        return 1

    total = ok = 0
    for sdir in seed_dirs:
        expected = FIXED + sorted(p.name for p in sdir.glob("epu_lambda*.pt"))
        log.info("--- %s : %d checkpoints ---", sdir.name, len(expected))
        for name in expected:
            total += 1
            path = sdir / name
            if not path.exists():
                log.error("[FAIL] %s/%s : file missing", sdir.name, name)
                continue
            ok += _verify_one(path, build_model, torch, log)

    log.info("SUMMARY: %d/%d checkpoints valid across %d seed(s)", ok, total, len(seed_dirs))
    if ok == total and total > 0:
        print("CHECKPOINT VERIFY OK")
        return 0
    print("CHECKPOINT VERIFY FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
