"""M3 verification: validate the saved federated artifacts and (re)write
history_summary.json. Read-only w.r.t. the model/history; does not retrain.

Run from the repo root:  python scripts/verify_m3.py
Checkpoint validation needs torch; everything else runs without it.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.federated.history import FederatedHistory, history_summary  # noqa: E402
from src.utils.config import load_config, get  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

log = get_logger("verify_m3")
CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, ok, detail))
    log.info("[%s] %s %s", "PASS" if ok else "FAIL", name, f"- {detail}" if detail else "")


def main() -> int:
    cfg = load_config(REPO_ROOT / "configs" / "data.yaml")
    cfg.update(load_config(REPO_ROOT / "configs" / "model.yaml"))
    cfg.update(load_config(REPO_ROOT / "configs" / "federated.yaml"))
    exp = REPO_ROOT / get(cfg, "federated.ckpt_dir", "experiments")
    hist_p = exp / "federated_history.pkl"
    ckpt_p = exp / "federated_model.pt"
    metrics_p = exp / "federated_val_metrics.json"

    # 1. history loads
    if not hist_p.exists():
        check("history file exists", False, f"missing {hist_p}; run scripts/run_federated.py")
        return _summary()
    hist = FederatedHistory.load(hist_p)
    check("history loads", True, f"{len(hist)} rounds")

    exp_rounds = cfg["federated"]["num_rounds"]
    exp_clients = cfg["data"]["num_clients"]

    # 2. rounds completed & sequential
    seq = [r["round"] for r in hist.rounds] == list(range(len(hist)))
    check("rounds sequential", seq, f"indices={[r['round'] for r in hist.rounds]}")
    check("expected round count", len(hist) == exp_rounds,
          f"got {len(hist)}, config says {exp_rounds}")

    # 3. per-round client structure & layer consistency
    n_layers = len(hist.rounds[0]["global_params"])
    ok_struct, detail = True, ""
    for r in hist.rounds:
        if len(r["clients"]) != exp_clients:
            ok_struct, detail = False, f"round {r['round']} has {len(r['clients'])} clients"
            break
        for c in r["clients"]:
            if c["n"] <= 0 or len(c["params"]) != n_layers:
                ok_struct, detail = False, f"round {r['round']} client bad n/layers"
                break
    check("client structure consistent", ok_struct, detail or f"{exp_clients} clients, {n_layers} tensors")

    # 4. global model updates every round (drift > 0 vs previous)
    drifts, prev = [], None
    for r in hist.rounds:
        flat = np.concatenate([a.ravel() for a in r["global_params"]]).astype(np.float64)
        if prev is not None:
            drifts.append(float(np.linalg.norm(flat - prev)))
        prev = flat
    check("global model updates each round", all(d > 0 for d in drifts) if drifts else True,
          f"drifts={[round(d, 4) for d in drifts]}")

    # 5. history round-trips (save -> load -> identical)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "h.pkl"
        hist.save(tmp)
        rl = FederatedHistory.load(tmp)
        rt = (len(rl) == len(hist)
              and all(np.allclose(a, b) for a, b in
                      zip(rl.rounds[0]["global_params"], hist.rounds[0]["global_params"])))
    check("history round-trips", rt)

    # 6. metrics saved & well-formed
    if metrics_p.exists():
        m = json.load(open(metrics_p))
        keys_ok = all(k in m for k in ("macro_auc", "mAP", "macro_f1", "micro_f1"))
        check("val metrics saved", keys_ok, f"keys={list(m)}")
    else:
        check("val metrics saved", False, f"missing {metrics_p}")
        m = None

    # 7. checkpoint valid & consistent with last-round global (torch-gated)
    try:
        import torch
        if ckpt_p.exists():
            ck = torch.load(ckpt_p, map_location="cpu")
            sd = ck["state_dict"]
            same_len = len(sd) == n_layers
            last = hist.rounds[-1]["global_params"]
            consistent = same_len and all(
                np.allclose(v.detach().cpu().numpy(), last[i], atol=1e-5)
                for i, v in enumerate(sd.values()))
            check("checkpoint loadable & == last-round global", consistent,
                  f"{len(sd)} tensors")
        else:
            check("checkpoint present", False, f"missing {ckpt_p}")
    except ImportError:
        log.warning("torch not installed; skipping checkpoint validation.")

    # (re)generate history_summary.json from existing artifacts
    meta = {"experiment": "M3 federated FedAvg (verification-generated)",
            "aggregation": "FedAvg (weighted by client sample count)",
            "model": cfg["model"]["name"], "seed": get(cfg, "seed", 42),
            "num_clients": exp_clients, "local_epochs": cfg["federated"]["local_epochs"],
            "lr": cfg["federated"]["lr"], "final_val_metrics": m,
            "source": {"history": hist_p.name, "model": ckpt_p.name}}
    with open(exp / "history_summary.json", "w") as f:
        json.dump(history_summary(hist, meta), f, indent=2)
    log.info("Wrote %s", exp / "history_summary.json")

    return _summary()


def _summary() -> int:
    passed = sum(1 for _, ok, _ in CHECKS if ok)
    total = len(CHECKS)
    if passed == total:
        print(f"M3 VERIFICATION PASSED ({passed}/{total})")
        return 0
    print(f"M3 VERIFICATION FAILED ({passed}/{total})")
    for name, ok, detail in CHECKS:
        if not ok:
            print(f"  FAIL: {name} {('- ' + detail) if detail else ''}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
