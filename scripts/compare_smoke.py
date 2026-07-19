"""Compare a fresh (post-optimization) smoke run against the frozen baseline in
experiments/_smoke_baseline/.

Usage (after re-running the smoke test so experiments/final_*.csv are regenerated):
    python scripts/compare_smoke.py

Exit code 0 => scientifically equivalent (all numeric metrics identical within
a tiny float tolerance, same file set, same row structure). Exit code 1 => a
difference was found (an optimization changed results and must be reverted).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments" / "_smoke_baseline"
NEW = ROOT / "experiments"
CSVS = ["final_results.csv", "final_multi_seed_summary.csv",
        "final_tables.csv", "final_ablation.csv"]
TOL = 1e-9  # metrics should be bit-for-bit identical; tolerance guards float repr


def _num_cols(df):
    return [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]


def compare_csv(name: str) -> list[str]:
    b, n = BASE / name, NEW / name
    if not b.exists():
        return [f"[skip] baseline missing: {name}"]
    if not n.exists():
        return [f"[FAIL] new output missing: {name}"]
    db, dn = pd.read_csv(b), pd.read_csv(n)
    issues = []
    if list(db.columns) != list(dn.columns):
        issues.append(f"[FAIL] {name}: column mismatch {list(db.columns)} vs {list(dn.columns)}")
        return issues
    if len(db) != len(dn):
        issues.append(f"[FAIL] {name}: row count {len(db)} vs {len(dn)}")
        return issues
    for c in _num_cols(db):
        a, z = db[c].to_numpy(float), dn[c].to_numpy(float)
        both_nan = np.isnan(a) & np.isnan(z)
        diff = ~both_nan & ~(np.abs(a - z) <= TOL)
        if diff.any():
            i = int(np.argmax(diff))
            issues.append(f"[FAIL] {name}: col '{c}' row {i}: {a[i]!r} != {z[i]!r}")
    if not issues:
        issues.append(f"[ok] {name}: identical ({len(db)} rows)")
    return issues


def main() -> int:
    all_msgs, failed = [], False
    for name in CSVS:
        for m in compare_csv(name):
            all_msgs.append(m)
            failed |= m.startswith("[FAIL]")
    print("\n".join(all_msgs))
    print("\nRESULT:", "SCIENTIFIC EQUIVALENCE CONFIRMED" if not failed
          else "DIFFERENCE FOUND — revert the offending optimization")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
