"""M7 tests: aggregation, table building, honest findings, and output writing
(all torch-free, driven by synthetic result rows)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.eval.experiment import (aggregate, build_tables, honest_findings,  # noqa: E402
                                 results_dataframe, write_outputs)


def _row(seed, method, lam, cos, mia, auc):
    return {
        "seed": seed, "method": method, "lambda": lam,
        "config": method if method != "epu" else f"epu@{lam:g}",
        "macro_auc": auc, "micro_auc": auc, "mAP": auc, "macro_f1": 0.0, "micro_f1": 0.0,
        "precision": 0.1, "recall": 0.5,
        "forget_loss": 0.5, "mia_auc": mia, "pred_dist_to_oracle": 1.0 - cos,
        "expl_cos_to_oracle": cos, "expl_pearson_to_oracle": cos,
        "expl_iou_to_oracle": cos, "expl_mse_to_oracle": 1.0 - cos,
        "wall_time_s": 1.0, "rounds": 2, "speedup_vs_oracle": 3.0,
    }


def _synthetic():
    rows = []
    for s in (42, 43):
        rows += [
            _row(s, "original", None, 0.30, 0.80, 0.60),
            _row(s, "finetune_retain", None, 0.50, 0.55, 0.58),
            _row(s, "oracle", None, 1.00, 0.50, 0.62),
            _row(s, "epu", 0.1, 0.60, 0.54, 0.59),   # beats plain on explanation
            _row(s, "epu", 1.0, 0.40, 0.53, 0.57),   # worse than plain
        ]
    return rows


def test_aggregate_mean_std():
    df = results_dataframe(_synthetic())
    summ = aggregate(df).set_index("config")
    assert summ.loc["finetune_retain", "expl_cos_to_oracle_mean"] == pytest.approx(0.50)
    assert summ.loc["epu@0.1", "expl_cos_to_oracle_mean"] == pytest.approx(0.60)


def test_build_tables_structure_and_ablation():
    df = results_dataframe(_synthetic())
    tables = build_tables(df)
    assert list(tables["comparison"]["config"]) == [
        "original", "finetune_retain", "epu@0.1", "epu@1", "oracle"]
    ab = tables["ablation_lambda"].set_index("config")
    # preservation gain vs plain(0.50): epu@0.1 -> +0.10, epu@1 -> -0.10
    assert ab.loc["epu@0.1", "preservation_gain_vs_plain"] == pytest.approx(0.10)
    assert ab.loc["epu@1", "preservation_gain_vs_plain"] == pytest.approx(-0.10)


def test_honest_findings_reports_truthfully():
    df = results_dataframe(_synthetic())
    f = honest_findings(df)
    assert f["best_epu_config"] == "epu@0.1"
    assert f["any_epu_beats_plain_on_explanation"] is True

    # if EPU never beats plain, findings must say so (no massaging)
    rows = [r for r in _synthetic() if not r["config"].startswith("epu")]
    rows += [_row(s, "epu", 1.0, 0.20, 0.5, 0.5) for s in (42, 43)]  # all worse
    f2 = honest_findings(results_dataframe(rows))
    assert f2["any_epu_beats_plain_on_explanation"] is False


def test_write_outputs_creates_all_files(tmp_path):
    write_outputs(tmp_path, _synthetic(), {"experiment": "test"})
    for name in ("evaluation_results.csv", "multi_seed_summary.csv",
                 "evaluation_tables.csv", "ablation_lambda.csv",
                 "evaluation_summary.json"):
        assert (tmp_path / name).exists(), name
    summary = json.load(open(tmp_path / "evaluation_summary.json"))
    assert "findings" in summary and "tables" in summary
