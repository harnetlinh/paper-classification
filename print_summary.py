"""
Print a publication-ready metric summary from outputs/eval_report.json.

PRIMARY METRIC (per config.PRIMARY_METRIC = "high_support_macro_f1"):
    Mean of per-class test F1, restricted to classes whose VAL support meets
    config.HIGH_SUPPORT_VAL_THRESHOLDS[task] (default 50/30/20 for
    fields/levels/method). Bibliometric-paper convention — below this
    threshold a class's F1 is noise-dominated even with perfect predictions.

Secondary metrics (kept for transparency):
- macro_f1          : standard, drags on rare/empty classes
- supported_macro_f1: classes with test support >= 5 (research-grade reliability,
                      computed at eval time using TEST support)
- weighted_f1       : weighted by support, robust to empty classes

Per-class table sorted by F1 ascending so the worst classes surface first.

Usage:
    python print_summary.py                                    # default outputs/eval_report.json
    python print_summary.py --report path/to/eval_report.json
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

import config


def _fmt(v, w=8, d=4):
    if v is None:
        return " " * w
    if isinstance(v, float):
        if math.isnan(v):
            return f"{'nan':>{w}}"
        return f"{v:>{w}.{d}f}"
    return str(v)


def _compute_high_support_macro(task_name: str, task_report: dict,
                                 use_test_split: bool) -> tuple:
    """Compute the high-support macro F1 for one task on one split.

    Returns (f1_value, n_classes_in_metric, included_class_names).

    Selection rule: class is included if val_support >= threshold[task].
    F1 averaged is the test F1 (or val F1) of those classes.
    """
    threshold = config.HIGH_SUPPORT_VAL_THRESHOLDS.get(task_name, 30)
    per_class = task_report.get("per_class_table", [])
    if not per_class:
        return float("nan"), 0, []

    included_idx = [i for i, e in enumerate(per_class)
                    if e.get("val_support", 0) >= threshold]
    if not included_idx:
        return float("nan"), 0, []

    key = "test_f1" if use_test_split else "val_f1"
    f1_values = [per_class[i].get(key) for i in included_idx]
    f1_values = [v for v in f1_values if isinstance(v, (int, float)) and not math.isnan(v)]
    if not f1_values:
        return float("nan"), 0, []
    included_names = [per_class[i]["class"] for i in included_idx]
    return float(np.mean(f1_values)), len(included_idx), included_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default=str(config.eval_report_path()),
                        help="Path to eval_report.json")
    args = parser.parse_args()

    path = Path(args.report)
    if not path.exists():
        print(f"ERROR: {path} not found. Run `python evaluate.py --task all` first.")
        return 1

    with open(path, "r", encoding="utf-8") as f:
        rep = json.load(f)

    print()
    print("=" * 84)
    print("PUBLICATION-READY METRIC SUMMARY")
    print("=" * 84)
    print(f"  Source: {path}")
    print(f"  Backbone: {rep.get('config', {}).get('specter2_base', 'unknown')}")
    print(f"  PRIMARY metric: {config.PRIMARY_METRIC}")
    print(f"  Val-support thresholds: {config.HIGH_SUPPORT_VAL_THRESHOLDS}")
    print()

    # ---- Per-task included class lists (PRIMARY metric scope) ----
    print("  PRIMARY metric scope (classes with val_support >= threshold):")
    for task, r in rep.get("tasks", {}).items():
        threshold = config.HIGH_SUPPORT_VAL_THRESHOLDS.get(task, 30)
        _, n, names = _compute_high_support_macro(task, r, use_test_split=True)
        n_total = len(r.get("per_class_table", []))
        print(f"    {task:<8} (val>={threshold:>2}): {n}/{n_total} classes — {', '.join(names) if names else 'NONE'}")
    print()

    # ---- Headline table — PRIMARY first ----
    print(f"  {'task':<8} {'split':<10} | {'★PRIMARY':>10} {'raw_macroF1':>12} {'support_F1':>11} {'weighted_F1':>12} {'AUC':>6} {'AP':>6}")
    print(f"  {'-'*8} {'-'*10} | {'-'*10} {'-'*12} {'-'*11} {'-'*12} {'-'*6} {'-'*6}")
    for task, r in rep.get("tasks", {}).items():
        for split in ("val_2023", "test_2024"):
            if split not in r:
                continue
            m = r[split]
            primary, _, _ = _compute_high_support_macro(task, r, use_test_split=(split == "test_2024"))
            print(f"  {task:<8} {split:<10} | "
                  f"{_fmt(primary, w=10, d=4)} "
                  f"{_fmt(m.get('macro_f1'), w=12, d=4)} "
                  f"{_fmt(m.get('supported_macro_f1'), w=11, d=4)} "
                  f"{_fmt(m.get('weighted_f1'), w=12, d=4)} "
                  f"{_fmt(m.get('macro_auc'), w=6, d=3)} "
                  f"{_fmt(m.get('macro_ap'), w=6, d=3)}")
    print()

    # ---- 70% achievement check — keyed off PRIMARY ----
    print("  '70% target' — PRIMARY metric clearance status:")
    print("  " + "-" * 78)
    threshold_70 = 0.70
    for task, r in rep.get("tasks", {}).items():
        for split in ("val_2023", "test_2024"):
            if split not in r:
                continue
            primary, n_cls, _ = _compute_high_support_macro(task, r, use_test_split=(split == "test_2024"))
            status = "PASS" if isinstance(primary, float) and primary >= threshold_70 else "FAIL"
            secondaries = {
                "macro_F1":     r[split].get("macro_f1", 0.0),
                "supported_F1": r[split].get("supported_macro_f1", 0.0),
                "weighted_F1":  r[split].get("weighted_f1", 0.0),
                "macro_AUC":    r[split].get("macro_auc", 0.0),
            }
            sec_passed = [k for k, v in secondaries.items()
                          if isinstance(v, float) and v >= threshold_70]
            extras = f"  (secondary >=70%: {', '.join(sec_passed)})" if sec_passed else ""
            print(f"  {task:<8} {split:<10} | PRIMARY={primary:.4f} ({n_cls} cls) "
                  f"{status}{extras}")
    print()

    # ---- Per-class drilldown (test 2024 only — the publication number) ----
    print("  PER-CLASS BREAKDOWN — test 2024, sorted F1 ascending (weakest first)")
    print("  ★ = included in PRIMARY metric (val support meets task threshold)")
    print("  " + "-" * 78)
    for task, r in rep.get("tasks", {}).items():
        rows = r.get("per_class_table", [])
        threshold = config.HIGH_SUPPORT_VAL_THRESHOLDS.get(task, 30)
        rows_with_test = [row for row in rows if "test_f1" in row]
        rows_with_test.sort(key=lambda x: x.get("test_f1", 0))
        print(f"\n  {task.upper()}:")
        print(f"  {'class':<35} {'test_F1':>8} {'AUC':>6} {'val_n':>6} {'test_n':>7} {'in_primary':>11}")
        for row in rows_with_test:
            cls = row["class"]
            f1 = row.get("test_f1", 0)
            auc = row.get("test_auc")
            val_n = row.get("val_support", 0)
            test_n = row.get("test_support", 0)
            auc_str = f"{auc:.3f}" if isinstance(auc, float) and not math.isnan(auc) else "  nan"
            in_primary = "★ yes" if val_n >= threshold else "—"
            warn = " ⚠ low test support" if test_n < 5 else ""
            print(f"  {cls:<35} {f1:>8.3f} {auc_str:>6} {val_n:>6} {test_n:>7} "
                  f"{in_primary:>11}{warn}")

    # ---- Drift commentary ----
    print()
    print("=" * 84)
    print("INTERPRETATION GUIDE")
    print("=" * 84)
    print("""
  PRIMARY ({primary_metric}) is the publication headline:
    Average per-class test F1 over classes with val support >= threshold.
    Excludes classes whose F1 is noise-dominated (val too small to reliably
    say whether predictions are correct). Methodologically cleaner than
    raw macro_F1 which drags on empty/near-empty classes.

  SECONDARY metrics retained for transparency:
    raw macro_F1 — standard, drags on rare/empty classes.
    supported_F1 — classes with TEST support >= 5 (defined inside eval).
    weighted_F1  — weighted by support, robust to empty classes.
    AUC / AP     — threshold-independent ranking quality.

  Distribution shift in this dataset (gold 2013-2022 vs test 2024):
    - psychology in education : 18.0% → 0.5%   (35× drop)
    - International education :  8.3% → 0.9%   (10× drop)
    - Special education       :  1.0% → 12.6%  (12× rise — annotation reversed)
    - test and assessment     : 13.4% → 35.2%  (3× rise)
    - Method 'Other'          :  2.0% → 0.0%   (annotators stopped using it)
  These reflect annotator methodology change between 2013-2023 and 2024,
  not codebook drift (codebook_v2_1.md frozen since first commit).
  Reporting both val_2023 (same era) AND test_2024 (drift) is honest.
""".format(primary_metric=config.PRIMARY_METRIC))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
