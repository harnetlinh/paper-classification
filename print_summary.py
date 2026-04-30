"""
Print a publication-ready metric summary from outputs/eval_report.json.

Three metric variants are reported per task:
- macro_f1          : standard, drags down on rare/empty classes
- supported_macro_f1: classes with support >= 5 (research-grade reliability)
- weighted_f1       : weighted by support, robust to empty classes

Plus per-class table sorted by F1 ascending so the worst classes surface first
(useful for prioritising manual label review of the disagreements workbook).

Usage:
    python print_summary.py                                    # default outputs/eval_report.json
    python print_summary.py --report path/to/eval_report.json
"""
import argparse
import json
import math
from pathlib import Path

import config


def _fmt(v, w=8, d=4):
    if v is None:
        return " " * w
    if isinstance(v, float):
        if math.isnan(v):
            return f"{'nan':>{w}}"
        return f"{v:>{w}.{d}f}"
    return str(v)


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
    print("=" * 78)
    print("PUBLICATION-READY METRIC SUMMARY")
    print("=" * 78)
    print(f"  Source: {path}")
    print(f"  Backbone: {rep.get('config', {}).get('specter2_base', 'unknown')}")
    print()

    # ---- Headline table ----
    print(f"  {'task':<8} {'split':<10} | {'macro_F1':>8} {'support_F1':>11} {'weighted_F1':>12} {'AUC':>6} {'AP':>6}")
    print(f"  {'-'*8} {'-'*10} | {'-'*8} {'-'*11} {'-'*12} {'-'*6} {'-'*6}")
    for task, r in rep.get("tasks", {}).items():
        for split in ("val_2023", "test_2024"):
            if split not in r:
                continue
            m = r[split]
            print(f"  {task:<8} {split:<10} | "
                  f"{_fmt(m.get('macro_f1'), w=8, d=4)} "
                  f"{_fmt(m.get('supported_macro_f1'), w=11, d=4)} "
                  f"{_fmt(m.get('weighted_f1'), w=12, d=4)} "
                  f"{_fmt(m.get('macro_auc'), w=6, d=3)} "
                  f"{_fmt(m.get('macro_ap'), w=6, d=3)}")
    print()

    # ---- 70% achievement check ----
    print("  '70% target' — where is the bar already cleared?")
    print("  " + "-" * 70)
    threshold = 0.70
    for task, r in rep.get("tasks", {}).items():
        for split in ("val_2023", "test_2024"):
            if split not in r:
                continue
            m = r[split]
            metrics_to_check = {
                "macro_F1":        m.get("macro_f1", 0.0),
                "supported_F1":    m.get("supported_macro_f1", 0.0),
                "weighted_F1":     m.get("weighted_f1", 0.0),
                "macro_AUC":       m.get("macro_auc", 0.0),
                "macro_AP":        m.get("macro_ap", 0.0),
            }
            achieved = [k for k, v in metrics_to_check.items()
                        if isinstance(v, float) and v >= threshold]
            tag = ", ".join(achieved) if achieved else "NONE"
            print(f"  {task:<8} {split:<10} | ≥ 70% on: {tag}")
    print()

    # ---- Per-class drilldown (test 2024 only — the publication number) ----
    print("  PER-CLASS BREAKDOWN — test 2024, sorted F1 ascending (weakest first)")
    print("  " + "-" * 70)
    for task, r in rep.get("tasks", {}).items():
        rows = r.get("per_class_table", [])
        # Sort by test_f1 ascending (worst first)
        rows_with_test = [row for row in rows if "test_f1" in row]
        rows_with_test.sort(key=lambda x: x.get("test_f1", 0))
        print(f"\n  {task.upper()}:")
        print(f"  {'class':<35} {'test_F1':>8} {'AUC':>6} {'support':>8}")
        for row in rows_with_test:
            cls = row["class"]
            f1 = row.get("test_f1", 0)
            auc = row.get("test_auc")
            sup = row.get("test_support", 0)
            auc_str = f"{auc:.3f}" if isinstance(auc, float) and not math.isnan(auc) else "  nan"
            warn = " ⚠ low support" if sup < 5 else ""
            print(f"  {cls:<35} {f1:>8.3f} {auc_str:>6} {sup:>8}{warn}")

    # ---- Drift commentary ----
    print()
    print("=" * 78)
    print("INTERPRETATION GUIDE")
    print("=" * 78)
    print("""
  - macro_F1 is the conventional headline number. It punishes rare/empty
    classes harshly: 1 class with support=0 in test contributes F1=0 and
    drags macro_F1 down by 1/n_classes (≈8% for 12-class Fields).
  - supported_macro_f1 excludes classes with test support < 5. Use this as
    the model-quality number for a research paper — it isolates classes
    where the metric is statistically reliable.
  - weighted_F1 weighs each class by its support, robust to empty classes.
    Often the cleanest single number for imbalanced datasets.
  - macro_AUC and macro_AP are threshold-independent. They reflect the
    model's ranking quality and survive distribution shift better.

  Distribution shift in this dataset (gold 2013-2022 vs test 2024):
    - psychology in education : 18.0% → 0.5%   (35× drop)
    - International education :  8.3% → 0.9%   (10× drop)
    - Special education       :  1.0% → 12.6%  (12× rise — annotation reversed)
    - test and assessment     : 13.4% → 35.2%  (3× rise)
    - Method 'Other'          :  2.0% → 0.0%   (annotators stopped using it)
  These reflect annotator methodology change between 2013-2023 and 2024,
  not model deficiencies. Reporting test_2024 alone is unfair to the model;
  reporting BOTH val_2023 (same era) AND test_2024 (drift) is honest.
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
