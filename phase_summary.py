"""
Phase summary: print a side-by-side comparison of all ensemble variants
against the SPECTER2-only baseline.

PRIMARY METRIC: high-support macro_F1 — average per-class test F1, restricted
to classes with VAL support >= HIGH_SUPPORT_VAL_THRESHOLD (default 50).

Why val support, not test support, drives the cutoff:
- Test labels are the metric target; using test support to select classes
  would let class composition leak into the metric definition. Val support
  is the methodologically clean alternative — same codebook v2.1 application
  as test, but separated by year and not used as the evaluation target.
- 50 is the bibliometric-paper convention threshold ("research-grade
  reliability"; below 50 positive samples a class's F1 is noise-dominated).

Raw macro_F1 (12-of-12 / 6-of-6 / 5-of-5 classes) is still printed but
clearly labeled as "contains low-support noise" — useful for completeness
in the publication but not the decision metric.

Usage: python phase_summary.py [--report outputs/eval_report.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np


HIGH_SUPPORT_VAL_THRESHOLD = 50


def _supp_idx_by_val(per_class_table, threshold: int) -> list[int]:
    """Indices of classes whose val_support meets the threshold."""
    return [i for i, e in enumerate(per_class_table)
            if e.get("val_support", 0) >= threshold]


def _supp_idx_by_test(support, threshold: int) -> list[int]:
    return [i for i, s in enumerate(support) if s >= threshold]


def _mean_at(per_class_f1, idx) -> float:
    if not idx:
        return float("nan")
    return float(np.mean([per_class_f1[i] for i in idx if i < len(per_class_f1)]))


def render(task_name, task_report):
    print()
    print("=" * 96)
    print(f" Task: {task_name.upper()}")
    print("=" * 96)
    base = task_report.get("test_2024")
    if not base:
        print(" no test_2024 in report")
        return
    base_pcf = base.get("per_class_f1", [])
    base_sup = base.get("support", [])
    base_macro = base.get("macro_f1", float("nan"))

    per_class_table = task_report.get("per_class_table", [])
    hi_idx = _supp_idx_by_val(per_class_table, HIGH_SUPPORT_VAL_THRESHOLD)
    hi_classes = [per_class_table[i]["class"] for i in hi_idx if i < len(per_class_table)]
    n_total = len(per_class_table)
    n_hi = len(hi_idx)

    base_hi = _mean_at(base_pcf, hi_idx)
    base_supp10 = _mean_at(base_pcf, _supp_idx_by_test(base_sup, 10))

    rows = [("SPECTER2 only (baseline)", base_macro, base_hi, base_supp10, 0.0)]

    # Variant comparison — covers all ensemble paths the eval may have produced.
    for key, label in [
        ("test_2024_quantified", "+ M2 Quantification"),
        ("test_2024_gpt5_ensemble", "+ Phase A GPT-5 panel"),
        ("test_2024_knn_ensemble", "+ Phase D kNN retrieval"),
        ("test_2024_full3_ensemble", "+ GPT-5 + kNN (3-way)"),
    ]:
        r = task_report.get(key)
        if not r:
            continue
        pcf = r.get("per_class_f1", [])
        sup = r.get("support", [])
        macro = r.get("macro_f1", float("nan"))
        hi = _mean_at(pcf, hi_idx)
        supp10 = _mean_at(pcf, _supp_idx_by_test(sup, 10))
        gain = hi - base_hi if not (np.isnan(hi) or np.isnan(base_hi)) else float("nan")
        rows.append((label, macro, hi, supp10, gain))

    # Header: PRIMARY is high-support (val n >= 50)
    print()
    print(f" PRIMARY metric: high-support macro_F1 ({n_hi}/{n_total} classes, "
          f"val support >= {HIGH_SUPPORT_VAL_THRESHOLD})")
    print(f"   Included: {', '.join(hi_classes)}")
    print()
    print(f" {'Variant':<28} {'raw macro_F1':>13} {'★PRIMARY':>11} {'supp(n>=10)':>13} {'PRIMARY Δ':>11}")
    print(f"   {'(noisy: low-supp)':<26} {'':>13} {'(val≥50)':>11} {'(test≥10)':>13} {'vs baseline':>11}")
    print(f" {'-'*28} {'-'*13} {'-'*11} {'-'*13} {'-'*11}")
    for label, macro, hi, supp10, gain in rows:
        gain_str = "  --   " if gain == 0.0 else f"{gain:+.4f}"
        print(f" {label:<28} {macro:>13.4f} {hi:>11.4f} {supp10:>13.4f} {gain_str:>11}")

    # Per-class breakdown using the best available ensemble variant.
    best_ens = (task_report.get("test_2024_full3_ensemble")
                or task_report.get("test_2024_gpt5_ensemble")
                or task_report.get("test_2024_quantified"))
    if best_ens and best_ens.get("per_class_f1") and base_pcf:
        ens_pcf = best_ens["per_class_f1"]
        print()
        print(" Per-class breakdown (test 2024):")
        print(f" {'Class':<35} {'baseline_F1':>12} {'best_ens_F1':>12} {'gain':>10} "
              f"{'val_n':>6} {'test_n':>7} {'in_primary':>11}")
        for i, entry in enumerate(per_class_table):
            cn = entry["class"]
            val_n = entry.get("val_support", 0)
            test_n = entry.get("test_support", 0)
            b = base_pcf[i] if i < len(base_pcf) else float("nan")
            e = ens_pcf[i] if i < len(ens_pcf) else float("nan")
            g = e - b if not (np.isnan(b) or np.isnan(e)) else float("nan")
            mark = " ★" if g > 0.05 else (" ↓" if g < -0.02 else "  ")
            in_primary = "yes" if val_n >= HIGH_SUPPORT_VAL_THRESHOLD else "—"
            print(f" {cn:<35} {b:>12.3f} {e:>12.3f} {g:>+10.3f} "
                  f"{val_n:>6d} {test_n:>7d} {in_primary:>11}{mark}")

    # M2: quantification detail block — estimated test prior + threshold shifts.
    quant = task_report.get("test_2024_quantified")
    if quant:
        print()
        print(f" M2 Quantification ({quant.get('estimator', '?')}):")
        thresholds_baseline = task_report.get("thresholds") or []
        adjusted = quant.get("adjusted_thresholds") or []
        priors = quant.get("estimated_test_prior") or []
        if per_class_table and adjusted and priors:
            print(f" {'Class':<35} {'val_thr':>9} {'adj_thr':>9} {'shift':>9} "
                  f"{'est_prior':>11} {'test_n':>7}")
            for i, entry in enumerate(per_class_table):
                cn = entry["class"]
                n = entry.get("test_support", 0)
                vt = thresholds_baseline[i] if i < len(thresholds_baseline) else float("nan")
                at = adjusted[i] if i < len(adjusted) else float("nan")
                pr = priors[i] if i < len(priors) else float("nan")
                shift = at - vt if not (np.isnan(vt) or np.isnan(at)) else float("nan")
                print(f" {cn:<35} {vt:>9.3f} {at:>9.3f} {shift:>+9.3f} "
                      f"{pr:>11.4f} {n:>7d}")


def main():
    global HIGH_SUPPORT_VAL_THRESHOLD
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="outputs/eval_report.json")
    parser.add_argument("--val-threshold", type=int, default=HIGH_SUPPORT_VAL_THRESHOLD,
                        help="Minimum val support for a class to count in the "
                             "primary high-support metric (default 50)")
    args = parser.parse_args()
    HIGH_SUPPORT_VAL_THRESHOLD = args.val_threshold

    report_path = Path(args.report)
    if not report_path.exists():
        print(f"Report not found: {report_path}")
        print("Run: python evaluate.py")
        return 1

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    print()
    print("=" * 96)
    print(" PHASE-BY-PHASE COMPARISON")
    print(f" report: {report_path}")
    print(f" backbone: {report.get('config', {}).get('specter2_base', '?')}")
    print(f" PRIMARY metric: high-support macro_F1 (val support >= {HIGH_SUPPORT_VAL_THRESHOLD})")
    print(" RAW macro_F1 is reported but flagged as containing low-support noise.")
    print("=" * 96)

    for task_name, task_report in report.get("tasks", {}).items():
        render(task_name, task_report)

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
