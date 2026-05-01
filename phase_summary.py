"""
Phase summary: print a side-by-side comparison of all ensemble variants
against the SPECTER2-only baseline, focused on supported_macro_F1 (n>=10),
the metric chosen for publication.

Reads the most recent eval_report.json and renders a single table per task:

    Variant                        macro_F1     supp_F1(n>=5)  supp_F1(n>=10)  gain
    SPECTER2 only (baseline)       0.4090       0.4727         0.4660          --
    + GPT-5 panel                  0.5xxx       0.5xxx         0.5xxx          +0.0xxx
    + kNN retrieval                0.4xxx       0.4xxx         0.4xxx          +0.0xxx
    + GPT-5 + kNN (full 3-way)     0.5xxx       0.5xxx         0.5xxx          +0.0xxx

Usage: python phase_summary.py [--report outputs/eval_report.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np


def supp_macro_at(per_class_f1, support, threshold):
    idx = [i for i, s in enumerate(support) if s >= threshold]
    if not idx:
        return float("nan")
    return float(np.mean([per_class_f1[i] for i in idx]))


def render(task_name, task_report):
    print()
    print("=" * 90)
    print(f" Task: {task_name.upper()}")
    print("=" * 90)
    base = task_report.get("test_2024")
    if not base:
        print(" no test_2024 in report")
        return
    base_pcf = base.get("per_class_f1", [])
    base_sup = base.get("support", [])
    base_macro = base.get("macro_f1", float("nan"))
    base_supp5 = supp_macro_at(base_pcf, base_sup, 5)
    base_supp10 = supp_macro_at(base_pcf, base_sup, 10)

    rows = [("SPECTER2 only (baseline)", base_macro, base_supp5, base_supp10, 0.0)]

    for key, label in [
        ("test_2024_gpt5_ensemble", "+ GPT-5 panel"),
        ("test_2024_knn_ensemble", "+ kNN retrieval"),
        ("test_2024_full3_ensemble", "+ GPT-5 + kNN (3-way)"),
    ]:
        r = task_report.get(key)
        if not r:
            continue
        pcf = r.get("per_class_f1", [])
        sup = r.get("support", [])
        macro = r.get("macro_f1", float("nan"))
        s5 = supp_macro_at(pcf, sup, 5)
        s10 = supp_macro_at(pcf, sup, 10)
        gain = s10 - base_supp10 if not np.isnan(s10) and not np.isnan(base_supp10) else float("nan")
        rows.append((label, macro, s5, s10, gain))

    print(f" {'Variant':<32} {'macro_F1':>10} {'supp(n>=5)':>12} {'supp(n>=10)':>13} {'gain':>10}")
    print(f" {'-'*32} {'-'*10} {'-'*12} {'-'*13} {'-'*10}")
    for label, macro, s5, s10, gain in rows:
        gain_str = "  --   " if gain == 0.0 else f"{gain:+.4f}"
        print(f" {label:<32} {macro:>10.4f} {s5:>12.4f} {s10:>13.4f} {gain_str:>10}")

    # Per-class gain breakdown (for rare/struggling classes)
    full3 = task_report.get("test_2024_full3_ensemble") or task_report.get("test_2024_gpt5_ensemble")
    if full3 and full3.get("per_class_f1") and base_pcf:
        ens_pcf = full3["per_class_f1"]
        per_class_table = task_report.get("per_class_table", [])
        print()
        print(" Per-class breakdown (test 2024):")
        print(f" {'Class':<35} {'baseline_F1':>12} {'best_ens_F1':>12} {'gain':>10} {'support':>8}")
        for i, entry in enumerate(per_class_table):
            cn = entry["class"]
            n = entry.get("test_support", 0)
            b = base_pcf[i] if i < len(base_pcf) else float("nan")
            e = ens_pcf[i] if i < len(ens_pcf) else float("nan")
            g = e - b if not np.isnan(b) and not np.isnan(e) else float("nan")
            mark = " ★" if g > 0.05 else (" ↓" if g < -0.02 else "  ")
            tag = " (n<10, excluded from supp)" if n < 10 else ""
            print(f" {cn:<35} {b:>12.3f} {e:>12.3f} {g:>+10.3f} {n:>8d}{mark}{tag}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="outputs/eval_report.json")
    args = parser.parse_args()

    report_path = Path(args.report)
    if not report_path.exists():
        print(f"Report not found: {report_path}")
        print("Run: python evaluate.py")
        return 1

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    print()
    print("=" * 90)
    print(" PHASE-BY-PHASE COMPARISON")
    print(f" report: {report_path}")
    print(f" backbone: {report.get('config', {}).get('specter2_base', '?')}")
    print("=" * 90)

    for task_name, task_report in report.get("tasks", {}).items():
        render(task_name, task_report)

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
