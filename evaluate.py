"""
Phase 3: Evaluation
====================

Generate comprehensive evaluation report:
- Macro-F1, Micro-F1 cho each task
- Per-class F1 với class names + tuned thresholds
- DUAL evaluation:
    - Val 2023 (gold, clean labels) → reliable F1
    - Test 2024 (sanitized labels, có drift) → real-world F1
- Drift comparison: report gap giữa val_F1 và test_F1 để detect distribution shift

Usage:
    python evaluate.py
    python evaluate.py --task fields           # only one task
    python evaluate.py --output custom.json    # custom output path
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score,
)

import config
import utils
from train_specter2 import (
    SpecterClassifier, load_train_val_test,
    get_target_cols, get_class_names, predict_probs,
    _model_save_path,
)
import ensemble
try:
    import llm_classify
    _LLM_CLASSIFY_AVAILABLE = True
except ImportError:
    _LLM_CLASSIFY_AVAILABLE = False
try:
    import knn_retrieval
    _KNN_AVAILABLE = True
except ImportError:
    _KNN_AVAILABLE = False


def _list_seed_models(task: str):
    """Return [(seed, path), ...] for every seed checkpoint that exists.

    Prefers Phase-F `_trainval` checkpoints when present (they're the strictly
    later, train+val-fitted refits), otherwise falls back to the standard
    train→val checkpoints. Always includes config.SEED plus every additional
    ENSEMBLE_SEEDS entry that has a saved file.
    """
    seeds_to_try = [config.SEED] + [
        s for s in getattr(config, "ENSEMBLE_SEEDS", []) if s != config.SEED
    ]
    found = []
    for seed in seeds_to_try:
        # Phase F refits take precedence if both exist.
        p_trainval = _model_save_path(task, seed, include_val=True)
        if p_trainval.exists():
            found.append((seed, p_trainval))
            continue
        p = _model_save_path(task, seed, include_val=False)
        if p.exists():
            found.append((seed, p))
    return found


def _ensemble_predict(models, loader, device, target_type):
    """Average probabilities across an ensemble of models."""
    sum_probs = None
    targets = None
    for m in models:
        probs, t = predict_probs(m, loader, device, target_type)
        sum_probs = probs if sum_probs is None else sum_probs + probs
        targets = t
    return sum_probs / len(models), targets


def _tta_loaders(df, tokenizer, target_cols, target_type, method_to_idx, max_length, batch_size):
    """Build a list of (variant_name, DataLoader) for the configured TTA variants.

    Each variant materialises a separate PaperDataset that reorders Title/Abstract
    in the [SEP]-separated input. Falls back to a single canonical loader if no
    TTA variants are configured.
    """
    variants = getattr(config, "TTA_VARIANTS", None) or ["title_then_abstract"]
    loaders = []
    for variant in variants:
        df_view = df.copy()
        if variant == "abstract_then_title":
            # Swap so the encoder sees Abstract before Title.
            df_view = df_view.rename(columns={"Title": "_Title_orig", "Abstract": "_Abstract_orig"})
            df_view["Title"] = df_view["_Abstract_orig"]
            df_view["Abstract"] = df_view["_Title_orig"]
        ds = utils.PaperDataset(
            df_view, tokenizer,
            target_cols=target_cols, target_type=target_type,
            method_to_idx=method_to_idx, max_length=max_length,
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        loaders.append((variant, loader))
    return loaders


def _ensemble_predict_with_tta(models, loaders, device, target_type):
    """Average probabilities across an ensemble of models AND TTA variants.

    Cardinality of the average: len(models) * len(loaders).
    Targets come from the first loader (all TTA loaders share the same labels).
    """
    sum_probs = None
    targets = None
    n = 0
    for m in models:
        for variant, loader in loaders:
            probs, t = predict_probs(m, loader, device, target_type)
            sum_probs = probs if sum_probs is None else sum_probs + probs
            if targets is None:
                targets = t
            n += 1
    return sum_probs / max(1, n), targets

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# Per-class support below this is considered statistically unreliable.
LOW_SUPPORT_THRESHOLD = 5
NAN_STR = "  nan"


def _per_class_auc_ap(targets, probs, target_type, n_classes):
    """Compute per-class AUC + Average Precision.

    Returns (auc_list, ap_list, per_class_support_2d) where each list has length
    n_classes. Classes with all-same labels (no positive or no negative) get
    NaN — those F1/AUC are undefined and should not contaminate macro means.
    """
    aucs, aps = [], []
    for c in range(n_classes):
        if target_type == "multi_label":
            y_true = targets[:, c].astype(int)
        else:
            y_true = (targets.astype(int) == c).astype(int)
        y_score = probs[:, c]

        n_pos = int(y_true.sum())
        n_neg = int(len(y_true) - n_pos)

        if n_pos == 0 or n_neg == 0:
            aucs.append(float("nan"))
            aps.append(float("nan"))
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            aucs.append(float(roc_auc_score(y_true, y_score)))
            aps.append(float(average_precision_score(y_true, y_score)))
    return aucs, aps


def evaluate_with_thresholds(probs, targets, target_type, n_classes, thresholds=None):
    """
    Compute metrics given probabilities + targets.
    
    Multi-label: applies per-class thresholds, returns macro/micro/per-class F1.
    Single-label: argmax, returns macro/micro/per-class F1.
    """
    if target_type == "multi_label":
        if thresholds is None:
            thresholds = [0.5] * n_classes
        preds = np.zeros_like(probs, dtype=int)
        for c in range(n_classes):
            preds[:, c] = (probs[:, c] >= thresholds[c]).astype(int)
        macro_f1 = f1_score(targets, preds, average="macro", zero_division=0)
        micro_f1 = f1_score(targets, preds, average="micro", zero_division=0)
        per_p, per_r, per_f1, support = precision_recall_fscore_support(
            targets, preds, average=None, zero_division=0
        )
    else:
        targets_int = targets.astype(int)
        preds = probs.argmax(axis=1)
        macro_f1 = f1_score(targets_int, preds, average="macro", zero_division=0)
        micro_f1 = f1_score(targets_int, preds, average="micro", zero_division=0)
        per_p, per_r, per_f1, support = precision_recall_fscore_support(
            targets_int, preds, average=None, zero_division=0,
            labels=list(range(n_classes)),
        )
    
    aucs, aps = _per_class_auc_ap(targets, probs, target_type, n_classes)
    macro_auc = float(np.nanmean(aucs)) if any(not np.isnan(a) for a in aucs) else float("nan")
    macro_ap = float(np.nanmean(aps)) if any(not np.isnan(a) for a in aps) else float("nan")

    # Weighted F1: each class contributes proportionally to its support.
    # Mathematically robust when one class has 0 support — it just gets weight 0.
    weighted_f1 = float(f1_score(targets, preds, average="weighted", zero_division=0))

    # Supported-class macro F1: only classes with support >= MIN_SUPPORT.
    # Reports what the model achieves on classes where the metric is reliable.
    # Particularly important when the dataset has very imbalanced classes
    # (e.g. Method 'Other' with 0 support, Fields 'Special edu' with val=1
    # — these contribute pure noise to standard macro F1).
    supports = list(support)
    MIN_SUPPORT = 5
    supported_idx = [i for i, s in enumerate(supports) if s >= MIN_SUPPORT]
    supported_f1 = (
        float(np.mean([per_f1[i] for i in supported_idx])) if supported_idx else float("nan")
    )

    # Test-30 macro: classes with support >= 30 (research-grade reliability).
    test30_idx = [i for i, s in enumerate(supports) if s >= 30]
    test30_f1 = (
        float(np.mean([per_f1[i] for i in test30_idx])) if test30_idx else float("nan")
    )

    return {
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "weighted_f1": weighted_f1,
        "macro_auc": macro_auc,
        "macro_ap": macro_ap,
        "supported_macro_f1": supported_f1,
        "supported_classes_count": len(supported_idx),
        "high_support_macro_f1": test30_f1,
        "high_support_classes_count": len(test30_idx),
        "per_class_precision": [float(x) for x in per_p],
        "per_class_recall": [float(x) for x in per_r],
        "per_class_f1": [float(x) for x in per_f1],
        "per_class_auc": aucs,
        "per_class_ap": aps,
        "support": [int(x) for x in support],
    }


def _maybe_load_gpt5_panel(split: str, df, n_classes: int, task: str):
    """If a GPT-5 panel parquet exists for this split, load it and return the
    [N, n_classes] aligned probabilities for the relevant task; else return None.

    The GPT-5 panel parquet contains predictions for ALL 3 tasks per paper;
    we extract only the task-relevant block here.
    """
    if not _LLM_CLASSIFY_AVAILABLE:
        return None, 0
    try:
        ids, fields_p, levels_p, method_p = llm_classify.load_panel_predictions(split)
    except FileNotFoundError:
        return None, 0
    if task == "fields":
        panel_probs = fields_p
    elif task == "levels":
        panel_probs = levels_p
    else:  # method
        panel_probs = method_p
    aligned, n_match = ensemble.align_gpt5_to_df(df, ids, panel_probs, n_classes)
    return aligned, n_match


def _maybe_load_knn(split: str, df, n_classes: int, task: str):
    """Load kNN-retrieval soft probs for the split×task if the parquet exists."""
    if not _KNN_AVAILABLE:
        return None, 0
    try:
        ids, probs = knn_retrieval.load_knn_probs(split, task)
    except FileNotFoundError:
        return None, 0
    aligned, n_match = ensemble.align_gpt5_to_df(df, ids, probs, n_classes)
    return aligned, n_match


def evaluate_task(task: str, device, val_loader, test_loader,
                  target_type: str, n_classes: int,
                  val_df=None, test_df=None, tokenizer=None,
                  target_cols=None, method_to_idx=None):
    """Run evaluation for one task on both val and test sets.

    Auto-detects ensemble: if multiple seed checkpoints exist
    (`model_{task}.pt` + `model_{task}_s{seed}.pt`), loads all of them and
    averages probabilities. Falls back to a single model if only one exists.

    If `val_df` / `tokenizer` are provided, also applies test-time augmentation
    using `config.TTA_VARIANTS` (defaults to title-then-abstract +
    abstract-then-title). Probabilities are averaged across (models × variants).
    """
    from transformers import AutoTokenizer

    seed_models_paths = _list_seed_models(task)
    if not seed_models_paths:
        print(f"[SKIP] {task}: no model checkpoint found at {config.model_path(task)}")
        return None

    print(f"\n=== Evaluating {task} ===")
    if len(seed_models_paths) > 1:
        print(f"  Ensemble: averaging {len(seed_models_paths)} seed models "
              f"({[s for s, _ in seed_models_paths]})")
    models = []
    for seed, path in seed_models_paths:
        m = SpecterClassifier(
            config.BACKBONE_MODEL, n_classes=n_classes,
            dropout=getattr(config, "DROPOUT", 0.1),
            revision=getattr(config, "BACKBONE_REVISION", None),
        ).to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        m.eval()
        models.append(m)
    # All probability prediction below goes through _ensemble_predict, which
    # handles the n=1 case identically to a single-model evaluation.
    
    # Load tuned thresholds
    thresholds = None
    if target_type == "multi_label" and config.threshold_path(task).exists():
        with open(config.threshold_path(task), "r", encoding="utf-8") as f:
            tdata = json.load(f)
        thresholds = tdata.get("thresholds")
        if thresholds is not None and len(thresholds) != n_classes:
            print(f"  WARNING: threshold file has {len(thresholds)} entries but task has {n_classes} classes — falling back to 0.5")
            thresholds = None
        if thresholds is not None:
            print(f"  Using tuned thresholds: {thresholds}")
    
    class_names = get_class_names(task)
    
    # Predict on val + test
    # If TTA inputs provided, build per-variant loaders and average across them.
    use_tta = (val_df is not None and tokenizer is not None
               and bool(getattr(config, "TTA_VARIANTS", None)))
    if use_tta:
        val_loaders = _tta_loaders(
            val_df, tokenizer, target_cols, target_type, method_to_idx,
            config.MAX_LENGTH, config.effective_batch_size(),
        )
        print(f"  TTA variants on val: {[v for v, _ in val_loaders]}")
        val_probs, val_targets = _ensemble_predict_with_tta(
            models, val_loaders, device, target_type,
        )
    else:
        val_probs, val_targets = _ensemble_predict(models, val_loader, device, target_type)
    val_metrics = evaluate_with_thresholds(
        val_probs, val_targets, target_type, n_classes, thresholds
    )
    
    test_metrics = None
    if test_loader is not None and len(test_loader.dataset) > 0:
        if use_tta and test_df is not None:
            test_loaders = _tta_loaders(
                test_df, tokenizer, target_cols, target_type, method_to_idx,
                config.MAX_LENGTH, config.effective_batch_size(),
            )
            print(f"  TTA variants on test: {[v for v, _ in test_loaders]}")
            test_probs, test_targets = _ensemble_predict_with_tta(
                models, test_loaders, device, target_type,
            )
        else:
            test_probs, test_targets = _ensemble_predict(models, test_loader, device, target_type)
        test_metrics = evaluate_with_thresholds(
            test_probs, test_targets, target_type, n_classes, thresholds
        )

    # ===== Phase A: GPT-5 panel ensemble (Phase A) =====
    # If GPT-5 panel parquets exist for both val and test, run a second pass
    # that ensembles SPECTER2 probs with GPT-5 panel probs via per-class lambda
    # tuned on val. Reports under "val_2023_gpt5_ensemble" / "test_2024_gpt5_ensemble"
    # keys; the original specter-only metrics stay as the primary report fields.
    val_gpt5_metrics = None
    test_gpt5_metrics = None
    val_gpt5 = None
    test_gpt5 = None
    if val_df is not None:
        val_gpt5, n_match_val = _maybe_load_gpt5_panel("val", val_df, n_classes, task)
    if test_df is not None and len(test_df) > 0:
        test_gpt5, n_match_test = _maybe_load_gpt5_panel("test", test_df, n_classes, task)

    ensemble_lambdas = None
    if val_gpt5 is not None and test_gpt5 is not None:
        print(f"\n  GPT-5 panel ensemble: val matched {n_match_val}/{len(val_df)}, "
              f"test matched {n_match_test}/{len(test_df)}")
        # Tune per-class lambda on val (use the same val_thresholds)
        thresh_arr = np.asarray(thresholds, dtype=np.float64) if thresholds else np.full(n_classes, 0.5)
        val_blend, test_blend, lambdas = ensemble.build_ensemble_probs(
            val_probs, val_gpt5, val_targets,
            test_probs, test_gpt5,
            target_type=target_type,
            val_thresholds=thresh_arr,
        )
        ensemble_lambdas = lambdas.tolist()
        val_gpt5_metrics = evaluate_with_thresholds(
            val_blend, val_targets, target_type, n_classes, thresholds
        )
        test_gpt5_metrics = evaluate_with_thresholds(
            test_blend, test_targets, target_type, n_classes, thresholds
        )
        print(f"  Ensemble lambdas: {[round(l, 2) for l in ensemble_lambdas]}")
        print("    (lambda=1.0 → SPECTER2 only; lambda=0.0 → GPT-5 only)")

    # ===== Phase D: kNN retrieval ensemble =====
    val_knn = None
    test_knn = None
    val_knn_metrics = None
    test_knn_metrics = None
    val_full3_metrics = None
    test_full3_metrics = None
    full3_weights = None
    if val_df is not None:
        val_knn, n_match_val_k = _maybe_load_knn("val", val_df, n_classes, task)
    if test_df is not None and len(test_df) > 0:
        test_knn, n_match_test_k = _maybe_load_knn("test", test_df, n_classes, task)

    if val_knn is not None and test_knn is not None and target_type == "multi_label":
        print(f"\n  kNN retrieval ensemble: val matched {n_match_val_k}/{len(val_df)}, "
              f"test matched {n_match_test_k}/{len(test_df)}")
        thresh_arr = np.asarray(thresholds, dtype=np.float64) if thresholds else np.full(n_classes, 0.5)
        # 2-way blend: SPECTER2 + kNN (per-class lambda)
        knn_lambdas = ensemble.tune_ensemble_lambda_multilabel(
            val_probs, val_knn, val_targets, thresh_arr,
        )
        val_knn_blend = ensemble.apply_ensemble(val_probs, val_knn, knn_lambdas)
        test_knn_blend = ensemble.apply_ensemble(test_probs, test_knn, knn_lambdas)
        val_knn_metrics = evaluate_with_thresholds(
            val_knn_blend, val_targets, target_type, n_classes, thresholds
        )
        test_knn_metrics = evaluate_with_thresholds(
            test_knn_blend, test_targets, target_type, n_classes, thresholds
        )
        # 3-way blend: SPECTER2 + GPT-5 + kNN (per-class triple)
        if val_gpt5 is not None and test_gpt5 is not None:
            full3_weights = ensemble.tune_ensemble_weights_3way_multilabel(
                val_probs, val_gpt5, val_knn, val_targets, thresh_arr,
            )
            val_full3 = ensemble.apply_ensemble_3way(val_probs, val_gpt5, val_knn, full3_weights)
            test_full3 = ensemble.apply_ensemble_3way(test_probs, test_gpt5, test_knn, full3_weights)
            val_full3_metrics = evaluate_with_thresholds(
                val_full3, val_targets, target_type, n_classes, thresholds
            )
            test_full3_metrics = evaluate_with_thresholds(
                test_full3, test_targets, target_type, n_classes, thresholds
            )

    # Build report
    report = {
        "task": task,
        "target_type": target_type,
        "n_classes": n_classes,
        "thresholds": thresholds,
        "val_2023": {
            "n_papers": len(val_loader.dataset),
            **val_metrics,
        },
    }
    if test_metrics is not None:
        report["test_2024"] = {
            "n_papers": len(test_loader.dataset),
            **test_metrics,
        }
        # Drift gap
        report["drift_gap"] = {
            "macro_f1_val_minus_test": round(
                val_metrics["macro_f1"] - test_metrics["macro_f1"], 4
            ),
            "interpretation": (
                "Negative or zero = test ≥ val (no/minimal drift). "
                "Large positive = val better than test (drift detected). "
                "Expected ~0.05-0.10 due to known annotator drift in 2024."
            ),
        }

    # Wire Phase A ensemble metrics into the report (when available).
    if val_gpt5_metrics is not None:
        report["val_2023_gpt5_ensemble"] = {
            "n_papers": len(val_loader.dataset),
            **val_gpt5_metrics,
        }
    if test_gpt5_metrics is not None:
        report["test_2024_gpt5_ensemble"] = {
            "n_papers": len(test_loader.dataset),
            **test_gpt5_metrics,
        }
    if ensemble_lambdas is not None:
        report["gpt5_ensemble_lambdas"] = ensemble_lambdas

    # Wire Phase D + 3-way ensemble metrics
    if val_knn_metrics is not None:
        report["val_2023_knn_ensemble"] = {
            "n_papers": len(val_loader.dataset),
            **val_knn_metrics,
        }
    if test_knn_metrics is not None:
        report["test_2024_knn_ensemble"] = {
            "n_papers": len(test_loader.dataset),
            **test_knn_metrics,
        }
    if val_full3_metrics is not None:
        report["val_2023_full3_ensemble"] = {
            "n_papers": len(val_loader.dataset),
            **val_full3_metrics,
        }
    if test_full3_metrics is not None:
        report["test_2024_full3_ensemble"] = {
            "n_papers": len(test_loader.dataset),
            **test_full3_metrics,
        }
    if full3_weights is not None:
        report["full3_ensemble_weights"] = full3_weights.tolist()
    
    # Pretty per-class table
    report["per_class_table"] = []
    low_support_classes = []
    for i, cn in enumerate(class_names):
        if target_type == "multi_label":
            threshold_value = thresholds[i] if thresholds else 0.5
        else:
            threshold_value = None
        entry = {
            "class": cn,
            "threshold": threshold_value,
            "val_f1": val_metrics["per_class_f1"][i],
            "val_precision": val_metrics["per_class_precision"][i],
            "val_recall": val_metrics["per_class_recall"][i],
            "val_auc": val_metrics["per_class_auc"][i],
            "val_ap": val_metrics["per_class_ap"][i],
            "val_support": val_metrics["support"][i],
        }
        if val_metrics["support"][i] < LOW_SUPPORT_THRESHOLD:
            entry["low_support_warning"] = (
                f"val support={val_metrics['support'][i]} "
                f"(< {LOW_SUPPORT_THRESHOLD}) — F1 estimate is noisy; prefer AUC/AP."
            )
            low_support_classes.append(cn)
        if test_metrics is not None:
            entry["test_f1"] = test_metrics["per_class_f1"][i]
            entry["test_precision"] = test_metrics["per_class_precision"][i]
            entry["test_recall"] = test_metrics["per_class_recall"][i]
            entry["test_auc"] = test_metrics["per_class_auc"][i]
            entry["test_ap"] = test_metrics["per_class_ap"][i]
            entry["test_support"] = test_metrics["support"][i]
        report["per_class_table"].append(entry)
    if low_support_classes:
        report["low_support_classes"] = low_support_classes

    # Print summary
    def _fmt(m, key, default=float("nan")):
        v = m.get(key, default)
        if isinstance(v, float) and np.isfinite(v):
            return f"{v:.4f}"
        return NAN_STR
    print(f"  Val 2023 (n={len(val_loader.dataset)}):")
    print(f"    Macro-F1:        {_fmt(val_metrics, 'macro_f1')}")
    print(f"    Micro-F1:        {_fmt(val_metrics, 'micro_f1')}")
    print(f"    Weighted-F1:     {_fmt(val_metrics, 'weighted_f1')}")
    print(f"    Macro-AUC:       {_fmt(val_metrics, 'macro_auc')}")
    print(f"    Macro-AP:        {_fmt(val_metrics, 'macro_ap')}")
    print(f"    Supported macro-F1 (n>=5):  {_fmt(val_metrics, 'supported_macro_f1')}  "
          f"({val_metrics.get('supported_classes_count', 0)} classes)")
    print(f"    High-support macro-F1 (n>=30): {_fmt(val_metrics, 'high_support_macro_f1')}  "
          f"({val_metrics.get('high_support_classes_count', 0)} classes)")
    if test_metrics:
        print(f"  Test 2024 (n={len(test_loader.dataset)}):")
        print(f"    Macro-F1:        {_fmt(test_metrics, 'macro_f1')}")
        print(f"    Micro-F1:        {_fmt(test_metrics, 'micro_f1')}")
        print(f"    Weighted-F1:     {_fmt(test_metrics, 'weighted_f1')}")
        print(f"    Macro-AUC:       {_fmt(test_metrics, 'macro_auc')}")
        print(f"    Macro-AP:        {_fmt(test_metrics, 'macro_ap')}")
        print(f"    Supported macro-F1 (n>=5):  {_fmt(test_metrics, 'supported_macro_f1')}  "
              f"({test_metrics.get('supported_classes_count', 0)} classes)")
        print(f"    High-support macro-F1 (n>=30): {_fmt(test_metrics, 'high_support_macro_f1')}  "
              f"({test_metrics.get('high_support_classes_count', 0)} classes)")
        print(f"    Drift gap (val - test macro-F1): "
              f"{report['drift_gap']['macro_f1_val_minus_test']:+.4f}")

    # Phase A: GPT-5 ensemble summary side-by-side with SPECTER2-only baseline
    if val_gpt5_metrics is not None and test_gpt5_metrics is not None:
        print("\n  ===== Phase A: GPT-5 panel ensemble =====")
        print("  Val 2023 ensemble:")
        print(f"    Macro-F1:                  {_fmt(val_gpt5_metrics, 'macro_f1')}  "
              f"(specter-only: {_fmt(val_metrics, 'macro_f1')})")
        print(f"    Supported macro-F1 (n>=5): {_fmt(val_gpt5_metrics, 'supported_macro_f1')}  "
              f"(specter-only: {_fmt(val_metrics, 'supported_macro_f1')})")
        print("  Test 2024 ensemble:")
        print(f"    Macro-F1:                  {_fmt(test_gpt5_metrics, 'macro_f1')}  "
              f"(specter-only: {_fmt(test_metrics, 'macro_f1')})")
        print(f"    Supported macro-F1 (n>=5): {_fmt(test_gpt5_metrics, 'supported_macro_f1')}  "
              f"(specter-only: {_fmt(test_metrics, 'supported_macro_f1')})")
        # supported_macro_F1 (n>=10) — the metric the user explicitly chose for publication
        support = test_gpt5_metrics.get("support", [])
        per_class_f1 = test_gpt5_metrics.get("per_class_f1", [])
        per_class_f1_baseline = test_metrics.get("per_class_f1", [])
        idx10 = [i for i, s in enumerate(support) if s >= 10]
        if idx10 and per_class_f1 and per_class_f1_baseline:
            ens10 = float(np.mean([per_class_f1[i] for i in idx10]))
            base10 = float(np.mean([per_class_f1_baseline[i] for i in idx10]))
            print(f"    Supported macro-F1 (n>=10, primary metric): "
                  f"{ens10:.4f}  (specter-only: {base10:.4f})  "
                  f"gain: {ens10 - base10:+.4f}")

    # Phase D: kNN ensemble + 3-way summary
    if test_knn_metrics is not None:
        print("\n  ===== Phase D: kNN retrieval ensemble (SPECTER2 + kNN) =====")
        print(f"  Test 2024 macro-F1:        {_fmt(test_knn_metrics, 'macro_f1')}  "
              f"(specter-only: {_fmt(test_metrics, 'macro_f1')})")
        print(f"  Test 2024 supp_macro_F1 (n>=5): {_fmt(test_knn_metrics, 'supported_macro_f1')}  "
              f"(specter-only: {_fmt(test_metrics, 'supported_macro_f1')})")
        sup = test_knn_metrics.get("support", [])
        pcf = test_knn_metrics.get("per_class_f1", [])
        pcf_b = test_metrics.get("per_class_f1", [])
        idx10k = [i for i, s in enumerate(sup) if s >= 10]
        if idx10k and pcf and pcf_b:
            print(f"  Test supp_macro_F1 (n>=10): "
                  f"{float(np.mean([pcf[i] for i in idx10k])):.4f}  "
                  f"(specter-only: {float(np.mean([pcf_b[i] for i in idx10k])):.4f})")
    if test_full3_metrics is not None:
        print("\n  ===== Full 3-way ensemble (SPECTER2 + GPT-5 + kNN) =====")
        print(f"  Test 2024 macro-F1:        {_fmt(test_full3_metrics, 'macro_f1')}")
        print(f"  Test 2024 supp_macro_F1 (n>=5): {_fmt(test_full3_metrics, 'supported_macro_f1')}")
        sup = test_full3_metrics.get("support", [])
        pcf = test_full3_metrics.get("per_class_f1", [])
        pcf_b = test_metrics.get("per_class_f1", [])
        idx10f = [i for i, s in enumerate(sup) if s >= 10]
        if idx10f and pcf and pcf_b:
            ens10 = float(np.mean([pcf[i] for i in idx10f]))
            base10 = float(np.mean([pcf_b[i] for i in idx10f]))
            print(f"  Test supp_macro_F1 (n>=10, primary metric): "
                  f"{ens10:.4f}  (specter-only: {base10:.4f})  "
                  f"gain: {ens10 - base10:+.4f}")

    print("  Per-class metrics (val 2023):")
    for entry in report["per_class_table"]:
        cn = entry["class"]
        v_f1 = entry["val_f1"]
        v_auc = entry["val_auc"]
        v_n = entry["val_support"]
        warn_tag = " ⚠low" if v_n < LOW_SUPPORT_THRESHOLD else ""
        auc_str = f"{v_auc:.3f}" if not np.isnan(v_auc) else NAN_STR
        if test_metrics:
            t_f1 = entry["test_f1"]
            t_auc = entry["test_auc"]
            t_n = entry["test_support"]
            t_auc_str = f"{t_auc:.3f}" if not np.isnan(t_auc) else NAN_STR
            print(f"    {cn:35s}  val_F1={v_f1:.3f} AUC={auc_str} (n={v_n}){warn_tag}  "
                  f"test_F1={t_f1:.3f} AUC={t_auc_str} (n={t_n})")
        else:
            print(f"    {cn:35s}  val_F1={v_f1:.3f} AUC={auc_str} (n={v_n}){warn_tag}")

    if low_support_classes:
        print(f"  Note: {len(low_support_classes)} class(es) with val support < "
              f"{LOW_SUPPORT_THRESHOLD} → F1 noisy. Trust AUC/AP + test_F1 for those.")

    return report


def build_eval_inputs(task: str):
    """Return everything needed to build val/test loaders (with optional TTA).

    Replaces the previous `build_loaders` two-DataLoader return — TTA needs
    the underlying DataFrame + tokenizer to materialise variant loaders, so
    we expose those here and let the caller build loaders on demand.
    """
    from transformers import AutoTokenizer

    train_df, val_df, test_df = load_train_val_test(task)

    if task in ("fields", "levels"):
        target_cols = get_target_cols(task)
        n_classes = len(target_cols)
        target_type = "multi_label"
    else:
        target_cols = None
        n_classes = len(config.METHODS_5)
        target_type = "single_label"

    tokenizer = AutoTokenizer.from_pretrained(config.BACKBONE_MODEL)
    method_to_idx = {m: i for i, m in enumerate(config.METHODS_5)}

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "tokenizer": tokenizer,
        "target_cols": target_cols,
        "target_type": target_type,
        "n_classes": n_classes,
        "method_to_idx": method_to_idx,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["fields", "levels", "method", "all"], default="all")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: outputs/eval_report.json)")
    args = parser.parse_args()
    
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    tasks = ["fields", "levels", "method"] if args.task == "all" else [args.task]
    
    full_report = {
        "config": {
            "seed": config.SEED,
            "specter2_base": config.BACKBONE_MODEL,
            "max_length": config.MAX_LENGTH,
        },
        "tasks": {},
    }
    
    for task in tasks:
        if not config.model_path(task).exists():
            print(f"\n[SKIP] {task}: model not found ({config.model_path(task)})")
            continue
        
        inputs = build_eval_inputs(task)
        val_ds = utils.PaperDataset(
            inputs["val_df"], inputs["tokenizer"],
            target_cols=inputs["target_cols"], target_type=inputs["target_type"],
            method_to_idx=inputs["method_to_idx"], max_length=config.MAX_LENGTH,
        )
        val_loader = DataLoader(val_ds, batch_size=config.effective_batch_size(), shuffle=False)
        test_loader = None
        if len(inputs["test_df"]) > 0:
            test_ds = utils.PaperDataset(
                inputs["test_df"], inputs["tokenizer"],
                target_cols=inputs["target_cols"], target_type=inputs["target_type"],
                method_to_idx=inputs["method_to_idx"], max_length=config.MAX_LENGTH,
            )
            test_loader = DataLoader(test_ds, batch_size=config.effective_batch_size(), shuffle=False)
        report = evaluate_task(
            task, device, val_loader, test_loader,
            inputs["target_type"], inputs["n_classes"],
            val_df=inputs["val_df"], test_df=inputs["test_df"],
            tokenizer=inputs["tokenizer"], target_cols=inputs["target_cols"],
            method_to_idx=inputs["method_to_idx"],
        )
        if report is not None:
            full_report["tasks"][task] = report
    
    output_path = Path(args.output) if args.output else config.eval_report_path()
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'=' * 80}")
    print(f"Full report saved: {output_path}")


if __name__ == "__main__":
    main()
