"""
Phase A: ensemble GPT-5 panel predictions with SPECTER2 ensemble at inference.

The SPECTER2 model gives per-class sigmoid (multi-label) or softmax (single-label)
probabilities. The GPT-5 panel gives per-class discrete probabilities in
{0, 0.33, 0.67, 1.0} (3 binary votes / 3). We linearly combine them per class:

    p_final[c] = lambda[c] * p_specter[c] + (1 - lambda[c]) * p_gpt5[c]

with lambda[c] tuned on val to maximize per-class F1 (multi-label) or to
maximize macro-F1 single-label task. Per-class lambda lets the model lean
heavier on whichever signal is stronger for each class — typically GPT-5
wins for rare classes (Special edu) where SPECTER2 underfits, and SPECTER2
wins for high-data classes (English Education) where it has high AUC.
"""
from __future__ import annotations
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import config


# ==================== Alignment ====================
def align_gpt5_to_df(
    df: pd.DataFrame,
    panel_paper_ids: np.ndarray,
    panel_probs: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """Reorder panel predictions to match df's row order via Total_ID.

    Why this helper exists: PaperDataset preserves df row order but the
    GPT-5 panel parquet is stored sorted by Total_ID. The two won't match
    out of the box. This re-indexes panel_probs into df's order, filling
    any df rows missing a panel prediction with neutral 0.5 (so they
    contribute nothing to a 50/50 ensemble).

    Args:
        df: the val or test DataFrame (must have Total_ID column)
        panel_paper_ids: [M] Total_IDs from the panel parquet (sorted asc)
        panel_probs: [M, n_classes] panel probabilities in same order
        n_classes: expected output dim

    Returns:
        aligned: [len(df), n_classes] reordered to df's row order
    """
    if "Total_ID" not in df.columns:
        raise ValueError("df must have Total_ID column")
    df_ids = df["Total_ID"].astype(int).to_numpy()
    panel_lookup = {int(pid): i for i, pid in enumerate(panel_paper_ids)}

    aligned = np.full((len(df), n_classes), 0.5, dtype=np.float64)
    n_matched = 0
    for row_i, df_id in enumerate(df_ids):
        if df_id in panel_lookup:
            aligned[row_i] = panel_probs[panel_lookup[df_id]]
            n_matched += 1
    return aligned, n_matched


# ==================== Lambda tuning ====================
def tune_ensemble_lambda_multilabel(
    val_specter: np.ndarray,
    val_gpt5: np.ndarray,
    val_targets: np.ndarray,
    val_thresholds: np.ndarray,
    lambda_grid: np.ndarray = None,
) -> np.ndarray:
    """Find per-class lambda maximizing val F1 at the same per-class thresholds.

    Args:
        val_specter: [N_val, C] sigmoid probs from SPECTER2 ensemble
        val_gpt5:    [N_val, C] panel probs (in {0, 0.33, 0.67, 1.0})
        val_targets: [N_val, C] binary 0/1
        val_thresholds: [C] tuned per-class thresholds (from utils.tune_thresholds_robust)
        lambda_grid: candidate lambda values; default 0.0..1.0 step 0.05

    Returns:
        best_lambdas: [C] one lambda per class

    A lambda of 1.0 means "use SPECTER2 only", 0.0 means "use GPT-5 only".
    Per-class tuning lets each class pick the right blend.
    """
    if lambda_grid is None:
        lambda_grid = np.linspace(0.0, 1.0, 21)
    n_classes = val_specter.shape[1]
    best_lambdas = np.full(n_classes, 0.5, dtype=np.float64)
    for c in range(n_classes):
        if int(val_targets[:, c].sum()) == 0:
            # No positives in val — F1 undefined regardless of lambda. Default 0.5.
            continue
        best_f1 = -1.0
        best_lam = 0.5
        for lam in lambda_grid:
            blend = lam * val_specter[:, c] + (1.0 - lam) * val_gpt5[:, c]
            preds = (blend >= val_thresholds[c]).astype(int)
            f1 = f1_score(val_targets[:, c], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_lam = lam
        best_lambdas[c] = best_lam
    return best_lambdas


def tune_ensemble_lambda_singlelabel(
    val_specter: np.ndarray,
    val_gpt5: np.ndarray,
    val_targets: np.ndarray,
    lambda_grid: np.ndarray = None,
) -> np.ndarray:
    """Find per-class lambda maximizing per-class one-vs-rest F1 for single-label.

    Single-label final prediction is argmax over the blended distribution.
    For tuning, we still optimize per-class: grid lambda values, blend, take
    argmax for the full prediction, compute F1 on the focal class only.

    Returns: [C] one lambda per class.
    """
    if lambda_grid is None:
        lambda_grid = np.linspace(0.0, 1.0, 21)
    n_classes = val_specter.shape[1]
    val_targets_int = val_targets.astype(int)
    best_lambdas = np.full(n_classes, 0.5, dtype=np.float64)

    # For single-label, lambda must be SHARED across classes (otherwise the
    # blend is inconsistent — each class would weight differently and the
    # argmax becomes meaningless). We pick the single lambda maximizing
    # macro-F1 instead of per-class.
    best_f1 = -1.0
    best_lam = 0.5
    for lam in lambda_grid:
        blend = lam * val_specter + (1.0 - lam) * val_gpt5
        preds = blend.argmax(axis=1)
        f1 = f1_score(val_targets_int, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_lam = lam
    best_lambdas[:] = best_lam
    return best_lambdas


# ==================== Ensemble application ====================
def apply_ensemble(
    specter_probs: np.ndarray,
    gpt5_probs: np.ndarray,
    lambdas: np.ndarray,
) -> np.ndarray:
    """Compute p_final[c] = lambda[c] * p_specter[c] + (1-lambda[c]) * p_gpt5[c]."""
    if specter_probs.shape != gpt5_probs.shape:
        raise ValueError(
            f"shape mismatch: specter {specter_probs.shape} vs gpt5 {gpt5_probs.shape}"
        )
    return lambdas[None, :] * specter_probs + (1.0 - lambdas[None, :]) * gpt5_probs


def tune_ensemble_weights_3way_multilabel(
    val_specter: np.ndarray,
    val_gpt5: np.ndarray,
    val_knn: np.ndarray,
    val_targets: np.ndarray,
    val_thresholds: np.ndarray,
    weight_grid: np.ndarray = None,
) -> np.ndarray:
    """Per-class 3-way blend: maximizes per-class F1 over a Dirichlet-like grid.

    For each class, sweeps (w_specter, w_gpt5, w_knn) tuples summing to 1
    and picks the tuple that maximizes F1 at the given threshold. Default
    grid uses 6 values per dim → 21 valid (sum=1) tuples per class.

    Returns:
        weights: [C, 3] per-class triples (specter, gpt5, knn).
    """
    if weight_grid is None:
        weight_grid = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    triples = [
        (a, b, c)
        for a in weight_grid for b in weight_grid for c in weight_grid
        if abs(a + b + c - 1.0) < 1e-6
    ]
    n_classes = val_specter.shape[1]
    best = np.zeros((n_classes, 3), dtype=np.float64)
    for k in range(n_classes):
        if int(val_targets[:, k].sum()) == 0:
            best[k] = (1.0, 0.0, 0.0)
            continue
        best_f1 = -1.0
        best_t = (1.0, 0.0, 0.0)
        for a, b, c in triples:
            blend = a * val_specter[:, k] + b * val_gpt5[:, k] + c * val_knn[:, k]
            preds = (blend >= val_thresholds[k]).astype(int)
            f1 = f1_score(val_targets[:, k], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = (a, b, c)
        best[k] = best_t
    return best


def apply_ensemble_3way(
    specter_probs: np.ndarray,
    gpt5_probs: np.ndarray,
    knn_probs: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Compute p_final[c] = w_specter[c]*p_s[c] + w_gpt5[c]*p_g[c] + w_knn[c]*p_k[c]."""
    if not (specter_probs.shape == gpt5_probs.shape == knn_probs.shape):
        raise ValueError("All three prob matrices must have the same shape")
    return (
        weights[None, :, 0] * specter_probs
        + weights[None, :, 1] * gpt5_probs
        + weights[None, :, 2] * knn_probs
    )


# ==================== One-shot helper for evaluate.py ====================
def build_ensemble_probs(
    val_specter: np.ndarray,
    val_gpt5: np.ndarray,
    val_targets: np.ndarray,
    test_specter: np.ndarray,
    test_gpt5: np.ndarray,
    target_type: str,
    val_thresholds: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tune lambda on val, apply to both val and test, return ensembled probs.

    Returns:
        (val_blend, test_blend, lambdas)
    """
    if target_type == "multi_label":
        if val_thresholds is None:
            val_thresholds = np.full(val_specter.shape[1], 0.5)
        lambdas = tune_ensemble_lambda_multilabel(
            val_specter, val_gpt5, val_targets, val_thresholds,
        )
    else:
        lambdas = tune_ensemble_lambda_singlelabel(
            val_specter, val_gpt5, val_targets,
        )
    val_blend = apply_ensemble(val_specter, val_gpt5, lambdas)
    test_blend = apply_ensemble(test_specter, test_gpt5, lambdas)
    return val_blend, test_blend, lambdas
