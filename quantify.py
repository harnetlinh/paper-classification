"""
Phase 1: Quantification — prior shift adaptation for multi-label / single-label.

The core problem this module solves: a classifier trained on data with class
prior π_train is evaluated on data with a DIFFERENT prior π_test. The model's
RANKING (AUC) is unaffected by prior shift, but its CALIBRATION (where to
threshold raw probabilities) is. Per-class thresholds tuned on a val set with
prior π_val transfer poorly to test if π_test ≠ π_val.

Concrete observed evidence in this project (test 2024 vs train 2013-2022):
    Field 'psychology in education':       train 18.0%  →  test  0.5%   (-17.5%)
    Field 'Special education':             train  1.0%  →  test 12.6%   (+11.7%)
    Field 'teaching & learning':           train 25.9%  →  test 64.4%   (+38.5%)
    Field 'International education':       AUC 0.98     →  F1 0.19      (threshold off)

References:
    Saerens, Latinne, Decaestecker (2002). "Adjusting the outputs of a
        classifier to new a priori probabilities". Neural Computation 14(1).
    Forman (2008). "Quantifying counts and costs via classification". Data Mining
        and Knowledge Discovery 17(2).
    Bella, Ferri, Hernández-Orallo, Ramírez-Quintana (2010). "Quantification via
        probability estimators". ICDM.

The module exposes two estimators (PACC for direct prior estimation, Saerens-EM
for full posterior re-weighting) and one threshold-adjustment helper that
plugs into evaluate.py / inference.py without requiring re-training.
"""
from __future__ import annotations
from typing import Tuple

import numpy as np


# ==================== Probabilistic Adjusted Classify Count ====================
def pacc_prior_estimate(
    val_probs: np.ndarray,
    val_targets: np.ndarray,
    test_probs: np.ndarray,
) -> np.ndarray:
    """Estimate per-class test prior via PACC (Bella et al. 2010).

    For each class c (one-vs-rest if multi-label, one-vs-rest over softmax
    probs for single-label):

        Let p_avg_test(c) = mean over test set of p(y=c | x)
        Let TPR_val(c)   = mean of p(y=c | x) for val samples WITH y=c
        Let FPR_val(c)   = mean of p(y=c | x) for val samples WITHOUT y=c

        π_test(c) ≈ (p_avg_test(c) - FPR_val(c)) / (TPR_val(c) - FPR_val(c))

    Clipped to [0.001, 0.999] to keep downstream divisions safe.

    Args:
        val_probs:   [N_val, C] sigmoid probs (multi-label) OR softmax probs
                     (single-label) on the val set.
        val_targets: [N_val, C] binary 0/1 (multi-label) OR [N_val] int (single).
                     If 1-D, treated as single-label and one-hot encoded internally.
        test_probs:  [N_test, C] same shape as val_probs.

    Returns:
        prior_test: [C] estimated test-set prior per class, in [0.001, 0.999].
    """
    test_probs = np.asarray(test_probs, dtype=np.float64)
    val_probs = np.asarray(val_probs, dtype=np.float64)
    val_targets = np.asarray(val_targets)

    if val_targets.ndim == 1:
        n_classes = val_probs.shape[1]
        oh = np.zeros((len(val_targets), n_classes), dtype=np.int64)
        oh[np.arange(len(val_targets)), val_targets.astype(int)] = 1
        val_targets = oh

    n_classes = val_probs.shape[1]
    prior_test = np.zeros(n_classes, dtype=np.float64)

    for c in range(n_classes):
        y = val_targets[:, c].astype(bool)
        if y.sum() == 0 or (~y).sum() == 0:
            # No positive or no negative in val — TPR or FPR undefined. Fall
            # back to the empirical mean of test probs as the prior estimate.
            prior_test[c] = float(np.clip(test_probs[:, c].mean(), 0.001, 0.999))
            continue
        tpr = val_probs[y, c].mean()
        fpr = val_probs[~y, c].mean()
        denom = tpr - fpr
        if abs(denom) < 1e-6:
            # Classifier has zero discriminative power on this class.
            prior_test[c] = float(np.clip(test_probs[:, c].mean(), 0.001, 0.999))
            continue
        p_avg_test = test_probs[:, c].mean()
        est = (p_avg_test - fpr) / denom
        prior_test[c] = float(np.clip(est, 0.001, 0.999))

    return prior_test


# ==================== Saerens-Latinne-Decaestecker EM ====================
def saerens_em_prior(
    val_probs: np.ndarray,
    val_targets: np.ndarray,
    test_probs: np.ndarray,
    max_iter: int = 200,
    tol: float = 1e-5,
    target_type: str = "multi_label",
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate π_test via the Saerens et al. (2002) EM and return RE-WEIGHTED
    posterior predictions on the test set.

    The full Saerens recipe is for single-label softmax probabilities — the
    posterior on the test set is re-weighted according to the estimated prior
    shift relative to the train (here: val) prior:

        p_test(y|x) ∝ p_train(y|x) * π_test(y) / π_train(y)

    For multi-label we apply the same recipe per-class one-vs-rest, treating
    each class independently (this is the standard approximation; see Vaz et al.
    2019 for analysis).

    Args:
        val_probs:    [N_val, C] sigmoid (multi) or softmax (single) probs on val
        val_targets:  [N_val, C] (multi) or [N_val] int (single)
        test_probs:   [N_test, C]
        max_iter:     EM iteration cap
        tol:          stop when ||π_new - π_old||_∞ < tol
        target_type:  "multi_label" or "single_label"

    Returns:
        prior_test:    [C] estimated test prior
        adjusted_test: [N_test, C] re-weighted test probabilities
                       (same shape as test_probs; rows sum to 1 for single_label,
                       per-class independent for multi_label).
    """
    val_probs = np.asarray(val_probs, dtype=np.float64)
    test_probs = np.asarray(test_probs, dtype=np.float64)
    val_targets = np.asarray(val_targets)

    # Saerens EM is known to diverge under extreme prior shifts (a class going
    # from 1% to 30% prevalence drives the iterative re-weighting toward the
    # extreme of "everything is positive"). Bootstrap from a PACC estimate
    # instead of the train prior — PACC is closed-form, doesn't iterate, and
    # gives a much closer starting point for Saerens to refine.
    pacc_init = pacc_prior_estimate(val_probs, val_targets, test_probs)

    if target_type == "multi_label":
        # One-vs-rest per class — each class is a Bernoulli(π_c).
        n_classes = val_probs.shape[1]
        prior_test = np.zeros(n_classes, dtype=np.float64)
        adjusted = np.zeros_like(test_probs, dtype=np.float64)
        for c in range(n_classes):
            y = val_targets[:, c].astype(int)
            pi_train = max(float(y.mean()), 1e-4)
            # EM in 2-class form. p_train(y=1|x) is provided directly by val/test_probs.
            pi_t = float(pacc_init[c])
            for _ in range(max_iter):
                # P_test(y=1|x) ∝ p_train(y=1|x) * (pi_t / pi_train)
                # P_test(y=0|x) ∝ (1 - p_train(y=1|x)) * ((1 - pi_t) / (1 - pi_train))
                w_pos = pi_t / pi_train
                w_neg = (1.0 - pi_t) / max(1.0 - pi_train, 1e-4)
                p1 = test_probs[:, c] * w_pos
                p0 = (1.0 - test_probs[:, c]) * w_neg
                denom = p1 + p0 + 1e-12
                p_test_y1 = p1 / denom
                pi_new = float(p_test_y1.mean())
                if abs(pi_new - pi_t) < tol:
                    pi_t = pi_new
                    break
                pi_t = pi_new
            prior_test[c] = float(np.clip(pi_t, 0.001, 0.999))
            # Re-compute adjusted at converged prior
            w_pos = pi_t / pi_train
            w_neg = (1.0 - pi_t) / max(1.0 - pi_train, 1e-4)
            p1 = test_probs[:, c] * w_pos
            p0 = (1.0 - test_probs[:, c]) * w_neg
            adjusted[:, c] = p1 / (p1 + p0 + 1e-12)
        return prior_test, adjusted

    # Single-label: full softmax EM.
    if val_targets.ndim != 1:
        val_targets = val_targets.argmax(axis=1)
    n_classes = val_probs.shape[1]
    counts = np.bincount(val_targets.astype(int), minlength=n_classes).astype(np.float64)
    pi_train = np.clip(counts / max(counts.sum(), 1.0), 1e-4, 1.0 - 1e-4)
    # Bootstrap from PACC for stability under large prior shifts (see comment above).
    pi_t = pacc_init.copy()
    for _ in range(max_iter):
        w = pi_t / pi_train  # [C]
        scaled = test_probs * w[None, :]
        denom = scaled.sum(axis=1, keepdims=True) + 1e-12
        post = scaled / denom  # [N_test, C]
        pi_new = post.mean(axis=0)
        if np.max(np.abs(pi_new - pi_t)) < tol:
            pi_t = pi_new
            break
        pi_t = pi_new
    prior_test = np.clip(pi_t, 0.001, 0.999)
    # Final re-weight at converged prior
    w = prior_test / pi_train
    scaled = test_probs * w[None, :]
    adjusted = scaled / (scaled.sum(axis=1, keepdims=True) + 1e-12)
    return prior_test, adjusted


# ==================== Threshold adjustment via prior shift ====================
def adjust_thresholds_for_prior_shift(
    val_thresholds: np.ndarray,
    val_prior: np.ndarray,
    test_prior: np.ndarray,
    method: str = "log_ratio",
) -> np.ndarray:
    """Shift per-class thresholds tuned on val to account for the prior shift
    from val to test, WITHOUT requiring re-training.

    Two methods supported:

    "log_ratio" (default):
        threshold_test = sigmoid(logit(threshold_val) + log(π_train(c) / π_test(c)))
        Equivalent to recalibrating the classifier's decision boundary by the
        estimated prior shift in log-odds space. Symmetric: a class whose
        prevalence DOUBLES gets its threshold pushed DOWN by log(2).

    "linear":
        threshold_test = threshold_val * (π_train(c) / π_test(c))
        Crude linear scaling, clipped to [0.05, 0.95]. Less principled but
        sometimes more stable when the classifier is well-calibrated.

    Args:
        val_thresholds: [C] thresholds tuned on val (e.g., from
                        utils.tune_thresholds_robust)
        val_prior:      [C] empirical class prevalence on val
        test_prior:     [C] estimated class prevalence on test (from PACC or
                        saerens_em_prior)
        method:         "log_ratio" or "linear"

    Returns:
        adjusted: [C] new thresholds
    """
    val_prior = np.clip(np.asarray(val_prior, dtype=np.float64), 1e-4, 1.0 - 1e-4)
    test_prior = np.clip(np.asarray(test_prior, dtype=np.float64), 1e-4, 1.0 - 1e-4)
    val_thresholds = np.asarray(val_thresholds, dtype=np.float64)

    if method == "linear":
        scaled = val_thresholds * (val_prior / test_prior)
        return np.clip(scaled, 0.05, 0.95)

    # log_ratio (preferred)
    safe_t = np.clip(val_thresholds, 1e-4, 1.0 - 1e-4)
    logit_t = np.log(safe_t / (1.0 - safe_t))
    log_ratio = np.log(val_prior / test_prior)
    new_logit = logit_t + log_ratio
    return 1.0 / (1.0 + np.exp(-new_logit))


# ==================== Convenience wrapper ====================
def quantified_thresholds(
    val_probs: np.ndarray,
    val_targets: np.ndarray,
    test_probs: np.ndarray,
    val_thresholds: np.ndarray,
    target_type: str = "multi_label",
    estimator: str = "pacc",
) -> Tuple[np.ndarray, np.ndarray]:
    """One-shot helper: estimate test prior + return adjusted per-class thresholds.

    Plug-and-play replacement for using val_thresholds directly on test data.

    Args:
        val_probs, val_targets, test_probs: as in saerens_em_prior
        val_thresholds: [C] from utils.tune_thresholds_robust (val-tuned)
        target_type:    "multi_label" (default) or "single_label" (note: test_thresholds
                        not meaningful for single-label argmax — returns val_priors only)
        estimator:      "pacc" (default, closed-form, robust under extreme shift)
                        or "saerens" (iterative; can diverge when shift is >5x).

    Returns:
        test_prior:        [C] estimated test prior
        adjusted_thresholds: [C] thresholds shifted via log-odds for the new prior

    Why PACC by default:
        Saerens EM is mathematically elegant but its iterative re-weighting
        amplifies any initial bias when the prior shift is severe. Concretely:
        if val prior = 0.002 and test prior = 0.126, Saerens converges to ~0.99
        (everything classified positive). PACC's closed-form estimate is
        empirically much closer to the true test prior under this regime —
        observed L1 error 0.05 vs Saerens 0.87 in synthetic tests matching the
        psychology / Special edu drift in this project's data.
    """
    val_targets_arr = np.asarray(val_targets)
    if target_type == "multi_label":
        if val_targets_arr.ndim == 1:
            raise ValueError("multi_label target must be 2-D")
        val_prior = val_targets_arr.mean(axis=0)
    else:
        n_classes = val_probs.shape[1]
        if val_targets_arr.ndim > 1:
            val_targets_arr = val_targets_arr.argmax(axis=1)
        counts = np.bincount(val_targets_arr.astype(int), minlength=n_classes)
        val_prior = counts.astype(np.float64) / max(len(val_targets_arr), 1)

    if estimator == "pacc":
        test_prior = pacc_prior_estimate(val_probs, val_targets_arr, test_probs)
    else:
        test_prior, _ = saerens_em_prior(
            val_probs, val_targets_arr, test_probs, target_type=target_type,
        )

    adjusted = adjust_thresholds_for_prior_shift(
        val_thresholds, val_prior, test_prior, method="log_ratio",
    )
    return test_prior, adjusted
