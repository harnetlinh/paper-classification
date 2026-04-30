"""
Utility functions: Asymmetric Loss, dataset class, deterministic setup.
"""
import os
import random
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import config


# ==================== Reproducibility ====================
def set_deterministic(seed: int = 42):
    """Force full determinism — bit-exact identical output across runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==================== Asymmetric Loss (Ridnik et al. 2021) ====================
class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification.
    
    Reference:
        Ridnik, T. et al. (2021). Asymmetric Loss for Multi-Label Classification.
        ICCV 2021. arXiv:2009.14119
    
    Args:
        gamma_pos: Focusing parameter for positive samples (default 0)
        gamma_neg: Focusing parameter for negative samples (default 4)
        clip: Probability margin for hard negatives (default 0.05)
        class_weight: Optional per-class weights (tensor of shape [num_classes])
    """
    def __init__(self, gamma_pos: float = 0, gamma_neg: float = 4,
                 clip: float = 0.05, class_weight=None, eps: float = 1e-8):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.class_weight = class_weight
        self.eps = eps
    
    def forward(self, logits, targets):
        """
        logits: [B, C] raw logits
        targets: [B, C] binary 0/1
        """
        # Sigmoid → probabilities
        x_sigmoid = torch.sigmoid(logits)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid
        
        # Asymmetric clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        
        # Basic CE
        los_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg
        
        # Asymmetric focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * targets
            pt1 = xs_neg * (1 - targets)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            loss *= one_sided_w
        
        # Apply class weights
        if self.class_weight is not None:
            loss *= self.class_weight.unsqueeze(0)
        
        return -loss.mean()


# ==================== Focal Cross-Entropy (single-label) ====================
class FocalCrossEntropyLoss(nn.Module):
    """Focal cross-entropy for multi-class single-label classification.

    Reference:
        Lin, T-Y. et al. (2017). Focal Loss for Dense Object Detection.
        arXiv:1708.02002

    Standard CE weights all examples equally; focal CE down-weights easy
    examples (where the model is confidently correct) so the gradient focuses
    on hard examples. Combined with class_weight, this helps very rare classes
    such as Method='Other' (~33 train samples in this project) where standard
    CE plus class weights can still leave the model under-fit on minority.

    Args:
        gamma: focusing parameter (default 2.0; gamma=0 reduces to weighted CE)
        class_weight: optional [n_classes] tensor — multiplied into the loss
                      per-sample using the target class index
    """
    def __init__(self, gamma: float = 2.0, class_weight=None, eps: float = 1e-8):
        super().__init__()
        self.gamma = gamma
        self.class_weight = class_weight
        self.eps = eps

    def forward(self, logits, targets):
        """logits: [B, C] raw, targets: [B] long class indices."""
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        log_p_t = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # [B]
        p_t = log_p_t.exp().clamp(min=self.eps, max=1.0)
        focal_weight = (1.0 - p_t) ** self.gamma
        loss = -focal_weight * log_p_t   # [B]

        if self.class_weight is not None:
            cw = self.class_weight.to(logits.device)
            sample_weight = cw.gather(0, targets)
            loss = loss * sample_weight

        return loss.mean()


# ==================== Input-text builder ====================
# Single source of truth for how a paper row gets serialised into the
# [SEP]-separated string we hand to the tokenizer. PaperDataset (training/eval)
# AND inference.py / export_review.py all call this so the text format is
# identical across train / eval / inference / review.
def build_input_texts(df, sep: str):
    """Concatenate per-paper input fields into one string each.

    Always uses Title + Abstract. When config.USE_RICH_FEATURES is True
    (default) and the dataframe has the corresponding columns, also appends
    Author Keywords / Source title / Document type — each prefixed with a
    short tag so the encoder can tell them apart from the abstract.

    Field-missing rows: the corresponding section is silently skipped; we
    never insert the literal string "None" or "nan" into the input.
    """
    use_rich = bool(getattr(config, "USE_RICH_FEATURES", False))
    has_kw = use_rich and "Author Keywords" in df.columns
    has_src = use_rich and "Source title" in df.columns
    has_dt = use_rich and "Document type" in df.columns

    texts = []
    for i in range(len(df)):
        row = df.iloc[i]
        parts = [str(row.get("Title", "")).strip(), str(row.get("Abstract", "")).strip()]
        if has_kw:
            kw = str(row.get("Author Keywords", "") or "").strip()
            if kw and kw.lower() not in ("nan", "none"):
                parts.append(f"Keywords: {kw}")
        if has_src:
            src = str(row.get("Source title", "") or "").strip()
            if src and src.lower() not in ("nan", "none"):
                parts.append(f"Journal: {src}")
        if has_dt:
            dt = str(row.get("Document type", "") or "").strip()
            if dt and dt.lower() not in ("nan", "none"):
                parts.append(f"Type: {dt}")
        texts.append(sep.join(parts))
    return texts


# ==================== Dataset ====================
class PaperDataset(Dataset):
    """
    PyTorch dataset for paper classification.
    Input format: title + [SEP] + abstract (SPECTER2 native pattern, Cohan et al. 2020)
    """
    def __init__(self, df, tokenizer, target_cols=None, target_type="multi_label",
                 method_to_idx=None, max_length=512):
        """
        df: DataFrame với cột Title, Abstract, và target columns
        tokenizer: HuggingFace tokenizer
        target_cols: list of column names cho multi-label (e.g., ['field_00', ..., 'field_11'])
        target_type: "multi_label" or "single_label"
        method_to_idx: dict mapping method string → int (cho single-label task)
        """
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_cols = target_cols
        self.target_type = target_type
        self.method_to_idx = method_to_idx
        
        # Pre-tokenize for speed
        self.encodings = self._tokenize_all()
    
    def _tokenize_all(self):
        sep = self.tokenizer.sep_token
        texts = build_input_texts(self.df, sep)
        return self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        item = {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
        }
        if self.target_type == "multi_label":
            labels = torch.tensor(
                [float(self.df.loc[idx, c]) for c in self.target_cols],
                dtype=torch.float32,
            )
            item["labels"] = labels
        elif self.target_type == "single_label":
            method = self.df.loc[idx, "method"]
            item["labels"] = torch.tensor(self.method_to_idx[method], dtype=torch.long)
        return item


# ==================== Class weights ====================
def compute_class_weights(df, target_cols, scheme: str = "inverse_frequency",
                           max_weight: float = 10.0):
    """
    Compute per-class weights for imbalanced multi-label.
    
    Args:
        df: DataFrame
        target_cols: List of binary indicator columns
        scheme: "inverse_frequency" or "sqrt_inverse"
        max_weight: Cap weight to avoid gradient explosion for very rare classes
    
    Returns: torch.Tensor of shape [num_classes], normalized to mean=1
    """
    n = len(df)
    pos_counts = np.array([df[c].sum() for c in target_cols], dtype=np.float64)
    pos_counts = np.maximum(pos_counts, 1)   # avoid div by 0
    
    if scheme == "inverse_frequency":
        weights = n / (len(target_cols) * pos_counts)
    elif scheme == "sqrt_inverse":
        weights = np.sqrt(n / pos_counts)
    else:
        weights = np.ones_like(pos_counts)
    
    # Normalize to mean=1, then clip to avoid overflow
    weights = weights / weights.mean()
    weights = np.minimum(weights, max_weight)
    # Re-normalize after clipping
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def compute_method_class_weights(df, methods, scheme: str = "inverse_frequency",
                                  max_weight: float = 10.0):
    """Single-label class weights for Method, with max clip."""
    counts = df["method"].value_counts().to_dict()
    weights = np.array([counts.get(m, 1) for m in methods], dtype=np.float64)
    weights = np.maximum(weights, 1)
    if scheme == "inverse_frequency":
        weights = 1.0 / weights
    elif scheme == "sqrt_inverse":
        weights = 1.0 / np.sqrt(weights)
    weights = weights / weights.mean()
    weights = np.minimum(weights, max_weight)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


# ==================== Threshold tuning ====================
def tune_thresholds_per_class(probs, targets, threshold_grid):
    """
    Find optimal threshold per class to maximize F1.
    
    Args:
        probs: [N, C] sigmoid probabilities
        targets: [N, C] binary 0/1
        threshold_grid: list of candidate thresholds
    
    Returns: list of optimal thresholds, list of corresponding F1
    """
    from sklearn.metrics import f1_score
    n_classes = targets.shape[1]
    best_thresholds = []
    best_f1s = []
    for c in range(n_classes):
        best_t, best_f1 = 0.5, 0.0
        # Class with zero positives in val: F1 is undefined regardless of
        # threshold. Return 0.0 so macro-F1 reflects the gap honestly.
        if targets[:, c].sum() == 0:
            best_thresholds.append(best_t)
            best_f1s.append(0.0)
            continue
        for t in threshold_grid:
            preds = (probs[:, c] >= t).astype(int)
            f1 = f1_score(targets[:, c], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        best_thresholds.append(best_t)
        best_f1s.append(best_f1)
    return best_thresholds, best_f1s


def youden_j_threshold(probs_c, targets_c):
    """Pick threshold that maximises Youden's J = TPR - FPR on the ROC curve.

    Used as a fallback when val support is too low for stable F1-grid tuning.
    Youden's J is support-robust: it does not weight rare positives
    disproportionately compared to F1 (which fails violently when support<10).
    Returns (threshold, j_value) or (0.5, 0.0) if undefined.
    """
    from sklearn.metrics import roc_curve
    targets_c = targets_c.astype(int)
    n_pos = int(targets_c.sum())
    n_neg = int(len(targets_c) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5, 0.0
    fpr, tpr, thresholds = roc_curve(targets_c, probs_c)
    j = tpr - fpr
    best_idx = j.argmax()
    t = float(thresholds[best_idx])
    # roc_curve sometimes returns infinity at threshold[0]; clamp to grid range
    if t > 1.0:
        t = 1.0
    if t < 0.0:
        t = 0.0
    return t, float(j[best_idx])


def tune_thresholds_robust(probs, targets, threshold_grid, low_support_threshold=10,
                            safe_low=0.3, safe_high=0.7, default_threshold=0.5):
    """Class-aware F1-driven threshold tuning with a safe range fallback.

    Why not Youden's J: in earlier runs Youden's J picked extreme thresholds
    (e.g. LLL=0.12, Special edu=0.99) that maximised TPR-FPR but destroyed
    F1 (recall 100% / precision 1%, or recall 0%). Our optimisation target is
    F1, so the fallback should also use F1 — just on a constrained range to
    avoid the over-fitting / spam-predictions modes on tiny val sets.

    Strategy per class:
    - 0 positives → default threshold (0.5), F1 = 0
    - support < low_support_threshold → F1-grid restricted to [safe_low, safe_high]
      (default 0.3-0.7, F1 cannot be optimised at extremes anyway)
    - support >= low_support_threshold → full F1-grid

    Returns:
        (thresholds, f1s, fallback_used)
        fallback_used: list[bool] — whether the constrained safe range was used.
    """
    from sklearn.metrics import f1_score
    n_classes = targets.shape[1]
    thresholds, f1s, fallback_used = [], [], []
    safe_grid = [t for t in threshold_grid if safe_low <= t <= safe_high]
    if not safe_grid:
        safe_grid = [default_threshold]

    for c in range(n_classes):
        n_pos = int(targets[:, c].sum())
        if n_pos == 0:
            thresholds.append(default_threshold)
            f1s.append(0.0)
            fallback_used.append(False)
            continue
        if n_pos <= low_support_threshold:
            grid = safe_grid
            used_fallback = True
        else:
            grid = threshold_grid
            used_fallback = False

        best_t, best_f1 = default_threshold, 0.0
        for t in grid:
            preds = (probs[:, c] >= t).astype(int)
            f1 = f1_score(targets[:, c], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        thresholds.append(round(float(best_t), 4))
        f1s.append(float(best_f1))
        fallback_used.append(used_fallback)
    return thresholds, f1s, fallback_used


# ==================== Calibrated weighted BCE (alternative loss) ====================
class WeightedBCEWithLogitsLoss(nn.Module):
    """BCE-with-logits using `pos_weight` per class + optional label smoothing.

    Unlike AsymmetricLoss, this keeps sigmoid outputs well-calibrated — meaning
    a single near-0.5 threshold works across most classes and threshold tuning
    does not have to compensate for a class-specific bias.

    Label smoothing (Müller et al. 2019): replace hard 0/1 targets with
    (smoothing/2, 1 - smoothing/2). Reduces over-confidence and combats the
    severe overfit observed at epoch 7+ (train_loss → 0.21 while val plateaus
    around 0.54). Typical empirical gain on small datasets: +1-3% F1 from
    smoother probability outputs that survive the val/test distribution shift
    better than perfectly-fit train predictions.

    Args:
        pos_weight: per-class [n_classes] tensor — passed straight to BCE
        class_weight: per-class [n_classes] tensor — multiplied per element
        label_smoothing: float in [0, 0.3]; 0 disables (default)
    """
    def __init__(self, pos_weight=None, class_weight=None, label_smoothing: float = 0.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.class_weight = class_weight
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits, targets):
        if self.label_smoothing > 0:
            half = self.label_smoothing / 2.0
            targets = targets * (1.0 - self.label_smoothing) + half
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=self.pos_weight.to(logits.device) if self.pos_weight is not None else None,
            reduction="none",
        )
        if self.class_weight is not None:
            loss = loss * self.class_weight.to(logits.device).unsqueeze(0)
        return loss.mean()


def compute_pos_weight(df, target_cols, max_pos_weight: float = 20.0):
    """Per-class pos_weight = (#negatives / #positives), clipped at max_pos_weight.

    Standard recipe for `BCEWithLogitsLoss(pos_weight=...)` to handle multi-label
    imbalance while keeping outputs calibrated.
    """
    n = len(df)
    pos_counts = np.array([df[c].sum() for c in target_cols], dtype=np.float64)
    pos_counts = np.maximum(pos_counts, 1.0)
    neg_counts = n - pos_counts
    neg_counts = np.maximum(neg_counts, 1.0)
    pw = neg_counts / pos_counts
    pw = np.minimum(pw, max_pos_weight)
    return torch.tensor(pw, dtype=torch.float32)
