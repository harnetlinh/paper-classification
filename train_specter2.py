"""
Phase 2: Train SPECTER2 deterministic
======================================

Train 3 separate models on Title+[SEP]+Abstract input:
- Fields:  multi-label, 12 classes, Asymmetric Loss + class weights
- Levels:  multi-label, 6 classes, Asymmetric Loss + class weights
- Method:  single-label, 5 classes, weighted Cross-Entropy

Architecture: SPECTER2 base (AutoModel) + custom classification head
- Pure transformers (no `adapters` library dependency for robustness)
- Title + [SEP] + Abstract concatenation (Cohan et al. 2020 SPECTER pattern)
- max_length=512 with truncation_side="right" (preserves Title)

CPU/GPU auto-detect:
- If CUDA available → train on GPU (~1-2h on T4)
- Otherwise → train on CPU (slow, ~10h for 5 epochs full data)
- Use --smoke to test pipeline with 1 epoch + small batch on CPU

Usage:
    python train_specter2.py --task fields
    python train_specter2.py --task all
    python train_specter2.py --task fields --smoke    # 1-epoch smoke test
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import f1_score
from tqdm import tqdm

import config
import utils

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


# ==================== Data Loading ====================
def load_train_val_test(task: str):
    """
    Load gold + main_2024 datasets, split temporal.
    
    Splits:
        Train: gold 2013-2022
        Val:   gold 2023
        Test:  main_2024_clean (sanitized 2024 labels)
    """
    if not config.GOLD_PARQUET.exists():
        raise FileNotFoundError(
            f"Gold dataset not found at {config.GOLD_PARQUET}. Run sanitize.py first."
        )
    gold = pd.read_parquet(config.GOLD_PARQUET)
    
    if config.MAIN_2024_PARQUET.exists():
        df_2024 = pd.read_parquet(config.MAIN_2024_PARQUET)
    else:
        print(f"WARNING: 2024 dataset not found. Test set will be empty.")
        df_2024 = pd.DataFrame()
    
    # Augmentation: LLM-augmented labels are applied ONLY to TRAIN-year rows
    # so val 2023 stays gold-truth and val_f1 measures real generalization,
    # not LLM agreement. Test 2024 is sanitized separately and never touched.
    # Augmentation is purely additive (label is added to the existing list);
    # this is why we restrict augmentation to multi-label tasks (fields, levels)
    # — single-label Method would require destructive label replacement.
    augment_specs = []
    if task == "fields" and config.SPECIAL_EDU_AUGMENT.exists():
        sp_idx = config.FIELDS_12.index("Special education")
        augment_specs.append({
            "label_value": "Special education",
            "list_col": "fields_list",
            "binary_col": f"field_{sp_idx:02d}",
            "aug_path": config.SPECIAL_EDU_AUGMENT,
        })
    if task == "levels":
        for level in ["ECE", "TVET", "LLL"]:
            aug_path = config.LEVEL_AUGMENT_OUTPUTS.get(level)
            if aug_path is not None and aug_path.exists():
                augment_specs.append({
                    "label_value": level,
                    "list_col": "levels_list",
                    "binary_col": f"level_{level}",
                    "aug_path": aug_path,
                })

    train_year_set = set(config.TRAIN_YEARS)
    for spec in augment_specs:
        aug = pd.read_parquet(spec["aug_path"])
        aug_yes = aug[aug["agreement"] == "unanimous_yes"]
        if len(aug_yes) == 0:
            continue
        aug_ids = set(aug_yes["Total_ID"].astype(float).tolist())
        train_mask_for_aug = gold["Year"].isin(train_year_set) & gold["Total_ID"].isin(aug_ids)
        n_train_aug = int(train_mask_for_aug.sum())
        n_skipped = len(aug_yes) - n_train_aug
        print(f"Augmenting {spec['label_value']}: +{n_train_aug} papers (train only)"
              f"{f', {n_skipped} skipped (val/test years)' if n_skipped else ''}")
        for idx in gold[train_mask_for_aug].index:
            current = gold.at[idx, spec["list_col"]]
            if not isinstance(current, list):
                current = list(current) if current is not None else []
            if spec["label_value"] not in current:
                gold.at[idx, spec["list_col"]] = current + [spec["label_value"]]
                gold.at[idx, spec["binary_col"]] = True

    train_df = gold[gold["Year"].isin(config.TRAIN_YEARS)].copy().reset_index(drop=True)
    val_df = gold[gold["Year"] == config.VAL_YEAR].copy().reset_index(drop=True)
    test_df = (df_2024.copy().reset_index(drop=True)
               if len(df_2024) > 0 else pd.DataFrame())

    return train_df, val_df, test_df


def load_train_val_test_with_optional_join(task: str, include_val_in_train: bool):
    """As load_train_val_test, but optionally fold val into train (Phase F).

    When include_val_in_train=True, train_df = train ∪ val and val_df is left
    intact (used only for in-training monitoring, not for best-model selection
    in this mode — caller must skip early stopping).

    Why keep val visible at all when folded in: the per-epoch val metrics still
    provide a sanity-check that the model isn't catastrophically diverging.
    Threshold tuning in this mode reuses the existing thresholds_*.json from
    a prior train→val run (caller responsibility).
    """
    train_df, val_df, test_df = load_train_val_test(task)
    if include_val_in_train and len(val_df) > 0:
        train_df = pd.concat([train_df, val_df], ignore_index=True)
        # Reset index so PaperDataset's df.iloc[] still works.
        train_df = train_df.reset_index(drop=True)
    return train_df, val_df, test_df


def get_target_cols(task: str) -> list:
    if task == "fields":
        return [f"field_{i:02d}" for i in range(len(config.FIELDS_12))]
    elif task == "levels":
        return [f"level_{l}" for l in config.LEVELS_6]
    return None


def get_class_names(task: str) -> list:
    if task == "fields":
        return config.FIELDS_12
    elif task == "levels":
        return config.LEVELS_6
    elif task == "method":
        return config.METHODS_5
    raise ValueError(f"Unknown task: {task}")


# ==================== Model ====================
class SpecterClassifier(nn.Module):
    """
    SPECTER2 base + classification head.
    
    Architecture:
        AutoModel(specter2_base) → [CLS] embedding (768 dims)
                                 → Dropout(0.1)
                                 → Linear(768, n_classes)
    
    For multi-label tasks: returns raw logits (apply sigmoid externally for probs).
    For single-label tasks: returns raw logits (apply softmax/argmax externally).
    """
    def __init__(self, base_model_name: str, n_classes: int, dropout: float = 0.1,
                 revision: str = None):
        super().__init__()
        from transformers import AutoModel
        kwargs = {}
        if revision is not None:
            kwargs["revision"] = revision
        self.encoder = AutoModel.from_pretrained(base_model_name, **kwargs)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, n_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # SPECTER2 uses [CLS] token at position 0 as document embedding
        cls_emb = outputs.last_hidden_state[:, 0, :]
        cls_emb = self.dropout(cls_emb)
        logits = self.classifier(cls_emb)
        return logits


def _model_save_path(task: str, seed: int, include_val: bool = False):
    """Save path for a model. seed=config.SEED uses the legacy single path
    (`model_{task}.pt`); other seeds use `model_{task}_s{seed}.pt`. This keeps
    a single-model run drop-in compatible with existing evaluate/inference,
    while ensemble runs produce sibling files that the loaders can pick up.

    include_val=True (Phase F) appends `_trainval` to the filename so the
    train+val refit is saved alongside the original train-only checkpoint
    rather than overwriting it.
    """
    suffix = "_trainval" if include_val else ""
    if seed == config.SEED and not include_val:
        return config.model_path(task)
    if seed == config.SEED:
        return config.OUTPUT_DIR / f"model_{task}{suffix}.pt"
    return config.OUTPUT_DIR / f"model_{task}_s{seed}{suffix}.pt"


# ==================== Training ====================
def train_model(task: str, smoke: bool = False, seed: int = None,
                include_val_in_train: bool = False):
    """Train one model for the given task.

    Args:
        include_val_in_train: Phase F flag. When True, train on train ∪ val.
            Caller must reuse thresholds from a prior train→val run because
            we no longer have a held-out val for threshold tuning. Best-model
            selection uses fixed-epoch (config.EPOCHS) instead of val-monitor
            early stopping.
    """
    if seed is None:
        seed = config.SEED
    is_ensemble_seed = seed != config.SEED
    print("=" * 80)
    suffix = f"  seed={seed}" + ("  [ENSEMBLE]" if is_ensemble_seed else "")
    if include_val_in_train:
        suffix += "  [TRAIN+VAL]"
    print(f"Training task: {task}{suffix}" + (" [SMOKE TEST MODE]" if smoke else ""))
    print("=" * 80)

    # Deterministic setup
    utils.set_deterministic(seed)

    # Device autodetect
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cpu":
        print("WARNING: Training on CPU will be VERY slow (~10h for 5 epochs full data)")
        print("         Recommended: use Colab GPU or local CUDA-enabled GPU")

    # Load data
    train_df, val_df, test_df = load_train_val_test_with_optional_join(
        task, include_val_in_train=include_val_in_train,
    )
    print(f"Data sizes — Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    if include_val_in_train:
        print("  [Phase F] Val 2023 has been folded into train. Val metrics during training "
              "are no longer held-out — they reflect train fit, not generalization. "
              "Threshold tuning will be SKIPPED; reuse thresholds from prior train→val run.")
    
    # Smoke test: subsample
    if smoke:
        train_df = train_df.head(50).copy().reset_index(drop=True)
        val_df = val_df.head(20).copy().reset_index(drop=True)
        print(f"[SMOKE] Subsampled to: Train {len(train_df)} | Val {len(val_df)}")
    
    # Task setup
    if task in ("fields", "levels"):
        target_cols = get_target_cols(task)
        n_classes = len(target_cols)
        target_type = "multi_label"
    else:
        target_cols = None
        n_classes = len(config.METHODS_5)
        target_type = "single_label"
    
    print(f"Task type: {target_type}, n_classes: {n_classes}")

    # M1: load TAPT-adapted encoder if available, else fall back to HF Hub
    # pretrained weights. TAPT_OUTPUT_DIR is produced by tapt.py and contains
    # both the SPECTER2 encoder weights (MLM-continued) and the matching
    # tokenizer (Guard #1) — same AutoModel API works either way.
    encoder_path = config.BACKBONE_MODEL
    if bool(getattr(config, "USE_TAPT", False)):
        tapt_dir = getattr(config, "TAPT_OUTPUT_DIR", None)
        if tapt_dir is not None and Path(tapt_dir).exists():
            encoder_path = str(tapt_dir)
            print(f"  Loading TAPT-adapted encoder from {encoder_path}")
        else:
            print(f"  WARNING: USE_TAPT=True but {tapt_dir} does not exist — "
                  f"falling back to HF Hub backbone ({config.BACKBONE_MODEL}). "
                  f"Run `python tapt.py` first to materialize the adapted encoder.")

    # Tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(encoder_path)
    
    # Datasets
    method_to_idx = {m: i for i, m in enumerate(config.METHODS_5)}
    
    train_ds = utils.PaperDataset(
        train_df, tokenizer,
        target_cols=target_cols,
        target_type=target_type,
        method_to_idx=method_to_idx,
        max_length=config.MAX_LENGTH,
    )
    val_ds = utils.PaperDataset(
        val_df, tokenizer,
        target_cols=target_cols,
        target_type=target_type,
        method_to_idx=method_to_idx,
        max_length=config.MAX_LENGTH,
    )
    
    batch_size = 4 if smoke else config.effective_batch_size()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(config.SEED),
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    # Model — same encoder_path as tokenizer above (TAPT-adapted when present).
    # When encoder_path == TAPT_OUTPUT_DIR, BACKBONE_REVISION is irrelevant
    # because the local path has no revision concept; pass None defensively.
    is_tapt_load = encoder_path != config.BACKBONE_MODEL
    print(f"Loading {'TAPT-adapted' if is_tapt_load else 'SPECTER2 base'} "
          f"model ({encoder_path})...")
    model = SpecterClassifier(
        encoder_path, n_classes=n_classes,
        dropout=getattr(config, "DROPOUT", 0.1),
        revision=None if is_tapt_load else getattr(config, "BACKBONE_REVISION", None),
    ).to(device)

    # Print the rich-features flag prominently — eval/inference need to be run
    # with the SAME flag value as training, otherwise the model sees a different
    # input format and predictions degrade silently.
    rich = bool(getattr(config, "USE_RICH_FEATURES", False))
    print(f"  Rich features (Author Keywords + Source title + Document type): "
          f"{'ON' if rich else 'OFF'} — eval/inference must match this flag")

    # Loss
    if target_type == "multi_label":
        class_weights = utils.compute_class_weights(train_df, target_cols).to(device)
        loss_choice = getattr(config, "MULTILABEL_LOSS", "asymmetric")
        if loss_choice == "bce_pos_weight":
            pos_weight = utils.compute_pos_weight(train_df, target_cols).to(device)
            ls = float(getattr(config, "LABEL_SMOOTHING", 0.0))
            loss_fn = utils.WeightedBCEWithLogitsLoss(
                pos_weight=pos_weight,
                class_weight=class_weights,
                label_smoothing=ls,
            )
            ls_tag = f", label_smoothing={ls}" if ls > 0 else ""
            print(f"  Loss: WeightedBCEWithLogits (pos_weight + class_weight{ls_tag}) — calibrated outputs")
        else:
            loss_fn = utils.AsymmetricLoss(
                gamma_pos=config.ASYMMETRIC_LOSS["gamma_pos"],
                gamma_neg=config.ASYMMETRIC_LOSS["gamma_neg"],
                clip=config.ASYMMETRIC_LOSS["clip"],
                class_weight=class_weights,
            )
            print(f"  Loss: AsymmetricLoss (gamma_neg={config.ASYMMETRIC_LOSS['gamma_neg']}) + class_weight")
    else:
        class_weights = utils.compute_method_class_weights(
            train_df, config.METHODS_5
        ).to(device)
        if getattr(config, "USE_FOCAL_LOSS_METHOD", False):
            loss_fn = utils.FocalCrossEntropyLoss(
                gamma=config.FOCAL_GAMMA,
                class_weight=class_weights,
            )
            print(f"  Loss: FocalCrossEntropy (gamma={config.FOCAL_GAMMA}) + class weights")
        else:
            loss_fn = nn.CrossEntropyLoss(weight=class_weights)
            print(f"  Loss: CrossEntropy + class weights")
    
    # Optimizer + scheduler
    optimizer = AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    n_epochs = 1 if smoke else config.EPOCHS
    total_steps = max(1, len(train_loader) * n_epochs)
    warmup_steps = max(1, int(total_steps * config.WARMUP_RATIO))
    
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))
    
    scheduler = LambdaLR(optimizer, lr_lambda)

    # Mixed precision (AMP): big throughput win on T4/V100/A100. No-op on CPU.
    use_amp = bool(getattr(config, "USE_AMP", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        print(f"  AMP mixed precision: enabled (fp16)")

    # Training loop
    best_val_metric = -1.0
    no_improve = 0
    log = []
    best_metric_name = getattr(config, "BEST_MODEL_METRIC", "macro_auc")

    for epoch in range(n_epochs):
        model.train()
        train_loss_sum = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(input_ids, attention_mask)
                    loss = loss_fn(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(input_ids, attention_mask)
                loss = loss_fn(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()

            train_loss_sum += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss_sum / max(1, n_batches)
        val_metrics = evaluate(model, val_loader, target_type, n_classes, device)
        val_f1 = val_metrics["macro_f1"]
        val_auc = val_metrics.get("macro_auc", float("nan"))

        # For multi-label tasks, also compute the tuned-F1 estimate this
        # epoch — that is what threshold tuning will produce at the end of
        # training, so it's the right metric to drive best-model selection.
        val_tuned_f1 = float("nan")
        if target_type == "multi_label":
            val_probs_ep, val_targets_ep = predict_probs(model, val_loader, device, target_type)
            low_support = getattr(config, "LOW_SUPPORT_THRESHOLD_FALLBACK", 10)
            _, f1s_ep, _ = utils.tune_thresholds_robust(
                val_probs_ep, val_targets_ep, config.THRESHOLD_GRID,
                low_support_threshold=low_support,
            )
            val_tuned_f1 = float(np.mean(f1s_ep))
            val_metrics["macro_f1_tuned"] = val_tuned_f1

        # Pick which metric drives best-model selection
        if best_metric_name == "tuned_macro_f1" and target_type == "multi_label":
            selected_metric = val_tuned_f1
        elif best_metric_name == "macro_auc":
            selected_metric = val_auc
        else:
            selected_metric = val_f1
        if not np.isfinite(selected_metric):
            selected_metric = val_f1   # fallback if NaN (e.g. all-empty class)

        epoch_log = {"epoch": epoch + 1, "train_loss": avg_train_loss, **val_metrics}
        log.append(epoch_log)
        if target_type == "multi_label":
            print(f"  Epoch {epoch+1}: loss={avg_train_loss:.4f}  "
                  f"f1@0.5={val_f1:.4f}  tuned_f1={val_tuned_f1:.4f}  auc={val_auc:.4f}")
        else:
            print(f"  Epoch {epoch+1}: loss={avg_train_loss:.4f}  "
                  f"f1={val_f1:.4f}  auc={val_auc:.4f}")

        is_first_epoch = epoch == 0
        improved = selected_metric > best_val_metric
        save_path = _model_save_path(task, seed, include_val=include_val_in_train)
        # Phase F: when val is folded into train, val metric is leaked, so
        # it's not a generalization signal. Save EVERY epoch (overwriting),
        # use config.EPOCHS as the fixed budget, no early stop.
        if include_val_in_train:
            torch.save(model.state_dict(), save_path)
            print(f"    Saved (epoch {epoch+1}/{n_epochs}, train+val mode — fixed budget)")
            continue
        if improved or is_first_epoch:
            if improved:
                best_val_metric = selected_metric
                no_improve = 0
            torch.save(model.state_dict(), save_path)
            tag = f"★ Saved best ({best_metric_name}={selected_metric:.4f})" if improved \
                else f"Saved baseline checkpoint ({best_metric_name}={selected_metric:.4f})"
            print(f"    {tag}")
        else:
            no_improve += 1
            if no_improve >= config.EARLY_STOPPING_PATIENCE and not smoke:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Reload best
    save_path = _model_save_path(task, seed, include_val=include_val_in_train)
    if save_path.exists():
        model.load_state_dict(torch.load(save_path, map_location=device))

    # Per-class threshold tuning (multi-label only) — only for the canonical
    # seed run; ensemble-seed runs leave threshold tuning to the ensemble
    # combiner (which averages probabilities across seeds first, then tunes).
    # Phase F (include_val_in_train): val is no longer held-out, so tuning
    # thresholds on it would be circular. Reuse existing thresholds_*.json
    # from the prior train→val run.
    if target_type == "multi_label" and not smoke and not is_ensemble_seed and not include_val_in_train:
        print("\n--- Tuning per-class thresholds (robust: F1-grid in safe range) ---")
        val_probs, val_targets = predict_probs(model, val_loader, device, target_type)
        low_support = getattr(config, "LOW_SUPPORT_THRESHOLD_FALLBACK", 10)
        thresholds, f1s, fallback_used = utils.tune_thresholds_robust(
            val_probs, val_targets, config.THRESHOLD_GRID,
            low_support_threshold=low_support,
        )
        class_names = get_class_names(task)
        for i, cn in enumerate(class_names):
            if fallback_used[i]:
                print(f"    [{cn}] used safe range [0.3, 0.7] (val support too low) → "
                      f"threshold={thresholds[i]:.3f}")
        threshold_data = {
            "thresholds": [float(t) for t in thresholds],
            "per_class_f1_at_optimal": [float(f) for f in f1s],
            "fallback_used": [bool(b) for b in fallback_used],
            "class_names": class_names,
            "macro_f1_with_optimal_thresholds": float(np.mean(f1s)),
            "low_support_threshold_fallback": int(low_support),
        }
        with open(config.threshold_path(task), "w", encoding="utf-8") as f:
            json.dump(threshold_data, f, indent=2, ensure_ascii=False)
        print(f"  Saved thresholds: {config.threshold_path(task)}")
        print(f"  Macro-F1 (tuned thresholds): {np.mean(f1s):.4f}")

    # Save log (per-seed file when ensemble, single file otherwise)
    log_suffix = f"_s{seed}" if is_ensemble_seed else ""
    log_path = config.OUTPUT_DIR / f"training_log_{task}{log_suffix}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "task": task,
            "seed": seed,
            "best_val_metric_name": best_metric_name,
            "best_val_metric": float(best_val_metric),
            "log": log,
            "config": {
                "seed": seed,
                "lr": config.LR,
                "batch_size": batch_size,
                "epochs": n_epochs,
                "dropout": getattr(config, "DROPOUT", 0.1),
                "use_amp": use_amp,
                "multilabel_loss": getattr(config, "MULTILABEL_LOSS", "asymmetric"),
                "smoke": smoke,
            },
        }, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] task={task}  seed={seed}  best_val_{best_metric_name}={best_val_metric:.4f}")
    return best_val_metric


# ==================== Ensemble combiner ====================
def build_ensemble(task: str):
    """Combine all seed-trained models into a single ensemble threshold file.

    Loads each seed's trained model, predicts probabilities on val, averages
    across seeds, then tunes thresholds on the averaged probabilities. The
    resulting thresholds_{task}.json is the artifact evaluate.py / inference.py
    pick up — no change needed downstream as long as those paths also load
    every seed model and average probabilities.
    """
    if "fields" not in task and "levels" not in task:
        # Single-label tasks don't use thresholds; nothing to tune.
        return None

    seeds = config.ENSEMBLE_SEEDS
    print("=" * 80)
    print(f"Building ensemble for {task} — averaging {len(seeds)} seed models")
    print("=" * 80)

    # Determine target_type / loader once
    train_df, val_df, test_df = load_train_val_test(task)
    if task in ("fields", "levels"):
        target_cols = get_target_cols(task)
        n_classes = len(target_cols)
        target_type = "multi_label"
    else:
        n_classes = len(config.METHODS_5)
        target_type = "single_label"

    # M1: same TAPT-aware encoder resolution as train_model().
    encoder_path = config.BACKBONE_MODEL
    if bool(getattr(config, "USE_TAPT", False)):
        tapt_dir = getattr(config, "TAPT_OUTPUT_DIR", None)
        if tapt_dir is not None and Path(tapt_dir).exists():
            encoder_path = str(tapt_dir)
            print(f"  Loading TAPT-adapted encoder from {encoder_path}")
    is_tapt_load = encoder_path != config.BACKBONE_MODEL

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(encoder_path)
    method_to_idx = {m: i for i, m in enumerate(config.METHODS_5)}
    val_ds = utils.PaperDataset(
        val_df, tokenizer, target_cols=target_cols, target_type=target_type,
        method_to_idx=method_to_idx, max_length=config.MAX_LENGTH,
    )
    val_loader = DataLoader(val_ds, batch_size=config.effective_batch_size(), shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Aggregate probabilities across seeds
    sum_probs = None
    val_targets = None
    n_loaded = 0
    for seed in seeds:
        path = _model_save_path(task, seed)
        if not path.exists():
            print(f"  [skip] seed={seed} not trained: {path}")
            continue
        model = SpecterClassifier(
            encoder_path, n_classes=n_classes,
            dropout=getattr(config, "DROPOUT", 0.1),
            revision=None if is_tapt_load else getattr(config, "BACKBONE_REVISION", None),
        ).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        probs_seed, val_targets = predict_probs(model, val_loader, device, target_type)
        if sum_probs is None:
            sum_probs = probs_seed
        else:
            sum_probs = sum_probs + probs_seed
        n_loaded += 1
        print(f"  loaded seed={seed} from {path.name}")
    if n_loaded == 0:
        print("  No seed models found; ensemble skipped.")
        return None
    avg_probs = sum_probs / n_loaded

    # Tune thresholds on averaged probabilities
    low_support = getattr(config, "LOW_SUPPORT_THRESHOLD_FALLBACK", 10)
    thresholds, f1s, fallback_used = utils.tune_thresholds_robust(
        avg_probs, val_targets, config.THRESHOLD_GRID,
        low_support_threshold=low_support,
    )
    class_names = get_class_names(task)
    threshold_data = {
        "thresholds": [float(t) for t in thresholds],
        "per_class_f1_at_optimal": [float(f) for f in f1s],
        "fallback_used": [bool(b) for b in fallback_used],
        "class_names": class_names,
        "macro_f1_with_optimal_thresholds": float(np.mean(f1s)),
        "low_support_threshold_fallback": int(low_support),
        "ensemble_seeds": list(seeds[:n_loaded]),
        "n_models_in_ensemble": n_loaded,
    }
    with open(config.threshold_path(task), "w", encoding="utf-8") as f:
        json.dump(threshold_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved ensemble thresholds: {config.threshold_path(task)}")
    print(f"  Ensemble macro-F1 (tuned): {np.mean(f1s):.4f}")
    return float(np.mean(f1s))


# ==================== Evaluation helpers ====================
def _compute_macro_auc(targets, probs, target_type, n_classes):
    """Macro AUC tolerating undefined classes (returns NaN-mean of valid classes).

    Empty-positive or empty-negative classes are skipped — they have no AUC.
    """
    from sklearn.metrics import roc_auc_score
    aucs = []
    if target_type == "multi_label":
        for c in range(n_classes):
            y = targets[:, c].astype(int)
            n_pos = int(y.sum())
            if n_pos == 0 or n_pos == len(y):
                continue
            try:
                aucs.append(float(roc_auc_score(y, probs[:, c])))
            except ValueError:
                continue
    else:
        # multi-class single-label, one-vs-rest macro
        targets_int = targets.astype(int)
        for c in range(n_classes):
            y = (targets_int == c).astype(int)
            n_pos = int(y.sum())
            if n_pos == 0 or n_pos == len(y):
                continue
            try:
                aucs.append(float(roc_auc_score(y, probs[:, c])))
            except ValueError:
                continue
    if not aucs:
        return float("nan")
    return float(np.mean(aucs))


def predict_probs(model, loader, device, target_type):
    """Return (probs, targets) numpy arrays."""
    model.eval()
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids, attention_mask)
            if target_type == "multi_label":
                probs = torch.sigmoid(logits).cpu().numpy()
            else:
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(batch["labels"].cpu().numpy())
    return np.concatenate(all_probs), np.concatenate(all_targets)


def evaluate(model, loader, target_type, n_classes, device, threshold=0.5):
    """Compute macro/micro F1 + macro AUC with default threshold.

    macro_auc is computed from raw probabilities and is threshold-independent
    — used by best-model selection because it's robust on small val sets where
    F1 fluctuates wildly with threshold.
    """
    model.eval()
    all_probs, all_preds, all_targets = [], [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids, attention_mask)
            if target_type == "multi_label":
                probs = torch.sigmoid(logits)
                preds = (probs >= threshold).int().cpu().numpy()
                all_probs.append(probs.cpu().numpy())
            else:
                probs = torch.softmax(logits, dim=-1)
                preds = logits.argmax(dim=-1).cpu().numpy()
                all_probs.append(probs.cpu().numpy())
            all_preds.append(preds)
            all_targets.append(batch["labels"].cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    if target_type == "single_label":
        all_targets = all_targets.astype(int)

    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    micro_f1 = f1_score(all_targets, all_preds, average="micro", zero_division=0)
    per_class = f1_score(all_targets, all_preds, average=None, zero_division=0).tolist()

    # macro AUC — robust to threshold + class imbalance
    macro_auc = _compute_macro_auc(all_targets, all_probs, target_type, n_classes)

    return {
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "macro_auc": float(macro_auc),
        "per_class_f1": [float(x) for x in per_class],
    }


# ==================== Main ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["fields", "levels", "method", "all"], required=True)
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: 1 epoch, 50 train + 20 val papers, batch 4")
    parser.add_argument("--ensemble", action="store_true",
                        help=f"Train {len(config.ENSEMBLE_SEEDS)} seeds and ensemble: "
                             f"{config.ENSEMBLE_SEEDS}. Linear training-time cost.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override single training seed (default: config.SEED)")
    parser.add_argument("--include-val-in-train", action="store_true",
                        help="Phase F: fold val 2023 into train, retrain for fixed "
                             "epochs (no early stop, no threshold tuning). Reuses "
                             "thresholds_*.json from a prior train→val run. Saves "
                             "models with `_trainval` suffix.")
    args = parser.parse_args()

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tasks = ["fields", "levels", "method"] if args.task == "all" else [args.task]
    seeds = config.ENSEMBLE_SEEDS if args.ensemble else [
        args.seed if args.seed is not None else config.SEED
    ]

    for t in tasks:
        for seed in seeds:
            train_model(t, smoke=args.smoke, seed=seed,
                        include_val_in_train=args.include_val_in_train)
        if args.ensemble and not args.smoke and not args.include_val_in_train:
            # build_ensemble tunes thresholds on val — also incompatible with
            # train+val mode for the same circular-tuning reason.
            build_ensemble(t)


if __name__ == "__main__":
    main()
