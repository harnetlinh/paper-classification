"""
Phase 4: Inference

Two modes:

1. CORPUS mode (NHIỆM VỤ 5) — predict on the hybrid 2015-2024 corpus and
   produce a parquet with both human labels (when present) and predictions
   for use by export_review.py's comprehensive review workbook:

       python inference.py --corpus outputs/inference_corpus_2015_2024.parquet

   Output: outputs/predictions_2015_2024.parquet with columns
       Total_ID, Year, Title, Abstract, has_human_labels,
       fields_human, levels_human, method_human   (None when unlabeled)
       fields_pred, levels_pred, method_pred       (concatenated strings)
       fields_<name>_prob × 12, levels_<name>_prob × 6, method_<name>_prob × 5

2. EXCEL mode (legacy) — predict on a raw Scopus-style Excel:

       python inference.py --input data/scopus_2025.xlsx --output preds.xlsx

Encoder resolution: prefers `outputs/specter2_tapt/` when USE_TAPT=True and
the directory exists, falling back to `config.BACKBONE_MODEL`. Matches
train_specter2.py logic so predictions use the same encoder the models
were trained with.
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import config
import utils
from train_specter2 import (
    SpecterClassifier, get_target_cols, get_class_names, _model_save_path,
)
from sanitize import normalize_whitespace

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


# ==================== Encoder path (TAPT-aware) ====================
def _resolve_encoder_path() -> tuple:
    """Return (encoder_path, is_tapt_load). Matches train_specter2.train_model."""
    encoder_path = config.BACKBONE_MODEL
    if bool(getattr(config, "USE_TAPT", False)):
        tapt_dir = getattr(config, "TAPT_OUTPUT_DIR", None)
        if tapt_dir is not None and Path(tapt_dir).exists():
            encoder_path = str(tapt_dir)
            print(f"  Loading TAPT-adapted encoder from {encoder_path}")
            return encoder_path, True
        if tapt_dir is not None:
            print(f"  WARNING: USE_TAPT=True but {tapt_dir} missing — "
                  f"falling back to HF Hub backbone ({config.BACKBONE_MODEL})")
    return encoder_path, False


def _list_seed_models(task: str):
    """Prefer Phase-F `_trainval` checkpoints; fall back to plain seed files."""
    seeds_to_try = [config.SEED] + [
        s for s in getattr(config, "ENSEMBLE_SEEDS", []) if s != config.SEED
    ]
    found = []
    for seed in seeds_to_try:
        p_trainval = _model_save_path(task, seed, include_val=True)
        if p_trainval.exists():
            found.append((seed, p_trainval))
            continue
        p = _model_save_path(task, seed, include_val=False)
        if p.exists():
            found.append((seed, p))
    return found


# ==================== Excel-mode loader (legacy) ====================
def load_input_excel(path: str, sheet: str = None) -> pd.DataFrame:
    """Load a raw Scopus-style Excel for legacy inference mode."""
    df = pd.read_excel(path, sheet_name=sheet) if sheet else pd.read_excel(path)
    required = {"Title", "Abstract"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )
    df["Title"] = df["Title"].apply(normalize_whitespace)
    df["Abstract"] = df["Abstract"].apply(normalize_whitespace)
    n_before = len(df)
    df = df[(df["Title"] != "") | (df["Abstract"] != "")].reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"Dropped {n_dropped} rows missing both Title and Abstract")
    print(f"Loaded {len(df)} papers for inference (Excel mode)")
    return df


# Backward-compatible alias kept for smoke_test.py and any external callers
# that imported the old name. The new code path is load_input_excel().
load_input = load_input_excel


# ==================== Corpus-mode loader (NHIỆM VỤ 5) ====================
def load_corpus_parquet(path: str) -> pd.DataFrame:
    """Load the 2015-2024 inference corpus parquet."""
    df = pd.read_parquet(path)
    required = {"Total_ID", "Year", "Title", "Abstract", "has_human_labels"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Corpus parquet missing columns: {missing}. "
            f"Available: {list(df.columns)}"
        )
    df = df.reset_index(drop=True)
    print(f"Loaded {len(df)} papers from corpus {path}")
    labeled = int(df["has_human_labels"].sum())
    print(f"  has_human_labels: {labeled} labeled, {len(df) - labeled} unlabeled")
    return df


# ==================== Core prediction ====================
def predict_task_raw(df: pd.DataFrame, task: str, device, tokenizer,
                      encoder_path: str, is_tapt_load: bool):
    """Predict probabilities (no derived label strings yet).

    Returns: (probs[N, n_classes], thresholds, class_names, n_classes, target_type)
    """
    if task in ("fields", "levels"):
        target_cols = get_target_cols(task)
        n_classes = len(target_cols)
        target_type = "multi_label"
    else:
        n_classes = len(config.METHODS_5)
        target_type = "single_label"

    seed_models_paths = _list_seed_models(task)
    if not seed_models_paths:
        raise FileNotFoundError(
            f"No model checkpoint for task={task}. Expected {config.model_path(task)}"
        )

    print(f"\n--- Inference: {task} ({n_classes} classes, "
          f"{len(seed_models_paths)} model{'s' if len(seed_models_paths) > 1 else ''}) ---")

    models = []
    for seed, path in seed_models_paths:
        m = SpecterClassifier(
            encoder_path, n_classes=n_classes,
            dropout=getattr(config, "DROPOUT", 0.1),
            revision=None if is_tapt_load else getattr(config, "BACKBONE_REVISION", None),
        ).to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        m.eval()
        models.append(m)

    # Thresholds (multi-label only)
    thresholds = None
    if target_type == "multi_label" and config.threshold_path(task).exists():
        with open(config.threshold_path(task), "r", encoding="utf-8") as f:
            thresholds = json.load(f).get("thresholds")
    if thresholds is None or (target_type == "multi_label" and len(thresholds) != n_classes):
        thresholds = [0.5] * n_classes

    sep = tokenizer.sep_token
    texts = utils.build_input_texts(df, sep)
    enc = tokenizer(
        texts, padding="max_length", truncation=True,
        max_length=config.MAX_LENGTH, return_tensors="pt",
    )

    n = len(df)
    bs = config.effective_batch_size()
    all_probs = []
    with torch.no_grad():
        for i in tqdm(range(0, n, bs), desc=f"{task} inference"):
            batch_ids = enc["input_ids"][i:i + bs].to(device)
            batch_mask = enc["attention_mask"][i:i + bs].to(device)
            sum_probs = None
            for m in models:
                logits = m(batch_ids, batch_mask)
                if target_type == "multi_label":
                    p = torch.sigmoid(logits).cpu().numpy()
                else:
                    p = torch.softmax(logits, dim=-1).cpu().numpy()
                sum_probs = p if sum_probs is None else sum_probs + p
            all_probs.append(sum_probs / len(models))
    probs = np.concatenate(all_probs)

    del models
    if device.type == "cuda":
        torch.cuda.empty_cache()

    class_names = get_class_names(task)
    return probs, thresholds, class_names, n_classes, target_type


def _build_pred_columns(probs, thresholds, class_names, target_type, task: str) -> pd.DataFrame:
    """Build the prediction columns dataframe for one task."""
    n = probs.shape[0]
    n_classes = probs.shape[1]
    out = pd.DataFrame()
    if target_type == "multi_label":
        for c, cname in enumerate(class_names):
            out[f"{task}_{cname}_prob"] = probs[:, c].round(4)
            out[f"{task}_{cname}_pred"] = (probs[:, c] >= thresholds[c]).astype(int)
        labels_str = []
        for i in range(n):
            labels = [class_names[c] for c in range(n_classes)
                      if probs[i, c] >= thresholds[c]]
            labels_str.append("; ".join(labels))
        out[f"{task}_predicted"] = labels_str
    else:
        preds = probs.argmax(axis=1)
        for c, cname in enumerate(class_names):
            out[f"{task}_{cname}_prob"] = probs[:, c].round(4)
        out[f"{task}_predicted"] = [class_names[p] for p in preds]
        out[f"{task}_confidence"] = probs.max(axis=1).round(4)
    return out


# ==================== Corpus-mode output (NHIỆM VỤ 5) ====================
def _format_human_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Build human-label string columns from corpus parquet.

    fields_human / levels_human = ';'-joined list when row.has_human_labels,
    else None. method_human = string or None.
    """
    out = pd.DataFrame()

    def _join_list(lst, has_label):
        if not has_label or lst is None:
            return None
        if isinstance(lst, np.ndarray):
            lst = lst.tolist()
        if not isinstance(lst, list):
            return None
        return "; ".join(str(x) for x in lst)

    out["fields_human"] = [_join_list(df.iloc[i].get("fields_list"),
                                        df.iloc[i]["has_human_labels"])
                            for i in range(len(df))]
    out["levels_human"] = [_join_list(df.iloc[i].get("levels_list"),
                                        df.iloc[i]["has_human_labels"])
                            for i in range(len(df))]
    out["method_human"] = [
        df.iloc[i].get("method") if df.iloc[i]["has_human_labels"] else None
        for i in range(len(df))
    ]
    return out


def run_corpus_inference(corpus_path: str, output_path: str):
    """Full inference pipeline for corpus mode."""
    print("=" * 80)
    print("Phase 4: Corpus inference (NHIỆM VỤ 5)")
    print("=" * 80)

    df = load_corpus_parquet(corpus_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    encoder_path, is_tapt_load = _resolve_encoder_path()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(encoder_path)

    # Predict 3 tasks
    fields_probs, fields_thr, _, _, _ = predict_task_raw(
        df, "fields", device, tokenizer, encoder_path, is_tapt_load,
    )
    levels_probs, levels_thr, _, _, _ = predict_task_raw(
        df, "levels", device, tokenizer, encoder_path, is_tapt_load,
    )
    method_probs, _, _, _, _ = predict_task_raw(
        df, "method", device, tokenizer, encoder_path, is_tapt_load,
    )

    # Assemble output dataframe — include rich features AND the original
    # human label list columns (fields_list, levels_list, method) so that
    # export_review.py can drive its sheet builders directly off this parquet
    # without joining back against the inference corpus parquet.
    base_cols = pd.DataFrame({
        "Total_ID": df["Total_ID"].astype("Int64"),  # nullable int
        "Year": df["Year"].astype(int),
        "Title": df["Title"],
        "Abstract": df["Abstract"],
        "Author Keywords": df.get("Author Keywords", ""),
        "Source title": df.get("Source title", ""),
        "Document type": df.get("Document type", ""),
        "fields_list": df.get("fields_list", pd.Series([[]] * len(df))),
        "levels_list": df.get("levels_list", pd.Series([[]] * len(df))),
        "method": df.get("method", pd.Series([None] * len(df))),
        "has_human_labels": df["has_human_labels"].astype(bool),
    })

    human_cols = _format_human_labels(df)
    fields_pred_cols = _build_pred_columns(
        fields_probs, fields_thr, config.FIELDS_12, "multi_label", "fields",
    )
    levels_pred_cols = _build_pred_columns(
        levels_probs, levels_thr, config.LEVELS_6, "multi_label", "levels",
    )
    method_pred_cols = _build_pred_columns(
        method_probs, None, config.METHODS_5, "single_label", "method",
    )

    out = pd.concat([
        base_cols.reset_index(drop=True),
        human_cols.reset_index(drop=True),
        fields_pred_cols.reset_index(drop=True),
        levels_pred_cols.reset_index(drop=True),
        method_pred_cols.reset_index(drop=True),
    ], axis=1)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    print(f"\nSaved predictions: {output_path}")
    print(f"Total papers: {len(out)}")
    print(f"  Labeled (have human truth):    {int(out['has_human_labels'].sum())}")
    print(f"  Unlabeled (predictions only):  {int((~out['has_human_labels']).sum())}")

    # Quick per-year predicted-field summary
    print("\nPredicted field positives per year (top-3 classes per year shown):")
    yearly = []
    for y in sorted(out["Year"].unique()):
        sub = out[out["Year"] == y]
        per_class = {}
        for f in config.FIELDS_12:
            col = f"fields_{f}_pred"
            if col in sub.columns:
                per_class[f] = int(sub[col].sum())
        top3 = sorted(per_class.items(), key=lambda x: -x[1])[:3]
        top3_str = ", ".join(f"{k}={v}" for k, v in top3)
        yearly.append((y, len(sub), top3_str))
    for y, n, top3 in yearly:
        print(f"  {y}: n={n:>4d}  top3: {top3}")

    return 0


# ==================== Excel-mode pipeline (legacy) ====================
def run_excel_inference(input_path: str, output_path: str, sheet: str = None):
    """Legacy Excel-in / Excel-out inference."""
    print("=" * 80)
    print("Phase 4: Excel inference (legacy)")
    print("=" * 80)

    df = load_input_excel(input_path, sheet)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    encoder_path, is_tapt_load = _resolve_encoder_path()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(encoder_path)

    pred_cols_list = []
    for task in ("fields", "levels", "method"):
        probs, thresholds, class_names, _, target_type = predict_task_raw(
            df, task, device, tokenizer, encoder_path, is_tapt_load,
        )
        pred_cols_list.append(
            _build_pred_columns(probs, thresholds, class_names, target_type, task)
        )

    out = pd.concat([df.reset_index(drop=True)] + pred_cols_list, axis=1)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_excel(output_path, index=False)
    print(f"\nSaved predictions to: {output_path}")
    print(f"Total papers: {len(out)}")

    print("\nField distribution (predicted):")
    for f in config.FIELDS_12:
        col = f"fields_{f}_pred"
        if col in out.columns:
            n = out[col].sum()
            print(f"  {f:35s}: {n:4d} ({n / len(out) * 100:.1f}%)")
    print("\nLevel distribution (predicted):")
    for l in config.LEVELS_6:
        col = f"levels_{l}_pred"
        if col in out.columns:
            n = out[col].sum()
            print(f"  {l:5s}: {n:4d} ({n / len(out) * 100:.1f}%)")
    print("\nMethod distribution (predicted):")
    print(out["method_predicted"].value_counts().to_string())

    return 0


# ==================== CLI ====================
def main():
    parser = argparse.ArgumentParser()
    # Two mutually exclusive input modes
    parser.add_argument(
        "--corpus",
        default=str(config.INFERENCE_CORPUS_2015_2024_PARQUET),
        help="Path to inference corpus parquet (default: "
             "outputs/inference_corpus_2015_2024.parquet — NHIỆM VỤ 5 mode)",
    )
    parser.add_argument(
        "--input", default=None,
        help="Path to raw Excel input (legacy Scopus mode). If set, "
             "overrides --corpus and runs Excel→Excel inference.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output path. Defaults: corpus mode → "
             "outputs/predictions_2015_2024.parquet; Excel mode required.",
    )
    parser.add_argument("--sheet", default=None,
                        help="Excel sheet name (only used in legacy --input mode)")
    args = parser.parse_args()

    # Verify all 3 models exist (either trainval or seed)
    missing_models = []
    for task in ["fields", "levels", "method"]:
        if not _list_seed_models(task):
            missing_models.append(task)
    if missing_models:
        print(f"ERROR: Missing trained models: {missing_models}")
        print("Run: python train_specter2.py --task all")
        return 1

    if args.input:
        if not args.output:
            print("ERROR: --output is required when --input is set (Excel mode).")
            return 1
        return run_excel_inference(args.input, args.output, args.sheet)

    # Corpus mode
    output = args.output or str(config.PREDICTIONS_2015_2024_PARQUET)
    return run_corpus_inference(args.corpus, output)


if __name__ == "__main__":
    raise SystemExit(main())
