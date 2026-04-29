"""
Phase 4: Inference on new Scopus data
======================================

Apply 3 trained models on a new Scopus Excel file (e.g., 2025+ data).

Input:  Excel file with columns Title, Abstract (other columns preserved in output)
Output: Excel with predictions + confidence/probability columns

Usage:
    python inference.py --input data/scopus_2025.xlsx --output outputs/predictions_2025.xlsx
    python inference.py --input data/scopus_2025.xlsx --output outputs/preds.xlsx --sheet "Sheet1"
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
import utils
from train_specter2 import SpecterClassifier, get_target_cols, get_class_names
from sanitize import normalize_whitespace

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


def load_input(path: str, sheet: str = None) -> pd.DataFrame:
    """Load and sanitize input Excel."""
    if sheet:
        df = pd.read_excel(path, sheet_name=sheet)
    else:
        df = pd.read_excel(path)
    
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
    
    print(f"Loaded {len(df)} papers for inference")
    return df


def predict_task(df: pd.DataFrame, task: str, device, tokenizer):
    """
    Run inference for one task.
    
    Returns DataFrame with new columns (probabilities + predictions).
    For multi-label tasks: per-class prob + per-class pred + concatenated label string.
    For single-label task (Method): single prediction + confidence.
    """
    print(f"\n--- Inference: {task} ---")
    
    if task in ("fields", "levels"):
        target_cols = get_target_cols(task)
        n_classes = len(target_cols)
        target_type = "multi_label"
    else:
        n_classes = len(config.METHODS_5)
        target_type = "single_label"
    
    # Load model
    model = SpecterClassifier(config.SPECTER2_BASE, n_classes=n_classes).to(device)
    state = torch.load(config.model_path(task), map_location=device)
    model.load_state_dict(state)
    model.eval()
    
    # Load tuned thresholds (multi-label only)
    thresholds = None
    if target_type == "multi_label" and config.threshold_path(task).exists():
        with open(config.threshold_path(task), "r", encoding="utf-8") as f:
            thresholds = json.load(f).get("thresholds")
    if thresholds is None or len(thresholds) != n_classes:
        thresholds = [0.5] * n_classes
    
    # Tokenize inputs
    sep = tokenizer.sep_token
    texts = [
        (str(t).strip() + sep + str(a).strip())
        for t, a in zip(df["Title"], df["Abstract"])
    ]
    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=config.MAX_LENGTH,
        return_tensors="pt",
    )
    
    # Batched inference
    n = len(df)
    all_probs = []
    with torch.no_grad():
        for i in tqdm(range(0, n, config.BATCH_SIZE), desc=f"{task} inference"):
            batch_ids = enc["input_ids"][i:i+config.BATCH_SIZE].to(device)
            batch_mask = enc["attention_mask"][i:i+config.BATCH_SIZE].to(device)
            logits = model(batch_ids, batch_mask)
            if target_type == "multi_label":
                probs = torch.sigmoid(logits).cpu().numpy()
            else:
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
    probs = np.concatenate(all_probs)
    
    # Build output columns
    class_names = get_class_names(task)
    out_cols = pd.DataFrame()
    
    if target_type == "multi_label":
        for c, cname in enumerate(class_names):
            out_cols[f"{task}_{cname}_prob"] = probs[:, c].round(4)
            out_cols[f"{task}_{cname}_pred"] = (probs[:, c] >= thresholds[c]).astype(int)
        # Concatenated label string (";" separator)
        labels_str = []
        for i in range(n):
            labels = [class_names[c] for c in range(n_classes)
                      if probs[i, c] >= thresholds[c]]
            labels_str.append("; ".join(labels))
        out_cols[f"{task}_predicted"] = labels_str
    else:
        preds = probs.argmax(axis=1)
        out_cols[f"{task}_predicted"] = [class_names[p] for p in preds]
        out_cols[f"{task}_confidence"] = probs.max(axis=1).round(4)
    
    # Free memory
    del model, state
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    return out_cols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input Excel file path")
    parser.add_argument("--output", required=True, help="Output Excel file path")
    parser.add_argument("--sheet", default=None, help="Sheet name (default: first sheet)")
    args = parser.parse_args()
    
    # Verify all 3 models exist
    missing_models = []
    for task in ["fields", "levels", "method"]:
        if not config.model_path(task).exists():
            missing_models.append(task)
    if missing_models:
        print(f"ERROR: Missing trained models: {missing_models}")
        print(f"Run: python train_specter2.py --task all")
        return 1
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load input
    df = load_input(args.input, args.sheet)
    
    # Tokenizer (loaded once, shared)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.SPECTER2_BASE)
    
    # Run inference for each task
    fields_out = predict_task(df, "fields", device, tokenizer)
    levels_out = predict_task(df, "levels", device, tokenizer)
    method_out = predict_task(df, "method", device, tokenizer)
    
    # Combine: original columns + predictions
    all_outputs = pd.concat([
        df.reset_index(drop=True),
        fields_out,
        levels_out,
        method_out,
    ], axis=1)
    
    # Save Excel
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    all_outputs.to_excel(args.output, index=False)
    print(f"\nSaved predictions to: {args.output}")
    print(f"Total papers: {len(all_outputs)}")
    
    # Print prediction distribution
    print(f"\nField distribution (predicted):")
    for f in config.FIELDS_12:
        col = f"fields_{f}_pred"
        if col in all_outputs.columns:
            n = all_outputs[col].sum()
            print(f"  {f:35s}: {n:4d} ({n/len(all_outputs)*100:.1f}%)")
    
    print(f"\nLevel distribution (predicted):")
    for l in config.LEVELS_6:
        col = f"levels_{l}_pred"
        if col in all_outputs.columns:
            n = all_outputs[col].sum()
            print(f"  {l:5s}: {n:4d} ({n/len(all_outputs)*100:.1f}%)")
    
    print(f"\nMethod distribution (predicted):")
    print(all_outputs["method_predicted"].value_counts().to_string())
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
