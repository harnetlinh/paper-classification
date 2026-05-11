"""
Phase 4b: Export human-review workbook (machine vs human classification).

Builds a multi-sheet Excel workbook comparing machine predictions against
gold/human labels on val 2023 and test 2024 splits, sorted by disagreement
severity so a human reviewer can prioritize where to look.

Sheets:
- summary:           1 row/paper, all 3 tasks side-by-side, missing/extra
                     labels broken out, jaccard similarity per task,
                     review_priority for sorting.
- disagreements:     same shape as summary, but filtered to papers where the
                     machine and human disagree on at least one task.
- fields_proba:      per-class human/pred/probability for the Fields task.
- levels_proba:      same, 6 levels.
- method_proba:      single-label argmax probabilities + softmax over 5 methods.
- stats:             per-class precision / recall / F1 / support over the
                     selected split (default test_2024).
- legend:            short column-by-column glossary for reviewer onboarding.

Auto-detects ensemble checkpoints (`model_{task}_s{seed}.pt`). Loads tuned
thresholds from `thresholds_{task}.json` if present, else falls back to 0.5.

Usage:
    python export_review.py
    python export_review.py --output outputs/review.xlsx
    python export_review.py --split test     # only test 2024
    python export_review.py --abstract-chars 500
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import config
import utils
from train_specter2 import (
    SpecterClassifier, get_target_cols, get_class_names, load_train_val_test,
    _model_save_path,
)

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning)


# ==================== Loaders ====================
def list_seed_models(task: str):
    """Return [(seed, path), ...] for every seed checkpoint that exists."""
    seeds = [config.SEED] + [
        s for s in getattr(config, "ENSEMBLE_SEEDS", []) if s != config.SEED
    ]
    found = []
    for seed in seeds:
        p = _model_save_path(task, seed)
        if p.exists():
            found.append((seed, p))
    return found


def load_thresholds(task: str, n_classes: int):
    """Load tuned thresholds; fallback to 0.5 if missing/wrong-shape."""
    if task == "method":
        return None   # single-label uses argmax
    path = config.threshold_path(task)
    if not path.exists():
        return [0.5] * n_classes
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    th = data.get("thresholds")
    if th is None or len(th) != n_classes:
        return [0.5] * n_classes
    return th


def predict_split(df: pd.DataFrame, task: str, device, tokenizer):
    """Predict probabilities for a split using ensemble (if available).

    Returns:
        probs: ndarray [N, C]
        target_type: 'multi_label' | 'single_label'
        n_classes: int
    """
    if task in ("fields", "levels"):
        target_cols = get_target_cols(task)
        n_classes = len(target_cols)
        target_type = "multi_label"
    else:
        target_cols = None
        n_classes = len(config.METHODS_5)
        target_type = "single_label"

    seed_models = list_seed_models(task)
    if not seed_models:
        raise FileNotFoundError(
            f"No checkpoint for task={task}. Run train_specter2.py --task {task} first."
        )

    # Pre-tokenize once — same input builder as training so the model sees
    # the same field layout it was trained on.
    sep = tokenizer.sep_token
    texts = utils.build_input_texts(df, sep)
    enc = tokenizer(
        texts, padding="max_length", truncation=True,
        max_length=config.MAX_LENGTH, return_tensors="pt",
    )

    n = len(df)
    sum_probs = None
    for seed, path in seed_models:
        model = SpecterClassifier(
            config.BACKBONE_MODEL, n_classes=n_classes,
            dropout=getattr(config, "DROPOUT", 0.1),
            revision=getattr(config, "BACKBONE_REVISION", None),
        ).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()

        all_probs = []
        with torch.no_grad():
            for i in range(0, n, config.effective_batch_size()):
                ids = enc["input_ids"][i:i+config.effective_batch_size()].to(device)
                msk = enc["attention_mask"][i:i+config.effective_batch_size()].to(device)
                logits = model(ids, msk)
                if target_type == "multi_label":
                    p = torch.sigmoid(logits).cpu().numpy()
                else:
                    p = torch.softmax(logits, dim=-1).cpu().numpy()
                all_probs.append(p)
        seed_probs = np.concatenate(all_probs)
        sum_probs = seed_probs if sum_probs is None else sum_probs + seed_probs

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    avg_probs = sum_probs / len(seed_models)
    return avg_probs, target_type, n_classes, len(seed_models)


# ==================== Builders ====================
def _truncate(s, n):
    if not isinstance(s, str):
        return ""
    return s if len(s) <= n else s[:n] + "..."


def _safe_list(v):
    """Coerce a parquet list-cell (Python list, numpy array, or None) to a list.

    Pandas/parquet round-trips list-of-strings as numpy arrays, which break
    `or []` truth-value checks. This helper is the canonical way to normalise.
    """
    if v is None:
        return []
    if isinstance(v, list):
        return v
    try:
        return list(v)
    except TypeError:
        return []


def build_summary_rows(df, fields_probs, levels_probs, method_probs,
                        fields_thr, levels_thr, abstract_chars: int):
    """Build summary rows with per-task human/machine label sets."""
    fields_names = config.FIELDS_12
    levels_names = config.LEVELS_6
    methods_names = config.METHODS_5

    fields_pred = (fields_probs >= np.array(fields_thr)[None, :]).astype(int)
    levels_pred = (levels_probs >= np.array(levels_thr)[None, :]).astype(int)
    method_pred = method_probs.argmax(axis=1)
    method_conf = method_probs.max(axis=1)

    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        # --- Fields ---
        fh = set(_safe_list(row.get("fields_list")))
        fm = {fields_names[c] for c in range(len(fields_names)) if fields_pred[i, c] == 1}
        f_missed = fh - fm
        f_extra = fm - fh
        f_inter = fh & fm
        f_union = fh | fm
        f_jacc = (len(f_inter) / len(f_union)) if f_union else 1.0
        f_disagree = len(f_missed) + len(f_extra)

        # --- Levels ---
        lh = set(_safe_list(row.get("levels_list")))
        lm = {levels_names[c] for c in range(len(levels_names)) if levels_pred[i, c] == 1}
        l_missed = lh - lm
        l_extra = lm - lh
        l_inter = lh & lm
        l_union = lh | lm
        l_jacc = (len(l_inter) / len(l_union)) if l_union else 1.0
        l_disagree = len(l_missed) + len(l_extra)

        # --- Method ---
        mh = row.get("method", "")
        mm = methods_names[method_pred[i]]
        method_match = (mh == mm)
        m_disagree = 0 if method_match else 1

        total_disagree = f_disagree + l_disagree + m_disagree
        # priority: HIGH if all 3 disagree, MED if 2, LOW if 1, OK if 0
        n_tasks_disagree = (1 if f_disagree else 0) + (1 if l_disagree else 0) + (1 if m_disagree else 0)
        priority = ["OK", "LOW", "MEDIUM", "HIGH"][n_tasks_disagree]

        rows.append({
            "Total_ID": int(row["Total_ID"]) if pd.notna(row.get("Total_ID")) else None,
            "Year": int(row["Year"]) if pd.notna(row.get("Year")) else None,
            "Title": _truncate(row.get("Title", ""), 200),
            "Abstract_excerpt": _truncate(row.get("Abstract", ""), abstract_chars),
            # Rich-feature metadata the model also sees during inference. Surfaced
            # here so the human reviewer can sanity-check what context the model
            # used (matters when machine and human labels disagree — sometimes
            # the keywords reveal the intent that the abstract obscures).
            "Author_Keywords": _truncate(str(row.get("Author Keywords", "") or ""), 200),
            "Source_title": _truncate(str(row.get("Source title", "") or ""), 100),
            "Document_type": str(row.get("Document type", "") or ""),
            "Fields_human": "; ".join(sorted(fh)),
            "Fields_machine": "; ".join(sorted(fm)),
            "Fields_missing_in_machine": "; ".join(sorted(f_missed)),
            "Fields_extra_in_machine": "; ".join(sorted(f_extra)),
            "Fields_jaccard": round(f_jacc, 3),
            "Fields_disagreement": f_disagree,
            "Levels_human": "; ".join(sorted(lh)),
            "Levels_machine": "; ".join(sorted(lm)),
            "Levels_missing_in_machine": "; ".join(sorted(l_missed)),
            "Levels_extra_in_machine": "; ".join(sorted(l_extra)),
            "Levels_jaccard": round(l_jacc, 3),
            "Levels_disagreement": l_disagree,
            "Method_human": str(mh),
            "Method_machine": str(mm),
            "Method_confidence": round(float(method_conf[i]), 3),
            "Method_match": method_match,
            "review_priority": priority,
            "total_disagreements": total_disagree,
        })
    return pd.DataFrame(rows)


def build_proba_rows(df, probs, thresholds, class_names, prefix=""):
    """Build per-class probability sheet for a multi-label task.

    Columns per class: {class}_human, {class}_pred, {class}_prob, {class}_status
    where status is TP/FP/FN/TN — lets reviewers filter by error type.
    """
    targets = np.zeros((len(df), len(class_names)), dtype=int)
    for i, (_, row) in enumerate(df.iterrows()):
        col = f"{prefix}_list" if prefix else "fields_list"
        labels = set(_safe_list(row.get(col)))
        for c, name in enumerate(class_names):
            if name in labels:
                targets[i, c] = 1

    pred = (probs >= np.array(thresholds)[None, :]).astype(int)
    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        item = {
            "Total_ID": int(row["Total_ID"]) if pd.notna(row.get("Total_ID")) else None,
            "Year": int(row["Year"]) if pd.notna(row.get("Year")) else None,
            "Title": _truncate(row.get("Title", ""), 120),
        }
        n_disagree = 0
        for c, name in enumerate(class_names):
            h, p, prob = int(targets[i, c]), int(pred[i, c]), float(probs[i, c])
            if h == 1 and p == 1:
                status = "TP"
            elif h == 0 and p == 1:
                status = "FP"
                n_disagree += 1
            elif h == 1 and p == 0:
                status = "FN"
                n_disagree += 1
            else:
                status = "TN"
            item[f"{name}__human"] = h
            item[f"{name}__pred"] = p
            item[f"{name}__prob"] = round(prob, 3)
            item[f"{name}__status"] = status
        item["disagreements"] = n_disagree
        rows.append(item)
    return pd.DataFrame(rows)


def build_method_proba_rows(df, probs):
    """Per-paper softmax probabilities over 5 methods + match status."""
    method_pred = probs.argmax(axis=1)
    method_conf = probs.max(axis=1)
    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        mh = row.get("method", "")
        mm = config.METHODS_5[method_pred[i]]
        item = {
            "Total_ID": int(row["Total_ID"]) if pd.notna(row.get("Total_ID")) else None,
            "Year": int(row["Year"]) if pd.notna(row.get("Year")) else None,
            "Title": _truncate(row.get("Title", ""), 120),
            "Method_human": str(mh),
            "Method_machine": str(mm),
            "Method_match": (mh == mm),
            "Method_confidence": round(float(method_conf[i]), 3),
        }
        for c, name in enumerate(config.METHODS_5):
            item[f"prob_{name}"] = round(float(probs[i, c]), 3)
        rows.append(item)
    return pd.DataFrame(rows)


def build_per_class_stats(df, probs, thresholds, class_names, target_type, prefix=""):
    """Compute precision / recall / F1 / support per class."""
    from sklearn.metrics import precision_recall_fscore_support

    if target_type == "multi_label":
        col = f"{prefix}_list" if prefix else "fields_list"
        targets = np.zeros((len(df), len(class_names)), dtype=int)
        for i, (_, row) in enumerate(df.iterrows()):
            labels = set(_safe_list(row.get(col)))
            for c, name in enumerate(class_names):
                if name in labels:
                    targets[i, c] = 1
        preds = (probs >= np.array(thresholds)[None, :]).astype(int)
        p, r, f, s = precision_recall_fscore_support(
            targets, preds, average=None, zero_division=0,
        )
        rows = []
        for c, name in enumerate(class_names):
            rows.append({
                "class": name,
                "support_human": int(s[c]),
                "support_predicted": int(preds[:, c].sum()),
                "true_positive": int(((targets[:, c] == 1) & (preds[:, c] == 1)).sum()),
                "false_positive": int(((targets[:, c] == 0) & (preds[:, c] == 1)).sum()),
                "false_negative": int(((targets[:, c] == 1) & (preds[:, c] == 0)).sum()),
                "precision": round(float(p[c]), 4),
                "recall": round(float(r[c]), 4),
                "f1": round(float(f[c]), 4),
                "threshold": round(float(thresholds[c]), 3),
            })
        return pd.DataFrame(rows)
    # single-label
    targets = np.array([config.METHODS_5.index(m) if m in config.METHODS_5 else -1
                         for m in df["method"]])
    preds = probs.argmax(axis=1)
    p, r, f, s = precision_recall_fscore_support(
        targets, preds, average=None, zero_division=0,
        labels=list(range(len(class_names))),
    )
    rows = []
    for c, name in enumerate(class_names):
        rows.append({
            "class": name,
            "support_human": int(s[c]),
            "support_predicted": int((preds == c).sum()),
            "precision": round(float(p[c]), 4),
            "recall": round(float(r[c]), 4),
            "f1": round(float(f[c]), 4),
        })
    return pd.DataFrame(rows)


# ==================== NV5 helpers — predictions parquet → sheets ====================
# When inference.py has produced predictions_2015_2024.parquet, export_review.py
# consumes it directly (no in-process inference) and produces the comprehensive
# 27-sheet workbook covering val 2023 + test 2024 + inference-only papers.

def _extract_probs_from_predictions(preds_df: pd.DataFrame, task: str):
    """Stack per-class `{task}_{name}_prob` columns into a [N, n_classes] array."""
    if task == "fields":
        names = config.FIELDS_12
    elif task == "levels":
        names = config.LEVELS_6
    else:
        names = config.METHODS_5
    cols = [f"{task}_{name}_prob" for name in names]
    missing = [c for c in cols if c not in preds_df.columns]
    if missing:
        raise ValueError(f"predictions parquet missing prob columns: {missing}")
    return preds_df[cols].to_numpy(dtype=np.float64)


def build_summary_inference_only(df: pd.DataFrame, fields_probs, levels_probs,
                                   method_probs, fields_thr, levels_thr,
                                   abstract_chars: int) -> pd.DataFrame:
    """Summary rows for inference-only papers (has_human_labels=False).

    No human/machine comparison columns — those would all be empty. Includes
    Year, Title, Abstract excerpt, rich features, and the 3 machine label
    sets + confidence.
    """
    fields_pred = (fields_probs >= np.array(fields_thr)[None, :]).astype(int)
    levels_pred = (levels_probs >= np.array(levels_thr)[None, :]).astype(int)
    method_pred = method_probs.argmax(axis=1)
    method_conf = method_probs.max(axis=1)

    rows = []
    for i in range(len(df)):
        row = df.iloc[i]
        fm = {config.FIELDS_12[c] for c in range(len(config.FIELDS_12))
              if fields_pred[i, c] == 1}
        lm = {config.LEVELS_6[c] for c in range(len(config.LEVELS_6))
              if levels_pred[i, c] == 1}
        mm = config.METHODS_5[method_pred[i]]
        rows.append({
            "Total_ID": int(row["Total_ID"]) if pd.notna(row.get("Total_ID")) else None,
            "Year": int(row["Year"]) if pd.notna(row.get("Year")) else None,
            "Title": _truncate(row.get("Title", ""), 200),
            "Abstract_excerpt": _truncate(row.get("Abstract", ""), abstract_chars),
            "Author_Keywords": _truncate(str(row.get("Author Keywords", "") or ""), 200),
            "Source_title": _truncate(str(row.get("Source title", "") or ""), 100),
            "Document_type": str(row.get("Document type", "") or ""),
            "Fields_machine": "; ".join(sorted(fm)),
            "Levels_machine": "; ".join(sorted(lm)),
            "Method_machine": mm,
            "Method_confidence": round(float(method_conf[i]), 3),
            "n_fields_predicted": len(fm),
            "n_levels_predicted": len(lm),
        })
    return pd.DataFrame(rows)


def build_proba_rows_inference_only(df: pd.DataFrame, probs, thresholds,
                                     class_names) -> pd.DataFrame:
    """Per-class probability sheet for inference-only papers.

    Columns: Total_ID, Year, Title + {class}__pred + {class}__prob (no
    __human and no __status because there's no truth to compare against).
    """
    pred = (probs >= np.array(thresholds)[None, :]).astype(int)
    rows = []
    for i in range(len(df)):
        row = df.iloc[i]
        item = {
            "Total_ID": int(row["Total_ID"]) if pd.notna(row.get("Total_ID")) else None,
            "Year": int(row["Year"]) if pd.notna(row.get("Year")) else None,
            "Title": _truncate(row.get("Title", ""), 120),
        }
        for c, name in enumerate(class_names):
            item[f"{name}__pred"] = int(pred[i, c])
            item[f"{name}__prob"] = round(float(probs[i, c]), 3)
        rows.append(item)
    return pd.DataFrame(rows)


def build_method_proba_rows_inference_only(df: pd.DataFrame, probs) -> pd.DataFrame:
    """Per-paper method probability sheet for inference-only papers.

    Method has 5 classes (single-label); columns: Total_ID, Year, Title,
    Method_machine, Method_confidence, prob_<Method> for each of 5.
    """
    method_pred = probs.argmax(axis=1)
    method_conf = probs.max(axis=1)
    rows = []
    for i in range(len(df)):
        row = df.iloc[i]
        item = {
            "Total_ID": int(row["Total_ID"]) if pd.notna(row.get("Total_ID")) else None,
            "Year": int(row["Year"]) if pd.notna(row.get("Year")) else None,
            "Title": _truncate(row.get("Title", ""), 120),
            "Method_machine": config.METHODS_5[method_pred[i]],
            "Method_confidence": round(float(method_conf[i]), 3),
        }
        for c, name in enumerate(config.METHODS_5):
            item[f"prob_{name}"] = round(float(probs[i, c]), 3)
        rows.append(item)
    return pd.DataFrame(rows)


def build_summary_all(df: pd.DataFrame, fields_probs, levels_probs, method_probs,
                       fields_thr, levels_thr, abstract_chars: int) -> pd.DataFrame:
    """All-papers summary across 2015-2024 with has_human_labels indicator.

    Mixed view — papers WITH human labels get the full comparison columns;
    papers WITHOUT get only the machine prediction columns (human columns
    will be empty / None). The has_human_labels column flags which is which.
    """
    fields_pred = (fields_probs >= np.array(fields_thr)[None, :]).astype(int)
    levels_pred = (levels_probs >= np.array(levels_thr)[None, :]).astype(int)
    method_pred = method_probs.argmax(axis=1)
    method_conf = method_probs.max(axis=1)

    rows = []
    for i in range(len(df)):
        row = df.iloc[i]
        has_labels = bool(row["has_human_labels"])
        fm = {config.FIELDS_12[c] for c in range(len(config.FIELDS_12))
              if fields_pred[i, c] == 1}
        lm = {config.LEVELS_6[c] for c in range(len(config.LEVELS_6))
              if levels_pred[i, c] == 1}
        mm = config.METHODS_5[method_pred[i]]

        if has_labels:
            fh = set(_safe_list(row.get("fields_list")))
            lh = set(_safe_list(row.get("levels_list")))
            mh = row.get("method", "")
            f_jacc = round(len(fh & fm) / len(fh | fm), 3) if (fh | fm) else 1.0
            l_jacc = round(len(lh & lm) / len(lh | lm), 3) if (lh | lm) else 1.0
            method_match = (mh == mm)
        else:
            fh, lh, mh = set(), set(), None
            f_jacc, l_jacc = None, None
            method_match = None

        rows.append({
            "Total_ID": int(row["Total_ID"]) if pd.notna(row.get("Total_ID")) else None,
            "Year": int(row["Year"]) if pd.notna(row.get("Year")) else None,
            "has_human_labels": has_labels,
            "Title": _truncate(row.get("Title", ""), 200),
            "Abstract_excerpt": _truncate(row.get("Abstract", ""), abstract_chars),
            "Author_Keywords": _truncate(str(row.get("Author Keywords", "") or ""), 200),
            "Source_title": _truncate(str(row.get("Source title", "") or ""), 100),
            "Document_type": str(row.get("Document type", "") or ""),
            "Fields_human": "; ".join(sorted(fh)) if has_labels else "",
            "Fields_machine": "; ".join(sorted(fm)),
            "Fields_jaccard": f_jacc,
            "Levels_human": "; ".join(sorted(lh)) if has_labels else "",
            "Levels_machine": "; ".join(sorted(lm)),
            "Levels_jaccard": l_jacc,
            "Method_human": str(mh) if (mh is not None) else "",
            "Method_machine": mm,
            "Method_confidence": round(float(method_conf[i]), 3),
            "Method_match": method_match,
        })
    return pd.DataFrame(rows)


def build_stats_per_year(df: pd.DataFrame, probs, thresholds_or_none,
                          class_names, target_type: str) -> pd.DataFrame:
    """Per-year predicted-positive counts per class.

    Schema: rows=class, columns=year. Cell = number of papers in that year
    predicted positive for that class (multi-label: pred=1; single-label
    Method: argmax == c). Covers ALL papers in df (labeled + unlabeled) so
    the per-year trends include the full 2015-2024 corpus.

    Returns:
        DataFrame with columns ['class', '<year1>', '<year2>', ...] sorted
        ascending by year. NaN-free integers.
    """
    years = sorted(df["Year"].astype(int).unique().tolist())
    if target_type == "multi_label":
        pred = (probs >= np.array(thresholds_or_none)[None, :]).astype(int)
    else:
        # Single-label: build a [N, n_classes] one-hot of argmax for parallel
        # treatment in the year-aggregation loop.
        argmax = probs.argmax(axis=1)
        pred = np.zeros((len(df), len(class_names)), dtype=int)
        pred[np.arange(len(df)), argmax] = 1

    df_idx = df.reset_index(drop=True)
    rows = []
    for c, name in enumerate(class_names):
        item = {"class": name}
        for y in years:
            year_mask = (df_idx["Year"].astype(int) == y).to_numpy()
            count = int(pred[year_mask, c].sum())
            item[str(y)] = count
        item["total"] = int(pred[:, c].sum())
        rows.append(item)
    return pd.DataFrame(rows)


# ==================== Excel writer ====================
LEGEND_ROWS = [
    ("Total_ID", "Paper identifier from the original Excel main sheet."),
    ("Year", "Publication year."),
    ("Title / Abstract_excerpt", "First 200 / N chars (configurable via --abstract-chars)."),
    ("Fields_human", "Human-curated multi-label Fields (semicolon list)."),
    ("Fields_machine", "Model-predicted Fields after threshold tuning."),
    ("Fields_missing_in_machine", "Labels human assigned that the model missed (false negatives)."),
    ("Fields_extra_in_machine", "Labels the model added that the human did not (false positives)."),
    ("Fields_jaccard", "Set similarity between human and machine label sets (1.0 = perfect)."),
    ("Levels_*", "Same fields applied to the 6-level taxonomy."),
    ("Method_human / Method_machine", "Single-label method (Quantitative/Qualitative/Mixed/Review/Other)."),
    ("Method_confidence", "Softmax probability of the predicted method."),
    ("review_priority", "HIGH = all 3 tasks disagree; MEDIUM = 2; LOW = 1; OK = perfect agreement."),
    ("total_disagreements", "Sum of FP+FN across Fields/Levels + (1 if method mismatch). Higher = needs more review."),
    ("status (per-class)", "TP=both label, FP=machine only, FN=human only, TN=neither."),
    ("threshold", "Per-class probability cutoff used for the binary prediction (tuned on val 2023)."),
    ("Notes", "Sheet 'disagreements' is a filtered copy of 'summary_*' — same data, easier to scan."),
]


def write_workbook(output_path, sheets):
    """Write the {sheet_name: dataframe} dict to Excel + apply minimal formatting."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
    # Post-process for column widths + freeze pane
    from openpyxl import load_workbook
    wb = load_workbook(output_path)
    for ws in wb.worksheets:
        ws.freeze_panes = "A2"
        # auto-fit-ish column widths
        for col in ws.columns:
            max_len = 0
            col_letter = col[0].column_letter
            for cell in col[:200]:   # sample first 200 rows
                v = cell.value
                if v is not None:
                    max_len = max(max_len, min(60, len(str(v))))
            ws.column_dimensions[col_letter].width = max(8, min(60, max_len + 2))
    wb.save(output_path)


def write_csv_bundle(output_path, sheets):
    """Write each sheet as a separate CSV under a directory, plus a manifest.

    Useful when the reviewer can't open .xlsx (corrupted download, no Excel
    installed, sandboxed environment that won't unzip xlsx). Each sheet
    becomes <output_dir>/<sheet_name>.csv plus an INDEX.txt with the
    column glossary.
    """
    base = Path(output_path)
    if base.suffix.lower() == ".xlsx":
        base = base.with_suffix("")
    base = base.parent / (base.name + "_csv")
    base.mkdir(parents=True, exist_ok=True)

    for name, df in sheets.items():
        csv_path = base / f"{name}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")   # BOM for Excel-friendly Unicode
    # Manifest
    index_path = base / "INDEX.txt"
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(f"Review bundle — {len(sheets)} sheets exported as CSV\n")
        f.write("=" * 60 + "\n\n")
        f.write("Sheets in this bundle (open any in Excel / Numbers / Google Sheets):\n\n")
        for name, df in sheets.items():
            f.write(f"  - {name}.csv  ({len(df)} rows, {len(df.columns)} cols)\n")
        f.write("\nSuggested workflow:\n")
        f.write("  1. Open `summary_test_2024.csv` first — papers sorted by\n")
        f.write("     review_priority. HIGH = model disagrees on all 3 task\n")
        f.write("     types, MEDIUM = 2/3, LOW = 1/3, OK = full agreement.\n")
        f.write("  2. `disagreements_test_2024.csv` is the same data filtered\n")
        f.write("     to HIGH priority only (the focused review backlog).\n")
        f.write("  3. For per-class drill-down, open `fields_test_2024.csv`,\n")
        f.write("     `levels_test_2024.csv`, `method_test_2024.csv`. Filter by\n")
        f.write("     <class>__status = FP / FN to see model errors.\n")
        f.write("  4. `stats_*.csv` — per-class precision/recall/F1.\n")
        f.write("  5. CSV files are UTF-8 with BOM, so Excel and Sheets show\n")
        f.write("     Vietnamese characters correctly without re-encoding.\n")
    return base


# ==================== NV5 mode driver ====================
def run_nv5_mode(predictions_path: str, output_path: str,
                  abstract_chars: int, also_csv: bool) -> int:
    """Build the comprehensive 27-sheet review workbook from the predictions
    parquet produced by inference.py in corpus mode.

    Sheets produced (27 total):
      Summary (4): summary_all, summary_val_2023, summary_test_2024,
                   summary_inference_only
      Disagreements (4): disagreements_{val,test} + major_disagreements_{val,test}
      Per-class (9): fields/levels/method × {val, test, inference_only}
      Stats per-split (6): stats_{val,test}_{fields,levels,method}
      Stats per-year (3): stats_per_year_{fields,levels,method}  [Row=class, Col=year]
      Legend (1)
    """
    print("=" * 80)
    print("NHIỆM VỤ 5: Comprehensive review workbook (2015-2024)")
    print("=" * 80)

    preds = pd.read_parquet(predictions_path)
    print(f"Loaded predictions: {len(preds)} papers from {predictions_path}")

    # Reconstruct probability matrices per task
    fields_probs = _extract_probs_from_predictions(preds, "fields")
    levels_probs = _extract_probs_from_predictions(preds, "levels")
    method_probs = _extract_probs_from_predictions(preds, "method")

    # Thresholds — same files the predictions were generated with.
    fields_thr = load_thresholds("fields", len(config.FIELDS_12))
    levels_thr = load_thresholds("levels", len(config.LEVELS_6))

    # Split predictions into 3 views by has_human_labels + Year.
    val_mask = (preds["Year"].astype(int) == config.VAL_YEAR) & preds["has_human_labels"]
    test_mask = (preds["Year"].astype(int) == config.TEST_YEAR) & preds["has_human_labels"]
    inf_mask = ~preds["has_human_labels"]

    val_idx = preds.index[val_mask].to_numpy()
    test_idx = preds.index[test_mask].to_numpy()
    inf_idx = preds.index[inf_mask].to_numpy()

    print(f"  val_2023 (labeled):       {len(val_idx)}")
    print(f"  test_2024 (labeled):      {len(test_idx)}")
    print(f"  inference_only:           {len(inf_idx)}")

    sheets = {}

    # === summary_all (across full 2015-2024 corpus) ===
    sheets["summary_all"] = build_summary_all(
        preds, fields_probs, levels_probs, method_probs,
        fields_thr, levels_thr, abstract_chars,
    )

    # === Labeled splits (val_2023, test_2024) — reuse existing builders ===
    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "OK": 3}
    for split_name, idx in [("val_2023", val_idx), ("test_2024", test_idx)]:
        df_split = preds.loc[idx].reset_index(drop=True)
        f_probs = fields_probs[idx]
        l_probs = levels_probs[idx]
        m_probs = method_probs[idx]

        summary = build_summary_rows(
            df_split, f_probs, l_probs, m_probs,
            fields_thr, levels_thr, abstract_chars,
        )
        summary["_p"] = summary["review_priority"].map(priority_order)
        summary = summary.sort_values(
            by=["_p", "total_disagreements"], ascending=[True, False]
        ).drop(columns=["_p"]).reset_index(drop=True)

        sheets[f"summary_{split_name}"] = summary
        sheets[f"disagreements_{split_name}"] = summary[
            summary["review_priority"] == "HIGH"
        ].reset_index(drop=True)
        sheets[f"major_disagreements_{split_name}"] = summary[
            summary["total_disagreements"] >= 4
        ].reset_index(drop=True)
        sheets[f"fields_{split_name}"] = build_proba_rows(
            df_split, f_probs, fields_thr, config.FIELDS_12, prefix="fields",
        )
        sheets[f"levels_{split_name}"] = build_proba_rows(
            df_split, l_probs, levels_thr, config.LEVELS_6, prefix="levels",
        )
        sheets[f"method_{split_name}"] = build_method_proba_rows(df_split, m_probs)
        sheets[f"stats_{split_name}_fields"] = build_per_class_stats(
            df_split, f_probs, fields_thr, config.FIELDS_12, "multi_label", prefix="fields",
        )
        sheets[f"stats_{split_name}_levels"] = build_per_class_stats(
            df_split, l_probs, levels_thr, config.LEVELS_6, "multi_label", prefix="levels",
        )
        sheets[f"stats_{split_name}_method"] = build_per_class_stats(
            df_split, m_probs, None, config.METHODS_5, "single_label",
        )

    # === Inference-only papers (no human labels) ===
    if len(inf_idx) > 0:
        df_inf = preds.loc[inf_idx].reset_index(drop=True)
        f_inf = fields_probs[inf_idx]
        l_inf = levels_probs[inf_idx]
        m_inf = method_probs[inf_idx]
        sheets["summary_inference_only"] = build_summary_inference_only(
            df_inf, f_inf, l_inf, m_inf, fields_thr, levels_thr, abstract_chars,
        )
        sheets["fields_inference_only"] = build_proba_rows_inference_only(
            df_inf, f_inf, fields_thr, config.FIELDS_12,
        )
        sheets["levels_inference_only"] = build_proba_rows_inference_only(
            df_inf, l_inf, levels_thr, config.LEVELS_6,
        )
        sheets["method_inference_only"] = build_method_proba_rows_inference_only(
            df_inf, m_inf,
        )
    else:
        print("  [warn] inference_only split is empty — sheets skipped")

    # === Stats per year (full corpus 2015-2024) ===
    sheets["stats_per_year_fields"] = build_stats_per_year(
        preds, fields_probs, fields_thr, config.FIELDS_12, "multi_label",
    )
    sheets["stats_per_year_levels"] = build_stats_per_year(
        preds, levels_probs, levels_thr, config.LEVELS_6, "multi_label",
    )
    sheets["stats_per_year_method"] = build_stats_per_year(
        preds, method_probs, None, config.METHODS_5, "single_label",
    )

    # === Legend ===
    legend_rows = list(LEGEND_ROWS) + [
        ("has_human_labels", "True = paper has Vietnam gold labels for comparison; "
                              "False = paper from main 2015-2023 with no gold (predictions only)."),
        ("summary_all", "All 2015-2024 papers in one sheet; has_human_labels flag "
                         "indicates whether human comparison columns are populated."),
        ("summary_inference_only", "Subset of summary_all where has_human_labels=False; "
                                     "machine predictions only — no human comparison."),
        ("stats_per_year_*", "Row=class, Column=year, Cell=number of papers in that "
                              "year predicted positive for that class. Tracks distribution "
                              "drift across years."),
    ]
    sheets["legend"] = pd.DataFrame(legend_rows, columns=["column", "meaning"])

    # === Write ===
    print(f"\nWriting {output_path} ({len(sheets)} sheets)...")
    write_workbook(output_path, sheets)
    if also_csv:
        bundle_dir = write_csv_bundle(output_path, sheets)
        print(f"  CSV bundle: {bundle_dir}/")

    print(f"\n{'=' * 60}")
    print("Export complete.")
    print(f"{'=' * 60}")
    print(f"  Output: {output_path}")
    print(f"  Sheets: {len(sheets)}")
    print("    Summary (4):     summary_all, summary_val_2023, "
          "summary_test_2024, summary_inference_only")
    print("    Disagreements (4): {,major_}disagreements_{val_2023,test_2024}")
    print("    Per-class (9): fields/levels/method × {val,test,inference_only}")
    print("    Stats per-split (6): stats_{val,test}_{fields,levels,method}")
    print("    Stats per-year (3):  stats_per_year_{fields,levels,method}")
    print("    Legend (1)")
    return 0


# ==================== Main ====================
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=None,
                        help="Output Excel path. Defaults: NV5 mode → "
                             "outputs/review_full_2015_2024.xlsx; legacy mode → "
                             "outputs/review.xlsx.")
    parser.add_argument("--split", choices=["all", "val", "test"], default="all",
                        help="(Legacy mode only) Which split(s) to export. "
                             "Ignored in NV5 mode — always exports all splits.")
    parser.add_argument("--abstract-chars", type=int, default=300,
                        help="Truncate abstract to N chars in output (default 300)")
    parser.add_argument("--csv", action="store_true",
                        help="ALSO emit a CSV bundle alongside the .xlsx — "
                             "useful when the reviewer can't open Excel files. "
                             "Each sheet becomes <output>_csv/<sheet>.csv with "
                             "UTF-8 BOM for Excel/Sheets-friendly Vietnamese.")
    parser.add_argument("--predictions", default=str(config.PREDICTIONS_2015_2024_PARQUET),
                        help="Path to predictions_2015_2024.parquet (NHIỆM VỤ 5 mode). "
                             "When this file exists, NV5 mode runs (27-sheet "
                             "workbook). Else falls back to legacy 19-sheet val+test.")
    parser.add_argument("--legacy", action="store_true",
                        help="Force legacy mode (predict val+test in-process, "
                             "19-sheet output) even when predictions parquet exists.")
    args = parser.parse_args()

    # ===== NV5 mode (default when predictions parquet exists) =====
    if not args.legacy and Path(args.predictions).exists():
        output = args.output or str(config.OUTPUT_DIR / "review_full_2015_2024.xlsx")
        return run_nv5_mode(args.predictions, output, args.abstract_chars, args.csv)

    # ===== Legacy mode =====
    output = args.output or str(config.OUTPUT_DIR / "review.xlsx")
    args.output = output  # downstream code reads args.output
    # Validate trained models
    missing = [t for t in ["fields", "levels", "method"] if not list_seed_models(t)]
    if missing:
        print(f"ERROR: trained models missing for: {missing}")
        print("Run: python train_specter2.py --task all   (or --ensemble for multi-seed)")
        return 1

    print("Loading data...")
    if not config.GOLD_PARQUET.exists() or not config.MAIN_2024_PARQUET.exists():
        print("ERROR: parquet files missing. Run sanitize.py first.")
        return 1
    # Load raw (un-augmented) labels — augmentation labels were synthesised by
    # the LLM ensemble, the human reviewer wants to compare model output
    # against the human-curated truth, not against the LLM agreement.
    raw_gold = pd.read_parquet(config.GOLD_PARQUET)
    val_df_raw = raw_gold[raw_gold["Year"] == config.VAL_YEAR].copy().reset_index(drop=True)
    test_df_raw = pd.read_parquet(config.MAIN_2024_PARQUET).copy().reset_index(drop=True)

    splits = []
    if args.split in ("all", "val"):
        splits.append(("val_2023", val_df_raw))
    if args.split in ("all", "test"):
        splits.append(("test_2024", test_df_raw))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.BACKBONE_MODEL)

    sheets = {}

    # Predict each split × each task once
    cache = {}
    for split_name, df in splits:
        print(f"\nPredicting {split_name} (n={len(df)})...")
        for task in ["fields", "levels", "method"]:
            print(f"  - {task}", end=" ")
            probs, target_type, n_classes, n_models = predict_split(df, task, device, tokenizer)
            cache[(split_name, task)] = (probs, target_type, n_classes, n_models)
            print(f"({n_models} model{'s' if n_models > 1 else ''})")

    # Build sheets per split
    for split_name, df in splits:
        fields_probs = cache[(split_name, "fields")][0]
        levels_probs = cache[(split_name, "levels")][0]
        method_probs = cache[(split_name, "method")][0]
        fields_thr = load_thresholds("fields", len(config.FIELDS_12))
        levels_thr = load_thresholds("levels", len(config.LEVELS_6))

        summary = build_summary_rows(
            df, fields_probs, levels_probs, method_probs,
            fields_thr, levels_thr, args.abstract_chars,
        )
        # Sort: HIGH disagreement first, then MEDIUM, LOW, OK; tie-break by total_disagreements desc
        priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "OK": 3}
        summary["_p"] = summary["review_priority"].map(priority_order)
        summary = summary.sort_values(
            by=["_p", "total_disagreements"], ascending=[True, False]
        ).drop(columns=["_p"]).reset_index(drop=True)

        sheets[f"summary_{split_name}"] = summary
        # Multi-label tasks rarely achieve full label-set agreement (12 fields ×
        # 6 levels means many ways to disagree), so "any disagreement" filters
        # almost nothing useful. The high-priority view focuses on papers where
        # the model disagrees on ALL THREE task types — that's the human review
        # backlog worth prioritising.
        sheets[f"disagreements_{split_name}"] = summary[
            summary["review_priority"] == "HIGH"
        ].reset_index(drop=True)
        # And a "needs major review" sheet: total disagreements >= 4 (cumulative
        # FP+FN across tasks + method mismatch). Catches cases where 1-2 tasks
        # are wrong by a lot, which the priority bucketing alone doesn't surface.
        sheets[f"major_disagreements_{split_name}"] = summary[
            summary["total_disagreements"] >= 4
        ].reset_index(drop=True)
        sheets[f"fields_{split_name}"] = build_proba_rows(
            df, fields_probs, fields_thr, config.FIELDS_12, prefix="fields",
        )
        sheets[f"levels_{split_name}"] = build_proba_rows(
            df, levels_probs, levels_thr, config.LEVELS_6, prefix="levels",
        )
        sheets[f"method_{split_name}"] = build_method_proba_rows(df, method_probs)
        sheets[f"stats_{split_name}_fields"] = build_per_class_stats(
            df, fields_probs, fields_thr, config.FIELDS_12, "multi_label", prefix="fields",
        )
        sheets[f"stats_{split_name}_levels"] = build_per_class_stats(
            df, levels_probs, levels_thr, config.LEVELS_6, "multi_label", prefix="levels",
        )
        sheets[f"stats_{split_name}_method"] = build_per_class_stats(
            df, method_probs, None, config.METHODS_5, "single_label",
        )

    # Legend
    sheets["legend"] = pd.DataFrame(LEGEND_ROWS, columns=["column", "meaning"])

    print(f"\nWriting {args.output}...")
    write_workbook(args.output, sheets)
    if args.csv:
        bundle_dir = write_csv_bundle(args.output, sheets)
        print(f"  CSV bundle: {bundle_dir}/")

    # Summary stats to stdout
    print(f"\n{'=' * 60}")
    print("Export complete.")
    print(f"{'=' * 60}")
    print(f"  Output: {args.output}")
    for split_name, _ in splits:
        s = sheets[f"summary_{split_name}"]
        n_total = len(s)
        n_high = (s["review_priority"] == "HIGH").sum()
        n_med = (s["review_priority"] == "MEDIUM").sum()
        n_low = (s["review_priority"] == "LOW").sum()
        n_ok = (s["review_priority"] == "OK").sum()
        print(f"\n  {split_name}: {n_total} papers")
        print(f"    HIGH (3/3 tasks disagree): {n_high}  ({n_high/n_total*100:.1f}%)")
        print(f"    MEDIUM (2/3):              {n_med}  ({n_med/n_total*100:.1f}%)")
        print(f"    LOW (1/3):                 {n_low}  ({n_low/n_total*100:.1f}%)")
        print(f"    OK (full agreement):       {n_ok}  ({n_ok/n_total*100:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
