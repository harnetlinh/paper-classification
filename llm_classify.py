"""
Phase A: GPT-5 panel as a third classifier ensembled with the SPECTER2 model
at INFERENCE TIME (no relabeling of training data — that was excluded by user
decision Q3).

Pipeline:
1. For each paper (val 2023 + test 2024), call the OpenAI panel of 3 models
   with the full v2.1 codebook + binary classification per task.
2. Aggregate the 3 votes per class into discrete probabilities {0, 0.33, 0.67, 1.0}.
3. evaluate.py / inference.py read these cached probs and ensemble them with
   the SPECTER2 sigmoid probs via per-class linear weights tuned on val.

Cache strategy:
- Per-(model, prompt_hash) using openai_clients.get_cache_path → re-runs are FREE.
- Per-paper progress JSONL via openai_clients.ProgressTracker → resumable.

Cost estimate (April 2026 OpenAI pricing, see config.OPENAI_PANEL):
- 562 test + 417 val = 979 papers
- 3 models × ~1500 input tokens (codebook+abstract) + ~300 output JSON tokens
- ≈ $15 total. Cached re-runs: $0.

Entry points:
    python llm_classify.py --split test           # run on test 2024
    python llm_classify.py --split val            # run on val 2023
    python llm_classify.py --split both           # both, in sequence
    python llm_classify.py --split test --limit 5 # smoke test
    python llm_classify.py --no-cache             # force re-call API
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

import config
import openai_clients
import prompts


# ==================== Output paths ====================
LLM_CLASSIFY_DIR = config.OUTPUT_DIR / "llm_classify"
LLM_CLASSIFY_DIR.mkdir(parents=True, exist_ok=True)


def predictions_path(split: str) -> Path:
    """Parquet of per-paper aggregated GPT-5 panel probs."""
    return LLM_CLASSIFY_DIR / f"gpt5_panel_{split}.parquet"


def progress_task_name(split: str) -> str:
    return f"gpt5_full_classify_{split}"


# ==================== Vote aggregation ====================
def _bool_or_none(v) -> bool | None:
    """Robustly cast LLM JSON value to True/False/None.

    Same coercion as llm_augment.coerce_bool — kept inline to avoid the
    cross-file dependency and to make this file self-contained.
    """
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if v == 1:
            return True
        if v == 0:
            return False
        return None
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "yes", "y", "1"):
            return True
        if s in ("false", "no", "n", "0"):
            return False
    return None


def aggregate_panel_votes(votes: List[Dict]) -> Dict:
    """Aggregate 3-model panel votes per paper into per-class discrete probabilities.

    Args:
        votes: list of per-model JSON responses (one per panel member). Each is
               the raw parsed JSON from the model (already validated to have
               'fields', 'levels', 'method' keys, or contains '_error').

    Returns:
        dict with:
            fields_probs: [12] floats in {0, 0.33, 0.67, 1.0}
            levels_probs: [6] floats
            method_probs: [5] floats (sums to 1 over the 5 methods if at least
                          one valid vote, else uniform 0.2)
            n_valid_votes: int (how many of the 3 models returned valid JSON)
            confidence: aggregate confidence label
    """
    fields_counts = np.zeros(len(config.FIELDS_12), dtype=np.float64)
    levels_counts = np.zeros(len(config.LEVELS_6), dtype=np.float64)
    method_counts = np.zeros(len(config.METHODS_5), dtype=np.float64)
    confidences = []
    n_valid = 0

    for vote in votes:
        if not isinstance(vote, dict) or "_error" in vote:
            continue
        f = vote.get("fields") or {}
        l = vote.get("levels") or {}
        m = vote.get("method")
        # All three blocks must be present for the vote to count toward
        # ANY task — partial responses get rejected to keep aggregation
        # interpretable. (A model that emitted only `method` would otherwise
        # silently bias method counts vs fields/levels.)
        if not isinstance(f, dict) or not isinstance(l, dict) or not isinstance(m, str):
            continue

        for i, field in enumerate(config.FIELDS_12):
            v = _bool_or_none(f.get(field))
            if v is True:
                fields_counts[i] += 1
        for i, level in enumerate(config.LEVELS_6):
            v = _bool_or_none(l.get(level))
            if v is True:
                levels_counts[i] += 1

        # Method: accept exact case match or alias-canonicalised match.
        m_clean = (m or "").strip()
        if m_clean in config.METHODS_5:
            method_counts[config.METHODS_5.index(m_clean)] += 1
        elif m_clean in config.METHOD_ALIASES:
            canon = config.METHOD_ALIASES[m_clean]
            if canon in config.METHODS_5:
                method_counts[config.METHODS_5.index(canon)] += 1

        n_valid += 1
        conf = vote.get("confidence", "")
        if conf in ("high", "medium", "low"):
            confidences.append(conf)

    if n_valid == 0:
        # All models failed — return uniform (least-information) priors.
        return {
            "fields_probs": np.zeros(len(config.FIELDS_12)).tolist(),
            "levels_probs": np.zeros(len(config.LEVELS_6)).tolist(),
            "method_probs": (np.ones(len(config.METHODS_5)) / len(config.METHODS_5)).tolist(),
            "n_valid_votes": 0,
            "confidence": "low",
        }

    fields_probs = (fields_counts / n_valid).tolist()
    levels_probs = (levels_counts / n_valid).tolist()
    # Method: normalise to a proper distribution over the 5 classes. If no
    # valid model output a method name, fall back to uniform.
    if method_counts.sum() > 0:
        method_probs = (method_counts / method_counts.sum()).tolist()
    else:
        method_probs = (np.ones(len(config.METHODS_5)) / len(config.METHODS_5)).tolist()

    # Aggregate confidence: most pessimistic of the three valid votes.
    if "low" in confidences:
        conf_agg = "low"
    elif "medium" in confidences:
        conf_agg = "medium"
    elif confidences:
        conf_agg = "high"
    else:
        conf_agg = "low"

    return {
        "fields_probs": fields_probs,
        "levels_probs": levels_probs,
        "method_probs": method_probs,
        "n_valid_votes": n_valid,
        "confidence": conf_agg,
    }


# ==================== Per-paper classification ====================
def classify_paper(panel, title: str, abstract: str, use_cache: bool = True) -> Dict:
    """Call all 3 GPT-5 models on one paper, aggregate, return panel result.

    Args:
        panel: list of OpenAIModelClient instances (from openai_clients.get_clients()).
        title, abstract: paper text.
        use_cache: whether to use the per-prompt cache.

    Returns:
        Dict with aggregated probs (see aggregate_panel_votes) plus raw votes.
    """
    sys_p, user_p = prompts.make_full_classification_prompt(title, abstract)
    votes = []
    for client in panel:
        resp = client.classify(sys_p, user_p, use_cache=use_cache)
        votes.append(resp)
    aggregated = aggregate_panel_votes(votes)
    aggregated["raw_votes"] = votes
    return aggregated


# ==================== Run on a split ====================
def _load_split(split: str) -> pd.DataFrame:
    if split not in ("train", "val", "test"):
        raise ValueError(f"Unknown split: {split!r}")
    if split == "test":
        if not config.MAIN_2024_PARQUET.exists():
            raise FileNotFoundError(f"Test set not found: {config.MAIN_2024_PARQUET}")
        return pd.read_parquet(config.MAIN_2024_PARQUET).reset_index(drop=True)
    if not config.GOLD_PARQUET.exists():
        raise FileNotFoundError(f"Gold dataset not found: {config.GOLD_PARQUET}")
    gold = pd.read_parquet(config.GOLD_PARQUET)
    if split == "val":
        return gold[gold["Year"] == config.VAL_YEAR].reset_index(drop=True)
    return gold[gold["Year"].isin(config.TRAIN_YEARS)].reset_index(drop=True)


def run_split(split: str, limit: int = None, use_cache: bool = True) -> pd.DataFrame:
    """Classify every paper in `split` via the GPT-5 panel; persist results.

    Returns the per-paper results DataFrame (also saved to parquet).
    """
    print("=" * 80)
    print(f"Phase A: GPT-5 panel full-classification — split={split}")
    print("=" * 80)

    df = _load_split(split)
    print(f"Loaded {len(df)} papers from split={split}")
    if limit and limit < len(df):
        df = df.head(limit).reset_index(drop=True)
        print(f"--limit applied: only first {limit} papers")

    panel = openai_clients.get_clients()
    cost = openai_clients.estimate_cost(
        panel, n_calls=len(df),
        avg_input_tokens=1500,   # codebook + abstract
        avg_output_tokens=300,    # structured JSON
    )
    print(f"Panel: {[c.model for c in panel]}")
    print(f"Estimated cost: ${cost['total_usd']:.4f}  (cached re-runs are FREE)")
    print(f"Total API calls: {len(df) * len(panel)}\n")

    tracker = openai_clients.ProgressTracker(progress_task_name(split))
    done_ids = tracker.load_done_ids()
    print(f"Already processed (resume): {len(done_ids)} papers")

    to_run = df[~df["Total_ID"].astype(int).isin(done_ids)].copy()
    print(f"Remaining to process: {len(to_run)}\n")

    for _, row in tqdm(to_run.iterrows(), total=len(to_run), desc=f"GPT-5 panel {split}"):
        paper_id = int(row["Total_ID"])
        try:
            agg = classify_paper(panel, row["Title"], row["Abstract"], use_cache=use_cache)
            tracker.append(paper_id, "done", {
                "Total_ID": paper_id,
                "fields_probs": agg["fields_probs"],
                "levels_probs": agg["levels_probs"],
                "method_probs": agg["method_probs"],
                "n_valid_votes": agg["n_valid_votes"],
                "confidence": agg["confidence"],
            })
        except Exception as e:
            tracker.append(paper_id, "error", {"error": f"{type(e).__name__}: {e}"})
            print(f"\n[ERROR] paper_id={paper_id}: {e}", file=sys.stderr)

    # Build the results DataFrame from the JSONL progress file.
    all_results = tracker.load_results()
    rows = []
    for paper_id, result in all_results.items():
        if result is None:
            continue
        rows.append({
            "Total_ID": paper_id,
            "fields_probs": result.get("fields_probs"),
            "levels_probs": result.get("levels_probs"),
            "method_probs": result.get("method_probs"),
            "n_valid_votes": result.get("n_valid_votes", 0),
            "confidence": result.get("confidence", "low"),
        })

    out = pd.DataFrame(rows)
    if len(out) > 0:
        out.to_parquet(predictions_path(split), index=False)
        print(f"\nSaved {len(out)} GPT-5 predictions to {predictions_path(split)}")

        # Summary diagnostics: how often did each class get majority vote?
        if "fields_probs" in out.columns:
            fields_arr = np.stack(out["fields_probs"].apply(np.asarray).values)
            print("\n=== GPT-5 panel field prevalence (paper rate >= 0.5 majority) ===")
            for i, f in enumerate(config.FIELDS_12):
                rate = float((fields_arr[:, i] >= 0.5).mean())
                print(f"  {f:35s}: {rate * 100:5.1f}%")
        if "levels_probs" in out.columns:
            levels_arr = np.stack(out["levels_probs"].apply(np.asarray).values)
            print("\n=== GPT-5 panel level prevalence (paper rate >= 0.5 majority) ===")
            for i, l in enumerate(config.LEVELS_6):
                rate = float((levels_arr[:, i] >= 0.5).mean())
                print(f"  {l:5s}: {rate * 100:5.1f}%")
        if "method_probs" in out.columns:
            method_arr = np.stack(out["method_probs"].apply(np.asarray).values)
            method_pred = method_arr.argmax(axis=1)
            print("\n=== GPT-5 panel method distribution (argmax) ===")
            for i, m in enumerate(config.METHODS_5):
                rate = float((method_pred == i).mean())
                print(f"  {m:13s}: {rate * 100:5.1f}%")
    else:
        print("No results saved.")
    return out


# ==================== Loader for evaluate.py ====================
def load_panel_predictions(split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load saved GPT-5 panel predictions for use in ensembling.

    Returns:
        (paper_ids, fields_probs[N,12], levels_probs[N,6], method_probs[N,5])
        ordered by paper_ids ascending.

    Raises FileNotFoundError if predictions parquet hasn't been generated yet.
    """
    path = predictions_path(split)
    if not path.exists():
        raise FileNotFoundError(
            f"GPT-5 predictions not found for split={split} at {path}. "
            f"Run: python llm_classify.py --split {split}"
        )
    df = pd.read_parquet(path).sort_values("Total_ID").reset_index(drop=True)
    paper_ids = df["Total_ID"].astype(int).to_numpy()
    fields = np.stack(df["fields_probs"].apply(np.asarray).values).astype(np.float64)
    levels = np.stack(df["levels_probs"].apply(np.asarray).values).astype(np.float64)
    method = np.stack(df["method_probs"].apply(np.asarray).values).astype(np.float64)
    return paper_ids, fields, levels, method


# ==================== CLI ====================
def main():
    parser = argparse.ArgumentParser(description="Phase A: GPT-5 panel full classification")
    parser.add_argument("--split", choices=["val", "test", "both", "train"], default="test",
                        help="Which split to classify")
    parser.add_argument("--limit", type=int, default=None,
                        help="Smoke test: only first N papers")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force re-call API (skip cache)")
    args = parser.parse_args()

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.LLM_LOG_DIR.mkdir(parents=True, exist_ok=True)
    config.LLM_PROGRESS_DIR.mkdir(parents=True, exist_ok=True)

    splits = ["val", "test"] if args.split == "both" else [args.split]
    for split in splits:
        run_split(split, limit=args.limit, use_cache=not args.no_cache)
        if len(splits) > 1:
            print()


if __name__ == "__main__":
    main()
