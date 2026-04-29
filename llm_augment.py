"""
Phase 1: LLM-based label augmentation for rare classes
======================================================

Originally built for "Special education" (Fields task, multi-label, 17 gold
positives). Extended to ECE, TVET, LLL (Levels task, multi-label) — same
3-model unanimous-vote ensemble, parameterized by class spec.

Method (single-label, 5 classes) is intentionally NOT augmented: switching a
paper's existing single label is destructive, whereas multi-label augmentation
is purely additive.

Pipeline (per class):
1. Pre-filter gold papers by class-specific keywords
2. For each candidate, ask 3 OpenAI models: "Does this paper match <class>?"
3. Aggregate by majority vote → unanimous_yes / majority_yes / split / all_failed
4. Apply augmentation in train_specter2.py (TRAIN years only, additive labels)

Resume support:
- Cache by (model, prompt) hash → re-running is FREE
- Progress JSONL file (one per class) → can interrupt anytime, resumes

Usage:
    python llm_augment.py                                  # default: Special education
    python llm_augment.py --class "Special education"
    python llm_augment.py --class ECE
    python llm_augment.py --class TVET
    python llm_augment.py --class LLL
    python llm_augment.py --class all                      # run all 4 in sequence
    python llm_augment.py --class ECE --limit 5            # smoke
    python llm_augment.py --no-cache                       # force re-call API
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict

import pandas as pd
from tqdm import tqdm

import config
import openai_clients
import prompts


# ==================== Helpers ====================
def matches_keywords(text, keywords):
    """Return True if text contains any of the given keywords (case-insensitive)."""
    if not isinstance(text, str) or not text:
        return False
    t = text.lower()
    return any(kw.lower() in t for kw in keywords)


def matches_special_edu_keywords(text: str) -> bool:
    """Backward-compatible wrapper used by smoke_test.py."""
    return matches_keywords(text, config.SPECIAL_EDU_KEYWORDS)


def coerce_bool(v):
    """Robustly cast LLM JSON value to True/False/None.

    Some models occasionally emit "true"/"false" strings, "yes"/"no", or 1/0
    instead of the strict JSON boolean schema we asked for. Treat those as
    valid signals so a single odd response doesn't silently get marked failed.
    Anything else (None, unrecognized) returns None.
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


def _aggregate_votes(votes_per_llm: List[Dict], response_field: str) -> Dict:
    """Aggregate votes from N LLMs reading `response_field` from each vote dict.

    Returns: {n_yes, n_no, n_failed, consensus, agreement, confidence}
    """
    yes, no, failed = 0, 0, 0
    for v in votes_per_llm:
        if not isinstance(v, dict) or "_error" in v:
            failed += 1
            continue
        is_match = coerce_bool(v.get(response_field))
        if is_match is True:
            yes += 1
        elif is_match is False:
            no += 1
        else:
            failed += 1

    valid = yes + no
    if valid == 0:
        return {
            "n_yes": 0, "n_no": 0, "n_failed": failed,
            "consensus": False, "agreement": "all_failed", "confidence": "low",
        }

    consensus = yes > no
    if yes == valid:
        agreement = "unanimous_yes"
        confidence = "high"
    elif no == valid:
        agreement = "unanimous_no"
        confidence = "high"
    else:
        agreement = "split"
        confidence = "medium" if consensus else "low"

    return {
        "n_yes": yes, "n_no": no, "n_failed": failed,
        "consensus": consensus, "agreement": agreement, "confidence": confidence,
    }


def aggregate_special_edu_votes(votes_per_llm: List[Dict]) -> Dict:
    """Backward-compatible wrapper used by smoke_test.py."""
    return _aggregate_votes(votes_per_llm, response_field="is_special_education")


# ==================== Class registry ====================
def _build_specs():
    """Class-to-spec registry. Built lazily so that import order is safe."""
    sp_idx = config.FIELDS_12.index("Special education")
    return {
        "Special education": {
            "task": "fields",
            "binary_col": f"field_{sp_idx:02d}",
            "label_value": "Special education",
            "list_col": "fields_list",
            "output_path": config.SPECIAL_EDU_AUGMENT,
            "keywords": config.SPECIAL_EDU_KEYWORDS,
            "progress_task_name": "special_edu_augment",
            "prompt_fn": lambda title, abstract: prompts.make_special_edu_filter_prompt(title, abstract),
            "response_field": "is_special_education",
            "gold_positive_count": None,  # populated at run time
        },
        "ECE": {
            "task": "levels",
            "binary_col": "level_ECE",
            "label_value": "ECE",
            "list_col": "levels_list",
            "output_path": config.LEVEL_AUGMENT_OUTPUTS["ECE"],
            "keywords": config.LEVEL_AUGMENT_KEYWORDS["ECE"],
            "progress_task_name": "ece_augment",
            "prompt_fn": lambda title, abstract: prompts.make_level_filter_prompt("ECE", title, abstract),
            "response_field": "is_match",
            "gold_positive_count": None,
        },
        "TVET": {
            "task": "levels",
            "binary_col": "level_TVET",
            "label_value": "TVET",
            "list_col": "levels_list",
            "output_path": config.LEVEL_AUGMENT_OUTPUTS["TVET"],
            "keywords": config.LEVEL_AUGMENT_KEYWORDS["TVET"],
            "progress_task_name": "tvet_augment",
            "prompt_fn": lambda title, abstract: prompts.make_level_filter_prompt("TVET", title, abstract),
            "response_field": "is_match",
            "gold_positive_count": None,
        },
        "LLL": {
            "task": "levels",
            "binary_col": "level_LLL",
            "label_value": "LLL",
            "list_col": "levels_list",
            "output_path": config.LEVEL_AUGMENT_OUTPUTS["LLL"],
            "keywords": config.LEVEL_AUGMENT_KEYWORDS["LLL"],
            "progress_task_name": "lll_augment",
            "prompt_fn": lambda title, abstract: prompts.make_level_filter_prompt("LLL", title, abstract),
            "response_field": "is_match",
            "gold_positive_count": None,
        },
    }


# ==================== Main task ====================
def run_augment(class_name: str, limit: int = None, use_cache: bool = True):
    """Run augmentation for one class via the LLM ensemble."""
    specs = _build_specs()
    if class_name not in specs:
        raise ValueError(f"Unknown class {class_name!r}. Choose from {list(specs.keys())}")
    spec = specs[class_name]

    print("=" * 80)
    print(f"Phase 1: LLM Augmentation — {class_name}  (task={spec['task']})")
    print("=" * 80)

    if not config.GOLD_PARQUET.exists():
        print(f"ERROR: Gold dataset not found at {config.GOLD_PARQUET}")
        print("Run sanitize.py first.")
        sys.exit(1)

    gold = pd.read_parquet(config.GOLD_PARQUET)
    print(f"Gold dataset: {len(gold)} papers")

    # Identify already-labeled (we will not relabel them — augmentation is additive)
    already_labeled = gold[spec["list_col"]].apply(
        lambda lst: spec["label_value"] in (list(lst) if lst is not None else [])
    )
    n_already = int(already_labeled.sum())
    print(f"Already labeled {class_name}: {n_already} (preserved, not re-verified)")

    # Pre-filter candidates by keyword
    title_match = gold["Title"].apply(lambda t: matches_keywords(t, spec["keywords"]))
    abs_match = gold["Abstract"].apply(lambda a: matches_keywords(a, spec["keywords"]))
    candidates_mask = (~already_labeled) & (title_match | abs_match)
    candidates = gold[candidates_mask].copy()
    print(f"Keyword candidates (not yet labeled): {len(candidates)}")

    if len(candidates) == 0:
        print("No candidates to verify. Done.")
        return

    if limit and limit < len(candidates):
        candidates = candidates.head(limit)
        print(f"--limit applied: only verifying first {limit} candidates")

    panel = openai_clients.get_clients()
    cost_est = openai_clients.estimate_cost(panel, len(candidates))
    print(f"\nLLM Panel: {[c.model for c in panel]}")
    print(f"Estimated cost: ${cost_est['total_usd']:.4f}")
    print(f"Total API calls: {len(candidates) * len(panel)}\n")

    tracker = openai_clients.ProgressTracker(task_name=spec["progress_task_name"])
    done_ids = tracker.load_done_ids()
    print(f"Already processed (resume): {len(done_ids)} papers")

    candidates_to_run = candidates[~candidates["Total_ID"].isin(done_ids)].copy()
    print(f"Remaining to process: {len(candidates_to_run)}\n")

    response_field = spec["response_field"]

    for _, row in tqdm(candidates_to_run.iterrows(), total=len(candidates_to_run),
                        desc=f"Verify {class_name}"):
        paper_id = int(row["Total_ID"])
        try:
            sys_p, user_p = spec["prompt_fn"](row["Title"], row["Abstract"])
            votes = []
            for client in panel:
                resp = client.classify(sys_p, user_p, use_cache=use_cache)
                votes.append({
                    "model": client.alias,
                    "raw": resp,
                    response_field: coerce_bool(resp.get(response_field)) if "_error" not in resp else None,
                    "reasoning": resp.get("reasoning", "") if "_error" not in resp else resp.get("_error", ""),
                    "confidence": resp.get("confidence", "") if "_error" not in resp else "low",
                })

            agg = _aggregate_votes(votes, response_field)
            result = {
                "Total_ID": paper_id,
                "Title": (row["Title"] or "")[:300],
                "abstract_preview": (row["Abstract"] or "")[:500],
                "votes": votes,
                **agg,
            }
            tracker.append(paper_id, "done", result)
        except Exception as e:
            tracker.append(paper_id, "error", {"error": f"{type(e).__name__}: {e}"})
            print(f"\n[ERROR] paper_id={paper_id}: {e}", file=sys.stderr)

    all_results = tracker.load_results()
    print(f"\nTotal processed: {len(all_results)}")

    rows = []
    for paper_id, result in all_results.items():
        if result is None:
            continue
        rows.append({
            "Total_ID": paper_id,
            "Title": result.get("Title", ""),
            "abstract_preview": result.get("abstract_preview", ""),
            "n_yes": result.get("n_yes", 0),
            "n_no": result.get("n_no", 0),
            "n_failed": result.get("n_failed", 0),
            "consensus": result.get("consensus", False),
            "agreement": result.get("agreement", ""),
            "confidence": result.get("confidence", ""),
            "votes_json": json.dumps(result.get("votes", []), ensure_ascii=False),
        })

    df_out = pd.DataFrame(rows)
    if len(df_out) > 0:
        df_out.to_parquet(spec["output_path"], index=False)

        n_unanimous = int((df_out["agreement"] == "unanimous_yes").sum())
        n_majority = int(((df_out["agreement"] == "split") & (df_out["consensus"] == True)).sum())
        n_split_no = int(((df_out["agreement"] == "split") & (df_out["consensus"] == False)).sum())
        n_unanimous_no = int((df_out["agreement"] == "unanimous_no").sum())
        n_failed = int((df_out["agreement"] == "all_failed").sum())

        print(f"\n=== {class_name} augmentation results ===")
        print(f"Total candidates verified:  {len(df_out)}")
        print(f"  Unanimous YES (3/3):      {n_unanimous}  will be added to gold (train years only)")
        print(f"  Majority YES (2/3):       {n_majority}  (skipped, conservative)")
        print(f"  Majority NO (1/3 yes):    {n_split_no}  (skipped)")
        print(f"  Unanimous NO (0/3):       {n_unanimous_no}  (skipped)")
        print(f"  All failed:               {n_failed}")
        print(f"\nFinal {class_name} count after augment: {n_already} -> up to {n_already + n_unanimous}")
        print(f"  (Actual gain in train_df depends on TRAIN-year filter applied at training time.)")
        print(f"\nSaved to: {spec['output_path']}")
    else:
        print("No results saved.")


# ==================== Backward-compat alias ====================
def augment_special_education(limit: int = None, use_cache: bool = True):
    """Backward-compatible alias retained for existing callers / docs."""
    return run_augment("Special education", limit=limit, use_cache=use_cache)


# ==================== Entry point ====================
def main():
    specs = _build_specs()
    parser = argparse.ArgumentParser(description="LLM ensemble augmentation for rare classes")
    parser.add_argument(
        "--class", dest="class_name",
        choices=list(specs.keys()) + ["all"],
        default="Special education",
        help="Target class to augment. 'all' runs every class in registry order.",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Smoke test mode: only process N candidates")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force re-call API (skip cache)")
    args = parser.parse_args()

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.LLM_LOG_DIR.mkdir(parents=True, exist_ok=True)
    config.LLM_PROGRESS_DIR.mkdir(parents=True, exist_ok=True)

    classes = list(specs.keys()) if args.class_name == "all" else [args.class_name]
    for cn in classes:
        run_augment(cn, limit=args.limit, use_cache=not args.no_cache)
        if len(classes) > 1:
            print()


if __name__ == "__main__":
    main()
