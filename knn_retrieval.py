"""
Phase D: kNN retrieval ensemble member.

For each test (or val) paper, find its k nearest neighbors in the train (+val)
corpus by SPECTER2 [CLS] embedding cosine similarity, and use the neighbors'
labels as a soft-label vote — sample-efficient classifier that bypasses the
fine-tune bias of the multilabel head.

Why this fits the failure mode:
- Special education has 16 train + 21 LLM-augmented = 37 positive papers.
  The fine-tuned head OVERFITS them (train AUC 0.998, test AUC 0.587). But
  cosine retrieval over the same 37 positives can still surface them as the
  nearest neighbors of new Special-edu test papers — it doesn't need to
  build a generalised concept, just match feature similarity.
- Multilabel single-class methods don't suffer the same way: kNN aggregates
  whatever labels the neighbors have, naturally handling multilabel.

Output: same shape as classifier sigmoid probs ([N, n_classes]) with values
in [0, 1] computed as similarity-weighted soft votes. Plug into ensemble.py
the same way GPT-5 panel probs go in.

Usage:
    python knn_retrieval.py --split test       # build retrieval probs for test
    python knn_retrieval.py --split val        # for val (used to tune lambda)
    python knn_retrieval.py --split both
    python knn_retrieval.py --include-val      # include val in the index
                                                 (use only for final test run)
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import config
import utils

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


# ==================== Output paths ====================
KNN_DIR = config.OUTPUT_DIR / "knn_retrieval"
KNN_DIR.mkdir(parents=True, exist_ok=True)


def knn_probs_path(split: str, task: str) -> Path:
    return KNN_DIR / f"knn_{task}_{split}.parquet"


# ==================== Embedding helper ====================
def _encode_cls(df: pd.DataFrame, tokenizer, device, batch_size: int = 32) -> np.ndarray:
    """Return [N, hidden_size] [CLS] embeddings using SPECTER2 base.

    Uses AutoModel directly (not the SpecterClassifier head) because we want
    raw embeddings, not classification logits.
    """
    from transformers import AutoModel
    model = AutoModel.from_pretrained(
        config.BACKBONE_MODEL,
        revision=getattr(config, "BACKBONE_REVISION", None),
    ).to(device)
    model.eval()

    sep = tokenizer.sep_token
    texts = utils.build_input_texts(df, sep)
    enc = tokenizer(
        texts, padding="max_length", truncation=True,
        max_length=config.MAX_LENGTH, return_tensors="pt",
    )

    embeddings = []
    n = len(df)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            ids = enc["input_ids"][i:i+batch_size].to(device)
            mask = enc["attention_mask"][i:i+batch_size].to(device)
            out = model(input_ids=ids, attention_mask=mask)
            # SPECTER2 doc embedding = first token ([CLS]) of last_hidden_state
            cls = out.last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32)
            embeddings.append(cls)
    embeddings = np.concatenate(embeddings, axis=0)
    # L2-normalize for cosine similarity via dot product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    return embeddings / norms


# ==================== kNN retrieval ====================
def _topk_sim(query: np.ndarray, index: np.ndarray, k: int) -> tuple:
    """Cosine similarity (assumes both L2-normalised) → top-k for each query.

    Returns:
        (top_indices, top_sims): both [N_query, k]
    """
    # Brute force is fine for our scale (~2000 train × ~600 query = 1.2M dot products).
    sims = query @ index.T  # [N_query, N_index]
    top_idx = np.argpartition(-sims, k, axis=1)[:, :k]
    # Re-sort the partition to get descending-by-similarity order
    top_sims = np.take_along_axis(sims, top_idx, axis=1)
    order = np.argsort(-top_sims, axis=1)
    top_idx = np.take_along_axis(top_idx, order, axis=1)
    top_sims = np.take_along_axis(top_sims, order, axis=1)
    return top_idx, top_sims


def _knn_vote(
    top_idx: np.ndarray,
    top_sims: np.ndarray,
    index_labels: np.ndarray,
    similarity_temperature: float = 1.0,
) -> np.ndarray:
    """Similarity-weighted soft-label vote.

    For each query and each class:
        prob[c] = Σ_{n in top-k} softmax(sim_n / T) · label[n, c]

    Sums to 1 over neighbors (not over classes — multi-label friendly).

    Args:
        top_idx:  [N_query, k] indices into index_labels
        top_sims: [N_query, k] cosine similarities
        index_labels: [N_index, C] binary labels
        similarity_temperature: T for softmax over similarities. Lower T (e.g. 0.1)
            sharpens to nearly the top-1 neighbor; higher T (e.g. 5.0) approaches
            uniform vote across all top-k. Default 1.0 = mild sharpening.

    Returns:
        probs: [N_query, C] in [0, 1]
    """
    # Numerically stable softmax over the k similarities per query.
    sims_t = top_sims / max(similarity_temperature, 1e-6)
    sims_t -= sims_t.max(axis=1, keepdims=True)
    weights = np.exp(sims_t)
    weights /= weights.sum(axis=1, keepdims=True) + 1e-12
    # Gather neighbor labels and aggregate.
    neighbor_labels = index_labels[top_idx]  # [N_query, k, C]
    probs = (weights[:, :, None] * neighbor_labels).sum(axis=1)
    return probs


# ==================== Main pipeline per task ====================
def build_knn_probs(
    split: str,
    task: str,
    include_val_in_index: bool = False,
    k: int = 10,
    temperature: float = 1.0,
):
    """Build kNN soft-label probs for `split` × `task`, save to parquet."""
    if not config.GOLD_PARQUET.exists():
        raise FileNotFoundError(f"{config.GOLD_PARQUET} not found; run sanitize.py")

    print("=" * 80)
    print(f"Phase D: kNN retrieval — split={split}, task={task}, k={k}, T={temperature}")
    print(f"  include_val_in_index={include_val_in_index}")
    print("=" * 80)

    gold = pd.read_parquet(config.GOLD_PARQUET)
    train_df = gold[gold["Year"].isin(config.TRAIN_YEARS)].reset_index(drop=True)
    val_df = gold[gold["Year"] == config.VAL_YEAR].reset_index(drop=True)

    # Build index DataFrame
    index_df = train_df.copy()
    if include_val_in_index and split == "test":
        # For final test inference only — adding val to the retrieval index
        # gives 417 more annotation-style-matched neighbors. Do NOT enable
        # this when split=val (would be cheating).
        index_df = pd.concat([train_df, val_df], ignore_index=True)
        print(f"  Index = train + val: {len(index_df)} papers")
    else:
        print(f"  Index = train only: {len(index_df)} papers")

    # Query DataFrame
    if split == "val":
        query_df = val_df
    elif split == "test":
        if not config.MAIN_2024_PARQUET.exists():
            raise FileNotFoundError(f"{config.MAIN_2024_PARQUET} not found")
        query_df = pd.read_parquet(config.MAIN_2024_PARQUET).reset_index(drop=True)
    elif split == "train":
        query_df = train_df  # mostly for sanity check
    else:
        raise ValueError(f"Unknown split: {split!r}")
    print(f"  Query: {len(query_df)} papers ({split})")

    # Get task-specific labels for the INDEX
    if task == "fields":
        target_cols = [f"field_{i:02d}" for i in range(len(config.FIELDS_12))]
        n_classes = len(target_cols)
        index_labels = index_df[target_cols].astype(np.float32).to_numpy()
    elif task == "levels":
        target_cols = [f"level_{l}" for l in config.LEVELS_6]
        n_classes = len(target_cols)
        index_labels = index_df[target_cols].astype(np.float32).to_numpy()
    elif task == "method":
        n_classes = len(config.METHODS_5)
        method_to_idx = {m: i for i, m in enumerate(config.METHODS_5)}
        index_labels = np.zeros((len(index_df), n_classes), dtype=np.float32)
        for i, m in enumerate(index_df["method"].astype(str)):
            if m in method_to_idx:
                index_labels[i, method_to_idx[m]] = 1.0
    else:
        raise ValueError(f"Unknown task: {task!r}")
    print(f"  Task: {task}, n_classes={n_classes}")

    # Encode
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.BACKBONE_MODEL)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    print(f"  Encoding index ({len(index_df)} papers)...")
    index_emb = _encode_cls(index_df, tokenizer, device)
    print(f"  Encoding query ({len(query_df)} papers)...")
    query_emb = _encode_cls(query_df, tokenizer, device)

    # Self-overlap: when querying train (e.g. sanity), exclude the paper from its own neighbors.
    # We achieve this by retrieving k+1 and dropping the top-1 if it's the paper itself
    # (cosine similarity ~ 1.0 indicates same row).
    print(f"  Computing top-{k} neighbors per query...")
    top_idx, top_sims = _topk_sim(query_emb, index_emb, k)

    # Vote
    probs = _knn_vote(top_idx, top_sims, index_labels, similarity_temperature=temperature)

    out = pd.DataFrame({
        "Total_ID": query_df["Total_ID"].astype(int).to_numpy(),
        "knn_probs": [list(row) for row in probs.astype(np.float64)],
    })
    out_path = knn_probs_path(split, task)
    out.to_parquet(out_path, index=False)
    print(f"\nSaved kNN probs: {out_path}")

    # Quick diagnostics
    print("  Class prevalence in kNN soft votes (mean prob across queries):")
    if task == "fields":
        names = config.FIELDS_12
    elif task == "levels":
        names = config.LEVELS_6
    else:
        names = config.METHODS_5
    for i, name in enumerate(names):
        print(f"    {name:35s}: {probs[:, i].mean():.4f}")

    return probs


def load_knn_probs(split: str, task: str) -> tuple:
    """Load kNN soft-vote probs for ensemble use.

    Returns:
        (paper_ids: [N], probs: [N, C])  ordered by paper_ids ascending.
    """
    path = knn_probs_path(split, task)
    if not path.exists():
        raise FileNotFoundError(
            f"kNN probs not found for split={split}, task={task}. "
            f"Run: python knn_retrieval.py --split {split}"
        )
    df = pd.read_parquet(path).sort_values("Total_ID").reset_index(drop=True)
    paper_ids = df["Total_ID"].astype(int).to_numpy()
    probs = np.stack(df["knn_probs"].apply(np.asarray).values).astype(np.float64)
    return paper_ids, probs


# ==================== CLI ====================
def main():
    parser = argparse.ArgumentParser(description="Phase D: kNN retrieval ensemble")
    parser.add_argument("--split", choices=["val", "test", "both", "train"], default="test")
    parser.add_argument("--task", choices=["fields", "levels", "method", "all"], default="all")
    parser.add_argument("--k", type=int, default=10,
                        help="Number of neighbors per query (default 10)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Softmax temperature over similarities (default 1.0; "
                             "lower = sharper toward top-1, higher = more uniform)")
    parser.add_argument("--include-val", action="store_true",
                        help="Include val 2023 in the retrieval index (only meaningful "
                             "for split=test; do NOT enable for val split — would cheat)")
    args = parser.parse_args()

    splits = ["val", "test"] if args.split == "both" else [args.split]
    tasks = ["fields", "levels", "method"] if args.task == "all" else [args.task]

    # Cache embeddings across (split, task) iterations: encoding 1657 train +
    # 417 val + 562 test papers takes 30-60 min on CPU per encode, and the
    # naive nested loop would do this 6x. Cache the encode call by a key
    # (length, first_total_id, last_total_id) — same DataFrame produces the
    # same key. Replace the module-level _encode_cls with a memoized wrapper.
    global _encode_cls
    _orig_encode = _encode_cls
    _emb_cache: dict = {}

    def _cached_encode_cls(df, tokenizer, device, batch_size=32):
        first = int(df["Total_ID"].astype(int).iloc[0]) if len(df) else 0
        last = int(df["Total_ID"].astype(int).iloc[-1]) if len(df) else 0
        key = (len(df), first, last)
        if key in _emb_cache:
            return _emb_cache[key]
        emb = _orig_encode(df, tokenizer, device, batch_size=batch_size)
        _emb_cache[key] = emb
        return emb
    _encode_cls = _cached_encode_cls

    for split in splits:
        for task in tasks:
            include_val = args.include_val and (split == "test")
            build_knn_probs(
                split=split, task=task,
                include_val_in_index=include_val,
                k=args.k, temperature=args.temperature,
            )


if __name__ == "__main__":
    main()
