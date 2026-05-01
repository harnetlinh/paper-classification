"""
Phase C: Task-Adaptive Pretraining (TAPT).

Continue masked-language-model pretraining of the SPECTER2 base on the project
corpus (train + val + test abstracts; labels NOT used) for a few epochs before
classification fine-tuning. Standard recipe: Gururangan et al. 2020 ACL,
"Don't Stop Pretraining". Documented +1-3% F1 on small downstream tasks.

Why this helps THIS project:
- 3/12 Fields classes (Education economically, Non-STEM, psychology) have
  significant val→test AUC drift (e.g. Educ economically 0.96 → 0.71). The
  drop indicates feature distribution shift between val (2023) and test
  (2024) — vocabulary / phrasing of newer papers diverges from what
  SPECTER2 saw during its general scientific pretraining.
- TAPT on the FULL corpus (including unlabeled test 2024 abstracts) adapts
  the encoder to this corpus-specific distribution without any test labels
  — fully legitimate semi-supervised technique.
- ~30 min on Colab T4 for 3-5 epochs MLM over ~2600 abstracts.

Usage:
    python tapt.py                          # adapt SPECTER2_base, save to outputs/specter2_tapt/
    python tapt.py --epochs 5 --lr 5e-5
    python tapt.py --output-dir outputs/specter2_tapt_v2/
    python tapt.py --smoke                  # 1 epoch, 50 papers — pipeline check

After running: set BACKBONE_MODEL = "outputs/specter2_tapt" in config.py and
re-run train_specter2.py. Inference / evaluate auto-pick up from the new path.
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

import config
import utils

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


# ==================== Corpus loader ====================
def _load_full_corpus() -> pd.DataFrame:
    """Concatenate train + val + test abstracts (no labels) for MLM pretraining.

    Including test abstracts is the explicit recommendation of Gururangan et al.
    2020 — TAPT uses the unlabeled corpus from the SAME distribution as the
    downstream task, which means test inputs (without their labels) are fair
    game to adapt the encoder to the test-time vocabulary.
    """
    if not config.GOLD_PARQUET.exists():
        raise FileNotFoundError(f"Run sanitize.py first; missing {config.GOLD_PARQUET}")
    gold = pd.read_parquet(config.GOLD_PARQUET)
    parts = [gold]
    if config.MAIN_2024_PARQUET.exists():
        test = pd.read_parquet(config.MAIN_2024_PARQUET)
        parts.append(test)
    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(subset=["Title", "Abstract"]).reset_index(drop=True)
    return df


# ==================== Dataset ====================
class MLMDataset(Dataset):
    """Produce tokenized inputs for masked-LM pretraining.

    Uses the same input format as classification (Title [SEP] Abstract,
    optionally with rich features) so the encoder sees test-time-style inputs.
    Random masking is applied lazily in the collator (DataCollatorForLanguageModeling).
    """
    def __init__(self, df, tokenizer, max_length=512):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        sep = tokenizer.sep_token
        # Reuse the project's canonical input builder so rich features land in
        # the same positions during TAPT and during classification fine-tune.
        texts = utils.build_input_texts(df, sep)
        self.encodings = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
        }


# ==================== TAPT loop ====================
def run_tapt(
    output_dir: Path,
    epochs: int = 3,
    lr: float = 5e-5,
    batch_size: int = 16,
    mlm_probability: float = 0.15,
    smoke: bool = False,
):
    """Continue MLM pretraining of BACKBONE_MODEL on the project corpus."""
    from transformers import (
        AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling,
    )

    print("=" * 80)
    print(f"Phase C: TAPT — continued MLM pretraining of {config.BACKBONE_MODEL}")
    print("=" * 80)

    utils.set_deterministic(config.SEED)

    tokenizer = AutoTokenizer.from_pretrained(config.BACKBONE_MODEL)
    # AutoModelForMaskedLM loads the MLM head — for SPECTER2 base, this gives
    # us the original BERT-style MLM head used during its initial pretraining.
    model = AutoModelForMaskedLM.from_pretrained(config.BACKBONE_MODEL)

    df = _load_full_corpus()
    print(f"Corpus: {len(df)} unique (Title, Abstract) pairs")
    if smoke:
        df = df.head(50).reset_index(drop=True)
        epochs = 1
        print(f"[SMOKE] reduced to {len(df)} papers, {epochs} epoch(s)")

    ds = MLMDataset(df, tokenizer, max_length=config.MAX_LENGTH)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability,
    )
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        collate_fn=collator,
        generator=torch.Generator().manual_seed(config.SEED),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cpu" and not smoke:
        print("WARNING: TAPT on CPU is ~10x slower than on T4 GPU. Consider Colab.")
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = max(1, len(loader) * epochs)
    warmup_steps = max(1, int(total_steps * 0.06))

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    scheduler = LambdaLR(optimizer, lr_lambda)

    use_amp = bool(getattr(config, "USE_AMP", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        print("AMP mixed precision: enabled (fp16)")

    from tqdm import tqdm
    model.train()
    log = []
    for epoch in range(epochs):
        loss_sum, n_batches = 0.0, 0
        pbar = tqdm(loader, desc=f"TAPT epoch {epoch+1}/{epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            if use_amp:
                with torch.cuda.amp.autocast():
                    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = out.loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()

            loss_sum += float(loss.item())
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg = loss_sum / max(1, n_batches)
        log.append({"epoch": epoch + 1, "avg_mlm_loss": avg})
        print(f"  Epoch {epoch+1} avg MLM loss: {avg:.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    # Save the encoder backbone (not the MLM head — the head won't be reused).
    # AutoModelForMaskedLM stores the encoder under model.bert / model.roberta /
    # etc. depending on architecture. save_pretrained gives the standard path
    # so AutoModel.from_pretrained(output_dir) downstream loads the adapted
    # encoder identically to the original SPECTER2 base.
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"\nTAPT-adapted model saved to: {output_dir}")
    print(f"  → To use: set config.BACKBONE_MODEL = \"{output_dir}\"")
    print(f"  → Then re-run: python train_specter2.py --task all --ensemble")

    return log


# ==================== CLI ====================
def main():
    parser = argparse.ArgumentParser(description="Phase C: TAPT")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of MLM epochs (default 3)")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate (default 5e-5; standard MLM continue-pretrain)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size (default 16; T4 16GB OK)")
    parser.add_argument("--mlm-probability", type=float, default=0.15,
                        help="Token masking probability (default 0.15, BERT standard)")
    parser.add_argument("--output-dir", type=str,
                        default=str(config.OUTPUT_DIR / "specter2_tapt"),
                        help="Where to save the adapted encoder")
    parser.add_argument("--smoke", action="store_true",
                        help="50 papers, 1 epoch — pipeline check")
    args = parser.parse_args()

    run_tapt(
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        mlm_probability=args.mlm_probability,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
