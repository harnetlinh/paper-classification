# Paper Classification — Vietnam Educational Research Bibliometrics

End-to-end pipeline phân loại Scopus papers theo 3 trục:
- **Fields** (multi-label, 12 lớp): teaching & learning, technology in education, English education, …
- **Levels** (multi-label, 6 lớp): ECE, GE, HE, TVET, LLL, ALL
- **Method** (single-label, 5 lớp): Quantitative, Qualitative, Mixed, Review, Other

Backbone là `allenai/specter2_base` fine-tuned trên gold labels 2013–2023 (~2074 papers), test trên 2024 (562 papers).

## Quickstart trên Colab T4 (khuyến nghị)

GPU T4 free chạy hết 3 task trong ~30–45 phút. CPU mất ~7 giờ.

1. Mở [colab_train.ipynb](colab_train.ipynb) trong Google Colab.
2. **Runtime → Change runtime type → T4 GPU**.
3. Chạy lần lượt từ trên xuống. Khi cell upload data hỏi file, chọn file Excel gốc.
4. Cell cuối tải `outputs.zip` về máy (chứa model `.pt` + thresholds + eval report).

Notebook đã đóng gói: clone repo → install deps → upload Excel → sanitize → train all → evaluate → download.

## Chạy local (CPU hoặc GPU)

```bash
git clone https://github.com/harnetlinh/paper-classification.git
cd paper-classification
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 1. Sanitize raw Excel
mkdir -p data outputs/llm_logs outputs/llm_progress
cp /path/to/your/scopus_2013_2024.xlsx data/
python sanitize.py

# 2. (Optional) Re-augment rare classes via LLM ensemble (~$2.7)
# Skip if outputs/{special_edu,ece,tvet,lll}_augmented.parquet are present.
echo "OPENAI_API_KEY=sk-..." > .env
python llm_augment.py --class all

# 3. Train 3 models
python train_specter2.py --task all

# 4. Evaluate
python evaluate.py --task all
cat outputs/eval_report.json | python -m json.tool | less

# 5. Export human-review workbook (compare machine vs gold/human labels)
python export_review.py --output outputs/review.xlsx
# Open outputs/review.xlsx in Excel — sheet `summary_test_2024` is sorted
# disagreements-first, sheet `disagreements_test_2024` is filtered to only
# papers needing human review.
```

## Pipeline tổng quan

```
Excel (2013–2024)
   │
   ▼ sanitize.py        ── normalize text/methods/levels/fields, drop research-only,
   │                       recover 417 NaN Total_IDs via title-lookup against main sheet
   ├──► outputs/gold_2013_2023.parquet      (2074 rows, training + val split by Year)
   └──► outputs/main_2024_clean.parquet     (562 rows, held-out test set)

Rare-class boost (one-time, ~$2.7 in OpenAI cost):
   llm_augment.py --class all
      │
      ▼ 3-model unanimous-vote ensemble (gpt-5.5 + gpt-5.4 + gpt-5.4-mini)
      ├──► special_edu_augmented.parquet   (Fields task — Special education: 21 YES of 46)
      ├──► ece_augmented.parquet           (Levels — ECE: 7/13)
      ├──► tvet_augmented.parquet          (Levels — TVET: 17/46)
      └──► lll_augmented.parquet           (Levels — LLL: 57/115)
                    Augmentation is additive and applied ONLY to TRAIN years (2013–2022)
                    so val 2023 stays gold-truth for honest measurement.

train_specter2.py
   │  AdamW + linear warmup (6%) decay, AMP fp16 on GPU,
   │  AsymmetricLoss for fields/levels (or weighted BCE via config),
   │  Focal CE for method, gradient clipping max_norm=1.0,
   │  best-model selection by macro_AUC (threshold-independent),
   │  robust per-class threshold tuning: F1-grid where val support ≥ 10,
   │  Youden's J fallback otherwise.
   │
   ├──► outputs/model_fields.pt + thresholds_fields.json
   ├──► outputs/model_levels.pt + thresholds_levels.json
   └──► outputs/model_method.pt

evaluate.py
   │
   ▼ Macro F1, micro F1, weighted F1, macro AUC, macro AP per task
     + supported_macro_f1 (classes with support ≥ 5) and high_support_macro_f1 (≥ 30)
     + per-class table (val_2023 + test_2024)
     + drift_gap (val − test macro F1)
     + low-support warnings for classes with val support < 5
   └──► outputs/eval_report.json

export_review.py
   │
   ▼ Side-by-side machine-vs-human label comparison for human review.
     Auto-detects ensemble checkpoints. Builds a multi-sheet Excel with:
     - summary_{split}: 1 row/paper, all 3 tasks side-by-side, Jaccard,
       review_priority (HIGH/MEDIUM/LOW/OK) sorted disagreements-first
     - disagreements_{split}: filtered to papers needing review
     - {fields,levels,method}_{split}: per-class probability + status
       (TP/FP/FN/TN) so reviewers can filter by error type
     - stats_{split}_{task}: per-class precision / recall / F1 / support
     - legend: column glossary
   └──► outputs/review.xlsx
```

## Cấu trúc

```
paper-classification/
├── colab_train.ipynb           # ← Colab one-click pipeline (run this)
├── README.md                   # ← This file
├── requirements.txt
├── .env.example                # OPENAI_API_KEY template (only needed for re-augment)
│
├── config.py                   # All knobs: LR, EPOCHS, BATCH_SIZE, loss type, panels
├── sanitize.py                 # Phase 0: Excel → parquet
├── prompts.py                  # LLM prompt templates (codebook v2.1)
├── openai_clients.py           # 3-model OpenAI ensemble + cache
├── llm_augment.py              # Phase 1: rare-class augmentation
├── utils.py                    # Loss functions, dataset, threshold tuning
├── train_specter2.py           # Phase 2: train 3 SPECTER2 fine-tunes
├── evaluate.py                 # Phase 3: dual val/test eval
├── inference.py                # Phase 4: predict on new Excel
├── smoke_test.py               # 31 unit + integration tests
│
├── data/                       # Place input Excel here
└── outputs/                    # Generated artifacts
    ├── gold_2013_2023.parquet
    ├── main_2024_clean.parquet
    ├── *_augmented.parquet
    ├── model_*.pt              # ~440 MB each
    ├── thresholds_*.json
    ├── training_log_*.json
    ├── eval_report.json
    └── llm_logs/, llm_progress/
```

## Performance levers đã apply

| # | Lever | Trước | Sau | Lý do |
|---|---|---|---|---|
| 1 | Recover Total_ID | 417 NaN rows excluded | 100% recovered via title lookup | Mất 20% gold data |
| 2 | LLM augment Special edu | 17 train | 35 train (+19) | Quá hiếm để học |
| 3 | LLM augment ECE/TVET/LLL | 48/28/6 train | 53/44/51 train | LLL train tăng 8.5x |
| 4 | Augment chỉ TRAIN years | val nhiễm LLM labels | val giữ gold-truth | Honest val_F1 |
| 5 | Gradient clipping max_norm=1 | none | 1.0 | Standard transformer fine-tune |
| 6 | Threshold grid step | 0.05 (17 candidates) | 0.02 (41 candidates) | Sharper decision boundary |
| 7 | Threshold tuning robust | F1-grid for all classes | Youden's J for low-support | Avoid tuning noise on 1-9 samples |
| 8 | LR | 1e-5 | 2e-5 | SPECTER2 chuẩn 2-5e-5 |
| 9 | EPOCHS | 5 | 10 | Train loss vẫn giảm ở epoch 5 |
| 10 | BATCH_SIZE | 16 | 32 | T4 đủ memory, gradient ổn hơn |
| 11 | Mixed precision (AMP) | off | fp16 on GPU | 2-3x throughput |
| 12 | Dropout | 0.1 | 0.2 | Combat overfit trên small data |
| 13 | Best-model selection | macro_F1@0.5 | macro_AUC | Threshold-independent, support-robust |
| 14 | Loss option | AsymmetricLoss only | AsymmetricLoss \| BCE+pos_weight | Calibrated outputs option |
| 15 | Focal CE for Method | weighted CE | Focal γ=2 | Rare class "Other" boost |
| 16 | Eval metrics | F1 only | F1 + AUC + AP + low-support warnings | Trust AUC khi support < 5 |

## Metrics dự kiến (sau khi train với hyperparameters mới trên T4)

Baseline đã chạy CPU với LR=1e-5, EPOCHS=5:

| Task | Test F1 (cũ) | Test AUC (cũ) | F1 mục tiêu |
|---|---|---|---|
| Fields | 0.405 | 0.770 | ≥ 0.55 |
| Levels | 0.464 | 0.817 | ≥ 0.60 |
| Method | 0.599 | 0.921 | ≥ 0.70 |

**Lưu ý:** test 2024 có distribution shift đáng kể với một số class rare (psychology in education val=71 → test=3). Macro F1 cho Fields có giới hạn cấu trúc do annotator drift, không thể đẩy qua mức ~0.6 chỉ bằng model improvement. Trust **macro AUC** + **per-class table** + **eval cho từng class có support ≥ 10**.

## Yêu cầu hệ thống

- Python 3.10+
- 8 GB RAM (CPU) hoặc T4 GPU 16 GB (Colab free)
- ~3 GB disk cho models + parquet outputs
- (Optional) OPENAI_API_KEY cho re-augment LLM

## License

MIT.
