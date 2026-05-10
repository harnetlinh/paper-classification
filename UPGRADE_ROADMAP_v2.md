# UPGRADE ROADMAP v2 — Pipeline KHGDVN

**Phân tích critical, không hype, không thấy mới là upgrade**

| Mục | Chi tiết |
|---|---|
| Tác giả | NCS. Hà Ngọc Linh (HUST) |
| Hướng dẫn | PGS.TS. Phạm Thị Thanh Hải |
| Phiên bản | v2 — sửa lại sau review critical, 10/05/2026 |
| Thay thế | Bản v1 (UPGRADE_ROADMAP.md) đã bị deprecated do thiếu critical analysis |

---

## TÓM TẮT QUYẾT ĐỊNH

Sau phân tích lại với context KHGDVN cụ thể (dataset nhỏ vài chục nghìn papers, temporal split, tiếng Việt, đã có nhiều cơ chế advanced), kết quả khác hẳn v1:

| Upgrade | v1 priority | v2 quyết định | Lý do thay đổi |
|---|---|---|---|
| A — Source title masking 75% | HIGH | **MODIFY**: giảm xuống 0.30 với curriculum | 0.75 quá aggressive cho dataset nhỏ — phù hợp Crossref triệu papers, không phải KHGDVN |
| B — Metadata ablation diagnostic | HIGH | **KEEP**, gộp vào drift report | Diagnostic, không gây hại |
| C — Class-wise F1 analysis | HIGH | **KEEP, mở rộng** với drift columns | Essential, nhưng phải drift-aware |
| D — Frequency-stratified eval | HIGH | **GỘP vào C** | Chỉ là aggregate view của C — file riêng là redundancy |
| E — Yang-format comparison | MEDIUM | **KEEP** | Low-cost reporting cho hội đồng |
| F — α weight scaling option | MEDIUM | **SKIP** | KHGDVN auto pos_weight ≡ Yang α=1 normalized — thêm flag không value |
| G — Co-occurrence analysis | MEDIUM | **SKIP** | Low-actionable cho KHGDVN flat schema |
| H — Per-class fine-tune | LOW | **SKIP** (như v1) | Compute prohibitive, gain marginal |

**Upgrade MỚI tìm ra (v1 missed):**

| Upgrade | Priority | Lý do |
|---|---|---|
| M1 — TAPT trên corpus 2024 unlabeled | **CRITICAL** | Module `tapt.py` đã có sẵn nhưng chưa wire vào pipeline. Đây là cơ chế chuẩn nhất để address temporal drift |
| M2 — Force-enable quantification at test | HIGH | Module `quantify.py` đã có nhưng cần verify được dùng và monitored cho test 2024 |
| M3 — Drift-aware diagnostic report | HIGH | Vấn đề cốt lõi train 2013-2022 / test 2024 cần diagnose rõ |

**Implementation order**: M1 → M2 → A modified → C+B+M3 (combined) → E. Không làm D, F, G, H.

---

## 1. Phân tích vấn đề cốt lõi: Temporal drift

### 1.1. Tại sao đây là vấn đề lớn nhất

KHGDVN dùng temporal split: train 2013-2022, val 2023, test 2024. Đây là kiểu split khó nhất vì có **concept drift** — distribution của input và label thay đổi theo thời gian:

- **Vocabulary shift**: Từ ngữ mới xuất hiện 2023-2024 (LLM, generative AI, ChatGPT in classroom, hybrid learning post-COVID...) — không có trong train data 2013-2022. Tokenizer SPECTER2 đã thấy các từ này từ pretrain, nhưng task-specific representation chưa
- **Topic shift**: Chủ đề hot 2013 (vd MOOC, online learning) khác hot 2024 (vd AI in education). Tỷ lệ Field thay đổi qua năm
- **Source title drift**: Tạp chí mới ra mắt 2023-2024 không có trong train. Tạp chí cũ đổi tên hoặc dừng phát hành
- **Label distribution shift**: STEM education có xu hướng tăng qua năm; "International education" có thể giảm sau COVID

### 1.2. Cơ chế nào trong pipeline hiện tại đã address drift?

KHGDVN đã có **3 cơ chế** đối phó drift, nhưng cần kiểm tra implementation:

| Cơ chế | Trạng thái | Vấn đề tiềm ẩn |
|---|---|---|
| Quantification adjustment (PACC, Saerens EM) trong `quantify.py` | Module có nhưng chưa rõ default enable | Cần verify gọi cho test 2024 |
| Multi-seed ensemble + TTA | Active | Không address drift, chỉ giảm noise |
| Rich features (Author Keywords, Document type) | Active | Source title trong rich features là điểm yếu vì drift mạnh nhất ở đây |

### 1.3. Cơ chế chưa được dùng

`tapt.py` có sẵn module `run_tapt()` để continue MLM pretrain SPECTER2 trên corpus mới. Đây chính là cơ chế **chuẩn** trong literature để address domain/temporal drift (Gururangan et al. 2020, "Don't Stop Pretraining" — paper Q1 nổi tiếng):

> "Continued pretraining on the unlabeled data of a target task" — gives consistent gains across 4 domains and 8 tasks.

KHGDVN có corpus 2024 unlabeled (Title + Abstract của tất cả paper 2024, bao gồm cả những paper không có trong test). Có thể dùng làm input cho TAPT.

**Kết luận**: Vấn đề drift cần được handle bằng kết hợp TAPT (M1) + quantification (M2) + diagnostic (M3), không phải bằng masking source title nặng tay (A v1).

---

## 2. Critical analysis từng upgrade v1

### 2.1. Upgrade A — Source title masking

**Lập luận v1 (đã sai)**: "Áp dụng 0.75 mask theo Gusenbauer 2025 sẽ improve robustness".

**Sai ở đâu**:

| Yếu tố | Gusenbauer 2025 | KHGDVN |
|---|---|---|
| Dataset size | Crossref large-scale (vài triệu paper) | Vài chục nghìn paper (2 bậc nhỏ hơn) |
| Số journal khác nhau | Hàng chục nghìn | Vài chục (giáo dục VN + một số international) |
| Vai trò source title | Có thể OVERFIT vì model có quá nhiều sample mỗi journal | Mỗi journal có ít sample, model không có nguy cơ memorize |
| Mục đích masking | Chống memorize "X journal → Y subject" | (Không có nguy cơ memorize) |

Ở Gusenbauer, dataset có 1.2M+ samples. Mỗi journal có thể có vài nghìn paper. Model dễ memorize "Nature → Science Field". Mask 75% là ÉP model học content vì có quá đủ data.

Ở KHGDVN, một tạp chí điển hình có 50–200 paper trong train 2013-2022. Mask 75% có nghĩa model chỉ thấy source title trên ~12-50 paper — quá ít để học được signal có giá trị từ source title. Risk: information loss > overfit prevention.

**Quyết định**: KEEP concept nhưng GIẢM mask_prob.

**Cấu hình mới**:
```python
SOURCE_TITLE_MASK_PROB = 0.30   # giảm từ 0.75
```

Thêm option curriculum (phức tạp hơn):
```python
# Hoặc: linear schedule. Epoch 1-3: 50% mask. Epoch 4-7: 25% mask. Epoch 8-10: 10% mask.
# Để model học content trước rồi gradually use full signal.
SOURCE_TITLE_MASK_SCHEDULE = "linear_decay"  # "constant" | "linear_decay"
```

**Lý do số 0.30**: 
- Gusenbauer paper Bảng kết quả cho thấy, gap weighted-F1 với/không source title: 0.892 vs 0.532 = gap 0.36 (40% drop). Nghĩa là source title là 40% signal
- Với corpus tiếng Việt và journal pool nhỏ, signal từ source title CAO HƠN tỷ lệ với content (vì content ngắn hơn, content tiếng Việt có thể nhiễu hơn so với English)
- Mask 30% là một compromise: vẫn buộc model có khả năng dự đoán khi không có source title, nhưng giữ đủ training signal trên data nhỏ

**Risk**: Ngay cả 0.30 cũng có thể quá nhiều cho Field thiểu số (vd Special education với <50 sample). Nên A/B test trên smoke set trước.

### 2.2. Upgrade B — Metadata ablation diagnostic

**Đánh giá**: ĐÚNG cho v2. Diagnostic, không gây hại, hữu ích để verify Upgrade A có hiệu quả không.

**Sửa**: Gộp output vào drift report thay vì file riêng.

### 2.3. Upgrade C — Class-wise F1 analysis

**Đánh giá**: ĐÚNG cho v2 nhưng CHƯA ĐỦ. Cần mở rộng thành drift-aware:

Ngoài per-class F1 trên val 2023, phải có:
- F1 trên test 2024 (drift)
- drift_gap = val_F1 − test_F1 cho từng class
- Support shift: train_support, val_support, test_support → identify class có distribution thay đổi
- Highlight class với drift_gap > 0.10 (severe drift)

### 2.4. Upgrade D — Frequency-stratified evaluation

**Đánh giá**: REDUNDANT với enhanced C.

D chỉ là aggregate view của C theo top/middle/bottom quartile. Nếu C đã có per-class breakdown, D chỉ thêm bảng aggregate ở dưới. Không cần file riêng, không cần function riêng. Inline trong phase summary là đủ.

**Quyết định**: GỘP vào C, không là upgrade riêng.

### 2.5. Upgrade E — Yang-format comparison table

**Đánh giá**: ĐÚNG. Low cost (1 function `format_yang_style_table()` 30 dòng code), high value (output để paste vào báo cáo hội đồng / paper draft).

### 2.6. Upgrade F — α weight scaling option

**Đánh giá**: SKIP. Không có evidence improve.

**Phân tích toán học**:

Yang Bảng 6: α ∈ {1, 10, 100}, α = 1 best trên CẢ 4 dataset. Paper kết luận:
> "α value of 1 provides the best results for BR-CNN for all datasets. Further increasing the value of α leads to a lowering of the performance."

KHGDVN auto pos_weight = N_neg / N_pos. So với Yang weighted BCE:

```
Yang loss (per sample):
  L_k = -(W_0 * y * log(p)) - (W_1 * (1-y) * log(1-p))
  với W_0 = 1/N_0, W_1 = α/N_1

PyTorch BCEWithLogitsLoss + pos_weight (per sample, KHGDVN):
  L_k = -(pos_weight * y * log(p)) - ((1-y) * log(1-p))
  với pos_weight = N_0/N_1 (auto compute)
```

Hai công thức ĐỀU áp dụng nguyên lý "weight up positive examples theo nghịch đảo positive frequency". Khác biệt là Yang có thêm constant term 1/N_0 cho negative samples — về bản chất là rescale toàn bộ loss bằng 1/N_0, không thay đổi gradient direction.

Khi auto pos_weight = N_0/N_1 và Yang α=1 dẫn đến W_1/W_0 = α * N_0/N_1 = N_0/N_1, **hai cấu hình tương đương về effective gradient**.

**Kết luận**: Thêm α grid là duplication không có evidence. Auto pos_weight đã hoạt động tốt theo principle Yang đã chứng minh. Skip.

**Thêm phân tích**: KHGDVN còn có MULTILABEL_LOSS option = "asymmetric" (Asymmetric Loss Ridnik 2021, γ⁻=4) đã handle imbalance theo cách khác. Nếu cần explore, dùng asymmetric thay vì α grid.

### 2.7. Upgrade G — Co-occurrence analysis

**Đánh giá**: SKIP cho KHGDVN.

Yang dùng co-occurrence để ranking class trong subset analysis (Hình 5-7). Nhưng action-able output là gì? Yang chỉ dùng để chứng minh BR-CNN tốt hơn LACO trên subset của 5 lớp ít overlap nhất.

KHGDVN không build hierarchical model, không build label-correlation model. Nếu compute cooccurrence matrix 12×12 cho Field, output là interesting-to-look-at nhưng không guide bất kỳ design decision nào trong pipeline.

Cost: 1 function ~50 dòng + add to phase_summary. Value: marginal.

**Quyết định**: SKIP. Nếu sau này build hierarchical model, có thể thêm sau.

### 2.8. Upgrade H — Per-class fine-tuning

**Đánh giá**: SKIP. Lý do trong v1 đã đúng.

Compute 12x training time (12 SPECTER2 chuyên cho 12 Field) cho cải thiện 1-2% lý thuyết. Trade-off không đáng. Hơn nữa, phá vỡ kiến trúc share encoder, làm code phức tạp.

---

## 3. Upgrade MỚI cốt lõi cho drift

### 3.1. Upgrade M1 — TAPT trên corpus 2024 unlabeled (CRITICAL)

**Cơ sở từ literature**:

Gururangan et al. (2020), "Don't Stop Pretraining: Adapt Language Models to Domains and Tasks", ACL 2020 (highly cited). Quote chính:

> "Continued pretraining on unlabeled data from the target task (TAPT) consistently improves performance, even when domain-adaptive pretraining (DAPT) is also performed."

Trong context KHGDVN, "target task data" = corpus paper 2024 (Title + Abstract, không có labels — đây là dữ liệu công khai có thể thu thập). TAPT trên corpus này adapt encoder vocabulary và representation cho:
- Vocabulary mới 2023-2024 (LLM, generative AI, hybrid learning...)
- Source title mới (tạp chí ra mắt 2023-2024)
- Style và terminology cập nhật

**Trạng thái hiện tại**:

`tapt.py` có sẵn với:
- `_load_full_corpus()` — load toàn corpus
- `MLMDataset` — dataset cho MLM
- `run_tapt(output_dir, epochs, lr, batch_size, mlm_probability, smoke)` — function chính

NHƯNG dependency map cho thấy KHÔNG ai gọi `tapt_py` từ pipeline chính. Nó là standalone, optional. Cần WIRE UP.

**Files cần modify**:

| File | Loại change |
|---|---|
| `config.py` | Add: `USE_TAPT = True`, `TAPT_OUTPUT_DIR`, `TAPT_EPOCHS = 3`, etc |
| `tapt.py` | Modify: `_load_full_corpus()` để chỉ load corpus 2024 (default) hoặc 2013-2024 (với flag) |
| `train_specter2.py` | Modify: nếu USE_TAPT và TAPT output exists, load từ TAPT_OUTPUT_DIR thay vì pretrained allenai/specter2_base |
| Pipeline scripts (run.sh hoặc Makefile) | Add: TAPT step trước train |

**Code design**:

```python
# config.py — thêm

# ==================== TAPT (Task-Adaptive PreTraining) ====================
# Continue MLM pretrain of SPECTER2 on target-domain unlabeled corpus to
# adapt representations for vocabulary and topic drift between train years
# (2013-2022) and test year (2024).
# Reference: Gururangan et al. 2020, "Don't Stop Pretraining", ACL.
# Recommended for any temporal-split scientometrics task with > 1 year gap
# between train and test.
USE_TAPT = True

# Where to save TAPT-adapted encoder. After TAPT, train_specter2 loads from
# this path instead of the HF Hub pretrained weights.
TAPT_OUTPUT_DIR = OUTPUT_DIR / "specter2_tapt"

# Corpus for TAPT: which years to include unlabeled
# - "test_only" (default): only Title+Abstract from main_2024 (most directly addresses drift)
# - "all": gold 2013-2023 + main 2024 (more data but mostly redundant with fine-tune corpus)
# - "recent": main 2024 + last 2 years of gold (2022, 2023) (compromise)
TAPT_CORPUS = "test_only"

# TAPT hyperparameters. SPECTER2 was pretrained with much larger compute;
# we just continue MLM for a few epochs to adapt to vocabulary drift.
# Don't overdo: too many epochs can cause forgetting of original pretrain knowledge.
TAPT_EPOCHS = 3
TAPT_LR = 5e-5         # higher than fine-tune LR (encoder pretrain rate)
TAPT_BATCH_SIZE = 16   # smaller because no labels = longer sequences viable
TAPT_MLM_PROBABILITY = 0.15   # standard BERT MLM rate
```

```python
# tapt.py — modify _load_full_corpus() để hỗ trợ TAPT_CORPUS flag

def _load_full_corpus():
    """Load Title + Abstract from configured corpus for TAPT.
    
    Behavior depends on config.TAPT_CORPUS:
    - "test_only": only main_2024_clean (most direct drift addressing)
    - "all": gold 2013-2023 + main 2024
    - "recent": main 2024 + 2022-2023 from gold
    """
    import pandas as pd
    import config
    
    corpus = config.TAPT_CORPUS
    dfs = []
    
    if corpus in ("test_only", "all", "recent"):
        if config.MAIN_2024_PARQUET.exists():
            df_2024 = pd.read_parquet(config.MAIN_2024_PARQUET)
            dfs.append(df_2024[["Title", "Abstract"]])
        else:
            raise FileNotFoundError(
                f"TAPT requires {config.MAIN_2024_PARQUET}. Run sanitize.py first."
            )
    
    if corpus == "all":
        gold = pd.read_parquet(config.GOLD_PARQUET)
        dfs.append(gold[["Title", "Abstract"]])
    elif corpus == "recent":
        gold = pd.read_parquet(config.GOLD_PARQUET)
        recent = gold[gold["Year"] >= 2022]
        dfs.append(recent[["Title", "Abstract"]])
    
    full_df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["Title"])
    print(f"TAPT corpus '{corpus}': {len(full_df)} unique papers loaded")
    return full_df
```

```python
# train_specter2.py — modify train_model() để load từ TAPT output

def train_model(task, smoke=False, seed=None, include_val_in_train=False):
    # ... existing code ...
    
    # Determine encoder path
    if config.USE_TAPT and config.TAPT_OUTPUT_DIR.exists():
        encoder_path = str(config.TAPT_OUTPUT_DIR)
        print(f"  Loading TAPT-adapted encoder from {encoder_path}")
    else:
        encoder_path = config.BACKBONE_MODEL
        if config.USE_TAPT:
            print(f"  WARNING: USE_TAPT=True but {config.TAPT_OUTPUT_DIR} does not exist. "
                  f"Run `python tapt.py` first. Falling back to {encoder_path}.")
    
    # Then load tokenizer + model from encoder_path
    tokenizer = AutoTokenizer.from_pretrained(encoder_path)
    # ... rest of existing code uses encoder_path ...
```

**Acceptance criteria**:

1. `python tapt.py --smoke` chạy thành công, output `outputs/specter2_tapt/` chứa pytorch_model.bin và config
2. `python tapt.py` (full) chạy ≤ 2 giờ trên T4
3. `python train_specter2.py --task fields` log ra dòng "Loading TAPT-adapted encoder from outputs/specter2_tapt"
4. Sau khi train xong với TAPT-adapted encoder, eval test 2024 macro-F1 phải HƠN bằng baseline (không TAPT) ≥ 1%. Nếu kém hơn → có vấn đề (overfit TAPT?)

**Test plan**:

```bash
# Step 1: Run TAPT on 2024 corpus
python tapt.py --smoke   # 10-15 phút, verify pipeline
python tapt.py           # 1-2 giờ, full TAPT

# Step 2: Train với TAPT-adapted encoder
USE_TAPT=true python train_specter2.py --task fields

# Step 3: Compare
USE_TAPT=false python train_specter2.py --task fields   # baseline rerun
# So sánh test_2024_macro_f1 với và không TAPT
```

**Risk**:
- TAPT trên corpus quá nhỏ (< 500 papers) có thể overfit → check `len(_load_full_corpus()) >= 500`, nếu ít hơn dùng TAPT_CORPUS="all"
- TAPT epochs quá nhiều có thể catastrophic forgetting → giữ TAPT_EPOCHS = 3, không tăng
- Nếu test 2024 macro-F1 GIẢM sau TAPT, có thể corpus TAPT có nhiễu (papers không phải giáo dục) → revert to USE_TAPT=False

**Expected impact**: +2 đến +6% macro-F1 trên test 2024 (drift) — đây là literature-supported impact của TAPT trên temporal/domain drift.

### 3.2. Upgrade M2 — Force-enable quantification at evaluation

**Cơ sở**:

KHGDVN có `quantify.py` với 4 functions: PACC prior estimate, Saerens EM prior, prior shift threshold adjustment, quantified thresholds. Đây là cơ chế chuẩn để adjust threshold khi label distribution shift giữa train (2013-2022) và test (2024).

**Vấn đề**: Không rõ quantification được DEFAULT-ENABLED trong evaluate.py hay phải bật bằng flag. Nếu chưa enable, đang lãng phí một cơ chế đã code sẵn.

**Files cần modify**:

| File | Loại change |
|---|---|
| `config.py` | Add: `USE_QUANTIFICATION_AT_TEST = True`, `QUANTIFICATION_ESTIMATOR = "saerens_em"` |
| `evaluate.py` | Verify: cho test 2024, ALWAYS run quantified_thresholds() và report cả 2 numbers (with vs without quantification) |
| `phase_summary.py` | Render: section "Quantification" trong eval report |

**Code design**:

```python
# config.py — thêm

# ==================== Quantification (Prior Shift Adjustment) ====================
# Adjust per-class thresholds at test time when test label distribution
# differs from train (Saerens et al. 2002, Neural Computation).
# CRITICAL for temporal-split tasks where test 2024 may have different
# Field/Level proportions than train 2013-2022.
USE_QUANTIFICATION_AT_TEST = True

# Estimator for test prior:
# - "pacc"        : Probabilistic Adjusted Classify and Count (fast, simple)
# - "saerens_em"  : Iterative EM-based (more accurate but ~10x slower)
# - "both"        : Run both and report — useful for comparison
QUANTIFICATION_ESTIMATOR = "saerens_em"

# Always report eval metrics WITHOUT quantification too, for comparison.
# This makes the impact of quantification visible in the eval report.
QUANTIFICATION_REPORT_BOTH = True
```

**Code wire-up trong `evaluate.py`** (pseudocode, integrate vào evaluate_task):

```python
# evaluate.py — within evaluate_task() for test split

if config.USE_QUANTIFICATION_AT_TEST and split_name == "test":
    from quantify import quantified_thresholds
    
    test_thresholds_quantified = quantified_thresholds(
        val_probs=val_probs,
        val_targets=val_targets,
        test_probs=test_probs,
        val_thresholds=tuned_thresholds,
        target_type=target_type,
        estimator=config.QUANTIFICATION_ESTIMATOR,
    )
    
    test_result_quantified = evaluate_with_thresholds(
        test_probs, test_targets, target_type, n_classes, test_thresholds_quantified
    )
    
    # Original test_result without quantification
    test_result_baseline = evaluate_with_thresholds(
        test_probs, test_targets, target_type, n_classes, tuned_thresholds
    )
    
    if config.QUANTIFICATION_REPORT_BOTH:
        eval_report["test_no_quantification"] = test_result_baseline
        eval_report["test_with_quantification"] = test_result_quantified
        # Default (deployment-ready) is the quantified version
        eval_report["test"] = test_result_quantified
    else:
        eval_report["test"] = test_result_quantified
    
    # Log diff
    delta_macro = test_result_quantified["macro_f1"] - test_result_baseline["macro_f1"]
    print(f"  Quantification adjustment: macro-F1 delta = {delta_macro:+.4f}")
```

**Acceptance criteria**:

1. Eval report có cả `test_no_quantification` và `test_with_quantification`
2. Phase summary in ra so sánh:
   ```
   Test 2024 macro-F1:
     Without quantification: 0.5234
     With Saerens EM:        0.5612 (+0.0378)
   ```
3. Nếu quantification làm WORSE > 0.02, log WARNING và keep no-quantification thành default

**Expected impact**: +1 đến +3% macro-F1 trên test 2024. Đây là cơ chế đã code sẵn, không tăng compute đáng kể.

### 3.3. Upgrade M3 — Drift-aware diagnostic report

Gộp Upgrade B (metadata ablation) + C (class-wise) + D (frequency strata) vào một report drift-aware duy nhất.

**Output format**:

`outputs/drift_report.md` chứa 4 section:

1. **Overall drift summary**:
   - val_macro_f1 (2023) vs test_macro_f1 (2024) gap
   - val support per class vs test support per class shift

2. **Per-class breakdown** (gộp upgrade C + D):
   ```
   | Class | Train support | Val sup | Test sup | Val F1 | Test F1 | Drift gap | Note |
   |---|---|---|---|---|---|---|---|
   | teaching & learning | 1234 | 145 | 178 | 0.78 | 0.72 | -0.06 | (large class, normal) |
   | Special education | 67 | 8 | 9 | 0.45 | 0.18 | -0.27 | DRIFT WARN |
   | ... | ... |
   ```

3. **Source title ablation** (gộp upgrade B):
   - Re-eval test 2024 với source title masked → measure how much drop
   - Acceptance: drop < 0.20 (model not over-reliant)

4. **Quantification impact**:
   - Test với và không quantification (từ M2)

**Files cần modify**:

| File | Loại change |
|---|---|
| `evaluate.py` | Add: `generate_drift_report(eval_results, train_df, val_df, test_df, output_path)` — gộp class-wise + ablation + quantification |
| `phase_summary.py` | Render drift_report.md từ eval_results |

**Quyết định**: Skip detailed code design ở đây vì code khá dài. Chỉ cần Claude Code implement combined report dựa trên các functions đã có (`evaluate_metadata_ablation`, `report_class_wise_analysis`, `quantified_thresholds`).

---

## 4. Final integrated upgrade plan

Sequence implementation (revised hoàn toàn so với v1):

### Pha 1 — Drift mitigation (CRITICAL, tuần 1)

**M1 — Wire up TAPT** (1-2 ngày):
- Add config flags
- Modify `tapt.py:_load_full_corpus()` để hỗ trợ TAPT_CORPUS option
- Modify `train_specter2.py` để load TAPT encoder nếu có
- Run TAPT smoke test
- Run TAPT full + train pipeline; compare vs baseline

**M2 — Verify + force quantification** (½ ngày):
- Add config flags
- Wire quantification vào `evaluate.py` cho test split
- Report cả with/without để monitor impact

**Verification gate**: Sau M1 + M2, run full eval. Test 2024 macro-F1 phải improve ít nhất +2% so với baseline. Nếu không, debug trước khi tiếp tục.

### Pha 2 — Modified upgrade A (½ ngày)

**A modified — Source title masking 0.30** (KHÔNG 0.75):
- Add `SOURCE_TITLE_MASK_PROB = 0.30` to config
- Modify `utils.build_input_texts()` để hỗ trợ training=True/False
- Modify `utils.PaperDataset.__init__` thêm training param
- Update call sites trong train_specter2.py, evaluate.py, inference.py
- Run smoke test
- A/B compare: với mask 0.30 vs không mask. Nếu mask làm xấu hơn 1%, revert hoặc giảm xuống 0.20

### Pha 3 — Diagnostic & reporting (1 ngày)

**M3 + C + B (combined drift report)**:
- Implement `generate_drift_report()` trong `evaluate.py`
- Output `outputs/drift_report.md` với 4 section
- Verify acceptance criteria
- Generate report sau full pipeline run

**E — Yang-format table** (½ ngày):
- Implement `format_yang_style_table()` trong `print_summary.py`
- Output `outputs/yang_format_table.md`

### Pha 4 — Verification

Run full pipeline end-to-end:
```bash
python sanitize.py
python tapt.py                      # NEW: TAPT step
python llm_augment.py               # existing
python train_specter2.py --task all # uses TAPT encoder
python knn_retrieval.py             # existing
python llm_classify.py              # existing
python evaluate.py --task all       # with quantification + drift report
python print_summary.py             # Yang-format
python export_review.py
```

Acceptance: 
- Smoke test pass
- Full pipeline runs < 4 hours on T4
- Test 2024 macro-F1 improve ≥ 2% so với baseline (no TAPT, no quantification, no masking)
- Drift report generated với class-level breakdown

---

## 5. Trade-off analysis

### 5.1. Compute cost so sánh

| Step | Baseline | Với upgrades đề xuất | Tăng |
|---|---|---|---|
| Sanitize | 1 phút | 1 phút | 0 |
| TAPT | (skip) | 1-2 giờ | +1-2h ONE-TIME |
| LLM augment | (đã có) | (không đổi) | 0 |
| Train SPECTER2 (3 task × 3 seed) | 2-3 giờ | 2-3 giờ | 0 |
| Evaluate | 10 phút | 12 phút (+ quantification + drift report) | +2 phút |
| **Tổng** | ~3 giờ | ~5 giờ | +1-2 giờ ONE-TIME (TAPT) |

TAPT là one-time cost. Sau khi có TAPT-adapted encoder, không cần re-run trừ khi có corpus 2025 mới.

### 5.2. Lợi ích kỳ vọng (cumulative)

| Upgrade | Test 2024 macro-F1 expected gain | Cumulative |
|---|---|---|
| Baseline | 0 | ~ 0.50 (giả định) |
| + M1 TAPT | +3 đến +6% | ~ 0.53–0.56 |
| + M2 Quantification | +1 đến +3% | ~ 0.54–0.59 |
| + A modified masking | +1 đến +2% | ~ 0.55–0.61 |
| **Total realistic** | +5 đến +11% | **~ 0.55–0.61** |

Đây là RANGE, không phải lời hứa. Có thể thấp hơn nếu drift KHGDVN không nặng như giả định, hoặc cao hơn nếu corpus 2024 đặc biệt khác 2013-2022.

### 5.3. Risk analysis

| Risk | Mitigation |
|---|---|
| TAPT overfit corpus 2024 | TAPT_EPOCHS = 3 only; A/B test với và không TAPT |
| Quantification làm WORSE (rare nhưng có thể) | QUANTIFICATION_REPORT_BOTH=True để compare; rollback nếu negative |
| Mask 0.30 vẫn quá nhiều cho Field thiểu số | A/B test trên smoke; giảm xuống 0.20 nếu cần |
| Code regression | Smoke test gate sau mỗi upgrade; commit riêng để dễ revert |
| Reproducibility break | Lock TAPT seed; verify `configs/codebook_hash.txt` chưa thay đổi |

---

## 6. Cập nhật lý thuyết: tại sao TAPT > masking nặng

Một insight quan trọng làm decision này khác hẳn v1:

**v1 nghĩ**: "Mask source title 75% sẽ chống drift vì model không lệ thuộc tạp chí cũ" — partial truth.

**v2 hiểu**: Có 2 nguồn drift khác nhau, cần handle khác nhau:

1. **Source-title drift**: Tạp chí mới ra mắt, tên đổi → handle bằng masking nhẹ (A modified, 0.30) buộc model có khả năng dự đoán không cần source title. ĐỦ.

2. **Vocabulary + topic drift**: Từ ngữ và chủ đề mới (LLM, generative AI...) → KHÔNG thể fix bằng masking (vì masking source title không adapt encoder cho từ mới). PHẢI dùng TAPT (M1) — adapt encoder representation cho corpus 2024.

→ Trước: v1 chỉ có A → cover một phần nhỏ của drift problem.
→ Sau: v2 có A modified + M1 + M2 → cover toàn bộ drift problem (vocabulary + topic + source title + label distribution).

Đây là lý do v2 SỬA quyết định v1.

---

## 7. Code đã có sẵn không cần thêm

KHGDVN đã có khá nhiều cơ chế advanced. Confirm trước khi đề xuất gì mới:

| Cơ chế | Module | Trạng thái |
|---|---|---|
| Multi-seed ensemble | `train_specter2.py` | ACTIVE (3 seeds) |
| TTA (test-time augmentation) | `evaluate.py` | ACTIVE (2 variants) |
| Per-class threshold tuning | `utils.tune_thresholds_robust` | ACTIVE |
| Robust threshold fallback | `utils.tune_thresholds_robust` | ACTIVE |
| LLM augmentation | `llm_augment.py` | ACTIVE (3-OpenAI panel) |
| LLM classification | `llm_classify.py` | ACTIVE (alternative path) |
| k-NN retrieval | `knn_retrieval.py` | ACTIVE (alternative path) |
| 3-way ensemble (specter + gpt5 + knn) | `evaluate.py + ensemble.py` | ACTIVE |
| Quantification | `quantify.py` | EXISTS — verify enabled (M2) |
| TAPT | `tapt.py` | EXISTS — wire up (M1) |
| Codebook SHA-256 hash | `sanitize.py` | ACTIVE |
| Asymmetric Loss (Ridnik 2021) | `utils.AsymmetricLoss` | EXISTS as option |
| Focal CE | `utils.FocalCrossEntropyLoss` | ACTIVE for method |
| Label smoothing | training loop | ACTIVE (0.05) |

**Không cần đề xuất** cho các cơ chế đã active. Chỉ M1 và M2 là wire-up cho cơ chế đã code sẵn.

---

## 8. Khuyến nghị cuối cùng

**Implement** (theo thứ tự):
1. **M1 TAPT** — CRITICAL nhất, fix vấn đề cốt lõi temporal drift
2. **M2 Quantification** — đảm bảo cơ chế đã có thực sự được dùng
3. **A modified (mask 0.30)** — robustness phụ trợ
4. **M3 + C + B (combined drift report)** — diagnostic
5. **E (Yang-format)** — reporting

**Skip** (với lý do rõ ràng):
- D — gộp vào C
- F — auto pos_weight đã tương đương, không evidence improve
- G — low-actionable cho schema flat
- H — compute prohibitive

**Tổng**: 5 upgrade thực sự (M1, M2, A modified, M3+C+B combined, E). Ít hơn v1 (8 upgrade) nhưng chất lượng cao hơn — mỗi cái có justification rõ và kết hợp với nhau coherent.

**Thời gian implement**: 4-5 ngày làm việc + 1 ngày verify + tune.

**Expected total impact**: Test 2024 macro-F1 +5 đến +11% (range realistic). Drift gap (val 2023 vs test 2024) thu hẹp đáng kể.

---

## 9. Hướng dẫn cho Claude Code

### Prompt template

Đối với MỖI upgrade, dùng prompt riêng — không apply đồng thời nhiều upgrade.

**Prompt cho M1**:
```
Implement Upgrade M1 — Wire up TAPT — từ UPGRADE_ROADMAP_v2.md mục 3.1.

Đọc mục 3.1 đầy đủ. Implement theo thứ tự:
1. Add config flags (USE_TAPT, TAPT_OUTPUT_DIR, TAPT_CORPUS, etc) vào config.py
2. Modify tapt.py:_load_full_corpus() để hỗ trợ TAPT_CORPUS = "test_only"|"all"|"recent"
3. Modify train_specter2.py:train_model() để load TAPT-adapted encoder nếu USE_TAPT=True và TAPT_OUTPUT_DIR exists
4. Run smoke: `python tapt.py --smoke` (10-15 phút)
5. Verify: tapt output dir tồn tại với pytorch_model.bin

Sau khi implement, report:
- File diff
- Smoke test output
- Memory + time consumed
```

Tương tự cho M2, A modified, drift report, E.

### Verification gate

Sau MỖI upgrade:
1. `python smoke_test.py` PASS
2. Acceptance criteria của upgrade đó đáp ứng
3. Git commit riêng với message rõ
4. Nếu fail, revert ngay và investigate

### Rollback plan

Nếu sau M1 + M2, test 2024 macro-F1 KHÔNG improve ≥ 2% so với baseline:

1. Disable M1: `USE_TAPT=False` (revert encoder load)
2. Re-run train + eval
3. Nếu vẫn không improve, disable M2: `USE_QUANTIFICATION_AT_TEST=False`
4. Investigate root cause: dataset issue? corpus 2024 quá nhỏ cho TAPT? smoke test corpus chosen?

---

**Hết tài liệu v2**

File này thay thế v1 (UPGRADE_ROADMAP.md). Khác biệt cốt lõi: SKIP 4 upgrade không có evidence (D, F, G, H), MODIFY 1 upgrade (A từ 0.75 xuống 0.30), thêm 3 upgrade mới critical cho temporal drift (M1 TAPT, M2 Quantification, M3 drift-aware diagnostic). Implementation thời gian ngắn hơn nhưng impact cao hơn vì targeted vào vấn đề cốt lõi của KHGDVN.
