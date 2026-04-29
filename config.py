"""
Central configuration for bibliometric classification pipeline.
All constants frozen here to ensure reproducibility.
"""
from pathlib import Path

# ==================== PATHS ====================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CONFIG_DIR = PROJECT_ROOT / "configs"

INPUT_FILE = DATA_DIR / "2025_11_09_-_biblio_-_du_lieu_phep_phan_tich_-_2013-2024__1_.xlsx"
GOLD_PARQUET = OUTPUT_DIR / "gold_2013_2023.parquet"
MAIN_2024_PARQUET = OUTPUT_DIR / "main_2024_clean.parquet"   # 2024 papers, sanitized + canonical labels
SPECIAL_EDU_AUGMENT = OUTPUT_DIR / "special_edu_augmented.parquet"
LLM_LOG_DIR = OUTPUT_DIR / "llm_logs"
LLM_PROGRESS_DIR = OUTPUT_DIR / "llm_progress"   # jsonl progress files for resume

# ==================== REPRODUCIBILITY ====================
SEED = 42

# ==================== TASK SCHEMA ====================
# 12 Fields (drop 'research')
FIELDS_12 = [
    "teaching & learning",
    "management, leadership & policy",
    "test and assessment",
    "Technology in education",
    "English Education",
    "curriculum",
    "psychology in education",      # renamed từ 'psychology eduation' (broad def)
    "Special education",
    "International education",
    "Education economically",
    "STEM education",
    "Non-STEM Education",
]

# Mapping aliases trong gold cũ → canonical
FIELDS_ALIASES = {
    "psychology eduation": "psychology in education",        # typo + rename
    "psychology education": "psychology in education",       # canonicalize
    "Special edu": "Special education",                       # alias
    "english education": "English Education",                 # case
    "stem education": "STEM education",                       # case
    "non-stem education": "Non-STEM Education",              # case
    "technology in education": "Technology in education",     # case
    "international education": "International education",     # case
    "education economically": "Education economically",       # case
}

# 6 Educational Levels
LEVELS_6 = ["ECE", "GE", "HE", "TVET", "LLL", "ALL"]
LEVELS_ALIASES = {
    "All": "ALL",
    "all": "ALL",
    "3L": "ALL",       # observed shorthand, no consistent meaning → default ALL
    "EGE": "ECE",      # typo
    "TEVE": "TVET",    # typo
    "TEVT": "TVET",    # typo
    "EVE": "TVET",     # likely typo of TVET (gần TEVE)
}

# 5 Methods (single-label)
METHODS_5 = ["Quantitative", "Qualitative", "Mixed", "Review", "Other"]
METHOD_ALIASES = {
    # Case variations
    "QUANTITATIVE": "Quantitative",
    "quantitative": "Quantitative",
    "Quanlitative": "Qualitative",     # typo
    "QUANLITATIVE": "Qualitative",     # typo
    "qualitative": "Qualitative",
    "MIXED": "Mixed",
    "mixed": "Mixed",
    "Mix": "Mixed",
    "REVIEW": "Review",
    "review": "Review",
    "OTHER": "Other",
    "quanli": "Qualitative",           # observed typo
    "Qualitativetative": "Qualitative",  # observed compound typo
    "qualitativetative": "Qualitative",
    "QUALITATIVETATIVE": "Qualitative",
}

# ==================== TEMPORAL SPLIT ====================
TRAIN_YEARS = list(range(2013, 2023))   # 2013-2022
VAL_YEAR = 2023
TEST_YEAR = 2024

# ==================== MODEL CONFIG ====================
SPECTER2_BASE = "allenai/specter2_base"
# Pin a known-good revision for full reproducibility across machines / Colab.
# Set to None to always use HF Hub `main` (less stable but always-fresh).
SPECTER2_REVISION = None
SPECTER2_ADAPTER = "allenai/specter2_classification"
MAX_LENGTH = 512
DROPOUT = 0.2   # Slightly higher than the previous 0.1 to combat overfit on small data.

# ==================== TRAINING CONFIG ====================
# CPU and GPU need different defaults. The torch.cuda.is_available() probe at
# train time picks the GPU branch — these constants are the GPU baseline.
# CPU users should drop BATCH_SIZE to 8 and EPOCHS to 3-5.
BATCH_SIZE = 32
LR = 2e-5            # SPECTER2 fine-tune typical: 2e-5 — 5e-5 (was 1e-5, too conservative)
EPOCHS = 10          # Was 5: training was still improving at epoch 5 (loss not converged)
WARMUP_RATIO = 0.06  # BERT default for fine-tuning (was 0.1)
WEIGHT_DECAY = 0.01
EARLY_STOPPING_PATIENCE = 3   # Was 2: slight noise in val_f1 was triggering early stop too aggressively
USE_AMP = True       # Mixed-precision on GPU: 2-3x training speedup, identical math on T4+

# Asymmetric Loss (Ridnik et al. 2021) — for multi-label
ASYMMETRIC_LOSS = {
    "gamma_pos": 0,
    "gamma_neg": 4,
    "clip": 0.05,
}

# Loss type for multi-label tasks. AsymmetricLoss handles imbalance natively
# but its outputs are poorly calibrated (different threshold per class — in
# practice optimal thresholds spanned 0.10-0.99). BCE with pos_weight keeps
# outputs calibrated and lets threshold tuning settle in a saner range.
# Default switched to "bce_pos_weight" after the AUC-best-model run regressed
# Method test F1 vs the original AsymmetricLoss CPU baseline.
MULTILABEL_LOSS = "bce_pos_weight"   # "asymmetric" | "bce_pos_weight"

# Per-class threshold tuning
# Step 0.02 (41 candidates per class) finds sharper decision boundaries than
# step 0.05 with negligible compute cost (~492 f1_score calls per task).
THRESHOLD_GRID = [round(0.10 + 0.02 * i, 2) for i in range(41)]   # 0.10 → 0.90 step 0.02

# Threshold tuning becomes unreliable when val support is too small. Below
# this many positive samples we fall back to AUC-derived Youden's J on val,
# or a safe default 0.5 if AUC is also undefined. This stops thresholds
# from chasing noise on classes with 1-3 val positives.
LOW_SUPPORT_THRESHOLD_FALLBACK = 10

# Best-model selection metric.
# - "tuned_macro_f1" (DEFAULT, recommended): tune per-class thresholds on val
#   each epoch, compute macro-F1 with those thresholds, select best epoch by
#   that. Matches the actual deployment metric. Adds ~1s/epoch overhead.
# - "macro_auc": threshold-independent ranking metric. Risk: AUC-best epoch
#   may not be F1-best (observed regression on Method: epoch 3 had peak AUC
#   F1=0.500 but epoch 4 had F1=0.532).
# - "macro_f1": F1 with default threshold 0.5 — biased on uncalibrated losses.
BEST_MODEL_METRIC = "tuned_macro_f1"

# ==================== LLM CONFIG (OpenAI-only ensemble) ====================
# 3 different OpenAI models for diversity. All use temperature=0 + seed=SEED for reproducibility.
# Pricing (April 2026):
#   gpt-5.5:        $5.00 in / $30.00 out per 1M tokens
#   gpt-5.4:        $2.50 in / $10.00 out per 1M tokens (with reasoning_effort)
#   gpt-5.4-mini:   $0.75 in / $4.50  out per 1M tokens
OPENAI_PANEL = [
    {
        "model": "gpt-5.5",
        "alias": "gpt55",
        "reasoning_effort": None,  # gpt-5.5 doesn't expose reasoning_effort param
    },
    {
        "model": "gpt-5.4",
        "alias": "gpt54",
        "reasoning_effort": "medium",
    },
    {
        "model": "gpt-5.4-mini",
        "alias": "gpt54mini",
        "reasoning_effort": None,
    },
]

LLM_RETRY_MAX = 3
LLM_RATE_LIMIT_PAUSE = 0.5   # seconds between calls (OpenAI handles rate limits well)
LLM_REQUEST_TIMEOUT = 90      # seconds per call

# ==================== KEYWORDS for filter ====================
# Special education keywords để pre-filter candidates trước khi LLM verify
SPECIAL_EDU_KEYWORDS = [
    "disability", "disabilities", "disabled", "impairment",
    "autism", "autistic", "asd",
    "adhd", "attention deficit",
    "dyslexia", "dyslexic", "learning disabil",
    "intellectual disabil", "cognitive disabil",
    "deaf", "blind", "visual impair", "hearing impair",
    "special needs", "special education",
    "individualized education", "iep",
    "gifted", "talented", "high-achiev",
    "inclusive education", "inclusion",
    "wheelchair", "mobility impair",
    "speech-language", "speech language",
    "down syndrome",
    "exceptional learner", "exceptional student",
]

# Per-Level keywords (rare classes only; GE/HE/ALL are common enough)
LEVEL_AUGMENT_KEYWORDS = {
    "ECE": [
        "early childhood", "preschool", "pre-school", "kindergarten",
        "daycare", "day-care", "nursery", "preprimary", "pre-primary",
        "young children", "0-6 years", "early years",
        "play group", "playgroup",
        "ecce", "eccd",
    ],
    "TVET": [
        "vocational", "technical education", "tvet",
        "apprentice", "trade school", "trade education",
        "technical college", "technical school", "vocational school",
        "skills training", "industrial training",
        "career and technical", "career-technical",
        "polytechnic",
    ],
    "LLL": [
        "lifelong learning", "lifelong",
        "adult learner", "adult learners", "adult education", "adult student",
        "continuing education", "continuing professional",
        "in-service training", "in-service teacher",
        "professional development", "cpd",
        "workplace learning",
        "non-formal education",
        "andragogy",
    ],
}

LEVEL_AUGMENT_OUTPUTS = {
    "ECE": OUTPUT_DIR / "ece_augmented.parquet",
    "TVET": OUTPUT_DIR / "tvet_augmented.parquet",
    "LLL": OUTPUT_DIR / "lll_augmented.parquet",
}

# ==================== LOSS CONFIG ====================
# Focal cross-entropy for the single-label Method task — focuses gradient on
# hard examples, which helps the very-rare 'Other' class (~33 train samples).
# AsymmetricLoss already handles multi-label imbalance, so it stays untouched.
USE_FOCAL_LOSS_METHOD = True
FOCAL_GAMMA = 2.0

# ==================== TRASH DETECTION (per Linh's Q5 decision) ====================
# Drop records meeting ANY of these criteria
MIN_TITLE_WORDS = 3       # Drop if Title < 3 words (low quality)
MIN_ABSTRACT_WORDS = 30   # Drop if Abstract < 30 words (low signal)
# Records also dropped if:
# - Missing Title or Abstract
# - Method cannot be canonicalized
# - Fields list empty after dropping 'research' and applying aliases

# ==================== FILE NAMES ====================
def model_path(task: str) -> Path:
    """Get model save path for a task."""
    assert task in ["fields", "levels", "method"]
    return OUTPUT_DIR / f"model_{task}.pt"

def threshold_path(task: str) -> Path:
    """Get per-class threshold save path."""
    return OUTPUT_DIR / f"thresholds_{task}.json"

def eval_report_path() -> Path:
    return OUTPUT_DIR / "eval_report.json"

def codebook_hash_path() -> Path:
    return CONFIG_DIR / "codebook_hash.txt"
