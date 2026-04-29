"""
Phase 0: Sanitization
======================
Đọc file Excel → clean → output 2 parquet files:
  - gold_2013_2023.parquet   (2107 papers, training set)
  - main_2024.parquet         (617 papers, test set, no labels)

Áp dụng quyết định:
- Option D (split by conflict)
- BROAD definition cho 'psychology in education'
- Drop 'research' label, keep papers nếu có lớp khác (272 papers research-only sẽ bị drop)
- Sanitize Method (21 → 5 case canonical)
- Sanitize Level (12 → 6)
- Resolve newline/whitespace bugs trong Fields ('English \\nEducation')

Usage:
    python sanitize.py
"""
import re
import json
import hashlib
import warnings
from pathlib import Path

import pandas as pd

import config

warnings.filterwarnings("ignore")


# ==================== HELPERS ====================
def normalize_whitespace(s):
    """Replace newlines/tabs with single space, collapse multiple spaces, strip."""
    if pd.isna(s):
        return ""
    return re.sub(r"\s+", " ", str(s)).strip()


def canonicalize_field_token(token: str) -> str | None:
    """Map raw Field token to canonical name in FIELDS_12. Returns None if invalid."""
    t = normalize_whitespace(token).lower()
    
    # Drop 'research' explicitly (Q3 decision)
    if t == "research":
        return None
    
    # Lookup aliases (case-insensitive)
    for alias, canonical in config.FIELDS_ALIASES.items():
        if t == alias.lower():
            return canonical
    
    # Direct match (case-insensitive) với FIELDS_12
    for canon in config.FIELDS_12:
        if t == canon.lower():
            return canon
    
    # Unknown — log warning and skip
    return None


def canonicalize_level(s):
    """Normalize Level value: handle 'All' → 'ALL', multi-label etc."""
    if pd.isna(s):
        return []
    s = normalize_whitespace(s)
    # Multi-label split (e.g., "GE; HE; LLL")
    parts = re.split(r"[;,/]\s*", s)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # Apply alias
        canon = config.LEVELS_ALIASES.get(p, p)
        if canon in config.LEVELS_6:
            out.append(canon)
    return out


def canonicalize_method(s):
    """Normalize Method value: handle case variations + typos."""
    if pd.isna(s):
        return None
    s = normalize_whitespace(s)
    # Apply alias
    canon = config.METHOD_ALIASES.get(s, s)
    # Title case fallback
    if canon not in config.METHODS_5:
        canon_tc = canon.title()
        if canon_tc in config.METHODS_5:
            return canon_tc
        # Match by first letter
        if canon.lower().startswith("quan"):
            if "li" in canon.lower():
                return "Qualitative"
            return "Quantitative"
        if canon.lower().startswith("qual"):
            return "Qualitative"
        if canon.lower().startswith("mix"):
            return "Mixed"
        if canon.lower().startswith("rev"):
            return "Review"
        if canon.lower().startswith("oth"):
            return "Other"
        return None
    return canon


# ==================== MAIN ====================
def load_gold():
    """Load sheet recoded- field & level (2380 papers)."""
    return pd.read_excel(config.INPUT_FILE, sheet_name="recoded- field & level")


def load_main():
    """Load main sheet (2997 papers all years)."""
    return pd.read_excel(config.INPUT_FILE, sheet_name="2025 11 09- VOS - du lieu phep")


def process_gold(df_gold: pd.DataFrame, df_main: pd.DataFrame) -> pd.DataFrame:
    """
    Process gold dataset (2013-2023).
    
    Steps:
    1. Parse Fields multi-label → list canonical names (drop 'research')
    2. Drop papers với 0 valid Fields after dropping 'research' (research-only papers)
    3. Canonicalize Level
    4. Canonicalize Method
    5. Join with main sheet để lấy Year actual
    6. Output 1 row per paper with: Total_ID, Year, Title, Abstract, fields_list, levels_list, method
    """
    print(f"Initial gold rows: {len(df_gold)}")
    
    # Title/Abstract sanitize
    df_gold["Title"] = df_gold["Title"].apply(normalize_whitespace)
    df_gold["Abstract"] = df_gold["Abstract"].apply(normalize_whitespace)
    
    # Trash detection: missing Title or Abstract
    n_before = len(df_gold)
    df_gold = df_gold[(df_gold["Title"] != "") & (df_gold["Abstract"] != "")].copy()
    n_dropped_missing = n_before - len(df_gold)
    if n_dropped_missing > 0:
        print(f"Dropped {n_dropped_missing} papers thiếu Title hoặc Abstract")
    
    # Trash detection: minimum word counts
    df_gold["title_words_tmp"] = df_gold["Title"].str.split().str.len()
    df_gold["abs_words_tmp"] = df_gold["Abstract"].str.split().str.len()
    n_before = len(df_gold)
    df_gold = df_gold[
        (df_gold["title_words_tmp"] >= config.MIN_TITLE_WORDS) &
        (df_gold["abs_words_tmp"] >= config.MIN_ABSTRACT_WORDS)
    ].copy()
    n_dropped_short = n_before - len(df_gold)
    if n_dropped_short > 0:
        print(f"Dropped {n_dropped_short} papers do Title/Abstract quá ngắn")
    df_gold = df_gold.drop(columns=["title_words_tmp", "abs_words_tmp"])
    
    # Parse Fields
    df_gold["Fields_raw"] = df_gold["Fields"].fillna("").astype(str)
    df_gold["fields_list"] = df_gold["Fields_raw"].apply(
        lambda x: [
            tok for tok in (canonicalize_field_token(t) for t in x.split(";"))
            if tok is not None
        ]
    )
    
    # Drop duplicates within paper's fields list (just in case)
    df_gold["fields_list"] = df_gold["fields_list"].apply(lambda lst: list(dict.fromkeys(lst)))
    df_gold["n_fields"] = df_gold["fields_list"].apply(len)
    
    # Drop papers với 0 fields after cleaning (= research-only papers + 1 paper trống)
    n_before = len(df_gold)
    df_gold = df_gold[df_gold["n_fields"] > 0].copy()
    n_dropped = n_before - len(df_gold)
    print(f"Dropped {n_dropped} papers (research-only or empty Fields after dropping 'research')")
    print(f"Remaining: {len(df_gold)}")
    
    # Levels (default fallback to ALL when empty per codebook v2.1)
    df_gold["levels_list"] = df_gold["Level"].apply(canonicalize_level)
    df_gold["levels_list"] = df_gold["levels_list"].apply(
        lambda lst: lst if len(lst) > 0 else ["ALL"]
    )
    
    # Method
    df_gold["method_clean"] = df_gold["Method"].apply(canonicalize_method)
    n_no_method = df_gold["method_clean"].isna().sum()
    if n_no_method > 0:
        print(f"WARNING: {n_no_method} papers có Method không parse được (sẽ bị drop)")
    df_gold = df_gold[df_gold["method_clean"].notna()].copy()
    
    # Recover missing Total_ID by Title match against main sheet.
    # Source Excel "recoded- field & level" has ~430 rows where both Total_ID and "Mã bài"
    # are blank; these are the 2023 batch where IDs were never filled in. They all match
    # exactly one Year=2023 entry in the main sheet by Title, so we restrict the lookup
    # to Year=2023 to avoid ambiguity with same-title papers reported in earlier years.
    main_2023 = df_main[df_main["Year"] == 2023].copy()
    main_2023["_title_norm"] = main_2023["Title"].apply(normalize_whitespace)
    title_to_id_2023 = (
        main_2023.dropna(subset=["Total_ID"])
        .drop_duplicates(subset="_title_norm")
        .set_index("_title_norm")["Total_ID"]
    )
    mask_missing_id = df_gold["Total_ID"].isna()
    n_missing = int(mask_missing_id.sum())
    if n_missing > 0:
        recovered = df_gold.loc[mask_missing_id, "Title"].map(title_to_id_2023)
        df_gold.loc[mask_missing_id, "Total_ID"] = recovered
        n_recovered = int(recovered.notna().sum())
        print(f"Recovered Total_ID for {n_recovered}/{n_missing} rows via Title match (Year=2023)")

    # Join with main để lấy Year
    df_main_subset = df_main[["Total_ID", "Year"]].copy()
    df_main_subset["Total_ID"] = pd.to_numeric(df_main_subset["Total_ID"], errors="coerce")
    df_gold["Total_ID"] = pd.to_numeric(df_gold["Total_ID"], errors="coerce")
    df_gold = df_gold.merge(df_main_subset, on="Total_ID", how="left", suffixes=("", "_main"))
    
    # 430 papers Time_Period=NaN → Year=2023 (verified earlier qua title match)
    df_gold.loc[df_gold["Year"].isna() & df_gold["Time_Period"].isna(), "Year"] = 2023
    
    n_no_year = df_gold["Year"].isna().sum()
    if n_no_year > 0:
        print(f"WARNING: {n_no_year} papers thiếu Year (sẽ bị drop)")
        df_gold = df_gold[df_gold["Year"].notna()].copy()
    
    df_gold["Year"] = df_gold["Year"].astype(int)
    
    # Final select columns
    out = df_gold[[
        "Total_ID", "Year", "Title", "Abstract",
        "fields_list", "levels_list", "method_clean",
    ]].rename(columns={"method_clean": "method"}).copy()
    
    # Add binary indicator columns cho Fields (12 cột)
    for f in config.FIELDS_12:
        out[f"field_{config.FIELDS_12.index(f):02d}"] = out["fields_list"].apply(lambda lst: f in lst)
    
    # Add binary indicator cho Levels (6 cột)
    for l in config.LEVELS_6:
        out[f"level_{l}"] = out["levels_list"].apply(lambda lst: l in lst)
    
    return out


def process_main_2024(df_main: pd.DataFrame) -> pd.DataFrame:
    """
    Process 2024 papers from main sheet (617 papers).
    
    Apply SAME canonicalization as gold 2013-2023:
    - Drop 'research' label
    - Canonicalize Field aliases ('Special edu' → 'Special education', etc.)
    - Canonicalize Method (case + typo handling)
    - Canonicalize Levels
    
    Apply trash detection:
    - Drop if missing Title or Abstract
    - Drop if Title < 3 words or Abstract < 30 words
    - Drop if Method not parseable
    - Drop if Fields empty after canonicalization
    
    Output: same schema as gold (with binary indicator columns).
    """
    df = df_main[df_main["Year"] == 2024].copy()
    print(f"\n2024 papers from main sheet: {len(df)}")
    
    # Sanitize text
    df["Title"] = df["Title"].apply(normalize_whitespace)
    df["Abstract"] = df["Abstract"].apply(normalize_whitespace)
    
    # Trash detection: missing Title or Abstract
    n_before = len(df)
    df = df[(df["Title"] != "") & (df["Abstract"] != "")].copy()
    n_dropped_missing = n_before - len(df)
    if n_dropped_missing > 0:
        print(f"Dropped {n_dropped_missing} papers thiếu Title hoặc Abstract")
    
    # Trash detection: minimum word counts
    df["title_words"] = df["Title"].str.split().str.len()
    df["abs_words"] = df["Abstract"].str.split().str.len()
    n_before = len(df)
    df = df[
        (df["title_words"] >= config.MIN_TITLE_WORDS) &
        (df["abs_words"] >= config.MIN_ABSTRACT_WORDS)
    ].copy()
    n_dropped_short = n_before - len(df)
    if n_dropped_short > 0:
        print(f"Dropped {n_dropped_short} papers do Title/Abstract quá ngắn "
              f"(< {config.MIN_TITLE_WORDS} title words hoặc < {config.MIN_ABSTRACT_WORDS} abstract words)")
    df = df.drop(columns=["title_words", "abs_words"])
    
    # Canonicalize Fields (drop 'research', apply aliases)
    df["Fields_raw"] = df["Fields"].fillna("").astype(str)
    df["fields_list"] = df["Fields_raw"].apply(
        lambda x: [
            tok for tok in (canonicalize_field_token(t) for t in x.split(";"))
            if tok is not None
        ]
    )
    df["fields_list"] = df["fields_list"].apply(lambda lst: list(dict.fromkeys(lst)))
    df["n_fields"] = df["fields_list"].apply(len)
    
    # Trash detection: papers with no valid fields after canonicalization
    n_before = len(df)
    df = df[df["n_fields"] > 0].copy()
    n_dropped_empty = n_before - len(df)
    if n_dropped_empty > 0:
        print(f"Dropped {n_dropped_empty} papers có Fields rỗng sau khi canonicalize "
              f"(research-only, hoặc all-unknown labels)")
    
    # Canonicalize Levels (default ALL if empty per codebook v2.1)
    df["levels_list"] = df["Educational level"].apply(canonicalize_level)
    # Default fallback to ALL when empty (per codebook v2.1 rule)
    df["levels_list"] = df["levels_list"].apply(lambda lst: lst if len(lst) > 0 else ["ALL"])
    
    # Canonicalize Method
    df["method_clean"] = df["Method"].apply(canonicalize_method)
    n_before = len(df)
    df = df[df["method_clean"].notna()].copy()
    n_dropped_method = n_before - len(df)
    if n_dropped_method > 0:
        print(f"Dropped {n_dropped_method} papers do Method không parse được")
    
    # Final fields
    df["Total_ID"] = pd.to_numeric(df["Total_ID"], errors="coerce")
    df["Year"] = df["Year"].astype(int)
    
    out = df[[
        "Total_ID", "Year", "Title", "Abstract",
        "fields_list", "levels_list", "method_clean",
    ]].rename(columns={"method_clean": "method"}).copy()
    
    # Add binary indicator columns matching gold schema
    for f in config.FIELDS_12:
        out[f"field_{config.FIELDS_12.index(f):02d}"] = out["fields_list"].apply(lambda lst: f in lst)
    for l in config.LEVELS_6:
        out[f"level_{l}"] = out["levels_list"].apply(lambda lst: l in lst)
    
    print(f"\nFinal 2024 clean: {len(out)} papers (dropped {617 - len(out)} 'rác' records)")
    
    return out


def compute_codebook_hash() -> str:
    """SHA-256 hash of FIELDS_12, LEVELS_6, METHODS_5 + aliases for audit."""
    payload = json.dumps({
        "fields": config.FIELDS_12,
        "fields_aliases": config.FIELDS_ALIASES,
        "levels": config.LEVELS_6,
        "levels_aliases": config.LEVELS_ALIASES,
        "methods": config.METHODS_5,
        "methods_aliases": config.METHOD_ALIASES,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def main():
    print("=" * 80)
    print("Phase 0: Sanitization")
    print("=" * 80)
    
    # Ensure output dirs
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load
    df_gold = load_gold()
    df_main = load_main()
    print(f"Loaded gold sheet: {len(df_gold)} rows")
    print(f"Loaded main sheet: {len(df_main)} rows")
    
    # Process
    print("\n--- Processing gold dataset ---")
    out_gold = process_gold(df_gold, df_main)
    
    print("\n--- Processing main 2024 ---")
    out_2024 = process_main_2024(df_main)
    
    # Save
    out_gold.to_parquet(config.GOLD_PARQUET, index=False)
    out_2024.to_parquet(config.MAIN_2024_PARQUET, index=False)
    
    print("\n" + "=" * 80)
    print("OUTPUT SUMMARY")
    print("=" * 80)
    print(f"Gold (2013-2023):  {len(out_gold)} papers → {config.GOLD_PARQUET}")
    print(f"  Year distribution:")
    for y, n in out_gold["Year"].value_counts().sort_index().items():
        print(f"    {y}: {n}")
    print(f"  Avg Fields/paper: {out_gold['fields_list'].apply(len).mean():.2f}")
    print(f"  Avg Levels/paper: {out_gold['levels_list'].apply(len).mean():.2f}")
    print(f"\n  Field distribution:")
    for f in config.FIELDS_12:
        n = out_gold[f"field_{config.FIELDS_12.index(f):02d}"].sum()
        print(f"    {f:40s}: {n:5d} ({n/len(out_gold)*100:.1f}%)")
    print(f"\n  Method distribution:")
    print(out_gold["method"].value_counts().to_string())
    
    print(f"\n2024 papers (test set): {len(out_2024)} → {config.MAIN_2024_PARQUET}")
    print(f"  Avg Fields/paper: {out_2024['fields_list'].apply(len).mean():.2f}")
    print(f"  Avg Levels/paper: {out_2024['levels_list'].apply(len).mean():.2f}")
    print(f"\n  2024 Field distribution (sau canonicalize):")
    for f in config.FIELDS_12:
        n = out_2024[f"field_{config.FIELDS_12.index(f):02d}"].sum()
        print(f"    {f:40s}: {n:4d} ({n/len(out_2024)*100:.1f}%)")
    print(f"\n  2024 Method distribution:")
    print(out_2024["method"].value_counts().to_string())
    
    # Save codebook hash
    h = compute_codebook_hash()
    config.codebook_hash_path().write_text(h)
    print(f"\nCodebook v2.1 hash (SHA-256): {h[:16]}...")
    
    print("\n[DONE]")


if __name__ == "__main__":
    main()
