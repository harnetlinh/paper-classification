"""
OpenAI-only LLM ensemble client.

Strategy: Use 3 different OpenAI models for diversity instead of 3 providers.
- gpt-5.5: newest flagship, strongest reasoning
- gpt-5.4: previous flagship, with reasoning_effort param
- gpt-5.4-mini: fast + cheap for diversity

All models use temperature=0 + seed=SEED for reproducibility.
JSON mode (response_format={"type": "json_object"}) ensures structured outputs.

Cache: SHA-256 hash of (model, system, user, reasoning_effort) → cached JSON file.
       Re-runs are FREE — same prompts return cached output.

Resume: Caller (llm_augment.py) maintains a JSONL progress file in addition to cache.
"""
import os
import json
import time
import hashlib
from pathlib import Path
from typing import Optional, Tuple

from openai import OpenAI, APITimeoutError, APIConnectionError, RateLimitError
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import config

load_dotenv()


# ==================== Cache helpers ====================
def cache_key(model: str, system_prompt: str, user_prompt: str, extra: str = "") -> str:
    """Deterministic 16-char hex cache key."""
    payload = f"{model}|{system_prompt}|{user_prompt}|{extra}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# Errors that indicate a transient mount / sync issue (Google Drive on Colab
# is the main culprit — small-file access at scale triggers ConnectionAbortedError
# and OSError under heavy load). We retry these instead of failing the run.
_DRIVE_TRANSIENT_ERRORS = (ConnectionAbortedError, OSError)


def _retry_on_transient(op_name: str, fn, *args, max_attempts: int = 4, **kwargs):
    """Call fn(*args, **kwargs) with retry on transient filesystem errors.

    Used to wrap LLM cache + progress file operations on Colab where Google
    Drive sync produces brief connection aborts when many tiny files are
    accessed in quick succession.
    """
    delay = 0.5
    for attempt in range(1, max_attempts + 1):
        try:
            return fn(*args, **kwargs)
        except _DRIVE_TRANSIENT_ERRORS as exc:
            # FileNotFoundError is technically OSError but it usually means
            # the parent directory was unmounted. Re-raise on final attempt.
            if attempt == max_attempts:
                raise
            print(f"  [retry] {op_name} hit {type(exc).__name__}: {exc} "
                  f"(attempt {attempt}/{max_attempts}, sleep {delay:.1f}s)")
            time.sleep(delay)
            delay = min(delay * 2, 5.0)
    return None  # unreachable


def get_cache_path(model_alias: str, key: str) -> Path:
    cache_dir = config.LLM_LOG_DIR / model_alias
    # Defensive mkdir each call — Drive symlinks can lose state between init
    # and read/write on Colab.
    _retry_on_transient(
        "mkdir cache_dir",
        lambda: cache_dir.mkdir(parents=True, exist_ok=True),
    )
    return cache_dir / f"{key}.json"


def load_cached(model_alias: str, key: str) -> Optional[dict]:
    p = get_cache_path(model_alias, key)

    def _check_and_read():
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    return _retry_on_transient(f"load_cached({key})", _check_and_read)


def save_cache(model_alias: str, key: str, response: dict) -> None:
    p = get_cache_path(model_alias, key)
    payload = json.dumps(response, ensure_ascii=False, indent=2)
    _retry_on_transient(
        f"save_cache({key})",
        lambda: p.write_text(payload, encoding="utf-8"),
    )


def parse_json_response(text: str) -> dict:
    """Strip markdown fences if present, then parse JSON. Raises on failure."""
    if text is None:
        raise ValueError("Empty response (text is None)")
    text = text.strip()
    if not text:
        raise ValueError("Empty response (whitespace only)")
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]   # drop first line (``` or ```json)
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return json.loads(text)


# ==================== Single OpenAI model client ====================
RETRYABLE_EXCEPTIONS = (APITimeoutError, APIConnectionError, RateLimitError)


class OpenAIModelClient:
    """
    Wrapper around one OpenAI model.
    
    Args:
        model_id: e.g. 'gpt-5.5', 'gpt-5.4', 'gpt-5.4-mini'
        alias:    short name for cache directory (e.g. 'gpt55')
        reasoning_effort: 'low'/'medium'/'high'/'xhigh' or None
    """
    def __init__(self, model_id: str, alias: str, reasoning_effort: Optional[str] = None):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Create .env file with OPENAI_API_KEY=sk-..."
            )
        self.client = OpenAI(
            api_key=api_key,
            timeout=config.LLM_REQUEST_TIMEOUT,
        )
        self.model = model_id
        self.alias = alias
        self.reasoning_effort = reasoning_effort
        self.name = alias
    
    @retry(
        stop=stop_after_attempt(config.LLM_RETRY_MAX),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        reraise=True,
    )
    def _call_chat(self, system: str, user: str) -> Tuple[Optional[str], dict]:
        """Call Chat Completions API. Returns (text, usage_info)."""
        kwargs = {
            "model": self.model,
            "temperature": 0.0,
            "seed": config.SEED,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        
        resp = self.client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content if resp.choices else None
        usage = {
            "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
            "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
            "total_tokens": resp.usage.total_tokens if resp.usage else 0,
        }
        return text, usage
    
    def classify(self, system: str, user: str, use_cache: bool = True) -> dict:
        """
        Classify a paper. Returns parsed JSON dict with `_meta` field.
        On error, returns dict with `_error` field (still has `_meta`).
        """
        key = cache_key(self.model, system, user, str(self.reasoning_effort or ""))
        
        if use_cache:
            cached = load_cached(self.alias, key)
            if cached is not None:
                return cached
        
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        try:
            text, usage = self._call_chat(system, user)
        except Exception as e:
            return {
                "_error": f"API call failed: {type(e).__name__}: {e}",
                "_meta": {"model": self.model, "alias": self.alias, "usage": usage},
            }
        
        try:
            parsed = parse_json_response(text)
        except (json.JSONDecodeError, ValueError) as e:
            parsed = {"_error": f"JSON parse failed: {e}", "_raw": text}

        parsed["_meta"] = {
            "model": self.model,
            "alias": self.alias,
            "usage": usage,
        }
        # Cache only successful parses. If we cache the error response, every
        # subsequent run with the same prompt re-reads the cached failure
        # without ever retrying the model — the user would have to know to
        # pass --no-cache to recover. Keep transient errors transient.
        if "_error" not in parsed:
            save_cache(self.alias, key, parsed)
        time.sleep(config.LLM_RATE_LIMIT_PAUSE)
        return parsed


# ==================== Factory ====================
def get_clients(panel_config: Optional[list] = None) -> list:
    """
    Instantiate the OpenAI ensemble.
    Default: 3 models from config.OPENAI_PANEL.
    """
    if panel_config is None:
        panel_config = config.OPENAI_PANEL
    
    return [
        OpenAIModelClient(
            model_id=p["model"],
            alias=p["alias"],
            reasoning_effort=p.get("reasoning_effort"),
        )
        for p in panel_config
    ]


# ==================== Cost estimation ====================
PRICING = {
    "gpt-5.5":      (5.00, 30.00),
    "gpt-5.4":      (2.50, 10.00),
    "gpt-5.4-mini": (0.75,  4.50),
    "gpt-5.4-nano": (0.20,  1.25),
    "gpt-5.5-pro":  (60.00, 360.00),
}


def estimate_cost(panel: list, n_calls: int,
                  avg_input_tokens: int = 800,
                  avg_output_tokens: int = 200) -> dict:
    """
    Estimate USD cost.
    Default token estimates based on Special Edu filter prompt:
        ~800 input (system + title + abstract) + ~200 output (JSON response)
    """
    total = 0.0
    breakdown = []
    for client in panel:
        prices = PRICING.get(client.model, (2.50, 10.00))
        in_cost = (n_calls * avg_input_tokens / 1_000_000) * prices[0]
        out_cost = (n_calls * avg_output_tokens / 1_000_000) * prices[1]
        sub = in_cost + out_cost
        total += sub
        breakdown.append({
            "model": client.model,
            "n_calls": n_calls,
            "input_cost_usd": round(in_cost, 4),
            "output_cost_usd": round(out_cost, 4),
            "subtotal_usd": round(sub, 4),
        })
    return {"total_usd": round(total, 4), "breakdown": breakdown}


# ==================== Progress file (resume support) ====================
def _json_default(o):
    """JSON encoder fallback for numpy/pandas types."""
    try:
        import numpy as np
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    return str(o)


class ProgressTracker:
    """
    JSONL append-only progress file for resume support.
    Each line: {"id": ..., "status": "done"|"error", "result": {...}, "ts": ...}
    """
    def __init__(self, task_name: str):
        self.task_name = task_name
        self.path = config.LLM_PROGRESS_DIR / f"{task_name}.jsonl"
        # Defensive mkdir with retry — Drive symlinks can lose state.
        _retry_on_transient(
            "tracker mkdir parent",
            lambda: self.path.parent.mkdir(parents=True, exist_ok=True),
        )

    def _ensure_parent(self) -> None:
        """Re-create parent dir before each write — defends against Drive
        unmount/remount cycles on Colab where the dir may transiently vanish."""
        _retry_on_transient(
            "tracker re-mkdir parent",
            lambda: self.path.parent.mkdir(parents=True, exist_ok=True),
        )
    
    def load_done_ids(self) -> set:
        """Return set of IDs already processed (status='done')."""
        if not self.path.exists():
            return set()
        done = set()
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("status") == "done":
                        done.add(rec["id"])
                except json.JSONDecodeError:
                    continue
        return done
    
    def load_results(self) -> dict:
        """Return dict {id: result} for all done records (last write wins)."""
        if not self.path.exists():
            return {}
        results = {}
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("status") == "done":
                        results[rec["id"]] = rec.get("result")
                except json.JSONDecodeError:
                    continue
        return results
    
    def append(self, paper_id, status: str, result: Optional[dict] = None) -> None:
        """Append one record. Flushes immediately. Defensive against Drive
        sync errors via parent re-mkdir + retry on transient OSError."""
        rec = {
            "id": paper_id,
            "status": status,
            "result": result,
            "ts": time.time(),
        }
        payload = json.dumps(rec, ensure_ascii=False, default=_json_default) + "\n"

        def _write():
            self._ensure_parent()   # idempotent — retried inside _retry_on_transient
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(payload)
                f.flush()

        _retry_on_transient(f"append({paper_id})", _write)
