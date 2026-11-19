import os
import re
import time

from executor import run_sql
from llm import generate
from taxonomy import classify_error

INTERVAL_S = float(os.environ.get("CALL_INTERVAL_S", "1"))
MAX_WAIT_S = 90
DAILY_QUOTA = re.compile(r"per\s*day", re.I)
RETRY_DELAY = re.compile(r"retry in ([\d.]+)\s*s|'retryDelay': '(\d+)s'", re.I)
TIMEOUT_MESSAGE = (
    "the query was stopped because it ran for too long; rewrite it to be more efficient"
)


class QuotaExhausted(Exception):
    pass


def is_quota_error(exc):
    text = str(exc)
    return "429" in text or "RESOURCE_EXHAUSTED" in text


def suggested_delay(exc):
    found = RETRY_DELAY.search(str(exc))
    if not found:
        return None
    return float(found.group(1) or found.group(2))


def wait_before_retry(exc, model, attempt):
    if not is_quota_error(exc):
        return 5 * (attempt + 1)
    delay = suggested_delay(exc)
    if DAILY_QUOTA.search(str(exc)) or (delay is not None and delay > MAX_WAIT_S):
        raise QuotaExhausted(
            f"Daily request quota exhausted for {model}. Calls already made are cached. "
            "Try again after the reset (midnight Pacific time) or enable billing."
        ) from exc
    if delay is None:
        return 5 * (attempt + 1)
    return min(delay + 1, MAX_WAIT_S)


def generate_with_retry(question, schema, model, run, feedback, attempts=5):
    for i in range(attempts):
        try:
            return generate(question, schema, model, run, feedback)
        except Exception as exc:
            wait = wait_before_retry(exc, model, i)
            if i == attempts - 1:
                raise
            time.sleep(wait)


def describe_error(exc):
    message = f"{type(exc).__name__}: {exc}"
    if classify_error(message) == "timeout":
        return TIMEOUT_MESSAGE
    return message


def answer(question, schema, db, model, run, max_retries=0):
    feedback = []
    attempts = []
    last_error = None
    cols, rows = [], []
    for _ in range(max_retries + 1):
        gen = generate_with_retry(question, schema, model, run, feedback or None)
        if not gen["cached"]:
            time.sleep(INTERVAL_S)
        attempts.append(gen)
        try:
            cols, rows = run_sql(db, gen["sql"])
            last_error = None
            break
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            feedback.append((gen["sql"], describe_error(e)))
    return {
        "sql": attempts[-1]["sql"],
        "cols": cols,
        "rows": rows,
        "error": last_error,
        "attempts": attempts,
        "model": model,
        "usage": [
            {
                "model": model,
                "input_tokens": a["input_tokens"] or 0,
                "output_tokens": a["output_tokens"] or 0,
            }
            for a in attempts
        ],
        "latency_s": round(sum(a["latency_s"] or 0 for a in attempts), 3),
        "input_tokens": sum(a["input_tokens"] or 0 for a in attempts),
        "output_tokens": sum(a["output_tokens"] or 0 for a in attempts),
        "cached": all(a["cached"] for a in attempts),
    }
