import time

from executor import run_sql
from llm import generate
from taxonomy import classify_error

TIMEOUT_MESSAGE = (
    "the query was stopped because it ran for too long; rewrite it to be more efficient"
)


def generate_with_retry(question, schema, model, run, feedback, attempts=5):
    for i in range(attempts):
        try:
            return generate(question, schema, model, run, feedback)
        except Exception:
            if i == attempts - 1:
                raise
            time.sleep(5 * (i + 1))


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
            time.sleep(1)
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
        "latency_s": round(sum(a["latency_s"] or 0 for a in attempts), 3),
        "input_tokens": sum(a["input_tokens"] or 0 for a in attempts),
        "output_tokens": sum(a["output_tokens"] or 0 for a in attempts),
        "cached": all(a["cached"] for a in attempts),
    }
