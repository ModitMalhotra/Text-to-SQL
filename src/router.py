import re

from pipeline import answer

STRING_LITERAL = re.compile(r"'[^']*'")
JOIN = re.compile(r"\bJOIN\b", re.I)
SELECT = re.compile(r"\bSELECT\b", re.I)
WINDOW = re.compile(r"\bOVER\s*\(", re.I)
CTE = re.compile(r"^\s*WITH\b", re.I)

TRIGGERS = ("error", "complex", "empty")


def sql_complexity(sql):
    text = STRING_LITERAL.sub("''", sql)
    return {
        "joins": len(JOIN.findall(text)),
        "selects": len(SELECT.findall(text)),
        "window": bool(WINDOW.search(text)),
        "cte": bool(CTE.match(text)),
    }


def is_complex(sql, complex_joins=3, complex_selects=3):
    c = sql_complexity(sql)
    return (
        c["joins"] >= complex_joins
        or c["selects"] >= complex_selects
        or c["window"]
        or c["cte"]
    )


def escalation_reason(res, triggers, complex_joins=3, complex_selects=3):
    if res["error"]:
        return "error" if "error" in triggers else None
    if "empty" in triggers and not res["rows"]:
        return "empty"
    if "complex" in triggers and is_complex(res["sql"], complex_joins, complex_selects):
        return "complex"
    return None


def answer_routed(
    question,
    schema,
    db,
    cheap,
    strong,
    run,
    max_retries=0,
    triggers=("error", "complex"),
    complex_joins=3,
    complex_selects=3,
):
    first = answer(question, schema, db, cheap, run, max_retries)
    reason = escalation_reason(first, triggers, complex_joins, complex_selects)
    if reason is None:
        return {**first, "model_used": cheap, "escalated": False, "reason": None}
    second = answer(question, schema, db, strong, run, max_retries)
    final = second
    used = strong
    if second["error"] and not first["error"]:
        final = first
        used = cheap
        reason = reason + "_strong_failed"
    return {
        **final,
        "attempts": first["attempts"] + second["attempts"],
        "latency_s": round(first["latency_s"] + second["latency_s"], 3),
        "input_tokens": first["input_tokens"] + second["input_tokens"],
        "output_tokens": first["output_tokens"] + second["output_tokens"],
        "usage": first["usage"] + second["usage"],
        "cached": first["cached"] and second["cached"],
        "model_used": used,
        "escalated": True,
        "reason": reason,
    }
