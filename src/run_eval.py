import argparse
import json
import statistics
import subprocess
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from executor import run_sql
from llm import MODEL, STRONG_MODEL
from fewshot import fewshot_block
from pipeline import answer
from pricing import PRICE_SETS, costs
from router import TRIGGERS, answer_routed
from schema import build_schema
from taxonomy import classify_error, classify_wrong

ROOT = Path(__file__).resolve().parent.parent
DB = str(ROOT / "olist.db")
EVAL_DIR = ROOT / "eval"
RUNS_DIR = EVAL_DIR / "runs"
LABELS = EVAL_DIR / "labels.json"
LEVELS = ("easy", "medium", "hard")


def canon(v):
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return v
    return v


def row_key(row):
    return tuple((0, "", v) if isinstance(v, float) else (1, str(v), 0.0) for v in row)


def values_match(a, b):
    if isinstance(a, float) and isinstance(b, float):
        if a.is_integer() and b.is_integer():
            return a == b
        return abs(a - b) <= 0.01
    return a == b


def same_result(gold_rows, pred_rows):
    if len(gold_rows) != len(pred_rows):
        return False
    g = sorted((tuple(canon(v) for v in r) for r in gold_rows), key=row_key)
    p = sorted((tuple(canon(v) for v in r) for r in pred_rows), key=row_key)
    for gr, pr in zip(g, p):
        if len(gr) != len(pr) or not all(values_match(a, b) for a, b in zip(gr, pr)):
            return False
    return True


def evaluate_run(questions, gold, schema, model, run, labels, retries, router=None):
    entries = []
    for q in questions:
        entry = {
            "run": run,
            "id": q["id"],
            "difficulty": q["difficulty"],
            "question": q["question"],
            "gold_sql": q["gold_sql"],
            "pred_sql": None,
            "correct": False,
            "category": None,
            "error": None,
            "attempts": None,
            "latency_s": None,
            "input_tokens": None,
            "output_tokens": None,
            "cached": None,
            "model_used": None,
            "escalated": False,
            "escalation_reason": None,
            "usage": None,
            "cost_usd": None,
        }
        try:
            if router:
                res = answer_routed(
                    q["question"],
                    schema,
                    DB,
                    model,
                    router["strong"],
                    run,
                    retries,
                    router["triggers"],
                    router["complex_joins"],
                    router["complex_selects"],
                )
            else:
                res = answer(q["question"], schema, DB, model, run, retries)
        except Exception as e:
            entry["category"] = "api_error"
            entry["error"] = f"{type(e).__name__}: {e}"
            entries.append(entry)
            print(f"run{run} {q['id']} {q['difficulty']:6s} BAD  api_error")
            continue
        entry["pred_sql"] = res["sql"]
        entry["attempts"] = len(res["attempts"])
        entry["model_used"] = res.get("model_used", model)
        entry["escalated"] = res.get("escalated", False)
        entry["escalation_reason"] = res.get("reason")
        entry["usage"] = res["usage"]
        entry["cost_usd"] = costs(res["usage"])
        for k in ("latency_s", "input_tokens", "output_tokens", "cached"):
            entry[k] = res[k]
        gold_cols, gold_rows = gold[q["id"]]
        if res["error"] is None:
            if same_result(gold_rows, res["rows"]):
                entry["correct"] = True
                entry["category"] = "correct"
            else:
                entry["category"] = classify_wrong(
                    gold_cols, gold_rows, res["cols"], res["rows"]
                )
                entry["category"] = labels.get(q["id"], entry["category"])
        else:
            entry["error"] = res["error"]
            entry["category"] = classify_error(res["error"])
        mark = "OK " if entry["correct"] else "BAD"
        note = entry["category"] if not entry["correct"] else ""
        if entry["correct"] and entry["escalated"]:
            note = f"escalated: {entry['escalation_reason']}"
        elif entry["correct"] and entry["attempts"] > 1:
            note = f"fixed after {entry['attempts']} attempts"
        elif entry["escalated"]:
            note = f"{note} (escalated: {entry['escalation_reason']})"
        print(f"run{run} {q['id']} {q['difficulty']:6s} {mark}  {note}")
        entries.append(entry)
    return entries


def git_state():
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=5,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=5,
        ).stdout.strip()
        if not commit:
            return {"commit": None, "uncommitted_changes": None}
        return {"commit": commit, "uncommitted_changes": bool(dirty)}
    except Exception:
        return {"commit": None, "uncommitted_changes": None}


def pct(a, b):
    return 100.0 * a / b if b else 0.0


def summarize(entries, runs):
    per_run = []
    for r in range(runs):
        es = [e for e in entries if e["run"] == r]
        per_run.append(pct(sum(e["correct"] for e in es), len(es)))
    out = {
        "accuracy_per_run": [round(a, 1) for a in per_run],
        "accuracy_mean": round(statistics.mean(per_run), 1),
        "accuracy_std": round(statistics.pstdev(per_run), 1) if runs > 1 else 0.0,
    }
    by_level = {}
    for level in LEVELS:
        accs = []
        for r in range(runs):
            es = [e for e in entries if e["run"] == r and e["difficulty"] == level]
            if es:
                accs.append(pct(sum(e["correct"] for e in es), len(es)))
        if accs:
            by_level[level] = round(statistics.mean(accs), 1)
    out["accuracy_by_difficulty"] = by_level
    cats = Counter(e["category"] for e in entries if not e["correct"])
    out["failure_categories_avg_per_run"] = {
        k: round(v / runs, 1) for k, v in cats.most_common()
    }
    by_q = defaultdict(list)
    for e in entries:
        by_q[e["id"]].append(e["correct"])
    tried = [e["attempts"] for e in entries if e["attempts"]]
    out["mean_attempts"] = round(statistics.mean(tried), 2) if tried else 1.0
    out["fixed_by_retry_per_run"] = round(
        sum(
            1
            for e in entries
            if e["correct"] and (e["attempts"] or 1) > 1 and not e.get("escalated")
        )
        / runs,
        1,
    )
    out["unstable_questions"] = sorted(
        i for i, v in by_q.items() if 0 < sum(v) < len(v)
    )
    out["always_wrong"] = sorted(i for i, v in by_q.items() if sum(v) == 0)
    lat = [e["latency_s"] for e in entries if e["latency_s"] is not None]
    out["mean_latency_s"] = round(statistics.mean(lat), 2) if lat else None
    out["input_tokens_per_run"] = round(
        sum(e["input_tokens"] or 0 for e in entries) / runs
    )
    out["output_tokens_per_run"] = round(
        sum(e["output_tokens"] or 0 for e in entries) / runs
    )
    esc = [e for e in entries if e.get("escalated")]
    kept = [e for e in entries if not e.get("escalated")]
    out["escalation_rate_pct"] = round(pct(len(esc), len(entries)), 1)
    out["escalation_reasons"] = dict(Counter(e["escalation_reason"] for e in esc))
    out["accuracy_escalated_pct"] = (
        round(pct(sum(e["correct"] for e in esc), len(esc)), 1) if esc else None
    )
    out["accuracy_kept_cheap_pct"] = (
        round(pct(sum(e["correct"] for e in kept), len(kept)), 1) if kept else None
    )
    if entries and all(e.get("cost_usd") for e in entries):
        correct = sum(e["correct"] for e in entries)
        out["cost_per_query_usd"] = {
            n: round(sum(e["cost_usd"][n] for e in entries) / len(entries), 6)
            for n in PRICE_SETS
        }
        out["cost_per_correct_usd"] = {
            n: (
                round(sum(e["cost_usd"][n] for e in entries) / correct, 6)
                if correct
                else None
            )
            for n in PRICE_SETS
        }
    else:
        out["cost_per_query_usd"] = None
        out["cost_per_correct_usd"] = None
    return out


def print_summary(s, runs):
    print("\n" + "=" * 52)
    if runs > 1:
        print(
            f"Execution accuracy: {s['accuracy_mean']}% +/- {s['accuracy_std']} over {runs} runs {s['accuracy_per_run']}"
        )
    else:
        print(f"Execution accuracy: {s['accuracy_mean']}%")
    for level, a in s["accuracy_by_difficulty"].items():
        print(f"  {level:6s} {a}%")
    print("Failure categories (avg per run):")
    for k, v in s["failure_categories_avg_per_run"].items():
        print(f"  {k:20s} {v}")
    if s["mean_attempts"] > 1:
        print(
            f"Mean attempts per question: {s['mean_attempts']} | fixed by retry (avg per run): {s['fixed_by_retry_per_run']}"
        )
    if runs > 1:
        print(f"Unstable questions: {s['unstable_questions']}")
    print(f"Always wrong: {s['always_wrong']}")
    print(
        f"Mean latency: {s['mean_latency_s']}s | tokens/run: {s['input_tokens_per_run']} in, {s['output_tokens_per_run']} out"
    )
    if s["escalation_rate_pct"]:
        print(
            f"Escalated to strong model: {s['escalation_rate_pct']}% of questions {s['escalation_reasons']}"
        )
        print(
            f"  accuracy on escalated: {s['accuracy_escalated_pct']}% | accuracy on kept-cheap: {s['accuracy_kept_cheap_pct']}%"
        )
    if s["cost_per_query_usd"]:
        for name in PRICE_SETS:
            print(
                f"Cost [{name}]: ${s['cost_per_query_usd'][name]} per query | ${s['cost_per_correct_usd'][name]} per correct answer"
            )
    else:
        print("Cost: not available (unpriced model or API errors in this run)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(EVAL_DIR / "questions_hard_v2.json"))
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--no-hints", action="store_true")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--retries", type=int, default=0)
    ap.add_argument("--fewshot", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--rules", action="store_true")
    ap.add_argument("--router", choices=["none", "cascade"], default="none")
    ap.add_argument("--strong-model", default=STRONG_MODEL)
    ap.add_argument("--escalate-on", default="error,complex")
    ap.add_argument("--complex-joins", type=int, default=3)
    ap.add_argument("--complex-selects", type=int, default=3)
    args = ap.parse_args()
    triggers = tuple(t for t in args.escalate_on.split(",") if t)
    for t in triggers:
        if t not in TRIGGERS:
            raise SystemExit(f"unknown trigger {t}, choose from {TRIGGERS}")
    router = None
    if args.router == "cascade":
        router = {
            "strong": args.strong_model,
            "triggers": triggers,
            "complex_joins": args.complex_joins,
            "complex_selects": args.complex_selects,
        }

    questions = json.loads(Path(args.questions).read_text())
    if args.limit:
        questions = questions[: args.limit]
    labels = json.loads(LABELS.read_text()) if LABELS.exists() else {}
    schema = build_schema(
        DB,
        include_hints=not args.no_hints,
        include_samples=args.samples,
        include_rules=args.rules,
    )
    if args.fewshot:
        schema = schema + "\n\n" + fewshot_block()
    gold = {q["id"]: run_sql(DB, q["gold_sql"]) for q in questions}

    entries = []
    for run in range(args.runs):
        entries += evaluate_run(
            questions, gold, schema, args.model, run, labels, args.retries, router
        )

    summary = summarize(entries, args.runs)
    print_summary(summary, args.runs)

    tag = (
        ("nohints" if args.no_hints else "hints")
        + ("_smp" if args.samples else "")
        + ("_rul" if args.rules else "")
        + ("_fs" if args.fewshot else "")
        + ("_cascade" if router else "")
    )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (
        RUNS_DIR
        / f"{stamp}_{Path(args.questions).stem}_{tag}_r{args.retries}_{args.model}{'-' + args.strong_model if router else ''}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "model": args.model,
        "hints": not args.no_hints,
        "questions": args.questions,
        "question_count": len(questions),
        "runs": args.runs,
        "retries": args.retries,
        "fewshot": args.fewshot,
        "samples": args.samples,
        "rules": args.rules,
        "router": router,
        "schema_chars": len(schema),
        "git": git_state(),
        "timestamp": stamp,
        "prices_per_1m_tokens": PRICE_SETS,
    }
    (out_dir / "results.json").write_text(
        json.dumps({"meta": meta, "entries": entries}, indent=2)
    )
    (out_dir / "summary.json").write_text(
        json.dumps({"meta": meta, **summary}, indent=2)
    )
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
