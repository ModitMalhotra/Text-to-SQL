import argparse
import json
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from executor import run_sql
from llm import MODEL, generate
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


def generate_with_retry(question, schema, model, run, attempts=5):
    for i in range(attempts):
        try:
            return generate(question, schema, model, run)
        except Exception:
            if i == attempts - 1:
                raise
            time.sleep(5 * (i + 1))


def evaluate_run(questions, gold, schema, model, run, labels):
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
            "latency_s": None,
            "input_tokens": None,
            "output_tokens": None,
            "cached": None,
        }
        try:
            gen = generate_with_retry(q["question"], schema, model, run)
        except Exception as e:
            entry["category"] = "api_error"
            entry["error"] = f"{type(e).__name__}: {e}"
            entries.append(entry)
            print(f"run{run} {q['id']} {q['difficulty']:6s} BAD  api_error")
            continue
        for k in ("latency_s", "input_tokens", "output_tokens", "cached"):
            entry[k] = gen[k]
        entry["pred_sql"] = gen["sql"]
        gold_cols, gold_rows = gold[q["id"]]
        try:
            pred_cols, pred_rows = run_sql(DB, gen["sql"])
            if same_result(gold_rows, pred_rows):
                entry["correct"] = True
                entry["category"] = "correct"
            else:
                entry["category"] = classify_wrong(gold_cols, gold_rows, pred_cols, pred_rows)
                entry["category"] = labels.get(q["id"], entry["category"])
        except Exception as e:
            entry["error"] = f"{type(e).__name__}: {e}"
            entry["category"] = classify_error(entry["error"])
        mark = "OK " if entry["correct"] else "BAD"
        print(f"run{run} {q['id']} {q['difficulty']:6s} {mark}  {entry['category'] if not entry['correct'] else ''}")
        entries.append(entry)
        if not gen["cached"]:
            time.sleep(1)
    return entries


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
    out["failure_categories_avg_per_run"] = {k: round(v / runs, 1) for k, v in cats.most_common()}
    by_q = defaultdict(list)
    for e in entries:
        by_q[e["id"]].append(e["correct"])
    out["unstable_questions"] = sorted(i for i, v in by_q.items() if 0 < sum(v) < len(v))
    out["always_wrong"] = sorted(i for i, v in by_q.items() if sum(v) == 0)
    lat = [e["latency_s"] for e in entries if e["latency_s"] is not None]
    out["mean_latency_s"] = round(statistics.mean(lat), 2) if lat else None
    out["input_tokens_per_run"] = round(sum(e["input_tokens"] or 0 for e in entries) / runs)
    out["output_tokens_per_run"] = round(sum(e["output_tokens"] or 0 for e in entries) / runs)
    return out


def print_summary(s, runs):
    print("\n" + "=" * 52)
    if runs > 1:
        print(f"Execution accuracy: {s['accuracy_mean']}% +/- {s['accuracy_std']} over {runs} runs {s['accuracy_per_run']}")
    else:
        print(f"Execution accuracy: {s['accuracy_mean']}%")
    for level, a in s["accuracy_by_difficulty"].items():
        print(f"  {level:6s} {a}%")
    print("Failure categories (avg per run):")
    for k, v in s["failure_categories_avg_per_run"].items():
        print(f"  {k:20s} {v}")
    if runs > 1:
        print(f"Unstable questions: {s['unstable_questions']}")
    print(f"Always wrong: {s['always_wrong']}")
    print(f"Mean latency: {s['mean_latency_s']}s | tokens/run: {s['input_tokens_per_run']} in, {s['output_tokens_per_run']} out")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(EVAL_DIR / "questions_hard_v2.json"))
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--no-hints", action="store_true")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    questions = json.loads(Path(args.questions).read_text())
    if args.limit:
        questions = questions[: args.limit]
    labels = json.loads(LABELS.read_text()) if LABELS.exists() else {}
    schema = build_schema(DB, include_hints=not args.no_hints)
    gold = {q["id"]: run_sql(DB, q["gold_sql"]) for q in questions}

    entries = []
    for run in range(args.runs):
        entries += evaluate_run(questions, gold, schema, args.model, run, labels)

    summary = summarize(entries, args.runs)
    print_summary(summary, args.runs)

    tag = "nohints" if args.no_hints else "hints"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RUNS_DIR / f"{stamp}_{Path(args.questions).stem}_{tag}_{args.model}"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"model": args.model, "hints": not args.no_hints, "questions": args.questions, "runs": args.runs}
    (out_dir / "results.json").write_text(json.dumps({"meta": meta, "entries": entries}, indent=2))
    (out_dir / "summary.json").write_text(json.dumps({"meta": meta, **summary}, indent=2))
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()