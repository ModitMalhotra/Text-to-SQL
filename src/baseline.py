import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from executor import UnsafeSQL, run_sql
from llm import MODEL, generate_sql
from schema import build_schema

ROOT = Path(__file__).resolve().parent.parent
DB = str(ROOT / "olist.db")
EVAL_DIR = ROOT / "eval"


def normalize(rows):
    cleaned = []
    for row in rows:
        cleaned.append(tuple(round(v, 2) if isinstance(v, float) else v for v in row))
    return sorted(cleaned, key=repr)


def same_result(gold_rows, pred_rows):
    return normalize(gold_rows) == normalize(pred_rows)


def ask_with_retry(question, schema, model, attempts=5):
    for i in range(attempts):
        try:
            return generate_sql(question, schema, model)
        except Exception as e:
            if i == attempts - 1:
                raise
            time.sleep(5 * (i + 1))


def evaluate(model, include_hints, questions_path, limit=None):
    schema = build_schema(DB, include_hints=include_hints)
    questions = json.loads(Path(questions_path).read_text())
    if limit:
        questions = questions[:limit]
    results = []
    for q in questions:
        _, gold_rows = run_sql(DB, q["gold_sql"])
        entry = {"id": q["id"], "difficulty": q["difficulty"], "question": q["question"], "gold_sql": q["gold_sql"]}
        try:
            pred_sql = ask_with_retry(q["question"], schema, model)
            entry["pred_sql"] = pred_sql
            _, pred_rows = run_sql(DB, pred_sql)
            entry["correct"] = same_result(gold_rows, pred_rows)
            entry["error"] = None
        except (UnsafeSQL, Exception) as e:
            entry["pred_sql"] = entry.get("pred_sql")
            entry["correct"] = False
            entry["error"] = f"{type(e).__name__}: {e}"
        mark = "OK " if entry["correct"] else "BAD"
        print(f"{entry['id']} {entry['difficulty']:6s} {mark} {entry['error'] or ''}")
        results.append(entry)
        time.sleep(1)
    return results


def summarize(results):
    by_level = defaultdict(lambda: [0, 0])
    for r in results:
        by_level[r["difficulty"]][1] += 1
        by_level[r["difficulty"]][0] += r["correct"]
    total = sum(r["correct"] for r in results)
    print(f"\nExecution accuracy: {total}/{len(results)} = {100 * total / len(results):.1f}%")
    for level in ("easy", "medium", "hard"):
        if level in by_level:
            c, n = by_level[level]
            print(f"  {level:6s} {c}/{n}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--no-hints", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--questions", default=str(EVAL_DIR / "questions.json"))
    args = ap.parse_args()
    res = evaluate(args.model, not args.no_hints, args.questions, args.limit)
    tag = "nohints" if args.no_hints else "hints"
    OUT = EVAL_DIR / f"results_{Path(args.questions).stem}_{tag}.json"
    OUT.write_text(json.dumps({"model": args.model, "hints": not args.no_hints, "results": res}, indent=2))
    summarize(res)
    print(f"\nSaved to {OUT}")