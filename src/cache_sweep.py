import argparse
import glob
import json
import os
import statistics

from executor import run_sql
from run_eval import same_result
from semantic_cache import Embedder, HashEmbedder, SemanticCache, conflicts

DEFAULT_LLM_LATENCY = 1.5


def parse_range(text):
    lo, hi, step = (float(x) for x in text.split(":"))
    count = int(round((hi - lo) / step)) + 1
    return [round(lo + i * step, 4) for i in range(count)]


def latest_llm_latency(runs_dir):
    best = None
    for path in glob.glob(os.path.join(runs_dir, "*", "summary.json")):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            continue
        latency = data.get("mean_latency_s")
        if latency is None:
            continue
        mtime = os.path.getmtime(path)
        if best is None or mtime > best[0]:
            best = (mtime, latency, os.path.basename(os.path.dirname(path)))
    return best


def execute(db, sql, memo):
    if sql not in memo:
        try:
            memo[sql] = run_sql(db, sql)[1]
        except Exception:
            memo[sql] = None
    return memo[sql]


def decide(cache, found, threshold, guard, gold_rows, db, memo):
    hit = cache.pick(found, threshold, guard)
    if hit is None:
        return None
    rows = execute(db, hit["sql"], memo)
    hit["correct"] = rows is not None and same_result(gold_rows, rows)
    return hit


def pct(num, den):
    return round(100.0 * num / den, 1) if den else 0.0


def metrics(queries, decisions, lookup_s, llm_s):
    n = len(queries)
    para = [i for i, q in enumerate(queries) if q["kind"] == "paraphrase"]
    other = [i for i, q in enumerate(queries) if q["kind"] != "paraphrase"]
    near = [i for i, q in enumerate(queries) if q["kind"] == "near_miss"]
    novel = [i for i, q in enumerate(queries) if q["kind"] == "novel"]
    hits = [i for i, d in enumerate(decisions) if d]
    wrong = [i for i in hits if not decisions[i]["correct"]]
    correct = [i for i in hits if decisions[i]["correct"]]
    para_correct = [i for i in para if decisions[i] and decisions[i]["correct"]]
    para_wrong = [i for i in para if decisions[i] and not decisions[i]["correct"]]
    other_wrong = [i for i in other if decisions[i] and not decisions[i]["correct"]]
    near_wrong = [i for i in near if decisions[i] and not decisions[i]["correct"]]
    novel_wrong = [i for i in novel if decisions[i] and not decisions[i]["correct"]]
    with_cache = (len(correct) * lookup_s + (n - len(correct)) * (lookup_s + llm_s)) / n
    return {
        "hits": len(hits),
        "hit_rate_pct": pct(len(hits), n),
        "para_correct_pct": pct(len(para_correct), len(para)),
        "para_wrong": len(para_wrong),
        "wrong_hits": len(wrong),
        "wrong_per_hit_pct": pct(len(wrong), len(hits)),
        "nonpara_wrong_pct": pct(len(other_wrong), len(other)),
        "near_miss_wrong_pct": pct(len(near_wrong), len(near)),
        "novel_wrong_pct": pct(len(novel_wrong), len(novel)),
        "latency_with_cache_s": round(with_cache, 3),
        "latency_reduction_pct": round(100.0 * (1 - with_cache / llm_s), 1) + 0.0,
    }


def print_table(title, rows):
    print(f"\n{title}")
    print(
        "thr    hits  para_ok%  para_wrong  wrong_hits  wrong/hit%  nearmiss_wrong%  novel_wrong%  latency_cut%"
    )
    for r in rows:
        print(
            f"{r['threshold']:<6} {r['hits']:<5} {r['para_correct_pct']:<9} {r['para_wrong']:<11} "
            f"{r['wrong_hits']:<11} {r['wrong_per_hit_pct']:<11} {r['near_miss_wrong_pct']:<16} "
            f"{r['novel_wrong_pct']:<13} {r['latency_reduction_pct']}"
        )


def recommend(rows, max_wrong):
    ok = [r for r in rows if r["wrong_hits"] <= max_wrong]
    if not ok:
        return None
    best = max(r["para_correct_pct"] for r in ok)
    return max(
        (r for r in ok if r["para_correct_pct"] == best), key=lambda r: r["threshold"]
    )


def plot(path, rows_by_guard, chosen):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return
    surface, ink, muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"
    blue, orange = "#2a78d6", "#eb6834"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, facecolor=surface)
    titles = {False: "Similarity threshold only", True: "Threshold + slot guard"}
    for ax, guard in zip(axes, (False, True)):
        rows = rows_by_guard[guard]
        xs = [r["threshold"] for r in rows]
        good = [r["para_correct_pct"] for r in rows]
        bad = [r["nonpara_wrong_pct"] for r in rows]
        ax.set_facecolor(surface)
        ax.plot(
            xs,
            good,
            color=blue,
            linewidth=1.6,
            marker="o",
            markersize=5,
            label="Paraphrases answered correctly from cache",
        )
        ax.plot(
            xs,
            bad,
            color=orange,
            linewidth=1.6,
            marker="s",
            markersize=5,
            label="Near-miss and new questions wrongly answered from cache",
        )
        pick = chosen.get(guard)
        if pick:
            ax.axvline(
                pick["threshold"], color=muted, linewidth=1, linestyle=(0, (4, 3))
            )
            left = pick["threshold"] > 0.92
            ax.text(
                pick["threshold"] + (-0.004 if left else 0.004),
                100,
                f"chosen {pick['threshold']}",
                color=muted,
                fontsize=8,
                ha="right" if left else "left",
                va="top",
            )
        ax.set_title(titles[guard], color=ink, fontsize=11, loc="left", pad=16)
        ax.set_xlabel("Similarity threshold", color=muted, fontsize=9)
        ax.set_ylim(-3, 108)
        ax.grid(axis="y", color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
        ax.tick_params(colors=muted, labelsize=8)
    axes[0].set_ylabel("% of queries in that group", color=muted, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=8,
        labelcolor=ink,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=160, facecolor=surface)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="eval/cache_pairs.json")
    ap.add_argument("--db", default="olist.db")
    ap.add_argument("--thresholds", default="0.60:0.98:0.02")
    ap.add_argument("--llm-latency", type=float, default=None)
    ap.add_argument("--max-wrong", type=int, default=0)
    ap.add_argument("--inspect", type=float, default=0.85)
    ap.add_argument("--embedder", default="minilm", choices=["minilm", "hash"])
    ap.add_argument("--out", default="eval/cache")
    args = ap.parse_args()

    with open(args.pairs, encoding="utf-8") as f:
        data = json.load(f)
    seed, queries = data["seed"], data["queries"]

    embedder = Embedder() if args.embedder == "minilm" else HashEmbedder()
    embedder.encode(["warm up"])
    cache = SemanticCache(embedder)
    cache.add_many([(s["question"], s["gold_sql"]) for s in seed])

    memo = {}
    gold = []
    for q in queries:
        rows = execute(args.db, q["gold_sql"], memo)
        if rows is None:
            raise SystemExit(f"gold SQL failed for {q['id']}")
        gold.append(rows)

    found = [cache.search(q["question"]) for q in queries]
    lookup_s = statistics.mean(f["ms"] for f in found) / 1000.0

    if args.llm_latency is not None:
        llm_s, source = args.llm_latency, "--llm-latency"
    else:
        latest = latest_llm_latency(os.path.join("eval", "runs"))
        llm_s, source = (
            (latest[1], latest[2]) if latest else (DEFAULT_LLM_LATENCY, "default")
        )

    print(f"embedder={embedder.name} seeds={len(cache)} queries={len(queries)}")
    print(
        f"mean lookup {lookup_s * 1000:.1f} ms | mean LLM latency {llm_s}s (from {source})"
    )

    top = []
    for q, f, rows in zip(queries, found, gold):
        idx = int(f["order"][0])
        sql_rows = execute(args.db, cache.sqls[idx], memo)
        top.append(
            {
                "id": q["id"],
                "kind": q["kind"],
                "question": q["question"],
                "top_score": round(float(f["scores"][idx]), 4),
                "top_match": cache.questions[idx],
                "top_correct": sql_rows is not None and same_result(rows, sql_rows),
            }
        )

    print(
        "\nTop-1 similarity by query type: mean [min, max], share of top-1 matches whose SQL is correct"
    )
    for kind in ("paraphrase", "near_miss", "novel"):
        items = [t for t in top if t["kind"] == kind]
        scores = [t["top_score"] for t in items]
        ok = sum(t["top_correct"] for t in items)
        print(
            f"  {kind:<11} {statistics.mean(scores):.3f} [{min(scores):.3f}, {max(scores):.3f}]  correct {ok}/{len(items)}"
        )

    thresholds = parse_range(args.thresholds)
    rows_by_guard = {False: [], True: []}
    decisions_by = {}
    for guard in (False, True):
        for thr in thresholds:
            decisions = [
                decide(cache, f, thr, guard, g, args.db, memo)
                for f, g in zip(found, gold)
            ]
            decisions_by[(guard, thr)] = decisions
            row = {
                "threshold": thr,
                "guard": guard,
                **metrics(queries, decisions, lookup_s, llm_s),
            }
            rows_by_guard[guard].append(row)
        print_table(
            "slot guard ON" if guard else "slot guard OFF", rows_by_guard[guard]
        )

    chosen = {g: recommend(rows_by_guard[g], args.max_wrong) for g in (False, True)}
    print(f"\nBest threshold with at most {args.max_wrong} wrong hits:")
    for guard in (False, True):
        r = chosen[guard]
        name = "guard ON " if guard else "guard OFF"
        if r:
            print(
                f"  {name} threshold {r['threshold']}: {r['para_correct_pct']}% of paraphrases answered from cache, latency cut {r['latency_reduction_pct']}%"
            )
        else:
            print(f"  {name} no threshold in the sweep has that few wrong hits")

    nearest = min(thresholds, key=lambda t: abs(t - args.inspect))
    print(f"\nWrong hits at threshold {nearest}:")
    for guard in (False, True):
        print(f"  guard {'ON' if guard else 'OFF'}")
        decisions = decisions_by[(guard, nearest)]
        shown = 0
        for q, d in zip(queries, decisions):
            if d and not d["correct"]:
                shown += 1
                print(
                    f"    {q['id']} {q['kind']} {d['score']:.3f}\n      asked:   {q['question']}\n      matched: {d['matched_question']}"
                )
        if not shown:
            print("    none")
    print(f"\nParaphrases the guard blocked at threshold {nearest}:")
    off, on = decisions_by[(False, nearest)], decisions_by[(True, nearest)]
    blocked = 0
    for q, a, b in zip(queries, off, on):
        if (
            q["kind"] == "paraphrase"
            and a
            and a["correct"]
            and not (b and b["correct"])
        ):
            blocked += 1
            print(
                f"    {q['id']} conflict on {conflicts(q['question'], a['matched_question'])}: {q['question']}"
            )
    if not blocked:
        print("    none")

    os.makedirs(args.out, exist_ok=True)
    report = {
        "config": {
            "embedder": embedder.name,
            "seeds": len(cache),
            "queries": len(queries),
            "per_kind": {
                k: sum(1 for q in queries if q["kind"] == k)
                for k in ("paraphrase", "near_miss", "novel")
            },
            "thresholds": thresholds,
            "mean_lookup_ms": round(lookup_s * 1000, 2),
            "llm_latency_s": llm_s,
            "llm_latency_source": source,
            "max_wrong": args.max_wrong,
        },
        "top1": top,
        "sweep": rows_by_guard[False] + rows_by_guard[True],
        "chosen": {("guard_on" if g else "guard_off"): r for g, r in chosen.items()},
    }
    with open(os.path.join(args.out, "sweep.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    plot(os.path.join(args.out, "sweep.png"), rows_by_guard, chosen)
    print(f"\nSaved to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
