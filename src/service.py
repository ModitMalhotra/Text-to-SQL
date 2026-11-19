import json
import os
import time

from assumptions import flag_assumptions
from executor import run_sql
from llm import MODEL, STRONG_MODEL
from pipeline import answer
from pricing import costs
from router import answer_routed
from schema import build_schema
from semantic_cache import SemanticCache, conflicts

CACHE_PATH = os.path.join("cache", "semantic")
MAX_STORED_ROWS = 500


def fingerprint(db):
    info = os.stat(db)
    return f"{info.st_size}-{int(info.st_mtime)}"


def pack(cols, rows):
    if len(rows) > MAX_STORED_ROWS:
        return None
    return {"cols": list(cols), "rows": [list(r) for r in rows]}


def unpack(result):
    return result["cols"], [tuple(r) for r in result["rows"]]


def new_cache(embedder, fp):
    cache = SemanticCache(embedder)
    cache.meta = {"db": fp}
    return cache


def load_cache(path, embedder, fp):
    try:
        with open(path + ".json", encoding="utf-8") as f:
            saved = json.load(f)
        if (
            saved.get("model") == embedder.name
            and saved.get("meta", {}).get("db") == fp
        ):
            return SemanticCache.load(path, embedder)
    except (OSError, ValueError, KeyError):
        pass
    return new_cache(embedder, fp)


class Assistant:
    def __init__(
        self, db, embedder, cache_path=CACHE_PATH, cheap=MODEL, strong=STRONG_MODEL
    ):
        self.db = db
        self.cheap = cheap
        self.strong = strong
        self.cache_path = cache_path
        self.schema = build_schema(
            db, include_hints=True, include_samples=True, include_rules=True
        )
        self.fp = fingerprint(db)
        self.cache = (
            load_cache(cache_path, embedder, self.fp)
            if cache_path
            else new_cache(embedder, self.fp)
        )

    def persist(self):
        if self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
            self.cache.save(self.cache_path)

    def store(self, question, sql, cols=None, rows=None):
        result = pack(cols, rows) if rows is not None else None
        self.cache.add(question, sql, result)
        self.persist()

    def seed(self, items):
        known = set(self.cache.questions)
        fresh = [(q, sql) for q, sql in items if q not in known]
        self.cache.add_many(fresh)
        self.persist()
        return len(fresh)

    def clear_cache(self):
        self.cache = new_cache(self.cache.embedder, self.fp)
        self.persist()

    def ask(
        self,
        question,
        threshold=0.88,
        guard=True,
        retries=2,
        escalate_on_error=False,
        run=0,
        store=True,
    ):
        text = question.strip()
        if not text:
            raise ValueError("empty question")
        start = time.perf_counter()
        found = self.cache.search(text)
        hit = self.cache.pick(found, threshold, guard)
        top = self.cache.pick(found, threshold, False)
        blocked = None
        if guard and top and (hit is None or hit["index"] != top["index"]):
            blocked = {
                "matched_question": top["matched_question"],
                "score": round(top["score"], 3),
                "differs_on": conflicts(text, top["matched_question"]),
            }
        result = {
            "question": text,
            "source": "model",
            "sql": None,
            "cols": [],
            "rows": [],
            "error": None,
            "cache_score": None,
            "cache_matched": None,
            "cache_blocked": blocked,
            "cache_error": None,
            "cache_lookup_ms": round(found["ms"], 2),
            "cache_size": len(self.cache),
            "result_cached": False,
            "stored_in_cache": False,
            "assumptions": [],
            "model_used": None,
            "escalated": False,
            "escalation_reason": None,
            "attempts": 0,
            "llm_cached": None,
            "llm_latency_s": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "total_s": 0.0,
        }
        if hit:
            try:
                if hit["result"]:
                    cols, rows = unpack(hit["result"])
                    result["result_cached"] = True
                else:
                    cols, rows = run_sql(self.db, hit["sql"])
                    packed = pack(cols, rows)
                    if packed:
                        self.cache.set_result(hit["index"], packed)
                        self.persist()
                result.update(
                    source="semantic_cache",
                    sql=hit["sql"],
                    cols=cols,
                    rows=rows,
                    cache_score=round(hit["score"], 3),
                    cache_matched=hit["matched_question"],
                    total_s=round(time.perf_counter() - start, 3),
                )
                return result
            except Exception as exc:
                result["cache_error"] = (
                    f"cached SQL failed, fell back to the model: {exc}"
                )
        if escalate_on_error:
            res = answer_routed(
                text,
                self.schema,
                self.db,
                self.cheap,
                self.strong,
                run,
                retries,
                ("error",),
            )
        else:
            res = answer(text, self.schema, self.db, self.cheap, run, retries)
        stored = False
        if store and not res["error"] and res["rows"]:
            self.store(text, res["sql"], res["cols"], res["rows"])
            stored = True
        price = costs(res["usage"])
        result.update(
            sql=res["sql"],
            cols=res["cols"],
            rows=res["rows"],
            error=res["error"],
            stored_in_cache=stored,
            assumptions=flag_assumptions(text, res["sql"]),
            cache_size=len(self.cache),
            model_used=res.get("model_used", res.get("model", self.cheap)),
            escalated=res.get("escalated", False),
            escalation_reason=res.get("reason"),
            attempts=len(res["attempts"]),
            llm_cached=res["cached"],
            llm_latency_s=res["latency_s"],
            input_tokens=res["input_tokens"],
            output_tokens=res["output_tokens"],
            cost_usd=price["intro"] if price else None,
            total_s=round(time.perf_counter() - start, 3),
        )
        return result
