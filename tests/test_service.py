import pytest

import service
from pipeline import QuotaExhausted
from service import Assistant


def fake_answer(sql, error=None):
    def run(question, schema, db, model, run_id, retries):
        from executor import run_sql

        cols, rows = ([], [])
        if not error:
            cols, rows = run_sql(db, sql)
        return {
            "sql": sql,
            "cols": cols,
            "rows": rows,
            "error": error,
            "attempts": [{}],
            "model": model,
            "usage": [{"model": model, "input_tokens": 100, "output_tokens": 20}],
            "latency_s": 0.4,
            "input_tokens": 100,
            "output_tokens": 20,
            "cached": False,
        }

    return run


@pytest.fixture
def assistant(db, embedder, monkeypatch, tmp_path):
    monkeypatch.setattr(service, "build_schema", lambda *a, **k: "schema")
    return Assistant(db, embedder, str(tmp_path / "sem"))


def test_empty_question_rejected(assistant):
    with pytest.raises(ValueError):
        assistant.ask("   ")


def test_miss_calls_model_and_stores(assistant, monkeypatch):
    monkeypatch.setattr(service, "answer", fake_answer("SELECT COUNT(*) FROM orders"))
    res = assistant.ask("How many orders are there?")
    assert res["source"] == "model"
    assert res["rows"] == [(4,)]
    assert res["stored_in_cache"]
    assert res["cost_usd"] > 0
    assert len(assistant.cache) == 1


def test_repeat_question_hits_cache_without_model(assistant, monkeypatch):
    monkeypatch.setattr(service, "answer", fake_answer("SELECT COUNT(*) FROM orders"))
    assistant.ask("How many orders are there?")

    def fail(*a, **k):
        raise AssertionError("model must not be called on a cache hit")

    monkeypatch.setattr(service, "answer", fail)
    res = assistant.ask("How many orders are there?")
    assert res["source"] == "semantic_cache"
    assert res["cache_score"] >= 0.99
    assert res["rows"] == [(4,)]
    assert res["cost_usd"] == 0.0


def test_guard_blocks_conflicting_numbers(assistant, monkeypatch):
    monkeypatch.setattr(service, "answer", fake_answer("SELECT COUNT(*) FROM orders"))
    assistant.ask("Show the top 5 orders by status")
    monkeypatch.setattr(service, "answer", fake_answer("SELECT 1"))
    res = assistant.ask("Show the top 10 orders by status", threshold=0.5)
    assert res["source"] == "model"
    assert res["cache_blocked"]
    assert res["cache_blocked"]["differs_on"]


def test_guard_off_allows_conflicting_hit(assistant, monkeypatch):
    monkeypatch.setattr(service, "answer", fake_answer("SELECT COUNT(*) FROM orders"))
    assistant.ask("Show the top 5 orders by status")
    res = assistant.ask("Show the top 10 orders by status", threshold=0.5, guard=False)
    assert res["source"] == "semantic_cache"


def test_failed_query_is_not_cached(assistant, monkeypatch):
    monkeypatch.setattr(
        service,
        "answer",
        fake_answer("SELECT nope", error="OperationalError: no such column"),
    )
    res = assistant.ask("Break things")
    assert res["error"]
    assert not res["stored_in_cache"]
    assert len(assistant.cache) == 0


def test_empty_result_is_not_cached(assistant, monkeypatch):
    monkeypatch.setattr(
        service, "answer", fake_answer("SELECT * FROM orders WHERE order_id = 'zzz'")
    )
    res = assistant.ask("Find a missing order")
    assert res["rows"] == []
    assert not res["stored_in_cache"]


def test_broken_cached_sql_falls_back_to_model(assistant, monkeypatch):
    assistant.cache.add("How many orders are there?", "SELECT nope FROM orders")
    monkeypatch.setattr(service, "answer", fake_answer("SELECT COUNT(*) FROM orders"))
    res = assistant.ask("How many orders are there?")
    assert res["source"] == "model"
    assert "cached SQL failed" in res["cache_error"]
    assert res["rows"] == [(4,)]


def test_quota_error_propagates(assistant, monkeypatch):
    def boom(*a, **k):
        raise QuotaExhausted("daily limit")

    monkeypatch.setattr(service, "answer", boom)
    with pytest.raises(QuotaExhausted):
        assistant.ask("anything at all")


def test_cache_persists_across_instances(db, embedder, monkeypatch, tmp_path):
    monkeypatch.setattr(service, "build_schema", lambda *a, **k: "schema")
    path = str(tmp_path / "sem")
    first = Assistant(db, embedder, path)
    monkeypatch.setattr(service, "answer", fake_answer("SELECT COUNT(*) FROM orders"))
    first.ask("How many orders are there?")
    second = Assistant(db, embedder, path)
    assert len(second.cache) == 1


def test_cache_from_other_embedder_is_ignored(db, embedder, monkeypatch, tmp_path):
    from semantic_cache import HashEmbedder

    monkeypatch.setattr(service, "build_schema", lambda *a, **k: "schema")
    path = str(tmp_path / "sem")
    first = Assistant(db, embedder, path)
    monkeypatch.setattr(service, "answer", fake_answer("SELECT COUNT(*) FROM orders"))
    first.ask("How many orders are there?")
    other = Assistant(db, HashEmbedder(dim=128), path)
    assert len(other.cache) == 0


def test_seed_skips_known_questions(assistant):
    items = [("q one", "SELECT 1"), ("q two", "SELECT 2")]
    assert assistant.seed(items) == 2
    assert assistant.seed(items) == 0
    assert len(assistant.cache) == 2


def test_escalate_on_error_uses_router(assistant, monkeypatch):
    seen = {}

    def fake_routed(question, schema, db, cheap, strong, run_id, retries, triggers):
        seen["triggers"] = triggers
        out = fake_answer("SELECT COUNT(*) FROM orders")(
            question, schema, db, strong, run_id, retries
        )
        return {**out, "model_used": strong, "escalated": True, "reason": "error"}

    monkeypatch.setattr(service, "answer_routed", fake_routed)
    res = assistant.ask("How many orders are there?", escalate_on_error=True)
    assert seen["triggers"] == ("error",)
    assert res["escalated"]
    assert res["model_used"] == assistant.strong


def test_store_false_skips_cache_until_confirmed(assistant, monkeypatch):
    monkeypatch.setattr(service, "answer", fake_answer("SELECT COUNT(*) FROM orders"))
    res = assistant.ask("How many orders are there?", store=False)
    assert not res["stored_in_cache"]
    assert len(assistant.cache) == 0
    assistant.store(res["question"], res["sql"])
    assert len(assistant.cache) == 1


def test_assumptions_reported(assistant, monkeypatch):
    monkeypatch.setattr(
        service,
        "answer",
        fake_answer("SELECT COUNT(*) FROM orders WHERE order_status = 'delivered'"),
    )
    res = assistant.ask("How many orders are there?")
    assert res["assumptions"]


def test_hit_serves_stored_result_without_running_sql(assistant, monkeypatch):
    monkeypatch.setattr(service, "answer", fake_answer("SELECT COUNT(*) FROM orders"))
    assistant.ask("How many orders are there?")

    def fail(*a, **k):
        raise AssertionError("sql must not run on a stored-result hit")

    monkeypatch.setattr(service, "run_sql", fail)
    res = assistant.ask("How many orders are there?")
    assert res["source"] == "semantic_cache"
    assert res["result_cached"]
    assert res["rows"] == [(4,)]


def test_seeded_entry_fills_result_on_first_hit(assistant):
    assistant.seed([("How many orders are there?", "SELECT COUNT(*) FROM orders")])
    first = assistant.ask("How many orders are there?")
    assert not first["result_cached"]
    second = assistant.ask("How many orders are there?")
    assert second["result_cached"]
    assert second["rows"] == [(4,)]


def test_large_results_are_not_stored(assistant, monkeypatch):
    monkeypatch.setattr(service, "MAX_STORED_ROWS", 2)
    monkeypatch.setattr(service, "answer", fake_answer("SELECT order_id FROM orders"))
    assistant.ask("List all order ids")
    assert assistant.cache.results == [None]
    again = assistant.ask("List all order ids")
    assert again["source"] == "semantic_cache"
    assert not again["result_cached"]
    assert len(again["rows"]) == 4


def test_stored_results_survive_restart(db, embedder, monkeypatch, tmp_path):
    monkeypatch.setattr(service, "build_schema", lambda *a, **k: "schema")
    path = str(tmp_path / "sem")
    first = Assistant(db, embedder, path)
    monkeypatch.setattr(service, "answer", fake_answer("SELECT COUNT(*) FROM orders"))
    first.ask("How many orders are there?")
    second = Assistant(db, embedder, path)
    res = second.ask("How many orders are there?")
    assert res["result_cached"]
    assert res["rows"] == [(4,)]


def test_changed_database_invalidates_cache(db, embedder, monkeypatch, tmp_path):
    import sqlite3

    monkeypatch.setattr(service, "build_schema", lambda *a, **k: "schema")
    path = str(tmp_path / "sem")
    first = Assistant(db, embedder, path)
    monkeypatch.setattr(service, "answer", fake_answer("SELECT COUNT(*) FROM orders"))
    first.ask("How many orders are there?")
    con = sqlite3.connect(db)
    con.execute("INSERT INTO orders VALUES ('e', 'delivered', '2017-04-01')")
    con.commit()
    con.close()
    import os

    stat = os.stat(db)
    os.utime(db, (stat.st_atime, stat.st_mtime + 10))
    second = Assistant(db, embedder, path)
    assert len(second.cache) == 0
