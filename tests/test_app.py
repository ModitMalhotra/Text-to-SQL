import os

import pytest
from streamlit.testing.v1 import AppTest

import service


@pytest.fixture
def app(db, monkeypatch, tmp_path):
    monkeypatch.setenv("OLIST_DB", db)
    monkeypatch.setenv("SQL_ASSISTANT_EMBEDDER", "hash")
    monkeypatch.setattr(service, "CACHE_PATH", str(tmp_path / "sem"))
    monkeypatch.setattr(service, "build_schema", lambda *a, **k: "schema")
    import streamlit as st

    st.cache_resource.clear()
    return AppTest.from_file(os.path.abspath("app.py"), default_timeout=30)


def ask(app):
    next(b for b in app.button if b.label == "Ask").click()


def fake(sql):
    def run(question, schema, db, model, run_id, retries):
        from executor import run_sql

        cols, rows = run_sql(db, sql)
        return {
            "sql": sql,
            "cols": cols,
            "rows": rows,
            "error": None,
            "attempts": [{}],
            "model": model,
            "usage": [{"model": model, "input_tokens": 50, "output_tokens": 10}],
            "latency_s": 0.2,
            "input_tokens": 50,
            "output_tokens": 10,
            "cached": False,
        }

    return run


def test_ask_then_hit(app, monkeypatch):
    monkeypatch.setattr(
        service,
        "answer",
        fake("SELECT order_status, COUNT(*) AS n FROM orders GROUP BY 1"),
    )
    app.run()
    assert not app.exception
    app.text_input(key="question").set_value("Orders per status?")
    ask(app)
    app.run()
    assert not app.exception
    assert any("Cache MISS" in w.value for w in app.warning)
    assert len(app.dataframe) == 1
    next(b for b in app.button if b.label == "Looks correct, save to cache").click()
    app.run()
    ask(app)
    app.run()
    assert any("Cache HIT" in s.value for s in app.success)


def test_quota_error_shown_not_crashed(app, monkeypatch):
    from pipeline import QuotaExhausted

    def boom(*a, **k):
        raise QuotaExhausted("daily limit")

    monkeypatch.setattr(service, "answer", boom)
    app.run()
    app.text_input(key="question").set_value("Anything")
    ask(app)
    app.run()
    assert not app.exception
    assert any("quota" in e.value.lower() for e in app.error)


def test_missing_database_message(monkeypatch):
    monkeypatch.setenv("OLIST_DB", "does_not_exist.db")
    import streamlit as st

    st.cache_resource.clear()
    at = AppTest.from_file(os.path.abspath("app.py"), default_timeout=30).run()
    assert any("Database not found" in e.value for e in at.error)
