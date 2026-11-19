import pytest

import pipeline
from pipeline import QuotaExhausted, answer, generate_with_retry, is_quota_error, suggested_delay, wait_before_retry


class Boom(Exception):
    pass


DAILY = Boom("429 RESOURCE_EXHAUSTED. Quota exceeded for metric GenerateRequestsPerDayPerProjectPerModel-FreeTier")
MINUTE = Boom("429 RESOURCE_EXHAUSTED. 'retryDelay': '12s'")


def test_quota_detection():
    assert is_quota_error(DAILY)
    assert not is_quota_error(Boom("500 internal"))


def test_delay_parsing():
    assert suggested_delay(MINUTE) == 12
    assert suggested_delay(Boom("nothing")) is None


def test_daily_quota_fails_fast():
    with pytest.raises(QuotaExhausted):
        wait_before_retry(DAILY, "m", 0)


def test_per_minute_limit_waits_suggested_delay():
    assert wait_before_retry(MINUTE, "m", 0) == 13


def test_non_quota_error_backs_off():
    assert wait_before_retry(Boom("500"), "m", 1) == 10


def test_generate_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def fake(question, schema, model, run, feedback):
        calls["n"] += 1
        if calls["n"] < 3:
            raise Boom("503 unavailable")
        return {"sql": "SELECT 1", "cached": False}

    monkeypatch.setattr(pipeline, "generate", fake)
    monkeypatch.setattr(pipeline.time, "sleep", lambda s: None)
    assert generate_with_retry("q", "s", "m", 0, None)["sql"] == "SELECT 1"
    assert calls["n"] == 3


def test_generate_stops_on_daily_quota(monkeypatch):
    calls = {"n": 0}

    def fake(*args):
        calls["n"] += 1
        raise DAILY

    monkeypatch.setattr(pipeline, "generate", fake)
    monkeypatch.setattr(pipeline.time, "sleep", lambda s: None)
    with pytest.raises(QuotaExhausted):
        generate_with_retry("q", "s", "m", 0, None)
    assert calls["n"] == 1


def gen_result(sql):
    return {"sql": sql, "cached": False, "input_tokens": 10, "output_tokens": 5, "latency_s": 0.5}


def test_answer_retries_after_sql_error(monkeypatch, db):
    sqls = iter(["SELECT nope FROM orders", "SELECT COUNT(*) FROM orders"])
    monkeypatch.setattr(pipeline, "generate", lambda *a: gen_result(next(sqls)))
    monkeypatch.setattr(pipeline.time, "sleep", lambda s: None)
    res = answer("how many orders", "schema", db, "m", 0, max_retries=2)
    assert res["error"] is None
    assert res["rows"] == [(4,)]
    assert len(res["attempts"]) == 2
    assert res["input_tokens"] == 20


def test_answer_reports_error_when_retries_run_out(monkeypatch, db):
    monkeypatch.setattr(pipeline, "generate", lambda *a: gen_result("SELECT nope FROM orders"))
    monkeypatch.setattr(pipeline.time, "sleep", lambda s: None)
    res = answer("q", "schema", db, "m", 0, max_retries=1)
    assert res["error"]
    assert len(res["attempts"]) == 2


def test_answer_blocks_unsafe_sql(monkeypatch, db):
    monkeypatch.setattr(pipeline, "generate", lambda *a: gen_result("DELETE FROM orders"))
    monkeypatch.setattr(pipeline.time, "sleep", lambda s: None)
    res = answer("q", "schema", db, "m", 0, max_retries=0)
    assert "UnsafeSQL" in res["error"]
