import sqlite3

import pytest

from executor import UnsafeSQL, check_safe, run_sql


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM orders",
        "DROP TABLE orders",
        "UPDATE orders SET order_status = 'x'",
        "SELECT 1; DELETE FROM orders",
        "PRAGMA table_info(orders)",
        "ATTACH DATABASE 'x.db' AS x",
        "/* hidden */ DELETE FROM orders",
        "",
        "   ;  ",
        "EXPLAIN SELECT 1",
    ],
)
def test_rejects_unsafe(sql):
    with pytest.raises(UnsafeSQL):
        check_safe(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1",
        "select * from orders;",
        "WITH t AS (SELECT 1 AS x) SELECT x FROM t",
        "-- note\nSELECT 1",
    ],
)
def test_accepts_read_queries(sql):
    assert check_safe(sql)


def test_run_sql_returns_columns_and_rows(db):
    cols, rows = run_sql(db, "SELECT order_status, COUNT(*) AS n FROM orders GROUP BY 1 ORDER BY 1")
    assert cols == ["order_status", "n"]
    assert rows == [("canceled", 1), ("delivered", 3)]


def test_run_sql_bad_column_raises(db):
    with pytest.raises(sqlite3.Error):
        run_sql(db, "SELECT nope FROM orders")


def test_database_stays_unchanged(db):
    with pytest.raises(UnsafeSQL):
        run_sql(db, "DELETE FROM orders")
    assert run_sql(db, "SELECT COUNT(*) FROM orders")[1] == [(4,)]
