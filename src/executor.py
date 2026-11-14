import re
import sqlite3
import time

MAX_ROWS = 10_000
TIMEOUT_S = 10
FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|vacuum|reindex)\b",
    re.I,
)


class UnsafeSQL(ValueError):
    pass


def _strip_comments(sql: str) -> str:
    sql = re.sub(r"--[^\n]*", " ", sql)
    return re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)


def check_safe(sql: str) -> str:
    s = _strip_comments(sql).strip().rstrip(";").strip()
    if not s:
        raise UnsafeSQL("empty query")
    if ";" in s:
        raise UnsafeSQL("multiple statements")
    if not re.match(r"(select|with)\b", s, re.I):
        raise UnsafeSQL("only SELECT/WITH allowed")
    if FORBIDDEN.search(s):
        raise UnsafeSQL("forbidden keyword")
    return s


def run_sql(db_path: str, sql: str):
    s = check_safe(sql)
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    deadline = time.time() + TIMEOUT_S
    con.set_progress_handler(lambda: 1 if time.time() > deadline else 0, 10_000)
    try:
        cur = con.execute(s)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchmany(MAX_ROWS)
        return cols, rows
    finally:
        con.close()