"""Build the schema section of the prompt from a SQLite db."""

import sqlite3

EXCLUDE = {"geolocation"}

JOIN_HINTS = """\
Join keys:
- orders.customer_id = customers.customer_id
- order_items.order_id = orders.order_id
- order_items.product_id = products.product_id
- order_items.seller_id = sellers.seller_id
- order_payments.order_id = orders.order_id
- order_reviews.order_id = orders.order_id
- products.product_category_name = product_category_name_translation.product_category_name

Notes:
- Dates are TEXT in 'YYYY-MM-DD HH:MM:SS' format (use strftime / julianday).
- order_status: delivered, invoiced, shipped, processing, unavailable, canceled, created, approved.
- order_items has one row per item; order_item_id is the item's sequence number in the order, NOT a quantity.
- Revenue is SUM(order_items.price). An order can have several items and several payment rows.
- customer_id is per order; customer_unique_id identifies a real person.
- Category names are Portuguese; use product_category_name_translation for English names.
"""


RULES = """\
Query-writing rules:
- order_items has one row per item and order_payments has one row per payment. When a question is about orders, count COUNT(DISTINCT order_id); do not count joined rows.
- An order can have several items, payments and reviews. Joining order_items to order_reviews or order_payments repeats rows. Reduce to distinct orders first (a subquery) or aggregate each table separately before combining.
- A payment method is order_payments.payment_type, not payment_sequential.
- When ranking or comparing groups, return both the group and the value used for the ranking.
"""

MAX_TEXT_VALUES = 30
MAX_INT_VALUES = 10


def column_values(con, table, column, col_type):
    limit = (
        MAX_TEXT_VALUES
        if col_type == "TEXT"
        else MAX_INT_VALUES if col_type == "INTEGER" else 0
    )
    if not limit:
        return None
    rows = con.execute(
        f"SELECT DISTINCT {column} FROM {table} WHERE {column} IS NOT NULL ORDER BY 1 LIMIT {limit + 1}"
    ).fetchall()
    if len(rows) > limit:
        return None
    return [r[0] for r in rows]


def format_values(values):
    return ", ".join(f"'{v}'" if isinstance(v, str) else str(v) for v in values)


def build_schema(
    db_path, include_hints=True, include_samples=False, include_rules=False
):
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    parts = []
    for (name,) in rows:
        if name in EXCLUDE:
            continue
        cols = con.execute(f"PRAGMA table_info({name})").fetchall()
        lines = []
        for i, c in enumerate(cols):
            line = f"  {c[1]} {c[2]}" + ("," if i < len(cols) - 1 else "")
            if include_samples:
                values = column_values(con, name, c[1], c[2])
                if values:
                    line += f" -- values: {format_values(values)}"
            lines.append(line)
        parts.append(f"CREATE TABLE {name} (\n" + "\n".join(lines) + "\n);")
    con.close()
    out = "\n\n".join(parts)
    if include_hints:
        out += "\n\n" + JOIN_HINTS
    if include_rules:
        out += "\n" + RULES
    return out


if __name__ == "__main__":
    import sys

    print(build_schema(sys.argv[1]))
