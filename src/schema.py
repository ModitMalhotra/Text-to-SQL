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


def build_schema(db_path: str, include_hints: bool = True) -> str:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    parts = []
    for (name,) in rows:
        if name in EXCLUDE:
            continue
        cols = con.execute(f"PRAGMA table_info({name})").fetchall()
        body = ",\n".join(f"  {c[1]} {c[2]}" for c in cols)
        parts.append(f"CREATE TABLE {name} (\n{body}\n);")
    con.close()
    out = "\n\n".join(parts)
    return out + ("\n\n" + JOIN_HINTS if include_hints else "")


if __name__ == "__main__":
    import sys
    print(build_schema(sys.argv[1]))