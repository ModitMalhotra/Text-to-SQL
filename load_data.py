import re
import sqlite3
import sys
import zipfile
from pathlib import Path

import pandas as pd

INDEXES = [
    ("orders", "order_id"), ("orders", "customer_id"),
    ("order_items", "order_id"), ("order_items", "product_id"), ("order_items", "seller_id"),
    ("order_payments", "order_id"), ("order_reviews", "order_id"),
    ("customers", "customer_id"), ("customers", "customer_unique_id"),
    ("products", "product_id"), ("sellers", "seller_id"),
]


def table_name(csv_name: str) -> str:
    stem = csv_name.rsplit("/", 1)[-1].removesuffix(".csv")
    stem = re.sub(r"^olist_", "", stem)
    return re.sub(r"_dataset$", "", stem)


def main(src_path: str, db_path: str) -> None:
    con = sqlite3.connect(db_path)
    src = Path(src_path)
    if src.is_dir():
        files = [(p.name, p) for p in sorted(src.rglob("*.csv"))]
        opener = lambda h: open(h, "rb")
    else:
        z = zipfile.ZipFile(src)
        files = [(n, n) for n in sorted(z.namelist()) if n.endswith(".csv")]
        opener = z.open
    for name, handle in files:
        with opener(handle) as f:
            df = pd.read_csv(f)
        t = table_name(name)
        df.to_sql(t, con, if_exists="replace", index=False)
        print(f"{t:40s} {len(df):>9,} rows")
    have = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for t, c in INDEXES:
        if t not in have:
            continue
        con.execute(f"CREATE INDEX IF NOT EXISTS idx_{t}_{c} ON {t}({c})")
    con.commit()
    print("\norder_status values:",
          [r[0] for r in con.execute("SELECT DISTINCT order_status FROM orders")])
    con.close()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])