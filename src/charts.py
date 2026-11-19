import re

import pandas as pd

MONTHLY = re.compile(r"^\d{4}-\d{2}(-\d{2})?$")
MAX_POINTS = 60


def suggest_chart(cols, rows):
    if len(cols) < 2 or len(rows) < 2 or len(rows) > MAX_POINTS:
        return None
    df = pd.DataFrame(rows, columns=cols)
    label = cols[0]
    values = [c for c in cols[1:] if pd.api.types.is_numeric_dtype(df[c])]
    if not values:
        return None
    value = values[0]
    labels = df[label].astype(str)
    kind = "line" if labels.map(lambda v: bool(MONTHLY.match(v))).all() else "bar"
    data = pd.DataFrame({value: df[value].to_numpy()}, index=labels.to_numpy())
    return {"kind": kind, "data": data, "label": label, "value": value}
