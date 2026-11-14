def classify_error(message):
    m = message.lower()
    if "unsafesql" in m:
        return "unsafe_sql"
    if "interrupted" in m:
        return "timeout"
    if "no such column" in m:
        return "hallucinated_column"
    if "no such table" in m:
        return "hallucinated_table"
    if "ambiguous column" in m:
        return "ambiguous_column"
    if "syntax error" in m:
        return "syntax_error"
    return "other_error"


def classify_wrong(gold_cols, gold_rows, pred_cols, pred_rows):
    if len(gold_cols) != len(pred_cols):
        return "wrong_columns"
    if len(gold_rows) != len(pred_rows):
        return "wrong_row_count"
    return "wrong_values"
