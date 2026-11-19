from charts import MAX_POINTS, suggest_chart


def test_monthly_series_becomes_line():
    chart = suggest_chart(["month", "n"], [("2017-01", 3), ("2017-02", 5), ("2017-03", 4)])
    assert chart["kind"] == "line"
    assert chart["value"] == "n"


def test_categories_become_bar():
    chart = suggest_chart(["state", "score"], [("SP", 4.1), ("RJ", 3.9)])
    assert chart["kind"] == "bar"
    assert list(chart["data"].index) == ["SP", "RJ"]


def test_single_row_has_no_chart():
    assert suggest_chart(["a", "b"], [("x", 1)]) is None


def test_single_column_has_no_chart():
    assert suggest_chart(["a"], [(1,), (2,)]) is None


def test_no_numeric_column_has_no_chart():
    assert suggest_chart(["a", "b"], [("x", "y"), ("z", "w")]) is None


def test_too_many_rows_has_no_chart():
    rows = [(str(i), i) for i in range(MAX_POINTS + 1)]
    assert suggest_chart(["a", "b"], rows) is None


def test_picks_first_numeric_value_column():
    chart = suggest_chart(["name", "tag", "total", "other"], [("a", "x", 1, 9), ("b", "y", 2, 8)])
    assert chart["value"] == "total"
