from assumptions import flag_assumptions

Q = "What is the average number of days between a customer's first and second order?"


def test_flags_unrequested_status_filter():
    sql = "SELECT 1 FROM orders WHERE order_status NOT IN ('canceled', 'unavailable')"
    flags = flag_assumptions(Q, sql)
    assert len(flags) == 1
    assert "canceled" in flags[0] and "unavailable" in flags[0]


def test_flags_equals_filter():
    assert flag_assumptions(Q, "SELECT 1 FROM orders WHERE order_status = 'delivered'")


def test_status_filter_allowed_when_question_mentions_it():
    sql = "SELECT 1 FROM orders WHERE order_status = 'delivered'"
    assert flag_assumptions("Revenue from delivered orders", sql) == []


def test_no_filter_no_flag():
    assert flag_assumptions(Q, "SELECT AVG(x) FROM orders") == []


def test_flags_hardcoded_year():
    flags = flag_assumptions("How many orders per month?", "SELECT 1 WHERE strftime('%Y', ts) = '2017'")
    assert flags and "2017" in flags[0]


def test_year_in_question_is_fine():
    assert flag_assumptions("Orders per month in 2017?", "SELECT 1 WHERE strftime('%Y', ts) = '2017'") == []


def test_empty_sql():
    assert flag_assumptions(Q, None) == []