import re

STATUS_WORDS = re.compile(
    r"\b(deliver\w*|cancel\w*|ship\w*|status|unavailable|approved|invoiced|processing|created)\b",
    re.I,
)
STATUS_FILTER = re.compile(
    r"order_status\s*(?:NOT\s+IN|IN|=|!=|<>)\s*(\([^)]*\)|'[^']*')", re.I
)
YEAR = re.compile(r"\b(20\d{2})\b")


def flag_assumptions(question, sql):
    flags = []
    if not sql:
        return flags
    if not STATUS_WORDS.search(question):
        seen = []
        for match in STATUS_FILTER.finditer(sql):
            seen.extend(re.findall(r"'([^']*)'", match.group(1)))
        if seen:
            values = ", ".join(sorted(set(seen)))
            flags.append(
                f"SQL filters on order status ({values}) but the question never mentioned status."
            )
    years = sorted(set(YEAR.findall(sql)) - set(YEAR.findall(question)))
    if years:
        flags.append(
            f"SQL hardcodes year {', '.join(years)} but the question did not name it."
        )
    return flags
