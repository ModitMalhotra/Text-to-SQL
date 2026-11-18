PRICE_SETS = {
    "intro": {
        "gemini-3.5-flash-lite": (0.30, 2.50),
        "gemini-3.6-flash": (0.75, 3.75),
    },
    "list_2027": {
        "gemini-3.5-flash-lite": (0.30, 2.50),
        "gemini-3.6-flash": (1.50, 7.50),
    },
}


def cost_usd(usage, price_set):
    prices = PRICE_SETS[price_set]
    total = 0.0
    for u in usage:
        if u["model"] not in prices:
            return None
        rate_in, rate_out = prices[u["model"]]
        total += (u["input_tokens"] or 0) * rate_in / 1_000_000
        total += (u["output_tokens"] or 0) * rate_out / 1_000_000
    return total


def costs(usage):
    out = {name: cost_usd(usage, name) for name in PRICE_SETS}
    if any(v is None for v in out.values()):
        return None
    return out
