EXAMPLES = [
    (
        "What percentage of products have appeared in more than one order?",
        "SELECT 100.0 * SUM(CASE WHEN n > 1 THEN 1 ELSE 0 END) / COUNT(*) FROM (SELECT product_id, COUNT(DISTINCT order_id) AS n FROM order_items GROUP BY product_id)",
        "Count distinct orders per product first, then take the ratio over products. Never count joined rows when the question is about orders or customers.",
    ),
    (
        "What is the average review score of orders that include at least one item priced above 500?",
        "SELECT AVG(review_score) FROM order_reviews WHERE order_id IN (SELECT order_id FROM order_items WHERE price > 500)",
        "An order has many items but its review is one row. Joining order_items to order_reviews directly would repeat each review once per item, so filter the orders in a subquery instead.",
    ),
    (
        "How many sellers have fulfilled more than 20 distinct orders?",
        "SELECT COUNT(*) FROM (SELECT seller_id FROM order_items GROUP BY seller_id HAVING COUNT(DISTINCT order_id) > 20)",
        "order_items has one row per item, so use COUNT(DISTINCT order_id) to count orders.",
    ),
]


def fewshot_block():
    parts = ["Worked examples on this same database (different questions):"]
    for question, sql, note in EXAMPLES:
        parts.append(f"Question: {question}\nSQL: {sql}\nNote: {note}")
    return "\n\n".join(parts)
