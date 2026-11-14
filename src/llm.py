import os
import re
import sys
import time

from dotenv import load_dotenv
from google import genai

import cache
from executor import UnsafeSQL, run_sql
from schema import build_schema

load_dotenv()

MODEL = "gemini-3.5-flash-lite"
STRONG_MODEL = "gemini-3.6-flash"

PROMPT = """You are an expert SQLite analyst. Given the schema below, write ONE SQLite query that answers the question.
Return only the SQL, with no explanation and no markdown.

{schema}

Question: {question}
SQL:"""


def clean_sql(text):
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S | re.I)
    return (m.group(1) if m else text).strip()


def build_prompt(question, schema):
    return PROMPT.format(schema=schema, question=question)


def generate(question, schema, model=MODEL, run=0):
    prompt = build_prompt(question, schema)
    key = cache.make_key(model, prompt, run)
    hit = cache.load(key)
    if hit is not None:
        return {**hit, "cached": True}
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    start = time.time()
    resp = client.models.generate_content(model=model, contents=prompt)
    latency = time.time() - start
    usage = getattr(resp, "usage_metadata", None)
    entry = {
        "sql": clean_sql(resp.text),
        "input_tokens": getattr(usage, "prompt_token_count", None),
        "output_tokens": getattr(usage, "candidates_token_count", None),
        "latency_s": round(latency, 3),
    }
    cache.save(key, entry)
    return {**entry, "cached": False}


def generate_sql(question, schema, model=MODEL, run=0):
    return generate(question, schema, model, run)["sql"]


if __name__ == "__main__":
    db = sys.argv[1] if len(sys.argv) > 1 else "olist.db"
    schema = build_schema(db)
    for q in [
        "How many orders are there in total?",
        "What are the top 5 product categories by revenue? Use English names.",
        "What is the average review score for delivered orders?",
    ]:
        sql = generate_sql(q, schema)
        print(f"\nQ: {q}\nSQL: {sql}")
        try:
            cols, rows = run_sql(db, sql)
            print(cols, rows[:5])
        except (UnsafeSQL, Exception) as e:
            print("ERROR:", e)
