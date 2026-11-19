import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import json
import os

import pandas as pd
import streamlit as st

from charts import suggest_chart
from pipeline import QuotaExhausted
from semantic_cache import Embedder, HashEmbedder
from service import CACHE_PATH, Assistant

DB_PATH = os.environ.get("OLIST_DB", "olist.db")
SEED_PATH = os.path.join("eval", "cache_pairs.json")
EXAMPLES = [
    "Which product category brings in the most money from delivered orders, and how much?",
    "How many orders were placed each month in 2017?",
    "Which 5 states have the highest average review score?",
    "What share of orders were delivered late?",
]

st.set_page_config(page_title="Text-to-SQL Assistant", layout="wide")


@st.cache_resource(show_spinner="Loading embedding model")
def get_assistant():
    embedder = (
        HashEmbedder()
        if os.environ.get("SQL_ASSISTANT_EMBEDDER") == "hash"
        else Embedder()
    )
    return Assistant(DB_PATH, embedder, CACHE_PATH)


def load_seed():
    with open(SEED_PATH, encoding="utf-8") as f:
        return [(x["question"], x["gold_sql"]) for x in json.load(f)["seed"]]


if not os.path.exists(DB_PATH):
    st.error(f"Database not found at {DB_PATH}. Run load_data.py first.")
    st.stop()

assistant = get_assistant()

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Cache similarity threshold", 0.60, 0.99, 0.88, 0.01)
    guard = st.toggle("Slot guard", value=True)
    retries = st.number_input("Error retries", 0, 3, 2)
    escalate = st.toggle("Escalate to strong model on error", value=False)
    auto_save = st.toggle("Auto-save answers to cache", value=False)
    st.divider()
    st.caption(f"Cache entries: {len(assistant.cache)}")
    if st.button("Seed cache with 30 gold questions"):
        added = assistant.seed(load_seed())
        st.success(f"Added {added}")
        st.rerun()
    if st.button("Clear cache"):
        assistant.clear_cache()
        st.rerun()

st.title("Text-to-SQL Assistant")
st.caption("Ask a question about the Olist e-commerce data.")

if "question" not in st.session_state:
    st.session_state["question"] = ""

cols = st.columns(len(EXAMPLES))
for col, example in zip(cols, EXAMPLES):
    if col.button(example, use_container_width=True):
        st.session_state["question"] = example

question = st.text_input("Question", key="question")
go = st.button("Ask", type="primary")

if go and question.strip():
    try:
        with st.spinner("Working"):
            result = assistant.ask(
                question,
                threshold=threshold,
                guard=guard,
                retries=int(retries),
                escalate_on_error=escalate,
                store=auto_save,
            )
        st.session_state["result"] = result
        st.session_state["failure"] = None
    except QuotaExhausted as exc:
        st.session_state["result"] = None
        st.session_state["failure"] = f"Gemini quota is used up for today: {exc}"
    except Exception as exc:
        st.session_state["result"] = None
        st.session_state["failure"] = f"{type(exc).__name__}: {exc}"

failure = st.session_state.get("failure")
result = st.session_state.get("result")

if failure:
    st.error(failure)

if result:
    main, side = st.columns([3, 1])
    with side:
        st.subheader("Under the hood")
        if result["source"] == "semantic_cache":
            st.success("Cache HIT")
            st.write(f"Similarity: {result['cache_score']}")
            st.write(f"Matched: {result['cache_matched']}")
            st.write("Model: none (no LLM call)")
            st.write(
                "Result: "
                + (
                    "served from cache"
                    if result["result_cached"]
                    else "re-executed SQL"
                )
            )
        else:
            st.warning("Cache MISS")
            if result["cache_blocked"]:
                b = result["cache_blocked"]
                st.write(
                    f"Guard blocked a near match ({b['score']}): {b['matched_question']}"
                )
                st.write(f"Differs on: {', '.join(b['differs_on'])}")
            st.write(f"Model: {result['model_used']}")
            if result["escalated"]:
                st.write(f"Escalated: {result['escalation_reason']}")
            st.write(f"Attempts: {result['attempts']}")
            st.write(
                f"Tokens in/out: {result['input_tokens']}/{result['output_tokens']}"
            )
            cost = result["cost_usd"]
            st.write("Cost: " + (f"${cost:.6f}" if cost is not None else "unpriced"))
            st.write(
                f"LLM latency: {result['llm_latency_s']}s"
                + (" (replayed from disk cache)" if result["llm_cached"] else "")
            )
            st.write(
                "Stored in cache"
                if result["stored_in_cache"]
                else "Not stored in cache"
            )
        if result["cache_error"]:
            st.warning(result["cache_error"])
        st.write(f"Cache lookup: {result['cache_lookup_ms']} ms")
        st.write(f"Total: {result['total_s']}s")
    with main:
        st.subheader("SQL")
        st.code(result["sql"] or "", language="sql")
        for note in result["assumptions"]:
            st.warning(note)
        if result["error"]:
            st.error(result["error"])
        elif not result["rows"]:
            st.info("The query ran but returned no rows.")
        else:
            df = pd.DataFrame(result["rows"], columns=result["cols"])
            st.subheader(f"Results ({len(df)} rows)")
            st.dataframe(df, use_container_width=True)
            chart = suggest_chart(result["cols"], result["rows"])
            if result["source"] == "model" and not result["stored_in_cache"]:
                if st.button("Looks correct, save to cache"):
                    assistant.store(
                        result["question"],
                        result["sql"],
                        result["cols"],
                        result["rows"],
                    )
                    result["stored_in_cache"] = True
                    st.success("Saved to cache")
            if chart:
                st.subheader("Chart")
                if chart["kind"] == "line":
                    st.line_chart(chart["data"])
                else:
                    st.bar_chart(chart["data"])
