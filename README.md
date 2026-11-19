# Text-to-SQL Assistant

Ask a question about the Olist e-commerce data in plain English. The app writes the SQL, runs it read-only on SQLite, shows the table and a chart, and tells you whether the answer came from the semantic cache or the model.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create `.env` in the project root:

```
GEMINI_API_KEY=your_key_here
```

Build the database once (needs the Olist CSVs in `raw_data/`):

```powershell
python load_data.py
```

## Run the app

```powershell
streamlit run app.py
```

Sidebar:

- Cache similarity threshold (default 0.88) and the slot guard toggle.
- Error retries (default 2).
- Escalate to the strong model on error (off by default, uses the free-tier-limited model).
- Seed cache with the 30 gold questions, so the demo shows hits straight away.
- Auto-save answers to cache (off by default): answers are saved only when you click "Looks correct, save to cache".
- Clear cache. Cached entries keep up to 500 result rows, so a hit skips both the model and the SQL re-run; rebuilding olist.db invalidates the cache.

Right panel per answer: cache hit or miss, similarity and matched question, model used, attempts, tokens, cost, LLM latency, cache lookup time.

## Run the tests

```powershell
pytest
```

No API key or internet needed. Tests stub the model and use a tiny temporary database and a hash-based embedder, so they check the plumbing (safety checks, retries, quota handling, cache logic, the app), not model quality.

## Quota errors

Gemini free tier has per-model daily limits. When the daily limit is hit the app shows a plain message instead of retrying for minutes. The eval harness stops, saves nothing for that run, and resumes from the disk cache when you rerun. Daily limits reset at midnight Pacific.

## What the SQL guard allows

Only one `SELECT` or `WITH` statement. Rejected before execution: `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `CREATE`, `ATTACH`, `DETACH`, `PRAGMA`, `VACUUM`, `REINDEX`, chained statements (`;`), anything else such as `EXPLAIN`. Comments are stripped first so `/* hidden */ DELETE` is caught. The database is also opened with `mode=ro`, results are capped at 10,000 rows and queries are stopped after 10 seconds.

## Layout

```
app.py                  Streamlit UI
src/service.py          cache lookup -> model -> execute -> store
src/semantic_cache.py   embeddings, cosine search, slot guard
src/pipeline.py         generate, execute, retry with error feedback, quota handling
src/router.py           cheap-then-strong cascade
src/executor.py         read-only SQL execution
src/charts.py           chart type chooser
src/assumptions.py      flags filters the question never asked for
src/pricing.py          token cost
src/run_eval.py         evaluation harness
tests/                  pytest suite
```