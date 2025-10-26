# Course Calendar Agent

Python agent that ingests course schedules, syncs them to Google Calendar, and now scans Gmail for new events using a LavaPayments-backed LLM.

## Quickstart

1. **Install dependencies**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Create OAuth credentials**
   - In the [Google Cloud Console](https://console.cloud.google.com/), create desktop-app OAuth credentials.
   - Save the JSON as `credentials.json` in the repo root.
3. **Run the CSV agent**
   ```bash
   python -m agentic_calendar.cli --calendar CalendarID
   ```
   - By default the CLI ingests `schedule.csv`; override with `--csv /path/to/file.csv`.
   - If the CSV omits a year (e.g., `Oct 05`), add `--csv-year 2025` so dates are interpreted correctly.
   - Use `--default-start HH:MM` and `--duration minutes` for files without explicit times.
   - Add `--console-oauth` to use the copy/paste OAuth flow.
   - Use `--replace-events` to delete matching events before inserting new ones.

## Gmail ingestion

The `agentic_calendar.gmail_ingest_cli` entry point authorizes Gmail (read-only) and Calendar (write) and sends each unread email to the LLM so commitments (deadlines, hackathon schedules, etc.) are added automatically.

```bash
python -m agentic_calendar.gmail_ingest_cli \
  --calendar primary \
  --gmail-credentials credentials.json \
  --gmail-token gmail_token.json \
  --query "label:unread newer_than:1d" \
  --forward-token "$LAVAPAY_FORWARD_TOKEN" \
  --apply
```

Workflow:

1. Gmail OAuth tokens are stored separately (`gmail_token.json`) so you can grant read-only mail access alongside Calendar writes.
2. The CLI searches Gmail with the provided query (default `label:unread newer_than:1d`), skips messages already logged in `.gmail_processed.json`, and feeds new ones into the extractor.
3. Preview results without `--apply` or add the flag to create the events immediately.

Environment variables:

- `LAVAPAY_FORWARD_TOKEN` / `LAVA_FORWARD_TOKEN` – LavaPayments forward token (or pass via `--forward-token`).
- `LAVAPAY_BASE_URL`, `LAVAPAY_TARGET_URL`, `LAVAPAY_PROVIDER`, `LAVAPAY_MODEL` – optional overrides for the Lava gateway/provider.

## CSV format

Two layouts are supported:

1. **Normalized headers** – `title,date,start_time,end_time,category,location,description`. These map directly to event fields.
2. **Scraper feed (default)** – `Date,Type,Description` as produced by `schedule.csv`. Titles/descriptions are inferred, type maps to categories, and missing times fall back to per-category defaults.

`Date` is required. Optional columns include `duration`, `location`, and explicit titles. When dates omit a year, provide `--csv-year` so parsing succeeds.
