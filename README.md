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

## Web UI + API server

The React dashboard (`src/App.js`) now talks to a FastAPI backend that wraps the calendar agent, the scraper, and the Gmail ingest loop.

1. **Start the API**
   ```bash
   # Same virtualenv as above
   export GOOGLE_CALENDAR_ID="primary"          # or a specific calendar id
   export CALENDAR_AGENT_DRY_RUN=false          # set true to preview without writing
   export ENABLE_GMAIL_POLLING=true             # polls Gmail every 5 minutes
   export LAVAPAY_FORWARD_TOKEN='{"token":"..."}'
   uvicorn backend.server:app --reload
   ```
   Useful overrides:
   - `SCHEDULE_CSV` – path to the working CSV (defaults to `schedule.csv`).
   - `CALENDAR_TIMEZONE`, `DEFAULT_START_TIME`, `DEFAULT_DURATION_MINUTES`, `SCHEDULE_FALLBACK_YEAR` – parser hints.
   - `FRONTEND_ORIGINS` – comma-separated list of allowed browser origins (default `http://localhost:3000`).
   - `COURSE_IMPORT_REPLACE=true` – delete+recreate matching events during course imports.
   - `GMAIL_*` variables mirror the CLI flags (`GMAIL_QUERY`, `GMAIL_TOKEN`, etc.).

2. **Run the frontend**
   ```bash
   npm install
   npm start
   ```
   Set `REACT_APP_API_BASE_URL` if the backend is not on `http://localhost:8000`.

### Using the chat box

- Natural-language requests (“add a study session tomorrow at 6pm”, “move HW4 to Friday”) are forwarded to the LLM agent that previously lived in the CLI.
- Pasting a course/syllabus URL and mentioning “course”, “syllabus”, etc. triggers the scraper, rewrites `schedule.csv`, reloads the events, and (when not in dry-run mode) pushes them to Google Calendar.
- Gmail ingestion runs automatically every five minutes whenever `ENABLE_GMAIL_POLLING=true` **and** `CALENDAR_AGENT_DRY_RUN=false`; the background task feeds new email-derived events into the same agent memory so you can reference them in chat.

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
