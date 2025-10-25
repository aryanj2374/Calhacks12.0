# Course Calendar Agent

This project hosts a Python agent that ingests a CSV of course events (lectures, sections, exams, homework) and syncs them to Google Calendar using the official API. The CSV typically comes from a teammate’s scraper or manual export, allowing this service to focus on robust normalization plus Calendar delivery.

## Quickstart

1. **Install dependencies**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Create OAuth credentials**
   - In the [Google Cloud Console](https://console.cloud.google.com/), create OAuth client credentials for a desktop app.
   - Save the JSON as `credentials.json` in the project root.
3. **Run the agent**
   ```bash
   python -m agentic_calendar.cli --calendar CalendarID
   ```
   - By default the CLI ingests `schedule.csv` in the repo root; override with `--csv /path/to/file.csv` if needed.
   - If the file omits times, pass `--default-start HH:MM --duration minutes` to match your lecture/discussion slots (per-category defaults exist for lectures, sections, labs, exams, assignments, etc.).
   - Add `--console-oauth` if you prefer to copy/paste the auth URL manually (no auto browser launch).
   - Use `--replace-events` to delete previously synced events (matched by title/time window) before inserting fresh ones.

The first run opens a browser window (or prints the authorization URL when `--console-oauth` is set) to authorize access. After authentication, the token is cached in `token.json`.

## Development status

- CSV ingestion with fuzzy date parsing and fallback durations
- Category hints tuned for lectures, discussions, labs, exams, quizzes, homework, and projects
- Google Calendar service wrapper with batching
- CLI that supports dry-run mode for testing without touching real calendars

Planned enhancements include integrating LLM-based extraction when course pages are inconsistent, and adding CSV/ICS export.

## CSV format

Two layouts are supported:

1. **Normalized headers** – `title,date,start_time,end_time,category,location,description`. These fields map directly to event attributes and override defaults.
2. **Scraper feed (default)** – `Date,Type,Description` as produced by `schedule.csv`. The loader infers titles from the description, maps `Type` to categories (Lecture, Discussion, Assignment, Exam, etc.), and hunts for time ranges like `7pm-9pm` inside the description. Missing times fall back to category-specific defaults (e.g., lectures at 11:00, assignments due 23:59) unless overridden via `--default-start/--duration`.

In both cases `date`/`Date` is required. Optional columns include `duration` (minutes), `location`, and explicit titles. Unrecognized categories fall back to keyword-based guessing (entries containing “midterm” become exams automatically).
