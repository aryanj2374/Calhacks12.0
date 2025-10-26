import requests
from bs4 import BeautifulSoup
import re
import csv
import dateparser
from datetime import datetime

def clean_text(text):
    return " ".join(text.split())

def looks_like_date(text):
    text = text.strip().lower()
    if re.fullmatch(r'\d+', text):
        return False
    return bool(re.search(r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|\d{1,2}[/-]\d{1,2})', text))

def parse_date(date_str):
    cleaned = clean_text(date_str)
    if not looks_like_date(cleaned):
        return None
    parsed = dateparser.parse(cleaned)
    if parsed:
        return parsed.strftime("%b %d")
    return cleaned

def extract_short_description(text, event_type):
    text = clean_text(text)
    if event_type == "Assignment":
        match = re.search(r'(homework\s*\d+|hw\s*\d+|lab\s*\d+|project\s*\d+|project\s+[a-zA-Z]+)', text, re.IGNORECASE)
        if match:
            return match.group(0).strip()
        return " ".join(text.split()[:4])
    if event_type == "Discussion":
        match = re.search(r'(discussion\s*\d*|disc\s*\d*)', text, re.IGNORECASE)
        if match:
            return match.group(0).strip()
        return text
    if event_type == "Exam":
        match = re.search(r'(midterm\s*\d*|final\s*exam|final)', text, re.IGNORECASE)
        if match:
            return match.group(0).strip().title()
        return text
    return text

def extract_time_from_text(text):
    text = re.sub(r'[()]', '', text)
    match = re.search(r'(\d{1,2}(:\d{2})?\s*(am|pm)?\s*-\s*\d{1,2}(:\d{2})?\s*(am|pm)?)', text, re.IGNORECASE)
    if match:
        return match.group(0).strip()
    return ""

def get_weekday_to_lecture_time(soup):
    """
    Extract lecture times from page text.
    Returns a dict mapping weekday name (Mon, Tue, etc.) -> time string.
    """
    page_text = soup.get_text(separator=" ")
    # Match patterns like "Mon, Wed, Fri 1-2 p.m."
    pattern = r'((?:Mon|Tue|Wed|Thu|Fri)(?:,\s*(?:Mon|Tue|Wed|Thu|Fri))*)\s+(\d{1,2}(:\d{2})?-(\d{1,2}(:\d{2})?)\s*(am|pm)?)'
    matches = re.findall(pattern, page_text, re.IGNORECASE)
    weekday_time = {}
    for match in matches:
        days_str = match[0]
        time_str = match[1]
        days = [d.strip() for d in days_str.split(',')]
        for d in days:
            weekday_time[d[:3]] = time_str  # Use 3-letter abbreviation
    return weekday_time

def scrape_course_schedule(url):
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    tables = soup.find_all("table")
    if not tables:
        raise ValueError("No tables found on the page.")

    weekday_time_map = get_weekday_to_lecture_time(soup)
    events = []

    for table in tables:
        rows = table.find_all("tr")
        if len(rows) < 2:
            continue

        for row in rows[1:]:
            cols = row.find_all(["td", "th"])
            cols_text = [clean_text(c.get_text()) for c in cols if c.get_text(strip=True)]
            if len(cols_text) < 2:
                continue

            # Determine correct date column
            if len(cols_text) > 1 and looks_like_date(cols_text[1]) and not looks_like_date(cols_text[0]):
                week_or_date = parse_date(cols_text[1])
                description = " ".join(cols_text[2:]).strip()
            else:
                week_or_date = parse_date(cols_text[0])
                description = " ".join(cols_text[1:]).strip()

            if not week_or_date:
                continue

            # Determine day of week from date
            try:
                dt = datetime.strptime(week_or_date + f" {datetime.now().year}", "%b %d %Y")
                weekday_abbr = dt.strftime("%a")
            except:
                weekday_abbr = ""

            desc_lower = description.lower()
            row_events = []

            # Lecture
            if "no lecture" not in desc_lower:
                lecture_time = weekday_time_map.get(weekday_abbr, "")
                row_events.append({"Date": week_or_date, "Time": lecture_time, "Type": "Lecture", "Description": "Lecture"})

            # Discussion
            if re.search(r'\b(disc|discussion)\b', desc_lower):
                time_str = extract_time_from_text(description)
                short_desc = extract_short_description(description, "Discussion")
                row_events.append({"Date": week_or_date, "Time": time_str, "Type": "Discussion", "Description": short_desc})

            # Exam
            if re.search(r'\b(exam|midterm|final)\b', desc_lower):
                time_str = extract_time_from_text(description)
                short_desc = extract_short_description(description, "Exam")
                row_events.append({"Date": week_or_date, "Time": time_str, "Type": "Exam", "Description": short_desc})

            # Assignment — no time
            if re.search(r'\b(homework|hw|lab|project)\b', desc_lower):
                assign_date = week_or_date
                if len(cols_text) > 2 and looks_like_date(cols_text[1]):
                    assign_date = parse_date(cols_text[1])
                short_desc = extract_short_description(description, "Assignment")
                row_events.append({"Date": assign_date, "Time": "", "Type": "Assignment", "Description": short_desc})

            events.extend(row_events)

    # Deduplicate exams only: keep one per Description, prefer one with time
    unique_events = []
    seen_exams = {}

    for e in events:
        if e["Type"] == "Exam":
            key = e["Description"]
            if key not in seen_exams:
                seen_exams[key] = e
            else:
                if seen_exams[key]["Time"] == "" and e["Time"] != "":
                    seen_exams[key] = e
        else:
            unique_events.append(e)

    # Add deduplicated exams
    unique_events.extend(seen_exams.values())

    return unique_events

def save_to_csv(events, filename="schedule.csv"):
    if not events:
        print("No events found to save.")
        return
    # Sort by date for convenience
    events_sorted = sorted(events, key=lambda x: datetime.strptime(x["Date"] + f" {datetime.now().year}", "%b %d %Y"))
    with open(filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Date", "Time", "Type", "Description"])
        writer.writeheader()
        for e in events_sorted:
            writer.writerow(e)
    print(f"✅ Schedule saved to {filename} successfully!")

if __name__ == "__main__":
    url = input("Enter course schedule URL: ").strip()
    try:
        events = scrape_course_schedule(url)
        save_to_csv(events)
    except Exception as e:
        print(f"Error: {e}")
