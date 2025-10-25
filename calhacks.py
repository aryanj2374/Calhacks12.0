import requests
from bs4 import BeautifulSoup
import re
import csv
import dateparser

def clean_text(text):
    """Strip extra whitespace and remove newlines"""
    return " ".join(text.split())

def parse_date(date_str):
    """
    Attempt to parse a date string into YYYY-MM-DD.
    If parsing fails, return cleaned original text.
    """
    parsed = dateparser.parse(date_str)
    if parsed:
        return parsed.strftime("%Y-%m-%d")
    return clean_text(date_str)

def scrape_course_schedule(url):
    """
    Scrapes a course website with table-based schedules, extracting multiple events per row:
    - Lecture
    - Discussion
    - Exam
    - Assignment (homework, hw, lab, project)
    """
    response = requests.get(url)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.text, "html.parser")
    tables = soup.find_all("table")
    
    if not tables:
        raise ValueError("No tables found on the page.")
    
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
            
            week_or_date = parse_date(cols_text[0])
            description = " ".join(cols_text[1:]).strip()
            desc_lower = description.lower()
            
            row_events = []
            
            # Lecture (skip if "no lecture")
            if "lecture" in desc_lower or "no lecture" not in desc_lower:
                if "no lecture" not in desc_lower:
                    row_events.append({"Date": week_or_date, "Type": "Lecture", "Description": description})
            
            # Discussion
            if re.search(r'\b(disc|discussion)\b', desc_lower):
                row_events.append({"Date": week_or_date, "Type": "Discussion", "Description": description})
            
            # Exam
            if re.search(r'\b(exam|midterm|final)\b', desc_lower):
                row_events.append({"Date": week_or_date, "Type": "Exam", "Description": description})
            
            # Assignment (homework, hw, lab, project)
            if re.search(r'\b(homework|hw|lab|project)\b', desc_lower):
                assign_date = week_or_date
                if len(cols_text) > 2:
                    # Prefer second column for due date if it exists
                    assign_date = parse_date(cols_text[1])
                row_events.append({"Date": assign_date, "Type": "Assignment", "Description": description})
            
            events.extend(row_events)
    
    return events

def save_to_csv(events, filename="schedule.csv"):
    """Save events to a CSV with clean formatting"""
    if not events:
        print("No events found to save.")
        return
    
    with open(filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Date", "Type", "Description"])
        writer.writeheader()
        for e in events:
            writer.writerow({
                "Date": e["Date"],
                "Type": e["Type"].title(),
                "Description": e["Description"]
            })
    
    print(f"Schedule saved to {filename} successfully!")

# Example usage
if __name__ == "__main__":
    url = input("Enter course schedule URL: ").strip()
    try:
        events = scrape_course_schedule(url)
        save_to_csv(events)
    except Exception as e:
        print(f"Error: {e}")
