#!/usr/bin/env python3
"""Scrape UC Berkeley dining hall menus into a structured JSON file."""

from __future__ import annotations

import argparse
import base64
import json
import re
import time as time_module
import xml.etree.ElementTree as ET
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup, NavigableString, Tag
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


MENU_URL = "https://dining.berkeley.edu/menus/"
SEARCH_API_URL = "https://dining.berkeley.edu/wp-json/wp/v2/search"
DEFAULT_OUTPUT = Path("menus.json")
DEFAULT_TIMEZONE = "America/Los_Angeles"


def fetch_html(url: str) -> str:
    """Retrieve the HTML for the menus page."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
        )
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.text


def decode_menu_source(encoded: Optional[str]) -> Optional[str]:
    """Decode the base64-encoded XML source link for a menu item."""
    if not encoded:
        return None

    padded = encoded.strip()
    # The data-location attribute is base64 without padding; add it back if missing.
    padding = len(padded) % 4
    if padding:
        padded += "=" * (4 - padding)

    try:
        return base64.b64decode(padded).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None


def text_or_none(element: Optional[Tag]) -> Optional[str]:
    """Return stripped text for a tag or None when missing."""
    if not element:
        return None
    value = element.get_text(strip=True)
    return value or None


def extract_meal_label(meal_el: Tag) -> Optional[str]:
    """Pull the visible label for a meal period."""
    label_span = meal_el.find("span")
    if not label_span:
        return None

    pieces: List[str] = []
    for child in label_span.contents:
        if isinstance(child, NavigableString):
            text = child.strip()
            if text:
                pieces.append(text)
    return " ".join(pieces) or None


def parse_icon(icon_el: Tag) -> Dict[str, Optional[str]]:
    """Extract metadata for a single informational icon."""
    img = icon_el.find("img")
    tooltip = icon_el.select_one(".allg-tooltip")

    label = text_or_none(tooltip)
    alt = img.get("alt") if img and img.has_attr("alt") else None
    src = img.get("src") if img and img.has_attr("src") else None

    code = None
    if src:
        parsed = urlparse(src)
        code = Path(parsed.path).stem if parsed.path else None

    return {
        "label": label or alt,
        "alt": alt,
        "image": src,
        "code": code,
    }


def parse_icons(wrapper: Optional[Tag]) -> List[Dict[str, Optional[str]]]:
    """Collect all icon descriptors for a menu item."""
    if not wrapper:
        return []
    return [parse_icon(icon_el) for icon_el in wrapper.select(".food-icon")]


def parse_menu_item(item_el: Tag) -> Dict[str, Any]:
    """Convert an individual menu item entry into a dictionary."""
    # Name is the first direct child span without a class attribute.
    name_span = None
    for child in item_el.find_all("span", recursive=False):
        if not child.has_attr("class"):
            name_span = child
            break

    name = text_or_none(name_span) or item_el.get_text(" ", strip=True)

    icons = parse_icons(item_el.select_one(".icons-wrap"))

    classes = [cls for cls in item_el.get("class", []) if cls != "recip"]

    menu_id_raw = item_el.get("data-menuid")
    menu_id = menu_id_raw.strip() if menu_id_raw else None
    if menu_id and menu_id.isdigit():
        menu_id_value: Optional[int | str] = int(menu_id)
    else:
        menu_id_value = menu_id

    return {
        "name": name,
        "id": item_el.get("data-id"),
        "menu_id": menu_id_value,
        "menu_source": decode_menu_source(item_el.get("data-location")),
        "tags": classes,
        "icons": icons,
    }


def parse_category(cat_el: Tag) -> Dict[str, Any]:
    """Extract a category block inside a meal period."""
    category_name = text_or_none(cat_el.find("span"))
    items = [parse_menu_item(item) for item in cat_el.select("ul.recipe-name > li.recip")]
    return {
        "category": category_name,
        "items": items,
    }


def parse_meal_period(meal_el: Tag) -> Dict[str, Any]:
    """Extract the details for a single meal period."""
    label = extract_meal_label(meal_el)
    categories = [
        parse_category(cat)
        for cat in meal_el.select(".recipes-main-wrap > div.cat-name")
    ]
    menu_ids = sorted(
        {
            item["menu_id"]
            for category in categories
            for item in category["items"]
            if item.get("menu_id") is not None
        },
        key=lambda value: (isinstance(value, str), value),
    )
    menu_sources = sorted(
        {
            item["menu_source"]
            for category in categories
            for item in category["items"]
            if item.get("menu_source")
        }
    )

    return {
        "label": label,
        "categories": categories,
        "menu_ids": menu_ids,
        "menu_sources": menu_sources,
    }


def parse_location(location_el: Tag) -> Dict[str, Any]:
    """Convert a dining location node into a dictionary."""
    name = text_or_none(location_el.select_one(".cafe-title"))
    status = text_or_none(location_el.select_one(".status"))
    serve_date_display = text_or_none(location_el.select_one(".serve-date"))
    hours = [
        span.get_text(strip=True)
        for span in location_el.select(".times span")
        if span.get_text(strip=True)
    ]

    classes = [cls for cls in location_el.get("class", []) if cls != "location-name"]
    slug = "-".join(classes) if classes else None

    meals = [
        parse_meal_period(meal_el)
        for meal_el in location_el.select("ul.meal-period > li")
    ]

    return {
        "name": name,
        "status": status,
        "serve_date": serve_date_display,
        "service_date": None,
        "hours": hours,
        "slug": slug,
        "location_page": None,
        "location_details": None,
        "meals": meals,
    }


def parse_document(html: str) -> Dict[str, Any]:
    """Parse the complete menus page into structured data."""
    soup = BeautifulSoup(html, "html.parser")
    locations = [
        parse_location(location_el)
        for location_el in soup.select("ul.cafe-location > li.location-name")
    ]
    return {
        "source_url": MENU_URL,
        "locations": locations,
    }


def parse_nutrient_values(labels: List[str], values: str) -> Dict[str, Optional[float | str]]:
    """Map nutrient labels to their corresponding values."""
    results: Dict[str, Optional[float | str]] = {}
    raw_values = [segment.strip() for segment in values.split("|")]
    for label, raw_value in zip(labels, raw_values):
        if not label:
            continue
        if not raw_value:
            results[label] = None
            continue
        try:
            results[label] = float(raw_value)
        except ValueError:
            results[label] = raw_value
    return results


def parse_served_date(raw: Optional[str]) -> Optional[str]:
    """Convert a served date in YYYYMMDD format to ISO date."""
    if not raw or len(raw) != 8 or not raw.isdigit():
        return None
    return f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]}"


def parse_menu_xml(content: bytes) -> Dict[str, Any]:
    """Parse a Cal Dining menu XML payload."""
    root = ET.fromstring(content)
    menus: Dict[str, Dict[str, Any]] = {}

    for menu_el in root.findall("menu"):
        menu_id = menu_el.attrib.get("id")
        if not menu_id:
            continue

        served_raw = menu_el.attrib.get("servedate")
        nutrient_labels_raw = menu_el.findtext("nutrients") or ""
        nutrient_labels = [
            label.strip()
            for label in nutrient_labels_raw.split("|")
            if label.strip()
        ]

        recipes: Dict[str, Dict[str, Any]] = {}
        recipes_el = menu_el.find("recipes")
        if recipes_el is not None:
            for recipe_el in recipes_el.findall("recipe"):
                recipe_id = recipe_el.attrib.get("id")
                if not recipe_id:
                    continue

                nutrients_map = parse_nutrient_values(
                    nutrient_labels,
                    recipe_el.attrib.get("nutrients", ""),
                )

                allergens = {
                    allergen.attrib.get("id"): (allergen.text or "").strip().lower() == "yes"
                    for allergen in recipe_el.findall("./allergens/allergen")
                    if allergen.attrib.get("id")
                }
                dietary_choices = {
                    dietary.attrib.get("id"): (dietary.text or "").strip().lower() == "yes"
                    for dietary in recipe_el.findall("./dietaryChoices/dietaryChoice")
                    if dietary.attrib.get("id")
                }

                ingredients_text = recipe_el.findtext("ingredients") or ""
                ingredients = ingredients_text.strip() or None

                recipes[recipe_id] = {
                    "id": recipe_id,
                    "category": recipe_el.attrib.get("category"),
                    "description": recipe_el.attrib.get("description"),
                    "short_name": recipe_el.attrib.get("shortName"),
                    "serving": {
                        "size": recipe_el.attrib.get("servingSize"),
                        "unit": recipe_el.attrib.get("servingSizeUnit"),
                        "description": recipe_el.attrib.get("servingDescription"),
                        "per_container": recipe_el.attrib.get("servingsPerContainer"),
                    },
                    "nutrients": nutrients_map,
                    "allergens": allergens or None,
                    "dietary_choices": dietary_choices or None,
                    "ingredients": ingredients,
                    "approved_nutrition": recipe_el.attrib.get("approvedNutrition"),
                    "show_ingredients": recipe_el.attrib.get("showIngredients"),
                    "local_items": recipe_el.attrib.get("localItems"),
                    "notes": recipe_el.attrib.get("itemDailyComment") or None,
                }

        menus[menu_id] = {
            "id": menu_id,
            "name": menu_el.attrib.get("name"),
            "served_date": served_raw,
            "served_date_iso": parse_served_date(served_raw),
            "location": menu_el.attrib.get("location"),
            "meal_period": menu_el.attrib.get("mealperiodname"),
            "start_time": menu_el.attrib.get("mealstarttime"),
            "end_time": menu_el.attrib.get("mealendtime"),
            "bulletin": menu_el.attrib.get("menubulletin") or None,
            "nutrient_labels": nutrient_labels,
            "recipes": recipes,
        }

    return {
        "menus": menus,
    }


def gather_menu_sources(locations: List[Dict[str, Any]]) -> List[str]:
    """Collect the unique menu source URLs referenced across locations."""
    sources = {
        item["menu_source"]
        for location in locations
        for meal in location.get("meals", [])
        for category in meal.get("categories", [])
        for item in category.get("items", [])
        if item.get("menu_source")
    }
    return sorted(sources)


def fetch_menu_sources_data(
    locations: List[Dict[str, Any]]
) -> tuple[Dict[tuple[str, str], Dict[str, Any]], List[Dict[str, Any]]]:
    """Fetch and parse menu XML feeds for all referenced menu sources."""
    menu_lookup: Dict[tuple[str, str], Dict[str, Any]] = {}
    summaries: List[Dict[str, Any]] = []

    for source in gather_menu_sources(locations):
        if not source:
            continue

        try:
            response = requests.get(source, timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:
            summaries.append(
                {
                    "url": source,
                    "error": str(exc),
                }
            )
            continue

        xml_bytes = response.content
        try:
            parsed = parse_menu_xml(xml_bytes)
        except ET.ParseError as exc:
            summaries.append(
                {
                    "url": source,
                    "error": f"XML parse error: {exc}",
                    "xml": xml_bytes.decode("utf-8", errors="replace"),
                }
            )
            continue

        menus = []
        for menu_id, menu_data in parsed["menus"].items():
            menu_lookup[(source, str(menu_id))] = menu_data
            menus.append(
                {
                    "id": int(menu_id) if isinstance(menu_id, str) and menu_id.isdigit() else menu_id,
                    "meal_period": menu_data.get("meal_period"),
                    "served_date": menu_data.get("served_date_iso") or menu_data.get("served_date"),
                    "served_date_raw": menu_data.get("served_date"),
                    "start_time": menu_data.get("start_time"),
                    "end_time": menu_data.get("end_time"),
                    "recipe_count": len(menu_data.get("recipes", {})),
                    "nutrient_labels": menu_data.get("nutrient_labels"),
                }
            )

        summaries.append(
            {
                "url": source,
                "menus": menus,
                "xml": xml_bytes.decode("utf-8", errors="replace"),
            }
        )

    return menu_lookup, summaries


def integrate_menu_data(
    locations: List[Dict[str, Any]],
    menu_lookup: Dict[tuple[str, str], Dict[str, Any]],
) -> None:
    """Enrich locations and menu items with data pulled from the XML feeds."""
    for location in locations:
        service_dates: set[str] = set()
        for meal in location.get("meals", []):
            meal_details: List[Dict[str, Any]] = []
            for menu_source in meal.get("menu_sources", []):
                if not menu_source:
                    continue
                for menu_id in meal.get("menu_ids", []):
                    key = (menu_source, str(menu_id))
                    menu_data = menu_lookup.get(key)
                    if not menu_data:
                        continue
                    detail = {
                        "menu_source": menu_source,
                        "menu_id": menu_id,
                        "meal_period": menu_data.get("meal_period"),
                        "served_date": menu_data.get("served_date_iso") or menu_data.get("served_date"),
                        "served_date_raw": menu_data.get("served_date"),
                        "start_time": menu_data.get("start_time"),
                        "end_time": menu_data.get("end_time"),
                        "bulletin": menu_data.get("bulletin"),
                        "nutrient_labels": menu_data.get("nutrient_labels"),
                    }
                    meal_details.append(detail)
                    if detail["served_date"]:
                        service_dates.add(detail["served_date"])

            if meal_details:
                # Deduplicate menu entries per meal.
                seen_keys: set[tuple[Any, Any]] = set()
                unique_details: List[Dict[str, Any]] = []
                for detail in meal_details:
                    key = (detail["menu_source"], detail["menu_id"])
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    unique_details.append(detail)
                meal["schedule"] = unique_details

            for category in meal.get("categories", []):
                for item in category.get("items", []):
                    menu_source = item.get("menu_source")
                    menu_id = item.get("menu_id")
                    recipe_id = item.get("id")
                    if not (menu_source and menu_id is not None and recipe_id):
                        continue

                    menu_data = menu_lookup.get((menu_source, str(menu_id)))
                    if not menu_data:
                        continue

                    recipe = menu_data.get("recipes", {}).get(str(recipe_id))
                    if not recipe:
                        continue

                    item["recipe_description"] = recipe.get("description")
                    item["short_name"] = recipe.get("short_name") or item.get("name")
                    item["serving"] = recipe.get("serving")
                    item["nutrition"] = recipe.get("nutrients")
                    if recipe.get("allergens") is not None:
                        item["allergens"] = recipe["allergens"]
                    if recipe.get("dietary_choices") is not None:
                        item["dietary_choices"] = recipe["dietary_choices"]
                    if recipe.get("ingredients") is not None:
                        item["ingredients"] = recipe["ingredients"]
                    if recipe.get("notes"):
                        item["notes"] = recipe["notes"]
                    item["menu_reference"] = {
                        "menu_source": menu_source,
                        "menu_id": menu_id,
                        "recipe_id": recipe_id,
                    }

        if service_dates:
            ordered_dates = sorted(service_dates)
            location["service_date"] = ordered_dates[0]
            location["service_dates"] = ordered_dates


def parse_time_range_display(value: str) -> Optional[Dict[str, Any]]:
    """Convert a textual time range into structured 24-hour time."""
    if not value:
        return None

    cleaned = value.replace("–", "-")
    parts = [part.strip() for part in cleaned.split("-", 1)]
    if len(parts) != 2:
        return {"display": value}

    try:
        start = datetime.strptime(parts[0].replace(".", "").upper(), "%I:%M %p").strftime("%H:%M")
        end = datetime.strptime(parts[1].replace(".", "").upper(), "%I:%M %p").strftime("%H:%M")
    except ValueError:
        return {"display": value}

    return {
        "display": value,
        "start_time": start,
        "end_time": end,
    }


def search_location_page(name: Optional[str]) -> Optional[str]:
    """Use the WordPress search API to find a location detail page."""
    if not name:
        return None

    try:
        response = requests.get(
            SEARCH_API_URL,
            params={
                "search": name,
                "per_page": 10,
            },
            timeout=30,
        )
        response.raise_for_status()
    except requests.RequestException:
        return None

    try:
        results = response.json()
    except ValueError:
        return None

    candidates = [
        result.get("url")
        for result in results
        if isinstance(result, dict) and result.get("url")
    ]
    for candidate in candidates:
        if "/locations/" in candidate:
            return candidate
    return candidates[0] if candidates else None


def extract_coordinates(map_url: str) -> Optional[Dict[str, float]]:
    """Parse latitude and longitude from a Google Maps URL."""
    if not map_url:
        return None

    match = re.search(r"@(-?\d+\.\d+),(-?\d+\.\d+)", map_url)
    if match:
        return {
            "latitude": float(match.group(1)),
            "longitude": float(match.group(2)),
        }

    lat_match = re.search(r"!3d(-?\d+\.\d+)", map_url)
    lng_match = re.search(r"!4d(-?\d+\.\d+)", map_url)
    if lat_match and lng_match:
        return {
            "latitude": float(lat_match.group(1)),
            "longitude": float(lng_match.group(1)),
        }

    return None


def fetch_location_details(name: Optional[str]) -> Optional[Dict[str, Any]]:
    """Fetch rich metadata for a dining location from its detail page."""
    page_url = search_location_page(name)
    if not page_url:
        return None

    try:
        response = requests.get(page_url, timeout=30)
        response.raise_for_status()
    except requests.RequestException:
        return {"url": page_url}

    soup = BeautifulSoup(response.text, "html.parser")

    entry = soup.select_one(".entry-content")
    lines: List[str] = []
    if entry:
        content = entry.get_text("\n", strip=True)
        lines = [line.strip() for line in content.splitlines() if line.strip()]

    address = next((line for line in lines if any(char.isdigit() for char in line)), None)
    body_lines = [line for line in lines if line != address]

    summary_meta = soup.find("meta", attrs={"property": "og:description"})
    summary = summary_meta.get("content", "").strip() if summary_meta and summary_meta.get("content") else None

    map_link = soup.select_one('a[href*="google.com/maps"]')
    map_url = map_link.get("href") if map_link else None

    modified_meta = soup.find("meta", attrs={"property": "article:modified_time"})
    published_meta = soup.find("meta", attrs={"property": "article:published_time"})

    return {
        "url": page_url,
        "address": address,
        "summary": summary or " ".join(body_lines) or None,
        "details": body_lines or None,
        "google_maps_url": map_url,
        "coordinates": extract_coordinates(map_url) if map_url else None,
        "published": published_meta.get("content") if published_meta and published_meta.get("content") else None,
        "last_modified": modified_meta.get("content") if modified_meta and modified_meta.get("content") else None,
    }


def enrich_locations_with_details(locations: List[Dict[str, Any]]) -> None:
    """Attach location page metadata to each location entry."""
    cache: Dict[str, Optional[Dict[str, Any]]] = {}
    for location in locations:
        name = location.get("name")
        if name in cache:
            details = cache[name]
        else:
            details = fetch_location_details(name)
            cache[name] = details

        if details:
            location["location_page"] = details.get("url")
            location["location_details"] = details


def build_icon_legend(locations: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Aggregate a legend for all icons encountered in the menus."""
    legend: Dict[str, Dict[str, Any]] = {}
    for location in locations:
        for meal in location.get("meals", []):
            for category in meal.get("categories", []):
                for item in category.get("items", []):
                    for icon in item.get("icons", []):
                        key = icon.get("code") or icon.get("image") or icon.get("label")
                        if not key:
                            continue
                        entry = legend.setdefault(
                            key,
                            {
                                "code": icon.get("code"),
                                "label": icon.get("label"),
                                "alt": icon.get("alt"),
                                "image": icon.get("image"),
                                "count": 0,
                            },
                        )
                        entry["count"] += 1
    return legend


def normalize_location_hours(locations: List[Dict[str, Any]]) -> None:
    """Populate structured hour ranges for each location."""
    for location in locations:
        structured = []
        for item in location.get("hours", []):
            parsed = parse_time_range_display(item)
            if parsed:
                structured.append(parsed)
        if structured:
            location["hours_structured"] = structured


def enrich_data(payload: Dict[str, Any]) -> None:
    """Augment the base scrape with additional metadata for downstream use."""
    locations = payload.get("locations", [])

    menu_lookup, menu_sources = fetch_menu_sources_data(locations)
    integrate_menu_data(locations, menu_lookup)
    enrich_locations_with_details(locations)
    normalize_location_hours(locations)

    payload["menu_sources"] = menu_sources
    icon_legend = build_icon_legend(locations)
    if icon_legend:
        payload["icon_legend"] = icon_legend

    tz = ZoneInfo(DEFAULT_TIMEZONE)
    payload["scraped_at"] = datetime.now(tz).isoformat(timespec="seconds")
    payload["timezone"] = DEFAULT_TIMEZONE


def run_scrape(url: str, output_path: Path) -> Dict[str, Any]:
    """Perform a single scrape cycle and persist the JSON."""
    html = fetch_html(url)
    parsed = parse_document(html)
    enrich_data(parsed)
    output_path.write_text(json.dumps(parsed, indent=2))
    return parsed


def parse_daily_time(value: str) -> time:
    """Parse a HH:MM string into a time object."""
    try:
        hour_str, minute_str = value.split(":", 1)
        hour = int(hour_str)
        minute = int(minute_str)
    except (ValueError, AttributeError):
        raise argparse.ArgumentTypeError("Time must be in HH:MM format") from None

    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise argparse.ArgumentTypeError("Hours must be 0-23 and minutes 0-59")

    return time(hour=hour, minute=minute)


def seconds_until(target: time, tz: ZoneInfo) -> float:
    """Compute seconds until the next occurrence of target time in the given timezone."""
    now = datetime.now(tz)
    today_target = datetime.combine(now.date(), target, tzinfo=tz)
    if today_target <= now:
        today_target += timedelta(days=1)
    return (today_target - now).total_seconds()


def run_daily(url: str, output_path: Path, target_time: time, tz_name: str) -> None:
    """Continuously run the scraper once per day at the requested time."""
    try:
        tz = ZoneInfo(tz_name)
    except ZoneInfoNotFoundError as exc:
        raise SystemExit(f"Unknown timezone '{tz_name}'.") from exc

    while True:
        wait_seconds = seconds_until(target_time, tz)
        next_run = datetime.now(tz) + timedelta(seconds=wait_seconds)
        print(f"Next scrape scheduled for {next_run.isoformat(timespec='minutes')}")
        time_module.sleep(wait_seconds)
        try:
            data = run_scrape(url, output_path)
            print(
                f"Updated menu data for {len(data['locations'])} locations at "
                f"{datetime.now(tz).isoformat(timespec='minutes')}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Scrape failed: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape the UC Berkeley dining menus page into JSON."
    )
    parser.add_argument(
        "--url",
        default=MENU_URL,
        help="Menus page URL (defaults to the Berkeley Dining menus page).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Where to write the JSON output (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--daily",
        action="store_true",
        help="Run continuously and refresh the JSON once per day.",
    )
    parser.add_argument(
        "--time",
        type=parse_daily_time,
        default=parse_daily_time("00:01"),
        help="Target time for the daily refresh in HH:MM (24h) format (default: 00:01).",
    )
    parser.add_argument(
        "--timezone",
        default=DEFAULT_TIMEZONE,
        help=f"Timezone name for scheduling daily refreshes (default: {DEFAULT_TIMEZONE}).",
    )
    args = parser.parse_args()

    if args.daily:
        print(
            f"Starting daily scraper targeting {args.time.strftime('%H:%M')} "
            f"{args.timezone} and writing to {args.output}"
        )
        try:
            run_daily(args.url, args.output, args.time, args.timezone)
        except KeyboardInterrupt:
            print("Daily scraper stopped.")
    else:
        parsed = run_scrape(args.url, args.output)
        print(f"Saved menu data for {len(parsed['locations'])} locations to {args.output}")


if __name__ == "__main__":
    main()
