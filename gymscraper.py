from __future__ import annotations

import re
from contextlib import suppress
from typing import Optional

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

RSF_WEIGHT_ROOM_URL = (
    "https://recwell.berkeley.edu/facilities/recreational-sports-facility-rsf/rsf-weight-room-crowd-meter/"
)


def _span_has_full_text(driver: webdriver.Chrome) -> Optional[str]:
    """Return the text content of the span that contains '% Full', if present."""
    spans = driver.find_elements(By.TAG_NAME, "span")
    for span in spans:
        if "% Full" in span.text:
            return span.text
    return None


def fetch_rsf_occupancy_percent(timeout: int = 20) -> int:
    """Scrape the RSF weight room crowd meter and return the occupancy percentage."""
    options = Options()
    # new headless helps avoid deprecated warning and better parity with Chrome 109+
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    try:
        driver.get(RSF_WEIGHT_ROOM_URL)
        iframe = WebDriverWait(driver, 10).until(
            lambda d: d.find_element(By.XPATH, '//iframe[@title="Weightroom Capacity"]')
        )
        driver.switch_to.frame(iframe)
        full_text = WebDriverWait(driver, timeout).until(_span_has_full_text)
        if not full_text:
            raise RuntimeError("RSF occupancy span not found")

        match = re.search(r"(\d+)", full_text)
        if not match:
            raise RuntimeError(f"Could not extract occupancy number from '{full_text}'")
        return int(match.group(1))
    finally:
        with suppress(Exception):
            driver.quit()


if __name__ == "__main__":
    try:
        occupancy = fetch_rsf_occupancy_percent()
    except Exception as exc:  # pragma: no cover - convenience path
        print(f"Could not extract occupancy number: {exc}")
    else:
        print(f"Current weight room occupancy: {occupancy}%")
