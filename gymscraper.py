import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Setup headless Chrome
options = Options()
options.add_argument("--headless")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

driver.get("https://recwell.berkeley.edu/facilities/recreational-sports-facility-rsf/rsf-weight-room-crowd-meter/")

# Switch to iframe
iframe = WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.XPATH, '//iframe[@title="Weightroom Capacity"]'))
)
driver.switch_to.frame(iframe)

# Wait until any span inside the iframe contains "% Full"
def span_has_full_text(driver):
    spans = driver.find_elements(By.TAG_NAME, "span")
    for span in spans:
        if "% Full" in span.text:
            return span
    return False

full_span = WebDriverWait(driver, 20).until(span_has_full_text)

# Extract only the numeric part using regex
match = re.search(r'\d+', full_span.text)
if match:
    occupancy_percent = int(match.group())
    print(f"Current weight room occupancy: {occupancy_percent}%")
else:
    print("Could not extract occupancy number.")

driver.quit()
