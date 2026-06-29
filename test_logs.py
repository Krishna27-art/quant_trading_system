import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

options = Options()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

try:
    driver = webdriver.Chrome(options=options)
    driver.get("http://localhost:3000")
    time.sleep(3)  # Wait for JS to run
    logs = driver.get_log("browser")
    for log in logs:
        print(f"[{log['level']}] {log['message']}")
    driver.quit()
except Exception as e:
    print(f"Error: {e}")
