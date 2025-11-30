import undetected_chromedriver as uc
from bs4 import BeautifulSoup
import csv
import time
import random
import logging
import os
from tqdm import tqdm

# ==========================
# CONFIG
# ==========================
BASE_URLS = [
    "https://batdongsan.com.vn/nha-dat-ban-nghe-an",
    "https://batdongsan.com.vn/nha-dat-ban-ha-tinh",
    "https://batdongsan.com.vn/nha-dat-ban-khanh-hoa",
    "https://batdongsan.com.vn/nha-dat-ban-quang-nam",
    "https://batdongsan.com.vn/nha-dat-ban-da-nang",
    "https://batdongsan.com.vn/nha-dat-ban-thua-thien-hue",
    "https://batdongsan.com.vn/nha-dat-ban-quang-ngai",
    "https://batdongsan.com.vn/nha-dat-ban-gia-lai",
    "https://batdongsan.com.vn/nha-dat-ban-quang-tri",
    "https://batdongsan.com.vn/nha-dat-ban-phu-yen",
    "https://batdongsan.com.vn/nha-dat-ban-binh-dinh"

]

TARGET_TOTAL_ITEMS = 2500
#OUTPUT_FILE = "data/01_raw/craw_data_.csv"
OUTPUT_FILE = "data/01_raw/data_.csv"

# ==========================
# SETUP DRIVER
# ==========================
def init_driver():
    options = uc.ChromeOptions()
    options.add_argument('--no-first-run')

    driver = uc.Chrome(options=options)
    return driver

def append_csv(filepath, fieldnames, rows):
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"-> Đã tạo thư mục mới: {directory}")

    file_exists = os.path.isfile(filepath)
    with open(filepath, mode="a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)

def main():
    driver = init_driver()
    collected = 0
    page_counters = {url: 1 for url in BASE_URLS}
    active_urls = list(BASE_URLS)
    
    existing_links = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_links.add(row['link'])
                collected += 1

    print(f"==================================================")
    print(f"Đã có sẵn: {collected} bài.")
    print(f"Mục tiêu: {TARGET_TOTAL_ITEMS} bài.")
    print(f"Cần crawl thêm: {TARGET_TOTAL_ITEMS - collected} bài nữa.")
    print(f"==================================================")

    try:
        while collected < TARGET_TOTAL_ITEMS and active_urls:
            for base_url in list(active_urls):
                current_page = page_counters[base_url]
                url = base_url if current_page == 1 else f"{base_url}/p{current_page}"
                
                print(f"--------------------------------------------------")
                print(f"Đang truy cập danh sách: {url}")
                
                try:
                    driver.get(url)
                    time.sleep(random.uniform(5, 8))
                    
                    # Kiểm tra nếu bị Cloudflare chặn (có thể cần giải captcha tay lần đầu)
                    if "challenge" in driver.title.lower():
                        print("!!! PHÁT HIỆN CAPTCHA. Hãy giải captcha trên trình duyệt trong 30s...")
                        time.sleep(30)

                    soup = BeautifulSoup(driver.page_source, "html.parser")
                    card_links = soup.select("a.js__product-link-for-product-id")
                    
                    if not card_links:
                        print(f"Không thấy bài viết nào ở {url}. Có thể hết trang.")
                        active_urls.remove(base_url)
                        continue

                    links_in_page = []
                    for a in card_links:
                        href = a.get("href")
                        if href:
                            if href.startswith("/"): href = "https://batdongsan.com.vn" + href
                            links_in_page.append(href)
                    
                    # Lọc trùng
                    links_in_page = list(set(links_in_page))
                    
                    # Vào từng bài chi tiết
                    for link in links_in_page:
                        if collected >= TARGET_TOTAL_ITEMS: break
                        if link in existing_links: continue
                        
                        driver.get(link)
                        time.sleep(random.uniform(3, 6))
                        
                        detail_soup = BeautifulSoup(driver.page_source, "html.parser")
                        
                        title = detail_soup.select_one("h1.pr-title")
                        content_div = detail_soup.select_one(".re__section-body, .re__detail-content, .pm-desc")
                        
                        if title and content_div:
                            row = {
                                "link": link,
                                "province": base_url.split("-")[-1],
                                "title": title.get_text(strip=True),
                                "content": " ".join(content_div.get_text(separator=" ").split()),
                                "price_summary": ""
                            }
                            append_csv(OUTPUT_FILE, ["link", "province", "title", "content", "price_summary"], [row])
                            existing_links.add(link)
                            collected += 1
                            print(f"-> [OK] {collected}/{TARGET_TOTAL_ITEMS}: {row['title'][:50]}...")
                        else:
                            print(f"-> [Lỗi] Không lấy được nội dung: {link}")

                except Exception as e:
                    print(f"Lỗi: {e}")
                    time.sleep(5)
                
                page_counters[base_url] += 1
                if collected >= TARGET_TOTAL_ITEMS: break

    except KeyboardInterrupt:
        print("Dừng chương trình.")
    finally:
        driver.quit()

if __name__ == "__main__":
    main()