# scripts/auto_crawl_vtv.py
import os
import sys
import json
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from urllib.parse import urljoin

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset_collection')))
from scrape_utils import vtv_parser, scrape_image

BASE_URL = "https://vtv.vn"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
CATEGORIES = ["cong-nghe", "xa-hoi", "the-gioi", "giao-duc", "kinh-te"]


def get_article_links(category, page):
    url = f"{BASE_URL}/{category}.htm" if page == 1 else f"{BASE_URL}/{category}.htm?trang={page}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
    except Exception as e:
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    links = set()

    # Chỉ lấy link bài viết, có pattern dạng -100[0-9]{12}.htm
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.endswith(".htm") and "-100" in href:
            full_url = urljoin(BASE_URL, href)
            links.add(full_url)

    return list(links)



def save_article_and_image(url):
    text, image_urls = vtv_parser(url)
    article_id = url.split("/")[-1].replace(".htm", "")

    article_path = f"dataset/article_crawl/{article_id}.txt"
    os.makedirs("dataset/article_crawl", exist_ok=True)
    with open(article_path, "w", encoding="utf-8") as f:
        f.write(f"URL: {url}\n{text}")

    local_images = []
    os.makedirs("dataset/img_crawl", exist_ok=True)
    for idx, im_url in enumerate(image_urls):
        img_path = scrape_image(im_url, f"{article_id}_{idx}")
        if img_path:
            local_images.append(img_path)

    return {
        "URL": url,
        "image_path": local_images[0] if local_images else None,
        "image_URL": image_urls[0] if image_urls else None,
        "org": "vtv",
        "publication_date": "not enough information",
        "claim": text[:2000] + "...",
        "provenance": "not enough information",
        "source": "not enough information",
        "date": "not enough information",
        "date_numeric_label": "not enough information",
        "location": "not enough information",
        "motivation": "not enough information",
        "type_of_image": "not enough information",
        "verification_strategy": [],
        "verification_tool": [],
        "claimed_location": "not enough information",
        "claimed_date": "not enough information"
    }


def crawl_articles_to_json():
    results = []
    for cat in CATEGORIES:
        print(f"[Crawling category] {cat}")
        collected_urls = set()
        for page in range(1, 6):
            links = get_article_links(cat, page)
            if not links:
                continue
            for link in tqdm(links):
                if link in collected_urls:
                    continue
                try:
                    article = save_article_and_image(link)
                    if article:
                        results.append(article)
                        collected_urls.add(link)
                except Exception as e:
                    print(f"[ERROR] Failed to parse/save {link}: {e}")

    with open("dataset/articles_vtv.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    crawl_articles_to_json()
