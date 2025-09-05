
import os
import sys
import json
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from urllib.parse import urljoin, urlparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset_collection')))
from scrape_utils import scrape_image, kenh14_parser

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

SOURCES = {
    "kenh14": {
        "BASE_URL": "https://kenh14.vn",
        "CATEGORIES": ["xa-hoi.chn", "doi-song.chn", "the-gioi-do-day.chn", "money-z.chn"],
        "PARSER": kenh14_parser,
        "ARTICLE_PATH": "dataset/article_crawl/kenh14",
        "IMAGE_PATH": "dataset/img_crawl/kenh14"
    }
}

def get_article_links(source, category, page):
    base_url = SOURCES[source]["BASE_URL"]
    if source == "tinmoi":
        url = f"{base_url}/{category}/" if page == 1 else f"{base_url}/{category}/trang-{page}.html"
    elif source == "kenh14":
        url = f"{base_url}/{category}"
        if page > 1:
            url += f"?page={page}"
    else:
        return []

    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
    except Exception:
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.endswith(".chn") or href.endswith(".html"):
            full_url = urljoin(base_url, href)
            links.add(full_url)
    return list(links)

def save_article_and_image(source, url):
    parser = SOURCES[source]["PARSER"]
    text, image_urls = parser(url)
    if not text:
        return None

    article_id = urlparse(url).path.strip("/").replace("/", "_").replace(".html", "").replace(".chn", "")
    article_path = os.path.join(SOURCES[source]["ARTICLE_PATH"], f"{article_id}.txt")
    os.makedirs(SOURCES[source]["ARTICLE_PATH"], exist_ok=True)
    with open(article_path, "w", encoding="utf-8") as f:
        f.write(f"URL: {url}\n{text}")

    local_images = []
    os.makedirs(SOURCES[source]["IMAGE_PATH"], exist_ok=True)
    for idx, im_url in enumerate(image_urls):
        img_path = scrape_image(im_url, f"{source}_{article_id}_{idx}")
        if img_path:
            local_images.append(img_path)

    return {
        "URL": url,
        "image_path": local_images[0] if local_images else None,
        "image_URL": image_urls[0] if image_urls else None,
        "org": source,
        "publication_date": "not enough information",
        "claim": text[:2000] + "...",
        "provenance": "not enough information",
        "source": "not enough information",
        "date": "not enough information",
        "date_numeric_label": "not enough information",
        "location": "not enough information",
        "motivation": "not enough information",
        "type_of_image": "manipulated",
        "verification_strategy": [],
        "verification_tool": [],
        "claimed_location": "not enough information",
        "claimed_date": "not enough information"
    }

def crawl_articles_to_json():
    results = []
    for source in SOURCES:
        print(f"[SOURCE] {source}")
        for cat in SOURCES[source]["CATEGORIES"]:
            print(f"[CATEGORY] {cat}")
            collected_urls = set()
            for page in range(1, 4):
                links = get_article_links(source, cat, page)
                for link in tqdm(links):
                    if link in collected_urls:
                        continue
                    try:
                        article = save_article_and_image(source, link)
                        if article:
                            results.append(article)
                            collected_urls.add(link)
                    except Exception as e:
                        print(f"[ERROR] Failed to parse/save {link}: {e}")

    with open("dataset/articles_kenh14.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    crawl_articles_to_json()
