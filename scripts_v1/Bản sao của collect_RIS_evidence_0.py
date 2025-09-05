from google.cloud import vision
import os 
from tqdm import tqdm
import time
import datetime
import sys 
import base64
import requests
import json
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import logging
import random
from urllib.parse import urlparse
import pandas as pd
import re

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils import *
from dataset_collection.scrape_utils import *
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('scraping_process.log'),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

def setup_driver(headless=True):
    """Configure WebDriver for Bing"""
    chrome_options = Options()
    if headless:
        chrome_options.add_argument('--headless=new')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('window-size=1920x1080')
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--lang=en-US")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("useAutomationExtension", False)
    return webdriver.Chrome(options=chrome_options)

def detect_web(image_path, api_key, how_many_queries=30):
    """Google Vision API implementation with better error handling"""
    if not api_key or api_key.strip() == " ":
        logger.warning("Google Vision API key not provided")
        return [], {}, {}
    
    # Thêm kiểm tra định dạng ảnh
    valid_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')
    if not image_path.lower().endswith(valid_extensions):
        logger.error(f"Unsupported image format: {image_path}")
        return [], {}, {}

    api_url = f'https://vision.googleapis.com/v1/images:annotate?key={api_key}'
    try:
        # Validate image file
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return [], {}, {}

        # Kiểm tra kích thước ảnh (tối đa 4MB)
        file_size = os.path.getsize(image_path)
        if file_size > 4 * 1024 * 1024:  # 4MB
            logger.error(f"Image file too large: {file_size/1024/1024:.2f}MB")
            return [], {}, {}

        with open(image_path, 'rb') as image_file:
            image_content = image_file.read()
            if len(image_content) == 0:
                logger.error(f"Empty image file: {image_path}")
                return [], {}, {}

            image_data = base64.b64encode(image_content).decode('utf-8')
            payload = {
                "requests": [{
                    "image": {"content": image_data},
                    "features": [{"type": "WEB_DETECTION", "maxResults": how_many_queries}],
                    "imageContext": {
                        "languageHints": ["en", "vi"]  # Thêm ngôn ngữ gợi ý
                    }
                }]
            }
            
            response = requests.post(api_url, 
                                   json=payload, 
                                   timeout=30,
                                   headers={'Content-Type': 'application/json'})
            
            # Thêm logging chi tiết khi có lỗi
            if response.status_code != 200:
                logger.error(f"Google Vision API error: {response.status_code} - {response.text}")
                return [], {}, {}

            annotations = response.json()['responses'][0]['webDetection']
            page_urls = []
            matching_image_urls = {}
            visual_entities = {}

            if 'pagesWithMatchingImages' in annotations:
                for page in annotations['pagesWithMatchingImages']:
                    page_urls.append(page['url'])
                    matching_image_urls[page['url']] = [
                        *(image['url'] for image in page.get('fullMatchingImages', [])),
                        *(image['url'] for image in page.get('partialMatchingImages', []))
                    ]

            if 'webEntities' in annotations:
                visual_entities = {
                    entity['description']: entity['score']
                    for entity in annotations['webEntities']
                    if 'description' in entity
                }

            return page_urls, matching_image_urls, visual_entities

    except requests.exceptions.RequestException as e:
        logger.error(f"Google Vision API request failed for {image_path}: {str(e)}")
    except Exception as e:
        logger.error(f"Google Vision API processing failed for {image_path}: {str(e)}")
    
    return [], {}, {}

def detect_bing(image_path, headless=True, max_wait=30):
    """Improved Bing Reverse Image Search with better error handling"""
    logger.info(f"[Bing] Processing {image_path}")
    
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        return []

    driver = None
    try:
        # Convert image to PNG if needed
        try:
            from PIL import Image
            img = Image.open(image_path)
            if img.format != 'PNG':
                new_path = os.path.splitext(image_path)[0] + '.png'
                img.save(new_path)
                image_path = new_path
        except Exception as e:
            logger.warning(f"Image conversion warning: {e}")

        driver = setup_driver(headless)
        driver.get('https://www.bing.com/images')
        time.sleep(2)

        # Try multiple selectors for image upload button
        camera_selectors = [
            'div#sb_sbi', 
            'div#sb_sbi[role="button"]',
            'button[aria-label*="Visual search"]',
            'label[for="vsbhtml_btn"]'
        ]
        
        clicked = False
        for sel in camera_selectors:
            try:
                btn = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, sel)))
                btn.click()
                clicked = True
                break
            except:
                continue

        if not clicked:
            logger.warning("Could not find camera button, trying direct upload")

        # Upload image
        file_input = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'input[type="file"]')))
        file_input.send_keys(os.path.abspath(image_path))

        # Wait for results with better conditions
        try:
            WebDriverWait(driver, max_wait).until(
                lambda d: d.find_elements(By.CSS_SELECTOR, '.imgpt, .iusc, .dg_u'))
        except:
            logger.warning(f"Timeout after {max_wait} seconds, trying to proceed anyway")

        # Parse results with more robust selectors
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        urls = set()
        
        # Improved selectors and parsing
        for tag in soup.select('a.iusc'):
            data = tag.get('m') or tag.get('data-m')
            if data:
                try:
                    data_json = json.loads(data)
                    for key in ['murl', 'purl', 'imgurl', 'turl']:
                        if key in data_json and data_json[key].startswith('http'):
                            urls.add(data_json[key])
                except json.JSONDecodeError:
                    continue

        # Additional fallback for direct links
        for a in soup.select('a[href^="http"]'):
            href = a.get('href')
            if href and any(x in href for x in ['/images/search?q=', '/images/detail/']):
                continue
            if href and href.startswith('http'):
                urls.add(href)

        return list(urls)

    except Exception as e:
        logger.error(f"Bing RIS failed: {str(e)}")
        if driver:
            driver.save_screenshot("bing_error.png")
        return []
    finally:
        if driver:
            driver.quit()

def is_scrapable_url(url):
    """Improved URL validation"""
    if not url or not isinstance(url, str):
        return False
        
    # Skip common media and social media URLs
    skip_exts = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.pdf', '.mp4', '.mp3']
    if any(url.lower().endswith(ext) for ext in skip_exts):
        return False
        
    skip_domains = [
        'facebook.com', 'twitter.com', 'youtube.com', 
        'instagram.com', 'tiktok.com', 'linkedin.com'
    ]
    
    try:
        domain = urlparse(url).netloc.lower()
        if any(sd in domain for sd in skip_domains):
            return False
    except:
        return False
        
    return True

def extract_with_retry(page_url, image_urls, max_retries=3, sleep_time=2):
    """
    Try to extract page info with retries.
    Now supports passing image_urls for Google + Bing RIS results.
    """
    for attempt in range(1, max_retries + 1):
        try:
            data = extract_info_trafilatura(page_url, image_urls=image_urls)
            # Nếu hàm trả về dict và không có error → coi như thành công
            if isinstance(data, dict) and 'error' not in data:
                return data
            else:
                print(f"[WARN] Attempt {attempt}: {data.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"[ERROR] Attempt {attempt}: {str(e)}")

        time.sleep(sleep_time)

    return {
        "url": page_url,
        "image_urls": image_urls,
        "error": f"Failed after {max_retries} attempts"
    }


def save_evidence_urls(filtered_results, output_path):
    """Save evidence URLs to a JSON file for future use"""
    evidence_urls = []
    for result in filtered_results:
        evidence_urls.append({
            "image_path": result["image_path"],
            "raw_url": result["raw_url"],
            "image_urls": result.get("image_urls", [])
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(evidence_urls, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved evidence URLs to {output_path}")

def main(args):
    """Main function with improved error handling and logging"""
    try:
        os.makedirs('dataset/retrieval_results', exist_ok=True)

        raw_ris_results = []
        image_files = [f for f in os.listdir(args.image_path)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for path in tqdm(image_files, desc="Processing images"):
            full_path = os.path.join(args.image_path, path)

            combined_urls = []
            combined_image_urls = {}
            google_visual_entities = {}

            if args.collect_google:
                google_page_urls, google_image_urls, google_visual_entities = detect_web(
                    full_path,
                    args.google_vision_api_key,
                    args.max_results
                )
                combined_urls.extend(google_page_urls)
                combined_image_urls.update(google_image_urls)

            if args.collect_bing:
                bing_page_urls = detect_bing(
                    full_path,
                    args.bing_headless,
                    args.bing_max_wait
                )
                combined_urls.extend(bing_page_urls)
                for u in bing_page_urls:
                    if u not in combined_image_urls:
                        combined_image_urls[u] = []

            raw_ris_results.append({
                'image_path': full_path,
                'urls': combined_urls,
                'image_urls': combined_image_urls,
                'visual_entities': google_visual_entities
            })

            time.sleep(args.sleep)
            print(f"Image: {full_path} - Found URLs: {combined_urls}")
        with open(args.raw_ris_urls_path, 'w', encoding='utf-8') as f:
            json.dump(raw_ris_results, f, indent=2, ensure_ascii=False)

        filtered_results = get_filtered_retrieval_results(
            args.raw_ris_urls_path,
        )

        if not filtered_results:
            logger.error("No URLs passed filtering. Skipping scraping to avoid missing file errors.")
            return

        save_evidence_urls(filtered_results, args.evidence_urls)

        if args.scrape_with_trafilatura:
            output = []
            failed_urls = []

            for row in tqdm(filtered_results, desc="Scraping URLs"):
                page_url = row['raw_url']
                image_urls = row.get('image_urls', [])
                result = extract_with_fallback(page_url, image_urls=image_urls)

                if 'error' in result:
                    failed_urls.append({'url': page_url, 'error': result['error']})
                    logger.warning(f"Failed to scrape {page_url}: {result['error']}")
                else:
                    output.append(result)

                if len(output) % 10 == 0 and output:
                    save_result(output, args.trafilatura_path)
                    output = []

            if output:
                save_result(output, args.trafilatura_path)
            if failed_urls:
                with open('failed_urls.json', 'w', encoding='utf-8') as f:
                    json.dump(failed_urls, f, indent=2, ensure_ascii=False)

        evidence_trafilatura = []
        if args.scrape_with_trafilatura:
            evidence_trafilatura = load_json(args.trafilatura_path)
        elif os.path.exists(args.trafilatura_path):
            evidence_trafilatura = load_json(args.trafilatura_path)

        dataset = []
        for split in ['train_custom', 'val_custom', 'test_custom']:
            dataset_path = f'dataset/{split}.json'
            if os.path.exists(dataset_path):
                dataset.extend(load_json(dataset_path))

        try:
            evidence = merge_data(
                evidence_trafilatura,
                filtered_results,
                dataset,
                apply_filtering=args.apply_filtering
            )

            if evidence is not None:
                evidence = evidence.fillna('')
                

                evidence_dict = evidence.to_dict(orient='records')

                def json_serializer(obj):
                    if pd.isna(obj):
                        return None
                    if isinstance(obj, (pd.Timestamp, datetime.datetime)):
                        return obj.isoformat()
                    if isinstance(obj, (np.integer)):
                        return int(obj)
                    if isinstance(obj, (np.floating)):
                        return float(obj)
                    if isinstance(obj, (np.ndarray)):
                        return obj.tolist()
                    return str(obj)

                with open(args.json_path, 'w', encoding='utf-8') as f:
                    json.dump(evidence_dict, f, indent=4, ensure_ascii=False, default=json_serializer)

                logger.info(f"Successfully saved evidence to {args.json_path}")
            else:
                logger.error("Merge data returned None, check your input data")

        except Exception as e:
            logger.error(f"Failed to merge data: {str(e)}")
            raise

    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collect evidence using Reverse Image Search.')
    parser.add_argument('--collect_google', type=int, default=0,
                      help='Whether to collect evidence URLs with Google Vision API')
    parser.add_argument('--collect_bing', type=int, default=0,
                      help='Whether to collect evidence URLs with Bing reverse image search')
    parser.add_argument('--evidence_urls', type=str, default='dataset/retrieval_results/evidence_urls.json',
                      help='Path to save evidence URLs for future use')
    parser.add_argument('--google_vision_api_key', type=str, default="",
                      help='API key for Google Vision API')
    parser.add_argument('--image_path', type=str, default='dataset/processed_img_vie/',
                      help='Path to directory containing images')
    parser.add_argument('--raw_ris_urls_path', type=str, default='dataset/retrieval_results/ris_results.json',
                      help='Path to save raw RIS results')
    parser.add_argument('--scrape_with_trafilatura', type=int, default=0,
                      help='Whether to scrape content from the URLs')
    parser.add_argument('--trafilatura_path', type=str, default='dataset/retrieval_results/trafilatura_data.json',
                      help='Path to save scraped content')
    parser.add_argument('--apply_filtering', type=int, default=0,
                      help='Whether to apply additional filtering (date, FC sources)')
    parser.add_argument('--json_path', type=str, default='dataset/retrieval_results/evidence.json',
                      help='Path to save final evidence JSON')
    parser.add_argument('--max_results', type=int, default=50,
                      help='Maximum number of results to collect per image')
    parser.add_argument('--sleep', type=int, default=3,
                      help='Seconds to sleep between API calls')
    parser.add_argument('--bing_headless', type=int, default=1,
                      help='Whether to run Bing in headless mode')
    parser.add_argument('--bing_max_wait', type=int, default=25,
                      help='Maximum wait time for Bing results in seconds')
    parser.add_argument(
        "--strict_filtering",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=True,
        help="Enable strict filtering of RIS results (default: True). Set to False to keep all URLs with image_path & raw_url."
    )
    args = parser.parse_args()
    
    # Validate Google API key
    if args.collect_google and (not args.google_vision_api_key or args.google_vision_api_key.strip() == ""):
        logger.error("Google Vision API key is required when collect_google is enabled")
        sys.exit(1)
    
    main(args)
