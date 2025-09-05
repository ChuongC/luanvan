import Levenshtein as lev
from dateutil.tz import tzutc
from dateutil import parser
import requests
#from trafilatura import bare_extraction
from bs4 import BeautifulSoup as bs
import pandas as pd
from PIL import Image
from io import BytesIO
import requests as rq
import re
import os
import time
import numpy as np
from tqdm import tqdm
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils import *

import random
import time
import imagehash

from bs4 import BeautifulSoup
import requests

import logging
logger = logging.getLogger(__name__)

def robust_get(url: str, timeout: int = 10) -> str:
    """
    Thực hiện yêu cầu HTTP một cách an toàn và trả về nội dung HTML (hoặc rỗng nếu thất bại).
    """
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        if response.status_code == 200:
            return response.text
    except Exception as e:
        print(f"[robust_get] Failed to fetch {url}: {e}")
    return ""

def scrape_metadata(soup: BeautifulSoup, url: str) -> dict:
    """
    Trích xuất metadata cơ bản từ nội dung HTML qua BeautifulSoup.
    """
    title = soup.title.string.strip() if soup.title and soup.title.string else ''

    description = ''
    desc_tag = soup.find('meta', attrs={'name': 'description'}) \
        or soup.find('meta', attrs={'property': 'og:description'})
    if desc_tag:
        description = desc_tag.get('content', '').strip()

    # Tổng hợp tất cả các thẻ <p>
    paragraphs = soup.find_all('p')
    full_text = ' '.join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))

    return {
        "title": title,
        "description": description,
        "full_text": full_text,
        "url": url
    }

def get_basename_from_url(url: str) -> str:
    base = url.rstrip('/').split('/')[-1]
    base = re.sub(r'[^a-zA-Z0-9_-]', '', base)
    return base or 'vietfact_article'

def get_initial_image_counter(folder='dataset/img/'):
    os.makedirs(folder, exist_ok=True)
    existing = [f for f in os.listdir(folder) if f.startswith('vietfact_') and f.endswith('.png')]
    nums = [int(re.search(r'vietfact_(\d+).png', f).group(1)) for f in existing if re.search(r'vietfact_(\d+).png', f)]
    return max(nums) + 1 if nums else 0

existing_hashes = set()
image_counter = get_initial_image_counter()

def is_valid_article_url(url: str) -> bool:
    """
    Kiểm tra xem URL có phải là link bài viết hợp lệ từ vietfactcheck.org không.
    """
    cleaned_url = url.split('?')[0].split('#')[0].strip()

    # Chấp nhận slug có chứa a-zA-Z0-9- và dấu kết thúc /
    pattern = re.compile(
        r'^https?://vietfactcheck\.org/\d{4}/\d{2}/\d{2}/[a-zA-Z0-9\-_]+/?$'
    )

    is_valid = bool(pattern.match(cleaned_url))
    print(f"[is_valid_article_url] Checking URL: {url} -> Cleaned: {cleaned_url} -> Valid: {is_valid}")
    return is_valid


def convert_to_wayback_url(original_url):
    return f"https://web.archive.org/cdx/search/cdx?url={original_url}&output=json&limit=1&filter=statuscode:200&collapse=digest"

def vietfactcheck_parser(url, only_vietnamese=False, skipped_log_path='dataset/skipped_urls.txt'):
    print(f"\n📰 Phân tích bài viết từ: {url}")
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
        soup = bs(res.text, 'html.parser')
    except Exception as e:
        print(f"❌ Không thể truy cập URL: {url}. Lỗi: {e}")
        return '', []

    # Chấp nhận bất kỳ ảnh nào trừ khi nó chứa từ khóa cấm
    image_candidates = soup.select('article img') or soup.select('div.entry-content img')
    main_img_tag = None
    for img in image_candidates:
        src = img.get('src', '')
        if not src:
            continue
        ban_keywords = ['logo', 'icon', 'scale', 'ruler', 'measure']
        if any(kw in src.lower() for kw in ban_keywords):
            continue
        main_img_tag = img
        break

    if not main_img_tag:
        print("🚫 Không tìm được ảnh hợp lệ")
        return '', []

    src = main_img_tag.get('src', '').split('?')[0]
    print(f"✅ Ảnh chính: {src}")
    image_urls = [src]

    if only_vietnamese:
        category_block = soup.find('span', class_='meta-category')
        is_vietnamese = False
        if category_block:
            is_vietnamese = any('category/vietnamese' in a.get('href', '') for a in category_block.find_all('a'))

        if not is_vietnamese:
            content_div = soup.find('div', class_='entry-content')
            viet_url = ''
            if content_div:
                for a in content_div.find_all('a', href=True):
                    href = a['href']
                    if 'vietfactcheck.org' in href and '202' in href and href != url:
                        viet_url = href
                        break
            if viet_url:
                viet_url = re.sub(r'([&?])preview=true', '', viet_url)
                print(f"🔁 Chuyển hướng tới bản tiếng Việt: {viet_url}")
                return vietfactcheck_parser(viet_url, only_vietnamese=True, skipped_log_path=skipped_log_path)
            else:
                print("🚫 Không tìm thấy bản tiếng Việt")
                with open(skipped_log_path, 'a', encoding='utf-8') as logf:
                    logf.write(f"{url}\n")
                return '', []

    title_tag = soup.find('h1', class_='entry-title')
    title = title_tag.get_text(strip=True) if title_tag else 'Không có tiêu đề'
    date_tag = soup.find('time', class_='entry-date')
    date_text = date_tag.get('datetime', '') if date_tag else 'Không có ngày'
    content_div = soup.find('div', class_='entry-content')

    if not content_div:
        print("⚠️ Không có div.entry-content")
        return '', []

    paragraphs = content_div.find_all('p')
    content = '\n'.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])

    print(f"📝 Tiêu đề: {title} | Ngày: {date_text} | Đoạn văn: {len(paragraphs)}")

    text_output = f"URL: {url}\n\nTiêu đề: {title}\nNgày: {date_text}\n\nNội dung:\n{content}\n\nĐường dẫn hình ảnh:\n" + '\n'.join(image_urls)
    return text_output, image_urls[:1], get_basename_from_url(url)


def scrape_image(original_url, basename):
    global image_counter
    print(f"🖼️ Đang tải ảnh từ: {original_url}")
    headers = {'User-Agent': 'Mozilla/5.0'}
    clean_url = original_url
    banned_keywords = ['logo', 'icon', 'scale', 'ruler', 'measure']

    if any(kw in clean_url.lower() for kw in banned_keywords):
        print(f"🚫 Bỏ qua ảnh không hợp lệ: {clean_url}")
        return False

    file_path = 'dataset/img/'
    os.makedirs(file_path, exist_ok=True)
    img_file_name = f"vietfact_{basename}.png"
    save_path = os.path.join(file_path, img_file_name)
    log_path = 'dataset/success_images.txt'

    cdx_query = convert_to_wayback_url(original_url)
    print(f"🔎 CDX Query: {cdx_query}")
    try:
        res = requests.get(cdx_query, headers=headers, timeout=15)
        if res.status_code == 200:
            data = res.json()
            if len(data) >= 2:
                timestamp = data[1][1]
                download_url = f"https://web.archive.org/web/{timestamp}/{original_url.split('?')[0]}"
                print(f"📸 Tải từ Wayback snapshot: {download_url}")
                req = requests.get(download_url, stream=True, timeout=(10, 10), headers=headers)
                if req.status_code == 200 and 'image' in req.headers.get('Content-Type', ''):
                    image_content = req.content
                    img = Image.open(BytesIO(image_content)).convert('RGB')
                    img_hash = str(imagehash.phash(img))
                    if img_hash in existing_hashes:
                        print(f"⚠️ Trùng ảnh (hash: {img_hash})")
                        return False
                    existing_hashes.add(img_hash)
                    img.save(save_path)
                    print(f"✅ Ảnh lưu từ Wayback: {save_path}")
                    with open(log_path, 'a', encoding='utf-8') as logf:
                        logf.write(f"Wayback | {save_path} | {original_url}\n")
                    image_counter += 1
                    time.sleep(random.uniform(1, 2))
                    return True
                else:
                    print(f"⚠️ Ảnh từ Wayback lỗi: status {req.status_code} | content-type {req.headers.get('Content-Type')}")
    except Exception as e:
        print(f"❌ CDX lỗi: {e}")

    print("🔁 Thử tải từ gốc")
    try:
        fallback_headers = {
            'User-Agent': 'Mozilla/5.0',
            'Referer': original_url
        }
        req = requests.get(clean_url, stream=True, timeout=(10, 10), headers=fallback_headers)
        print(f"🧪 Fallback URL: {req.url} | Status: {req.status_code} | Content-Type: {req.headers.get('Content-Type')}")

        if req.status_code == 200 and 'image' in req.headers.get('Content-Type', ''):
            image_content = req.content
            img = Image.open(BytesIO(image_content)).convert('RGB')
            img_hash = str(imagehash.phash(img))
            if img_hash in existing_hashes:
                print(f"⚠️ Trùng ảnh (hash: {img_hash})")
                return False
            existing_hashes.add(img_hash)
            img.save(save_path)
            print(f"✅ Fallback lưu ảnh từ gốc: {save_path}")
            with open(log_path, 'a', encoding='utf-8') as logf:
                logf.write(f"Fallback | {save_path} | {original_url}\n")
            image_counter += 1
            return True

        if req.status_code == 404 and 'i0.wp.com' in clean_url:
            fallback_with_query = clean_url + '?resize=1320%2C880&ssl=1'
            print(f'🔁 Thử lại fallback với query: {fallback_with_query}')
            try:
                req = requests.get(fallback_with_query, stream=True, timeout=(10, 10), headers=fallback_headers)
                if req.status_code == 200 and 'image' in req.headers.get('Content-Type', ''):
                    image_content = req.content
                    img = Image.open(BytesIO(image_content)).convert('RGB')
                    img_hash = str(imagehash.phash(img))
                    if img_hash in existing_hashes:
                        print(f"⚠️ Trùng ảnh (hash: {img_hash})")
                        return False
                    existing_hashes.add(img_hash)
                    img.save(save_path)
                    print(f"✅ Fallback lưu ảnh từ bản resize: {save_path}")
                    with open(log_path, 'a', encoding='utf-8') as logf:
                        logf.write(f"Fallback Query | {save_path} | {fallback_with_query}\n")
                    image_counter += 1
                    return True
                else:
                    print(f"❌ Fallback query cũng thất bại: {req.status_code} | {req.headers.get('Content-Type')}")
            except Exception as e:
                print(f"❌ Lỗi khi thử fallback query: {e}")

        print(f"❌ Fallback thất bại: status {req.status_code} | content-type {req.headers.get('Content-Type')}")
        with open('dataset/missing_images.txt', 'a', encoding='utf-8') as logf:
            logf.write(f"{original_url} | Fallback failed | status: {req.status_code} | content-type: {req.headers.get('Content-Type')}\n")
    except Exception as e:
        print(f"❌ Fallback lỗi: {e}")
        with open('dataset/missing_images.txt', 'a', encoding='utf-8') as logf:
            logf.write(f"{original_url} | Fallback exception: {e}\n")

    if not os.path.exists(save_path):
        with open('dataset/missing_images.txt', 'a', encoding='utf-8') as logf:
            logf.write(f"{original_url} | Không thể lưu ảnh\n")

    return False


def load_urls(file_path):
    '''
    Load all fact-checking articles URLs as a list
    '''
    with open(file_path, 'r') as file:
        # Read the lines and create a list
        url_list = [line.strip() for line in file]
        #Remove duplicates
        url_list = list(set(url_list))
        file.close()
    return url_list


def is_english_article(url):
    '''
    Verify if the  article is saved in English.
    Some articles of Factly are written in Kannada and Telugu.
    Some articles of Pesacheck are written in French.
    '''
    if 'telugu' in url or 'kannada' in url:
        return False
    if 'faux' in url or "intox" in url or 'ces-photo' in url or 'cette-photo' in url or 'cette-image' in url or 'ces-images' in url:
        return False
    return True


def pesacheck_parser(url):
    '''
    Scrape an URL using request and parse it with BeautifulSoup to collect the FC article data and the image url.
    Custom script for Pesacheck articles.
    '''
    try:
        headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        req = rq.get(url,headers=headers).text
    except :
        return '', []
    soup = bs(req, 'html.parser')
    title = soup.title.text + '\n'
    pub_date_tag = soup.find('meta', {'property': 'article:published_time'})

    if pub_date_tag:
        publication_date = "Publication Date:" + pub_date_tag.get('content', 'No publication date found')
    else:
        publication_date = "Publication date not found in the HTML."
    filtered_paragraphs = soup.find_all('p')
    try:
        text = title + '\n' + publication_date + '\n' + '\n'.join([p.get_text() for p in filtered_paragraphs]).split('This post is part of an ongoing series of PesaCheck')[0].split('--')[1]
    except:
        text = title + '\n' + publication_date + '\n' + '\n'.join([p.get_text() for p in filtered_paragraphs])
    image_urls = [img['srcset']  for img in  soup.find_all('source', type='image/webp')]
    try:
        image_urls = [image_urls[1].split(',')[0].split()[0]]
    except:
        image_urls = ''
    text += '\nImage URLs :\n' + '\n'.join(image_urls)
    return text, image_urls


def two11org_parser(url):
    '''
    Scrape an URL using request and parse it with BeautifulSoup to collect the FC article data and the image url.
    Custom script for 211Check articles.
    '''
    try:
        headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        req = rq.get(url,headers=headers).text
    except :
        return '', []
    soup = bs(req, 'html.parser')
    title = soup.title.text + '\n'
    pub_date_tag = soup.find('meta', {'property': 'article:published_time'})

    if pub_date_tag:
        publication_date = "Publication Date:" + pub_date_tag.get('content', 'No publication date found')
    else:
        publication_date = "Publication date not found in the HTML."
    filtered_paragraphs = soup.find_all('p')
    text = title + '\n' + publication_date + '\n' + '\n'.join([p.get_text() for p in filtered_paragraphs]).split('Name *')[0]
    image_urls = [[img['src'] for img in soup.find_all('img')][2]]
    text += '\nImage URLs :\n' + '\n'.join(image_urls)

    return text, image_urls


def factly_parser(url):
    '''
    Scrape an URL using request and parse it with BeautifulSoup to collect the FC article data and the image url.
    Custom script for Factly articles.
    '''
    try:
        headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        req = rq.get(url,headers=headers).text
    except :
        return '', []
    soup = bs(req, 'html.parser')
    title = soup.title.text + '\n'
    pub_date_tag = soup.find('meta', {'property': 'article:published_time'})

    if pub_date_tag:
        publication_date = "Publication Date:" + pub_date_tag.get('content', 'No publication date found')
    else:
        publication_date = "Publication date not found in the HTML."
    filtered_paragraphs = soup.find_all('p')
    text = title + '\n' + publication_date + '\n' + '\n'.join([p.get_text() for p in filtered_paragraphs]).split('FACTLY is one of the well known Data Journalism/Public Information portals in India.')[0]
    image_urls = [img['src']  for img in  soup.find_all('img')]
    image_urls = [i for i in image_urls if 'logo' not in i.lower() and 'thumbnail' not in i.lower()][5:6]
    text += '\nImage URLs :\n' + '\n'.join(image_urls)
    return text, image_urls


def collect_articles(urls,
                     parser,
                     scrape_images=True,
                     image_urls=None,
                     sleep=10):
    '''
    Collect the fact-checking articles and images based on their URLs.
    '''
    img_urls_unique = set()
    if 'article' not in os.listdir('dataset/'):
        os.mkdir('dataset/article/')
    for u in tqdm(range(len(urls))):
        files = [f.split('.txt')[0] for f in os.listdir('dataset/article/')]
        is_new_article=True
        if  urls[u].split('/')[-1].split('?')[0]  in files:
            is_new_article = False
            print('Already scraped : ' + urls[u].split('/')[-1].split('?')[0])
        if is_new_article:
            path = 'dataset/article/'+ urls[u].split('/')[-1].split('?')[0] + '.txt'
            text, scraped_image_urls = parser(urls[u]) #Use a platform specific parser
            scraped_image_urls = [img for img in scraped_image_urls if img not in img_urls_unique]
            for img in scraped_image_urls:
                    img_urls_unique.add(img)
            #Save text
            with open(path,'w',encoding='utf-8') as f:
                text = 'URL: ' + urls[u] + '\n' + text
                f.write(text)
            if scrape_images:
                #Scrape the image and save the content
                if 'img' not in os.listdir('dataset/'):
                    os.mkdir('dataset/img/')
                if image_urls!= None:
                    #A reference image url is already provided as part of the dataset
                    scrape_image(image_urls[u], path.split('/')[-1])
                else:
                    #If no existing image urls are provided, default to the scraped ones
                    for im_url in scraped_image_urls:
                        scrape_image(im_url, path.split('/')[-1])
            time.sleep(sleep)


def is_obfuscated_or_encoded(url):
    '''
    Check that the evidence url is not obfuscated or encoded.
    '''
    unquoted_url = url
    try:
        return '%' in unquoted_url or '//' in unquoted_url.split('/')[2]
    except:
        return True


def is_likely_html(url):
    '''
    Check that the evidence url is html
    '''
    # List of common file extensions
    file_extensions = ['.pdf', '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.doc', '.docx', '.ppt', '.pptx', '.xls',
                       '.xlsx', '.txt', '.zip', '.rar', '.exe', '.svg', '.mp4', '.avi', '.mp3']

    # Extract the extension from the URL
    extension = '.' + url.rsplit('.', 1)[-1].lower()

    # Check if the URL ends with a common file extension
    if extension in file_extensions:
        return False
    else:
        return True


def is_fc_organization(url):
    '''
    Check that the evidence url does not come from a FC organization
    Note: the provided list does not include every single existing FC organization. Some FC articles might still pass through this filter.
    '''
    fc_domains = ['https://www.fastcheck.cl','https://pesacheck.org','https://africacheck.org','https://www.snopes.com',
            'https://newsmobile.in', 'https://211check.org', 'factcrescendo.com/', 'https://leadstories.com', 'https://www.sochfactcheck.com',
            'https://newschecker.in','https://www.altnews.in', 'https://dubawa.org', 'https://factcheck.afp.com', 'factly.in',
            'https://misbar.com/factcheck/', 'larepublica.pe/verificador/', 'fatabyyano.net/', 'https://www.vishvasnews.com/', "newsmeter.in" ,
            "boomlive", "politifact","youturn.in", "lemonde.fr/les-decodeurs","factuel.afp.com","thequint.com", "logicalindian.com/fact-check/",
            "timesofindia.com/times-fact-check", "indiatoday.in/fact-check/", "vietfactcheck.org", "factly.in", "en.wikipedia.org", "smhoaxslayer.com", "facthunt.in", "aajtak.in/fact-check/",
            "bhaskar.com/no-fake-news", "theprint.in/hoaxposed/", 'firstdraftnews.org']
    for d in fc_domains :
        if d in url:
            return True
    return False


def is_banned(url):
    '''
    Check if the evidence url is in the list of banned urls
    '''
    banned = [
        #Those websites are flagged as potential unsafe or block the webscraping process
        "legalaiddc-prod.qed42.net", "windows8theme.net", "darkroom.baltimoresun.com", "dn.meethk.com", "hotcore.info", "pre-inscription.supmti.ac.ma",
        "hideaways.in", "www.alhurra.com/search?st=articleEx", "anonup.com", "hiliventures", "middleagerealestateagent", "nonchalantslip.fr",
        "corseprestige.com", ".aa.com.tr",  "landing.rassan.ir", "aiohotzgirl.com", "hotzxgirl.com",
        #The content of those social media websites is not scrapable.
        "facebook.com", "twitter.c", "youtube.co", "linkedin.co", "tiktok.c", "quora.c", "gettyimages.", "reddit." ]
    for b in banned:
        if b in url:
            return True
    return False


import re
from urllib.parse import urlparse

BANNED_EXTENSIONS = ('.pdf', '.zip', '.exe', '.rar', '.mp4', '.mp3', '.avi', '.wmv', '.svg')
SPAM_KEYWORDS = ('utm=', 'tracking', 'adclick', 'doubleclick.net', 'facebook.com/l.php', 't.co/')

def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme in ('http', 'https'), result.netloc])
    except:
        return False

def contains_spam_keyword(url):
    return any(k in url for k in SPAM_KEYWORDS)

def has_banned_extension(url):
    return url.lower().endswith(BANNED_EXTENSIONS)

import tldextract

MIN_URL_LENGTH = 15
MAX_QUERY_LENGTH = 150

BAD_TLDS = ('.xyz', '.top', '.club', '.click', '.work', '.party', '.gq', '.tk', '.ml')

def has_suspicious_tld(url):
    ext = tldextract.extract(url)
    return any(ext.suffix.endswith(tld.lstrip('.')) for tld in BAD_TLDS)

def has_excessive_query(url):
    parsed = urlparse(url)
    return parsed.query and len(parsed.query) > MAX_QUERY_LENGTH

def has_no_content_path(url):
    parsed = urlparse(url)
    # chỉ là domain, không có /path gì -> ít hữu ích
    return parsed.path in ('', '/')

def is_too_short(url):
    return len(url) < MIN_URL_LENGTH


def get_filtered_retrieval_results(path):
    """
    Filter RIS results: mức lọc nâng cao hơn, loại bỏ rác và file media.
    """
    ris_results = load_json(path)
    print(f"Loaded {len(ris_results)} RIS items from {path}")
    retrieval_results = []

    for item in ris_results:
        image_path = item['image_path']
        for url in item.get('urls', []):
            if not url or not is_valid_url(url):
                continue

            ris_data = {
                'image_path': image_path,
                'raw_url': url,
                'image_urls': item.get('image_urls', {}).get(url, []),
                'is_fc': is_fc_organization('/'.join(url.split('/')[:3])),
                'is_https': url.startswith('https')
            }
            ris_data['is_banned'] = is_banned(url)
            ris_data['is_obfuscated'] = is_obfuscated_or_encoded(url)
            ris_data['is_html'] = is_likely_html(url)
            ris_data['has_banned_ext'] = has_banned_extension(url)
            ris_data['has_spam_kw'] = contains_spam_keyword(url)

            # Advanced filtering
            ris_data['selection'] = (
                ris_data['is_html']
                and ris_data['is_https']
                and not ris_data['is_obfuscated']
                and not ris_data['is_banned']
                and not ris_data['has_banned_ext']
                and not ris_data['has_spam_kw']
                and not has_suspicious_tld(url)
                and not has_excessive_query(url)
                and not has_no_content_path(url)
                and not is_too_short(url)
            )

            retrieval_results.append(ris_data)

    selected_retrieval_results = [d for d in retrieval_results if d['selection']]
    print(f"URLs after filtering: {len(selected_retrieval_results)}")
    return selected_retrieval_results


# =========================
# Cleaning & Metadata
# =========================
def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    noise_patterns = [
        r'Xem thêm.*$',
        r'Related Articles.*$',
        r'Subscribe.*$'
    ]
    for pat in noise_patterns:
        text = re.sub(pat, '', text, flags=re.IGNORECASE)
    return text.strip()

import os
import json
import logging
from urllib.parse import urlparse
from datetime import datetime
import trafilatura
from newspaper import Article

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ===== Load pre-crawled data if exists =====
TRAFILATURA_JSON = "dataset/retrieval_results/trafilatura_data.json"
RIS_RESULTS_JSON = "dataset/retrieval_results/ris_results.json"

def load_json_if_exists(path):
    return json.load(open(path, "r", encoding="utf-8")) if os.path.exists(path) else []

trafilatura_data = load_json_if_exists(TRAFILATURA_JSON)
ris_results_data = load_json_if_exists(RIS_RESULTS_JSON)

# ===== Helper: standardize output =====
def standard_output(**kwargs):
    keys = [
        "url", "title", "author", "date", "description", "text",
        "image_urls", "image_captions", "hostname", "sitename", "source"
    ]
    return {k: kwargs.get(k) for k in keys}

import pandas as pd
from urllib.parse import urlparse


def clean_path(p):
    if not isinstance(p, str):
        return p
    p = p.replace('\\', '/').rstrip('/')
    return p


def compute_url_distance(url1,url2,threshold):
    distance = lev.distance(url1,url2)
    if distance < threshold:
        return True
    else:
        return False

from bs4 import BeautifulSoup
from urllib.parse import urlparse
import json
import logging
from newspaper import Article
import trafilatura

# ===== Hàm tìm caption nâng cao =====
def find_image_caption(soup, image_url, threshold=25):
    """
    Retrieve the caption corresponding to an image url by searching the html in BeautifulSoup format.
    Also tries to get captions from meta tags if not found in the HTML body.
    """
    # --- Strategy 1: Search image tag in HTML ---
    img_tag = None
    for img in soup.find_all('img'):
        src = img.get('src') or img.get('data-src') or img.get('data-original')
        if src and compute_url_distance(src, image_url, threshold):
            img_tag = img
            break
    if img_tag:
        # Check figcaption
        figure = img_tag.find_parent('figure')
        if figure:
            figcaption = figure.find('figcaption')
            if figcaption and figcaption.get_text().strip():
                return figcaption.get_text().strip()
        # Check sibling elements
        for sibling in img_tag.find_next_siblings(['div', 'p', 'small']):
            if sibling.get_text().strip():
                return sibling.get_text().strip()
        # Check title/alt attributes
        title = img_tag.get('title')
        if title:
            return title.strip()
        alt_text = img_tag.get('alt')
        if alt_text:
            return alt_text.strip()

    # --- Strategy 2: Meta tags ---
    meta_tags = [
        ('property', 'og:image:alt'),
        ('name', 'twitter:image:alt'),
        ('property', 'og:description'),
        ('name', 'twitter:description'),
        ('name', 'description')
    ]
    for attr, value in meta_tags:
        tag = soup.find('meta', {attr: value})
        if tag and tag.get('content'):
            return tag['content'].strip()

    # --- Strategy 3: No match ---
    if not img_tag:
        return "Image not found"
    return "Caption not found"

# ===== Step 1: Trafilatura extraction =====
def extract_info_trafilatura(url, image_urls=None):
    if image_urls is None:
        image_urls = []
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None

        soup = BeautifulSoup(downloaded, "html.parser")

        result_json = trafilatura.extract(downloaded, output_format="json", with_metadata=True)
        if not result_json:
            return None
        data = json.loads(result_json)
        hostname = urlparse(url).hostname

        if data.get("image") and data["image"] not in image_urls:
            image_urls.append(data["image"])

        image_captions = [find_image_caption(soup, img) for img in image_urls]

        return standard_output(
            url=url,
            title=data.get("title"),
            author=data.get("author"),
            date=data.get("date"),
            description=data.get("description"),
            text=data.get("text"),
            image_urls=image_urls,
            image_captions=image_captions,
            hostname=hostname,
            sitename=None,
            source="trafilatura"
        )
    except Exception as e:
        logging.warning(f"Trafilatura failed for {url}: {e}")
        return None

# ===== Step 2: Newspaper3k extraction =====
def extract_with_newspaper3k(url):
    try:
        article = Article(url)
        article.download()
        article.parse()

        soup = BeautifulSoup(article.html, "html.parser")

        image_urls = list(article.images) if article.images else []
        image_captions = [find_image_caption(soup, img) for img in image_urls]

        hostname = urlparse(url).hostname
        return standard_output(
            url=url,
            title=article.title,
            author=article.authors,
            date=article.publish_date.isoformat() if article.publish_date else None,
            description=article.meta_description,
            text=article.text,
            image_urls=image_urls,
            image_captions=image_captions,
            hostname=hostname,
            sitename=getattr(article, "source_url", None),
            source="newspaper3k"
        )
    except Exception as e:
        logging.warning(f"Newspaper3k failed for {url}: {e}")
        return None

# ===== Step 3: Fallback from pre-crawled JSON =====
def extract_from_pre_crawled(url):
    def find_in(data_list):
        for item in data_list:
            if item.get("url") == url:
                return item
        return None

    t_data = find_in(trafilatura_data)
    r_data = find_in(ris_results_data)
    if not t_data and not r_data:
        return None

    merged = {**(t_data or {}), **(r_data or {})}
    hostname = urlparse(url).hostname

    image_urls = merged.get("image_urls", [])
    if isinstance(image_urls, str):
        image_urls = [u.strip() for u in image_urls.split(";") if u.strip()]
    elif not isinstance(image_urls, list):
        image_urls = []

    # Không có HTML để parse caption -> gán rỗng hoặc dùng caption có sẵn
    image_captions = merged.get("image_captions", []) or ["Caption not found"] * len(image_urls)

    return standard_output(
        url=url,
        title=merged.get("title"),
        author=merged.get("author"),
        date=merged.get("date"),
        description=merged.get("description"),
        text=merged.get("text"),
        image_urls=image_urls,
        image_captions=image_captions,
        hostname=hostname,
        sitename=merged.get("sitename"),
        source="pre_crawled"
    )

# ===== Step 4: Combined extraction with fallback =====
def extract_with_fallback(url, image_urls=None):
    if image_urls is None:
        image_urls = []

    result = extract_info_trafilatura(url, image_urls=image_urls)
    if result:
        return result

    result = extract_with_newspaper3k(url)
    if result:
        return result

    result = extract_from_pre_crawled(url)
    if result:
        return result

    hostname = urlparse(url).hostname
    return {
        "url": url,
        "image_urls": image_urls,
        "image_captions": ["Caption not found"] * len(image_urls),
        "hostname": hostname,
        "error": "Failed to extract content from all methods"
    }



def time_difference(date1, date2):
    '''
    Compute whether date1 preceeds date2
    '''
    # Parse the dates
    dt1 = parser.parse(date1)
    dt2 = parser.parse(date2)
    # Make both dates offset-aware, assuming UTC if no timezone is provided
    if dt1.tzinfo is None:
        dt1 = dt1.replace(tzinfo=tzutc())
    if dt2.tzinfo is None:
        dt2 = dt2.replace(tzinfo=tzutc())
    return dt1 < dt2


import pandas as pd
from urllib.parse import urlparse

def normalize_url(url):
    if not isinstance(url, str):
        return url
    parsed = urlparse(url)
    normalized = parsed.netloc.lower() + parsed.path.rstrip('/')
    return normalized

# def merge_data(evidence, evidence_metadata, dataset, apply_filtering=False):
#     evidence_df = pd.DataFrame(evidence)
#     evidence_metadata_df = pd.DataFrame(evidence_metadata)
#     dataset_df = pd.DataFrame(dataset)
#     print("Columns in evidence_df:", evidence_df.columns)

#     # Chuẩn hóa URL trước khi merge
#     evidence_df['url'] = evidence_df['url'].apply(normalize_url)
#     evidence_metadata_df['raw_url'] = evidence_metadata_df['raw_url'].apply(normalize_url)

#     # Tạo dict url->image_urls
#     url_to_images = dict(zip(evidence_metadata_df['raw_url'], evidence_metadata_df['image_urls']))

#     # Merge evidence và metadata lấy tất cả cột metadata
#     merged_data = pd.merge(
#         evidence_df,
#         evidence_metadata_df.drop_duplicates(subset='raw_url').rename(columns={'raw_url': 'url'}),
#         on='url',
#         how='left'
#     )

#     # Gán lại image_url theo dict url_to_images
#     merged_data['image_url'] = merged_data['url'].map(url_to_images)
#     merged_data['image_url'] = merged_data['image_url'].apply(lambda x: x if isinstance(x, list) else [])

#     # Chuẩn hóa image_path để merge với dataset
#     merged_data['image_file'] = merged_data['image_path'].apply(lambda x: os.path.basename(x) if isinstance(x, str) else x)
#     dataset_df['image_file'] = dataset_df['image_path'].apply(lambda x: os.path.basename(x) if isinstance(x, str) else x)

#     # Merge với dataset lấy publication_date (date_filter)
#     merged_data = pd.merge(
#         merged_data,
#         dataset_df[['image_file', 'publication_date']],
#         on='image_file',
#         how='left'
#     ).rename(columns={'publication_date': 'date_filter'})

#     # Xử lý cột trùng do merge (ví dụ: title_x, title_y)
#     for col in ['title', 'author', 'description', 'sitename', 'date', 'evidence_url']:
#         col_x = f"{col}_x"
#         col_y = f"{col}_y"
#         if col_x in merged_data.columns and col_y in merged_data.columns:
#             merged_data[col] = merged_data[col_x].combine_first(merged_data[col_y])
#             merged_data = merged_data.drop(columns=[col_x, col_y])

#     # Nếu còn cột 'url' thì đổi tên thành 'evidence_url'
#     if 'url' in merged_data.columns:
#         merged_data = merged_data.rename(columns={'url': 'evidence_url'})

#     # Đặt org = hostname
#     if 'hostname' in merged_data.columns:
#         merged_data['org'] = merged_data['hostname']
#     else:
#         merged_data['org'] = None

#     # Nếu cần lọc dữ liệu theo điều kiện thời gian, org, url ...
#     if apply_filtering:
#         def filter_row(row):
#             org = str(row.get('org', '') or '')  # luôn là string
            
#             ev_url = row.get('evidence_url', '')
#             if not isinstance(ev_url, str):
#                 ev_url = ''  # tránh NaN hoặc None

#             img_urls = row.get('image_url', [])
#             if not isinstance(img_urls, list):
#                 img_urls = []
#             else:
#                 img_urls = [str(u) for u in img_urls if isinstance(u, str)]  # chỉ giữ string

#             # điều kiện lọc
#             if org in ev_url or any(org in u for u in img_urls):
#                 return False

#             if pd.isnull(row.get('date')) or pd.isnull(row.get('date_filter')):
#                 return False

#             if not time_difference(row['date'], row['date_filter']):
#                 return False

#             return True

#     # Đảm bảo các cột cần thiết tồn tại
#     required_cols = [
#         'image_path', 'org', 'evidence_url', 'title', 'author', 'hostname',
#         'description', 'sitename', 'date', 'image', 'image_url', 'image_caption'
#     ]
#     for col in required_cols:
#         if col not in merged_data.columns:
#             merged_data[col] = None

#     # Lọc duplicate
#     merged_data = merged_data[required_cols].drop_duplicates(subset=['evidence_url', 'image_path'])

#     return merged_data


def merge_data(evidence, evidence_metadata, dataset, apply_filtering=False):
    """
    Merge evidence + metadata + dataset.
    Đảm bảo an toàn cột, tránh KeyError, và lọc dữ liệu nếu cần.
    """

    print("\n========== [MERGE START] ==========")
    print("[DEBUG] Input evidence size:", len(evidence))
    print("[DEBUG] Input evidence_metadata size:", len(evidence_metadata))
    print("[DEBUG] Input dataset type:", type(dataset), "len:", len(dataset))

    # --- Convert to DataFrame ---
    evidence_df = pd.DataFrame(evidence)
    evidence_metadata_df = pd.DataFrame(evidence_metadata)

    print("\n--- Evidence DF ---")
    print("[DEBUG] evidence_df shape:", evidence_df.shape)
    print("[DEBUG] evidence_df columns:", evidence_df.columns.tolist())
    print("[DEBUG] evidence_df head:", evidence_df.head(2).to_dict(orient="records"))

    print("\n--- Evidence Metadata DF ---")
    print("[DEBUG] evidence_metadata_df shape:", evidence_metadata_df.shape)
    print("[DEBUG] evidence_metadata_df columns:", evidence_metadata_df.columns.tolist())
    print("[DEBUG] evidence_metadata_df head:", evidence_metadata_df.head(2).to_dict(orient="records"))

    try:
        dataset_df = pd.DataFrame(dataset)
        print("\n--- Dataset DF ---")
        print("[DEBUG] dataset_df shape:", dataset_df.shape)
        print("[DEBUG] dataset_df columns:", dataset_df.columns.tolist())
        print("[DEBUG] dataset_df head:", dataset_df.head(2).to_dict(orient="records"))
    except Exception as e:
        print("[ERROR] Failed to create dataset_df:", e)
        return pd.DataFrame()

    # --- Chuẩn hóa URL ---
    if 'url' in evidence_df.columns:
        evidence_df['url'] = evidence_df['url'].apply(normalize_url)
    if 'raw_url' in evidence_metadata_df.columns:
        evidence_metadata_df['raw_url'] = evidence_metadata_df['raw_url'].apply(normalize_url)

    print("\n--- Merge evidence + metadata ---")
    try:
        merged_data = pd.merge(
            evidence_df,
            evidence_metadata_df.drop_duplicates(subset='raw_url').rename(columns={'raw_url': 'url'}),
            on='url',
            how='left'
        )
        print("[DEBUG] merged_data after evidence+metadata:", merged_data.shape)
        print("[DEBUG] merged_data columns:", merged_data.columns.tolist())
        print("[DEBUG] merged_data sample:", merged_data.head(2).to_dict(orient="records"))
    except Exception as e:
        print("[ERROR] Failed to merge evidence + metadata:", e)
        return pd.DataFrame()

    # Rename url -> evidence_url
    if 'url' in merged_data.columns:
        merged_data = merged_data.rename(columns={'url': 'evidence_url'})

    # --- Chuẩn hóa path để merge với dataset ---
    if 'image_path' not in merged_data.columns:
        print("[WARNING] 'image_path' not in merged_data → sẽ thêm cột None")
        merged_data['image_path'] = None

    merged_data['image_file'] = merged_data['image_path'].apply(
        lambda x: os.path.basename(x) if isinstance(x, str) else x
    )
    dataset_df['image_file'] = dataset_df['image_path'].apply(
        lambda x: os.path.basename(x) if isinstance(x, str) else x
    )

    print("\n--- Merge với dataset ---")
    try:
        merged_data = pd.merge(
            merged_data,
            dataset_df[['org', 'image_file', 'publication_date']].rename(columns={'publication_date': 'date_filter'}),
            on='image_file',
            how='left'
        )
        print("[DEBUG] merged_data after merge with dataset:", merged_data.shape)
        print("[DEBUG] merged_data sample:", merged_data.head(2).to_dict(orient="records"))
    except Exception as e:
        print("[ERROR] Failed to merge with dataset:", e)
        return pd.DataFrame()

    # --- Đặt org fallback nếu thiếu ---
    if 'hostname' in merged_data.columns and 'org' not in merged_data.columns:
        merged_data['org'] = merged_data['hostname']

    # --- Đảm bảo các cột cần thiết tồn tại ---
    required_cols = [
        'image_path','org','evidence_url','title','author','hostname',
        'description','sitename','date','image','image_url','image_caption','date_filter'
    ]
    for col in required_cols:
        if col not in merged_data.columns:
            print(f"[DEBUG] Missing col {col} → fill None")
            merged_data[col] = None

    # --- Filtering ---
    if apply_filtering:
        print("\n--- Filtering rows ---")
        before = len(merged_data)

        def valid_row(row):
            org = str(row.get('org', '') or '')
            ev_url = str(row.get('evidence_url', '') or '')
            img_urls = row.get('image_url', [])
            if not isinstance(img_urls, list): 
                img_urls = []

            if org in ev_url or any(org in u for u in img_urls):
                print(f"[FILTER-OUT] Same org ({org}) in URL/images: {ev_url}")
                return False
            if pd.isnull(row.get('date')) or pd.isnull(row.get('date_filter')):
                print(f"[FILTER-OUT] Missing date/date_filter for {ev_url}")
                return False
            if not time_difference(row['date'], row['date_filter']):
                print(f"[FILTER-OUT] Evidence after FC article: {row['date']} vs {row['date_filter']}")
                return False
            return True

        merged_data = merged_data[merged_data.apply(valid_row, axis=1)]
        after = len(merged_data)
        print(f"[DEBUG] Filtering reduced {before} → {after} rows")

    # --- Drop duplicate ---
    merged_data = merged_data[required_cols].drop_duplicates(subset=['evidence_url','image_path'])
    print("\n========== [MERGE END] Final shape:", merged_data.shape, "==========")
    print("[DEBUG] Final sample:", merged_data.head(2).to_dict(orient="records"))

    return merged_data



def download_image_as_pil(url: str, timeout: int = 10) -> Image.Image | None:
    """
    Download image and return as PIL.Image.
    Return None if invalid content or error.
    """
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=timeout)
        if response.status_code == 200 and 'image' in response.headers.get('Content-Type', ''):
            return Image.open(BytesIO(response.content)).convert('RGB')
    except Exception as e:
        print(f"[download_image_as_pil] Failed to download {url} | {e}")
    return None


def download_image(url, file_path, max_size_mb=10):
    '''
    Download evidence images. Only used for images predicted as manipulated to replace them by their predicted original version.
    '''
    try:
        # Send a GET request to the URL
        headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        response = rq.get(url, stream=True, timeout=(10,10),headers=headers)
        # Check if the request was successful
        if response.status_code != 200:
            print(f"Failed to download. Status code: {response.status_code}")
            return None
        # Check the content type to be an image
        if 'image' not in response.headers.get('Content-Type', ''):
            print("URL does not point to an image.")
            return None
        # Check the size of the image
        if int(response.headers.get('Content-Length', 0)) > max_size_mb * 1024 * 1024:
            print(f"Image is larger than {max_size_mb} MB.")
            return None
        # Read the image content
        image_data = response.content
        if not image_data:
            print("No image data received.")
            return None
        image = Image.open(BytesIO(image_data))
        image.verify()
        image = Image.open(BytesIO(image_data))
        # Save the image to a file
        image.save(file_path + '.png')
        print("Image downloaded and saved successfully.")
    except rq.RequestException as e:
        print(f"Request failed: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


def keep_longest_non_greyscale_area(image, color_threshold=30):
    '''
    Automatically remove social media sidebars for screenshots coming from social media platforms.
    '''
    def find_longest_sequence(arr, color_threshold):
        # Calculate the range (max-min) within each row for each color channel
        color_ranges = np.ptp(arr, axis=1)
        # Determine rows that have enough color variation to not be considered greyscale
        is_colorful = np.any(color_ranges > color_threshold, axis=1)
        # Identify transitions between greyscale and non-greyscale rows
        transitions = np.diff(is_colorful.astype(int))
        start_indices = np.where(transitions == 1)[0] + 1  # Start of colorful sequence
        end_indices = np.where(transitions == -1)[0]  # End of colorful sequence
        # Handle cases where the sequence starts from the first row or ends at the last row
        if len(start_indices) == 0 or (len(end_indices) > 0 and start_indices[0] > end_indices[0]):
            start_indices = np.insert(start_indices, 0, 0)
        if len(end_indices) == 0 or (len(start_indices) > 0 and end_indices[-1] < start_indices[-1]):
            end_indices = np.append(end_indices, arr.shape[0] - 1)
        # Find the longest sequence of colorful rows
        max_length = 0
        max_seq_start = max_seq_end = 0
        for start, end in zip(start_indices, end_indices):
            if end - start > max_length:
                max_length = end - start
                max_seq_start, max_seq_end = start, end
        # Keep only the longest colorful sequence
        return arr[max_seq_start:max_seq_end]

    # Convert to numpy array for analysis
    image_array = np.array(image)

    # Keep the longest sequence of non-greyscale rows
    longest_row_sequence = find_longest_sequence(image_array, color_threshold)

    # Transpose the array to treat columns as rows and repeat the process
    transposed_array = np.transpose(longest_row_sequence, (1, 0, 2))
    longest_col_sequence = find_longest_sequence(transposed_array, color_threshold)

    # Transpose back to original orientation
    final_array = np.transpose(longest_col_sequence, (1, 0, 2))

    # Convert the array back to an image
    longest_sequence_image = Image.fromarray(final_array)
    return longest_sequence_image


def apply_instructions(image, instructions):
    '''
    Applies processing instructions to the given image.
    Instructions include standard processing, cropping, or downloading a new image and then cropping.
    '''
    if instructions.startswith("Standard processing"):
        return keep_longest_non_greyscale_area(image)
    elif instructions.startswith("Cropped"):
        crop_coords = eval(instructions.split(": ")[1])
        return image.crop(crop_coords)
    elif instructions.startswith("Replaced with URL"):
        parts = instructions.split("; ")
        url = parts[0].split(": ")[1]
        new_image = download_image(url)
        if len(parts) > 1 and parts[1].startswith("Standard processing"):
            return keep_longest_non_greyscale_area(image)
        if len(parts) > 1 and parts[1].startswith("Cropped"):
            crop_coords = eval(parts[1].split(": ")[1])
            return new_image.crop(crop_coords)
        return new_image
    else:
        return image


def process_images_from_instructions(instructions_file, source_folder, target_folder):
    '''
    Loads processing instructions from a file and applies them to each corresponding image.
    '''
    processed_info = []
    with open(instructions_file, 'r') as file:
        for line in file:
            image_name, instructions = line.strip().split(": ", 1)
            image_path = os.path.join(source_folder, image_name)
            # Check if the image exists before processing
            if not os.path.exists(image_path):
                continue
            image = Image.open(image_path)
            processed_image = apply_instructions(image, instructions)
            # Save the processed image to the target folder
            target_image_path = os.path.join(target_folder, image_name)
            processed_image.save(target_image_path)
            print(f"Processed and saved: {image_name}")
            processed_info.append({
                "image": image_name,
                "instruction": instructions,
                "output_path": target_image_path
            })
