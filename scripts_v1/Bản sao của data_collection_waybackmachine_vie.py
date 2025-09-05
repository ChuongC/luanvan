
import os
import sys
import requests as rq
import time  


import json
from tqdm import tqdm
import os
import argparse
import re
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils import *
from dataset_collection.scrape_utils import *
import requests

def collect_data_wayback(domain_url,
                         output_file,
                         start_date,
                         end_date,
                         max_urls=1000,
                         chunk_size=100,
                         sleep=10):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    collected_urls = set()
    offset = 0
    headers = {'User-Agent': 'Mozilla/5.0'}
    no_progress_streak = 0  # Đếm số lần liên tiếp không thu thập được thêm URL mới

    print(f"✨ Đang thu thập từ: {domain_url} (từ {start_date} đến {end_date})")

    while len(collected_urls) < max_urls:
        url = (
            f"https://web.archive.org/cdx/search/cdx"
            f"?url={domain_url}*"
            f"&from={start_date}&to={end_date}"
            f"&output=json&fl=original&collapse=urlkey"
            f"&filter=statuscode:200&limit={chunk_size}&offset={offset}"
        )

        print(f"🔍 Đang truy vấn offset {offset}...")
        try:
            res = requests.get(url, headers=headers, timeout=60)
            if res.status_code != 200 or not res.text.strip().startswith('['):
                print(f"⚠️ Phản hồi không hợp lệ (status: {res.status_code})")
                break
        except Exception as e:
            print(f"❌ Lỗi truy vấn offset {offset}: {e}")
            break

        try:
            data = res.json()
            urls_fetched = 0

            if len(data) <= 1:
                print("⛔️ Không còn dữ liệu để thu thập.")
                break

            for row in data[1:]:
                if isinstance(row, list) and len(row) > 0:
                    url_candidate = row[0]
                    if not any(x in url_candidate for x in ['/feed/', '/amp', '?__twitter_']):
                        if url_candidate not in collected_urls:
                            collected_urls.add(url_candidate)
                            urls_fetched += 1

            print(f"✔️ Offset {offset} nhận được {urls_fetched} URL mới")

            # Nếu số lượng URL mới thu được < chunk_size => gần hết rồi, dừng lại
            if urls_fetched < chunk_size:
                no_progress_streak += 1
            else:
                no_progress_streak = 0

            if no_progress_streak >= 3:
                print("📉 Không có thêm URL mới trong 3 lượt liên tiếp, kết thúc.")
                break

            offset += chunk_size
            time.sleep(sleep)

        except Exception as e:
            print(f"❌ Lỗi khi xử lý dữ liệu offset {offset}: {e}")
            break

    print(f"\n✅ Đã thu thập được {len(collected_urls)} URL duy nhất.")
    with open(output_file, 'w', encoding='utf-8') as f:
        for u in sorted(collected_urls):
            f.write(u + '\n')

def extract_basename(url):
    """
    Tạo tên ngắn gọn từ URL để dùng làm tên file lưu bài viết.
    """
    basename = url.strip().rstrip('/').split('/')[-1]
    # Nếu basename trống, lấy tên theo hash
    if not basename:
        import hashlib
        return hashlib.md5(url.encode('utf-8')).hexdigest()
    return basename

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download articles and images from the Wayback machine.')
    parser.add_argument('--url_domain', type=str, default='vietfactcheck.org/', help='The domain to query.')
    parser.add_argument('--org', type=str, default='vietfactcheck', help='FC organization')
    parser.add_argument('--file_path', type=str, default='dataset/url/vietfactcheck.txt')
    parser.add_argument('--only_vietnamese', type=int, default=1, help='Chỉ thu thập bài viết tiếng Việt')
    parser.add_argument('--scrape_image', type=int, default=1)
    parser.add_argument('--process_image', type=int, default=1)
    parser.add_argument('--image_processing_script', type=str, default='dataset/image_processing_instructions.txt')
    parser.add_argument('--start_date', type=int, default=20200801)
    parser.add_argument('--end_date', type=int, default=20250801)
    parser.add_argument('--max_count', type=int, default=3000)
    parser.add_argument('--chunk_size', type=int, default=600)
    parser.add_argument('--sleep', type=int, default=10)

    args = parser.parse_args()

    collect_data_wayback(args.url_domain,
                         args.file_path,
                         start_date=args.start_date,
                         end_date=args.end_date,
                         max_urls=args.max_count,
                         chunk_size=args.chunk_size,
                         sleep=args.sleep)

    urls_all = load_urls(args.file_path)
    print(f"\U0001F4DD Tổng URL thu thập được: {len(urls_all)}")

    filtered_urls = [u for u in urls_all if is_valid_article_url(u)]
    print(f"\U0001F9F9 Đã lọc được {len(filtered_urls)} bài viết hợp lệ")
    # Các bài viết đã có nội dung (tránh scrape lại)
    existing_articles = {
        os.path.splitext(f)[0]
        for f in os.listdir('dataset/article/')
        if f.endswith('.txt')
    }

    # Giữ lại bài viết chưa scrape
    filtered_urls = [
        u for u in filtered_urls
        if extract_basename(u) not in existing_articles
    ]

    print(f"📂 Đã bỏ qua {len(urls_all) - len(filtered_urls)} bài viết đã thu thập.")

    urls_with_images = []
    failed_urls = []
    fail_reasons = {}
    success_count = 0

    for url in tqdm(filtered_urls):
        try:
            content, image_urls, basename = vietfactcheck_parser(url, only_vietnamese=bool(args.only_vietnamese))
            if not image_urls:
                failed_urls.append(url)
                fail_reasons[url] = 'no_image_found'
                continue
            urls_with_images.append((url, content, image_urls[:1], basename))
        except Exception as e:
            failed_urls.append(url)
            fail_reasons[url] = str(e)

    print(f"\n✅ Tổng bài viết có ảnh: {len(urls_with_images)}")

    os.makedirs('dataset/article/', exist_ok=True)

    for i, (url, content, image_urls, basename) in enumerate(urls_with_images):
        file_name = f"vietfact_{basename}.txt"
        success = False

        if args.scrape_image:
            for j, img_url in enumerate(image_urls):
                success = scrape_image(img_url, basename)

        if success:
            with open(f"dataset/article/{file_name}", 'w', encoding='utf-8') as f:
                f.write(content)
            success_count += 1
        else:
            failed_urls.append(url)
            fail_reasons[url] = 'image_download_failed'

    # In thống kê cho báo cáo
    print("\n📊 Thống kê quá trình thu thập:")
    print(f"- Tổng URL ban đầu: {len(urls_all)}")
    print(f"- Bài viết hợp lệ sau lọc: {len(filtered_urls) + len(existing_articles)}")
    print(f"- Bài viết đã tồn tại (bỏ qua): {len(existing_articles)}")
    print(f"- Bài viết mới cần xử lý: {len(filtered_urls)}")
    print(f"- Thành công (có nội dung + ảnh): {success_count}")
    print(f"- Thất bại: {len(failed_urls)}")

    from collections import Counter
    reason_stats = Counter(fail_reasons.values())
    print("\n🧾 Thống kê lỗi:")
    for reason, count in reason_stats.items():
        print(f"- {reason}: {count} trường hợp")
