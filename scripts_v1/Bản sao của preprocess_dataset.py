import os
import sys
import time
import json
import pandas as pd
import argparse
import stanza
import spacy_stanza
from deep_translator import GoogleTranslator
from dateutil import parser as date_parser
import re
from urllib.parse import urlparse
import fnmatch

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from dataset_collection.preprocessing_utils import *
from evaluation.geonames_collection import *
from utils import *

def translate_to_en(text):
    try:
        return GoogleTranslator(source='vi', target='en').translate(text)
    except Exception as e:
        print(f"[⚠️] Dịch thất bại '{text}': {e}")
        return text

def extract_image_info_from_article(json_data, article_files_map):
    url = json_data.get('URL', '').strip()
    if not url:
        return 'not enough information', 'not enough information'

    # Bước 1: Trích xuất ID từ URL (phần cuối sau dấu '-')
    url_parts = urlparse(url).path.strip('/').split('/')
    article_id = url_parts[-1].split('-')[-1].lower()
    
    # Bước 2: Tìm file article tương ứng
    matched_article = None
    for file_path in article_files_map:
        file_name = os.path.basename(file_path).lower()
        if article_id in file_name.replace('.txt', ''):
            matched_article = file_path
            break

    if not matched_article:
        return 'not enough information', 'not enough information'

    # Bước 3: Trích xuất URL ảnh từ article
    image_url = 'not enough information'
    try:
        with open(matched_article, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Tìm URL ảnh với nhiều pattern khác nhau
        url_patterns = [
            r'Đường dẫn hình ảnh:\s*(http[s]?://[^\s]+)',
            r'Image URL:\s*(http[s]?://[^\s]+)',
            r'(http[s]?://[^\s]+\.(?:jpg|jpeg|png|gif))'
        ]
        
        for pattern in url_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                image_url = match.group(1).strip()
                break
    except Exception as e:
        print(f"Error reading article file: {e}")
        return 'not enough information', 'not enough information'

    # Bước 4: Tìm file ảnh trong thư mục dataset/img
    image_path = 'not enough information'
    img_folder = 'dataset/img/'
    
    # Cách 1: Tìm bằng ID bài báo (phần cuối URL)
    possible_image_names = [
        f"vietfact_{url_parts[-1].lower()}.txt.png",
        f"{url_parts[-1].lower()}.txt.png",
        f"*{article_id}*.png"
    ]
    
    for pattern in possible_image_names:
        for root, _, files in os.walk(img_folder):
            for file in files:
                if fnmatch.fnmatch(file.lower(), pattern.lower()):
                    image_path = os.path.join(root, file)
                    return image_url, image_path

    # Cách 2: Nếu có URL ảnh, tìm bằng tên file ảnh
    if image_url != 'not enough information':
        image_filename = os.path.basename(image_url)
        for root, _, files in os.walk(img_folder):
            if image_filename.lower() in [f.lower() for f in files]:
                image_path = os.path.join(root, image_filename)
                break

    return image_url, image_path

def map_image_type_to_label(image_type):
    t = image_type.lower().strip()
    if t == 'non-manipulated':
        return 'non-manipulated'
    else:
        return 'manipulated'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess the raw data to create train, val, and test sets.')
    parser.add_argument('--json_file_path', type=str, default='dataset/gpt4_annotations/annotations.json')
    parser.add_argument('--geonames_username', type=str, default="chuong")
    parser.add_argument('--sleep_geonames', type=int, default=2)
    parser.add_argument('--geonames_data', type=str, default='dataset/geonames_results.json')
    args = parser.parse_args()

    stanza.download('vi')
    nlp = spacy_stanza.load_pipeline('vi')

    raw_data = load_json(args.json_file_path)
    print(f"[📊] Tổng số bài báo ban đầu: {len(raw_data)}")

    normalized_data = pd.DataFrame([normalize_json_fields(d) for d in raw_data])
    print(f"[✅] Đã chuẩn hoá xong dữ liệu: {normalized_data.shape[0]} dòng")

    def format_iso_z(date_str):
        try:
            return date_parser.parse(date_str).replace(tzinfo=None).isoformat() + 'Z'
        except:
            return 'not enough information'

    normalized_data['publication_date'] = normalized_data['publication_date'].apply(format_iso_z)
    normalized_data = normalized_data.sort_values(by='publication_date')
    normalized_data = normalized_data.fillna('not enough information').to_dict(orient='records')

    print(f"[🧹] Dữ liệu đã được sắp xếp theo ngày và thay thế missing")

    all_locs = set()
    translated_locs = 0
    geonames_found = 0
    geonames_not_found = 0
    os.makedirs(os.path.dirname(args.geonames_data), exist_ok=True)
    
    for d in normalized_data:
        location_text = d['location']
        if location_text.lower() != "not enough information":
            if location_text not in all_locs:
                location_en = translate_to_en(location_text)
                translated_locs += 1
                results = search_location(location_en, args.geonames_username, args.sleep_geonames)
                time.sleep(args.sleep_geonames)
                if results:
                    geonames_found += 1
                    save_result(results, args.geonames_data)
                else:
                    geonames_not_found += 1
                all_locs.add(location_text)

    print(f"\n[🌐] Tổng địa điểm duy nhất: {len(all_locs)}")
    print(f"[🌍] Đã dịch: {translated_locs}")
    print(f"[✅] Có GeoNames: {geonames_found}")
    print(f"[❌] Không có GeoNames: {geonames_not_found}")

    article_folder = 'dataset/article/'
    article_files = [os.path.join(article_folder, f) for f in os.listdir(article_folder) if f.endswith('.txt')]

    final_data = []
    count_url = 0
    count_local_image = 0
    label_counts = {'manipulated': 0, 'non-manipulated': 0, 'not enough information': 0}

    for i, d in enumerate(normalized_data):
        d['claimed_location'] = raw_data[i].get('claimed location', raw_data[i].get('real location', 'not enough information'))
        d['claimed_date'] = raw_data[i].get('claimed date', raw_data[i].get('real date', 'not enough information'))
        
        image_url, image_path = extract_image_info_from_article(raw_data[i], article_files)
        
        # Thay đổi image_path nếu hợp lệ
        if image_path != 'not enough information':
            image_path = image_path.replace('dataset/img/', 'dataset/processed_img_vie/')
        
        d['image_URL'] = image_url
        d['image_path'] = image_path
        
        if d['image_URL'] != 'not enough information':
            count_url += 1
        if d['image_path'] != 'not enough information':
            count_local_image += 1
            
        image_type = raw_data[i].get('type of image') or raw_data[i].get('type_of_image', 'not enough information')
        d['label'] = map_image_type_to_label(image_type)
        d['type_of_image'] = image_type 
        label_counts[d['label']] += 1

        if d['label'] != 'not enough information':
            final_data.append(d)

    print(f"\n[🖼️] Số bài báo có đường dẫn hình ảnh: {count_url}")
    print(f"[📁] Số bài báo có ảnh trong local: {count_local_image}")
    print(f"[🏷️] Số ảnh theo nhãn:")
    for label, count in label_counts.items():
        print(f"    - {label}: {count}")

    train_idx = int(len(final_data) * 0.60)
    val_idx = int(len(final_data) * 0.70)

    train = final_data[:train_idx]
    val = final_data[train_idx:val_idx]
    test = final_data[val_idx:]

    print(f"\n[📂] Tổng ảnh hợp lệ: {len(final_data)}")
    print(f"[📁] Train set: {len(train)}")
    print(f"[📁] Val set: {len(val)}")
    print(f"[📁] Test set: {len(test)}")

    with open('dataset/train_custom.json', 'w', encoding='utf-8') as file:
        json.dump(train, file, indent=4, ensure_ascii=False)
    with open('dataset/val_custom.json', 'w', encoding='utf-8') as file:
        json.dump(val, file, indent=4, ensure_ascii=False)
    with open('dataset/test_custom.json', 'w', encoding='utf-8') as file:
        json.dump(test, file, indent=4, ensure_ascii=False)

    print("\n✅ [HOÀN TẤT] Đã chia và lưu tập dữ liệu.")
