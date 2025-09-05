import argparse
import os
import sys
import json
import pandas as pd
import requests

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils import *
from dataset_collection.scrape_utils import *

def get_filename(path):
    return os.path.basename(path)

def try_download_multiple_versions(urls, save_path):
    for url in urls:
        try:
            print(f"[DOWNLOAD] Trying URL: {url}")
            response = requests.get(url, timeout=10)
            if response.status_code == 200 and 'image' in response.headers.get('Content-Type', ''):
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                print(f"[SUCCESS] Image saved to {save_path}")
                return True
            else:
                print(f"[SKIPPED] URL not valid image: {url}")
        except Exception as e:
            print(f"[FAILED] Error downloading {url}: {e}")
    return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Heuristic to identify the unaltered, original image among the RIS results.')
    parser.add_argument('--json_path', type=str, default='dataset/manipulation_detection_test.json',
                        help='Path to the manipulation detection predictions')
    parser.add_argument('--download_image', type=int, default=0,
                        help='If True, download the images retrieved by RIS for images predicted as manipulated.')
    parser.add_argument('--map_json_path', type=str, default='dataset/map_manipulated_original.json',
                        help='Path to store the file that maps manipulated images to their identified original version.')
    args = parser.parse_args()

    os.makedirs('dataset/manipulated_original_img/', exist_ok=True)

    print("[LOAD] Loading JSON data...")
    test = load_json('dataset/test.json')
    #test = load_json('dataset/train_sample.json')
    evidence = load_json('dataset/retrieval_results/evidence.json')

    manipulated = [im['image_path'] for im in load_json(args.json_path) if im['manipulation_detection'] == 'manipulated']
    manipulated_filenames = [get_filename(p) for p in manipulated]

    print(f"[INFO] Detected manipulated images: {len(manipulated)}")
    for m in manipulated_filenames:
        print(f"    - {m}")

    if args.download_image:
        print(f"[INFO] Starting download of original images...")
        for ev in evidence:
            filename = get_filename(ev['image_path'])
            if filename in manipulated_filenames:
                urls_raw = ev['image_url']
                if isinstance(urls_raw, str):
                    urls = [u.strip() for u in urls_raw.split(',') if u.strip().startswith('http')]
                elif isinstance(urls_raw, list):
                    urls = urls_raw
                else:
                    urls = []
                save_path = f'dataset/manipulated_original_img/{filename}'
                try_download_multiple_versions(urls, save_path)

    dict_original_image = {}
    failed_reason = {}
    print(f"[PROCESS] Mapping manipulated images to original candidates...")
    for img_path in manipulated:
        filename = get_filename(img_path)
        print(f"\n[PROCESSING] {filename}")
        subset = [ev for ev in evidence if get_filename(ev['image_path']) == filename]
        subset_index = [i for i, ev in enumerate(evidence) if get_filename(ev['image_path']) == filename]

        if not subset:
            print("  [SKIP] No evidence found.")
            failed_reason[img_path] = 'No matching evidence found'
            continue

        try:
            sorted_df = pd.DataFrame(subset)
            sorted_df['date'] = pd.to_datetime(sorted_df['evidence_date'], errors='coerce')
            sorted_evidence_by_date = sorted_df.sort_values(by='date')
            if sorted_evidence_by_date['date'].isnull().all():
                print("  [FAIL] All evidence dates are invalid.")
                failed_reason[img_path] = 'All dates are invalid or missing'
                continue
            idx = sorted_df.index.get_loc(sorted_evidence_by_date.index[0])
            image_filename = get_filename(subset[idx]['image_path'])
            image_path = f"dataset/manipulated_original_img/{image_filename}"
            if image_filename in os.listdir('dataset/manipulated_original_img/'):
                print(f"  [OK] Mapped to original: {image_path}")
                dict_original_image[img_path] = f"dataset/manipulated_original_img/{image_filename}"
            else:
                print(f"  [FAIL] Original image not downloaded: {image_path}")
                failed_reason[img_path] = 'Original image file not downloaded'
        except Exception as e:
            print(f"  [ERROR] Unexpected error: {str(e)}")
            failed_reason[img_path] = f'Error during processing: {str(e)}'

    with open(args.map_json_path, 'w') as file:
        json.dump(dict_original_image, file, indent=4)

    print(f"\n[SUMMARY] Mapping complete.")
    print(f"  Total manipulated images: {len(manipulated)}")
    print(f"  Successfully mapped     : {len(dict_original_image)}")
    print(f"  Failed to map           : {len(failed_reason)}")

    print("\n[FAILED CASES]")
    reason_stats = {}
    for k, v in failed_reason.items():
        print(f" - {get_filename(k)}: {v}")
        reason_stats[v] = reason_stats.get(v, 0) + 1

    print("\n[FAILURE BREAKDOWN]")
    for reason, count in reason_stats.items():
        print(f"  {reason}: {count} images")
