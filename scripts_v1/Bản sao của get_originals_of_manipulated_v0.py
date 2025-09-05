
# scripts/get_originals_of_manipulated_v1.py
import argparse
import os
import sys
import pandas as pd
import json

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils import *
from dataset_collection.scrape_utils import *

def normalize_path(p: str) -> str:
    return os.path.basename(p.strip().lower())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Identify the original image among the RIS results.')
    parser.add_argument('--json_path', type=str, default='dataset/manipulation_detection_test.json',
                        help='Path to the manipulation detection predictions')
    parser.add_argument('--download_image', type=int, default=0,
                        help='If True, download RIS images for manipulated images.')
    parser.add_argument('--map_json_path', type=str, default='dataset/map_manipulated_original.json',
                        help='Output JSON mapping manipulated to original images.')
    args = parser.parse_args()

    os.makedirs('dataset/manipulated_original_img', exist_ok=True)

    test = load_json('dataset/test_custom.json')
    evidence = load_json('dataset/retrieval_results/evidence.json')

    manipulated_paths = [im['image_path'] for im in load_json(args.json_path)
                         if im['manipulation_detection'] == 'manipulated']

    print(f"[INFO] Total manipulated images: {len(manipulated_paths)}")

    if args.download_image:
        image_to_download = []

        for i, ev in enumerate(evidence):
            if normalize_path(ev['image_path']) in [normalize_path(p) for p in manipulated_paths]:
                url = ev.get('image_url', '').split(';')[0]
                if url:
                    image_to_download.append((url, i))

        print(f"[INFO] Downloading {len(image_to_download)} images...")
        for url, idx in image_to_download:
            save_path = f'dataset/manipulated_original_img/{idx}'  # no extension
            print(f"[DOWNLOAD] {url} -> {save_path}")
            download_image(url, save_path)

    dict_original_image = {}
    for img_path in manipulated_paths:
        print(f"\n[CHECK] Manipulated image: {img_path}")
        subset = [ev for ev in evidence if normalize_path(ev['image_path']) == normalize_path(img_path)]
        subset_index = [i for i, ev in enumerate(evidence) if normalize_path(ev['image_path']) == normalize_path(img_path)]

        print(f"[INFO] Found {len(subset)} evidence matches")

        if not subset:
            print(f"[WARN] No evidence found for: {img_path}")
            continue

        df = pd.DataFrame(subset)
        if 'date' not in df.columns:
            print(f"[WARN] No 'date' field for evidence of: {img_path}")
            continue

        sorted_df = df.sort_values(by='date')
        for i, (_, row) in enumerate(sorted_df.iterrows()):
            if i >= len(subset_index):
                continue
            idx = subset_index[i]
            image_file = f'{idx}.png.png'  # match download_image output
            image_path = os.path.join('dataset/manipulated_original_img', image_file)
            print(f"[DEBUG] Checking if image exists: {image_path}")
            if os.path.exists(image_path):
                dict_original_image[img_path] = image_path
                print(f"[OK] Mapped: {img_path} -> {image_path}")
                break
            else:
                print(f"[MISSING] File not found: {image_path}")

    print(f"\n[INFO] Total mapped manipulated images: {len(dict_original_image)}")

    with open(args.map_json_path, 'w') as f:
        json.dump(dict_original_image, f, indent=4)

    print(f"[INFO] Mapping saved to {args.map_json_path}")
