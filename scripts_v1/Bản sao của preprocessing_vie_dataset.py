import os
import sys
import time
import json
import pandas as pd
import argparse

import stanza
import spacy_stanza

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from dataset_collection.preprocessing_utils import *
from evaluation.geonames_collection import *
from utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess the raw data to create a train, val, and test sets.')
    parser.add_argument('--json_file_path', type=str, default='dataset/gpt4_annotations/annotations.json',
                        help='Path to the GPT4 annotations')
    parser.add_argument('--geonames_username', type=str, default="chuong",
                        help='Username to access GeoNames API.')
    parser.add_argument('--sleep_geonames', type=int, default=2,
                        help='Waiting time between two API calls of the GeoNames API.')
    parser.add_argument('--geonames_data', type=str, default='dataset/geonames_results.json',
                        help='File to store the geonames results.')

    args = parser.parse_args()

    # Download and load Stanza pipeline for Vietnamese
    stanza.download('vi')
    nlp = spacy_stanza.load_pipeline('vi')

    raw_data = load_json(args.json_file_path)

    normalized_data = pd.DataFrame([normalize_json_fields(d) for d in raw_data])
    duplicates = get_duplicates(normalized_data['image_path'].to_list())
    duplicates_mask = normalized_data['image_path'].apply(lambda row : False if row in duplicates else True)
    normalized_data = normalized_data[duplicates_mask]

    annotation_count = normalized_data[['provenance','source','location', 'date', 'motivation']]
    for c in annotation_count.columns:
        annotation_count.loc[:,c] = annotation_count[c].apply(lambda row : row if str(row).lower() != 'not enough information' else None)
    null_rows = annotation_count.isnull().all(axis=1)
    normalized_data = normalized_data.drop(normalized_data.index[null_rows])

    normalized_data = normalized_data.sort_values(by='publication_date')
    normalized_data = normalized_data.fillna('not enough information').to_dict(orient='records')

    all_locs = []
    print(f"[INFO] Total normalized entries: {len(normalized_data)}")

    for d in normalized_data:
        print(f"[📍] Location text: {d['location']}")
        locations = extract_named_entities(d['location'], nlp, 'locations')
        print(f"[📦] Named entities extracted: {locations}")

        for l in locations:
            if l not in all_locs:
                results = search_location(l, args.geonames_username, args.sleep_geonames)
                time.sleep(args.sleep_geonames)
                print(f"[GeoNames] Saving {len(results)} entries for: {l}")
                save_result(results, args.geonames_data)
                all_locs += locations

    # Normalize image_URL path to local path
    for d in normalized_data:
        if 'image_URL' in d and isinstance(d['image_URL'], str):
            filename = os.path.basename(d['image_URL'])
            d['image_URL'] = f"dataset/img/{filename}"

    # Split data
    train_idx = int(len(normalized_data) * 0.60)
    val_idx = int(len(normalized_data) * 0.70)

    train = normalized_data[:train_idx]
    val = normalized_data[train_idx:val_idx]
    test = normalized_data[val_idx:]

    # Save as UTF-8 JSON with proper Vietnamese rendering
    with open('dataset/train_custom.json', 'w', encoding='utf-8') as file:
        json.dump(train, file, indent=4, ensure_ascii=False)
    with open('dataset/val_custom.json', 'w', encoding='utf-8') as file:
        json.dump(val, file, indent=4, ensure_ascii=False)
    with open('dataset/test_custom.json', 'w', encoding='utf-8') as file:
        json.dump(test, file, indent=4, ensure_ascii=False)
