
import os
import time
import argparse
import json
import sys
import re
from tqdm import tqdm
from urllib.parse import urlparse
from dateutil import parser
import openai
from dotenv import load_dotenv

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils import *

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

openai.api_key = os.getenv("OPENAI_API_KEY")

def normalize_date(value: str) -> str:
    try:
        return parser.parse(value, dayfirst=True).strftime("%Y-%m-%dT00:00:00+00:00")
    except:
        return value

def extract_org_from_url(url: str) -> str:
    try:
        domain = urlparse(url).netloc
        return domain.replace("www.", "").split('.')[0]
    except Exception:
        return "Not Enough Information"

def extract_publication_date_from_url(url: str) -> str:
    try:
        date_match = re.search(r'(\d{4})(\d{2})(\d{2})', url)
        if date_match:
            y, m, d = date_match.groups()
            return f"{y}-{m}-{d}T00:00:00+00:00"
    except Exception:
        pass
    return "Not Enough Information"

def validate_json_output(output: str, min_filled_ratio: float = 0.8) -> bool:
    try:
        obj = json.loads(output) if isinstance(output, str) else output
        required_keys = [
            "URL", "image_path", "org", "publication_date", "claim", "provenance",
            "source", "date", "date_numeric_label", "location", "motivation",
            "type_of_image", "verification_strategy", "verification_tool",
            "claimed_location", "claimed_date", "image_URL"
        ]
        filled_keys = [k for k in required_keys if k in obj and str(obj[k]).strip() != ""]
        if len(filled_keys) / len(required_keys) >= min_filled_ratio:
            return True
        else:
            missing = [k for k in required_keys if k not in filled_keys]
            print(f"[WARN] Missing fields: {missing}")
            return False
    except Exception:
        return False

def label_corpus(corpus, system_prompt_path, json_path, sleep=20):
    with open(system_prompt_path, encoding='utf-8') as f:
        system_prompt = f.read()

    for filename, text, image_path_list, image_url_list in tqdm(corpus):
        for image_path, image_url in zip(image_path_list, image_url_list):
            try:
                url = filename.replace('.txt', '')
                user_prompt = (
                    f"URL: {url}\n\n"
                    f"Article:\n{text}\n\n"
                    f"Local Image Path: {image_path}\n"
                    f"Image URL: {image_url}"
                )

                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini-2024-07-18",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=1500,
                    temperature=0.2
                )

                output = response.choices[0].message.content
                if isinstance(output, str):
                    try:
                        json_obj = json.loads(output)

                        for k in ["publication_date", "claimed_date", "date"]:
                            if k in json_obj:
                                json_obj[k] = normalize_date(json_obj[k])

                        if json_obj.get("date") != "Not Enough Information":
                            json_obj["date_numeric_label"] = [normalize_date(json_obj["date"])]
                        else:
                            json_obj["date_numeric_label"] = ["Not Enough Information"]

                        if validate_json_output(json_obj):
                            save_result(json_obj, json_path)
                        else:
                            print(f"[WARN] Invalid JSON format for {filename} - image {image_path}")
                    except Exception as e:
                        print(f"[ERROR] JSON decode/normalization failed for {filename}: {e}")

                time.sleep(sleep)

            except Exception as e:
                print(f"[ERROR] GPT-4 failed on {filename} with image {image_path}: {e}")
                continue

def main():
    parser = argparse.ArgumentParser(description='Annotate Vietnamese news articles with GPT-4.')
    parser.add_argument('--image_dir_path', type=str, default='dataset/crawled_img/',
                        help='Directory containing the related images.')
    parser.add_argument('--article_dir_path', type=str, default='dataset/article_vie/',
                        help='Directory containing article text files.')
    parser.add_argument('--json_file_path', type=str, default='dataset/gpt4_annotations/annotations_vie.json',
                        help='Path to store the GPT-4 annotated results.')
    parser.add_argument('--system_prompt', type=str, default='dataset_collection/system_prompt.txt',
                        help='System prompt that defines the labeling task in Vietnamese.')
    parser.add_argument('--sleep', type=int, default=10,
                        help='Pause between API calls to avoid rate limits.')

    args = parser.parse_args()
    os.makedirs('dataset/gpt4_annotations', exist_ok=True)

    corpus = []
    article_files = [
        f for f in os.listdir(args.article_dir_path)
        if f.endswith(('.txt', '.chn', '.html'))
    ]
    all_images = [
        os.path.join(dp, f)
        for dp, _, filenames in os.walk(args.image_dir_path)
        for f in filenames
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    for file in article_files:
        file_stub = os.path.splitext(os.path.splitext(file)[0])[0]
        print(f"[DEBUG] Checking: {file_stub}")

        if entry_exists(args.json_file_path, file):
            print(f"[SKIP] Already exists in JSON: {file}")
            continue

        matching_images = [
            img for img in all_images
            if os.path.splitext(os.path.basename(img))[0] == file_stub
        ]

        if not matching_images:
            print(f"[SKIP] No matching images for {file_stub}")
            continue

        print(f"[APPEND] Matched {len(matching_images)} images for {file_stub}")

        image_path_list = matching_images
        image_url_list = [
            f"https://example.com/images/{os.path.basename(p)}"
            for p in image_path_list
        ]

        txt_path = os.path.join(args.article_dir_path, file)
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read().split('Image URLs')[0].strip()

        corpus.append((file, text, image_path_list, image_url_list))


    label_corpus(corpus, args.system_prompt, args.json_file_path, sleep=args.sleep)

if __name__ == '__main__':
    main()
