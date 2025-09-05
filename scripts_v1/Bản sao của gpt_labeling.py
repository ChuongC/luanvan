
import os
import json
import openai
from tqdm import tqdm
import time

openai.api_key = ""
INPUT_DIR = "dataset/article_vie"
OUTPUT_DIR = "dataset/labeled_articles_vie"
DEBUG = True

INSTRUCTION = """..."""  # giữ nguyên như trước

def is_valid_label(label):
    expected_keys = [
        "URL", "image_URL", "image_path", "publication_date", "provenance", "source", "date",
        "date_numeric_label", "location", "motivation", "org", "claim", "type_of_image",
        "verification_strategy", "verification_tool", "claimed_location", "claimed_date"
    ]
    return isinstance(label, dict) and all(k in label for k in expected_keys)

def clean_label(label):
    for key, value in label.items():
        if isinstance(value, list):
            label[key] = [v for v in value if isinstance(v, str) and v.strip()]
        elif isinstance(value, str) and not value.strip():
            label[key] = "unknown"
    if "date_numeric_label" in label:
        label["date_numeric_label"] = [v.strip() for v in label["date_numeric_label"] if v.strip()]
    if not label.get("claim", "").strip():
        label["claim"] = "unknown"
    return label

def gpt_label_article(article, retries=5, delay=5):
    prompt = f"""
    {INSTRUCTION}

    Bài viết:
    URL: {article.get('URL', '')}
    Text: {article.get('claim', '')[:1000]}...
    """
    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini-2024-07-18",
                messages=[
                    {"role": "system", "content": "Bạn là chuyên gia phân tích dữ liệu tin giả"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=700
            )
            content = response.choices[0].message.content.strip()
            if DEBUG:
                print("[DEBUG] Raw output:", content)
            label = json.loads(content)
            label = clean_label(label)
            if not is_valid_label(label):
                print("[WARN] Invalid label schema, retrying...")
                continue
            article.update(label)
            return article
        except json.JSONDecodeError:
            print("[ERROR] JSON decode failed.")
            if DEBUG:
                print(f"[RAW OUTPUT]:\n{content}\n")
        except Exception as e:
            print(f"[ERROR] GPT labeling failed: {e}")
        time.sleep(delay * (attempt + 1))
    return None

def validate_output_file(output_path):
    with open(output_path, "r", encoding="utf-8") as f:
        labeled = json.load(f)

    errors = []
    for idx, art in enumerate(labeled):
        if isinstance(art.get("date_numeric_label"), list):
            if any(d == "" for d in art["date_numeric_label"]):
                errors.append((idx, "Empty string in date_numeric_label"))
        for key in ["verification_strategy", "verification_tool"]:
            if isinstance(art.get(key), list):
                if any(v == "" for v in art[key]):
                    errors.append((idx, f"Empty string in {key}"))
        if not art.get("claim", "").strip():
            errors.append((idx, "Empty claim"))

    print(f"[VALIDATION] Tổng cộng {len(errors)} lỗi:")
    for e in errors:
        print(f" - Bài {e[0]}: {e[1]}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    json_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json")]

    for file_name in tqdm(json_files, desc="Processing files"):
        input_path = os.path.join(INPUT_DIR, file_name)
        output_path = os.path.join(OUTPUT_DIR, file_name)

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        results = []
        for article in data:
            labeled = gpt_label_article(article)
            if labeled:
                results.append(labeled)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        validate_output_file(output_path)

if __name__ == "__main__":
    main()
