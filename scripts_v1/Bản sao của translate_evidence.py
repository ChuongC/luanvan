
# scripts/translate_evidence.py

import openai
import json
import argparse
import time
from tqdm import tqdm

def translate_text(text: str, model: str = "gpt-3.5-turbo") -> str:
    if not text.strip():
        return text
    prompt = f"Dịch đoạn văn sau sang tiếng Việt, giữ nguyên ngữ cảnh và ý nghĩa:\n\n{text}"
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "Bạn là một trợ lý dịch thuật chính xác, ngắn gọn và tự nhiên."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print("❌ Lỗi khi dịch:", e)
        return text  # fallback giữ nguyên nếu lỗi

def translate_evidence(input_path: str, output_path: str, api_key: str, model: str):
    openai.api_key = api_key

    with open(input_path, 'r', encoding='utf-8') as f:
        evidence_data = json.load(f)

    translated_data = []
    for item in tqdm(evidence_data):
        new_item = item.copy()
        for field in ["title", "description", "image_caption"]:
            if field in item and isinstance(item[field], str):
                new_item[field] = translate_text(item[field], model)
                time.sleep(1.5)
        translated_data.append(new_item)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate evidence fields from English to Vietnamese.")
    parser.add_argument('--openai_api_key', type=str, required=True)
    parser.add_argument('--input', type=str, default='dataset/retrieval_results/evidence.json')
    parser.add_argument('--output', type=str, default='dataset/retrieval_results/evidence_translated.json')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo')
    args = parser.parse_args()

    translate_evidence(args.input, args.output, args.openai_api_key, args.model)

