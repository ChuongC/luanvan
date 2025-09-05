

import openai
import json
import argparse
import time
import re
import unicodedata

def clean_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = ''.join(c for c in text if unicodedata.category(c)[0] != 'C' or c in '\n\t')
    text = text.replace('\\n', '\n').strip()
    return text

def split_long_text(text, max_words=250):
    sentences = re.split(r'(?<=[.?!])\s+', text)
    chunks, current = [], ""
    for sentence in sentences:
        if len(current.split()) + len(sentence.split()) < max_words:
            current += " " + sentence
        else:
            chunks.append(current.strip())
            current = sentence
    if current:
        chunks.append(current.strip())
    return chunks

def translate_text(text, model="gpt-3.5-turbo"):
    prompt = f"Dịch đoạn văn sau sang tiếng Việt:\n\n{text}"
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "Bạn là một trợ lý dịch thuật chính xác, tự nhiên."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print("❌ Lỗi dịch:", e)
        return text

def translate_list_json(input_path, output_path, api_key, model):
    openai.api_key = api_key

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("❌ Dữ liệu không phải là danh sách JSON.")
        return

    FIELDS_TO_TRANSLATE = ["title", "description", "text", "image_caption"]
    translated_list = []

    for idx, item in enumerate(data):
        print(f"\n🔄 Dịch item {idx+1}/{len(data)}")
        new_item = item.copy()
        for field in FIELDS_TO_TRANSLATE:
            if field in item and isinstance(item[field], str):
                print(f"  ➤ Trường: {field}")
                cleaned = clean_text(item[field])
                if field == "text":
                    chunks = split_long_text(cleaned)
                    translated_chunks = []
                    for i, chunk in enumerate(chunks):
                        print(f"    ➤ Đoạn {i+1}/{len(chunks)}")
                        translated_chunk = translate_text(chunk, model)
                        translated_chunks.append(translated_chunk)
                        time.sleep(1.5)
                    new_item[field] = "\n\n".join(translated_chunks)
                else:
                    new_item[field] = translate_text(cleaned, model)
                    time.sleep(1.5)
        translated_list.append(new_item)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(translated_list, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Đã lưu danh sách JSON dịch: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate a single Trafilatura JSON entry to Vietnamese.")
    parser.add_argument('--openai_api_key', type=str, required=True)
    parser.add_argument('--input', type=str, default='dataset/retrieval_results/trafilatura_data.json')
    parser.add_argument('--output', type=str, default='dataset/retrieval_results/trafilatura_data_translated.json')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo')
    args = parser.parse_args()

    translate_list_json(args.input, args.output, args.openai_api_key, args.model)

