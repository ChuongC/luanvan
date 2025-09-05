
import os
import sys
import time
import openai
import argparse
from tqdm import tqdm
from dotenv import load_dotenv

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utils import *
from baseline.llm_prompting_sample import *

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def label_corpus(corpus, system_prompt_path, json_path, sleep=20):
    system_prompt = open(system_prompt_path, encoding='utf-8').read()
    total = len(corpus)
    success_count = 0
    start_time = time.time()

    for idx, article in enumerate(tqdm(corpus, desc="Processing articles")):
        article_text, local_img_path, metadata = article
        full_prompt = system_prompt.strip() + "\n\n" + f"Bài viết:\n{article_text.strip()}"

        output = call_openai_gpt4o(
            full_prompt,
            image_path=local_img_path,
            temperature=0.2,
            max_tokens=1024
        )

        if isinstance(output, str):
            save_result(output, json_path)
            success_count += 1

            # === PRINT KẾT QUẢ CHI TIẾT ===
            print(f"\n📌 [{idx+1}/{total}] ĐÃ GÁN NHÃN CHO:")
            print(f"Ảnh       : {local_img_path}")
            if metadata:
                print(f"Metadata  : {metadata}")
            print(f"Văn bản   : {article_text.strip()[:150]}...")  # rút gọn nếu quá dài
            print(f"✅ Gán nhãn:\n{output.strip()}")
            print("=" * 80)

            time.sleep(sleep)


    end_time = time.time()
    duration = end_time - start_time
    avg_speed = success_count / duration if duration > 0 else 0

    print("\n================= BÁO CÁO GÁN NHÃN =================")
    print(f"Tổng số bài viết cần xử lý : {total}")
    print(f"Tổng số bài đã xử lý       : {success_count}")
    print(f"Tổng thời gian thực hiện   : {duration:.2f} giây")
    print(f"Tốc độ trung bình          : {avg_speed:.2f} bài/giây")
    print("====================================================\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Annotate Vietnamese articles using GPT-4o.')
    parser.add_argument('--image_dir_path', type=str, default='dataset/img/')
    parser.add_argument('--article_dir_path', type=str, default='dataset/article/')
    parser.add_argument('--json_file_path', type=str, default='dataset/gpt4_annotations/annotations.json')
    parser.add_argument('--system_prompt', type=str, default='dataset_collection/system_prompt.txt')

    args = parser.parse_args()

    if 'gpt4_annotations' not in os.listdir('dataset'):
        os.mkdir('dataset/gpt4_annotations/')

    corpus = get_corpus(args.article_dir_path, args.json_file_path, args.image_dir_path)
    label_corpus(corpus, args.system_prompt, args.json_file_path)
