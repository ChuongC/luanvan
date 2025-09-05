import torch
import clip
from multilingual_clip import pt_multilingual_clip
from transformers import AutoTokenizer
import numpy as np
import os
import sys
import time
import json

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils import *
from baseline.generation_utils import *


def log(msg):
    print(f"[INFO] {msg}")


if __name__ == '__main__':
    log("Khởi động tiến trình...")

    if 'embeddings' not in os.listdir('dataset/'):
        os.mkdir('dataset/embeddings')
        log("Tạo thư mục dataset/embeddings")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Sử dụng thiết bị: {device.upper()}")

    model_name = 'M-CLIP/XLM-Roberta-Large-Vit-L-14'
    log(f"Tải mô hình text: {model_name}")

    text_model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Evidence
    log("Đọc dữ liệu bằng chứng từ JSON...")
    evidence = load_json('dataset/retrieval_results/evidence.json')
    log(f"Tổng số bằng chứng: {len(evidence)}")

    log("Tokenizing evidence...")
    text_list = get_tokenized_evidence(evidence, tokenizer)
    log(f"Số đoạn văn đã tokenize: {len(text_list)}")
    log(f"Ví dụ: {text_list[0][:100]}...")

    log("Tính toán embedding cho evidence...")
    start = time.time()
    evidence_embeddings = compute_clip_text_embeddings(text_list, text_model, tokenizer, batch_size=16)
    log(f"Hoàn tất trích xuất text embedding trong {time.time() - start:.2f} giây")
    log(f"Kích thước embedding: {evidence_embeddings.shape}")

    np.save('dataset/embeddings/evidence_embeddings.npy', evidence_embeddings)
    log("Đã lưu evidence_embeddings.npy")

    # Images
    log("Tải mô hình CLIP ảnh...")
    image_model, preprocess = clip.load('ViT-L/14', device=device)

    image_paths = ['dataset/processed_img/' + i for i in os.listdir('dataset/processed_img/')]
    log(f"Tổng số ảnh: {len(image_paths)}")
    log(f"Ví dụ ảnh: {image_paths[0]}")

    list_dict = {image_paths[i]: str(i) for i in range(len(image_paths))}
    with open('dataset/embeddings/image_embeddings_map.json', 'w') as json_file:
        json.dump(list_dict, json_file)
    log("Đã lưu ánh xạ image_embeddings_map.json")

    log("Tính toán embedding cho ảnh...")
    start = time.time()
    image_embeddings = compute_clip_image_embeddings(image_paths, preprocess, image_model)
    log(f"Hoàn tất trích xuất image embedding trong {time.time() - start:.2f} giây")
    log(f"Kích thước embedding ảnh: {image_embeddings.shape}")

    np.save('dataset/embeddings/image_embeddings.npy', image_embeddings)
    log("Đã lưu image_embeddings.npy")

    log("--- HOÀN TẤT ---")
