
import os
import sys
import json
import time
import argparse
import torch
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import clip
from bs4 import BeautifulSoup
import requests
from langdetect import detect
from typing import Dict, Any, List, Tuple
from sentence_transformers import SentenceTransformer, util

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from retrievers.multi_ris_engine import retrieve_ris
from utils import load_json, save_result
from dataset_collection.scrape_utils import *

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def is_scrapable(url):
    SKIP_DOMAINS = ["reddit.com", "x.com", "twitter.com", "instagram.com", "youtube.com"]
    return not any(domain in url for domain in SKIP_DOMAINS)

def rerank_by_clip(base_image_path, image_urls_dict, top_k=3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    base_img = preprocess(Image.open(base_image_path).convert("RGB")).unsqueeze(0).to(device)
    base_feat = model.encode_image(base_img).detach()

    scored_urls = []
    for url, img_urls in image_urls_dict.items():
        best_score = -1
        for img_url in img_urls:
            img = download_image_as_pil(img_url)
            if img is None:
                continue
            try:
                input_tensor = preprocess(img).unsqueeze(0).to(device)
                feat = model.encode_image(input_tensor).detach()
                score = torch.cosine_similarity(base_feat, feat).item()
                best_score = max(best_score, score)
            except:
                continue
        if best_score > 0:
            scored_urls.append((url, best_score))

    scored_urls.sort(key=lambda x: x[1], reverse=True)
    return [url for url, _ in scored_urls[:top_k]]

def ris_semantic_match(article_text: str, candidate_texts: List[str], threshold: float = 0.6) -> List[Tuple[str, float]]:
    if not candidate_texts:
        return []

    corpus_embeddings = model.encode(candidate_texts, convert_to_tensor=True)
    query_embedding = model.encode(article_text, convert_to_tensor=True)
    similarities = util.cos_sim(query_embedding, corpus_embeddings)[0]

    matches = []
    for text, score in zip(candidate_texts, similarities):
        if score >= threshold:
            matches.append((text, float(score)))

    matches.sort(key=lambda x: x[1], reverse=True)
    return matches

def robust_scraper(url: str, image_path: str = '', timeout: int = 10) -> Dict[str, Any]:
    try:
        html = robust_get(url, timeout=timeout)
        if not html:
            raise ValueError("Empty HTML")

        soup = BeautifulSoup(html, 'html.parser')
        metadata = scrape_metadata(soup, url)

        title = metadata.get("title", "")
        desc = metadata.get("description", "")
        text = metadata.get("full_text", "")

        full_content = f"{title} {desc} {text}".strip()

        if not full_content or len(full_content) < 200:
            raise ValueError("Content too short")

        lang = detect(full_content)
        metadata["language"] = lang

        ris_candidates: List[str] = metadata.get("ris_candidates", [])
        if ris_candidates:
            matches = ris_semantic_match(full_content, ris_candidates, threshold=0.6)
            matched_urls = [m[0] for m in matches]
            metadata["filtered_ris"] = matched_urls
            metadata["ris_scores"] = [round(m[1], 4) for m in matches]

        metadata["evidence_url"] = url
        metadata["image_path"] = image_path
        return metadata

    except Exception as e:
        print(f"[ERROR] Robust scraping failed for {url}: {e}")
        return {}

def collect_evidence_from_images(image_dir, output_path, ris_engine, api_key, max_results, top_k, sleep):
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    results = []
    for fname in tqdm(os.listdir(image_dir)):
        img_path = os.path.join(image_dir, fname)
        urls, image_urls_dict, _ = retrieve_ris(img_path, engine=ris_engine, api_key=api_key, max_results=max_results)
        top_urls = rerank_by_clip(img_path, image_urls_dict, top_k=top_k)
        entry = {
            "image_path": img_path,
            "urls": top_urls,
            "image_urls": {u: image_urls_dict[u] for u in top_urls if u in image_urls_dict}
        }
        results.append(entry)
        time.sleep(sleep)

    evidence_urls_format = []
    for entry in results:
        for url in entry["urls"]:
            evidence_urls_format.append({
                "image_path": entry["image_path"],
                "raw_url": url,
                "image_urls": entry["image_urls"].get(url, [])
            })

    save_result(evidence_urls_format, "dataset/retrieval_results/evidence_urls_vn.json")
    return results

def extract_scrapable_evidence(entries, scrape_output_path):
    all_scraped = []
    skipped_log_path = 'dataset/skipped_urls.txt'
    os.makedirs(os.path.dirname(scrape_output_path), exist_ok=True)
    skipped = []

    for entry in tqdm(entries):
        image_path = entry.get('image_path', 'unknown')
        for url in entry.get('urls', []):
            if not is_scrapable(url):
                skipped.append(f"SKIPPED: Not scrapable → {url}")
                continue

            result = robust_scraper(url, image_path=image_path)
            if not result:
                skipped.append(f"SKIPPED: Scraper returned None → {url}")
                continue

            all_scraped.append(result)

    with open(scrape_output_path, 'w') as f:
        json.dump(all_scraped, f, indent=4)

    if skipped:
        with open(skipped_log_path, 'w', encoding='utf-8') as logf:
            logf.write('\n'.join(skipped))

    return all_scraped

def validate_evidence_structure(entries):
    return [e for e in entries if isinstance(e, dict) and 'image_path' in e and 'urls' in e and 'image_urls' in e]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Improved RIS Evidence Collection (Vietnamese-aware)')
    parser.add_argument('--collect_google', type=int, default=1)
    parser.add_argument('--google_vision_api_key', type=str, default='')
    parser.add_argument('--image_path', type=str, default='dataset/img/')
    parser.add_argument('--ris_engine', type=str, default='google', choices=['google', 'yandex'])
    parser.add_argument('--max_results', type=int, default=30)
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--sleep', type=int, default=2)
    parser.add_argument('--raw_ris_path', type=str, default='dataset/retrieval_results/ris_vie.json')
    parser.add_argument('--scrape_output', type=str, default='dataset/retrieval_results/scraped_vie.json')
    parser.add_argument('--final_output', type=str, default='dataset/retrieval_results/evidence_vie.json')
    parser.add_argument('--apply_filtering', type=int, default=1)
    args = parser.parse_args()

    api_key = os.environ.get(args.google_vision_api_key) or args.google_vision_api_key

    if args.collect_google:
        ris_entries = collect_evidence_from_images(
            args.image_path,
            args.raw_ris_path,
            args.ris_engine,
            api_key,
            args.max_results,
            args.top_k,
            args.sleep
        )
        validated = validate_evidence_structure(ris_entries)
        evidence_metadata = [
            {"image_path": e["image_path"], "raw_url": url}
            for e in validated for url in e.get("urls", [])
        ]
    else:
        evidence_urls = load_json("dataset/retrieval_results/evidence_urls_vn.json")
        validated = []
        for entry in evidence_urls:
            img_path = entry["image_path"]
            raw_url = entry["raw_url"]
            image_urls = entry.get("image_urls", [])
            existing = next((v for v in validated if v["image_path"] == img_path), None)
            if existing:
                existing["urls"].append(raw_url)
                existing["image_urls"][raw_url] = image_urls
            else:
                validated.append({
                    "image_path": img_path,
                    "urls": [raw_url],
                    "image_urls": {raw_url: image_urls}
                })
        evidence_metadata = [
            {"image_path": e["image_path"], "raw_url": url}
            for e in validated for url in e.get("urls", [])
        ]

    # Filter metadata by processed image files
    processed_image_dir = 'dataset/processed_img_vie'
    valid_images = set(os.listdir(processed_image_dir))

    filtered_metadata = []
    for item in evidence_metadata:
        image_path = os.path.basename(item["image_path"])
        if image_path in valid_images:
            item["image_path"] = image_path
            filtered_metadata.append(item)

    evidence_metadata = filtered_metadata
    print(f"[INFO] {len(evidence_metadata)} metadata entries matched with processed images.")

    scraped = extract_scrapable_evidence(validated, args.scrape_output)
    scraped = [e for e in scraped if isinstance(e, dict) and 'image_path' in e and 'evidence_url' in e]
    print(f"[INFO] {len(scraped)} valid entries after filtering scraped results.")

    dataset = (
        load_json('dataset/train_custom.json') +
        load_json('dataset/val_custom.json') +
        load_json('dataset/test_custom.json')
    )

    final_merged = merge_data(scraped, evidence_metadata, dataset, apply_filtering=bool(args.apply_filtering))
    save_result(final_merged.to_dict(orient='records'), args.final_output)
