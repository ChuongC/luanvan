import os
import urllib.parse
import re
import traceback
from datetime import datetime

import numpy as np


def compute_metadata_scores(evidence_list):
    """
    Nhận list of evidence dicts, trả về array các điểm metadata.
    Added verbose debug printing per evidence item.
    """
    #print("[DEBUG:compute_metadata_scores] START - num evidence:", len(evidence_list))
    scores = []
    trusted_hosts = [
        # Quốc tế
        "bbc.com", "reuters.com", "nytimes.com", "cnn.com", "theguardian.com",
        "aljazeera.com", "bloomberg.com", "apnews.com", "associatedpress.com",
        "washingtonpost.com", "france24.com", "hrw.org", "voanews.com",

        # Ảnh và dữ liệu media
        "gettyimages.com", "alamy.com", "shutterstock.com", "wikimedia.org", "wikipedia.org",

        # Ấn Độ
        "indiatoday.in", "timesofindia.indiatimes.com", "thehindu.com", "ndtv.com",

        # Kenya / Ethiopia / châu Phi
        "businessdailyafrica.com", "citizen.digital", "dailytrust.com",
        "fanabc.com", "addisstandard.com", "ena.et",

        # Việt Nam
        "vnexpress.net", "thanhnien.vn", "tuoitre.vn", "zingnews.vn", "dantri.com.vn",
        "vietnamnet.vn", "nhandan.vn", "vov.vn", "vnmedia.vn"
    ]


    for i, evidence_item in enumerate(evidence_list):
        try:
            #print(f"[DEBUG:compute_metadata_scores] Processing evidence {i+1}/{len(evidence_list)}")
            score = 0.0

            hostname = str(evidence_item.get('hostname', '')).lower()
            #print(f"  hostname: '{hostname}'")
            if any(th in hostname for th in trusted_hosts):
                score += 1.0
                #print("  +1.0 for trusted hostname")

            # date_str = evidence_item.get('date', '')
            # print(f"  date_str: '{date_str}'")
            # if date_str:
            #     try:
            #         from dateutil import parser
            #         date_obj = parser.parse(date_str)
            #         days_diff = (datetime.now() - date_obj).days
            #         max_days = 365 * 5
            #         reversed_score = min(days_diff / max_days, 1.0)
            #         score += 1.0
            #         score += reversed_score
            #         print(f"  +1.0 for having a parsable date, +{reversed_score:.3f} for recency (days_diff={days_diff})")
            #     except Exception as e:
            #         print(f"  [WARN] Could not parse date '{date_str}': {e}")

            date_str = evidence_item.get('date', '')
            #print(f"  date_str: '{date_str}'")
            if date_str:
                try:
                    date_obj = parser.parse(date_str)
                    # Không để ngày tương lai được điểm
                    if date_obj > datetime.now():
                       # print(f"  [WARN] Future date {date_obj}, skip scoring")
                        return score
                    # Càng gần hiện tại thì càng tốt (ngược với code cũ)
                    days_diff = (datetime.now() - date_obj).days
                    max_days = 365 * 20  # cho 20 năm là xa nhất
                    proximity_score = max(0.0, 1.0 - days_diff / max_days)

                    score += 1.0  # có date hợp lệ
                    score += proximity_score
                    #print(f"  +1.0 for valid date, +{proximity_score:.3f} for being recent (days_diff={days_diff})")

                except Exception as e:
                    print(f"  [WARN] Could not parse date '{date_str}': {e}")

            desc_len = len(str(evidence_item.get('description', '')))
            title_len = len(str(evidence_item.get('title', '')))
            caption_len = len(str(evidence_item.get('image_caption', '')))
            #print(f"  lengths - description: {desc_len}, title: {title_len}, image_caption: {caption_len}")

            if desc_len > 10:
                score += 1.0
                #print("  +1.0 for description length > 10")
            if title_len > 10:
                score += 1.0
               # print("  +1.0 for title length > 10")
            if caption_len > 10:
                score += 1.0
                #print("  +1.0 for image_caption length > 10")

            #print(f"  => metadata score for evidence {i}: {score}")
            scores.append(score)
        except Exception as e:
            #print(f"[ERROR:compute_metadata_scores] Exception on evidence {i}: {e}")
            traceback.print_exc()
            scores.append(0.0)

    arr = np.array(scores, dtype=float)
   # print(f"[DEBUG:compute_metadata_scores] END - scores (first 20): {arr[:20]}")
    return arr


# small utility cosine similarity (vectorized) - keep for internal use
def cosine_similarity_vec(a, b):
    """
    Cosine similarity giữa hai vector 1D
    """
    try:
        a_norm = a / (np.linalg.norm(a) + 1e-8)
        b_norm = b / (np.linalg.norm(b) + 1e-8)
        sim = float(np.dot(a_norm, b_norm))
        # Debuggable print - keep it lightweight
        print(f"[DEBUG:cosine_similarity_vec] sim: {sim:.6f}")
        return sim
    except Exception as e:
        print(f"[ERROR:cosine_similarity_vec] Exception: {e}")
        return 0.0


def rerank_evidence_with_metadata(image_index,
                                  candidate_evidence_indices,
                                  image_embeddings,
                                  clip_evidence_embeddings,
                                  evidence,
                                  alpha=0.7):
    """
    Rerank candidate evidence.
    image_index: int
    candidate_evidence_indices: list of indices (global indices into evidence list)
    image_embeddings: np.array [n_images, emb_dim]
    clip_evidence_embeddings: np.array [n_evidence, emb_dim]
    evidence: list of dicts
    alpha: trọng số giữa similarity và metadata score
    """
   # print(f"[DEBUG:rerank] START image_index={image_index} num_candidates={len(candidate_evidence_indices)} alpha={alpha}")

    # Defensive checks
    if not isinstance(candidate_evidence_indices, (list, np.ndarray)) or len(candidate_evidence_indices) == 0:
      #  print("[WARN:rerank] No candidate_evidence_indices provided, returning empty list")
        return []

    try:
        image_emb = image_embeddings[int(image_index)]  # 1D
        print(f"[DEBUG:rerank] image_emb shape: {image_emb.shape}")
    except Exception as e:
      #  print(f"[ERROR:rerank] Could not fetch image embedding for index {image_index}: {e}")
        traceback.print_exc()
        return candidate_evidence_indices

    try:
        candidates_emb = clip_evidence_embeddings[candidate_evidence_indices]  # 2D
       # print(f"[DEBUG:rerank] candidates_emb shape: {candidates_emb.shape}")
    except Exception as e:
      #  print(f"[ERROR:rerank] Could not index clip_evidence_embeddings with provided indices: {e}")
        traceback.print_exc()
        return candidate_evidence_indices

    # Cosine similarity vectorized
    image_norm = np.linalg.norm(image_emb) + 1e-8
    candidates_norm = np.linalg.norm(candidates_emb, axis=1) + 1e-8
    sim_scores = (candidates_emb @ image_emb) / (candidates_norm * image_norm)
  #  print(f"[DEBUG:rerank] sim_scores (first 10): {sim_scores[:10]}")

    # Metadata scores
    meta_scores_all = compute_metadata_scores(evidence)
    try:
        meta_scores = meta_scores_all[candidate_evidence_indices]
    except Exception as e:
      #  print(f"[ERROR:rerank] Could not index meta_scores_all with candidate indices: {e}")
        meta_scores = np.zeros_like(sim_scores)

   # print(f"[DEBUG:rerank] meta_scores (first 10): {meta_scores[:10]}")

    # Kết hợp final score
    final_scores = alpha * sim_scores + (1 - alpha) * meta_scores
   # print(f"[DEBUG:rerank] final_scores (first 10): {final_scores[:10]}")

    # Sắp xếp giảm dần
    sorted_order = np.argsort(-final_scores)
    sorted_global_indices = [candidate_evidence_indices[int(i)] for i in sorted_order]
   # print(f"[DEBUG:rerank] sorted_global_indices (first 20): {sorted_global_indices[:20]}")
   # print(f"[DEBUG:rerank] END")
    return sorted_global_indices


def sanitize_prompt(prompt):
    forbidden_keywords = [
        "biden", "trump", "ukraine", "covid", "vaccine", "virus", "china", "phá thai", "đảng cộng hòa",
        "ivermectin", "hydroxychloroquine", "russia", "ukrain", "abort", "abortion", "pandemic",
        "sars", "pfizer", "astrazeneca", "moderna"
    ]

    original = str(prompt)
    sanitized = original
    for word in forbidden_keywords:
        sanitized = re.sub(rf"\b{re.escape(word)}\b", "[REDACTED]", sanitized, flags=re.IGNORECASE | re.UNICODE)

    if sanitized != original:
        # Print only when something was redacted to reduce noise
        print(f"[DEBUG:sanitize_prompt] Redacted keywords in prompt. Original len={len(original)} -> sanitized len={len(sanitized)}")
    return sanitized


# Keep a separate name to avoid collision with other imports
def cosine_similarity_safe(vec1, vec2):
    """
    Compute cosine similarity between two vectors (safe wrapper).
    """
    try:
        norm1 = np.linalg.norm(vec1) + 1e-10
        norm2 = np.linalg.norm(vec2) + 1e-10
        sim = float(np.dot(vec1, vec2) / (norm1 * norm2))
        print(f"[DEBUG:cosine_similarity_safe] sim={sim}")
        return sim
    except Exception as e:
        print(f"[ERROR:cosine_similarity_safe] Exception: {e}")
        return 0.0


def sort_with_clip_score(image_index,
                        evidence_index_list,
                        image_embeddings,
                        clip_evidence_embeddings):
    """
    Sort the candidate evidence based on CLIP score.
    """
    print(f"[DEBUG:sort_with_clip_score] START for image_index={image_index} num_candidates={len(evidence_index_list)}")
    image = image_embeddings[int(image_index)]
    similarities = []

    for idx in evidence_index_list:
        try:
            sim = cosine_similarity_safe(image, clip_evidence_embeddings[int(idx)])
            similarities.append((idx, sim))
            print(f"  candidate idx={idx} sim={sim}")
        except Exception as e:
            print(f"  [ERROR] computing sim for idx={idx}: {e}")

    # Sort by similarity in descending order
    sorted_indices = sorted(similarities, key=lambda x: x[1], reverse=True)
    print(f"[DEBUG:sort_with_clip_score] sorted top 10: {sorted_indices[:10]}")
    # Return only indices, not similarities
    return [idx for idx, _ in sorted_indices]


# We will import sklearn cosine_similarity locally inside this function to avoid global name collision
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity


def sort_with_image_similarity(image_index,
                               train_images_index_list,
                               image_embeddings):
    """
    Sort the candidate demonstrations based on CLIP similarity between images.
    Expects train_images_index_list to be a list of (train_list_idx, image_map_idx) OR just indices.
    """
    print(f"[DEBUG:sort_with_image_similarity] START image_index={image_index} num_train_candidates={len(train_images_index_list)}")
    image = image_embeddings[int(image_index)].reshape(1, -1)  # reshape 1D -> 2D (1 sample, n_features)
    similarities = []

    for item in train_images_index_list:
        try:
            if isinstance(item, tuple):
                idx, emb_idx = item
            else:
                idx, emb_idx = item, item

            emb_vec = image_embeddings[int(emb_idx)].reshape(1, -1)
            sim = float(sklearn_cosine_similarity(image, emb_vec)[0][0])
            similarities.append((idx, sim))
            print(f"  candidate train idx={idx}, emb_idx={emb_idx}, sim={sim}")
        except Exception as e:
            print(f"  [ERROR] computing image similarity for item={item}: {e}")

    # Sort by similarity in descending order
    sorted_indices = sorted(similarities, key=lambda x: x[1], reverse=True)
    print(f"[DEBUG:sort_with_image_similarity] sorted top 10: {sorted_indices[:10]}")
    return [idx for idx, _ in sorted_indices]


def normalize_path(path):
    # Giải mã URL encoding nếu có
    path = urllib.parse.unquote(path)
    # Chuẩn hóa dấu slash, đường dẫn
    norm = os.path.normpath(path)
    #print(f"[DEBUG:normalize_path] '{path}' -> '{norm}'")
    return norm


def get_topk_evidence(image_path,
                      evidence,
                      image_embeddings,
                      clip_evidence_embeddings,
                      image_map,
                      k=3,
                      alpha=0.7):

    #print(f"[DEBUG:get_topk_evidence] START for image_path: {image_path}")
    # Ưu tiên khớp full path trước
    matching_evidence = [
        ev for ev in evidence
        if normalize_path(ev.get('image_path', '')) == normalize_path(image_path)
    ]

    if not matching_evidence:
        # Fallback basename
        image_basename = os.path.basename(image_path)
        matching_evidence = [
            ev for ev in evidence
            if os.path.basename(ev.get('image_path', '')) == image_basename
        ]

    if not matching_evidence:
     #   print(f"[WARN:get_topk_evidence] No matching evidence for image: {image_path}")
      #  print(f"[DEBUG:get_topk_evidence] evidence sample keys (first 3): {[list(ev.keys()) for ev in evidence[:3]]}")
        return []

    # compute global indices
    evidence_index = [i for i, ev in enumerate(evidence) if ev in matching_evidence]
    #print(f"[DEBUG:get_topk_evidence] Found matching evidence global indices: {evidence_index}")

    if len(evidence_index) > k:
        normalized_path = normalize_path(image_path)
        image_index = None
        if normalized_path in image_map:
            image_index = int(image_map[normalized_path])
     #       print(f"[DEBUG:get_topk_evidence] Found image_index in image_map: {image_index}")
        else:
            basename_map = {os.path.basename(p): idx for p, idx in image_map.items()}
            basename_img = os.path.basename(image_path)
            if basename_img in basename_map:
                image_index = int(basename_map[basename_img])
       #         print(f"[DEBUG:get_topk_evidence] Found image_index via basename map: {image_index}")
            else:
      #          print(f"[ERROR:get_topk_evidence] {image_path} not found in image_map keys (normalized: {normalized_path})")
                return []

        sorted_evidence = rerank_evidence_with_metadata(
            image_index,
            evidence_index,
            image_embeddings,
            clip_evidence_embeddings,
            evidence,
            alpha=alpha
        )
        #print(f"[DEBUG:get_topk_evidence] sorted_evidence (first {k}): {sorted_evidence[:k]}")
        return sorted_evidence[:k]
    else:
        #print(f"[DEBUG:get_topk_evidence] returning evidence_index (<=k): {evidence_index}")
        return evidence_index


def get_topk_demonstrations(image_path,
                            question,
                            train,
                            image_embeddings,
                            image_map,
                            k=3):

    #print(f"[DEBUG:get_topk_demonstrations] START for image_path={image_path} question={question} k={k}")
    train_image_idx = []
    subset_train_index = [
        t for t in range(len(train))
        if str(train[t].get(question, '')).lower() != 'not enough information'
    ]
    subset_train = [train[t] for t in range(len(train)) if t in subset_train_index]
    #print(f"[DEBUG:get_topk_demonstrations] subset_train size: {len(subset_train)}")

    for idx, i in zip(subset_train_index, [t.get('image_path', '') for t in subset_train]):
        normalized_i = normalize_path(i)
        try:
            if normalized_i in image_map:
                train_image_idx.append((idx, int(image_map[normalized_i])))
                #print(f"  mapped train idx {idx} -> image_map[{normalized_i}] = {image_map[normalized_i]}")
            else:
                basename_map = {os.path.basename(p): idx2 for p, idx2 in image_map.items()}
                basename_i = os.path.basename(normalized_i)
                if basename_i in basename_map:
                    train_image_idx.append((idx, int(basename_map[basename_i])))
                   # print(f"  mapped train idx {idx} via basename {basename_i} -> {basename_map[basename_i]}")
                else:
                   print(f"[WARN:get_topk_demonstrations] Train image {i} not found in image_map")
        except Exception as e:
           # print(f"[ERROR:get_topk_demonstrations] Mapping train image {i}: {e}")
            traceback.print_exc()

    normalized_path = normalize_path(image_path)
    image_idx = None
    if normalized_path in image_map:
        image_idx = int(image_map[normalized_path])
       # print(f"[DEBUG:get_topk_demonstrations] Found test image_idx in image_map: {image_idx}")
    else:
        basename_map = {os.path.basename(p): idx for p, idx in image_map.items()}
        basename_img = os.path.basename(normalized_path)
        if basename_img in basename_map:
            image_idx = int(basename_map[basename_img])
           # print(f"[DEBUG:get_topk_demonstrations] Found test image_idx via basename map: {image_idx}")
        else:
            print(f"[ERROR:get_topk_demonstrations] Test image {image_path} not found in image_map")
            return []

    sorted_candidates = sort_with_image_similarity(image_idx, train_image_idx, image_embeddings)
  #  print(f"[DEBUG:get_topk_demonstrations] sorted_candidates (first {k}): {sorted_candidates[:k]}")
    return sorted_candidates[:k]


# Text processing helpers
from tqdm import tqdm
import torch
from PIL import Image


def sanitize_text_fields(evidence_item):
    keys_to_clean = ['title', 'author', 'description', 'image_caption']
    print(f"[DEBUG:sanitize_text_fields] START for evidence item keys: {list(evidence_item.keys())}")
    for key in keys_to_clean:
        if key in evidence_item and evidence_item[key]:
            before = evidence_item[key]
            after = sanitize_prompt(before)
            evidence_item[key] = after
            if before != after:
                print(f"  [DEBUG] sanitized field '{key}': changed")
    print(f"[DEBUG:sanitize_text_fields] END")
    return evidence_item


def get_evidence_prompt(evidence_list):
    prompt = ''
    print(f"[DEBUG:get_evidence_prompt] START building prompt from {len(evidence_list)} evidence items")
    for i, ev in enumerate(evidence_list):
        ev = sanitize_text_fields(ev)
        text = f"Evidence {i}\n"
        for key in ['evidence_url', 'hostname', 'sitename', 'title', 'author', 'date', 'description', 'image_caption']:
            field_val = ev.get(key, '')
            if field_val and 'Caption not found' not in field_val and 'Image not found' not in field_val:
                field = sanitize_prompt(field_val)
                label = key.replace('_', ' ').capitalize()
                text += f"{label}: {field}\n"
        text += "\n"
        prompt += text
    print(f"[DEBUG:get_evidence_prompt] END prompt length: {len(prompt)} chars")
    return prompt


# Tokenization/truncation helpers

def truncate_text(evidence_text, tokenizer, max_length=128):
    '''
    Truncate the evidence text to the maximum token length accepted by the multilingual CLIP model.
    '''
    # Tokenize the input
    try:
        tokens = tokenizer.encode(evidence_text, add_special_tokens=True)
        print(f"[DEBUG:truncate_text] original tokens len={len(tokens)} max_length={max_length}")
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        return tokenizer.decode(tokens, skip_special_tokens=True)
    except Exception as e:
        print(f"[ERROR:truncate_text] Tokenization failed: {e}")
        traceback.print_exc()
        # Fallback: simple truncation of characters
        return evidence_text[:max_length]


def get_tokenized_evidence(evidence, tokenizer):
    '''
    Get tokenized to compute the text embeddings.
    '''
    text_list = []
    print(f"[DEBUG:get_tokenized_evidence] START num evidence={len(evidence)}")
    for s in tqdm(range(len(evidence))):
        try:
            text = ''
            text += evidence[s].get('title', '') or ''
            if 'Caption not found' not in evidence[s].get('image_caption', '') and 'Image not found' not in evidence[s].get('image_caption', ''):
                text += evidence[s].get('image_caption', '') or ''
            if text == '':
                text += evidence[s].get('description', '') or ''
            text = truncate_text(text, tokenizer)
            text_list.append(text)
        except Exception as e:
            print(f"[ERROR:get_tokenized_evidence] at index {s}: {e}")
            traceback.print_exc()
            text_list.append('')
    print(f"[DEBUG:get_tokenized_evidence] END - produced {len(text_list)} texts")
    return text_list


def compute_clip_text_embeddings(texts, model, tokenizer,  batch_size=16):
    '''
    Compute the embeddings of evidence text passages
    '''
    all_embeddings = []
    print(f"[DEBUG:compute_clip_text_embeddings] START num_texts={len(texts)} batch_size={batch_size}")
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        try:
            with torch.no_grad():
                batch_embeddings = model.forward(batch_texts, tokenizer).detach()
                all_embeddings.append(batch_embeddings.cpu().numpy())
                print(f"  [DEBUG] processed batch {i//batch_size} size={len(batch_texts)}")
        except Exception as e:
            print(f"  [ERROR] compute_clip_text_embeddings batch {i//batch_size}: {e}")
            traceback.print_exc()
    if len(all_embeddings) == 0:
        print("[WARN:compute_clip_text_embeddings] No embeddings computed, returning empty array")
        return np.zeros((0,))
    result = np.concatenate(all_embeddings, axis=0)
    print(f"[DEBUG:compute_clip_text_embeddings] END result shape: {result.shape}")
    return result


def compute_clip_image_embeddings(image_paths, preprocess, model, batch_size=32):
    '''
    Compute image embeddings.
    '''
    all_embeddings = []
    import torch
    from PIL import Image
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DEBUG:compute_clip_image_embeddings] START num_images={len(image_paths)} device={device}")
    for i in tqdm(range(0, len(image_paths), batch_size)):
        try:
            batch_images = [preprocess(Image.open(path)).unsqueeze(0) for path in image_paths[i:i+batch_size]]
            batch_images_tensor = torch.cat(batch_images).to(device)

            # Compute embeddings
            with torch.no_grad():
                batch_embeddings = model.encode_image(batch_images_tensor).detach()
                all_embeddings.append(batch_embeddings.cpu().numpy())
            print(f"  [DEBUG] processed image batch {i//batch_size} size={len(batch_images)}")
        except Exception as e:
            print(f"  [ERROR] compute_clip_image_embeddings batch {i//batch_size}: {e}")
            traceback.print_exc()

    if len(all_embeddings) == 0:
        print("[WARN:compute_clip_image_embeddings] No image embeddings computed, returning empty array")
        return np.zeros((0,))

    result = np.concatenate(all_embeddings, axis=0)
    print(f"[DEBUG:compute_clip_image_embeddings] END result shape: {result.shape}")
    return result
