import os
import sys
import re
import string
import time
import calendar
import numpy as np
import pandas as pd
from itertools import combinations
from datetime import datetime, timezone
from dateutil import parser
from haversine import haversine, Unit
from sklearn.metrics import ndcg_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer
from scipy.optimize import linear_sum_assignment

# --- NLTK setup ---
import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize

NLTK_DATA_DIR = '/root/nltk_data'
os.environ['NLTK_DATA'] = NLTK_DATA_DIR
for pkg in ["wordnet", "omw-1.4", "punkt", "punkt_tab"]:
    try:
        nltk.download(pkg, download_dir=NLTK_DATA_DIR, quiet=True)
    except:
        pass

# --- ROUGE & BERTScore ---

#from rouge_score import rouge_scorer
import evaluate
rouge = evaluate.load("rouge")

bertscore = evaluate.load("bertscore")

from bert_score import score as bert_score

# --- Local imports ---
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils import *
from dataset_collection.preprocessing_utils import *
from evaluation.geonames_collection import *



#################################
# Text normalization & metrics  #
#################################
# def normalize_text(text: str) -> str:
#     """Lowercase + remove punctuation + strip spaces"""
#     if not isinstance(text, str):
#         return ""
#     text = text.lower()
#     text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
#     return re.sub(r"\s+", " ", text).strip()

ABBREVIATIONS = {
    # Công ty / tổ chức
    "ltd": "limited",
    "inc": "incorporated",
    "co": "company",
    "corp": "corporation",
    "govt": "government",
    "org": "organization",
    "dept": "department",
    "assoc": "association",
    "univ": "university",
    "inst": "institute",
    "intl": "international",
    "natl": "national",
    "adm": "administration",
    "comm": "commission",
    "fed": "federal",
    
    # Địa danh
    "st": "street",
    "rd": "road",
    "ave": "avenue",
    "blvd": "boulevard",
    "mt": "mount",
    "ft": "fort",
    
    # Đơn vị hành chính
    "dist": "district",
    "prov": "province",
    "reg": "region",
    "cty": "county",
    "mun": "municipality",
    
    # Các từ phổ biến khác
    "etc": "et cetera",
    "vs": "versus",
    "no": "number",
    "yr": "year"
}


def normalize_text(text: str) -> str:
    """Lowercase + strip + expand common abbreviations, giữ token hữu ích."""
    if not isinstance(text, str):
        return ""
    # lowercase + trim
    text = text.lower().strip()

    # tokenize để tránh gộp từ sau khi xóa dấu
    tokens = word_tokenize(text)

    # mapping viết tắt
    expanded = [ABBREVIATIONS.get(tok, tok) for tok in tokens]

    # bỏ dấu câu đơn lẻ (., , , !, ?)
    expanded = [tok for tok in expanded if tok.isalnum()]

    return " ".join(expanded)


# def compute_meteor(predictions, references):
#     norm_preds = [normalize_text(p) for p in predictions]
#     norm_refs = [normalize_text(r) for r in references]
#     return np.mean([
#         meteor_score([ref], pred)
#         for pred, ref in zip(norm_preds, norm_refs)
#     ])
def compute_meteor(predictions, references):
    return np.mean([
        meteor_score([word_tokenize(str(ref))], word_tokenize(str(pred)))
        for pred, ref in zip(predictions, references)
    ])



def compute_rouge(predictions, references):
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = []
    for pred, ref in zip(predictions, references):
        result = rouge_scorer_obj.score(ref, pred)
        scores.append(result["rougeL"].fmeasure)
    return sum(scores) / len(scores)


def compute_bertscore(predictions, references):
    P, R, F1 = bert_score(predictions, references, lang='en')
    return float(F1.mean())


#################################
# Date utilities                #
#################################
def parse_date_safe(date_str):
    """Parse date string safely, return pandas.Timestamp (UTC) or None."""
    if not date_str or not isinstance(date_str, str):
        return None
    date_str = date_str.strip()
    try:
        dt = parser.parse(date_str, dayfirst=True, fuzzy=True, default=datetime(1900, 1, 1))
        return pd.Timestamp(dt).tz_convert("UTC")
    except Exception:
        # Nếu chỉ có năm
        if re.fullmatch(r"\d{4}", date_str):
            return pd.Timestamp(datetime(int(date_str), 7, 1, tzinfo=timezone.utc))
        # Nếu có năm-tháng
        if re.fullmatch(r"\d{4}[-/]\d{1,2}", date_str):
            y, m = map(int, re.split("[-/]", date_str))
            return pd.Timestamp(datetime(y, m, 15, tzinfo=timezone.utc))
        try:
            dt = pd.to_datetime(date_str, errors='coerce', dayfirst=True)
            if dt is pd.NaT:
                return None
            return pd.Timestamp(dt).tz_convert("UTC")
        except Exception:
            return None


def date_distance(dt1, dt2):
    """Compute relaxed distance between two dates."""
    if dt1 is None or dt2 is None:
        return np.inf
    dt1 = pd.Timestamp(dt1).tz_convert("UTC")
    dt2 = pd.Timestamp(dt2).tz_convert("UTC")
    if dt1 == dt2:
        return 0
    if dt1.year == dt2.year and dt1.month == dt2.month:
        return 1
    if dt1.year == dt2.year:
        return 30
    return abs(dt1.year - dt2.year) * 365


def extract_dates(text):
    """Extract candidate dates from text, return list pandas.Timestamp (UTC)."""
    if not text or not isinstance(text, str):
        return []
    date_patterns = re.findall(
        r'\b('
        r'\d{4}[-/]\d{1,2}[-/]\d{1,2}'
        r'|\d{1,2}[-/]\d{1,2}[-/]\d{4}'
        r'|\d{1,2}\s+\w+\s+\d{4}'
        r'|\w+\s+\d{1,2},?\s+\d{4}'
        r'|\w+\s+\d{4}'
        r'|\d{4}[-/]\d{1,2}'
        r'|\d{4}'
        r')\b', text
    )
    dates = []
    for d in date_patterns:
        try:
            dt = pd.to_datetime(d, errors="coerce", utc=True, dayfirst=True)
            if pd.notna(dt):
                dates.append(dt)
        except Exception:
            continue
    return dates


#################################
# Location utilities            #
#################################
def location_coordinate_distance(coordinates1, coordinates2, unit=1000):
    d = min([haversine(c1, c2, unit=Unit.KILOMETERS) for c1 in coordinates1 for c2 in coordinates2])
    d /= unit
    return d


def hierarchical_distance_metric(pred_hierarchy, gt_hierarchy):
    if all(i in pred_hierarchy for i in gt_hierarchy):
        return 0
    common_length = 0
    for p, g in zip(pred_hierarchy, gt_hierarchy):
        if p == g:
            common_length += 1
        else:
            break
    return len(pred_hierarchy) + len(gt_hierarchy) - 2 * common_length


def location_hierarchy_distance(hierarchy1, hierarchy2):
    d = min([hierarchical_distance_metric(h1, h2) for h1 in hierarchy1 for h2 in hierarchy2])
    return d


def is_strict_subset(sublist, mainlist):
    return set(sublist).issubset(set(mainlist)) and len(sublist) < len(mainlist)


def find_locations_to_remove(l):
    indices_to_remove = []
    def contains_strict_subset(outer_list, other_lists):
        for sublist in outer_list:
            for other_list in other_lists:
                for other_sublist in other_list:
                    if is_strict_subset(sublist, other_sublist):
                        return True
        return False
    for i, outer_list in enumerate(l):
        if contains_strict_subset(outer_list, [other_list for j, other_list in enumerate(l) if i != j]):
            indices_to_remove.append(i)
    return indices_to_remove


#################################
# Evaluation function           #
#################################
def evaluate(prediction,
             ground_truth,
             task,
             NER_model=None,
             geonames_data_path=None,
             geonames_username=None,
             sleep_geonames=2):
    # --- Source ---
    if task == "source":
        if not isinstance(prediction, list):
            prediction = [prediction]
        if not isinstance(ground_truth, list):
            ground_truth = [ground_truth]
        rouge_result = rouge.compute(predictions=prediction, references=[ground_truth])['rougeL']
        meteor_result = compute_meteor(prediction, [ground_truth])
        return {'rougeL': rouge_result, "meteor": meteor_result}

    # --- Motivation ---
    elif task == "motivation":
        if not isinstance(prediction, list):
            prediction = [prediction]
        if not isinstance(ground_truth, list):
            ground_truth = [ground_truth]
        rouge_result = rouge.compute(predictions=prediction, references=[ground_truth])['rougeL']
        meteor_result = compute_meteor(prediction, [ground_truth])
        berts_result = bertscore.compute(predictions=prediction,
                                         references=ground_truth,
                                         lang='en',
                                         model_type="distilbert-base-uncased")['f1'][0]

        return {'rougeL': rouge_result, "meteor": meteor_result, 'BertS': berts_result}

    # --- Location ---
    elif task == "location":
        if not isinstance(prediction, list):
            prediction = [prediction]
        if not isinstance(ground_truth, list):
            ground_truth = [ground_truth]
        rouge_result = rouge.compute(predictions=prediction, references=[ground_truth])['rougeL']
        meteor_result = compute_meteor(prediction, [ground_truth])
        return {'rougeL': rouge_result, "meteor": meteor_result}

    # --- Location NER ---
    elif task == "location NER":
        geonames_data = load_json(geonames_data_path)
        geonames_entries = list(set([d['query'].lower() for d in geonames_data]))
        prediction_location_NER = [l for l in extract_named_entities(prediction, NER_model, 'locations')]
        prediction_coordinates, prediction_hierarchies = [], []
        for p in prediction_location_NER:
            if p.lower() not in geonames_entries:
                matching_records = search_location(p, geonames_username, sleep_geonames)
                time.sleep(sleep_geonames)
                save_result(matching_records, geonames_data_path)
            else:
                matching_records = [d for d in geonames_data if 'coordinates' in d.keys() and d['query'].lower()==p.lower()]
            if len(matching_records) > 0:
                prediction_coordinates.append([r['coordinates'] for r in matching_records])
                prediction_hierarchies.append([r['hierarchy'] for r in matching_records])
        ground_truth_location_NER = [l for l in extract_named_entities(ground_truth, NER_model, 'locations')]
        ground_truth_coordinates, ground_truth_hierarchies = [], []
        for g in ground_truth_location_NER:
            matching_records = [d for d in geonames_data if 'coordinates' in d.keys() and d['query'].lower()==g.lower()]
            if len(matching_records) > 0:
                ground_truth_hierarchies.append([r['hierarchy'] for r in matching_records])
                ground_truth_coordinates.append([r['coordinates'] for r in matching_records])
        idx_to_remove = find_locations_to_remove(ground_truth_hierarchies)
        ground_truth_coordinates = [ground_truth_coordinates[i] for i in range(len(ground_truth_coordinates)) if i not in idx_to_remove]
        ground_truth_hierarchies = [ground_truth_hierarchies[i] for i in range(len(ground_truth_hierarchies)) if i not in idx_to_remove]
        # --- Compute codelta ---
        best_codelta = 0
        if len(prediction_coordinates) > 0:
            if len(prediction_coordinates) > len(ground_truth_coordinates):
                candidates = list(combinations(prediction_coordinates, len(ground_truth_coordinates)))
            else:
                candidates = [prediction_coordinates]
            for candidate in candidates:
                distances = np.array([[location_coordinate_distance(pc, gc) for gc in ground_truth_coordinates] for pc in candidate])
                row_ind, col_ind = linear_sum_assignment(distances)
                scores = sum(1/(1+d) for d in [distances[r,c] for r,c in zip(row_ind,col_ind)])
                coefficient = 1 / len(ground_truth_coordinates)
                codelta = coefficient * scores
                if codelta > best_codelta:
                    best_codelta = codelta
        # --- Compute hierarchy delta ---
        best_hierarchy_delta = 0
        if len(prediction_hierarchies) > 0:
            if len(prediction_hierarchies) > len(ground_truth_hierarchies):
                candidates = list(combinations(prediction_hierarchies, len(ground_truth_hierarchies)))
            else:
                candidates = [prediction_hierarchies]
            for candidate in candidates:
                distances = np.array([[location_hierarchy_distance(pc, gc) for gc in ground_truth_hierarchies] for pc in candidate])
                row_ind, col_ind = linear_sum_assignment(distances)
                scores = sum(1/(1+d) for d in [distances[r,c] for r,c in zip(row_ind,col_ind)])
                coefficient = 1 / len(ground_truth_hierarchies)
                hierarchy_delta  = coefficient * scores
                if hierarchy_delta > best_hierarchy_delta:
                    best_hierarchy_delta = hierarchy_delta
        return {"codelta": best_codelta, "hldelta": best_hierarchy_delta}

    # --- Date ---
    elif task == "date":
        print("\n[DEBUG] ===== DATE EVALUATION START =====")
        if isinstance(prediction, str) and prediction.startswith("[") and prediction.endswith("]"):
            prediction = prediction[1:-1]
        prediction_dates = extract_dates(prediction)
        print(f"[DEBUG] prediction raw: {prediction}")
        print(f"[DEBUG] prediction dates parsed: {prediction_dates}")
        if isinstance(ground_truth, list):
            ground_truth_dates = []
            for g in ground_truth:
                ground_truth_dates.extend(extract_dates(g))
        else:
            ground_truth_dates = extract_dates(ground_truth)
        print(f"[DEBUG] ground_truth raw: {ground_truth}")
        print(f"[DEBUG] ground_truth dates parsed: {ground_truth_dates}")
        if len(ground_truth_dates) == 0 or len(prediction_dates) == 0:
            print(f"[WARN] No valid dates: prediction={prediction_dates}, ground_truth={ground_truth_dates}")
            return {"exact_match": 0, "delta": 0}
        if len(prediction_dates) > len(ground_truth_dates):
            candidates = list(combinations(prediction_dates, len(ground_truth_dates)))
        else:
            candidates = [prediction_dates]
        best_delta, best_EM = 0, 0
        for idx, candidate in enumerate(candidates):
            distances = np.array([[date_distance(pd, gd) for gd in ground_truth_dates] for pd in candidate])
            print(f"[DEBUG] Candidate {idx+1}: {candidate}")
            print(f"[DEBUG] Distances matrix:\n{distances}")
            if np.all(np.isinf(distances)):
                print("[DEBUG] All distances are inf → skip candidate")
                continue
            row_ind, col_ind = linear_sum_assignment(distances)
            matched_distances = [distances[r, c] for r, c in zip(row_ind, col_ind)]
            print(f"[DEBUG] row_ind={row_ind}, col_ind={col_ind}")
            print(f"[DEBUG] matched_distances={matched_distances}")
            scores = sum(1/(1+d) for d in matched_distances)
            exact_match = np.all(np.array(matched_distances) == 0)
            coefficient = 1 / len(ground_truth_dates)
            delta = coefficient * scores
            print(f"[DEBUG] delta={delta:.4f}, exact_match={int(exact_match)}")
            if delta > best_delta:
                best_delta = delta
                best_EM = int(exact_match)
        if len(prediction_dates) > len(ground_truth_dates):
            best_EM = 0
        print(f"[DEBUG] BEST → delta={best_delta:.4f}, exact_match={best_EM}")
        print("[DEBUG] ===== DATE EVALUATION END =====\n")
        return {"exact_match": best_EM, "delta": best_delta}

    else:
        raise ValueError("Invalid task name")


#################################
# Evidence ranking              #
#################################
def cosine_similarity(vec1, vec2):
    norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    return np.dot(vec1, vec2) / (norm1 * norm2)


def compute_clip_score(image_index, evidence_index_list, image_embeddings, clip_evidence_embeddings):
    image = image_embeddings[image_index]
    similarities = []
    for idx in evidence_index_list:
        sim = cosine_similarity(image, clip_evidence_embeddings[idx])
        similarities.append((idx, sim))
    return [score for _, score in similarities]


def generate_ngrams(text, n=3):
    try:
        vectorizer = CountVectorizer(ngram_range=(1, n), stop_words='english')
        vectorizer.fit_transform([text])
        return set(vectorizer.get_feature_names_out())
    except:
        return None


def ngram_overlap_score(passage, answer, n=2):
    passage_ngrams = generate_ngrams(passage, n)
    answer_ngrams = generate_ngrams(answer, n)
    if answer_ngrams and passage_ngrams:
        overlap = passage_ngrams.intersection(answer_ngrams)
        return len(overlap) / max(len(passage_ngrams), len(answer_ngrams), 1)
    else:
        return 0


def eval_ranking(evidence, answer, predicted_ranking):
    ngram_scores = [ngram_overlap_score(e, answer) for e in evidence]
    target_ranking = np.array(ngram_scores)
    if len(target_ranking) > 1 and len(target_ranking) == len(predicted_ranking):
        ndcg = ndcg_score([target_ranking], [predicted_ranking])
        return ndcg
    else:
        return None


def get_ndcg_score(dataset, task, evidence, image_embeddings, clip_evidence_embeddings, image_embeddings_map, sort_with_date=False):
    total_ndcg, count = 0, 0
    img_corpus = [image['image_path'] for image in dataset]
    ground_truth = [image[task] for image in dataset]
    for i in range(len(img_corpus)):
        evidence_subset = [ev for ev in evidence if ev['image_path'] == img_corpus[i]]
        evidence_subset_index = [evidence.index(ev) for ev in evidence if ev['image_path'] == img_corpus[i]]
        if len(evidence_subset) > 3:
            image_index = int(image_embeddings_map[img_corpus[i]])
            if sort_with_date:
                date_sort = pd.DataFrame(evidence_subset).reset_index().sort_values(by='date',ascending=False).index.to_list()
                predicted_ranking = [date_sort.index(i) for i in pd.DataFrame(evidence_subset).reset_index().index.to_list()]
            else:
                predicted_ranking  = compute_clip_score(image_index,evidence_subset_index,image_embeddings, clip_evidence_embeddings)
            evidence_text = [ text[2:] for text in get_evidence_prompt(evidence_subset).split('Evidence ')[1:]]
                #only evaluate ranking when there are more than 3 evidence to select
            ndcg = eval_ranking(evidence_text,ground_truth[i],predicted_ranking)
            if ndcg !=None:
                total_ndcg +=ndcg
                count+=1
    if count != 0:
        return round(100*total_ndcg/count,2)
    else:
        return 'No matching evidence for those images'
