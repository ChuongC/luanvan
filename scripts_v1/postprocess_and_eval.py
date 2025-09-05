
import json
from collections import defaultdict, Counter
from pathlib import Path
import argparse
from tabulate import tabulate

UNKNOWN_KEYWORDS = [
    "not provided", "do not provide", "not mentioned", "not available",
    "not enough information", "no specific", "no information",
    "cannot be determined", "unspecified", "unclear"
]

def is_unknown(text):
    lower = text.lower().strip()
    if not lower or lower in ["none", "-", "n/a"]:
        return True
    return any(kw in lower for kw in UNKNOWN_KEYWORDS)

def clean_output(raw):
    raw = raw.strip()
    if is_unknown(raw):
        return "not enough information"
    return raw

from dateutil import parser

def to_iso8601(date_str):
    try:
        dt = parser.parse(date_str)
        return dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    except Exception:
        return None

def evaluate(predictions, task):
    stats = {
        "total": 0,
        "correct": 0,
        "incorrect": 0,
        "gt_unknown": 0,
        "gt_known": 0,
        "out_unknown": 0,
        "out_known": 0,
    }
    mismatches = []
    answers = []

    for pred in predictions:
        img = pred["img_path"]
        raw_gt = pred["ground_truth"]
        raw_out = pred["output"]

        gt = clean_output(raw_gt).lower()
        out = clean_output(raw_out).lower()

        stats["total"] += 1
        answers.append(out)

        if gt == "not enough information":
            stats["gt_unknown"] += 1
        else:
            stats["gt_known"] += 1

        if out == "not enough information":
            stats["out_unknown"] += 1
        else:
            stats["out_known"] += 1

        if task == "date":
            gt_norm = to_iso8601(gt) or gt
            out_norm = to_iso8601(out) or out
        else:
            gt_norm = gt
            out_norm = out

        if gt_norm == out_norm:
            stats["correct"] += 1
        else:
            stats["incorrect"] += 1
            mismatches.append({
                "img_path": img,
                "ground_truth": gt,
                "output": out
            })

    stats["accuracy"] = stats["correct"] / stats["total"] * 100
    return stats, mismatches, answers

def print_stats(stats, task, answers):
    print(f"\n📊 Evaluation Report for Task: {task.upper()}")
    print(f"  ✅ Accuracy:       {stats['accuracy']:.2f}%")
    print(f"  🟢 Correct:        {stats['correct']} / {stats['total']}")
    print(f"  🔴 Incorrect:      {stats['incorrect']}")
    print(f"\n  Ground Truth Types:")
    print(f"    • Unknown:       {stats['gt_unknown']}")
    print(f"    • Known:         {stats['gt_known']}")
    print(f"\n  Output Types:")
    print(f"    • Unknown:       {stats['out_unknown']}")
    print(f"    • Known:         {stats['out_known']}")

    print("\n  🔠 Top 10 Answer Frequencies:")
    top_answers = Counter(answers).most_common(10)
    print(tabulate(top_answers, headers=["Answer", "Count"]))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default="output/results.json", help="Path to results.json")
    parser.add_argument('--task', type=str, required=True, choices=['date', 'source', 'location', 'motivation'])
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    grouped = defaultdict(list)
    for item in raw_data:
        grouped[item["img_path"]].append(item)

    deduped = [group[0] for group in grouped.values()]

    stats, mismatches, answers = evaluate(deduped, args.task)
    print_stats(stats, args.task, answers)

    if mismatches:
        err_path = Path(args.input).with_name(f"mismatches_{args.task}.json")
        with open(err_path, "w", encoding="utf-8") as f:
            json.dump(mismatches, f, indent=2, ensure_ascii=False)
        print(f"\n⚠️  Mismatched samples saved to: {err_path}")

if __name__ == "__main__":
    main()
