import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import ViTImageProcessor, ViTForImageClassification, TrainingArguments, Trainer, pipeline
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from PIL import Image
from datasets import Dataset
import argparse
import os
import sys
import json
from collections import Counter

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils import *

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    refs = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(refs, preds, average='binary')
    accuracy = accuracy_score(refs, preds)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def load_json(json_path):
    print(f"\n[INFO] Loading data from: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def normalize_label(raw_label):
    if not isinstance(raw_label, str):
        return None
    label = raw_label.strip().lower()
    if label == "manipulated":
        return "manipulated"
    elif label == "non-manipulated":
        return "non-manipulated"
    return None

def prepare_dataset(json_data):
    image_paths, labels = [], []
    seen = set()
    for im in json_data:
        label = normalize_label(im.get("type_of_image", ""))
        if label is not None:
            path = im["image_path"]
            if path not in seen:
                image_paths.append(path)
                labels.append(label)
                seen.add(path)
    print(f"[INFO] Prepared dataset with {len(labels)} unique samples")
    return Dataset.from_dict({"image_path": image_paths, "labels": labels})

def print_dataset_info(name, json_data):
    image_paths = [im['image_path'] for im in json_data if 'image_path' in im]
    print(f"[INFO] {name} dataset: total samples = {len(image_paths)}, unique images = {len(set(image_paths))}")

def print_label_distribution(name, dataset):
    counter = Counter(dataset["labels"])
    print(f"[INFO] Label distribution ({name}):", dict(counter))

def transform(example_batch):
    image_list = [Image.open(p).convert("RGB") for p in example_batch["image_path"]]
    processed = processor(image_list, return_tensors="pt")
    processed["labels"] = example_batch["labels"]
    return processed

def collate_fn(batch):
    label_map = {'manipulated': 1, 'non-manipulated': 0}
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([label_map[x['labels'].strip().lower()] for x in batch])
    }


def train_manipulation_detector(prepared_train_dataset,
                                prepared_val_dataset,
                                model_name_or_path,
                                save_folder,
                                epochs,
                                learning_rate):
    print("\n[INFO] Starting training...")
    print(f"[INFO] Model: {model_name_or_path} | Epochs: {epochs} | LR: {learning_rate}")
    labels = ["non-manipulated", "manipulated"]
    model = ViTForImageClassification.from_pretrained(
        model_name_or_path,
        num_labels=len(labels),
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)}
    )
    training_args = TrainingArguments(
        output_dir=save_folder,
        per_device_train_batch_size=16,
        evaluation_strategy="steps",
        num_train_epochs=epochs,
        fp16=torch.cuda.is_available(),
        save_steps=500,
        eval_steps=50,
        logging_steps=500,
        learning_rate=learning_rate,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to='tensorboard',
        load_best_model_at_end=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=prepared_train_dataset,
        eval_dataset=prepared_val_dataset,
        tokenizer=processor,
    )
    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()
    print("[INFO] Training completed. Metrics:", train_results.metrics)
    with open(os.path.join(save_folder, "metrics_train.json"), "w") as f:
        json.dump(train_results.metrics, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify images as manipulated or not with a ViT model.')
    parser.add_argument('--model_name', type=str, default='google/vit-base-patch16-224-in21k')
    parser.add_argument('--train', type=int, default=0)
    parser.add_argument('--model_folder', type=str, default='baseline/vit-manipulation')
    parser.add_argument('--json_path', type=str, default='dataset/manipulation_detection_test.json')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_raw = load_json('dataset/train.json')
    val_raw = load_json('dataset/val.json')
    test_raw = load_json('dataset/test.json')
    
    print_dataset_info("Train", train_raw)
    print_dataset_info("Val", val_raw)
    print_dataset_info("Test", test_raw)

    train_dataset = prepare_dataset(train_raw)
    val_dataset = prepare_dataset(val_raw)
    test_dataset = prepare_dataset(test_raw)

    print_label_distribution("train", train_dataset)
    print_label_distribution("val", val_dataset)
    print_label_distribution("test", test_dataset)

    processor = ViTImageProcessor.from_pretrained(args.model_name)
    prepared_train_dataset = train_dataset.with_transform(transform)
    prepared_val_dataset = val_dataset.with_transform(transform)
    prepared_test_dataset = test_dataset.with_transform(transform)

    if args.train:
        train_manipulation_detector(prepared_train_dataset, prepared_val_dataset,
                                    args.model_name, args.model_folder,
                                    args.epochs, args.learning_rate)

    print("\n[INFO] Loading model for prediction from:", args.model_folder)
    model = ViTForImageClassification.from_pretrained(args.model_folder)
    test_loader = DataLoader(prepared_test_dataset, batch_size=32)

    print("[INFO] Running inference on test dataset...")
    test_predictions = []
    with torch.no_grad():
        for batch in test_loader:
            pixel_values = batch['pixel_values']
            outputs = model(pixel_values)
            preds = torch.argmax(outputs.logits, dim=1)
            test_predictions.extend(preds.tolist())

    test_predictions = ['non-manipulated' if p == 0 else 'manipulated' for p in test_predictions]
    unique_results = {}
    for i in range(len(test_predictions)):
        path = test_dataset[i]['image_path']
        label = test_predictions[i]
        unique_results[path] = label  # nếu trùng sẽ ghi đè, giữ nhãn mới nhất

    results = [{'image_path': k, 'manipulation_detection': v} for k, v in unique_results.items()]

    save_result(results, args.json_path)
    print(f"[INFO] Saved {len(results)} unique predictions to {args.json_path}")



    summary = Counter(test_predictions)
    print("\n[INFO] Prediction summary:", dict(summary))
    with open("results_summary.json", "w", encoding='utf-8') as f:
        json.dump({"summary": dict(summary), "total": len(test_predictions)}, f, indent=2, ensure_ascii='False')
