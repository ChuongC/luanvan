import os
import sys
import json
import numpy as np
import gradio as gr
import openai

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from baseline.answer_generation_sample import run_model
from baseline.generation_utils import *
from baseline.llm_prompting_sample import *

# ==== CONFIG ====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # thay key của bạn
MODEL_NAME = "gpt-4o-mini"
MAX_TOKENS = 1000
TEMPERATURE = 0.2
SLEEP_TIME = 5
N_SHOTS = 3

# ==== LOAD DATA ONCE ====
print("[INFO] Loading datasets and embeddings...")
map_manipulated = load_json('dataset/map_manipulated_original.json')
train_data = load_json('dataset/train_custom.json')
test_data = load_json('dataset/test_custom.json')
all_data = train_data + test_data

clip_evidence_embeddings = np.load('dataset/embeddings/evidence_embeddings.npy')
image_embeddings = np.load('dataset/embeddings/image_embeddings.npy')
image_embeddings_map = load_json('dataset/embeddings/image_embeddings_map.json')
evidence = load_json('dataset/retrieval_results/evidence.json')

images_with_evidence = [os.path.basename(ev['image_path']) for ev in evidence]
demonstration_candidates = [
    t for t in train_data if os.path.basename(t['image_path']) in images_with_evidence
]
print("[INFO] Data loaded successfully!")

# ==== MAIN PROCESS FUNCTION ====
def process_image(api_key, image_file, task, modality):
    if not api_key:
        return "❌ Vui lòng nhập OpenAI API key", None
    openai.api_key = api_key

    # Lấy basename của file upload từ Gradio
    image_basename = os.path.basename(image_file)

    # Tìm đúng item trong dataset dựa trên endswith basename
    item = next((t for t in all_data if t['image_path'].endswith(image_basename)), None)
    if not item:
        return f"❌ Không tìm thấy {image_basename} trong dataset!", None

    # Luôn dùng path gốc từ dataset, không dùng file tạm Gradio
    image_paths = [item['image_path']]

    # Lấy ground truth
    if task == 'date':
        ground_truth = [item.get('date_numeric_label', "")]
    else:
        ground_truth = [item.get(task, "")]

    # Evidence retrieval with reranking
    evidence_idx = []
    idx = get_topk_evidence(
        image_paths[0],
        evidence,
        image_embeddings,
        clip_evidence_embeddings,
        image_embeddings_map,
        alpha=0.7
    )
    evidence_idx.append(idx)

    # Demonstrations
    demonstrations = []
    if N_SHOTS > 0:
        demonstrations_idx = get_topk_demonstrations(
            image_paths[0],
            task,
            demonstration_candidates,
            image_embeddings,
            image_embeddings_map,
            N_SHOTS
        )
        instance_demonstrations = []
        for di in demonstrations_idx:
            demo_image = demonstration_candidates[di]['image_path']
            demo_answer = demonstration_candidates[di][task]

            demo_evidence_idx = get_topk_evidence(
                demo_image,
                evidence,
                image_embeddings,
                clip_evidence_embeddings,
                image_embeddings_map,
                alpha=0.7
            )

            demo_image_basename = os.path.basename(demo_image)
            evidence_image_subset = [
                ev for ev in evidence if os.path.basename(ev['image_path']) == demo_image_basename
            ]
            demo_evidence = [
                evidence_image_subset[j]
                for j in demo_evidence_idx if j < len(evidence_image_subset)
            ]
            print(f"[DEBUG] Demo image: {demo_image} has {len(demo_evidence)} evidence items")
            instance_demonstrations.append((demo_image, demo_answer, demo_evidence))
        demonstrations.append(instance_demonstrations)
    else:
        demonstrations.append([])

    results_file = "output/gradio_result.json"

    run_model(
        image_paths=image_paths,
        task=task,
        ground_truth=ground_truth,
        results_json=results_file,
        map_manipulated=map_manipulated,
        modality=modality,
        model=MODEL_NAME,
        evidence=evidence,
        evidence_idx=evidence_idx,
        demonstrations=demonstrations,
        client=None,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        sleep=SLEEP_TIME
    )

    # Xử lý kết quả
    if os.path.exists(results_file):
        with open(results_file, 'r', encoding='utf-8') as f:
            res = json.load(f)
        from collections import defaultdict

        if res and len(res) > 0:
            grouped = defaultdict(list)
            for r in res:
                img = r.get("img_path", "unknown")
                gt = r.get("ground_truth", "").strip()
                pred = r.get("pred") or r.get("output", "No prediction")
                grouped[img].append((gt, pred))

            lines = []
            for img, vals in grouped.items():
                lines.append(f"🖼 {img}")
                for i, (gt, pred) in enumerate(vals, start=1):
                    lines.append(f"  Pillar {i}: GT → {gt} | Pred → {pred}")

            pred_text = "\n".join(lines)
            return f"✅ Predictions:\n{pred_text}", results_file
        else:
            return "⚠️ No predictions found", results_file

    return "❌ Không có kết quả", None


# ==== GRADIO UI ====
with gr.Blocks() as demo:
    gr.Markdown("## 🖼️ 5Pillars Prediction Gradio App")

    with gr.Row():
        api_key_input = gr.Textbox(label="OpenAI API Key", type="password", value=OPENAI_API_KEY)
        task_dropdown = gr.Dropdown(
            choices=["source", "date", "location", "motivation"],
            value="source", label="task"
        )
        modality_dropdown = gr.Dropdown(
            choices=["vision", "evidence", "multimodal"],
            value="evidence", label="modality"
        )

    image_input = gr.Image(type="filepath", label="Upload Image from dataset")
    output_text = gr.Textbox(label="Prediction")
    output_file = gr.File(label="Download Raw JSON", file_types=[".json"])

    run_btn = gr.Button("Generate")

    run_btn.click(
        fn=process_image,
        inputs=[api_key_input, image_input, task_dropdown, modality_dropdown],
        outputs=[output_text, output_file]
    )

if __name__ == "__main__":
    demo.launch(debug=True, share=True)
