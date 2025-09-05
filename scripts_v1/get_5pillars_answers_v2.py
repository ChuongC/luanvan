
import argparse
import openai
import os
import sys
import numpy as np

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from baseline.answer_generation_sample import run_model
from baseline.generation_utils import *
from baseline.llm_prompting_sample import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate 5 pillars answers with LLMs.')
    parser.add_argument('--openai_api_key', type=str, default='',
                        help='Your OpenAI API key.')
    parser.add_argument('--map_manipulated_original', type=str, default='dataset/map_manipulated_original.json',
                        help='Path to the file that maps manipulated images to their identified original version.')
    parser.add_argument('--results_file', type=str, default='output/results.json',
                        help='Path to store the predicted answers.')
    parser.add_argument('--task', type=str, default='source',
                        help='The task to perform. One of [source, date, location, motivation]')
    parser.add_argument('--modality', type=str, default='evidence',
                        help='Which input modality to use. One of [vision, evidence, multimodal]')
    parser.add_argument('--n_shots', type=int, default=0,
                        help='How many demonstrations to include.')
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                        help='Which LLM to use for generating answers.')
    parser.add_argument('--max_tokens', type=int, default=200,
                        help='The maximum number of tokens to generate as output.')
    parser.add_argument('--temperature', type=float, default=0.2,
                        help='The temperature of the model. Lower values make the output more deterministic.')
    parser.add_argument('--sleep', type=int, default=5,
                        help='The waiting time between two answer generation.')

    args = parser.parse_args()

    # Debug: In toàn bộ tham số
    print("\n[DEBUG] ====== Script Arguments ======")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("=====================================\n")

    openai.api_key = args.openai_api_key

    # Tạo thư mục output nếu chưa có
    if 'output' not in os.listdir():
        os.mkdir('output/')
        print("[DEBUG] Created output/ directory")

    # Load dữ liệu
    print(f"[DEBUG] Loading map_manipulated_original from: {args.map_manipulated_original}")
    map_manipulated = load_json(args.map_manipulated_original)
    print(f"[DEBUG] Loaded map_manipulated_original: {len(map_manipulated)} entries")

    print("[DEBUG] Loading dataset/train.json")
    train = load_json('dataset/train.json')
    print(f"[DEBUG] Loaded train set: {len(train)} samples")

    print("[DEBUG] Loading dataset/test.json")
    test = load_json('dataset/test.json')
    print(f"[DEBUG] Loaded test set: {len(test)} samples")

    task_test = test
    print(f"[DEBUG] Total test items after task filter: {len(task_test)}")

    # Lấy danh sách đường dẫn ảnh
    image_paths = [t['image_path'] for t in task_test]
    print(f"[DEBUG] Total image_paths: {len(image_paths)}")
    print(f"[DEBUG] First 3 image_paths: {image_paths[:3]}")

    # Lấy ground truth
    if args.task == 'date':
        ground_truth = [t['date_numeric_label'] for t in task_test]
    else:
        ground_truth = [t[args.task] for t in task_test]
    print(f"[DEBUG] Example ground truth labels: {ground_truth[:5]}")

    # Load embeddings và evidence
    print("[DEBUG] Loading embeddings and evidence...")
    clip_evidence_embeddings = np.load('dataset/embeddings/evidence_embeddings.npy')
    image_embeddings = np.load('dataset/embeddings/image_embeddings.npy')
    image_embeddings_map = load_json('dataset/embeddings/image_embeddings_map.json')
    evidence = load_json('dataset/retrieval_results/evidence.json')
    print(f"[DEBUG] Evidence loaded: {len(evidence)} entries")
    print(f"[DEBUG] Image embeddings shape: {image_embeddings.shape}")
    print(f"[DEBUG] Evidence embeddings shape: {clip_evidence_embeddings.shape}")

    evidence_idx = []
    if args.modality in ['evidence', 'multimodal']:
        print("[DEBUG] Computing top-k evidence for each test image...")
        for i in range(len(image_paths)):
            idx = get_topk_evidence(
                image_paths[i],
                evidence,
                image_embeddings,
                clip_evidence_embeddings,
                image_embeddings_map
            )
            print(f"[DEBUG] Evidence idx for image {os.path.basename(image_paths[i])}: {idx}")
            evidence_idx.append(idx)

    # Tạo tập demonstration candidates
    images_with_evidence = [os.path.basename(ev['image_path']) for ev in evidence]
    demonstration_candidates = [t for t in train if os.path.basename(t['image_path']) in images_with_evidence]
    print(f"[DEBUG] Demonstration candidates: {len(demonstration_candidates)} samples")

    demonstrations = []
    for i in range(len(image_paths)):
        if args.n_shots > 0:
            print(f"[DEBUG] Getting top-{args.n_shots} demonstrations for image {os.path.basename(image_paths[i])}")
            demonstrations_idx = get_topk_demonstrations(
                image_paths[i],
                args.task,
                demonstration_candidates,
                image_embeddings,
                image_embeddings_map,
                args.n_shots
            )
            print(f"[DEBUG] Demonstration idx: {demonstrations_idx}")
            instance_demonstrations = []
            for idx in demonstrations_idx:
                demo_image = demonstration_candidates[idx]['image_path']
                demo_answer = demonstration_candidates[idx][args.task]
                print(f"[DEBUG] Demo image: {demo_image}, Demo answer: {demo_answer}")

                demo_evidence_idx = get_topk_evidence(
                    demo_image,
                    evidence,
                    image_embeddings,
                    clip_evidence_embeddings,
                    image_embeddings_map
                )
                print(f"[DEBUG] Demo evidence idx: {demo_evidence_idx}")

                demo_image_basename = os.path.basename(demo_image)
                evidence_image_subset = [ev for ev in evidence if os.path.basename(ev['image_path']) == demo_image_basename]
                demo_evidence = [evidence_image_subset[j] for j in demo_evidence_idx if j < len(evidence_image_subset)]
                instance_demonstrations.append((demo_image, demo_answer, demo_evidence))
            demonstrations.append(instance_demonstrations)
        else:
            demonstrations.append([])

    print("[DEBUG] Starting run_model()...")
    print(f"[DEBUG] Model: {args.model}, Task: {args.task}, Modality: {args.modality}, N_shots: {args.n_shots}")

    run_model(
        image_paths=image_paths,
        task=args.task,
        ground_truth=ground_truth,
        results_json=args.results_file,
        map_manipulated=map_manipulated,
        modality=args.modality,
        model=args.model,
        evidence=evidence,
        evidence_idx=evidence_idx,
        demonstrations=demonstrations,
        client=None,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        sleep=args.sleep
    )
