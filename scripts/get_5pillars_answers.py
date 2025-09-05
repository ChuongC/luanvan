import argparse
import openai
import os
import sys
import numpy as np
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from baseline.answer_generation import *
from baseline.generation_utils import *
from baseline.llm_prompting import *

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Generate 5 pillars answers with LLMs.')
    parser.add_argument('--openai_api_key', type=str, default='',
                        help='Your OpenAI API key.')
    parser.add_argument('--map_manipulated_original', type=str, default='dataset/map_manipulated_original.json',
                        help='Path to the file that maps manipulated images to their identified original version.')
    parser.add_argument('--results_file', type=str, default='output/results.json',
                        help='Path to store the predicted answers.')
    parser.add_argument('--task', type=str, default='source',
                        help='The task to perform. One of [source, date, location, motivation]')
    parser.add_argument('--modality', type=str, default='multimodal',
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

    print(f"[INFO] ====== Starting get_5pillars_answers ======")
    print(f"[INFO] Arguments: {args}")

    openai.api_key = args.openai_api_key
    if not openai.api_key:
        print("[WARNING] No OpenAI API key provided!")

    if 'output' not in os.listdir():
        os.mkdir('output/')
        print("[INFO] Created 'output/' directory.")

  #  print(f"[INFO] Loading map_manipulated_original from: {args.map_manipulated_original}")
    map_manipulated = load_json(args.map_manipulated_original)
  #  print(f"[INFO] Loaded {len(map_manipulated)} entries from map_manipulated_original.")

    # Prepare data
    print("[INFO] Loading train.json and test.json ...")
    train = load_json('dataset/train.json')
    test = load_json('dataset/test.json')
   # print(f"[INFO] Train size: {len(train)}, Test size: {len(test)}")

    task_test = [t for t in test if t[args.task] != 'not enough information']
   # print(f"[INFO] Filtered {len(task_test)} test samples for task='{args.task}' (removed 'not enough information').")

    image_paths = [t['image_path'] for t in task_test]
    if args.task == 'date':
        ground_truth = [t['date_numeric_label'] for t in task_test]
    else:
        ground_truth = [t[args.task] for t in task_test]
  #  print(f"[DEBUG] First 3 image_paths: {image_paths[:3]}")
   # print(f"[DEBUG] First 3 ground_truth: {ground_truth[:3]}")

    # Load embeddings and evidence
    print("[INFO] Loading embeddings and evidence ...")
    clip_evidence_embeddings = np.load('dataset/embeddings/evidence_embeddings.npy')
    image_embeddings = np.load('dataset/embeddings/image_embeddings.npy')
    image_embeddings_map = load_json('dataset/embeddings/image_embeddings_map.json')
    evidence = load_json('dataset/retrieval_results/evidence.json')
   #print(f"[INFO] Evidence entries loaded: {len(evidence)}")
   #print(f"[INFO] image_embeddings shape: {image_embeddings.shape}, clip_evidence_embeddings shape: {clip_evidence_embeddings.shape}")
    if len(evidence) > 0:
        print(f"[DEBUG] sample evidence[0] keys: {list(evidence[0].keys())}")

    # Select evidence and demonstrations
    evidence_idx = []
    if args.modality in ['evidence', 'multimodal']:
       # print(f"[INFO] Selecting top-k evidence for each image (modality='{args.modality}')...")
        for i, img_path in enumerate(image_paths):
           # print(f"[DEBUG] Processing evidence for image {i+1}/{len(image_paths)}: {img_path}")
            idxs = get_topk_evidence(img_path, evidence, image_embeddings, clip_evidence_embeddings, image_embeddings_map)
           # print(f"[DEBUG] Got evidence global indices for image {img_path}: {idxs}")
            evidence_idx.append(idxs)
    else:
        evidence_idx = [[] for _ in image_paths]

    # Select demonstrations
    images_with_evidence = [ev['image_path'] for ev in evidence]
    demonstration_candidates = [t for t in train if t['image_path'] in images_with_evidence]
   # print(f"[INFO] Found {len(demonstration_candidates)} demonstration candidates.")

    demonstrations = []
    for i, img_path in enumerate(image_paths):
        if args.n_shots > 0:
          #  print(f"[INFO] Selecting demonstrations for image {i+1}/{len(image_paths)}...")
            demonstrations_idx = get_topk_demonstrations(
                img_path,
                args.task,
                demonstration_candidates,
                image_embeddings,
                image_embeddings_map,
                args.n_shots
            )
          #  print(f"[DEBUG] Demonstrations indices (into demonstration_candidates): {demonstrations_idx}")

            instance_demonstrations = []
            # --- IMPORTANT: demo_evidence_idx are global indices into evidence[] ---
            for idx in demonstrations_idx:
                demo_image = demonstration_candidates[idx]['image_path']
                demo_answer = demonstration_candidates[idx][args.task]
             #  print(f"[DEBUG] For demo candidate image: {demo_image}, answer: {demo_answer}")

                # Get global indices for evidence related to demo_image
                demo_evidence_global_idx = get_topk_evidence(demo_image, evidence, image_embeddings, clip_evidence_embeddings, image_embeddings_map)
               # print(f"[DEBUG] demo_evidence_global_idx (global indices in evidence list): {demo_evidence_global_idx}")

                # Select evidence items directly from global evidence list (avoid local-index confusion)
                demo_evidence = []
                for gidx in demo_evidence_global_idx:
                    if 0 <= gidx < len(evidence):
                        ev_item = evidence[gidx]
                        if os.path.basename(ev_item['image_path']) == os.path.basename(demo_image) or ev_item['image_path'] == demo_image:
                            demo_evidence.append(ev_item)
                        else:
                            print(f"[WARN] evidence[{gidx}] image_path mismatch for demo_image {demo_image}: {ev_item['image_path']}")
                    else:
                        print(f"[WARN] demo evidence global index {gidx} out of range (evidence len {len(evidence)})")

            #    print(f"[DEBUG] Selected {len(demo_evidence)} demo_evidence items for demo_image {demo_image}")
                instance_demonstrations.append((demo_image, demo_answer, demo_evidence))
            demonstrations.append(instance_demonstrations)
        else:
            demonstrations.append([])

    # Run the main loop
    print("[INFO] Running model inference...")
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
        client=openai,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        sleep=args.sleep
    )
    print(f"[INFO] Finished! Results saved to {args.results_file}")
