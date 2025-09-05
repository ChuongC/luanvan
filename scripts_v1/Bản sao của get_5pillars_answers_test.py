import argparse
import openai
import os
import sys
import numpy as np

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from baseline.answer_generation_sample import run_model
from baseline.generation_utils import *
from baseline.llm_prompting_sample import *


def load_data():
    return {
        "map_manipulated": load_json('dataset/map_manipulated_original.json'),
        "train": load_json('dataset/train_custom.json'),
        "test": load_json('dataset/test_custom.json'),
        "clip_evidence_embeddings": np.load('dataset/embeddings/evidence_embeddings.npy'),
        "image_embeddings": np.load('dataset/embeddings/image_embeddings.npy'),
        "image_embeddings_map": load_json('dataset/embeddings/image_embeddings_map.json'),
        "evidence": load_json('dataset/retrieval_results/evidence.json')
    }


def get_ground_truth_for_task(test_data, task):
    if task == 'date':
        return [t['date_numeric_label'] for t in test_data]
    return [t[task] for t in test_data]


def prepare_evidence_indices(image_paths, modality, evidence, image_embeddings, clip_evidence_embeddings, image_embeddings_map):
    evidence_idx = []
    if modality in ['evidence', 'multimodal']:
        for img_path in image_paths:
            idx = get_topk_evidence(
                img_path,
                evidence,
                image_embeddings,
                clip_evidence_embeddings,
                image_embeddings_map
            )
            evidence_idx.append(idx)
    return evidence_idx


def prepare_demonstrations(image_paths, n_shots, task, train, evidence, image_embeddings, clip_evidence_embeddings, image_embeddings_map):
    images_with_evidence = {os.path.basename(ev['image_path']) for ev in evidence}
    demonstration_candidates = [t for t in train if os.path.basename(t['image_path']) in images_with_evidence]

    demonstrations = []
    for img_path in image_paths:
        if n_shots > 0:
            demonstrations_idx = get_topk_demonstrations(
                img_path,
                task,
                demonstration_candidates,
                image_embeddings,
                image_embeddings_map,
                n_shots
            )
            instance_demonstrations = []
            for idx in demonstrations_idx:
                demo_image = demonstration_candidates[idx]['image_path']
                demo_answer = demonstration_candidates[idx][task]
                demo_evidence_idx = get_topk_evidence(
                    demo_image,
                    evidence,
                    image_embeddings,
                    clip_evidence_embeddings,
                    image_embeddings_map
                )
                demo_image_basename = os.path.basename(demo_image)
                evidence_image_subset = [ev for ev in evidence if os.path.basename(ev['image_path']) == demo_image_basename]
                demo_evidence = [evidence_image_subset[j] for j in demo_evidence_idx if j < len(evidence_image_subset)]
                instance_demonstrations.append((demo_image, demo_answer, demo_evidence))
            demonstrations.append(instance_demonstrations)
        else:
            demonstrations.append([])
    return demonstrations


def main():
    parser = argparse.ArgumentParser(description='Generate 5 pillars answers with LLMs.')
    parser.add_argument('--openai_api_key', type=str, default='')
    parser.add_argument('--results_file', type=str, default='output/results.json')
    parser.add_argument('--task', type=str, default='source', help='One of [source, date, location, motivation, all]')
    parser.add_argument('--modality', type=str, default='vision')
    parser.add_argument('--n_shots', type=int, default=0)
    parser.add_argument('--model', type=str, default='gpt-4o-mini')
    parser.add_argument('--max_tokens', type=int, default=500)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--sleep', type=int, default=5)
    args = parser.parse_args()

    openai.api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY", "")
    if not openai.api_key:
        raise ValueError("OpenAI API key is required.")

    if 'output' not in os.listdir():
        os.mkdir('output/')

    tasks = [args.task] if args.task != 'all' else ['source', 'date', 'location', 'motivation']
    data = load_data()
    image_paths = [t['image_path'] for t in data['test']]

    for task in tasks:
        ground_truth = get_ground_truth_for_task(data['test'], task)
        evidence_idx = prepare_evidence_indices(
            image_paths,
            args.modality,
            data['evidence'],
            data['image_embeddings'],
            data['clip_evidence_embeddings'],
            data['image_embeddings_map']
        )
        demonstrations = prepare_demonstrations(
            image_paths,
            args.n_shots,
            task,
            data['train'],
            data['evidence'],
            data['image_embeddings'],
            data['clip_evidence_embeddings'],
            data['image_embeddings_map']
        )

        result_file = args.results_file.replace('.json', f'_{task}.json')
        run_model(
            image_paths=image_paths,
            task=task,
            ground_truth=ground_truth,
            results_json=result_file,
            map_manipulated=data['map_manipulated'],
            modality=args.modality,
            model=args.model,
            evidence=data['evidence'],
            evidence_idx=evidence_idx,
            demonstrations=demonstrations,
            client=None,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            sleep=args.sleep
        )

        if os.path.exists(result_file):
            results = load_json(result_file)
            print(f"\n===== Task: {task} =====")
            for r in results:
                print(r.get('answer', ''))


if __name__ == '__main__':
    main()
