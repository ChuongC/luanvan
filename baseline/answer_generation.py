import time
from tqdm import tqdm
import os
from transformers import BitsAndBytesConfig, pipeline, AutoTokenizer
import torch
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from baseline.generation_utils import *
from baseline.llm_prompting import *


def run_model(image_paths, 
              task, 
              ground_truth, 
              results_json, 
              map_manipulated, 
              modality, 
              model,  
              evidence=[],
              evidence_idx=[], 
              demonstrations=[], 
              client=None,
              max_tokens=50, 
              temperature=0.2, 
              sleep=5):
    """
    Main loop to perform question answering with LLMs.
    """
    # print("\n[INFO] ==== MODEL EXECUTION STARTED ====")
    # print(f"[INFO] Model: {model}")
    # print(f"[INFO] Task: {task}")
    # print(f"[INFO] Modality: {modality}")
    # print(f"[INFO] Total images: {len(image_paths)}")
    # print(f"[INFO] Output file: {results_json}")
    # print("=========================================\n")

    if model == 'llava':
        print("[DEBUG] Initializing LLaVA pipeline...")
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model_id = "llava-hf/llava-1.5-7b-hf"
        pipe = pipeline("image-to-text", 
                        model=model_id, 
                        model_kwargs={"quantization_config": quantization_config})
        print("[DEBUG] LLaVA pipeline loaded successfully.")

    if model == 'llama':
        print("[DEBUG] Initializing LLaMA pipeline...")
        model_id = "meta-llama/Llama-2-7b-chat-hf"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pipe = pipeline("text-generation",
                        model=model_id, 
                        torch_dtype=torch.float16,
                        device_map="auto")
        print("[DEBUG] LLaMA pipeline loaded successfully.")

    # Select the prompt assembler
    prompt_assembler_dict = {
        'gpt-4o-mini-2024-07-18': assemble_prompt_gpt4,
        'llava': assemble_prompt_llava,
        'llama': assemble_prompt_llama
    }
    assembler = prompt_assembler_dict[model]

    questions = {
        'source': 'Who is the source/author of this image? Answer only with one or more persons or entities in a few words.',
        'date': 'When was this image taken? Answer only with one or more dates in a few words.',
        'location': 'Where was this image taken? Answer only with one or more locations in a few words.',
        'motivation': 'Why was this image taken? Answer in a few words.'
    }
    
    question = questions[task]

    # Main loop
    for i in tqdm(range(len(image_paths))):
        print(f"\n[INFO] Processing image {i+1}/{len(image_paths)}")
        print(f"[DEBUG] Original image path: {image_paths[i]}")

        real_image_path = map_manipulated.get(image_paths[i], image_paths[i])
        print(f"[DEBUG] Mapped image path (if manipulated): {real_image_path}")

        if not os.path.exists(real_image_path):
            print(f"[WARN] Image file not found: {real_image_path}")
            continue

        # --- Select evidence ---
        evidence_image_subset = [ev for ev in evidence if ev['image_path'] == image_paths[i]]
        print(f"[DEBUG] Found {len(evidence_image_subset)} evidence items for this image.")

        evidence_selection = []
        for idx in evidence_idx[i]:
            if idx < len(evidence_image_subset):
                evidence_selection.append(evidence_image_subset[idx])
            else:
                print(f"[WARN] Evidence index {idx} out of range for image {image_paths[i]}")

        print(f"[DEBUG] Selected {len(evidence_selection)} evidence for prompt.")
        for ev_num, ev in enumerate(evidence_selection):
            print(f"    Evidence {ev_num}: {ev.get('url', 'No URL')} | {ev.get('title', 'No Title')}")

        # --- Assemble prompt ---
        if len(demonstrations[i]) != 0:
            print("[DEBUG] Assembling prompt with demonstrations...")
            prompt = assembler(
                question,
                evidence=evidence_selection if modality != 'vision' else None,
                modality=modality,
                demonstrations=demonstrations[i]
            )
        else:
            print("[DEBUG] Assembling prompt without demonstrations...")
            prompt = assembler(
                question,
                evidence=evidence_selection if modality != 'vision' else None,
                modality=modality
            )

        print(f"[PROMPT] ===== Prompt for image {i+1} =====\n{prompt}\n========================\n")

        # --- Call model ---
        try:
            if model == 'gpt-4o-mini-2024-07-18':
                print("[DEBUG] Calling GPT-4o-mini...")
                output = gpt4_vision_prompting(
                    prompt, client, real_image_path,
                    modality=modality,
                    map_manipulated_original=map_manipulated,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            elif model == 'llava':
                print("[DEBUG] Calling LLaVA...")
                output = llava_prompting(
                    prompt, real_image_path, pipe, map_manipulated,
                    temperature, max_tokens
                )
            elif model == 'llama':
                print("[DEBUG] Calling LLaMA...")
                output = llama_prompting(
                    prompt, pipe, tokenizer, temperature,
                    max_tokens, image_path=real_image_path
                )
            else:
                print(f"[ERROR] Wrong model provided: {model}")
                break

            print(f"[OUTPUT] Model output: {output}")

        except Exception as e:
            print(f"[ERROR] Model call failed for {real_image_path}: {e}")
            output = ""

        # --- Save result ---
        print("[DEBUG] Saving result to JSON...")
        save_result({
            'img_path': image_paths[i],
            'ground_truth': ground_truth[i],
            'output': output
        }, results_json)
        print(f"[DEBUG] Result saved for image {i+1}/{len(image_paths)}.")

        print(f"[DEBUG] Sleeping for {sleep} seconds to avoid rate limit...")
        time.sleep(sleep)

    print("\n[INFO] ==== MODEL EXECUTION FINISHED ====\n")
