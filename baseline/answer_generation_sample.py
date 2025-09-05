
import time
from tqdm import tqdm
import os
from transformers import BitsAndBytesConfig, pipeline, AutoTokenizer
import torch
import sys
import json
import openai
import datetime

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from baseline.generation_utils import *
from baseline.llm_prompting_sample import *


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
              max_tokens=1000,
              temperature=0.2,
              sleep=5):

    print("[DEBUG] ====== START run_model ======")
    print(f"[DEBUG] image_paths length      : {len(image_paths)}")
    print(f"[DEBUG] ground_truth length     : {len(ground_truth)}")
    print(f"[DEBUG] results_json path       : {results_json}")
    print(f"[DEBUG] map_manipulated size    : {len(map_manipulated)}")
    print(f"[DEBUG] modality                : {modality}")
    print(f"[DEBUG] model                   : {model}")
    print(f"[DEBUG] evidence length         : {len(evidence)}")
    print(f"[DEBUG] evidence_idx length     : {len(evidence_idx)}")
    print(f"[DEBUG] demonstrations length   : {len(demonstrations)}")
    print(f"[DEBUG] max_tokens              : {max_tokens}")
    print(f"[DEBUG] temperature             : {temperature}")
    print(f"[DEBUG] sleep                   : {sleep}")

    # Khởi tạo pipeline theo model
    if model == 'llava':
        print("[DEBUG] Initializing LLaVA pipeline...")
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model_id = "llava-hf/llava-1.5-7b-hf"
        pipe = pipeline("image-to-text",
                        model=model_id,
                        model_kwargs={"quantization_config": quantization_config})
        print("[DEBUG] LLaVA pipeline initialized.")

    elif model == 'llama':
        print("[DEBUG] Initializing LLaMA pipeline...")
        model_id = "meta-llama/Llama-2-7b-chat-hf"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pipe = pipeline("text-generation",
                        model=model_id,
                        torch_dtype=torch.float16,
                        device_map="auto")
        print("[DEBUG] LLaMA pipeline initialized.")

    else:
        print(f"[DEBUG] Model '{model}' will use OpenAI API prompting.")

    # Map assembler
    prompt_assembler_dict = {
        'gpt-4o-mini': assemble_prompt_gpt4,
        'llava': assemble_prompt_llava,
        'llama': assemble_prompt_llama
    }
    assembler = prompt_assembler_dict[model]
    print(f"[DEBUG] Assembler function: {assembler.__name__}")

    # Câu hỏi cho từng task
    questions = {
        'source': "...",
        'date': "...",
        'location': "...",
        'motivation': "..."
    }
    question = questions[task]
    print(f"[DEBUG] Task: {task}")
    print(f"[DEBUG] Question: {question}")

    # Chuẩn bị debug log
    os.makedirs("output", exist_ok=True)
    debug_log = open("output/debug_prompt.txt", "a", encoding="utf-8")
    debug_path = os.path.join(os.getcwd(), "debug_evidence.jsonl")
    open(debug_path, "w", encoding="utf-8").close()
    print(f"[DEBUG] Initialized debug files: debug_prompt.txt & {debug_path}")

    # Giới hạn số mẫu
    total = min(len(image_paths), len(evidence_idx), len(demonstrations))
    print(f"[DEBUG] Total samples to process: {total}")

    for i in tqdm(range(total)):
        print(f"\n[DEBUG] === Processing sample {i+1}/{total} ===")
        print(f"[DEBUG] Image path: {image_paths[i]}")
        print(f"[DEBUG] Ground truth: {ground_truth[i] if i < len(ground_truth) else '[Missing]'}")
        print(f"[DEBUG] Evidence idx list: {evidence_idx[i] if i < len(evidence_idx) else '[Missing]'}")
        print(f"[DEBUG] Demonstrations for sample: {len(demonstrations[i]) if i < len(demonstrations) else 0}")

        if i >= len(evidence_idx) or i >= len(demonstrations):
            print(f"[ERROR] Index {i} out of range for evidence_idx or demonstrations")
            continue

        # Chọn evidence nếu có
        if modality in ['evidence', 'multimodal']:
            evidence_selection = []
            if len(evidence_idx[i]) != 0:
                print(f"[DEBUG] evidence_idx[{i}] = {evidence_idx[i]}")
                evidence_selection = [evidence[idx] for idx in evidence_idx[i]]
                print(f"[DEBUG] Selected {len(evidence_selection)} evidence entries")
            else:
                print(f"[WARN] No evidence index found for image {image_paths[i]}")

            # Ghi log evidence
            with open(debug_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "img_path": image_paths[i],
                    "evidence_idx": evidence_idx[i],
                    "evidence_selection": evidence_selection
                }, ensure_ascii=False) + "\n")

            demo_prompt_str = ""
            if len(demonstrations[i]) != 0:
                print(f"[DEBUG] Formatting demonstrations for sample {i}")
                demo_prompt_str = format_demonstrations(demonstrations[i], task, questions)
                print(f"[DEBUG] Demonstrations string length: {len(demo_prompt_str)}")

            prompt = assembler(
                question,
                evidence=evidence_selection if modality in ['evidence', 'multimodal'] else None,
                modality=modality,
                demonstrations=demo_prompt_str if demo_prompt_str else None
            )
            print(f"[DEBUG] Prompt length: {len(prompt)}")

        else:  # vision-only
            demo_prompt_str = ""
            if demonstrations[i]:
                print(f"[DEBUG] Formatting vision-only demonstrations for sample {i}")
                demo_prompt_str = format_demonstrations(demonstrations[i], task, questions)
                print(f"[DEBUG] Demonstrations string length: {len(demo_prompt_str)}")

            prompt = assembler(question, modality='vision', demonstrations=demo_prompt_str if demo_prompt_str else None)
            print(f"[DEBUG] Prompt length: {len(prompt)}")

        # Logging the prompt
        debug_log.write("========== {} ==========\n".format(datetime.datetime.now()))
        debug_log.write(f"Image path: {image_paths[i]}\n")
        debug_log.write(f"Prompt:\n{prompt}\n\n")
        debug_log.flush()

        # Run model
        print(f"[DEBUG] Running model '{model}' for sample {i}")
        if modality == 'evidence' and not evidence_idx[i]:
            print(f"[WARN] No evidence for evidence-only modality, skipping model call.")
            output = ''
        else:
            if model == 'gpt-4o-mini':
                output = gpt4_vision_prompting(
                    prompt,
                    client,
                    map_manipulated.get(image_paths[i], image_paths[i]),
                    map_manipulated,
                    modality=modality,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            elif model == 'llava':
                output = llava_prompting(
                    prompt,
                    image_paths[i],
                    pipe,
                    map_manipulated,
                    temperature,
                    max_tokens
                )
            elif model == 'llama':
                output = llama_prompting(
                    prompt,
                    pipe,
                    tokenizer,
                    temperature,
                    max_tokens
                )
            else:
                print('[ERROR] Wrong model provided')
                continue

        print(f"[DEBUG] Model output type: {type(output)}")
        if isinstance(output, str):
            print(f"[DEBUG] Output length: {len(output)}")
            data = {
                'img_path': image_paths[i],
                'ground_truth': ground_truth[i],
                'output': output
            }
            print(f"[DEBUG] Saving result for image: {image_paths[i]}")
            save_result(data, results_json)
            time.sleep(sleep)
        else:
            print(f"[ERROR] Output is not string: {output}")

    debug_log.close()
    print("[DEBUG] ====== END run_model ======")
