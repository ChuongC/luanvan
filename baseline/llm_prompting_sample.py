from PIL import Image
from baseline.generation_utils import *
import os
import openai
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils import *

#############
#  CLEANING #
#############

def format_demonstrations(demos, task_name: str, question_dict: dict) -> str:
    print(f"[DEBUG] format_demonstrations() called with {len(demos)} demos, task_name={task_name}")
    formatted = []
    for i, (img_path, answer, evidence_list) in enumerate(demos):
        print(f"[DEBUG] Demo {i+1} image_path={img_path}, answer={answer}, evidence_count={len(evidence_list)}")
        # Nếu có 'text' thì dùng, không thì gộp 'title' + 'content'
        evidence_text = "\n".join(
            ev.get('text') or f"{ev.get('title', '')}\n{ev.get('content', '')}"
            for ev in evidence_list
        )
        block = f"""[Example {i + 1}]
Image Path: {img_path}
Evidence: {evidence_text.strip()}
Question: {question_dict[task_name]}
Answer: {answer}
"""
        formatted.append(block.strip())
    print(f"[DEBUG] format_demonstrations() output length={len(formatted)} blocks")
    return "\n\n".join(formatted)


#############
#  EVIDENCE #
#############

import base64
import traceback

def call_openai_gpt4o(prompt: str, image_path: str = None, temperature=0.2, max_tokens=512) -> str:
    """
    Call OpenAI GPT-4o (vision + text) via OpenAI API.
    Returns model response or error message.
    """
    try:
        print(f"[DEBUG] call_openai_gpt4o() prompt length={len(prompt)}, temperature={temperature}, max_tokens={max_tokens}")
        print("[DEBUG] Prompt preview:", prompt[:100].replace("\n", " "), "...")
        print("[DEBUG] Image path:", image_path)
        print("[DEBUG] openai version:", openai.__version__)
        api_key = os.getenv("OPENAI_API_KEY")
        print("[DEBUG] API key prefix:", api_key[:10] + "..." if api_key else "None")

        content = [{"type": "text", "text": prompt}]
        if image_path:
            with open(image_path, "rb") as img_file:
                b64_image = base64.b64encode(img_file.read()).decode("utf-8")
            print(f"[DEBUG] Encoded image length: {len(b64_image)} characters")
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}})

        messages = [{"role": "user", "content": content}]
        print(f"[DEBUG] Sending {len(messages)} messages to model")

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini-2024-07-18",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        print(f"[DEBUG] API response keys: {list(response.keys())}")
        if response and hasattr(response, "choices") and len(response.choices) > 0:
            message = response.choices[0].message
            if message and hasattr(message, "content"):
                output = message.content.strip()
                print(f"[DEBUG] Model output length={len(output)}")
                return output
            else:
                print("[ERROR] No content in response message.")
                return "[Error] No content in response message."
        else:
            print("[ERROR] No choices returned by model.")
            return "[Error] No choices returned by model."

    except openai.OpenAIError as oe:
        print("[DEBUG] OpenAI API Error:")
        traceback.print_exc()
        return f"[Error] OpenAI API error: {oe}"

    except Exception as e:
        print("[DEBUG] Unexpected Error in call_openai_gpt4o():")
        traceback.print_exc()
        return f"[Error] {e}"


def gpt4_prompting(content, client, max_tokens=1000):
    deployment_name = 'gpt-4o-mini'
    print(f"[DEBUG] gpt4_prompting() called, content length={len(content)}, max_tokens={max_tokens}")
    messages = [{"role": "user", 'content': content}]
    try:
        completion = openai.ChatCompletion.create(
            model=deployment_name,
            messages=messages,
            max_tokens=max_tokens
        )
        output = completion.choices[0].message.content
        usage = completion.usage.total_tokens
        print(f"[DEBUG] gpt4_prompting() output length={len(output)}, tokens used={usage}")
        return output, usage
    except Exception as e:
        print(f"[ERROR] gpt4_prompting() failed: {e}")
        traceback.print_exc()
        return "[Error] GPT call failed", 0


def gpt4_vision_prompting(prompt, client, image_path, map_manipulated_original={}, modality='vision',
                          temperature=0.2, max_tokens=1000):
    deployment_name = 'gpt-4o-mini'
    sanitized_prompt = sanitize_prompt(prompt)
    print(f"[DEBUG] gpt4_vision_prompting() called, modality={modality}, prompt length={len(sanitized_prompt)}")
    content = [{"type": "text", "text": sanitized_prompt}]

    if modality in ['vision', 'multimodal']:
        basename = os.path.basename(image_path)
        print(f"[DEBUG] Original basename: {basename}")
        if basename in map_manipulated_original:
            print(f"[DEBUG] Image {basename} replaced with original: {map_manipulated_original[basename]}")
            image_path = map_manipulated_original[basename]

        image64 = encode_image(image_path)
        print(f"[DEBUG] Encoded image64 length={len(image64)}")
        content += [{"type": "image_url", "image_url": {"url": image64}}]

    # Log prompt ra file
    with open("debug_prompt.txt", "a", encoding="utf-8") as f:
        f.write("==== PROMPT START ====\n")
        f.write(sanitized_prompt + "\n")
        f.write("==== PROMPT END ======\n\n")
    print(f"[DEBUG] Prompt saved to debug_prompt.txt (length={len(sanitized_prompt)})")

    messages = [{"role": "user", "content": content}]
    try:
        completion = openai.ChatCompletion.create(
            model=deployment_name,
            temperature=temperature,
            messages=messages,
            max_tokens=max_tokens
        )
        output = completion.choices[0].message.content
        print(f"[DEBUG] gpt4_vision_prompting() output length={len(output)}")
    except Exception as e:
        print(f"[ERROR] GPT Call failed: {e}")
        traceback.print_exc()
        output = 'Content Filtering error'
    return output


def assemble_prompt_gpt4(
    question,
    answer=None,
    evidence=[],
    demonstrations=None,
    modality='vision'
):
    print(f"[DEBUG] assemble_prompt_gpt4() called with modality={modality}, question length={len(question)}, "
          f"evidence_count={len(evidence)}, has_answer={answer is not None}, demonstrations_count={len(demonstrations) if demonstrations else 0}")
    prompt = ""

    # 1️⃣ Add demonstrations
    if demonstrations and isinstance(demonstrations, list) and len(demonstrations) > 0:
        for demo_group in demonstrations:
            for demo_image, demo_answer, demo_evidence in demo_group:
                print(f"[DEBUG] Adding demonstration: image={demo_image}, answer={demo_answer}, evidence_count={len(demo_evidence)}")
                prompt += "=== Ví dụ ===\n"
                prompt += get_evidence_prompt(demo_evidence)
                prompt += f"Question: {question}\n"
                prompt += f"Answer: {demo_answer}\n\n"
        prompt += "\n"

    # 2️⃣ Add instruction
    if modality == 'evidence':
        prompt += "Bạn được cung cấp các bài viết trực tuyến sử dụng cùng một hình ảnh. "
        prompt += "Nhiệm vụ của bạn là trả lời câu hỏi về hình ảnh dựa trên các bài viết này.\n\n"
    elif modality == 'multimodal':
        prompt += "Bạn được cung cấp một hình ảnh và các bài viết trực tuyến sử dụng hình ảnh đó. "
        prompt += "Nhiệm vụ của bạn là trả lời câu hỏi về hình ảnh bằng cách sử dụng cả hình ảnh và các bài viết.\n\n"
    else:  # vision
        prompt += "Bạn được cung cấp một hình ảnh. "
        prompt += "Nhiệm vụ của bạn là trả lời câu hỏi về hình ảnh dựa trên các thông tin có sẵn.\n\n"

    # 3️⃣ Add evidence
    if evidence:
        prompt += get_evidence_prompt(evidence)

    # 4️⃣ Add question & answer
    prompt += f"Question: {question.strip()}\n"
    prompt += "Answer: "
    if answer:
        prompt += f"{answer}\n"

    print(f"[DEBUG] Final assembled prompt length={len(prompt)}")
    return prompt

#############
#   Llava   #
#############

def llava_prompting(prompt,
                    image_path,
                    pipe,
                    map_manipulated_original={},
                    temperature=0.2,
                    max_tokens=200):
    try:
        basename = os.path.basename(image_path)
        if basename in map_manipulated_original:
            image_path = map_manipulated_original[basename]
        image = Image.open(image_path)
        outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": max_tokens,
                                                               "temperature": temperature,
                                                               "do_sample": True})
        return outputs[0]['generated_text'].split('ASSISTANT:')[1].split('\n\n')[0]
    except RuntimeError as e:
        print(e)
        return ''

def assemble_prompt_llava(question,
                          answer=None,
                          evidence=[],
                          demonstrations=[],
                          modality='vision'):
    prompt = ''
    for d in range(len(demonstrations)):
        prompt += assemble_prompt_llava(question, answer=demonstrations[d][1],
                                        evidence=demonstrations[d][2], demonstrations=[], modality='evidence')
        prompt += '\n\n'
    prompt += 'USER:'
    if modality == 'evidence':
        prompt += 'You are given online articles that used a certain image. Your task is to answer a question about the image.\n\n'
    elif modality == 'multimodal':
        prompt += 'You are given an image and online articles that used that image. Your task is to answer a question about the image using the image and the articles.\n\n'
        prompt += '<image>'
    else:
        prompt += 'You are given an image. Your task is to answer a question about the image.\n\n'
        prompt += '<image>'
    if len(evidence) != 0:
        prompt += get_evidence_prompt(evidence)
    prompt += 'Question: ' + question + '\n'
    if answer:
        prompt += 'Answer:'
        prompt += answer + '\n'
    else:
        prompt += 'ASSISTANT:'
    return prompt

#############
#   Llama   #
#############

def llama_prompting(prompt,
                    pipeline,
                    tokenizer,
                    temperature,
                    max_tokens):
    output = pipeline(prompt,
                      eos_token_id=tokenizer.eos_token_id,
                      max_new_tokens=max_tokens,
                      temperature=temperature,
                      do_sample=True)[0]['generated_text']
    if '[/INST]' in output:
        output = output.split('[/INST]')[1]
    return output

def assemble_prompt_llama(question,
                          answer=None,
                          evidence=[],
                          demonstrations=[],
                          modality='evidence'):
    prompt = '<s>[INST] <<SYS>>'
    prompt += 'You are given online articles that used a certain image. Your task is to answer a question about the image.<</SYS>>\n'
    if len(evidence) != 0:
        prompt += get_evidence_prompt(evidence)
    prompt += 'Question: ' + question + '\n'
    prompt += '[/INST]'
    return prompt
