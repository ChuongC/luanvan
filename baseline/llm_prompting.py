from PIL import Image
from baseline.generation_utils import *
import os
import sys 
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils import *

import openai


def gpt4_prompting(content, client=None, max_tokens=1000):
    '''
    Prompting the standard GPT4 model. Used for data labeling.
    '''
    print("\n[DEBUG] ===== gpt4_prompting START =====")
    print(f"[DEBUG] content:\n{content}")
    print(f"[DEBUG] max_tokens: {max_tokens}")

    deployment_name = 'gpt-4o-mini-2024-07-18'
    messages = [
        {
            "role": "user",
            'content': content
        }
    ]
    print(f"[DEBUG] deployment_name: {deployment_name}")
    print(f"[DEBUG] messages: {messages}")

    try:
        api = client if client is not None else openai
        completion = api.ChatCompletion.create(
            model=deployment_name,
            messages=messages,
            max_tokens=max_tokens
        )

        # Safely extract output
        output = None
        usage = None
        try:
            output = completion.choices[0].message.content
        except Exception:
            try:
                output = completion.choices[0].get('text')
            except Exception:
                output = str(completion)

        try:
            usage = completion.usage.total_tokens
        except Exception:
            usage = None

        print(f"[DEBUG] Raw completion object (truncated): {str(completion)[:800]}")
        print(f"[DEBUG] Output: {output}")
        print(f"[DEBUG] Token usage: {usage}")
        print("[DEBUG] ===== gpt4_prompting END =====\n")
        return output, usage

    except Exception as e:
        print(f"[ERROR] Exception when calling gpt4_prompting: {e}")
        traceback.print_exc()
        return None, None



def gpt4_vision_prompting(prompt, client=None, image_path=None, map_manipulated_original=None, modality='vision',
                          temperature=0.2, max_tokens=50):
    '''
    Prompting GPT4 multimodal.
    '''
    print("\n[DEBUG] ===== gpt4_vision_prompting START =====")
    print(f"[DEBUG] prompt:\n{prompt}")
    print(f"[DEBUG] image_path: {image_path}")
    print(f"[DEBUG] modality: {modality}")
    if map_manipulated_original is not None:
        print(f"[DEBUG] map_manipulated_original size: {len(map_manipulated_original)}")

    deployment_name = 'gpt-4o-mini-2024-07-18'
    content = [{"type": "text", "text": prompt}]

    api = client if client is not None else openai

    if modality in ['vision','evidence', 'multimodal'] and image_path is not None:
        try:
            if map_manipulated_original and image_path in map_manipulated_original:
                print(f"[DEBUG] Image is manipulated. Switching to original: {map_manipulated_original[image_path]}")
                image_path = map_manipulated_original[image_path]

            print(f"[DEBUG] Encoding image: {image_path}")
            # encode_image should be defined elsewhere in your project
            image64 = encode_image(image_path)
            print(f"[DEBUG] Encoded image (first 120 chars): {image64[:120]}...")
            content += [{"type": "image_url", "image_url": {"url": image64}}]
        except Exception as e:
            print(f"[ERROR] Could not encode/add image to prompt: {e}")
            traceback.print_exc()

    messages = [{"role": "user", "content": content}]
    print(f"[DEBUG] deployment_name: {deployment_name}")
    print(f"[DEBUG] messages (truncated): {str(messages)[:1000]}")

    try:
        api = client if client is not None else openai
        completion = api.ChatCompletion.create(
            model=deployment_name,
            temperature=temperature,
            messages=messages,
            max_tokens=max_tokens
        )

        # Safely extract output
        try:
            output = completion.choices[0].message.content
        except Exception:
            try:
                output = completion.choices[0].get('text')
            except Exception:
                output = str(completion)

        print(f"[DEBUG] Raw completion object (truncated): {str(completion)[:900]}")
        print(f"[DEBUG] Output: {output}")
        print("[DEBUG] ===== gpt4_vision_prompting END =====\n")
        return output

    except Exception as e:
        print(f"[ERROR] Exception occurred during API call: {e}")
        traceback.print_exc()
        print("[DEBUG] ===== gpt4_vision_prompting END (with exception) =====\n")
        return 'Content Filtering error'

def assemble_prompt_gpt4(question,
                         answer=None,
                         evidence=[], 
                         demonstrations=[], 
                         modality='vision'):
    '''
    Assemble the prompt for GPT4.
    '''
    print("\n[DEBUG] ===== assemble_prompt_gpt4 START =====")
    print(f"[DEBUG] question: {question}")
    print(f"[DEBUG] answer: {answer}")
    print(f"[DEBUG] evidence count: {len(evidence)}")
    print(f"[DEBUG] demonstrations count: {len(demonstrations)}")
    print(f"[DEBUG] modality: {modality}")

    # demonstrations are tuples (image_paths, answer, evidence_df)
    prompt = ''
    for d in range(len(demonstrations)):
        print(f"[DEBUG] Processing demonstration {d+1}/{len(demonstrations)}")
        demo_answer = demonstrations[d][1]
        demo_evidence = demonstrations[d][2]
        prompt += assemble_prompt_gpt4(
            question,
            answer=demo_answer,
            evidence=demo_evidence,
            demonstrations=[], 
            modality='evidence'  # Demonstrations are provided without images
        )
        prompt += '\n\n'
    
    if modality == 'evidence':
        prompt += 'You are given online articles that used a certain image. Your task is to answer a question about the image.\n\n'
    elif modality == 'multimodal':
        prompt += 'You are given an image and online articles that used that image. Your task is to answer a question about the image using the image and the articles.\n\n'          
    else:
        prompt += 'You are given an image. Your task is to answer a question about the image.\n\n'  
    
    if len(evidence) != 0:
        print(f"[DEBUG] Adding {len(evidence)} evidence to prompt")
        prompt += get_evidence_prompt(evidence)

    prompt += 'Question: ' + question + '\n'
    prompt += 'Answer: '
    if answer:
        prompt += answer + '\n'

    print(f"[DEBUG] Final assembled prompt:\n{prompt}")
    print("[DEBUG] ===== assemble_prompt_gpt4 END =====\n")
    return prompt


#############
#   Llava   #
#############

def llava_prompting(prompt,
                    image_path,
                    pipe,
                    map_manipulated_original={},
                    temperature=0.2,
                    max_tokens=50):
  '''
  Prompting Llava model for vision and multimodal input.
  '''
  try:
    if image_path in map_manipulated_original.keys():
        #Convert to the original image if it is detected as manipulated and an original is available
        image_path = map_manipulated_original[image_path]
    image = Image.open(image_path)
    outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": max_tokens,
                                                        "temperature":temperature,
                                                        "do_sample":True})
    return outputs[0]['generated_text'].split('ASSISTANT:')[1].split('\n\n')[0]
  except RuntimeError as e:
    print(e)
    return ''


def assemble_prompt_llava(question,
                          answer=None,
                          evidence=[], 
                          demonstrations=[], 
                          modality='vision',
                          ):
    '''
    Assemble the prompt for Llava.
    '''
    prompt=''
    #Demonstrations are tuples (image_paths,answer,evidence_df)
    for d in range(len(demonstrations)):
        prompt += assemble_prompt_llava(question, answer=demonstrations[d][1],
                                  evidence=demonstrations[d][2],demonstrations=[],modality='evidence')
        prompt += '\n\n'
    prompt += 'USER:'
    if modality=='evidence':
        prompt += 'You are given online articles that used a certain image. Your task is to answer a question about the image.\n\n'
    elif modality=='multimodal':
        prompt += 'You are given an image and online articles that used that image. Your task is to answer a question about the image using the image and the articles.\n\n'
        prompt += '<image>'
    else:
        prompt += 'You are given an image. Your task is to answer a question about the image.\n\n'
        prompt += '<image>'
    if len(evidence)!=0:
        prompt += get_evidence_prompt(evidence)
    prompt += 'Question: ' + question + '\n'
    if answer:
        #Provide the answer for demonstrations
        prompt += 'Answer:'
        prompt +=  answer + '\n'
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
                    max_tokens
                    ):
    '''
    Prompting Llama2 model for text input.
    '''
    output = pipeline(prompt, 
                      eos_token_id=tokenizer.eos_token_id, 
                      max_new_tokens= max_tokens, 
                      temperature = temperature, 
                      do_sample=True)[0]['generated_text']
    if '[/INST]' in output:
        output = output.split('[/INST]')[1]
    return output


def assemble_prompt_llama(question,
                          answer=None,
                          evidence=[],
                          demonstrations=[],
                          modality='evidence'):
    '''
    Assemble the prompt for Llama2.
    '''
    prompt='<s>[INST] <<SYS>>'
    prompt += 'You are given online articles that used a certain image. Your task is to answer a question about the image.<</SYS>>\n'
    if len(evidence)!=0:
        prompt += get_evidence_prompt(evidence)
    prompt += 'Question: ' + question + '\n'
    prompt += '[/INST]'
    return prompt
