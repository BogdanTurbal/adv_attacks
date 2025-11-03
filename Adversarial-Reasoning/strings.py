import torch
import numpy as np
from utils import prompt_togetherai_batch, prompt_togetherai_multi, prompt_openrouter_batch, prompt_openrouter_multi
from convs import get_conv_feedbacker
import json
import re
import concurrent.futures
import time
import ast
import os

def get_attacks_string_with_timeout(address, conv, batch, timeout=120, retries=3, wait=60):
    def call_api():
        return get_attacks_string(address, conv, batch)

    for attempt in range(retries):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(call_api)
            try:
                result = future.result(timeout=timeout)
                return result
            except concurrent.futures.TimeoutError:
                print(f"Attempt {attempt + 1} timed out. Retrying after {wait} seconds...")
                time.sleep(wait)
    
    raise TimeoutError(f"All {retries} attempts timed out.")


def get_attacks_string(attacker_address, conv, batch, use_openrouter=False, api_key=None, base_url=None):
    """
    Get attack strings either from TogetherAI (via litellm) or OpenRouter API.
    
    Args:
        attacker_address: Model identifier (TogetherAI format or OpenRouter model name)
        conv: Conversation template
        batch: Number of prompts to generate
        use_openrouter: If True, use OpenRouter API instead of TogetherAI
        api_key: OpenRouter API key (required if use_openrouter is True)
        base_url: OpenRouter base URL (optional, defaults to https://openrouter.ai/api/v1)
    """
    messages = []
    generated = 0
    
    while True:
        try:
            if use_openrouter:
                # Use OpenRouter API
                if base_url is None:
                    base_url = "https://openrouter.ai/api/v1"
                outputs = prompt_openrouter_batch(attacker_address, conv, batch - generated, api_key=api_key, base_url=base_url)
            else:
                # Use TogetherAI (via litellm)
            outputs = prompt_togetherai_batch(attacker_address, conv, batch - generated)    
            
            for output in outputs:          
                result = extract_strings(output)
                if result:
                    messages.append(result)
                    generated += 1
                    
        except Exception as e:
            print(f"Error in get_attacks_string: {e}")
            pass 
        
        print(generated)
        if generated == batch:
            break
    
    return messages


def get_attacks_string_with_timeout(address, conv, batch, timeout=120, retries=3, wait=60, use_openrouter=False, api_key=None, base_url=None):
    def call_api():
        return get_attacks_string(address, conv, batch, use_openrouter=use_openrouter, api_key=api_key, base_url=base_url)

    for attempt in range(retries):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(call_api)
            try:
                result = future.result(timeout=timeout)
                return result
            except concurrent.futures.TimeoutError:
                print(f"Attempt {attempt + 1} timed out. Retrying after {wait} seconds...")
                time.sleep(wait)
    
    raise TimeoutError(f"All {retries} attempts timed out.")


def get_feedbacks(name, address, goal, target, messages, idx, divs, num_branches, use_openrouter=False, api_key=None, base_url=None):
    """
    Get feedbacks either from TogetherAI (via litellm) or OpenRouter API.
    
    Args:
        name: Model name identifier
        address: Model address/identifier
        goal: Goal string
        target: Target string
        messages: List of messages
        idx: Indices
        divs: Divisions for randomization
        num_branches: Number of branches
        use_openrouter: If True, use OpenRouter API instead of TogetherAI
        api_key: OpenRouter API key (required if use_openrouter is True)
        base_url: OpenRouter base URL (optional)
    """
    convs = [get_conv_feedbacker(name, goal, target, gen_string_feedbacker_rand(messages, idx, divs), len(messages)//divs) for _ in range(num_branches)]
    final_feedbacks = []
    indices_to_generate = np.arange(len(convs))
    
    while True:
        convs_to_generate = [convs[i] for i in indices_to_generate]
        
        if use_openrouter:
            if base_url is None:
                base_url = "https://openrouter.ai/api/v1"
            outputs = prompt_openrouter_multi(address, convs_to_generate, api_key=api_key, base_url=base_url)
        else:
        outputs = prompt_togetherai_multi(address, convs_to_generate)
        
        indices_copy = np.copy(indices_to_generate)
        
        print(indices_to_generate)
        
        for i, idx in enumerate(indices_to_generate):
            outputs[i] = outputs[i].replace("\\", "")
            final_feedback = extract_final_feedback(outputs[i])
            
            if final_feedback is not None:
                final_feedbacks.append(final_feedback)
                indices_copy = np.delete(indices_copy, np.where(indices_copy == idx)[0])
        
        indices_to_generate = indices_copy
         
                
        if len(indices_to_generate) == 0:
            break 
    
    return final_feedbacks


def get_new_prompts(convs, address, use_openrouter=False, api_key=None, base_url=None): 
    """
    Get new prompts either from TogetherAI (via litellm) or OpenRouter API.
    
    Args:
        convs: List of conversation templates
        address: Model address/identifier
        use_openrouter: If True, use OpenRouter API instead of TogetherAI
        api_key: OpenRouter API key (required if use_openrouter is True)
        base_url: OpenRouter base URL (optional)
    """
    new_prompts = []
    indices_to_generate = np.arange(len(convs))
    
    while True:
        convs_to_generate = [convs[i] for i in indices_to_generate]
        
        if use_openrouter:
            if base_url is None:
                base_url = "https://openrouter.ai/api/v1"
            outputs = prompt_openrouter_multi(address, convs_to_generate, api_key=api_key, base_url=base_url)
        else:
        outputs = prompt_togetherai_multi(address, convs_to_generate)

        indices_copy = np.copy(indices_to_generate)
        
        print(indices_to_generate)
        for i, idx in enumerate(indices_to_generate):
            outputs[i] =outputs[i].replace("\\", "")
            prompt = extract_new_prompt(outputs[i])
            
            if prompt is not None:
                new_prompts.append(prompt)
                indices_copy = np.delete(indices_copy, np.where(indices_copy == idx)[0])
        
        indices_to_generate = indices_copy   
                
        if len(indices_to_generate) == 0:
            break 
    
    return new_prompts


def gen_string_feedbacker_rand(messages, idx, div =8):
    assert len(messages) == len(idx) 
    l = len(idx)
        
    idx = idx.reshape(-1, div)[torch.arange(l//div), torch.randint(div, size = (l//div, ))]
    string = f""
    for i in range(len(idx)):
        string += f"Prompt_{i+1}:\n'{messages[idx[i]]}'\n\n"
    
    return string
    

def gen_string_optimizer(variable, feedback):
    string = f"Variable:\n'{variable}'\n\nFeedback:\n{feedback}"
    
    return string


def extract_strings(text):
    # Split the string into lines
    lines = text.split('\n')
    key_text = '"Prompt P": '
    for line in lines:
        # Check if key_text is in the line
        if key_text in line:
            # Find the index of key_text and get everything after it
            feedback_index = line.find(key_text)
            prompt = line[feedback_index + len(key_text):].strip()
            # Remove trailing brace if present
            if prompt.endswith('}'):
                prompt = prompt[:-1]
            try:
                # Safely evaluate the string to unescape it
                prompt = ast.literal_eval(prompt)
            except Exception as e:
                # Handle cases where evaluation fails
                pass
            return prompt
    return None


def extract_final_feedback(text):
    # Split the string into lines
    lines = text.split('\n')
    text = f""""Final_feedback": """
    for line in lines:
        # Check if 'final_feedback' is in the line
        if text in line:
            # Find the index of 'final_feedback' and get everything after it
            feedback_index = line.find(text)
            return line[feedback_index + len(text):].strip()
    
    return None


def extract_new_prompt(text):
    # Split the string into lines
    lines = text.split('\n')
    text = f""""Improved_variable": """
    for line in lines:
        # Check if 'final_feedback' is in the line
        if text in line:
            # Find the index of 'final_feedback' and get everything after it
            feedback_index = line.find(text)
            return line[feedback_index + len(text):].strip()

    return None