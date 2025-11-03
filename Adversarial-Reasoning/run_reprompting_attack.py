#!/usr/bin/env python3
"""
Reprompting Attack Runner
Runs reprompting-based adversarial attacks on reasoning models.
This is called by run_reprompting_unified.sh
"""

import argparse
import os
import sys
import pandas as pd
import torch
import numpy as np
from pathlib import Path
import multiprocessing as mp
import time
from math import ceil

from tqdm import tqdm
import fcntl

# Add Adversarial-Reasoning to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from convs import get_conv_attacker, get_init_msg, get_conv_feedbacker, get_conv_optimizer
from utils import load_model_and_tokenizer, get_losses, get_target_responses_local
from strings import gen_string_optimizer, gen_string_feedbacker_rand, extract_strings, extract_final_feedback, extract_new_prompt, get_attacks_string_with_timeout, get_feedbacks, get_new_prompts
from alg import GWW_dfs_min
from fastchat.model import get_conversation_template


# Global variables for multiprocessing workers (one per GPU)
_worker_target_model = None
_worker_target_tokenizer = None
_worker_attacker_model = None
_worker_attacker_tokenizer = None
_worker_attacker_name = None
_worker_gpu_id = None


def print_gpu_memory(prefix=""):
    """Print current GPU memory usage for all available GPUs."""
    if not torch.cuda.is_available():
        print(f"{prefix}No CUDA GPUs available")
        return
    
    num_gpus = torch.cuda.device_count()
    print(f"{prefix}GPU Memory Usage:")
    for i in range(num_gpus):
        allocated = torch.cuda.memory_allocated(i) / 1e9  # GB
        reserved = torch.cuda.memory_reserved(i) / 1e9  # GB
        max_allocated = torch.cuda.max_memory_allocated(i) / 1e9  # GB
        props = torch.cuda.get_device_properties(i)
        total_memory = props.total_memory / 1e9  # GB
        
        allocated_pct = (allocated / total_memory) * 100
        reserved_pct = (reserved / total_memory) * 100
        
        print(f"  GPU {i} ({props.name}):")
        print(f"    Allocated: {allocated:.2f} GB ({allocated_pct:.1f}%)")
        print(f"    Reserved:  {reserved:.2f} GB ({reserved_pct:.1f}%)")
        print(f"    Max Allocated: {max_allocated:.2f} GB")
        print(f"    Total Memory: {total_memory:.2f} GB")
    
    print()  # Empty line for readability


def get_feedbacks_local(name, model, tokenizer, goal, target, messages, idx, divs, num_branches, device):
    """Get feedbacks using local model instead of API. Batched for efficiency."""
    from strings import gen_string_feedbacker_rand, extract_final_feedback
    
    convs = [
        get_conv_feedbacker(name, goal, target, gen_string_feedbacker_rand(messages, idx, divs), len(messages)//divs) 
        for _ in range(num_branches)
    ]
    
    # Batch all convs together for efficient generation
    all_input_sequences = []
    for conv in convs:
        conv_input = get_conversation_template(name)
        conv_input.sep2 = conv_input.sep2.strip()
        conv_input.set_system_message(conv.system_message)
        conv_input.append_message(conv_input.roles[0], conv.messages[0][1])
        
        input_text = tokenizer.apply_chat_template(
            conv_input.to_openai_api_messages(),
            tokenize=False,
            add_generation_prompt=True
        )
        all_input_sequences.append(input_text)
    
    # Generate all at once in a single batch
    input_ids = tokenizer(all_input_sequences, return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **input_ids,
            max_new_tokens=512,
            temperature=1.0,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    final_feedbacks = []
    
    for generated_text in generated_texts:
        generated_text = generated_text.replace("\\", "")
        final_feedback = extract_final_feedback(generated_text)
        if final_feedback is not None:
            final_feedbacks.append(final_feedback)
    
    return final_feedbacks if final_feedbacks else ["Improve the prompt to better elicit the target behavior."] * num_branches


def get_new_prompts_local(convs, model, tokenizer, device):
    """Get new prompts using local model instead of API. Batched for efficiency."""
    from strings import extract_new_prompt
    
    if not convs:
        return []
    
    # Batch all convs together for efficient generation
    all_input_sequences = []
    for conv in convs:
        conv_input = get_conversation_template("mixtral")  # Use mixtral template
        conv_input.sep2 = conv_input.sep2.strip()
        conv_input.set_system_message(conv.system_message)
        conv_input.append_message(conv_input.roles[0], conv.messages[0][1])
        
        input_text = tokenizer.apply_chat_template(
            conv_input.to_openai_api_messages(),
            tokenize=False,
            add_generation_prompt=True
        )
        all_input_sequences.append(input_text)
    
    # Generate all at once in a single batch
    input_ids = tokenizer(all_input_sequences, return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **input_ids,
            max_new_tokens=512,
            temperature=1.0,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    new_prompts = []
    
    for generated_text in generated_texts:
        generated_text = generated_text.replace("\\", "")
        prompt = extract_new_prompt(generated_text)
        if prompt is not None:
            new_prompts.append(prompt)
    
    return new_prompts if new_prompts else [convs[0].messages[0][1] if convs and convs[0].messages else ""]


def load_dataset(dataset_path: str, example_range: str = None, max_examples: int = -1):
    """Load dataset from CSV file."""
    if not os.path.isabs(dataset_path):
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dataset_path = os.path.join(script_dir, dataset_path)
    
    df = pd.read_csv(dataset_path)
    
    # Parse example range
    if example_range:
        if ':' in example_range:
            start, end = example_range.split(':')
            if '.' in start:  # Percentage
                start_idx = int(float(start) * len(df))
                end_idx = int(float(end) * len(df))
            else:  # Integer
                start_idx = int(start)
                end_idx = int(end)
            df = df.iloc[start_idx:end_idx].reset_index(drop=True)
    
    if max_examples > 0:
        df = df.head(max_examples)
    
    return df


def extract_target_from_response(response: str, tokenizer=None, default_target: str = "Sure", num_tokens_after_tag: int = 20) -> str:
    """Extract target string from response for reprompting attack.
    
    Extracts: </think> + next num_tokens_after_tag tokens
    
    Args:
        response: The full response string from the dataset
        tokenizer: Optional tokenizer to use for token counting. If None, uses simple word-based approximation
        default_target: Default target if extraction fails
        num_tokens_after_tag: Number of tokens to include after the </think> tag (default: 20)
    """
    if pd.isna(response) or not response:
        return default_target
    
    tag = "</think>"
    
    # Find the tag in the response
    if tag not in response:
        # Fallback: return first 50 characters if tag not found
        return response[:50].strip() if response.strip() else default_target
    
    # Split at the tag position
    parts = response.split(tag, 1)
    if len(parts) < 2:
        return default_target
    
    # Get text after the tag
    text_after_tag = parts[1]
    
    # Extract tag + next num_tokens_after_tag tokens
    if tokenizer is not None:
        # Use actual tokenizer for accurate token counting
        try:
            # Tokenize the text after the tag
            tokens = tokenizer.encode(text_after_tag, add_special_tokens=False)
            
            # Take first num_tokens_after_tag tokens
            tokens_to_include = tokens[:num_tokens_after_tag]
            
            # Decode back to text
            target_after = tokenizer.decode(tokens_to_include, skip_special_tokens=False)
            
            # Combine tag + target text
            target = tag + "\n" + target_after if target_after.strip() else tag
        except Exception as e:
            print(f"Warning: Tokenizer-based extraction failed: {e}, falling back to word-based method")
            # Fallback to word-based method
            words = text_after_tag.split()
            words_to_include = words[:num_tokens_after_tag]
            target_after = " ".join(words_to_include)
            target = tag + "\n" + target_after if target_after.strip() else tag
    else:
        # Simple word-based approximation (assume ~1 word ≈ 1 token)
        words = text_after_tag.split()
        words_to_include = words[:num_tokens_after_tag]
        target_after = " ".join(words_to_include)
        target = tag + "\n" + target_after if target_after.strip() else tag
    
    return target.strip() if target.strip() else default_target


def flatten_result(result):
    """Flatten a single result dictionary for CSV saving."""
    base_result = {
        'prompt_idx': result['prompt_idx'],
        'goal': result['goal'],
        'target': result['target'],
        'best_prompt': result['best_prompt'],
        'best_loss': result['best_loss'],
        'mean_loss': result['mean_loss'],
        'initial_loss_mean': result['initial_loss_mean'],
        'initial_loss_min': result['initial_loss_min'],
        'final_loss_mean': result['final_loss_mean'],
        'final_loss_min': result['final_loss_min'],
        'loss_improvement': result['initial_loss_min'] - result['final_loss_min']  # Positive means improvement
    }
    # Add iteration losses as separate columns (for analysis)
    if result.get('iteration_losses'):
        for iter_loss in result['iteration_losses']:
            iter_num = iter_loss['iteration']
            base_result[f'iter_{iter_num}_loss_mean'] = iter_loss['loss_mean']
            base_result[f'iter_{iter_num}_loss_min'] = iter_loss['loss_min']
    # Add outputs
    if result.get('outputs'):
        for i, output in enumerate(result['outputs']):
            base_result[f'output_{i}'] = output
    return base_result


def save_result_to_csv(result, output_path):
    """Append a single result to CSV file with file locking for thread safety."""
    try:
        flat_result = flatten_result(result)
        result_df = pd.DataFrame([flat_result])
        
        # Use file locking to prevent concurrent write conflicts
        # Open in append mode and check file existence after acquiring lock
        with open(output_path, 'a+') as f:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # Exclusive lock
                
                # Check if file has content (seek to end, then check position)
                f.seek(0, 2)  # Seek to end
                file_pos = f.tell()
                file_exists = file_pos > 0
                
                if file_exists:
                    # Append without header
                    result_df.to_csv(f, index=False, header=False)
                else:
                    # Write with header (first write)
                    result_df.to_csv(f, index=False, header=True)
                
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # Release lock
            except (BlockingIOError, IOError, AttributeError) as e:
                # If lock fails or fcntl not available (Windows), fallback without locking
                # Note: On Windows, fcntl is not available, but this code runs on Linux
                f.seek(0, 2)  # Seek to end
                file_pos = f.tell()
                file_exists = file_pos > 0
                if file_exists:
                    result_df.to_csv(f, index=False, header=False)
                else:
                    result_df.to_csv(f, index=False, header=True)
        
    except Exception as e:
        print(f"Warning: Failed to save result to CSV: {e}")
        # Don't fail the entire process if CSV saving fails


def generate_new_messages_for_prompt(
    prompt_new: str,
    attacker_name: str,
    goal: str,
    target: str,
    batch_size: int,
    use_attacker_api: bool,
    attacker_api_model: str = None,
    attacker_api_key: str = None,
    attacker_api_base_url: str = None,
    attacker_model=None,
    attacker_tokenizer=None,
    device: str = "cuda:0",
    iter_num: int = 0
):
    """Generate new messages for a single prompt_new. Can be used for parallelization."""
    conv = get_conv_attacker(attacker_name, goal, target, prompt_new)
    
    if use_attacker_api:
        # Use API to generate new messages
        new_messages = get_attacks_string_with_timeout(
            attacker_api_model,
            conv,
            batch_size,
            use_openrouter=True,
            api_key=attacker_api_key,
            base_url=attacker_api_base_url
        )
        if not new_messages:
            new_messages = [prompt_new] * batch_size
    else:
        # Generate new messages using local attacker model
        attacker_conv = get_conversation_template(attacker_name)
        attacker_conv.sep2 = attacker_conv.sep2.strip()
        attacker_conv.set_system_message(conv.system_message)
        attacker_conv.append_message(attacker_conv.roles[0], prompt_new)
        
        attacker_inputs = attacker_tokenizer.apply_chat_template(
            attacker_conv.to_openai_api_messages(),
            tokenize=False,
            add_generation_prompt=True
        )
        input_ids = attacker_tokenizer([attacker_inputs] * batch_size, return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            outputs = attacker_model.generate(
                **input_ids,
                max_new_tokens=512,
                temperature=1.0,
                top_p=0.9,
                do_sample=True,
                pad_token_id=attacker_tokenizer.pad_token_id,
                eos_token_id=attacker_tokenizer.eos_token_id
            )
        
        generated_texts = attacker_tokenizer.batch_decode(outputs, skip_special_tokens=False)
        new_messages = []
        from strings import extract_strings
        for text in generated_texts:
            extracted = extract_strings(text)
            if extracted:
                new_messages.append(extracted)
            else:
                # Fallback
                if '"Prompt P":' in text or '"Prompt P" :' in text:
                    try:
                        if '"Prompt P":' in text:
                            prompt_part = text.split('"Prompt P":')[1].strip()
                        else:
                            prompt_part = text.split('"Prompt P" :')[1].strip()
                        prompt_part = prompt_part.strip('"').strip(',').strip('}').strip()
                        new_messages.append(prompt_part)
                    except:
                        new_messages.append(text)
        
        if not new_messages:
            new_messages = [prompt_new] * batch_size
    
    return prompt_new, new_messages


def run_reprompting_attack(
    goal: str,
    target: str,
    model,
    tokenizer,
    attacker_model,
    attacker_tokenizer,
    attacker_name: str,
    num_iters: int,
    num_branches: int,
    memory: int,
    K: int,
    batch_size: int = 16,
    device: str = "cuda:0",
    use_attacker_api: bool = False,
    attacker_api_model: str = None,
    attacker_api_key: str = None,
    attacker_api_base_url: str = None
):
    """Run a single reprompting attack.
    
    Args:
        use_attacker_api: If True, use OpenRouter API instead of local attacker model
        attacker_api_model: Model identifier for OpenRouter (e.g., "mistralai/mixtral-8x7b-instruct")
        attacker_api_key: OpenRouter API key
        attacker_api_base_url: OpenRouter base URL (defaults to https://openrouter.ai/api/v1)
    """
    
    # Initialize prompt
    prompt = get_init_msg(goal, target)
    
    # Initialize GWW algorithm
    prompt_class = GWW_dfs_min(memory)
    
    # Generate initial attacks
    conv = get_conv_attacker(attacker_name, goal, target, prompt)
    
    if use_attacker_api:
        # Use API for initial prompt generation
        messages = get_attacks_string_with_timeout(
            attacker_api_model,
            conv,
            batch_size,
            use_openrouter=True,
            api_key=attacker_api_key,
            base_url=attacker_api_base_url
        )
        if not messages:
            # Fallback: use original prompt
            messages = [prompt] * batch_size
    else:
        # Use local attacker model for initial prompt generation
        attacker_conv = get_conversation_template(attacker_name)
        attacker_conv.sep2 = attacker_conv.sep2.strip()
        attacker_conv.set_system_message(conv.system_message)
        attacker_conv.append_message(attacker_conv.roles[0], prompt)
        
        # Generate batch of prompts
        messages = []
        attacker_inputs = attacker_tokenizer.apply_chat_template(
        attacker_conv.to_openai_api_messages(), 
        tokenize=False, 
        add_generation_prompt=True
        )
        input_ids = attacker_tokenizer([attacker_inputs] * batch_size, return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            outputs = attacker_model.generate(
            **input_ids,
            max_new_tokens=512,
            temperature=1.0,
            top_p=0.9,
            do_sample=True,
            pad_token_id=attacker_tokenizer.pad_token_id,
            eos_token_id=attacker_tokenizer.eos_token_id
        )
        
        generated_texts = attacker_tokenizer.batch_decode(outputs, skip_special_tokens=False)
        
        # Extract prompt P from JSON responses (as per the notebook)
        from strings import extract_strings
        for text in tqdm(generated_texts, desc="Extracting initial prompts", leave=False):
            # Use the extract_strings function from strings.py
            extracted = extract_strings(text)
            if extracted:
                messages.append(extracted)
            else:
                # Fallback: extract manually
                if '"Prompt P":' in text or '"Prompt P" :' in text:
                    try:
                        if '"Prompt P":' in text:
                            prompt_part = text.split('"Prompt P":')[1].strip()
                        else:
                            prompt_part = text.split('"Prompt P" :')[1].strip()
                        prompt_part = prompt_part.strip('"').strip(',').strip('}').strip()
                        messages.append(prompt_part)
                    except:
                        messages.append(text)
        
        if not messages:
            # Fallback: use original prompt
            messages = [prompt] * batch_size
    
    # Compute losses on target model
    #print_gpu_memory("[Before computing initial losses] ")
    losses, _ = get_losses(model, tokenizer, messages, target, "deepseek")
    #print_gpu_memory("[After computing initial losses] ")
    prompt_class.add_prompt(prompt, losses, messages)
    
    # Track initial losses
    initial_loss_mean = torch.mean(losses).item()
    initial_loss_min = torch.min(losses).item()
    
    # Track best loss and corresponding prompt across all iterations (starting with initial)
    best_loss_overall = initial_loss_min
    best_prompt_overall = messages[losses.argmin().item()] if len(messages) > 0 else prompt
    print(f"Initial loss: mean={initial_loss_mean:.4f}, min={initial_loss_min:.4f} (best so far: {best_loss_overall:.4f})")
    print(f"Best attack prompt (initial): {best_prompt_overall[:200]}..." if len(best_prompt_overall) > 200 else f"Best attack prompt (initial): {best_prompt_overall}")
    
    # Main iteration loop
    iteration_losses = []  # Track losses at each iteration
    for iter in tqdm(range(num_iters), desc="Main iterations", leave=False):
        #print_gpu_memory(f"[Iter {iter}: Start] ")
        init_prompt, losses, messages = prompt_class.get_prompt()
        idx = losses.argsort()
        
        # Log iteration losses
        iter_loss_mean = torch.mean(losses).item()
        iter_loss_min = torch.min(losses).item()
        
        # Update best loss and prompt overall
        if iter_loss_min < best_loss_overall:
            best_loss_overall = iter_loss_min
            best_prompt_overall = messages[losses.argmin().item()] if len(messages) > 0 else best_prompt_overall
        
        iteration_losses.append({
            'iteration': iter,
            'loss_mean': iter_loss_mean,
            'loss_min': iter_loss_min
        })
        
        # Print current best loss and prompt for sanity check
        print(f"Iter {iter}: loss_mean={iter_loss_mean:.4f}, loss_min={iter_loss_min:.4f}, best_loss_overall={best_loss_overall:.4f}")
        print(f"  Best attack prompt so far: {best_prompt_overall[:200]}..." if len(best_prompt_overall) > 200 else f"  Best attack prompt so far: {best_prompt_overall}")
        
        # Generate output based on currently best prompt after each 5th iteration
        if (iter + 1) % 5 == 0:
            print(f"\n[Iter {iter}] Generating output using current best prompt (loss: {best_loss_overall:.4f})...")
            try:
                best_output = get_target_responses_local(model, tokenizer, [best_prompt_overall], max_n_tokens=1024)
                if best_output and len(best_output) > 0:
                    generated_text = best_output[0]
                    # Truncate for display if too long
                    display_text = generated_text[:500] + "..." if len(generated_text) > 500 else generated_text
                    print(f"  Generated output: {display_text}")
                    print(f"  Full output length: {len(generated_text)} characters")
                else:
                    print(f"  Warning: No output generated from best prompt")
            except Exception as e:
                print(f"  Error generating output: {e}")
                import traceback
                traceback.print_exc()
            print()  # Empty line for readability
        
        # Get feedbacks
        #print_gpu_memory(f"[Iter {iter}: Before getting feedbacks] ")
        if use_attacker_api:
            final_feedbacks = get_feedbacks(
                attacker_name,
                attacker_api_model,
                goal,
                target,
                messages,
                idx,
                K,
                num_branches,
                use_openrouter=True,
                api_key=attacker_api_key,
                base_url=attacker_api_base_url
            )
        else:
            final_feedbacks = get_feedbacks_local(
            attacker_name,
            attacker_model,
            attacker_tokenizer,
            goal,
            target,
            messages,
            idx,
            K,
            num_branches,
            device
        )
        #print_gpu_memory(f"[Iter {iter}: After getting feedbacks] ")
        
        # Generate optimized prompts
        collections_opt = [gen_string_optimizer(init_prompt, final_feedback) for final_feedback in final_feedbacks]
        convs_opt = [get_conv_optimizer(attacker_name, goal, target, collection_opt) for collection_opt in collections_opt]
        #print_gpu_memory(f"[Iter {iter}: Before generating new prompts] ")
        if use_attacker_api:
            new_prompts = get_new_prompts(
                convs_opt,
                attacker_api_model,
                use_openrouter=True,
                api_key=attacker_api_key,
                base_url=attacker_api_base_url
            )
        else:
            new_prompts = get_new_prompts_local(convs_opt, attacker_model, attacker_tokenizer, device)
        #print_gpu_memory(f"[Iter {iter}: After generating new prompts] ")
        
        # Evaluate new prompts - batched version
        # Step 1: Generate all new_messages in a single batched API call
        prompt_new_messages_map = {}  # Maps prompt_new -> list of new_messages
        
        # Import extract_strings at the top level so it's available in both branches
        from strings import extract_strings
        
        if use_attacker_api:
            # Use parallel individual API calls instead of batched (more reliable)
            from concurrent.futures import ThreadPoolExecutor, as_completed
            from utils import prompt_openrouter_batch
            
            def generate_messages_for_single_prompt(prompt_new):
                """Generate messages for a single prompt using individual API calls."""
                try:
                    conv = get_conv_attacker(attacker_name, goal, target, prompt_new)
                    
                    if attacker_api_base_url is None:
                        attacker_api_base_url_local = "https://openrouter.ai/api/v1"
                    else:
                        attacker_api_base_url_local = attacker_api_base_url
                    
                    # Use prompt_openrouter_batch which handles retries internally
                    generated_texts = prompt_openrouter_batch(
                        attacker_api_model,
                        conv,
                        batch_size,
                        api_key=attacker_api_key,
                        base_url=attacker_api_base_url_local
                    )
                    
                    new_messages = []
                    for text in generated_texts:
                        if text and text.strip():  # Only process non-empty responses
                            text = text.replace("\\", "")
                            extracted = extract_strings(text)
                            if extracted:
                                new_messages.append(extracted)
                            else:
                                # Fallback parsing
                                if '"Prompt P":' in text or '"Prompt P" :' in text:
                                    try:
                                        if '"Prompt P":' in text:
                                            prompt_part = text.split('"Prompt P":')[1].strip()
                                        else:
                                            prompt_part = text.split('"Prompt P" :')[1].strip()
                                        prompt_part = prompt_part.strip('"').strip(',').strip('}').strip()
                                        new_messages.append(prompt_part)
                                    except:
                                        new_messages.append(text)
                                else:
                                    new_messages.append(text)
                    
                    # Ensure we have batch_size messages
                    if len(new_messages) < batch_size:
                        while len(new_messages) < batch_size:
                            new_messages.append(prompt_new)
                    
                    if not new_messages:
                        new_messages = [prompt_new] * batch_size
                    
                    return prompt_new, new_messages
                except Exception as e2:
                    print(f"Error generating messages for prompt '{prompt_new[:50]}...': {e2}")
                    # Retry once more
                    try:
                        conv = get_conv_attacker(attacker_name, goal, target, prompt_new)
                        from strings import get_attacks_string_with_timeout
                        retry_messages = get_attacks_string_with_timeout(
                            attacker_api_model,
                            conv,
                            batch_size,
                            timeout=120,
                            retries=2,
                            wait=30,
                            use_openrouter=True,
                            api_key=attacker_api_key,
                            base_url=attacker_api_base_url_local
                        )
                        if retry_messages and len(retry_messages) >= batch_size:
                            return prompt_new, retry_messages[:batch_size]
                    except:
                        pass
                    return prompt_new, [prompt_new] * batch_size
            
            # Use ThreadPoolExecutor for parallel individual calls (more reliable than batched)
            print(f"Generating messages for {len(new_prompts)} prompts using parallel individual API calls...")
            executor = None
            try:
                executor = ThreadPoolExecutor(max_workers=min(len(new_prompts) * 2, 20))
                future_to_prompt = {
                    executor.submit(generate_messages_for_single_prompt, prompt_new): prompt_new
                    for prompt_new in new_prompts
                }
                
                for future in tqdm(as_completed(future_to_prompt), total=len(new_prompts), desc=f"Iter {iter}: Generating messages"):
                    try:
                        prompt_new, new_messages = future.result()
                        prompt_new_messages_map[prompt_new] = new_messages
                    except Exception as e2:
                        prompt_new = future_to_prompt[future]
                        print(f"Failed to get result for prompt '{prompt_new[:50]}...': {e2}")
                        prompt_new_messages_map[prompt_new] = [prompt_new] * batch_size
            finally:
                # Ensure executor is properly cleaned up to prevent semaphore leaks
                if executor is not None:
                    try:
                        # Python 3.9+ supports cancel_futures parameter
                        executor.shutdown(wait=True, cancel_futures=False)
                    except TypeError:
                        # Older Python versions don't have cancel_futures
                        executor.shutdown(wait=True)
            
            print(f"Completed: generated messages for {len(prompt_new_messages_map)}/{len(new_prompts)} prompts")
        else:
            # For local model, batch all prompts together for efficient generation
            # Step 1: Prepare all input sequences
            all_input_sequences = []
            prompt_to_indices = {}  # Maps prompt_new -> (start_idx, end_idx) in the batched input
            total_batch_size = len(new_prompts) * batch_size
            
            for idx, prompt_new in enumerate(new_prompts):
                conv = get_conv_attacker(attacker_name, goal, target, prompt_new)
                attacker_conv = get_conversation_template(attacker_name)
                attacker_conv.sep2 = attacker_conv.sep2.strip()
                attacker_conv.set_system_message(conv.system_message)
                attacker_conv.append_message(attacker_conv.roles[0], prompt_new)
                
                attacker_inputs = attacker_tokenizer.apply_chat_template(
                    attacker_conv.to_openai_api_messages(),
                    tokenize=False,
                    add_generation_prompt=True
                )
                # Create batch_size copies for this prompt
                all_input_sequences.extend([attacker_inputs] * batch_size)
                
                # Track indices: this prompt will generate batch_size outputs
                start_idx = idx * batch_size
                end_idx = start_idx + batch_size
                prompt_to_indices[prompt_new] = (start_idx, end_idx)
            
            # Step 2: Generate all sequences in one batch
            try:
                input_ids = attacker_tokenizer(all_input_sequences, return_tensors="pt", padding=True).to(device)
                
                with torch.no_grad():
                    outputs = attacker_model.generate(
                        **input_ids,
                        max_new_tokens=512,
                        temperature=1.0,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=attacker_tokenizer.pad_token_id,
                        eos_token_id=attacker_tokenizer.eos_token_id
                    )
                
                generated_texts = attacker_tokenizer.batch_decode(outputs, skip_special_tokens=False)
                
                # Step 3: Extract messages and map back to prompts
                # extract_strings already imported at the top of this block
                
                for prompt_new in tqdm(new_prompts, desc=f"Iter {iter}: Extracting messages from batch", leave=False):
                    start_idx, end_idx = prompt_to_indices[prompt_new]
                    prompt_generated_texts = generated_texts[start_idx:end_idx]
                    
                    new_messages = []
                    for text in prompt_generated_texts:
                        extracted = extract_strings(text)
                        if extracted:
                            new_messages.append(extracted)
                        else:
                            # Fallback
                            if '"Prompt P":' in text or '"Prompt P" :' in text:
                                try:
                                    if '"Prompt P":' in text:
                                        prompt_part = text.split('"Prompt P":')[1].strip()
                                    else:
                                        prompt_part = text.split('"Prompt P" :')[1].strip()
                                    prompt_part = prompt_part.strip('"').strip(',').strip('}').strip()
                                    new_messages.append(prompt_part)
                                except:
                                    new_messages.append(text)
                    
                    if not new_messages:
                        new_messages = [prompt_new] * batch_size
                    
                    prompt_new_messages_map[prompt_new] = new_messages
                    
            except Exception as e:
                print(f"Error in batched generation: {e}")
                # Fallback to sequential if batch fails
                for prompt_new in new_prompts:
                    try:
                        _, new_messages = generate_new_messages_for_prompt(
                            prompt_new,
                            attacker_name,
                            goal,
                            target,
                            batch_size,
                            use_attacker_api,
                            None,  # attacker_api_model
                            None,  # attacker_api_key
                            None,  # attacker_api_base_url
                            attacker_model,
                            attacker_tokenizer,
                            device,
                            iter
                        )
                        prompt_new_messages_map[prompt_new] = new_messages
                    except Exception as e2:
                        print(f"Error generating messages for prompt: {e2}")
                        prompt_new_messages_map[prompt_new] = [prompt_new] * batch_size
        
        # Step 2: Compute losses for all prompts in a single batched operation
        # Collect all messages and their corresponding prompts
        all_prompt_new_list = []
        all_new_messages_list = []
        
        for prompt_new in new_prompts:
            if prompt_new in prompt_new_messages_map:
                all_prompt_new_list.append(prompt_new)
                all_new_messages_list.extend(prompt_new_messages_map[prompt_new])
        
        # Compute losses in batches for efficiency
        # We'll process all messages together, then split the results back
        if all_new_messages_list:
            #print_gpu_memory(f"[Iter {iter}: Before computing losses for all prompts] ")
            all_losses, _ = get_losses(model, tokenizer, all_new_messages_list, target, "deepseek")
            #print_gpu_memory(f"[Iter {iter}: After computing losses for all prompts] ")
            
            # Split losses back to each prompt_new
            loss_idx = 0
            for prompt_new in all_prompt_new_list:
                if prompt_new in prompt_new_messages_map:
                    new_messages = prompt_new_messages_map[prompt_new]
                    num_messages = len(new_messages)
                    losses_new = all_losses[loss_idx:loss_idx + num_messages]
                    loss_idx += num_messages
                    prompt_class.add_prompt(prompt_new, losses_new, new_messages)
    
    # Get final best prompt
    #print_gpu_memory("[After all iterations: Before final prompt selection] ")
    final_prompt, final_losses, final_messages = prompt_class.get_prompt()
    
    # Get target responses
    #print_gpu_memory("[Before getting target responses] ")
    outputs = get_target_responses_local(model, tokenizer, final_messages, max_n_tokens=1024)
    #print_gpu_memory("[After getting target responses] ")
    
    # Compute final losses
    final_loss_mean = final_losses.mean().item() if len(final_losses) > 0 else float('inf')
    final_loss_min = final_losses.min().item() if len(final_losses) > 0 else float('inf')
    
    # Update best loss and prompt if final is better
    if final_loss_min < best_loss_overall:
        best_loss_overall = final_loss_min
        best_prompt_overall = final_messages[final_losses.argmin().item()] if len(final_messages) > 0 else best_prompt_overall
    
    print(f"Final loss: mean={final_loss_mean:.4f}, min={final_loss_min:.4f}, best_loss_overall={best_loss_overall:.4f}")
    print(f"Best attack prompt (final): {best_prompt_overall[:200]}..." if len(best_prompt_overall) > 200 else f"Best attack prompt (final): {best_prompt_overall}")
    
    return {
        'best_prompt': final_messages[final_losses.argsort()[0].item()] if len(final_messages) > 0 else "",
        'best_loss': final_loss_min,
        'mean_loss': final_loss_mean,
        'initial_loss_mean': initial_loss_mean,
        'initial_loss_min': initial_loss_min,
        'final_loss_mean': final_loss_mean,
        'final_loss_min': final_loss_min,
        'outputs': outputs,
        'losses': final_losses.cpu().numpy().tolist() if len(final_losses) > 0 else [],
        'iteration_losses': iteration_losses  # Store iteration-by-iteration losses
    }


def worker_init(gpu_id: int, target_model_name: str, attacker_model_name: str, attacker_quantize: bool, attacker_quantize_bits: int, attacker_use_flash_attention: bool, use_attacker_api: bool = False, target_quantize: bool = False, target_quantize_bits: int = 8):
    """Initialize worker process with models and tokenizers on a specific GPU."""
    global _worker_target_model, _worker_target_tokenizer
    global _worker_attacker_model, _worker_attacker_tokenizer, _worker_attacker_name, _worker_gpu_id
    
    _worker_gpu_id = gpu_id
    
    print(f"[GPU {gpu_id}] Initializing worker process...")
    print(f"[GPU {gpu_id}] CUDA available: {torch.cuda.is_available()}")
    print(f"[GPU {gpu_id}] CUDA device count: {torch.cuda.device_count()}")
    
    try:
        # Set CUDA device for this worker
        torch.cuda.set_device(gpu_id)
        device = f"cuda:{gpu_id}"
        
        # Clear any existing cache
        torch.cuda.empty_cache()
        
        # Load target model on this GPU
        target_quantize_status = f"quantized ({target_quantize_bits}-bit)" if target_quantize else "not quantized"
        print(f"[GPU {gpu_id}] Loading target model: {target_model_name} ({target_quantize_status})")
        _worker_target_model, _worker_target_tokenizer = load_model_and_tokenizer(
            target_model_name,
            low_cpu_mem_usage=True,
            cache_dir=os.environ.get("HF_HOME", "/u/bt4811/huggingface"),
            device=device,
            quantize=target_quantize,
            quantization_bits=target_quantize_bits,
            device_map=device  # Load entire model on this specific GPU
        )
        print(f"[GPU {gpu_id}] ✓ Target model loaded successfully")
        
        # Load attacker model only if not using API
        if not use_attacker_api:
            flash_attention_status = "enabled" if attacker_use_flash_attention else "disabled"
            print(f"[GPU {gpu_id}] Loading attacker model: {attacker_model_name} (quantized: {attacker_quantize}, bits: {attacker_quantize_bits if attacker_quantize else 'N/A'}, flash_attention: {flash_attention_status})")
            attacker_kwargs = {}
            if attacker_use_flash_attention:
                attacker_kwargs['use_flash_attention_2'] = True
            
            _worker_attacker_model, _worker_attacker_tokenizer = load_model_and_tokenizer(
                attacker_model_name,
                low_cpu_mem_usage=True,
                cache_dir=os.environ.get("HF_HOME", "/u/bt4811/huggingface"),
                device=device,
                quantize=attacker_quantize,
                quantization_bits=attacker_quantize_bits,
                device_map=device,  # Load entire model on this specific GPU
                **attacker_kwargs
            )
            _worker_attacker_name = "mixtral"
            print(f"[GPU {gpu_id}] ✓ Attacker model loaded successfully")
        else:
            print(f"[GPU {gpu_id}] Using OpenRouter API for attacker model (skipping local model loading)")
            _worker_attacker_model = None
            _worker_attacker_tokenizer = None
            _worker_attacker_name = "mixtral"
        
        # Monitor memory after loading
        allocated = torch.cuda.memory_allocated(gpu_id) / 1e9
        reserved = torch.cuda.memory_reserved(gpu_id) / 1e9
        print(f"[GPU {gpu_id}] Models loaded. Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
        
    except Exception as e:
        print(f"[GPU {gpu_id}] Error initializing worker: {e}")
        print(f"[GPU {gpu_id}] Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        raise


def run_attacks_on_gpu_worker(
    gpu_id: int,
    target_model_name: str,
    attacker_model_name: str,
    attacker_quantize: bool,
    attacker_quantize_bits: int,
    attacker_use_flash_attention: bool,
    examples: list,
    num_iters: int,
    num_branches: int,
    memory: int,
    K: int,
    batch_size: int,
    verbose: bool,
    use_attacker_api: bool = False,
    attacker_api_model: str = None,
    attacker_api_key: str = None,
    attacker_api_base_url: str = None,
    output_path: str = None,
    target_quantize: bool = False,
    target_quantize_bits: int = 8
) -> list:
    """Worker function to initialize GPU and run attacks."""
    # Initialize models on this GPU
    worker_init(gpu_id, target_model_name, attacker_model_name, attacker_quantize, attacker_quantize_bits, attacker_use_flash_attention, use_attacker_api, target_quantize, target_quantize_bits)
    
    # Run attacks on this GPU
    return run_attacks_on_gpu(gpu_id, examples, num_iters, num_branches, memory, K, batch_size, verbose, use_attacker_api, attacker_api_model, attacker_api_key, attacker_api_base_url, output_path)


def run_attacks_on_gpu(gpu_id: int, examples: list, num_iters: int, num_branches: int, memory: int, K: int, batch_size: int, verbose: bool, use_attacker_api: bool = False, attacker_api_model: str = None, attacker_api_key: str = None, attacker_api_base_url: str = None, output_path: str = None) -> list:
    """Run reprompting attacks on a specific GPU for assigned examples."""
    global _worker_target_model, _worker_target_tokenizer
    global _worker_attacker_model, _worker_attacker_tokenizer, _worker_attacker_name, _worker_gpu_id
    
    if _worker_target_model is None:
        raise RuntimeError(f"Worker target model not initialized on GPU {gpu_id}")
    
    if not use_attacker_api and (_worker_attacker_model is None or _worker_attacker_tokenizer is None):
        raise RuntimeError(f"Worker attacker model not initialized on GPU {gpu_id}")
    
    device = f"cuda:{gpu_id}"
    results = []
    
    print(f"[GPU {gpu_id}] Starting {len(examples)} experiments...")
    
    for example_idx, (df_idx, row) in enumerate(examples):
        goal = row['forbidden_prompt']
        response = row.get('response', '')
        target = extract_target_from_response(response, tokenizer=_worker_target_tokenizer, num_tokens_after_tag=20)
        
        if verbose:
            print(f"[GPU {gpu_id}] Processing example {df_idx} ({example_idx+1}/{len(examples)})")
            print(f"[GPU {gpu_id}] Goal: {goal[:100]}...")
            print(f"[GPU {gpu_id}] Target: {target[:50]}...")
        
        try:
            start_time = time.time()
            
            result = run_reprompting_attack(
                goal=goal,
                target=target,
                model=_worker_target_model,
                tokenizer=_worker_target_tokenizer,
                attacker_model=_worker_attacker_model,
                attacker_tokenizer=_worker_attacker_tokenizer,
                attacker_name=_worker_attacker_name,
                num_iters=num_iters,
                num_branches=num_branches,
                memory=memory,
                K=K,
                batch_size=batch_size,
                device=device,
                use_attacker_api=use_attacker_api,
                attacker_api_model=attacker_api_model,
                attacker_api_key=attacker_api_key,
                attacker_api_base_url=attacker_api_base_url
            )
            
            runtime = time.time() - start_time
            
            result['prompt_idx'] = df_idx
            result['goal'] = goal
            result['target'] = target
            result['original_response'] = response
            result['gpu_id'] = gpu_id
            result['runtime'] = runtime
            
            if verbose:
                print(f"[GPU {gpu_id}] Example {df_idx} completed in {runtime:.1f}s")
                print(f"[GPU {gpu_id}] Initial loss: mean={result['initial_loss_mean']:.4f}, min={result['initial_loss_min']:.4f}")
                for iter_loss in result['iteration_losses']:
                    print(f"[GPU {gpu_id}] Iteration {iter_loss['iteration']}: loss_mean={iter_loss['loss_mean']:.4f}, loss_min={iter_loss['loss_min']:.4f}")
                print(f"[GPU {gpu_id}] Final loss: mean={result['final_loss_mean']:.4f}, min={result['final_loss_min']:.4f}")
                print(f"[GPU {gpu_id}] Loss improvement: {result['initial_loss_min'] - result['final_loss_min']:.4f}")
            
            results.append(result)
            
            # Save result to CSV immediately after processing each example
            if output_path:
                save_result_to_csv(result, output_path)
                if verbose:
                    print(f"[GPU {gpu_id}] Saved result for example {df_idx} to CSV")
            
            # Print GPU memory after each example
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(gpu_id) / 1e9
                reserved = torch.cuda.memory_reserved(gpu_id) / 1e9
                print(f"[GPU {gpu_id}] After example {df_idx}: Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
            
        except Exception as e:
            print(f"[GPU {gpu_id}] Error processing example {df_idx}: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            continue
    
    print(f"[GPU {gpu_id}] Completed {len(results)}/{len(examples)} experiments")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run reprompting attack experiments")
    parser.add_argument("--example-range", help="Example range (e.g., '0.0:0.5' or '0:100')")
    parser.add_argument("--job-id", help="Job ID for parallel execution")
    parser.add_argument("--results-dir", required=True, help="Results directory")
    parser.add_argument("--num-iters", type=int, required=True, help="Number of iterations")
    parser.add_argument("--num-branches", type=int, required=True, help="Number of branches per reasoning string")
    parser.add_argument("--memory", type=int, required=True, help="Buffer size for GWW algorithm")
    parser.add_argument("--K", type=int, required=True, help="Bucket size for randomization")
    parser.add_argument("--batch-size", type=int, required=True, help="Batch size for generation")
    parser.add_argument("--max-examples", type=int, default=-1, help="Maximum examples to process")
    parser.add_argument("--input-csv", required=True, help="Input CSV file")
    parser.add_argument("--model-name", required=True, help="Target model name")
    parser.add_argument("--attacker-model-name", required=True, help="Attacker model name")
    parser.add_argument("--wandb-project", default="reprompting_attacks", help="W&B project")
    parser.add_argument("--wandb-entity", default="bogdan-turbal-y", help="W&B entity")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--wandb-offline", action="store_true", help="W&B offline mode")
    parser.add_argument("--attacker-api", action="store_true", help="Use OpenRouter API for attacker model instead of local model")
    parser.add_argument("--attacker-api-model", help="OpenRouter model identifier (e.g., mistralai/mixtral-8x7b-instruct)")
    parser.add_argument("--attacker-api-key", help="OpenRouter API key (or set OPENROUTER_API_KEY env var)")
    parser.add_argument("--attacker-api-base-url", default="https://openrouter.ai/api/v1", help="OpenRouter base URL")
    parser.add_argument("--attacker-quantize", action="store_true", help="Enable quantization for attacker model (ignored if --attacker-api)")
    parser.add_argument("--attacker-quantize-bits", type=int, default=8, choices=[4, 8], help="Quantization bits for attacker model (4 or 8)")
    parser.add_argument("--attacker-use-flash-attention", action="store_true", help="Enable Flash Attention 2 for attacker model (ignored if --attacker-api)")
    parser.add_argument("--target-quantize", action="store_true", help="Enable quantization for target model (DeepSeek) to reduce memory usage")
    parser.add_argument("--target-quantize-bits", type=int, default=8, choices=[4, 8], help="Quantization bits for target model (4 or 8)")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs to use for parallel processing")
    
    args = parser.parse_args()
    
    # Set API key from environment if not provided via command line
    if args.attacker_api and not args.attacker_api_key:
        args.attacker_api_key = os.environ.get("OPENROUTER_API_KEY")
        if not args.attacker_api_key:
            print("Warning: --attacker-api is set but no API key provided (via --attacker-api-key or OPENROUTER_API_KEY env var)")
    
    # Validate API parameters if API mode is enabled
    if args.attacker_api:
        if not args.attacker_api_model:
            print("Error: --attacker-api requires --attacker-api-model")
            sys.exit(1)
        if not args.attacker_api_key:
            print("Error: --attacker-api requires --attacker-api-key or OPENROUTER_API_KEY environment variable")
            sys.exit(1)
    
    # Load dataset
    df = load_dataset(args.input_csv, args.example_range, args.max_examples)
    
    # Check available GPUs
    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if args.num_gpus > available_gpus:
        print(f"⚠️  Warning: Requested {args.num_gpus} GPUs but only {available_gpus} are available.")
        if available_gpus == 0:
            print("Error: No CUDA GPUs available. Cannot run in multi-GPU mode.")
            sys.exit(1)
        args.num_gpus = available_gpus
        print(f"Using {available_gpus} GPU(s) instead.")
    elif available_gpus > 0:
        print(f"✓ CUDA available: {available_gpus} GPU(s) detected, using {args.num_gpus}")
    else:
        print("⚠️  No CUDA GPUs available, falling back to CPU (will be slow)")
        args.num_gpus = 1
    
    if args.verbose:
        print(f"Loaded {len(df)} examples from {args.input_csv}")
        print(f"Target Model: {args.model_name}")
        print(f"Attacker Model: {args.attacker_model_name}")
        for i in range(args.num_gpus):
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name}, {props.total_memory / 1e9:.2f} GB")
    
    # Create results directory
    results_dir = args.results_dir
    if args.job_id:
        results_dir = os.path.join(results_dir, args.job_id)
    os.makedirs(results_dir, exist_ok=True)
    print(f'Created results directory: {results_dir}')
    
    # Set up output CSV path for incremental saving
    output_path = os.path.join(results_dir, 'results.csv')
    
    # Prepare experiments list (each example becomes an experiment)
    experiments = [(idx, row) for idx, row in df.iterrows()]
    total_examples = len(experiments)
    
    # Distribute experiments across GPUs (uniform distribution like reasoning attacks)
    if args.num_gpus > 1 and torch.cuda.is_available():
        print(f"\nDistributing {total_examples} examples across {args.num_gpus} GPUs...")
        
        # Compute uniform examples per GPU to ensure balanced distribution
        # Similar to reasoning attacks: ensure most GPUs get the same number of examples
        # We use ceil to ensure we have enough capacity, then distribute evenly
        examples_per_gpu = ceil(total_examples / args.num_gpus)
        
        experiments_per_gpu = [[] for _ in range(args.num_gpus)]
        
        # Distribute examples uniformly across GPUs using round-robin
        # This ensures the most balanced distribution: each GPU gets either 
        # floor(total_examples/num_gpus) or ceil(total_examples/num_gpus) examples
        # Priority: distribute evenly in round-robin fashion
        base_examples = total_examples // args.num_gpus
        extra_examples = total_examples % args.num_gpus
        
        current_idx = 0
        
        # Assign examples round-robin: each GPU gets base_examples, 
        # then first 'extra_examples' GPUs get one more
        for gpu_id in range(args.num_gpus):
            examples_for_this_gpu = base_examples + (1 if gpu_id < extra_examples else 0)
            for _ in range(examples_for_this_gpu):
                if current_idx < total_examples:
                    experiments_per_gpu[gpu_id].append(experiments[current_idx])
                    current_idx += 1
        
        print(f"Starting {len(experiments)} experiments across {args.num_gpus} GPUs...")
        for gpu_id in range(args.num_gpus):
            exp_list = experiments_per_gpu[gpu_id]
            if len(exp_list) > 0:
                example_ids = [exp[0] for exp in exp_list]
                if len(example_ids) <= 10:
                    print(f"  GPU {gpu_id}: {len(exp_list)} experiments with example IDs: {example_ids}")
                else:
                    print(f"  GPU {gpu_id}: {len(exp_list)} experiments with example IDs: {example_ids[:5]} ... {example_ids[-5:]} (range {min(example_ids)}-{max(example_ids)})")
            else:
                print(f"  GPU {gpu_id}: 0 experiments")
        
        start_time = time.time()
        
        # Create process pool with one process per GPU
        print("\nCreating multiprocessing pool...")
        with mp.Pool(processes=args.num_gpus) as pool:
            # Prepare arguments for each GPU worker
            gpu_args = [
                (
                    gpu_id,
                    args.model_name,
                    args.attacker_model_name,
                    args.attacker_quantize,
                    args.attacker_quantize_bits,
                    args.attacker_use_flash_attention,
                    experiments_per_gpu[gpu_id],
                    args.num_iters,
                    args.num_branches,
                    args.memory,
                    args.K,
                    args.batch_size,
                    args.verbose,
                    args.attacker_api,
                    args.attacker_api_model,
                    args.attacker_api_key,
                    args.attacker_api_base_url,
                    output_path,
                    args.target_quantize,
                    args.target_quantize_bits
                )
                for gpu_id in range(args.num_gpus)
                if experiments_per_gpu[gpu_id]  # Only include GPUs with experiments
            ]
            
            print(f"Running experiments on {len(gpu_args)} GPUs...")
            # Initialize workers and run experiments in parallel
            all_results = pool.starmap(run_attacks_on_gpu_worker, gpu_args)
        
        # Flatten results from all GPUs
        results = []
        for gpu_results in all_results:
            results.extend(gpu_results)
        
        total_time = time.time() - start_time
        print(f"\n✓ Completed {len(results)} experiments across {args.num_gpus} GPUs in {total_time:.1f}s")
    
    else:
        # Single GPU or CPU mode (sequential processing)
        print(f"\nProcessing {len(experiments)} examples sequentially...")
        
        # Initialize worker on GPU 0 (or CPU)
        if torch.cuda.is_available():
            worker_init(0, args.model_name, args.attacker_model_name, args.attacker_quantize, args.attacker_quantize_bits, args.attacker_use_flash_attention, args.attacker_api, args.target_quantize, args.target_quantize_bits)
            device = "cuda:0"
        else:
            device = "cpu"
            print("Warning: CPU mode - will be very slow!")
        
        attacker_name = "mixtral"
        results = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing examples"):
            goal = row['forbidden_prompt']
            response = row.get('response', '')
            target = extract_target_from_response(response, tokenizer=_worker_target_tokenizer, num_tokens_after_tag=20)
            
            if args.verbose:
                print(f"Processing example {idx+1}/{len(df)}")
                print(f"Goal: {goal[:100]}...")
                print(f"Target: {target[:50]}...")
            
            try:
                result = run_reprompting_attack(
                    goal=goal,
                    target=target,
                    model=_worker_target_model,
                    tokenizer=_worker_target_tokenizer,
                    attacker_model=_worker_attacker_model,
                    attacker_tokenizer=_worker_attacker_tokenizer,
                    attacker_name=_worker_attacker_name,
                    num_iters=args.num_iters,
                    num_branches=args.num_branches,
                    memory=args.memory,
                    K=args.K,
                    batch_size=args.batch_size,
                    device=device,
                    use_attacker_api=args.attacker_api,
                    attacker_api_model=args.attacker_api_model,
                    attacker_api_key=args.attacker_api_key,
                    attacker_api_base_url=args.attacker_api_base_url
                )
                
                result['prompt_idx'] = idx
                result['goal'] = goal
                result['target'] = target
                result['original_response'] = response
                result['gpu_id'] = 0 if torch.cuda.is_available() else -1
                
                if args.verbose:
                    print(f"  Initial loss: mean={result['initial_loss_mean']:.4f}, min={result['initial_loss_min']:.4f}")
                    for iter_loss in result['iteration_losses']:
                        print(f"  Iteration {iter_loss['iteration']}: loss_mean={iter_loss['loss_mean']:.4f}, loss_min={iter_loss['loss_min']:.4f}")
                    print(f"  Final loss: mean={result['final_loss_mean']:.4f}, min={result['final_loss_min']:.4f}")
                    print(f"  Loss improvement: {result['initial_loss_min'] - result['final_loss_min']:.4f}")
                
                results.append(result)
                
                # Save result to CSV immediately after processing each example
                save_result_to_csv(result, output_path)
                if args.verbose:
                    print(f"  Saved result for example {idx} to CSV")
                
            except Exception as e:
                print(f"Error processing example {idx}: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
                continue
    
    # Print final GPU memory
    #print_gpu_memory("[After attack loop completed] ")
    
    # Note: Results are already saved incrementally via save_result_to_csv() after each example
    # No need to overwrite at the end - this preserves results from parallel jobs and prevents data loss
    if results:
        if args.verbose:
            print(f"\nProcessed {len(results)} examples total")
            print(f"Results saved incrementally to {output_path} (append mode)")
            print("Each result was saved immediately after processing to prevent data loss")
    else:
        print("No results to save")


if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    # This must be done before any multiprocessing operations
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # Start method already set (e.g., in parent process), continue
        pass
    main()

