# Requires: pip install torch datasets transformers trl accelerate bitsandbytes
import os
import sys
import torch
from typing import List # Keep List for type hinting reward function
import re # Added for regex in reward function

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig

# Add project root to sys.path if necessary for imports to work in test environment
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))) 

# --- Helper functions for the reward function ---
def extract_gsm8k_ground_truth_answer(answer_str: str) -> str | None:
    """
    Parses strings like "...#### 123" from GSM8K and returns the numerical part.
    """
    match = re.search(r"####\s*([\d\.,]+)", answer_str)
    if match:
        return match.group(1).strip()
    # Fallback: if #### is not present, try to find the last number in the string
    # This might be useful if the format is inconsistent for some reason.
    numbers = re.findall(r"([\d\.,]+)", answer_str)
    if numbers:
        return numbers[-1].strip().rstrip('.')
    return None

def extract_answer_from_completion(completion_str: str) -> str | None:
    """
    Extracts the final numerical answer from a model's completion string.
    Heuristic: looks for the last number.
    """
    # Remove potential "Question:" or "Prompt:" prefix if the model regenerates it
    # completion_str = re.sub(r"^(Question:|Prompt:).*\n", "", completion_str, flags=re.IGNORECASE)

    # Look for numbers (integers or decimals, potentially with commas)
    numbers = re.findall(r"([\d\.,]+)", completion_str)
    if numbers:
        # Return the last number found
        return numbers[-1].strip().rstrip('.') # Remove trailing dots if any
    return None

# --- Main Reward Function ---
def reasoning_reward_function(prompts: list[str], completions: list[str], completions_ids: list[list[int]], **kwargs) -> list[float]:
    """
    Reward function for reasoning tasks, focusing on extracting and comparing final numerical answers.
    kwargs will contain other columns from the dataset, like "answer" (ground truth for GSM8K).
    """
    rewards = []
    ground_truth_answer_texts = kwargs.get("answer", []) 

    # print(f"Reward Function Called. Batch size: {len(prompts)}") # Debug print

    if len(ground_truth_answer_texts) != len(completions):
        print(f"Warning: Mismatch in lengths of completions ({len(completions)}) and ground_truth_answers ({len(ground_truth_answer_texts)}). Returning default rewards.")
        return [0.0] * len(completions)

    for i in range(len(completions)):
        completion_text = completions[i]
        gt_answer_text = ground_truth_answer_texts[i]

        # print(f"\nProcessing item {i}:") # Debug print
        # print(f"  Prompt: {prompts[i][:100]}...") # Debug print
        # print(f"  Completion: {completion_text[:150]}...") # Debug print
        # print(f"  Ground Truth Text: {gt_answer_text[:150]}...") # Debug print

        gt_final_answer = extract_gsm8k_ground_truth_answer(gt_answer_text)
        pred_final_answer = extract_answer_from_completion(completion_text)

        # print(f"  Extracted GT final answer: {gt_final_answer}") # Debug print
        # print(f"  Extracted Predicted final answer: {pred_final_answer}") # Debug print

        reward = 0.0  # Default reward

        if gt_final_answer is not None and pred_final_answer is not None:
            # Normalize for comparison: remove commas, strip whitespace
            gt_final_answer_norm = gt_final_answer.replace(",", "").strip()
            pred_final_answer_norm = pred_final_answer.replace(",", "").strip()
            
            try:
                # Attempt float conversion for robust numerical comparison
                if abs(float(gt_final_answer_norm) - float(pred_final_answer_norm)) < 1e-3: # Tolerance for float comparison
                    reward = 1.0  # Correct final answer
                    # print("    Reward: 1.0 (Correct final answer)") # Debug print
                else:
                    reward = -0.1 # Incorrect final answer (but both parsable)
                    # print("    Reward: -0.1 (Incorrect final answer)") # Debug print
            except ValueError:
                # Fallback to string comparison if not clearly numeric (e.g., if extraction yields non-numeric strings despite regex)
                if gt_final_answer_norm == pred_final_answer_norm:
                    reward = 0.8 # String match, slightly less than perfect float match
                    # print("    Reward: 0.8 (String match on final answer)") # Debug print
                else:
                    reward = -0.2 # Penalize if format is weird and not matching by string
                    # print("    Reward: -0.2 (String mismatch, format/extraction issue)") # Debug print
        elif gt_final_answer is None:
            # Should not happen with well-formatted gsm8k data
            # print("    Reward: -0.5 (Could not parse ground truth answer!)") # Debug print
            reward = -0.5 
        else: # pred_final_answer is None, but gt_final_answer was parsable
            # print("    Reward: -0.3 (Could not parse answer from model completion)") # Debug print
            reward = -0.3 
        
        # Optional: Small penalty for very short completions if the answer was not perfect
        if reward < 1.0 and len(completion_text) < 30: # Arbitrary short length
            # print(f"    Short completion penalty applied. Original reward: {reward}") # Debug print
            reward -= 0.1
            reward = max(-1.0, reward) # Cap minimum reward

        rewards.append(reward)
    
    # print(f"Calculated rewards (batch): {rewards}") # Debug print
    return rewards


def run_grpo_training():
    """
    Main function to run GRPO training.
    """
    # --- 1. Script Configuration ---
    sft_model_path = "./fw14k-gsm8k-sft"
    dataset_name = "openai/gsm8k"
    dataset_config = "main" 
    grpo_output_dir = "./fw14k-gsm8k-sft-grpo-refined-reward" # Updated output dir name
    use_subset = True 
    subset_size = 200 


    print("--- GRPO Training Configuration (Refined Reward) ---")
    print(f"SFT model path: {sft_model_path}")
    print(f"Dataset: {dataset_name} (config: {dataset_config})")
    print(f"GRPO output directory: {grpo_output_dir}")
    print(f"Using subset of dataset: {use_subset} (size: {subset_size if use_subset else 'all'})")
    print("-----------------------------------\n")

    # --- 2. Load SFT Model and Tokenizer ---
    print(f"Loading SFT model from: {sft_model_path}")
    try:
        model = AutoModelForCausalLM.from_pretrained(sft_model_path)
        tokenizer = AutoTokenizer.from_pretrained(sft_model_path)
        print("SFT Model and Tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading SFT model or tokenizer from {sft_model_path}: {e}")
        print("Please ensure the path is correct and the SFT model was saved properly.")
        sys.exit(1)

    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        print("Tokenizer does not have a pad_token. Setting pad_token = eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id 
    
    tokenizer.padding_side = "left" 
    
    if model.config.pad_token_id != tokenizer.pad_token_id:
        print(f"Updating model.config.pad_token_id from {model.config.pad_token_id} to {tokenizer.pad_token_id}")
        model.config.pad_token_id = tokenizer.pad_token_id

    print(f"Tokenizer configured: pad_token='{tokenizer.pad_token}', pad_token_id={tokenizer.pad_token_id}, padding_side='{tokenizer.padding_side}'")
    print("-----------------------------------\n")

    # --- 3. Prepare Dataset for GRPOTrainer ---
    print(f"Loading dataset: {dataset_name} (config: {dataset_config})")
    full_dataset = load_dataset(dataset_name, name=dataset_config, split="train")
    print(f"Full dataset loaded. Number of examples: {len(full_dataset)}")

    def prepare_prompts(example):
        return {"prompt": example["question"], "answer": example["answer"]}

    print("Processing dataset to create 'prompt' and 'answer' columns...")
    dataset_map_num_proc = os.cpu_count() // 2 if os.cpu_count() and os.cpu_count() > 2 else 1
    if use_subset:
        print(f"Selecting a subset of {subset_size} examples for processing.")
        actual_subset_size = min(subset_size, len(full_dataset))
        processed_dataset = full_dataset.select(range(actual_subset_size)).map(
            prepare_prompts, 
            num_proc=dataset_map_num_proc,
            remove_columns=[col for col in full_dataset.column_names if col not in ["question", "answer"]]
        )
    else:
        processed_dataset = full_dataset.map(
            prepare_prompts,
            num_proc=dataset_map_num_proc,
            remove_columns=[col for col in full_dataset.column_names if col not in ["question", "answer"]]
        )
    
    print(f"Dataset processed. Number of examples in processed_dataset: {len(processed_dataset)}")
    print(f"Columns in processed_dataset: {processed_dataset.column_names}")
    if len(processed_dataset) > 0:
        print(f"First example - Prompt: {processed_dataset[0]['prompt'][:100]}...")
        # print(f"First example - Answer (for reward): {processed_dataset[0]['answer'][:100]}...") # For debugging
    print("-----------------------------------\n")
    
    # --- 4. Initialize GRPOConfig ---
    print("Initializing GRPOConfig...")
    config = GRPOConfig(
        output_dir=grpo_output_dir,
        learning_rate=5e-7, 
        batch_size=16,      
        mini_batch_size=2,  
        gradient_accumulation_steps=4, 
        log_with="none",    
        kl_penalty="kl",    
        beta=0.02,          
        adap_kl_ctrl=False, 
        max_prompt_length=512,      
        max_completion_length=512,  
        logging_steps=5, # Log more frequently with refined reward
        save_steps=10, # Save more frequently for smaller datasets/testing
        ppo_epochs=2,       
        num_iterations=2, # Keep low for quick test with refined reward
    )
    print("GRPOConfig initialized.")
    print(f"  Output directory: {config.output_dir}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Num iterations: {config.num_iterations}")
    print("-----------------------------------\n")

    # --- 5. Instantiate GRPOTrainer ---
    print("Instantiating GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=processed_dataset, 
        reward_funcs=[reasoning_reward_function], 
        processing_class=tokenizer, 
    )
    print("GRPOTrainer instantiated.")
    print("-----------------------------------\n")

    # --- 6. Training Loop ---
    print("Starting GRPO training with refined reward function...")
    try:
        trainer.train() 
        print("GRPO training completed.")
    except Exception as e:
        print(f"An error occurred during GRPOTrainer.train(): {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 
    print("-----------------------------------\n")

    # --- 7. Save Model ---
    print(f"Saving GRPO fine-tuned model to {grpo_output_dir}...")
    try:
        trainer.save_model(grpo_output_dir)
        print(f"GRPO fine-tuned model (and tokenizer) saved successfully to {grpo_output_dir}")
    except Exception as e:
        print(f"An error occurred during model saving: {e}")
        import traceback
        traceback.print_exc()
    
    print("-----------------------------------\n")
    print("GRPO training script finished.")


if __name__ == '__main__':
    print("Executing GRPO training script: train-rl.py (with refined reward function)")
    run_grpo_training()
    print("Script execution complete.")
