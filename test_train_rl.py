# Requires: pip install pytest transformers datasets torch accelerate bitsandbytes
import pytest
import torch
from transformers import AutoTokenizer

# Assuming model.py and train_rl.py are in the same directory or in PYTHONPATH
# Add project root to sys.path if necessary for imports to work in test environment
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))) 

# Import functions to be tested from train_rl.py (refactored for GRPOTrainer)
from train_rl import (
    extract_gsm8k_ground_truth_answer,
    extract_answer_from_completion,
    reasoning_reward_function,
    # prepare_prompts function is defined inside run_grpo_training, 
    # so we'll define a similar one for testing or test its effect via dataset mapping.
    # For now, let's define a test version or assume its logic is simple.
)

# --- Fixtures ---

@pytest.fixture(scope="module") 
def tokenizer_fixture():
    """Fixture to provide a Hugging Face tokenizer."""
    try:
        # Using a common, small tokenizer for tests to avoid large downloads if not cached
        tokenizer_name = "gpt2" # "KoboldAI/llama2-tokenizer" is larger
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        # GPT2 specific BOS/EOS if needed, though not strictly required for reward function tests
        # if tokenizer.bos_token_id is None: tokenizer.bos_token_id = tokenizer.eos_token_id 
        # if tokenizer.eos_token_id is None: tokenizer.eos_token_id = 50256
        return tokenizer
    except Exception as e:
        pytest.skip(f"Failed to load tokenizer {tokenizer_name}, skipping tests that need it: {e}")

@pytest.fixture(scope="session") 
def device_fixture():
    """Fixture to provide the device for torch tensors."""
    return "cuda" if torch.cuda.is_available() else "cpu"

# --- Tests for Reward Function Components ---

def test_extract_gsm8k_ground_truth_answer_logic():
    """Test extraction of ground truth answers from GSM8K format."""
    assert extract_gsm8k_ground_truth_answer("The answer is #### 123") == "123"
    assert extract_gsm8k_ground_truth_answer("Some other text #### 456.78 then more.") == "456.78"
    assert extract_gsm8k_ground_truth_answer("Final Answer: #### 1,000") == "1,000"
    assert extract_gsm8k_ground_truth_answer("The final answer is ####3.14.") == "3.14" # Space variation
    assert extract_gsm8k_ground_truth_answer("The final answer is ####  0.5  .") == "0.5" # More spaces and trailing dot
    
    # Fallback logic tests (if implemented in train_rl.py to find last number)
    assert extract_gsm8k_ground_truth_answer("The answer is 123") == "123" # Assuming fallback
    assert extract_gsm8k_ground_truth_answer("Answer: 456.78.") == "456.78" # Assuming fallback
    assert extract_gsm8k_ground_truth_answer("No #### but result 1,000.") == "1,000" # Assuming fallback

    # Negative cases
    assert extract_gsm8k_ground_truth_answer("No number here ####") is None
    assert extract_gsm8k_ground_truth_answer("Only text here") is None
    assert extract_gsm8k_ground_truth_answer("#### (no number after)") is None
    assert extract_gsm8k_ground_truth_answer("The price is $50 but this is not the format.") == "50" # Fallback picks last number

def test_extract_answer_from_completion_logic():
    """Test extraction of numerical answers from model completions."""
    assert extract_answer_from_completion("The model thinks the answer is 123.") == "123"
    assert extract_answer_from_completion("So, the final value is 456.78.") == "456.78"
    assert extract_answer_from_completion("It might be 1,000, I guess.") == "1,000"
    assert extract_answer_from_completion("The result is 3.14") == "3.14"
    assert extract_answer_from_completion("The result is 0.5 .") == "0.5" # Trailing dot
    assert extract_answer_from_completion("The result is 5,000.00.") == "5,000.00"

    # Cases with multiple numbers (should pick the last one)
    assert extract_answer_from_completion("First try 1, then try 2, finally 3.") == "3"
    assert extract_answer_from_completion("Step 1: 10. Step 2: 20. Final answer: 30.") == "30"

    # Negative cases
    assert extract_answer_from_completion("There is no number here.") is None
    assert extract_answer_from_completion("The answer is five.") is None # Word, not number
    assert extract_answer_from_completion("The model is unsure.") is None

# --- Tests for reasoning_reward_function ---

def test_reasoning_reward_function_correct_answer():
    prompts = ["What is 2+2?"]
    completions = ["The answer is <<2+2=4>>4."] # Completion contains "4"
    # Dummy completions_ids, not used by this reward function logic
    completions_ids = [[101, 102]] 
    kwargs = {"answer": ["Question: What is 2+2?\nAnswer: The final answer is #### 4"]}
    
    rewards = reasoning_reward_function(prompts, completions, completions_ids, **kwargs)
    assert len(rewards) == 1
    assert rewards[0] == 1.0, "Reward should be 1.0 for correct numerical answer"

def test_reasoning_reward_function_correct_answer_float():
    prompts = ["What is 10/4?"]
    completions = ["The answer is 2.5"] 
    completions_ids = [[101, 102]] 
    kwargs = {"answer": ["Question: What is 10/4?\nAnswer: The final answer is #### 2.5"]}
    
    rewards = reasoning_reward_function(prompts, completions, completions_ids, **kwargs)
    assert len(rewards) == 1
    assert rewards[0] == 1.0, "Reward should be 1.0 for correct float answer"

def test_reasoning_reward_function_incorrect_answer():
    prompts = ["What is 2+2?"]
    completions = ["I think it is <<2+2=5>>5."] # Completion contains "5"
    completions_ids = [[101, 102]]
    kwargs = {"answer": ["Question: What is 2+2?\nAnswer: The final answer is #### 4"]}
    
    rewards = reasoning_reward_function(prompts, completions, completions_ids, **kwargs)
    assert len(rewards) == 1
    assert rewards[0] < 0.0, "Reward should be negative for incorrect answer (e.g., -0.1)"
    assert rewards[0] == -0.1, "Expected -0.1 for incorrect but parsable answer"

def test_reasoning_reward_function_no_answer_in_completion():
    prompts = ["What is 2+2?"]
    completions = ["I am not sure about that calculation."] # No numerical answer
    completions_ids = [[101, 102]]
    kwargs = {"answer": ["Question: What is 2+2?\nAnswer: The final answer is #### 4"]}
    
    rewards = reasoning_reward_function(prompts, completions, completions_ids, **kwargs)
    assert len(rewards) == 1
    # Expected penalty for not finding an answer in completion, e.g., -0.3
    # This might also be affected by short completion penalty if applicable
    expected_reward = -0.3 
    if len(completions[0]) < 30: # Check if short completion penalty applies
        expected_reward -= 0.1
    assert rewards[0] == pytest.approx(expected_reward), "Reward should reflect penalty for no parsable answer in completion"


def test_reasoning_reward_function_gt_missing_format():
    prompts = ["What is 2+2?"]
    completions = ["The answer is 4."] # Model gives correct number
    completions_ids = [[101, 102]]
    # Ground truth is missing the "#### <number>" format but has a fallback number
    kwargs = {"answer": ["Question: What is 2+2?\nAnswer: The final answer is four, which is 4."]} 
    
    rewards = reasoning_reward_function(prompts, completions, completions_ids, **kwargs)
    assert len(rewards) == 1
    # If GT fallback extracts "4", and completion extracts "4", reward should be 1.0
    assert rewards[0] == 1.0, "Reward should be 1.0 if GT fallback and completion match"

def test_reasoning_reward_function_gt_unparsable():
    prompts = ["What is 2+2?"]
    completions = ["The answer is 4."]
    completions_ids = [[101, 102]]
    # Ground truth is unparsable by extract_gsm8k_ground_truth_answer
    kwargs = {"answer": ["Question: What is 2+2?\nAnswer: It's just four."]} 
    
    rewards = reasoning_reward_function(prompts, completions, completions_ids, **kwargs)
    assert len(rewards) == 1
    # Expected penalty for GT answer not being parsable, e.g., -0.5
    assert rewards[0] == -0.5, "Reward should reflect penalty for unparsable GT answer"


def test_reasoning_reward_function_batch():
    prompts = ["Q1: 2+2?", "Q2: 3*3?", "Q3: 10/2?"]
    completions = ["It is 4.", "The result is 9", "I think it is 6."]
    completions_ids = [[1,2],[3,4],[5,6]] # Dummy
    kwargs = {
        "answer": [
            "Q1: Answer is #### 4",
            "Q2: Answer is #### 9",
            "Q3: Answer is #### 5" 
        ]
    }
    rewards = reasoning_reward_function(prompts, completions, completions_ids, **kwargs)
    assert len(rewards) == 3
    assert rewards[0] == 1.0 # Q1: Correct
    assert rewards[1] == 1.0 # Q2: Correct
    assert rewards[2] == -0.1 # Q3: Incorrect (6 vs 5)

def test_reasoning_reward_function_mismatched_lengths_kwargs():
    """Test behavior when completions and kwargs['answer'] have different lengths."""
    prompts = ["Q1", "Q2"]
    completions = ["Ans1", "Ans2"]
    completions_ids = [[1],[2]]
    kwargs_mismatched = {"answer": ["GT_Ans1"]} # Only one GT answer for two completions

    rewards = reasoning_reward_function(prompts, completions, completions_ids, **kwargs_mismatched)
    assert len(rewards) == len(completions), "Should return rewards for all completions"
    # Expect default reward (e.g., 0.0) for all if it cannot process due to mismatch
    assert all(r == 0.0 for r in rewards), "Expected default rewards for mismatched lengths"

# --- Tests for Dataset Preparation Logic ---
# The prepare_prompts function is defined within run_grpo_training in train_rl.py.
# To test it directly, we either need to extract it or replicate its simple logic here.
# For this test, we'll replicate its core transformation.

def test_prepare_prompts_mapping_replication():
    """
    Tests the logic similar to prepare_prompts used in dataset.map().
    """
    def prepare_prompts_for_test(example): # Replicating the mapping function
        return {"prompt": example["question"], "answer": example["answer"]}

    dummy_gsm8k_example = {"question": "What is 3 multiplied by 7?", 
                           "answer": "To find 3 multiplied by 7, we calculate 3 * 7 = 21. #### 21",
                           "extra_info": "This is a multiplication problem."}
    
    processed_example = prepare_prompts_for_test(dummy_gsm8k_example)
    
    assert "prompt" in processed_example, "Output should have a 'prompt' key"
    assert "answer" in processed_example, "Output should have an 'answer' key"
    assert processed_example["prompt"] == dummy_gsm8k_example["question"], "'prompt' field mismatch"
    assert processed_example["answer"] == dummy_gsm8k_example["answer"], "'answer' field mismatch"
    
    # Check that other columns are not implicitly carried over by this simple function
    # (though the .map(remove_columns=...) in train_rl.py handles the final dataset structure)
    assert "extra_info" not in processed_example, "Other columns should not be in the output of this specific function"
    assert "question" not in processed_example, "Original 'question' key should be replaced by 'prompt' or removed by map in actual script"

# Note: A more integrated test for dataset preparation would involve
# creating a dummy Hugging Face Dataset object and applying the .map()
# call with the actual prepare_prompts function and remove_columns argument
# from train_rl.py. However, that's closer to an integration test.
# This unit test focuses on the transformation logic itself.

# Removed GRPOTrainer instantiation test as it would require significant mocking
# or actual model/config setup, making it more of an integration test.
# The focus here is on unit testing the reward logic and data prep function.
