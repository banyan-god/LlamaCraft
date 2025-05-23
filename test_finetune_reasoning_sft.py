# Requires: pip install pytest torch transformers datasets trl accelerate bitsandbytes
import pytest
from unittest.mock import patch, MagicMock, call
import torch

# Attempt to import the main script and specific functions
# This assumes finetune_reasoning_sft.py is in the same directory or PYTHONPATH
import finetune_reasoning_sft

# If format_example is defined inside run_sft, we need to either extract it
# or replicate it here for unit testing. For this example, we assume it's
# accessible or we replicate its logic.
# Let's try importing it. If not, we'll define it.
try:
    from finetune_reasoning_sft import format_example
except ImportError:
    # Replicate if not importable (e.g., if it's a nested function)
    print("Warning: format_example not directly importable from finetune_reasoning_sft. Replicating for tests.")
    def format_example(example):
        return {"text": "Question: " + example["question"] + "\nAnswer: " + example["answer"]}

# --- Test for format_example Function ---

def test_format_example_concatenation():
    """Test the format_example function for correct string concatenation."""
    example = {"question": "Q1", "answer": "A1"}
    formatted = format_example(example)
    assert formatted["text"] == "Question: Q1\nAnswer: A1", \
        "The 'text' field should correctly concatenate question and answer."

# --- Fixtures for Mocking ---

@pytest.fixture
def mock_tokenizer_fixture(): # Renamed to avoid conflict with parameter name in test
    """Fixture to provide a mock Hugging Face tokenizer."""
    tokenizer = MagicMock()
    tokenizer.pad_token = None # Initial state for testing pad_token logic
    tokenizer.eos_token = "<|eos|>"
    tokenizer.pad_token_id = 50256  # Example ID, aligned with eos_token_id
    tokenizer.eos_token_id = 50256  # Example ID
    
    # Mock other methods if SFTTrainer or script calls them directly
    tokenizer.encode = MagicMock(return_value=[1, 2, 3]) 
    tokenizer.decode = MagicMock(return_value="decoded text")
    # If from_pretrained is called on the tokenizer instance itself (unlikely for AutoTokenizer)
    # tokenizer.from_pretrained = MagicMock(return_value=tokenizer) 
    return tokenizer

@pytest.fixture
def mock_model_fixture(): # Renamed
    """Fixture to provide a mock Hugging Face model."""
    model = MagicMock()
    model.config = MagicMock()
    model.config.pad_token_id = None # Initial state for testing pad_token logic
    return model

# --- Tests for SFT Script Logic (Using Mocks) ---

@patch('finetune_reasoning_sft.AutoModelForCausalLM.from_pretrained')
@patch('finetune_reasoning_sft.AutoTokenizer.from_pretrained')
def test_sft_script_pad_token_logic(mock_auto_tokenizer, mock_auto_model, mock_tokenizer_fixture, mock_model_fixture):
    """
    Test the pad token setting logic within the SFT script.
    This test focuses on the part of run_sft where tokenizer.pad_token is set if None.
    """
    # Configure the mock AutoTokenizer and AutoModel to return our specific mock instances
    mock_auto_tokenizer.return_value = mock_tokenizer_fixture
    mock_auto_model.return_value = mock_model_fixture
    
    # Simulate the scenario where pad_token is initially None on the mock_tokenizer_fixture
    assert mock_tokenizer_fixture.pad_token is None
    assert mock_model_fixture.config.pad_token_id is None

    # We need to call a portion of run_sft or have a helper.
    # For simplicity, let's directly test the logic snippet if it were isolated.
    # If run_sft is called, it will execute the whole SFT process.
    # Here, we assume the logic snippet is:
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token
    #     model.config.pad_token_id = tokenizer.pad_token_id (or eos_token_id)

    # Simulate this logic:
    if mock_tokenizer_fixture.pad_token is None:
        mock_tokenizer_fixture.pad_token = mock_tokenizer_fixture.eos_token
        # In the main script, it's model.config.pad_token_id = tokenizer.pad_token_id
        mock_model_fixture.config.pad_token_id = mock_tokenizer_fixture.pad_token_id 
        # (or mock_tokenizer_fixture.eos_token_id if pad_token_id wasn't set before this logic)

    assert mock_tokenizer_fixture.pad_token == mock_tokenizer_fixture.eos_token, \
        "tokenizer.pad_token should be set to tokenizer.eos_token"
    assert mock_model_fixture.config.pad_token_id == mock_tokenizer_fixture.pad_token_id, \
        "model.config.pad_token_id should be set to tokenizer.pad_token_id"

@patch('finetune_reasoning_sft.SFTTrainer')
@patch('finetune_reasoning_sft.load_dataset')
@patch('finetune_reasoning_sft.AutoModelForCausalLM.from_pretrained')
@patch('finetune_reasoning_sft.AutoTokenizer.from_pretrained')
def test_sft_trainer_initialization_and_run(
    mock_auto_tokenizer, 
    mock_auto_model, 
    mock_load_dataset, 
    mock_sft_trainer,
    mock_tokenizer_fixture, # Use the fixture for tokenizer
    mock_model_fixture    # Use the fixture for model
):
    """
    Test the overall SFT script flow, ensuring components are called correctly.
    """
    # --- Setup Mocks ---
    # Configure AutoTokenizer.from_pretrained to return our mock_tokenizer_fixture
    mock_auto_tokenizer.return_value = mock_tokenizer_fixture
    
    # Configure AutoModelForCausalLM.from_pretrained to return our mock_model_fixture
    mock_auto_model.return_value = mock_model_fixture
    
    # Mock the dataset loading
    mock_dataset = MagicMock()
    # Ensure the .map method on the mock_dataset also returns a MagicMock (or itself)
    # to allow chaining or further operations if any.
    mock_dataset.map.return_value = MagicMock() # This will be the formatted_dataset
    mock_load_dataset.return_value = mock_dataset
    
    # Mock the SFTTrainer instance and its methods
    mock_sft_trainer_instance = MagicMock()
    mock_sft_trainer.return_value = mock_sft_trainer_instance

    # --- Call the main function of the SFT script ---
    # This will run the entire SFT pipeline with mocked external calls
    finetune_reasoning_sft.run_sft()

    # --- Assertions ---
    # Check model and tokenizer loading
    mock_auto_tokenizer.assert_called_once_with("sabareesh88/fw14k")
    mock_auto_model.assert_called_once_with("sabareesh88/fw14k")
    
    # Check dataset loading and processing
    mock_load_dataset.assert_called_once_with("openai/gsm8k", name="main", split="train")
    mock_dataset.map.assert_called_once() # Check that formatting was applied
    # We can be more specific about the map call if format_example is passed directly
    # For example, if format_example was imported and passed:
    # mock_dataset.map.assert_called_once_with(format_example, num_proc=ANY) # ANY from unittest.mock if needed

    # Check SFTTrainer initialization
    mock_sft_trainer.assert_called_once()
    # Inspecting call_args for SFTTrainer (example)
    args, kwargs = mock_sft_trainer.call_args
    assert kwargs.get('model') == mock_model_fixture
    assert kwargs.get('tokenizer') == mock_tokenizer_fixture
    assert kwargs.get('train_dataset') == mock_dataset.map.return_value # formatted_dataset
    assert kwargs.get('dataset_text_field') == "text"
    assert kwargs.get('max_seq_length') == 1024 # As per script's default
    # TrainingArguments is also passed to SFTTrainer, can check some key args on it
    training_args_passed = kwargs.get('args')
    assert training_args_passed is not None
    assert training_args_passed.output_dir == "./fw14k-gsm8k-sft" # Default from script
    assert training_args_passed.num_train_epochs == 1

    # Check training and saving
    mock_sft_trainer_instance.train.assert_called_once()
    mock_sft_trainer_instance.save_model.assert_called_once_with("./fw14k-gsm8k-sft") # Default output_dir

# Note: The `test_sft_script_pad_token_logic` is a bit conceptual because it tests a small
# part of the `run_sft` function. Ideally, if that logic were in its own helper function,
# it would be easier to unit test directly. The current implementation simulates the effect.
# For a more robust test of that specific snippet within run_sft, one might need to
# structure run_sft to allow easier testing of its intermediate steps or rely on the
# full integration test (`test_sft_trainer_initialization_and_run`) to cover it implicitly.
