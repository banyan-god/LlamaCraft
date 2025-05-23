# Requires: pip install pytest transformers sentencepiece torch
import pytest
import torch
from transformers import AutoTokenizer
import torch.optim as optim # For checking optimizer in GRPOPolicy

# Assuming model.py and train_rl.py are in the same directory or in PYTHONPATH
# Add project root to sys.path if necessary for imports to work in test environment
import sys
import os
# This assumes the tests are run from the root of the repository or that model.py and train_rl.py are findable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))) 

from model import Transformer, ModelArgs
from train_rl import RLEnvironment, load_pretrained_model, GRPOPolicy

@pytest.fixture(scope="module") 
def tokenizer_fixture():
    """Fixture to provide a Hugging Face tokenizer."""
    try:
        tokenizer_name = "KoboldAI/llama2-tokenizer"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.bos_token_id is None: 
             if hasattr(tokenizer, 'bos_token') and isinstance(tokenizer.bos_token, str):
                tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.bos_token)
             if tokenizer.bos_token_id is None: 
                tokenizer.bos_token_id = 1 
        if tokenizer.eos_token_id is None: 
            if hasattr(tokenizer, 'eos_token') and isinstance(tokenizer.eos_token, str):
                tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
            if tokenizer.eos_token_id is None:
                tokenizer.eos_token_id = 2 
        return tokenizer
    except Exception as e:
        pytest.skip(f"Failed to load tokenizer {tokenizer_name}, skipping tests that need it: {e}")


@pytest.fixture(scope="module")
def dummy_model_fixture(tokenizer_fixture):
    """Fixture to provide a dummy Transformer model."""
    model_args = ModelArgs(
        dim=64, 
        n_layers=1, 
        n_heads=2, 
        vocab_size=tokenizer_fixture.vocab_size, 
        max_seq_len=128 
    )
    model = Transformer(model_args)
    # Ensure model parameters require gradients for update test
    for param in model.parameters():
        param.requires_grad_(True)
    return model

@pytest.fixture(scope="session") 
def device_fixture():
    """Fixture to provide the device for torch tensors."""
    return "cuda" if torch.cuda.is_available() else "cpu"

@pytest.fixture(scope="function") # Use function scope for tmp_path to get a fresh dir each test
def dummy_checkpoint_fixture(tmp_path, dummy_model_fixture):
    """Fixture to create and save a dummy checkpoint."""
    dummy_model = dummy_model_fixture
    # model_args should be a dict, vars() works well for dataclasses like ModelArgs
    model_args_dict = vars(dummy_model.params) 
    
    checkpoint = {
        "model": dummy_model.state_dict(),
        "model_args": model_args_dict,
        "iter_num": 100, # Example additional data
        "best_val_loss": 0.5, # Example additional data
        "config": {"lr": 0.001} # Example additional data
    }
    
    ckpt_path = tmp_path / "dummy_ckpt.pt"
    torch.save(checkpoint, ckpt_path)
    
    return str(ckpt_path), model_args_dict # Return path as string and original args


# --- Tests for RLEnvironment ---

def test_env_initialization(dummy_model_fixture, tokenizer_fixture, device_fixture):
    """Test RLEnvironment initialization."""
    env = RLEnvironment(model_for_max_len=dummy_model_fixture, tokenizer=tokenizer_fixture, device=device_fixture)
    assert env.tokenizer == tokenizer_fixture
    assert env.device == device_fixture
    expected_max_seq_len = dummy_model_fixture.params.max_seq_len
    if hasattr(tokenizer_fixture, 'model_max_length') and tokenizer_fixture.model_max_length < 10000:
         expected_max_seq_len = tokenizer_fixture.model_max_length
    assert env.max_seq_len == expected_max_seq_len
    assert env.get_action_space_size() == tokenizer_fixture.vocab_size

def test_env_reset_empty_prompt(dummy_model_fixture, tokenizer_fixture, device_fixture):
    """Test RLEnvironment reset with an empty prompt."""
    env = RLEnvironment(model_for_max_len=dummy_model_fixture, tokenizer=tokenizer_fixture, device=device_fixture)
    state = env.reset(initial_prompt_text="")
    assert isinstance(state, torch.Tensor)
    assert state.device.type == device_fixture
    if tokenizer_fixture.bos_token_id is not None:
        assert state.tolist() == [[tokenizer_fixture.bos_token_id]]
    else:
        assert state.nelement() > 0

def test_env_reset_with_prompt(dummy_model_fixture, tokenizer_fixture, device_fixture):
    """Test RLEnvironment reset with a specific prompt."""
    env = RLEnvironment(model_for_max_len=dummy_model_fixture, tokenizer=tokenizer_fixture, device=device_fixture)
    prompt = "Hello world"
    state = env.reset(initial_prompt_text=prompt)
    expected_ids = tokenizer_fixture.encode(prompt, add_special_tokens=True, return_tensors="pt")
    assert isinstance(state, torch.Tensor)
    assert torch.equal(state, expected_ids.to(device_fixture))
    assert state.device.type == device_fixture

def test_env_step(dummy_model_fixture, tokenizer_fixture, device_fixture):
    """Test a single step in the RLEnvironment."""
    env = RLEnvironment(model_for_max_len=dummy_model_fixture, tokenizer=tokenizer_fixture, device=device_fixture)
    initial_state = env.reset(initial_prompt_text="Test")
    initial_len = initial_state.size(1)
    action = 50 
    if action == tokenizer_fixture.eos_token_id: action = 51
    if action == tokenizer_fixture.eos_token_id: action = 52 # Highly unlikely
    next_state, reward, done = env.step(action)
    assert isinstance(next_state, torch.Tensor)
    assert next_state.size(1) == initial_len + 1
    assert next_state[0, -1].item() == action
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    if env.max_seq_len > initial_len + 1 and (tokenizer_fixture.eos_token_id is None or action != tokenizer_fixture.eos_token_id):
        assert not done

def test_env_step_done_eos(dummy_model_fixture, tokenizer_fixture, device_fixture):
    """Test if 'done' is True when EOS token is the action."""
    env = RLEnvironment(model_for_max_len=dummy_model_fixture, tokenizer=tokenizer_fixture, device=device_fixture)
    env.reset() 
    if tokenizer_fixture.eos_token_id is None:
        pytest.skip("Tokenizer does not have an EOS token ID defined.")
    action = tokenizer_fixture.eos_token_id
    _, _, done = env.step(action)
    assert done

def test_env_step_done_max_length(dummy_model_fixture, tokenizer_fixture, device_fixture):
    """Test if 'done' is True when max sequence length is reached."""
    test_max_len = 5 
    env = RLEnvironment(model_for_max_len=dummy_model_fixture, tokenizer=tokenizer_fixture, device=device_fixture, max_seq_len=test_max_len)
    env.reset(initial_prompt_text="Hi") 
    current_len = env.current_state.size(1)
    non_eos_action = 100 
    if tokenizer_fixture.eos_token_id is not None and non_eos_action == tokenizer_fixture.eos_token_id:
        non_eos_action = 101
    for _ in range(test_max_len - current_len - 1):
        _, _, step_done = env.step(non_eos_action)
        assert not step_done
    assert env.current_state.size(1) == test_max_len - 1
    _, _, done = env.step(non_eos_action)
    assert env.current_state.size(1) == test_max_len
    assert done

def test_calculate_reward_basic(dummy_model_fixture, tokenizer_fixture, device_fixture):
    """Basic test for the calculate_reward method."""
    env = RLEnvironment(model_for_max_len=dummy_model_fixture, tokenizer=tokenizer_fixture, device=device_fixture)
    env.reset() 
    dummy_current_state = torch.tensor([[10, 20, 30]], dtype=torch.long, device=device_fixture)
    action = 40 
    if tokenizer_fixture.eos_token_id is not None and action == tokenizer_fixture.eos_token_id: action = 41
    reward = env.calculate_reward(dummy_current_state, action)
    assert isinstance(reward, float)
    if tokenizer_fixture.eos_token_id is None or action != tokenizer_fixture.eos_token_id:
         assert reward == 0.1

def test_calculate_reward_eos(dummy_model_fixture, tokenizer_fixture, device_fixture):
    """Test reward calculation for EOS action."""
    env = RLEnvironment(model_for_max_len=dummy_model_fixture, tokenizer=tokenizer_fixture, device=device_fixture)
    env.reset()
    dummy_current_state = torch.tensor([[10, 20, 30]], dtype=torch.long, device=device_fixture)
    if tokenizer_fixture.eos_token_id is None:
        pytest.skip("Tokenizer does not have an EOS token ID defined for this test.")
    action = tokenizer_fixture.eos_token_id
    reward = env.calculate_reward(dummy_current_state, action)
    assert isinstance(reward, float)
    assert reward == 0.0

def test_calculate_reward_repetition(dummy_model_fixture, tokenizer_fixture, device_fixture):
    """Test reward calculation for repetitive sequence."""
    env = RLEnvironment(model_for_max_len=dummy_model_fixture, tokenizer=tokenizer_fixture, device=device_fixture)
    env.reset()
    repetitive_state = torch.tensor([[10, 20, 30, 10, 20, 30]], dtype=torch.long, device=device_fixture)
    action = 40 
    if tokenizer_fixture.eos_token_id is not None and action == tokenizer_fixture.eos_token_id: action = 41
    reward = env.calculate_reward(repetitive_state, action)
    assert isinstance(reward, float)
    assert reward == 0.1 - 0.5

# --- Tests for load_pretrained_model ---

def test_load_pretrained_model(dummy_checkpoint_fixture, device_fixture):
    """Test loading a model from a dummy checkpoint."""
    ckpt_path, original_model_args_dict = dummy_checkpoint_fixture
    
    loaded_model = load_pretrained_model(checkpoint_path=ckpt_path, device=device_fixture)
    
    assert loaded_model is not None, "Loaded model should not be None"
    assert isinstance(loaded_model, Transformer), "Loaded model should be an instance of Transformer"
    
    # Compare ModelArgs parameters
    assert loaded_model.params.dim == original_model_args_dict['dim'], "Dimension mismatch"
    assert loaded_model.params.vocab_size == original_model_args_dict['vocab_size'], "Vocab size mismatch"
    assert loaded_model.params.n_layers == original_model_args_dict['n_layers'], "Number of layers mismatch"
    
    # Perform a basic forward pass
    # Ensure input tensor is on the same device as the model
    dummy_input = torch.tensor([[1, 2, 3]], dtype=torch.long).to(device_fixture)
    try:
        logits, loss = loaded_model(dummy_input) # Model might return loss as well if targets are passed
        assert logits is not None, "Logits should not be None after forward pass"
        # If your model's forward pass always returns loss (even if None), that's fine.
        # If it only returns loss when targets are provided, this test is okay.
    except Exception as e:
        pytest.fail(f"Loaded model failed forward pass: {e}")

# --- Tests for GRPOPolicy ---

def test_grpo_policy_initialization(dummy_model_fixture, tokenizer_fixture, device_fixture):
    """Test GRPOPolicy initialization."""
    model = dummy_model_fixture.to(device_fixture) # Ensure model is on the correct device
    tokenizer = tokenizer_fixture
    
    policy = GRPOPolicy(model=model, tokenizer=tokenizer, device=device_fixture, learning_rate=1e-4)
    
    assert policy.model == model, "Model not set correctly in policy"
    assert policy.device == device_fixture, "Device not set correctly in policy"
    assert policy.tokenizer == tokenizer, "Tokenizer not set correctly in policy"
    assert isinstance(policy.optimizer, optim.AdamW), "Optimizer is not AdamW"
    
    # Check if optimizer has parameters
    assert len(policy.optimizer.param_groups) > 0, "Optimizer has no param groups"
    assert len(policy.optimizer.param_groups[0]['params']) > 0, "Optimizer has no parameters in the first group"
    # Check that a model parameter is indeed part of the optimizer's parameters
    assert any(model_param is opt_param for opt_param in policy.optimizer.param_groups[0]['params'] for model_param in model.parameters()), "Model parameters not in optimizer"


def test_grpo_policy_select_action(dummy_model_fixture, tokenizer_fixture, device_fixture):
    """Test GRPOPolicy select_action method."""
    model = dummy_model_fixture.to(device_fixture)
    tokenizer = tokenizer_fixture
    policy = GRPOPolicy(model=model, tokenizer=tokenizer, device=device_fixture)
    
    prompt = "Test prompt for action selection"
    # Encode prompt and ensure it's on the correct device
    state = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to(device_fixture)
    
    # Handle case where encoding might result in an empty tensor (e.g. if prompt is only special tokens that are filtered)
    if state.nelement() == 0 or state.size(1) == 0:
        if tokenizer.bos_token_id is not None:
            state = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long, device=device_fixture)
        else: # Fallback if BOS is also None
            state = torch.tensor([[1]], dtype=torch.long, device=device_fixture) # Use a generic token

    action, log_prob = policy.select_action(state)
    
    assert isinstance(action, int), "Action should be an integer"
    assert 0 <= action < tokenizer.vocab_size, f"Action {action} out of vocab range [0, {tokenizer.vocab_size-1}]"
    assert isinstance(log_prob, torch.Tensor), "Log probability should be a torch.Tensor"
    assert log_prob.shape == (), "Log probability should be a scalar tensor"
    assert log_prob.device.type == device_fixture, "Log probability is on the wrong device"

def test_grpo_policy_update_basic(dummy_model_fixture, tokenizer_fixture, device_fixture):
    """Test basic functionality of GRPOPolicy update_policy method."""
    model = dummy_model_fixture.to(device_fixture)
    tokenizer = tokenizer_fixture
    
    # Ensure model is in training mode for the update
    model.train()
    
    policy = GRPOPolicy(model=model, tokenizer=tokenizer, device=device_fixture, learning_rate=1e-3)
    
    # Get initial state of a parameter to check if it changes
    # Accessing a specific parameter, e.g., from the first attention layer's query matrix
    # This might need adjustment if model structure changes.
    param_to_check = model.layers[0].attention.wq.weight
    initial_param_value = param_to_check.clone().detach()
    
    # Dummy data for update
    rewards = [0.1, 0.2, 0.05]
    # Create log_probs that require gradients
    log_probs = [
        torch.tensor(-0.5, device=device_fixture, requires_grad=True),
        torch.tensor(-0.2, device=device_fixture, requires_grad=True),
        torch.tensor(-0.8, device=device_fixture, requires_grad=True)
    ]
    
    policy.update_policy(rewards, log_probs)
    
    updated_param_value = param_to_check.clone().detach()
    
    # Assert that parameters have changed (optimizer step was called and grads were non-zero)
    assert not torch.equal(initial_param_value, updated_param_value), \
        "Model parameter did not change after policy update. Check learning rate, gradients, or optimizer."

    # Check if requires_grad is still True for model parameters after update (it should be)
    for param in model.parameters():
        assert param.requires_grad, "Model parameter lost requires_grad after update"

    # Further check: ensure log_probs are detached or don't have grad_fn if they were part of a larger graph not intended
    # In REINFORCE, log_probs are typically leaves in the loss computation for THIS update step.
    # The policy.update_policy method calculates loss and calls backward.
    # The original log_prob tensors passed in might still have requires_grad=True if they were created that way.
    # This is generally fine.
    assert log_probs[0].requires_grad, "Original log_prob tensor's requires_grad status changed unexpectedly."

    # Test with empty rewards/log_probs (should not error)
    try:
        policy.update_policy([], [])
        policy.update_policy([0.1], []) # Mismatched lengths, current impl might handle, but good to be aware
        policy.update_policy([], [torch.tensor(-0.1, requires_grad=True)])
    except Exception as e:
        pytest.fail(f"policy.update_policy failed with empty or mismatched inputs: {e}")
