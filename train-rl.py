import os
import sys
import torch
from typing import Tuple, Any, List 
import torch.optim as optim 
import torch.nn.functional as F 
from transformers import AutoTokenizer # Added for Hugging Face Tokenizer

# Requires: pip install transformers sentencepiece torch

# Add the current directory to sys.path to ensure model.py can be imported
sys.path.append(os.path.dirname(__file__))

from model import Transformer, ModelArgs

def load_pretrained_model(checkpoint_path: str, device: str, out_dir: str = "out") -> Transformer:
    """
    Loads a pretrained model from a checkpoint file.
    """
    if checkpoint_path is None:
        checkpoint_path = os.path.join(out_dir, "ckpt.pt")

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_args_dict = checkpoint["model_args"]
    # Ensure all keys in model_args_dict are valid arguments for ModelArgs
    # This is important if the checkpoint was saved with a different set of args
    # For now, assume compatibility or that ModelArgs handles extra/missing keys gracefully.
    gptconf = ModelArgs(**model_args_dict)
    model = Transformer(gptconf)
    state_dict = checkpoint["model"]

    unwanted_prefix = "_orig_mod."
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(unwanted_prefix):
            cleaned_state_dict[k[len(unwanted_prefix):]] = v
        else:
            cleaned_state_dict[k] = v
    state_dict = cleaned_state_dict

    model.load_state_dict(state_dict, strict=False) # Set strict=False to handle potential mismatches gracefully
    model.to(device)
    model.eval() 
    print("Pretrained model loaded successfully.")
    return model

class RLEnvironment:
    """
    Reinforcement Learning Environment for text generation.
    Uses a Hugging Face tokenizer.
    """
    def __init__(self, model_for_max_len: Transformer, tokenizer: Any, device: str, max_seq_len: int = 1024):
        # model_for_max_len is used to potentially derive max_seq_len, not for generation within env
        self.tokenizer = tokenizer
        self.device = device
        
        # Determine max_seq_len: Use tokenizer's model_max_length if available and reasonable,
        # otherwise use model's block_size, or fallback to provided max_seq_len.
        if hasattr(self.tokenizer, 'model_max_length') and self.tokenizer.model_max_length < 10000: # Check for a sensible value
            self.max_seq_len = self.tokenizer.model_max_length
        elif hasattr(model_for_max_len, 'params') and hasattr(model_for_max_len.params, 'block_size'):
            self.max_seq_len = model_for_max_len.params.block_size
        else:
            self.max_seq_len = max_seq_len
        
        self.current_state: torch.Tensor = None
        print(f"RLEnvironment initialized with tokenizer: {self.tokenizer.name_or_path}, max_seq_len: {self.max_seq_len}")
        if self.tokenizer.bos_token_id is None or self.tokenizer.eos_token_id is None:
             print("Warning: Tokenizer BOS or EOS token ID is not set. This might cause issues.")


    def reset(self, initial_prompt_text: str = "") -> torch.Tensor:
        if not initial_prompt_text:
            if self.tokenizer.bos_token_id is None:
                print("Error: Tokenizer does not have a BOS token defined, and prompt is empty.")
                # Fallback: use a common token like 0 or 1 if BOS is not defined, or raise error
                # For now, using 1, but this should be handled based on tokenizer specifics.
                self.current_state = torch.tensor([[1]], dtype=torch.long, device=self.device)
            else:
                self.current_state = torch.tensor([[self.tokenizer.bos_token_id]], dtype=torch.long, device=self.device)
        else:
            # Encode the prompt, ensuring add_special_tokens is handled correctly
            # Some tokenizers add BOS by default, others need add_special_tokens=True explicitly.
            # For Llama tokenizers, add_special_tokens=True usually adds BOS.
            encoded_prompt = self.tokenizer.encode(initial_prompt_text, return_tensors="pt", add_special_tokens=True)
            self.current_state = encoded_prompt.to(self.device)
            # If encoding results in an empty tensor (e.g. tokenizer quirk or very short/special prompt)
            if self.current_state.nelement() == 0 or self.current_state.size(1) == 0:
                print(f"Warning: Encoding prompt '{initial_prompt_text}' resulted in empty tensor. Using BOS.")
                if self.tokenizer.bos_token_id is not None:
                    self.current_state = torch.tensor([[self.tokenizer.bos_token_id]], dtype=torch.long, device=self.device)
                else: # Fallback if BOS is also None
                    self.current_state = torch.tensor([[1]], dtype=torch.long, device=self.device)


        if self.current_state.dim() == 1: # Ensure it's [batch_size, seq_len]
            self.current_state = self.current_state.unsqueeze(0)
        
        return self.current_state

    def calculate_reward(self, current_state: torch.Tensor, action: int) -> float:
        reward = 0.1 
        if self.tokenizer.eos_token_id is not None and action == self.tokenizer.eos_token_id:
            reward = 0.0 
        if current_state.size(1) >= 6: 
            if torch.equal(current_state[0, -3:], current_state[0, -6:-3]):
                reward -= 0.5 
        return reward

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool]:
        action_tensor = torch.tensor([[action]], dtype=torch.long, device=self.device)
        reward = self.calculate_reward(self.current_state, action)
        self.current_state = torch.cat((self.current_state, action_tensor), dim=1)
        
        done = False
        if self.tokenizer.eos_token_id is not None and action == self.tokenizer.eos_token_id:
            done = True
        if self.current_state.size(1) >= self.max_seq_len:
            done = True
            
        return self.current_state, reward, done

    def get_action_space_size(self) -> int:
        return self.tokenizer.vocab_size

class GRPOPolicy:
    """
    GRPO (Generative Reward Policy Optimization) Policy/Agent.
    """
    def __init__(self, model: Transformer, tokenizer: Any, device: str, learning_rate: float = 3e-4):
        self.model = model  
        self.device = device
        self.tokenizer = tokenizer # Store tokenizer for vocab_size access
        self.vocab_size = self.tokenizer.vocab_size # Use tokenizer's vocab_size
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        print(f"GRPOPolicy initialized with vocab_size: {self.vocab_size} (from tokenizer).")

    def select_action(self, state: torch.Tensor, temperature: float = 1.0, top_k: int = 50) -> Tuple[int, torch.Tensor]:
        logits, _ = self.model(state) 
        last_token_logits = logits[:, -1, :]  
        
        if temperature != 1.0:
            last_token_logits = last_token_logits / temperature
        
        if top_k is not None and top_k > 0:
            k = min(top_k, last_token_logits.size(-1))
            if k > 0 : 
                v, _ = torch.topk(last_token_logits, k) 
                last_token_logits[last_token_logits < v[:, [-1]]] = -float('Inf') 
            
        probs = F.softmax(last_token_logits, dim=-1) 
        
        if torch.isnan(probs).any() or torch.isinf(probs).any() or probs.sum().item() == 0:
            # print("Warning: Invalid probabilities in select_action. Falling back to random token from vocab.")
            action = torch.randint(0, self.vocab_size, (1,), device=self.device).item()
            log_prob = torch.log(torch.tensor(1.0 / self.vocab_size, device=self.device))
        else:
            action_tensor = torch.multinomial(probs, num_samples=1) 
            action = action_tensor.item() 
            log_prob = torch.log(probs.squeeze(0)[action]) if probs.size(0) == 1 else torch.log(probs[0, action])

        return action, log_prob

    def update_policy(self, rewards: List[float], log_probs: List[torch.Tensor], gamma: float = 0.99):
        if not rewards or not log_probs:
            return

        discounted_returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            discounted_returns.insert(0, R)
        
        discounted_returns = torch.tensor(discounted_returns, device=self.device)
        
        if len(discounted_returns) > 1:
            discounted_returns = (discounted_returns - discounted_returns.mean()) / (discounted_returns.std() + 1e-9)
        elif len(discounted_returns) == 1:
             discounted_returns = (discounted_returns - discounted_returns.mean()) / (torch.abs(discounted_returns.mean()) + 1e-9)


        policy_loss = []
        for log_prob, G_t in zip(log_probs, discounted_returns):
            if isinstance(log_prob, torch.Tensor): 
                policy_loss.append(-log_prob * G_t)
            else:
                continue 
        
        if not policy_loss:
            return

        self.optimizer.zero_grad()
        policy_loss_tensor = torch.stack(policy_loss).sum() 
        policy_loss_tensor.backward()
        self.optimizer.step()

def train_one_episode(policy: GRPOPolicy, env: RLEnvironment, initial_prompt: str, max_episode_steps: int = 50, temperature: float = 1.0, top_k: int = 50) -> Tuple[List[float], List[torch.Tensor]]:
    current_env_state = env.reset(initial_prompt_text=initial_prompt)
    collected_rewards = []
    collected_log_probs = []

    # Check if reset state is valid (not empty)
    if current_env_state is None or current_env_state.nelement() == 0:
        print(f"Error: Environment reset with prompt '{initial_prompt}' resulted in empty state. Skipping episode.")
        return collected_rewards, collected_log_probs

    for step_num in range(max_episode_steps):
        current_env_state = current_env_state.to(policy.device) 
        
        action, log_prob = policy.select_action(current_env_state, temperature=temperature, top_k=top_k)
        next_env_state, reward, done = env.step(action)
        
        collected_rewards.append(reward)
        collected_log_probs.append(log_prob)
        
        current_env_state = next_env_state
        if done:
            break
            
    return collected_rewards, collected_log_probs


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    out_dir = "out" 
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Attempting to load model from {ckpt_path}...")

    learning_rate = 1e-5 # Further reduced LR for HF tokenizer
    num_episodes = 20    
    max_steps_per_episode = 30 
    rl_temperature = 0.8 
    rl_top_k = 40        
    
    prompts = [
        "The problem with modern art is", 
        "Once upon a time, in a kingdom built on clouds", 
        "To truly understand recursion, one must first understand the nature of a mirror.", 
        "The key to effective machine learning is often found in",
        "A spaceship drifted silently through the void, its only passenger a cat named"
    ]

    try:
        model = load_pretrained_model(checkpoint_path=ckpt_path, device=device, out_dir=out_dir)
        print("Pretrained model loaded successfully.")

        try:
            tokenizer_name = "KoboldAI/llama2-tokenizer"
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token 
                print(f"Set tokenizer.pad_token_id to tokenizer.eos_token_id ({tokenizer.eos_token_id})")
            if tokenizer.bos_token_id is None: # Llama tokenizer might have bos_token but not bos_token_id set in some HF versions
                if hasattr(tokenizer, 'bos_token') and isinstance(tokenizer.bos_token, str):
                    tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.bos_token)
                print(f"Set tokenizer.bos_token_id to {tokenizer.bos_token_id}")

        except Exception as e:
            print(f"Error loading tokenizer {tokenizer_name}: {e}")
            print("Please ensure you have `transformers` and `sentencepiece` installed and internet connectivity.")
            print("Install with: pip install transformers sentencepiece")
            sys.exit(1)
        
        print(f"Hugging Face tokenizer '{tokenizer_name}' loaded. Vocab size: {tokenizer.vocab_size}")
        print(f"BOS ID: {tokenizer.bos_token_id}, EOS ID: {tokenizer.eos_token_id}, PAD ID: {tokenizer.pad_token_id}")


        if hasattr(model, 'params') and model.params.vocab_size != tokenizer.vocab_size:
            print(f"Warning: Model vocab size ({model.params.vocab_size}) and tokenizer vocab size ({tokenizer.vocab_size}) differ.")
            # This is a critical warning. For GRPO, we will use tokenizer.vocab_size for the policy.
            # The model's output layer should ideally match this. If not, it might lead to issues.
            # Consider re-initializing the model's output layer if this mismatch is problematic.
            # For now, GRPOPolicy will use tokenizer.vocab_size.
        
        # Use tokenizer.model_max_length if available, else model.params.block_size
        env_max_seq_len = tokenizer.model_max_length if hasattr(tokenizer, 'model_max_length') and tokenizer.model_max_length < 10000 else model.params.block_size
        env = RLEnvironment(model_for_max_len=model, tokenizer=tokenizer, device=device, max_seq_len=min(env_max_seq_len, 256))
        
        policy = GRPOPolicy(model=model, tokenizer=tokenizer, device=device, learning_rate=learning_rate)
        
        model.train() 
        print("Model set to train() mode for GRPO.")

        # Optional: Test generation after model load and with new tokenizer
        # print("\nTesting generation with loaded model and new tokenizer...")
        # test_prompt = "Hello, world! My name is"
        # test_start_tokens = tokenizer.encode(test_prompt, return_tensors="pt", add_special_tokens=True).to(device)
        # if test_start_tokens.size(1) == 0: 
        #    test_start_tokens = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long, device=device)
        # generated_output_test = model.generate(test_start_tokens, max_new_tokens=20)
        # generated_text_test = tokenizer.decode(generated_output_test[0], skip_special_tokens=True)
        # print(f"Test prompt: '{test_prompt}'")
        # print(f"Generated text: '{generated_text_test}'\n")


        print(f"\nStarting GRPO training for {num_episodes} episodes...")
        all_episode_rewards = []

        for episode_num in range(num_episodes):
            current_prompt = prompts[episode_num % len(prompts)]
            
            rewards, log_probs = train_one_episode(
                policy, 
                env, 
                initial_prompt=current_prompt,
                max_episode_steps=max_steps_per_episode,
                temperature=rl_temperature,
                top_k=rl_top_k
            )
            
            if not rewards or not log_probs:
                print(f"Episode {episode_num + 1}/{num_episodes}: No data collected. Prompt: '{current_prompt[:30]}...'. Skipping update.")
                continue

            policy.update_policy(rewards, log_probs) 
            
            total_reward_episode = sum(rewards)
            all_episode_rewards.append(total_reward_episode)
            print(f"Episode {episode_num + 1}/{num_episodes}: Total Reward: {total_reward_episode:.3f}, Steps: {len(rewards)}, Prompt: '{current_prompt[:30]}...'")
            
            if (episode_num + 1) % 10 == 0: 
                if all_episode_rewards:
                    avg_reward = sum(all_episode_rewards[-10:]) / len(all_episode_rewards[-10:])
                    print(f"Average reward for last 10 episodes: {avg_reward:.3f}")

        print("\nGRPO training complete.")

        fine_tuned_model_path = os.path.join(out_dir, "grpo_fine_tuned_ckpt_hf_tokenizer.pt")
        model_params_to_save = {}
        if hasattr(model, 'params') and model.params is not None:
             if isinstance(model.params, ModelArgs): 
                 model_params_to_save = vars(model.params)
             else: 
                 try: model_params_to_save = dict(model.params)
                 except: print("Warning: Could not serialize model.params to dict for saving.")

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': policy.optimizer.state_dict(),
            'model_args': model_params_to_save, 
            'grpo_config': { 
                'learning_rate': learning_rate, 'num_episodes': num_episodes,
                'max_steps_per_episode': max_steps_per_episode, 'rl_temperature': rl_temperature,
                'rl_top_k': rl_top_k, 'tokenizer_name': tokenizer.name_or_path
            }
        }, fine_tuned_model_path)
        print(f"Fine-tuned model and GRPO config saved to {fine_tuned_model_path}")

    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {ckpt_path}. Ensure 'out/ckpt.pt' exists for pretraining.")
    except AttributeError as e:
        print(f"AttributeError: {e}. This might be due to differences in model structure or ModelArgs.")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

    print("\ntrain-rl.py execution finished.")
