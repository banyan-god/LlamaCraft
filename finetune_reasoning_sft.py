# Requires: pip install torch datasets transformers trl accelerate bitsandbytes
import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

def run_sft():
    """
    Runs the Supervised Fine-Tuning (SFT) process.
    """
    # --- 1. Configuration Variables ---
    model_name = "sabareesh88/fw14k"
    dataset_name = "openai/gsm8k"
    dataset_config = "main"  # "main" or "socratic" for gsm8k
    new_model_name = "fw14k-gsm8k-sft"
    output_dir = f"./{new_model_name}"

    print("--- Configuration ---")
    print(f"Base model: {model_name}")
    print(f"Dataset: {dataset_name} ({dataset_config})")
    print(f"Output directory for fine-tuned model: {output_dir}")
    print("---------------------\n")

    # --- 2. Load Model and Tokenizer ---
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Tokenizer loaded.")

    print(f"Loading model {model_name}...")
    # To potentially save memory if running on limited resources, one might add:
    # model_kwargs = {"torch_dtype": torch.bfloat16} # if supported by GPU and model
    # model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    print("Model loaded.")

    # Ensure tokenizer has a pad token
    if tokenizer.pad_token is None:
        print("Tokenizer does not have a pad_token. Setting pad_token = eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        # Important: The model's config might also need to be updated if it relies on pad_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
    
    print(f"Tokenizer pad_token: {tokenizer.pad_token}, pad_token_id: {tokenizer.pad_token_id}")
    print(f"Tokenizer eos_token: {tokenizer.eos_token}, eos_token_id: {tokenizer.eos_token_id}")
    print("---------------------\n")

    # --- 3. Load and Preprocess Dataset ---
    print(f"Loading dataset {dataset_name} (config: {dataset_config})...")
    # Using a subset for faster example run: dataset = load_dataset(dataset_name, name=dataset_config, split="train[:1%]")
    dataset = load_dataset(dataset_name, name=dataset_config, split="train") 
    print(f"Dataset loaded. Number of examples: {len(dataset)}")

    def format_example(example):
        """
        Formats a single example from the GSM8K dataset for SFTTrainer.
        The SFTTrainer expects a column named "text" (or specified by dataset_text_field).
        """
        return {"text": "Question: " + example["question"] + "\nAnswer: " + example["answer"]}

    print("Formatting dataset...")
    # Using num_proc for potentially faster mapping, adjust based on your CPU cores
    # For very large datasets, consider dataset.set_transform for on-the-fly formatting.
    formatted_dataset = dataset.map(format_example, num_proc=os.cpu_count() // 2 if os.cpu_count() > 1 else 1)
    print("Dataset formatted.")
    print(f"First formatted example:\n{formatted_dataset[0]['text']}")
    print("---------------------\n")

    # --- 4. Training Arguments ---
    print("Defining TrainingArguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,  # Start with 1 for a quick test, increase for better results
        per_device_train_batch_size=2,  # Reduced from 4 to save memory, adjust based on GPU
        gradient_accumulation_steps=4,  # Increased from 2, effective batch size = 2*4=8
        learning_rate=2e-5,
        logging_steps=10,
        save_steps=50,  # Save a checkpoint every 50 steps
        fp16=torch.cuda.is_available(),  # Use mixed precision if CUDA is available
        # bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(), # Alternative precision
        report_to="none",  # Set to "wandb" or "tensorboard" if you want to log metrics
        # Further options to consider for memory saving:
        # gradient_checkpointing=True, # Can save memory but slows down training
        # optim="adamw_torch_fused", # If using PyTorch 2.0+ and CUDA
    )
    print("TrainingArguments defined.")
    print(f"  Output directory: {training_args.output_dir}")
    print(f"  Num train epochs: {training_args.num_train_epochs}")
    print(f"  Batch size per device: {training_args.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  FP16 enabled: {training_args.fp16}")
    print("---------------------\n")

    # --- 5. Initialize SFTTrainer ---
    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=formatted_dataset,
        dataset_text_field="text",  # Must match the column name from format_example
        max_seq_length=1024,  # Adjust based on model's capabilities and dataset analysis
        args=training_args,
        # packing=True, # Optional: packs multiple short examples into one sequence for efficiency
    )
    print("SFTTrainer initialized.")
    print("---------------------\n")

    # --- 6. Train the Model ---
    print("Starting model training...")
    try:
        train_result = trainer.train()
        print("Training completed.")
        # You can access training metrics via train_result if needed
        # print(f"Training metrics: {train_result.metrics}")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        print("Ensure you have enough GPU memory. Try reducing batch size or sequence length if it's an OOM error.")
        # Consider adding more specific error handling or cleanup if needed
        raise # Re-raise the exception to stop the script if training fails

    print("---------------------\n")

    # --- 7. Save the Model and Tokenizer ---
    print(f"Saving fine-tuned model and tokenizer to {output_dir}...")
    try:
        trainer.save_model(output_dir)  # Saves both model and tokenizer
        # If you want to save tokenizer explicitly (though SFTTrainer.save_model should handle it)
        # tokenizer.save_pretrained(output_dir)
        print(f"SFT fine-tuned model and tokenizer saved successfully to {output_dir}")
    except Exception as e:
        print(f"An error occurred during model saving: {e}")
        raise
    
    print("---------------------\n")
    print("SFT script finished.")

if __name__ == '__main__':
    print("Executing SFT script: finetune_reasoning_sft.py")
    run_sft()
    print("Script execution complete.")
