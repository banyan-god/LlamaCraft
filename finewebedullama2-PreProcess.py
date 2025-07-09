from datasets import load_dataset
import transformers


class Task:
    """Tokenise the FineWeb-Edu dataset once and save it to disk.

    The resulting Arrow/Parquet files contain *full* documents (preceded by a
    single BOS token) without any length truncation.  At training time those
    documents are concatenated and cut into context-length blocks by
    ``finewebedullama2.py`` according to the sequence length requested in
    ``train.py``.
    """

    def __init__(self, batch_size: int):
        self.batch_size = batch_size  # currently unused but kept for API symmetry

        self.tokenizer = transformers.AutoTokenizer.from_pretrained("KoboldAI/llama2-tokenizer")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.initialize()

    # ---------------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------------
    def initialize(self):
        dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=False,
            num_proc=24,
        )

        train_dataset, val_dataset = dataset.train_test_split(test_size=0.1).values()

        columns_to_remove = [
            "id",
            "url",
            "text",
            "dump",
            "file_path",
            "language",
            "language_score",
            "token_count",
            "score",
            "int_score",
        ]

        print("Tokenising train split …")
        tokenized_train = train_dataset.map(
            self.tokenize_function,
            batched=True,
            num_proc=24,
            remove_columns=columns_to_remove,
        )
        tokenized_train.save_to_disk("data/tokenized_datasets")

        print("Tokenising validation split …")
        tokenized_val = val_dataset.map(
            self.tokenize_function,
            batched=True,
            num_proc=24,
            remove_columns=columns_to_remove,
        )
        tokenized_val.save_to_disk("data/tokenized_val_datasets")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def tokenize_function(self, examples):
        # Tokenise *without* truncation so that sequence length can be decided
        # later at training time.
        tokenized_output = self.tokenizer(
            examples["text"],
            truncation=False,
            add_special_tokens=False,
        )

        # Prepend BOS token to every document.
        input_ids = [
            [self.tokenizer.bos_token_id] + ids for ids in tokenized_output["input_ids"]
        ]

        return {"input_ids": input_ids}


if __name__ == "__main__":
    # Simple CLI entry-point.
    Task(batch_size=4)
