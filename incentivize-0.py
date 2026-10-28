import os
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType


MODEL_NAME = "distilgpt2"
OUTPUT_DIR = "./model_output"
CHECKPOINT_DIR = "./checkpoints"
MAX_LENGTH = 512
SEED = 42

set_seed(SEED)
random.seed(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def load_data():
    dataset = load_dataset(
        "json",
        data_files={
            "train": "data/train.jsonl",
            "validation": "data/validation.jsonl",
            "test": "data/test.jsonl",
            "preferences": "data/preferences.jsonl",
        },
    )

    required = {
        "train": {"text"},
        "validation": {"text"},
        "test": {"text"},
        "preferences": {"prompt", "chosen", "rejected"},
    }

    for split, columns in required.items():
        if not columns.issubset(dataset[split].column_names):
            raise ValueError(f"Invalid columns in {split}")

        if len(dataset[split]) == 0:
            raise ValueError(f"Empty dataset split: {split}")

    return dataset


def prepare_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.config.pad_token_id = tokenizer.pad_token_id

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["c_attn"],
        bias="none",
    )

    model = get_peft_model(model, lora_config)

    return model, tokenizer


def tokenize_dataset(dataset, tokenizer):
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    tokenized = {}

    for split in ("train", "validation", "test"):
        tokenized[split] = dataset[split].map(
            tokenize,
            batched=True,
            remove_columns=dataset[split].column_names,
        )

    return tokenized


def run_supervised_training(model, tokenizer, tokenized):
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=2e-4,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=25,
        report_to="none",
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator,
    )

    trainer.train(resume_from_checkpoint=True if
                  any(os.path.isdir(os.path.join(CHECKPOINT_DIR, d))
                      for d in os.listdir(CHECKPOINT_DIR))
                  else None)

    validation_metrics = trainer.evaluate(
        eval_dataset=tokenized["validation"]
    )

    test_metrics = trainer.evaluate(
        eval_dataset=tokenized["test"],
        metric_key_prefix="test",
    )

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    return trainer, validation_metrics, test_metrics


class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.encoder = base_model
        self.score = nn.Linear(
            base_model.config.hidden_size, 1
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        hidden = outputs.last_hidden_state
        positions = attention_mask.sum(dim=1) - 1
        batch = torch.arange(hidden.size(0), device=hidden.device)

        last_hidden = hidden[batch, positions]
        return self.score(last_hidden).squeeze(-1)


def preference_loss(reward_model, batch):
    chosen_reward = reward_model(
        batch["chosen_input_ids"],
        batch["chosen_attention_mask"],
    )

    rejected_reward = reward_model(
        batch["rejected_input_ids"],
        batch["rejected_attention_mask"],
    )

    return -F.logsigmoid(
        chosen_reward - rejected_reward
    ).mean()


def train_reward_model(
    reward_model,
    preference_loader,
    optimizer,
    epochs=1,
):
    reward_model.train()

    for _ in range(epochs):
        for batch in preference_loader:
            batch = {
                key: value.to(DEVICE)
                for key, value in batch.items()
            }

            optimizer.zero_grad(set_to_none=True)
            loss = preference_loss(reward_model, batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                reward_model.parameters(), 1.0
            )

            optimizer.step()


def evaluate_reward_model(reward_model, preference_loader):
    reward_model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in preference_loader:
            batch = {
                key: value.to(DEVICE)
                for key, value in batch.items()
            }

            chosen = reward_model(
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"],
            )

            rejected = reward_model(
                batch["rejected_input_ids"],
                batch["rejected_attention_mask"],
            )

            correct += (chosen > rejected).sum().item()
            total += chosen.numel()

    return correct / max(total, 1)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    dataset = load_data()
    model, tokenizer = prepare_model()

    model.to(DEVICE)

    tokenized = tokenize_dataset(dataset, tokenizer)

    trainer, validation_metrics, test_metrics = (
        run_supervised_training(model, tokenizer, tokenized)
    )

    print(json.dumps({
        "validation": validation_metrics,
        "test": test_metrics,
    }, indent=2))

    preference_records = dataset["preferences"]

    if len(preference_records) < 2:
        raise ValueError("Insufficient preference examples")

    model.save_pretrained(
        os.path.join(OUTPUT_DIR, "sft_adapter")
    )

    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()
