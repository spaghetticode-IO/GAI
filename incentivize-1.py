
import os
import inspect
import random
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    DataCollator,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, TaskType, get_peft_model


MODEL_NAME = "distilgpt2"
DATA_DIR = "./data"
OUTPUT_DIR = "./output"
CHECKPOINT_DIR = "./checkpoints"
MAX_LENGTH = 512
SEED = 42
SFT_EPOCHS = 3
REWARD_EPOCHS = 3
BATCH_SIZE = 2
LEARNING_RATE = 2e-4
REWARD_LEARNING_RATE = 1e-5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

set_seed(SEED)
random.seed(SEED)


# --------------------------------------------------
# 1. Data validation
# --------------------------------------------------

def load_and_validate_data():
    data = load_dataset(
        "json",
        data_files={
            "train": f"{DATA_DIR}/train.jsonl",
            "validation": f"{DATA_DIR}/validation.jsonl",
            "test": f"{DATA_DIR}/test.jsonl",
            "preferences": f"{DATA_DIR}/preferences.jsonl",
        },
    )

    for split in ("train", "validation", "test"):
        if "text" not in data[split].column_names:
            raise ValueError(f"{split}: missing text column")

        data[split] = data[split].filter(
            lambda row: isinstance(row["text"], str)
            and bool(row["text"].strip())
        )

        if len(data[split]) == 0:
            raise ValueError(f"{split}: no valid text records")

    required = {"prompt", "chosen", "rejected"}

    if not required.issubset(data["preferences"].column_names):
        raise ValueError("Invalid preference columns")

    data["preferences"] = data["preferences"].filter(
        lambda row: all(
            isinstance(row[key], str) and bool(row[key].strip())
            for key in ("prompt", "chosen", "rejected")
        )
        and row["chosen"].strip() != row["rejected"].strip()
    )

    if len(data["preferences"]) < 2:
        raise ValueError(
            "At least two valid preference pairs are required"
        )

    split_data = data["preferences"].train_test_split(
        test_size=max(1, int(len(data["preferences"]) * 0.1)),
        seed=SEED,
    )

    data["preference_train"] = split_data["train"]
    data["preference_validation"] = split_data["test"]

    return data


# --------------------------------------------------
# 2. Dynamic padding for SFT
# --------------------------------------------------

class CausalLMCollator(DataCollator):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels

        return batch


def tokenize_sft(data, tokenizer):
    def encode(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            add_special_tokens=True,
        )

    result = {}

    for split in ("train", "validation", "test"):
        result[split] = data[split].map(
            encode,
            batched=True,
            remove_columns=data[split].column_names,
        )

    return result


# --------------------------------------------------
# 3. SFT model and training
# --------------------------------------------------

def create_sft_model():
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["c_attn"],
        bias="none",
    )

    model = get_peft_model(model, config)
    return model


def train_sft(tokenizer, tokenized):
    model = create_sft_model()

    args_kwargs = dict(
        output_dir=CHECKPOINT_DIR,
        num_train_epochs=SFT_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=25,
        report_to="none",
        seed=SEED,
    )

    signature = inspect.signature(TrainingArguments.__init__).parameters

    if "eval_strategy" in signature:
        args_kwargs["eval_strategy"] = "epoch"
    else:
        args_kwargs["evaluation_strategy"] = "epoch"

    args = TrainingArguments(**args_kwargs)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=CausalLMCollator(tokenizer),
    )

    last_checkpoint = get_last_checkpoint(CHECKPOINT_DIR)

    trainer.train(resume_from_checkpoint=last_checkpoint)

    # Evaluate on test only after model selection is complete.
    test_metrics = trainer.evaluate(
        eval_dataset=tokenized["test"],
        metric_key_prefix="test",
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.save_model(os.path.join(OUTPUT_DIR, "sft"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "sft"))

    return test_metrics


# --------------------------------------------------
# 4. Preference dataset and dynamic padding
# --------------------------------------------------

class PreferenceDataset(Dataset):
    def __init__(self, records, tokenizer):
        self.examples = []

        for row in records:
            prompt = row["prompt"].strip()

            chosen = tokenizer(
                prompt + "\n" + row["chosen"].strip(),
                add_special_tokens=True,
                truncation=True,
                max_length=MAX_LENGTH,
            )["input_ids"]

            rejected = tokenizer(
                prompt + "\n" + row["rejected"].strip(),
                add_special_tokens=True,
                truncation=True,
                max_length=MAX_LENGTH,
            )["input_ids"]

            if chosen and rejected:
                self.examples.append({
                    "chosen": chosen,
                    "rejected": rejected,
                })

        if not self.examples:
            raise ValueError("No valid tokenized preferences")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class PreferenceCollator:
    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id

    def pad(self, sequences):
        max_len = max(len(seq) for seq in sequences)
        ids = torch.full(
            (len(sequences), max_len),
            self.pad_id,
            dtype=torch.long,
        )
        mask = torch.zeros_like(ids)

        for i, seq in enumerate(sequences):
            length = len(seq)
            ids[i, :length] = torch.tensor(seq)
            mask[i, :length] = 1

        return ids, mask

    def __call__(self, examples):
        chosen_ids, chosen_mask = self.pad(
            [x["chosen"] for x in examples]
        )
        rejected_ids, rejected_mask = self.pad(
            [x["rejected"] for x in examples]
        )

        return {
            "chosen_input_ids": chosen_ids,
            "chosen_attention_mask": chosen_mask,
            "rejected_input_ids": rejected_ids,
            "rejected_attention_mask": rejected_mask,
        }


# --------------------------------------------------
# 5. Reward model
# --------------------------------------------------

class RewardModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(model_name)

        hidden_size = self.backbone.config.hidden_size

        self.score = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        hidden = outputs.last_hidden_state

        # Select the final non-padding token.
        positions = torch.arange(
            attention_mask.shape[1],
            device=attention_mask.device,
        ).unsqueeze(0).expand_as(attention_mask)

        positions = positions.masked_fill(
            attention_mask == 0, -1
        )

        last_positions = positions.max(dim=1).values

        if (last_positions < 0).any():
            raise ValueError("Empty token sequence")

        batch_indices = torch.arange(
            hidden.shape[0], device=hidden.device
        )

        last_hidden = hidden[batch_indices, last_positions]

        return self.score(last_hidden).squeeze(-1)


def preference_loss(model, batch):
    chosen = model(
        batch["chosen_input_ids"],
        batch["chosen_attention_mask"],
    )

    rejected = model(
        batch["rejected_input_ids"],
        batch["rejected_attention_mask"],
    )

    return -F.logsigmoid(chosen - rejected).mean()


# --------------------------------------------------
# 6. Reward training, evaluation and checkpoints
# --------------------------------------------------

def move_batch(batch):
    return {
        key: value.to(DEVICE)
        for key, value in batch.items()
    }


@torch.no_grad()
def evaluate_reward(model, loader):
    model.eval()

    correct = 0
    total = 0
    total_loss = 0.0

    for batch in loader:
        batch = move_batch(batch)

        chosen = model(
            batch["chosen_input_ids"],
            batch["chosen_attention_mask"],
        )

        rejected = model(
            batch["rejected_input_ids"],
            batch["rejected_attention_mask"],
        )

        total_loss += -F.logsigmoid(
            chosen - rejected
        ).mean().item()

        correct += (chosen > rejected).sum().item()
        total += chosen.numel()

    return {
        "loss": total_loss / max(len(loader), 1),
        "accuracy": correct / max(total, 1),
    }


def train_reward_model(train_loader, validation_loader):
    model = RewardModel(MODEL_NAME).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=REWARD_LEARNING_RATE,
        weight_decay=0.01,
    )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    latest_path = os.path.join(
        CHECKPOINT_DIR, "reward_latest.pt"
    )
    best_path = os.path.join(
        OUTPUT_DIR, "reward_best.pt"
    )

    start_epoch = 0
    best_accuracy = -1.0

    if os.path.isfile(latest_path):
        state = torch.load(
            latest_path,
            map_location=DEVICE,
            weights_only=False,
        )

        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"] + 1
        best_accuracy = state["best_accuracy"]

    for epoch in range(start_epoch, REWARD_EPOCHS):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            batch = move_batch(batch)

            optimizer.zero_grad(set_to_none=True)

            loss = preference_loss(model, batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )

            optimizer.step()
            running_loss += loss.item()

        metrics = evaluate_reward(model, validation_loader)

        print({
            "reward_epoch": epoch + 1,
            "train_loss": running_loss / max(len(train_loader), 1),
            "validation_loss": metrics["loss"],
            "validation_accuracy": metrics["accuracy"],
        })

        # Save full training state for resuming.
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_accuracy": max(
                best_accuracy, metrics["accuracy"]
            ),
        }, latest_path)

        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]

            torch.save({
                "model": model.state_dict(),
                "validation_metrics": metrics,
                "model_name": MODEL_NAME,
            }, best_path)

    return best_path


# --------------------------------------------------
# 7. Main pipeline
# --------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    data = load_and_validate_data()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    tokenized = tokenize_sft(data, tokenizer)

    sft_test_metrics = train_sft(tokenizer, tokenized)

    print("SFT test metrics:", sft_test_metrics)

    preference_train = PreferenceDataset(
        data["preference_train"], tokenizer
    )
    preference_validation = PreferenceDataset(
        data["preference_validation"], tokenizer
    )

    collator = PreferenceCollator(tokenizer)

    train_loader = DataLoader(
        preference_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
    )

    validation_loader = DataLoader(
        preference_validation,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
    )

    best_reward_path = train_reward_model(
        train_loader,
        validation_loader,
    )

    print("Best reward model:", best_reward_path)


if __name__ == "__main__":
    main()
