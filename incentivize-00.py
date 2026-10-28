import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def supervised_training(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad(set_to_none=True)

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )

        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    batches = 0

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

        total_loss += outputs.loss.item()
        batches += 1

    return total_loss / max(batches, 1)


def collect_feedback():
    prompt = input("Prompt: ")
    response = input("Response: ")
    rating = float(input("Rating (0-1): "))

    if not 0.0 <= rating <= 1.0:
        raise ValueError("Rating must be between 0 and 1")

    return {
        "prompt": prompt,
        "response": response,
        "rating": rating,
    }


def reward_model_loss(reward_model, chosen, rejected):
    chosen_reward = reward_model(**chosen).logits.squeeze(-1)
    rejected_reward = reward_model(**rejected).logits.squeeze(-1)

    return -F.logsigmoid(
        chosen_reward - rejected_reward
    ).mean()


def ppo_clipped_objective(
    new_log_probs,
    old_log_probs,
    advantages,
    clip_range=0.2,
):
    ratio = torch.exp(new_log_probs - old_log_probs)

    unclipped = ratio * advantages
    clipped = torch.clamp(
        ratio, 1.0 - clip_range, 1.0 + clip_range
    ) * advantages

    return -torch.minimum(unclipped, clipped).mean()


def train():
    model_name = "distilgpt2"
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)

    training_texts = [
        "Example training text.",
        "Another example training text.",
    ]

    dataset = TextDataset(training_texts, tokenizer)
    dataloader = DataLoader(
        dataset, batch_size=2, shuffle=True
    )

    optimizer = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(3):
        train_loss = supervised_training(
            model, dataloader, optimizer, device
        )

        validation_loss = evaluate(
            model, dataloader, device
        )

        print({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
        })

    model.save_pretrained("./trained_model")
    tokenizer.save_pretrained("./trained_model")


if __name__ == "__main__":
    train()
