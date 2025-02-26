import os
import os.path as op
from datasets import load_dataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch
from tqdm import tqdm
import evaluate


class IMDBDataset(Dataset):
    def __init__(self, dataset_dict, partition_key="train"):
        self.partition = dataset_dict[partition_key]

    def __getitem__(self, index):
        return self.partition[index]

    def __len__(self):
        return self.partition.num_rows


def tokenization():
    imdb_dataset = load_dataset(
        "csv",
        data_files={
            "train": op.join("data", "train.csv"),
            "validation": op.join("data", "val.csv"),
            "test": op.join("data", "test.csv"),
        },
    )
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize_text(batch):
        return tokenizer(batch["text"], truncation=True, padding=True)

    imdb_tokenized = imdb_dataset.map(tokenize_text, batched=True, batch_size=None)
    imdb_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    return imdb_tokenized


def setup_dataloaders(imdb_tokenized):
    train_dataset = IMDBDataset(imdb_tokenized, partition_key="train")
    val_dataset = IMDBDataset(imdb_tokenized, partition_key="validation")
    test_dataset = IMDBDataset(imdb_tokenized, partition_key="test")

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=12,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=12,
        num_workers=4
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=12,
        num_workers=4
    )
    return train_loader, val_loader, test_loader


class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, tokenizer, optimizer, criterion,
                 device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        print("Using device:", self.device)

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        with tqdm(total=len(self.train_loader), desc="Training") as progress_bar:
            for batch in self.train_loader:
                inputs = {
                    "input_ids": batch["input_ids"].to(self.device),
                    "attention_mask": batch["attention_mask"].to(self.device),
                    "labels": batch["label"].to(self.device)
                }
                self.optimizer.zero_grad()
                outputs = self.model(**inputs)
                loss = self.criterion(outputs.logits, inputs["labels"])
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

                progress_bar.update(1)
                progress_bar.set_postfix({'loss': total_loss / (progress_bar.n + 1e-5)})

        return total_loss / len(self.train_loader)

    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for batch in data_loader:
                inputs = {
                    "input_ids": batch["input_ids"].to(self.device),
                    "attention_mask": batch["attention_mask"].to(self.device),
                    "labels": batch["label"].to(self.device)
                }
                outputs = self.model(**inputs)
                loss = self.criterion(outputs.logits, inputs["labels"])
                total_loss += loss.item()
                predictions = torch.argmax(outputs.logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(inputs["labels"].cpu().numpy())

        return total_loss / len(data_loader), all_predictions, all_labels

    def train(self, num_epochs):
        best_val_loss = float('inf')
        accuracy_metric = evaluate.load("accuracy")
        f1_metric = evaluate.load("f1")

        for epoch in range(num_epochs):
            print(f"Starting Epoch {epoch + 1}/{num_epochs}")
            train_loss = self.train_epoch()
            val_loss, val_preds, val_labels = self.evaluate(self.val_loader)
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Calculate accuracy
            accuracy_metric.add_batch(predictions=val_preds, references=val_labels)
            accuracy = accuracy_metric.compute()["accuracy"]
            print(f"Validation Accuracy: {accuracy:.4f}")

            # Calculate F1 score
            f1_metric.add_batch(predictions=val_preds, references=val_labels)
            f1_score = f1_metric.compute(average="weighted")["f1"]
            print(f"Validation F1 Score: {f1_score:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), "best_model.pth")

        test_loss, test_preds, test_labels = self.evaluate(self.test_loader)
        print(f"Test Loss: {test_loss:.4f}")

        # Calculate test accuracy
        accuracy_metric.add_batch(predictions=test_preds, references=test_labels)
        test_accuracy = accuracy_metric.compute()["accuracy"]
        print(f"Test Accuracy: {test_accuracy:.4f}")

        # Calculate test F1 score
        f1_metric.add_batch(predictions=test_preds, references=test_labels)
        test_f1_score = f1_metric.compute(average="weighted")["f1"]
        print(f"Test F1 Score: {test_f1_score:.4f}")
