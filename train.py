import time
import warnings
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.utils import Trainer, setup_dataloaders, tokenization
from src.lora_engine import replace_modules_with_lora


warnings.filterwarnings('ignore')

def get_device():
    if torch.backends.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def load_base_model(
        model_name: str,
        num_labels: int = 2,
        verbose: bool = True
    ) -> AutoModelForSequenceClassification:
    """Loads sequence classification model and prints number of trainable parameters to the console."""
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels)
    if verbose:
        base_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("\nTrainable Parameters:")
        print("Base trainable: ", base_trainable, "(100%)")
    return model


def train_model(
        model,
        train_loader: Dataloader,
        val_loader: Dataloader, 
        test_loader: Dataloader, 
        tokenizer: AutoTokenizer, 
        lr: float = 1e-5, 
        num_epochs: int = 5, 
        device: str = 'cpu'
    ) -> None:
    """Trains model with given parameters and times it."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(model=model,
                      train_loader=train_loader,
                      val_loader=val_loader,
                      test_loader=test_loader,
                      tokenizer=tokenizer,
                      optimizer=optimizer,
                      criterion=criterion,
                      device=device)

    start = time.time()
    trainer.train(num_epochs=num_epochs)
    end = time.time()
    elapsed = end - start
    print(f"Time elapsed {elapsed / 60:.2f} min")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Default arguments for model training")

    parser.add_argument('--task', type=str, required=True, default='fft', choices=['fft', 'lora', 'cls'], )
    parser.add_argument('--rank', type=int, default=8, help='Rank of the model')
    parser.add_argument('--alpha', type=int, default=16, help='Alpha parameter')
    parser.add_argument('--modules_to_replace', type=str, nargs='+', default=['q_lin', 'v_lin', 'lin1', 'lin2'],
                        help='List of modules to replace')
    parser.add_argument('--model_name', type=str, default='distilbert-base-uncased', help='Name of the model')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--device', type=str, default=get_device(),
                        help='Device to use for training (mps, cuda, or cpu)')

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    imdb_tokenized = tokenization()
    train_loader, val_loader, test_loader = setup_dataloaders(imdb_tokenized)
    model = load_base_model(args.model_name)
    base_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if args.task == 'cls': # classification head finetuning
        for param in model.parameters():
            param.requires_grad = False
        for param in model.pre_classifier.parameters():
            param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True
        cls_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Cls trainable: ", cls_trainable, "(", (cls_trainable / base_trainable * 100), "%)")
    if args.task == 'lora':
        model = replace_modules_with_lora(model, args.modules_to_replace, rank=args.rank, alpha=args.alpha)
        lora_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("LoRA trainable: ", lora_trainable, "(", (lora_trainable / base_trainable * 100), "%)")

    train_model(model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                tokenizer=tokenizer,
                lr=args.lr,
                num_epochs=args.num_epochs,
                device=args.device)
