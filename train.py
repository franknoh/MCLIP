import sys
import wandb
import tqdm
import pandas as pd
from accelerate import Accelerator
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import XLMRobertaTokenizer, XLMRobertaModel
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split

from MCLIP import MCLIP


class MCLIPDataset(Dataset):
    def __init__(self, data, tokenizer, max_length, clip_model, clip_tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.clip_model = clip_model
        self.clip_tokenizer = clip_tokenizer

    def __len__(self):
        return len(self.data)

    def clip_embedding(self, text):
        tokens = self.clip_tokenizer(text, return_tensors="pt", padding="max_length", truncation=True)
        with torch.no_grad():
            outputs = self.clip_model.get_text_features(**tokens)
        return outputs

    def __getitem__(self, idx):
        text = self.data.iloc[idx, 1]
        inputs = self.tokenizer(text, max_length=self.max_length, padding="max_length", truncation=True,
                                return_tensors="pt")
        return inputs, self.clip_embedding(text)


def train(model: MCLIP, train_loader: DataLoader, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler, device: torch.device):
    model.train()
    running_loss = 0.0
    for inputs, targets in tqdm.tqdm(train_loader, total=len(train_loader)):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        optimizer.zero_grad()
        outputs = model.forward_before_clip(inputs["input_ids"])
        loss = F.mse_loss(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()
        wandb.log({"loss": loss.item()})
    return running_loss / len(train_loader)


def main():
    wandb.init(project="mclip")
    accelerator = Accelerator()
    device = accelerator.device

    wandb.config.update({"batch_size": 8, "learning_rate": 1e-5})

    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Accelerate: {accelerator.state}")

    clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16")
    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
    xlmr_model = XLMRobertaModel.from_pretrained("xlm-roberta-base")
    xlmr_tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
    model = MCLIP(clip_model, xlmr_model)

    model = model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=1e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=1000)

    data = pd.read_csv("data.csv")
    train_data, test_data = train_test_split(data, test_size=0.2)
    train_dataset = MCLIPDataset(train_data, xlmr_tokenizer, 77, clip_model, clip_tokenizer)
    test_dataset = MCLIPDataset(test_data, xlmr_tokenizer, 77, clip_model, clip_tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    for epoch in range(10):
        train_loss = train(model, train_loader, optimizer, scheduler, device)
        print(f"Epoch {epoch + 1} Train Loss: {train_loss:.4f}")

    model.eval()
    running_loss = 0.0
    for inputs, targets in tqdm.tqdm(test_loader, total=len(test_loader)):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model.forward_before_clip(inputs["input_ids"])
        loss = F.mse_loss(outputs, targets)
        running_loss += loss.item()
        wandb.log({"test_loss": loss.item()})
    print(f"Test Loss: {running_loss / len(test_loader):.4f}")
    wandb.finish()


if __name__ == "__main__":
    main()
