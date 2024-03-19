import sys
import wandb
import tqdm
import pandas as pd
from accelerate import Accelerator
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, XLMRobertaModel, CLIPTextModel
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split

from MCLIP import MCLIP


class MCLIPDataset(Dataset):
    def __init__(self, data, tokenizer, max_length, clip_model, clip_tokenizer, device):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.clip_model = clip_model
        self.clip_tokenizer = clip_tokenizer
        self.device = device

    def __len__(self):
        return len(self.data)

    def clip_embedding(self, text):
        tokens = self.clip_tokenizer(text, return_tensors="pt", padding="max_length", truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.clip_model(**tokens)
        return outputs

    def __getitem__(self, idx):
        text = self.data.iloc[idx, 2]
        target = self.data.iloc[idx, 1]
        inputs = self.tokenizer(text, max_length=self.max_length, padding="max_length", truncation=True,
                                return_tensors="pt")
        return inputs, self.clip_embedding(target)


def train(model: MCLIP, train_loader: DataLoader, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler,
          device: torch.device, accelerator: Accelerator, use_wandb: bool) -> float:
    model.train()
    running_loss = 0.0
    for inputs, targets in tqdm.tqdm(train_loader, total=len(train_loader)):
        inputs = {k: v.squeeze(1).to(device) for k, v in inputs.items()}
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.mse_loss(outputs, targets)
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()
        if use_wandb:
            wandb.log({"loss": loss.item()})
    return running_loss / len(train_loader)


def main():
    batch_size = 8
    learning_rate = 1e-5
    n_epochs = 10

    use_wandb = False

    if use_wandb:
        wandb.init(project="mclip")
    accelerator = Accelerator()
    device = accelerator.device

    if use_wandb:
        wandb.config.update({
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "device": device.type,
            "n_epochs": n_epochs
        })

    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(accelerator.state)

    clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    clip_model.to(device)
    clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    xlmr_model = XLMRobertaModel.from_pretrained("xlm-roberta-base")
    xlmr_model.to(device)
    xlmr_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    model = MCLIP(clip_model, xlmr_model)

    model = model.to(device)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=1000)

    data = pd.read_csv("data.csv")
    train_data, test_data = train_test_split(data, test_size=0.2)
    train_dataset = MCLIPDataset(train_data, xlmr_tokenizer, 512, clip_model, clip_tokenizer, device)
    test_dataset = MCLIPDataset(test_data, xlmr_tokenizer, 512, clip_model, clip_tokenizer, device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    for epoch in range(n_epochs):
        train_loss = train(model, train_loader, optimizer, scheduler, device, accelerator, use_wandb)
        torch.save(model.state_dict(), f"models/mclip_epoch_{epoch + 1}.pt")
        print(f"Epoch {epoch + 1} Train Loss: {train_loss:.4f}")

    torch.save(model.state_dict(), "models/mclip_final.pt")

    model.eval()
    running_loss = 0.0
    if use_wandb:
        wandb.init(project="mclip", job_type="test")
    for inputs, targets in tqdm.tqdm(test_loader, total=len(test_loader)):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(inputs["input_ids"])
        loss = F.mse_loss(outputs, targets)
        running_loss += loss.item()
        if use_wandb:
            wandb.log({"test_loss": loss.item()})
    print(f"Test Loss: {running_loss / len(test_loader):.4f}")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
