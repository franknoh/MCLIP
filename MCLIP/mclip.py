import torch
from torch import nn
from torch.nn.modules.module import T
from transformers import XLMRobertaModel, CLIPTextModel
from .clip import CLIPLayer


class MCLIP(nn.Module):
    def __init__(self, clip_model: CLIPTextModel, xlmr_model: XLMRobertaModel, n_layers: int = 6):
        super().__init__()
        self.xlmr_model = xlmr_model
        self.embedding = nn.Linear(512, 77)
        self.layers = nn.ModuleList([
            CLIPLayer(12, 768) for i in range(n_layers)
        ])
        self.clip_layers = clip_model.text_model.encoder.layers
        self.clip_layers = self.clip_layers[:len(self.clip_layers) - n_layers]
        self.layernorm = clip_model.text_model.final_layer_norm

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)
        state = self.xlmr_model(tokens).last_hidden_state

        state = self.embedding(state)
        for layer in self.layers:
            state = layer(state)

        for layer in self.clip_layers:
            state = layer(state)
        output = self.layernorm(state)
        return output

    def train(self: T, mode: bool = True) -> T:
        self.xlmr_model.train(False)
        self.embedding.train(mode)
        self.layers.train(mode)
        self.clip_layers.train(False)
        self.layernorm.train(False)
        return self

    def eval(self: T) -> T:
        self.xlmr_model.eval()
        self.embedding.eval()
        self.layers.eval()
        self.clip_layers.eval()
        self.layernorm.eval()
        return self
