import torch
from torch import nn, Tensor
from typing import List
from transformers import AutoTokenizer, T5EncoderModel
from einops import rearrange

class TextEmbedder(nn.Module):
    def __init__(self, model: str = "t5-base", max_length: int = 64, enable_grad = False):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model, model_max_length = max_length)
        self.transformer = T5EncoderModel.from_pretrained(model, max_length=max_length)

        if not enable_grad:
            self.transformer.requires_grad_(False)

        self.max_length = max_length
        self.enable_grad = enable_grad

    def forward(self, texts: List[str]) -> Tensor:

        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        device = next(self.transformer.parameters()).device
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device).to(torch.bool)

        if not self.enable_grad:
            self.transformer.eval()

        embedding = self.transformer(
            input_ids=input_ids, attention_mask=attention_mask
        )["last_hidden_state"]

        return embedding, attention_mask