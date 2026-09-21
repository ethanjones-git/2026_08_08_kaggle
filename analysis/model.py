
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import BertTokenizer, BertModel

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.bert = BertModel.from_pretrained("bert-base-uncased")

        self.classifier = nn.Linear(
            self.bert.config.hidden_size,
            3
        )

    def forward(self, x):

        # Transformer
        x = self.bert(**x)

        # Extract token embeddings
        x = x.last_hidden_state

        # Average across tokens
        x = x.mean(dim=1)

        # Classification
        logits = self.classifier(x)

        return logits