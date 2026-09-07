
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import BertTokenizer, BertModel


bert = BertModel.from_pretrained("bert-base-uncased")
outputs = bert(**tokens)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # bert transformer
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        # max pool
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits