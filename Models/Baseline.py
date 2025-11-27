import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch import nn
import torch

conv_num_features = 128 

class ConvBackbone(nn.Module):
    """Arquitectura Conv-4 adaptada para Pokémon (Input 84x84)"""
    def __init__(self, in_channels=3, num_features=conv_num_features):
        super().__init__()

        # Input: (B, 3, 84, 84)
        self.conv1 = nn.Conv2d(in_channels, num_features, 3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=num_features//4, num_channels=num_features)
        
        self.conv2 = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=num_features//4, num_channels=num_features)

        self.conv3 = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.norm3 = nn.GroupNorm(num_groups=num_features//4, num_channels=num_features)
        
        self.conv4 = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.norm4 = nn.GroupNorm(num_groups=num_features//4, num_channels=num_features)

        self.maxpool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        
        # Detectar GPU automáticamente
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        # 84 -> 42
        x = self.maxpool(self.relu(self.norm1(self.conv1(x))))
        # 42 -> 21
        x = self.maxpool(self.relu(self.norm2(self.conv2(x))))
        # 21 -> 10
        x = self.maxpool(self.relu(self.norm3(self.conv3(x))))
        # 10 -> 5
        x = self.maxpool(self.norm4(self.conv4(x)))
        
        # Output final: 5x5x128. Aplanamos.
        x = x.view(x.size(0), -1) 
        return x

class ClassifierHead(nn.Module):
    def __init__(self, num_classes, input_size=conv_num_features*5*5, dropout=True):
        # NOTA: input_size es 128 * 5 * 5 = 3200 features
        super().__init__()
        self.fc = nn.Linear(input_size, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5) if dropout else nn.Identity()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        x = self.fc(self.relu(self.dropout(x)))
        return x