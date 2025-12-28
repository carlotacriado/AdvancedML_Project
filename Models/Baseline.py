import torch
import torch.nn as nn

# --- CONFIGURATION ---
CONV_NUM_FEATURES = 128
INPUT_IMAGE_SIZE = 84

class ConvBackbone(nn.Module):
    """
    Conv-4 Backbone Architecture.
    Designed for input images of size 84x84 (Standard for Few-Shot Learning / ProtoNets).
    """
    def __init__(self, in_channels=3, num_features=CONV_NUM_FEATURES):
        super().__init__()

        # Block 1: Input (Batch, 3, 84, 84)
        self.conv1 = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=num_features//4, num_channels=num_features)
        
        # Block 2
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=num_features//4, num_channels=num_features)

        # Block 3
        self.conv3 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.norm3 = nn.GroupNorm(num_groups=num_features//4, num_channels=num_features)
        
        # Block 4
        self.conv4 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.norm4 = nn.GroupNorm(num_groups=num_features//4, num_channels=num_features)

        # Shared layers
        self.maxpool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Block 1: 84x84 -> 42x42
        x = self.maxpool(self.relu(self.norm1(self.conv1(x))))
        
        # Block 2: 42x42 -> 21x21
        x = self.maxpool(self.relu(self.norm2(self.conv2(x))))
        
        # Block 3: 21x21 -> 10x10 (Padding allows for keeping size, MaxPool halves it)
        x = self.maxpool(self.relu(self.norm3(self.conv3(x))))
        
        # Block 4: 10x10 -> 5x5
        # Note: The original code did not apply ReLU in the final block, keeping features raw before the classifier.
        x = self.maxpool(self.norm4(self.conv4(x)))
        
        # Flatten: (Batch, 128, 5, 5) -> (Batch, 3200)
        x = x.view(x.size(0), -1) 
        return x

class ClassifierHead(nn.Module):
    """
    Simple Linear Classifier Head.
    """
    def __init__(self, num_classes, input_size=CONV_NUM_FEATURES*5*5, dropout=True):
        super().__init__()
        
        # Default input size calculation: 128 * 5 * 5 = 3200 features
        self.fc = nn.Linear(input_size, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5) if dropout else nn.Identity()

    def forward(self, x):
        # Apply activation and dropout before the final projection
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc(x)
        return x