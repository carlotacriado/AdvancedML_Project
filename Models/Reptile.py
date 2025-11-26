import torch
from torch import nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.nn.functional as F
import torchvision
from torchvision import transforms

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import pandas as pd
from PIL import Image
from tqdm import tqdm
import os

from IPython.display import display, clear_output
#from torchinfo import summary

import sys
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from utils.utils import *
from utils.globals import *
from Baseline import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
import torch.nn as nn
import torch.optim as optim
import copy

# --- HYPERPARAMETERS ---
META_LR = 1e-3       # Outer loop learning rate (Adam)
INNER_LR = 0.01      # Inner loop learning rate (SGD usually works well for inner)
INNER_STEPS = 5      # How many steps to train on support set
EPSILON = 0.1        # Reptile step size (Soft update)

# --- SETUP ---
meta_model = ConvBackbone().to(device)
meta_optimizer = optim.Adam(meta_model.parameters(), lr=META_LR)
criterion = nn.CrossEntropyLoss()

# --- TRAINING LOOP ---
print("Starting Reptile Meta-Training...")

# Iterate over the DataLoader (which generates episodes automatically)
for batch_idx, (images, _) in enumerate(tqdm(train_loader)):
    
    # 1. PREPARE DATA
    # Reshape: [N_way, K_shot + Q_query, C, H, W]
    p, k, q = 5, SUPPORT_SIZE, QUERY_SIZE
    images = images.to(device)
    
    # View as [5, K+Q, C, H, W]
    # Note: Using labels from DataLoader is tricky because they are global indices.
    # In Meta-learning, we usually re-label them 0..N-1 for the current episode.
    
    data = images.view(p, k + q, 3, 84, 84)

    # Split Support (Inner Train) and Query (Inner Val/Outer Update)
    x_support = data[:, :k].contiguous().view(-1, 3, 84, 84) # [5*K, C, H, W]
    y_support = torch.arange(p).repeat_interleave(k).to(device) # [0,0,.., 1,1,..]
    
    x_query   = data[:, k:].contiguous().view(-1, 3, 84, 84) # [5*Q, C, H, W]
    y_query   = torch.arange(p).repeat_interleave(q).to(device)

    # 2. CREATE A CLONE FOR INNER LOOP (The "Fast" Model)
    # We clone the state dict to ensure we don't mess up the meta-model gradients yet
    fast_model = ConvBackbone().to(device)
    fast_model.load_state_dict(meta_model.state_dict())
    
    # Inner Optimizer (usually SGD for few-shot)
    inner_optimizer = optim.SGD(fast_model.parameters(), lr=INNER_LR)
    
    # 3. INNER LOOP (Fine-tune on Support Set)
    fast_model.train()
    for _ in range(INNER_STEPS):
        logits = fast_model(x_support)
        loss = criterion(logits, y_support)
        
        inner_optimizer.zero_grad()
        loss.backward()
        inner_optimizer.step()
        
    # 4. REPTILE UPDATE (Outer Loop)
    # Theta_new = Theta_old + Epsilon * (Theta_fast - Theta_old)
    # Which is equivalent to: Gradient = (Theta_old - Theta_fast)
    
    meta_optimizer.zero_grad()
    
    # We manually set the gradients of the meta_model
    for meta_param, fast_param in zip(meta_model.parameters(), fast_model.parameters()):
        # Calculate the "gradient" that moves meta towards fast
        # Note the sign: we want to move TOWARDS fast_param
        # Update rule: meta = meta + eps * (fast - meta)
        # Standard optimizers do: param = param - lr * grad
        # So: -lr * grad = eps * (fast - meta)
        # grad = -(eps/lr) * (fast - meta)
        # Let's just do the manual update without optimizer.step() to be clear, 
        # OR use the pseudo-gradient method you had:
        
        pseudo_grad = (meta_param.data - fast_param.data) # Direction: Meta -> Fast
        # If we use SGD on outer loop, we would subtract this. 
        # But you used Adam. Standard Reptile implementation often just does:
        # meta_param.data = meta_param.data + EPSILON * (fast_param.data - meta_param.data)
        
        # YOUR IMPLEMENTATION (Soft Update):
        meta_param.data.add_(fast_param.data - meta_param.data, alpha=EPSILON)

    # Note: If you do the manual .data update above, you do NOT call meta_optimizer.step()
    # If you want to use Adam for the outer loop (Meta-SGD / MAML style), you need to compute gradients properly.
    # But pure Reptile is just a soft weight update.
    
    # 5. EVALUATE (Optional: Check Query Set performance for logging)
    if batch_idx % 10 == 0:
        with torch.no_grad():
            fast_model.eval()
            q_logits = fast_model(x_query)
            q_loss = criterion(q_logits, y_query)
            q_acc = (q_logits.argmax(dim=1) == y_query).float().mean()
            print(f"Episode {batch_idx}: Query Loss {q_loss.item():.4f}, Acc {q_acc.item():.4f}")