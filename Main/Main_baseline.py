import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import numpy as np
import time
import wandb
import os

# --- LOCAL MODULES ---
from Utils.utils import set_all_seeds
from Dataloaders.dataloader_baseline import get_baseline_dataloaders
from Dataloaders.dataloader import PokemonMetaDataset, get_structured_splits 
from Models.Baseline import ConvBackbone, ClassifierHead
from Utils.globals import *

# ==========================================
# 1. CONFIGURATION & HYPERPARAMETERS
# ==========================================
SEED = 151
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Experiment Settings
TASK = 'pokedex'      # Standard Classification
SPLIT_MODE = 'type'   # Options: 'random', 'generation', 'type'

# Training Hyperparameters
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = MAX_EPOCHS  
SAVE_DIR = "Results/Models_pth/Baseline_pth/"

# WandB Configuration
WANDB_KEY = "93d025aa0577b011c6d4081b9d4dc7daeb60ee6b" 
WANDB_PROJECT = "Baseline_model_Tiempos"

# Data Paths
CSV_PATH = "Data/pokemon_data_linked.csv"
ROOT_DIR = "Data/pokemon_sprites"

# ==========================================
# 2. SETUP
# ==========================================
set_all_seeds(SEED)

def train_epoch(model, loader, criterion, optimizer, device):
    """Standard Supervised Training Epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    return running_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device):
    """Validation Loop"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return running_loss / len(loader), 100. * correct / total

def main():
    # --- WandB Init ---
    wandb.login(key=WANDB_KEY)
    wandb.init(
        project=WANDB_PROJECT,
        name=f"Baseline_{TASK}_{SPLIT_MODE}_seed{SEED}_Augmented",
        config={
            "learning_rate": LR,
            "architecture": "Conv4",
            "dataset": "Pokemon",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "task": TASK,
            "split_mode": SPLIT_MODE,
            "augmentation": True
        }
    )

    print(f"--- Starting Baseline Training (Augmented) | Seed: {SEED} ---")
    
    # --- 3. Data Preparation ---
    
    # Transform for Validation (Clean)
    eval_transform = transforms.Compose([
        transforms.Resize((84,84)),
        transforms.ToTensor()
    ])

    # Transform for Training (Heavily Augmented)
    train_transform = transforms.Compose([
        transforms.Resize((120, 120)),              
        transforms.RandomCrop((84, 84)),            
        transforms.RandomHorizontalFlip(p=0.5),     
        transforms.RandomRotation(degrees=15),      
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), 
        transforms.ToTensor()
    ])
    
    # Initialize Datasets
    val_dataset = PokemonMetaDataset(csv_file=CSV_PATH, root_dir=ROOT_DIR, transform=eval_transform)
    train_dataset = PokemonMetaDataset(csv_file=CSV_PATH, root_dir=ROOT_DIR, transform=train_transform)

    # Configure Splits
    print(f"Configuring splits for mode: {SPLIT_MODE}")

    if SPLIT_MODE == 'random':
        train_labels, test_labels, val_labels = get_structured_splits(val_dataset, split_mode='random')

    elif SPLIT_MODE == 'generation':
        train_labels, test_labels, val_labels = get_structured_splits(
            val_dataset, 
            split_mode='generation', 
            train_vals=['generation-i', 'generation-iii'], 
            val_vals=['generation-ii'],                     
            test_vals=['generation-iv']                     
        )

    elif SPLIT_MODE == 'type':
        train_types = ['water', 'normal', 'grass', 'bug', 'fire', 'psychic', 'poison']
        val_types   = ['electric', 'ground', 'rock', 'fighting']
        test_types  = ['ghost', 'dragon', 'ice', 'steel', 'dark', 'flying', 'fairy']

        train_labels, test_labels, val_labels = get_structured_splits(
            val_dataset,
            split_mode='type',
            train_vals=train_types,
            val_vals=val_types,
            test_vals=test_types
        )
    else:
        raise ValueError(f"Unknown Split Mode: {SPLIT_MODE}")
    
    # Create Dataloaders
    train_loader, val_loader, num_classes = get_baseline_dataloaders(
        train_dataset=train_dataset, 
        val_dataset=val_dataset,       
        train_labels=train_labels,    
        val_labels=val_labels,
        task_mode=TASK,
        batch_size=BATCH_SIZE,
        seed=SEED
    )
    
    print(f"Training on {len(train_labels)} classes | Validating on {len(val_labels)} classes")
    print(f"Classifier Output Size: {num_classes}")

    # --- 4. Model Initialization ---
    backbone = ConvBackbone()
    classifier = ClassifierHead(num_classes=num_classes)
    full_model = nn.Sequential(backbone, classifier).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(full_model.parameters(), lr=LR)
    
    wandb.watch(full_model, log="all")

    # --- 5. Training Loop ---
    best_val_acc = 0.0
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, f"baseline_{TASK}_{SPLIT_MODE}_seed{SEED}.pth")

    # --- [TIMING START] TOTAL TRAINING TIME ---
    if torch.cuda.is_available(): torch.cuda.synchronize()
    total_train_start = time.time()

    for epoch in range(EPOCHS):
        start_time = time.time()
        
        train_loss, train_acc = train_epoch(full_model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(full_model, val_loader, criterion, DEVICE)
        
        elapsed = time.time() - start_time
        
        # Logging
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "epoch_time": elapsed # Log time per epoch
        })
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Time: {elapsed:.1f}s")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Save Best Model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(full_model.state_dict(), save_path)
            print(f"  --> New Best Model Saved (Acc: {best_val_acc:.2f}%)")

    # --- [TIMING END] TOTAL TRAINING TIME ---
    if torch.cuda.is_available(): torch.cuda.synchronize()
    total_train_end = time.time()
    total_train_time = total_train_end - total_train_start
    
    print(f"\n--- RESUMEN DE TIEMPOS BASELINE (MAIN) ---")
    print(f"Tiempo Total de Entrenamiento: {total_train_time:.2f} segundos")

    # --- [TIMING] INFERENCE LATENCY CHECK ---
    # Measure raw inference speed (without backward pass)
    full_model.eval()
    sample_img, _ = next(iter(val_loader))
    sample_img = sample_img[:1].to(DEVICE) # Single image batch
    
    # Warmup
    for _ in range(10): _ = full_model(sample_img)
    
    if torch.cuda.is_available(): torch.cuda.synchronize()
    inf_start = time.time()
    for _ in range(100):
        _ = full_model(sample_img)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    inf_end = time.time()
    
    avg_inf_latency = (inf_end - inf_start) / 100.0
    print(f"Latencia media de inferencia (1 imagen): {avg_inf_latency:.6f} segundos")
    
    wandb.log({
        "total_train_time": total_train_time,
        "inference_latency": avg_inf_latency
    })

    print(f"\nTraining Finished. Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {save_path}")
    wandb.finish()
    
if __name__ == '__main__':
    main()