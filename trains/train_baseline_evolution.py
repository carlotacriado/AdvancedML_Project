import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import numpy as np
import pandas as pd
import time
import wandb
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# --- LOCAL MODULES ---
from Utils.utils import set_all_seeds, seed_worker
from Dataloaders.dataloader import PokemonMetaDataset 
from Models.Baseline import ConvBackbone, ClassifierHead
from Utils.globals import *

# ==========================================
# 1. CONFIGURATION
# ==========================================
SEED = 151
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = MAX_EPOCHS  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# WandB Configuration
WANDB_KEY = "93d025aa0577b011c6d4081b9d4dc7daeb60ee6b" 
WANDB_PROJECT = "Baseline_Evolution_Task"

# Data Paths
CSV_PATH = "Data/pokemon_data_linked.csv"
ROOT_DIR = "Data/pokemon_sprites"
SAVE_DIR = "Results/Models_pth/Baseline_pth/"

# ==========================================
# 2. DATASET WRAPPER (SPECIES -> FAMILIES)
# ==========================================
class FamilyMappedDataset(Dataset):
    def __init__(self, original_dataset, indices, species_to_family_map, family_to_label_map):
        self.dataset = original_dataset
        self.indices = indices
        self.species_to_family_map = species_to_family_map
        self.family_to_label_map = family_to_label_map

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        image, species_id = self.dataset[real_idx]
        fam_id = self.species_to_family_map.get(species_id)
        target_label = self.family_to_label_map.get(fam_id, -100)
        return image, target_label

# ==========================================
# 3. DATA PREPARATION
# ==========================================
def get_evolution_dataloaders(batch_size, seed):
    print("\n--- Setting up Dataloaders by FAMILIES ---")
    df = pd.read_csv(CSV_PATH)
    species_to_family = pd.Series(df.family_id.values, index=df['dex_number'].values).to_dict()
    unique_families = df['family_id'].unique()
    
    train_fams, temp_fams = train_test_split(unique_families, test_size=0.2, random_state=seed)
    val_fams, test_fams   = train_test_split(temp_fams, test_size=0.5, random_state=seed)
    
    family_to_label = {fam_id: i for i, fam_id in enumerate(train_fams)}
    num_classes = len(train_fams)
    
    train_transform = transforms.Compose([
        transforms.Resize((120, 120)),              
        transforms.RandomCrop((84, 84)),            
        transforms.RandomHorizontalFlip(p=0.5),     
        transforms.RandomRotation(degrees=15),      
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), 
        transforms.ToTensor()
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize((84,84)),
        transforms.ToTensor()
    ])
    
    full_ds_train = PokemonMetaDataset(csv_file=CSV_PATH, root_dir=ROOT_DIR, transform=train_transform)
    full_ds_val   = PokemonMetaDataset(csv_file=CSV_PATH, root_dir=ROOT_DIR, transform=eval_transform)
    
    all_train_indices = []
    for species_id, indices_list in full_ds_train.indices_by_label.items():
        fam_id = species_to_family.get(species_id)
        if fam_id in train_fams:
            all_train_indices.extend(indices_list)

    train_idx, val_idx = train_test_split(all_train_indices, test_size=0.2, random_state=seed)
    
    train_ds = FamilyMappedDataset(full_ds_train, train_idx, species_to_family, family_to_label)
    val_ds   = FamilyMappedDataset(full_ds_val, val_idx, species_to_family, family_to_label)
    
    g = torch.Generator()
    g.manual_seed(seed)
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, 
        num_workers=2, worker_init_fn=seed_worker, generator=g
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, 
        num_workers=2, worker_init_fn=seed_worker, generator=g
    )
    
    return train_loader, val_loader, num_classes

# ==========================================
# 4. TRAINING LOOPS
# ==========================================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
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
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
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

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
def main():
    set_all_seeds(SEED)
    wandb.login(key=WANDB_KEY)
    
    run_name = f"Baseline_EVOLUTION_Random_Seed{SEED}"
    wandb.init(
        project=WANDB_PROJECT,
        name=run_name,
        config={"lr": LR, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "seed": SEED}
    )
    
    print(f"--- Starting Baseline Training: EVOLUTION TASK ---")
    
    # 1. Setup Data
    train_loader, val_loader, num_classes = get_evolution_dataloaders(BATCH_SIZE, SEED)
    print(f"Classifier Output Size: {num_classes} Families")
    
    # 2. Model Initialization
    backbone = ConvBackbone()
    classifier = ClassifierHead(num_classes=num_classes)
    full_model = nn.Sequential(backbone, classifier).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(full_model.parameters(), lr=LR)
    
    # 3. Training Loop
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, f"baseline_evolution_random_seed{SEED}.pth")
    best_val_acc = 0.0
    
    # --- [TIMING START] TOTAL TRAINING TIME ---
    if torch.cuda.is_available(): torch.cuda.synchronize()
    total_train_start = time.time()

    for epoch in range(EPOCHS):
        start = time.time()
        
        train_loss, train_acc = train_epoch(full_model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc     = validate(full_model, val_loader, criterion, DEVICE)
        
        elapsed = time.time() - start
        
        print(f"Epoch {epoch+1}/{EPOCHS} | {elapsed:.1f}s | "
              f"Train: {train_loss:.4f} ({train_acc:.2f}%) | "
              f"Val: {val_loss:.4f} ({val_acc:.2f}%)")
        
        wandb.log({
            "epoch": epoch+1,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
            "epoch_time": elapsed
        })
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(full_model.state_dict(), save_path)
            print(f"  --> New Best Model Saved (Acc: {best_val_acc:.2f}%)")

    # --- [TIMING END] TOTAL TRAINING TIME ---
    if torch.cuda.is_available(): torch.cuda.synchronize()
    total_train_end = time.time()
    total_train_time = total_train_end - total_train_start

    print(f"\n--- RESUMEN DE TIEMPOS BASELINE (EVOLUTION) ---")
    print(f"Tiempo Total de Entrenamiento: {total_train_time:.2f} segundos")
    wandb.log({"total_train_time": total_train_time})

    print(f"\nTraining Finished. Best Acc: {best_val_acc:.2f}%")
    wandb.finish()

if __name__ == '__main__':
    main()