import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import numpy as np
import time
from pathlib import Path
import wandb
import os

# --- IMPORTS ---
from Utils.utils import set_all_seeds
from Dataloaders.dataloader_baseline import get_baseline_dataloaders
from Dataloaders.dataloader import PokemonMetaDataset, get_structured_splits 
from Models.Baseline import ConvBackbone, ClassifierHead
from Utils.globals import *

# ==========================================
# 1. CONFIGURACION Y HIPERPARAMETROS
# ==========================================
SEED = 151
TASK = 'pokedex'     # 'pokedex' o 'oak'
SPLIT_MODE = 'random'
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = MAX_EPOCHS  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# WandB Config
WANDB_KEY = "93d025aa0577b011c6d4081b9d4dc7daeb60ee6b" 
WANDB_PROJECT = "Baseline_model"

# Rutas
CSV_PATH = "Data/pokemon_data_linked.csv"
ROOT_DIR = "Data/pokemon_sprites"

# ==========================================
# 2. FIJAR SEMILLA
# ==========================================
set_all_seeds(SEED)

def train_epoch(model, loader, criterion, optimizer, device):
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
    wandb.login(key=WANDB_KEY)
    wandb.init(
        project=WANDB_PROJECT,
        name=f"Baseline_{TASK}_{SPLIT_MODE}_seed{SEED}_Augmented", # Anado tag Augmented
        config={
            "learning_rate": LR,
            "architecture": "Conv4",
            "dataset": "Pokemon",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "task": TASK,
            "split_mode": SPLIT_MODE,
            "augmentation": True # Para que quede constancia en WandB
        }
    )

    print(f"--- Iniciando Entrenamiento Baseline CON AUGMENTATION (Seed: {SEED}) ---")
    
    # 3.1 Definir Transformaciones
    # Transformacion LIMPIA (Para validacion y splits)
    eval_transform = transforms.Compose([
        transforms.Resize((84,84)),
        transforms.ToTensor()
    ])

    # Transformacion AUMENTADA (Igual que en Reptile)
    # Nota: Reptile usaba Resize 120 -> RandomCrop 84. Esto es clave.
    train_transform = transforms.Compose([
        transforms.Resize((120, 120)),            
        transforms.RandomCrop((84, 84)),          
        transforms.RandomHorizontalFlip(p=0.5),   
        transforms.RandomRotation(degrees=15),    
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), 
        transforms.ToTensor()
    ])
    
    # 3.2 Crear DOS Datasets
    # Este se usa para calcular que pokemons van a train/val (porque no altera los IDs)
    # y para la validacion real.
    val_dataset = PokemonMetaDataset(csv_file=CSV_PATH, root_dir=ROOT_DIR, transform=eval_transform)
    
    # Este se usa SOLO para alimentar el train_loader con imagenes variadas
    train_dataset = PokemonMetaDataset(csv_file=CSV_PATH, root_dir=ROOT_DIR, transform=train_transform)

    # 3.3 Obtener los Splits (usando el dataset limpio)
    if SPLIT_MODE == 'random':
        train_labels, test_labels, val_labels = get_structured_splits(val_dataset, split_mode='random')
    else:
        train_labels, test_labels, val_labels = get_structured_splits(
            val_dataset, 
            split_mode='generation', 
            train_vals=['generation-i', 'generation-ii', 'generation-iii'],
            test_vals=['generation-iv']
        )
    
    # 3.4 Crear Dataloaders con la nueva funcion que acepta dos datasets
    train_loader, val_loader, num_classes = get_baseline_dataloaders(
        train_dataset=train_dataset,  # Pasa el dataset aumentado
        val_dataset=val_dataset,      # Pasa el dataset limpio
        train_labels=train_labels,    # Labels permitidos
        task_mode=TASK,
        batch_size=BATCH_SIZE,
        seed=SEED
    )
    
    print(f"Clases de salida detectadas: {num_classes}")

    # ------------------------------------------
    # 4. INICIALIZAR MODELO (Igual que antes)
    # ------------------------------------------
    backbone = ConvBackbone()
    classifier = ClassifierHead(num_classes=num_classes)
    full_model = nn.Sequential(backbone, classifier).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(full_model.parameters(), lr=LR)
    
    wandb.watch(full_model, log="all")

    # ------------------------------------------
    # 5. BUCLE DE ENTRENAMIENTO (Igual que antes)
    # ------------------------------------------
    best_acc = 0.0
    
    # Crear directorio si no existe
    save_dir = "Results/Models_pth/Baseline_pth"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(EPOCHS):
        start_time = time.time()
        
        train_loss, train_acc = train_epoch(full_model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(full_model, val_loader, criterion, DEVICE)
        
        elapsed = time.time() - start_time
        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Time: {elapsed:.1f}s")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(save_dir, f"baseline_{TASK}_{SPLIT_MODE}_seed{SEED}_aug.pth")
            torch.save(full_model.state_dict(), save_path)
            print(f"  --> Modelo guardado en {save_path}")

    print(f"\nEntrenamiento finalizado. Mejor Val Acc: {best_acc:.2f}%")
    wandb.finish()
    
if __name__ == '__main__':
    main()