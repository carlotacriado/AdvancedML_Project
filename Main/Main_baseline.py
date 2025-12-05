import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import numpy as np
import time
from pathlib import Path

# --- IMPORTS ---
from Utils.utils import set_all_seeds, save_plots
from Dataloaders.dataloader_baseline import get_baseline_dataloaders
from Dataloaders.dataloader import PokemonMetaDataset, get_structured_splits 
from Models.Baseline import ConvBackbone, ClassifierHead
from Utils.globals import *

# ==========================================
# 1. CONFIGURACION Y HIPERPARAMETROS
# ==========================================
SEED = SEED if 'SEED' in globals() else 151        # La misma que usaras en Meta-Learning
TASK = 'pokedex'     # Opciones: 'pokedex' (Especies) o 'oak' (Familias)
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Rutas (Ajusta esto a tu proyecto)
CSV_PATH = "Data/pokemon_data_linked.csv"
ROOT_DIR = "Data/pokemon_sprites"

# ==========================================
# 2. FIJAR SEMILLA (El paso mas importante)
# ==========================================
# Esto garantiza que el split de datos sea identico al de tu experimento de Meta-Learning
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
    print(f"--- Iniciando Entrenamiento Baseline (Seed: {SEED}) ---")
    print(f"--- Tarea: {TASK.upper()} ---")
    
    # ------------------------------------------
    # 3. PREPARAR DATOS
    # ------------------------------------------
    transform_pipeline = transforms.Compose([
      transforms.Resize((84,84)),
      transforms.ToTensor()
      ])
    
    dataset = PokemonMetaDataset(csv_file=CSV_PATH, root_dir=ROOT_DIR, transform=transform_pipeline) 
    # NOTA: Anade tu transform (ToTensor, Resize) en el dataset si no lo has hecho ya dentro

    # 3.1. Obtener los mismos Splits que en Meta-Learning
    # Ajusta train_vals/val_vals segun tu experimento real
    train_labels, test_labels, val_labels = get_structured_splits(
        dataset, 
        split_mode='generation', 
        train_vals=['generation-i', 'generation-ii', 'generation-iii'],
        # val_vals=['generation-iii'],        
        test_vals=['generation-iv']
    )

    # 3.2. Crear Dataloaders Baseline (Aqui ocurre la magia del mapeo)
    train_loader, val_loader, num_classes = get_baseline_dataloaders(
        dataset, 
        train_labels, 
        val_labels, 
        task_mode=TASK, 
        batch_size=BATCH_SIZE, 
        seed=SEED
    )
    
    print(f"Clases de salida detectadas: {num_classes}")

    # ------------------------------------------
    # 4. INICIALIZAR MODELO
    # ------------------------------------------
    backbone = ConvBackbone()
    
    # IMPORTANTE: Inicializamos la cabeza con el numero exacto de clases activas
    # Si TASK='pokedex', num_classes sera ~60
    # Si TASK='oak', num_classes sera ~25
    classifier = ClassifierHead(num_classes=num_classes)
    
    full_model = nn.Sequential(backbone, classifier).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(full_model.parameters(), lr=LR)

    # ------------------------------------------
    # 5. BUCLE DE ENTRENAMIENTO
    # ------------------------------------------
    best_acc = 0.0
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        train_loss, train_acc = train_epoch(full_model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(full_model, val_loader, criterion, DEVICE)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        elapsed = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Time: {elapsed:.1f}s")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Guardar el mejor modelo
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = f"baseline_{TASK}_seed{SEED}.pth"
            torch.save(full_model.state_dict(), save_path)
            print(f"  --> Modelo guardado en {save_path}")

    print(f"\nEntrenamiento finalizado. Mejor Val Acc: {best_acc:.2f}%")

    save_plots(
        history['train_acc'], 
        history['val_acc'], 
        history['train_loss'], 
        history['val_loss'], 
        filename=f"curves_{TASK}_seed{SEED}.png"
    )
    
if __name__ == '__main__':
    main()