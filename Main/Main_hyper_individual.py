import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import time
import wandb
import numpy as np

# --- IMPORTS DE TUS ARCHIVOS ---
from Utils.utils import set_all_seeds
from Utils.globals import *
from Dataloaders.dataloader import PokemonMetaDataset, get_structured_splits, get_meta_dataloaders_pokedex, get_meta_dataloaders_oak
from Models.Baseline import ConvBackbone # Reusamos el backbone
from Models.Hypernetwork import HyperNetworkModel # Tu nuevo modelo
from trains.train_hyper import train_episode, validate_episode # Las funciones de arriba

# --- CONFIG ---
TASK = 'pokedex'  # 'pokedex' o 'oak'
SPLIT_MODE = 'random' # 'random' o 'generation'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_WAY = 5
N_SHOT = 1     # Support Size (K)
N_QUERY = 1   # Query Size (Q)

# Hypernetwork Config
EMBEDDING_SIZE = 128 * 5 * 5 # 3200 features salen del Conv4 antes del flatten
# IMPORTANTE: Si tu ConvBackbone ya tiene un flatten final a 128, cambia esto a 128.
# Mirando Baseline.py: "x = x.view(x.size(0), -1)". El output es 5x5x128 = 3200.
# A MENOS QUE cambies el backbone para tener un FC final.
# Vamos a asumir que usas el ConvBackbone tal cual: output es 3200.

WANDB_PROJECT = "Hypernetwork"
WANDB_KEY = "93d025aa0577b011c6d4081b9d4dc7daeb60ee6b"

def main():
    # 1. SETUP
    set_all_seeds(SEED)
    
    wandb.login(key=WANDB_KEY)
    wandb.init(
        project=WANDB_PROJECT,
        name=f"Hypernet_{TASK}_{SPLIT_MODE}_{N_WAY}way_{N_SHOT}shot",
        config={
            "task": TASK,
            "n_way": N_WAY,
            "k_shot": N_SHOT,
            "q_query": N_QUERY,
            "lr": 1e-4, # Hypernets suelen necesitar LRs más bajos
            "epochs": MAX_EPOCHS
        }
    )

    print(f"--- Hypernetwork Training: {TASK} ---")

    # 2. DATASETS
    transform = transforms.Compose([
        transforms.Resize((84,84)),
        transforms.ToTensor()
    ])
    dataset = PokemonMetaDataset(csv_file="Data/pokemon_data_linked.csv", root_dir="Data/pokemon_sprites", transform=transform)

    # Splits
    if SPLIT_MODE == 'random':
        train_labels, test_labels, val_labels = get_structured_splits(dataset, split_mode='random')
    else:
        train_labels, test_labels, val_labels = get_structured_splits(
          dataset, split_mode='generation', 
          train_vals=['generation-i', 'generation-ii', 'generation-iii'],
          test_vals=['generation-iv']
        )

    # Loaders (Episodic)
    loader_func = get_meta_dataloaders_pokedex if TASK == 'pokedex' else get_meta_dataloaders_oak
    
    train_loader, test_loader, val_loader = loader_func(
        dataset, train_labels, test_labels, val_labels,
        n_way=N_WAY, n_shot=N_SHOT, n_query=N_QUERY, episodes=EPISODES_PER_EPOCH
    )

    # 3. MODEL SETUP
    # Backbone
    backbone = ConvBackbone() 
    # Como el output del backbone es 5x5x128 = 3200, se lo pasamos a la hypernetwork
    hyper_model = HyperNetworkModel(backbone, feature_dim=3200, num_classes=N_WAY).to(DEVICE)

    optimizer = optim.Adam(hyper_model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # 4. TRAINING LOOP
    best_val_acc = 0.0

    for epoch in range(MAX_EPOCHS):
        start = time.time()
        
        # --- TRAIN ---
        train_loss_accum = 0.0
        train_acc_accum = 0.0
        
        for batch in train_loader:
            loss, acc = train_episode(hyper_model, batch, optimizer, criterion, N_WAY, N_SHOT, N_QUERY, DEVICE)
            train_loss_accum += loss
            train_acc_accum += acc
            
        avg_train_loss = train_loss_accum / EPISODES_PER_EPOCH
        avg_train_acc = train_acc_accum / EPISODES_PER_EPOCH

        # --- VALIDATION ---
        val_loss_accum = 0.0
        val_acc_accum = 0.0
        
        for batch in val_loader:
            loss, acc = validate_episode(hyper_model, batch, criterion, N_WAY, N_SHOT, N_QUERY, DEVICE)
            val_loss_accum += loss
            val_acc_accum += acc
            
        avg_val_loss = val_loss_accum / len(val_loader) # OJO: len(val_loader) = num_episodes
        avg_val_acc = val_acc_accum / len(val_loader)

        # --- LOGGING ---
        elapsed = time.time() - start
        print(f"Epoch {epoch+1}/{MAX_EPOCHS} | Time: {elapsed:.1f}s")
        print(f"  Train Loss: {avg_train_loss:.4f} | Acc: {avg_train_acc:.2f}%")
        print(f"  Val Loss:   {avg_val_loss:.4f} | Acc: {avg_val_acc:.2f}%")
        
        wandb.log({
            "epoch": epoch+1,
            "train_loss": avg_train_loss,
            "train_acc": avg_train_acc,
            "val_loss": avg_val_loss,
            "val_acc": avg_val_acc
        })

        # --- SAVE ---
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            save_path = f"Results/Models_pth/Hypernetwork_pth/Hypernet_{TASK}.pth"
            torch.save(hyper_model.state_dict(), save_path)
            print(f"  --> Model Saved! Best Acc: {best_val_acc:.2f}%")

    print("\nTraining Complete!")
    wandb.finish()

if __name__ == '__main__':
    main()