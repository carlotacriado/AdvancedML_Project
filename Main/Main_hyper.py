import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import time
import wandb
import numpy as np
import os
import gc

# --- LOCAL MODULES ---
from Utils.utils import set_all_seeds
from Utils.globals import *
from Dataloaders.dataloader import PokemonMetaDataset, get_structured_splits, get_meta_dataloaders_pokedex, get_meta_dataloaders_oak
from Models.Baseline import ConvBackbone 
from Models.Hypernetwork import HyperNetworkModel 
from trains.train_hyper import train_episode, validate_episode 

# ==========================================
# 1. CONFIGURATION & HYPERPARAMETERS
# ==========================================
TASK = 'oak'  # 'pokedex' o 'oak'
SPLIT_MODE = 'generation' # 'random' o 'generation'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_WAY_LIST = [5]
N_SHOT_LIST = [5]     # Support Size (K)
N_QUERY = 1   # Query Size (Q)

# Hypernetwork Config
EMBEDDING_SIZE = 128 * 5 * 5 # 3200 features salen del Conv4 antes del flatten

# WandB Configuration
WANDB_PROJECT = "Hypernetwork_oak_tiempos"
WANDB_KEY = "93d025aa0577b011c6d4081b9d4dc7daeb60ee6b"

def main():
    wandb.login(key=WANDB_KEY)

    # Data Paths
    save_dir = "Results/Models_pth/Hypernetwork_pth/oak"
    os.makedirs(save_dir, exist_ok=True)
    
    for N_WAY in N_WAY_LIST:
        for N_SHOT in N_SHOT_LIST:
            
            print(f"\n" + "="*40)
            print(f">>> ENTRENANDO: {N_WAY}-Way {N_SHOT}-Shot")
            print("="*40)
            
            # 1. SETUP
            set_all_seeds(SEED)
            
            run_name = f"HN_{TASK}_{SPLIT_MODE}_{N_WAY}way_{N_SHOT}shot"
    
            wandb.init(
                project=WANDB_PROJECT,
                name=run_name,
                group=f"{N_WAY}Way_Experiments",
                reinit=True,       
                config={
                    "task": TASK,
                    "n_way": N_WAY,
                    "k_shot": N_SHOT,
                    "q_query": N_QUERY,
                    "lr": 1e-4, # Hypernets usually need lower LRs
                    "epochs": MAX_EPOCHS
                }
            )


            # 2. DATASETS
            dataset = PokemonMetaDataset(csv_file="Data/pokemon_data_linked.csv", root_dir="Data/pokemon_sprites", transform=EVAL_TRANSFORMS)
            
            # Splits
            if SPLIT_MODE == 'random':
                train_labels, test_labels, val_labels = get_structured_splits(dataset, split_mode='random')
            else:
                train_labels, test_labels, val_labels = get_structured_splits(
                dataset, split_mode='type', 
                train_vals=['fairy', 'dark', 'dragon', 'rock', 'bug', 'psychic', 'flying', 'water', 'fire', 'grass'],
                val_vals=['steel', 'ground', 'ghost'],
                test_vals=['ice', 'poison', 'fighting', 'electric', 'normal']
                )

            # Loaders (Episodic)
            loader_func = get_meta_dataloaders_pokedex if TASK == 'pokedex' else get_meta_dataloaders_oak
            
            # We pass the variables of the current loop
            train_loader, test_loader, val_loader = loader_func(
                dataset, None, train_labels, test_labels, val_labels,
                n_way=N_WAY, n_shot=N_SHOT, n_query=N_QUERY, episodes=EPISODES_PER_EPOCH
                )

            # 3. MODEL SETUP
            # Backbone
            backbone = ConvBackbone() 
            hyper_model = HyperNetworkModel(backbone, feature_dim=EMBEDDING_SIZE, num_classes=N_WAY).to(DEVICE)

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
                    
                avg_val_loss = val_loss_accum / len(val_loader) 
                avg_val_acc = val_acc_accum / len(val_loader)

                # --- LOGGING ---
                if (epoch + 1) % 5 == 0 or epoch == 0:
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
                    filename = f"Hypernet_{TASK}_{SPLIT_MODE}_{N_WAY}way_{N_SHOT}shot.pth"
                    save_path = os.path.join(save_dir, filename)                    
                    torch.save(hyper_model.state_dict(), save_path)
                    print(f"  --> Model Saved! Best Acc: {best_val_acc:.2f}%")

            print(f">>> FINISHED. Best Acc: {best_val_acc:.2f}%. Model saved: {filename}")
            wandb.finish()
            
            # Clean up GPU memory before the next loop
            del hyper_model
            del optimizer
            del backbone
            torch.cuda.empty_cache()
            gc.collect()
            
    print("\n=== TODOS LOS EXPERIMENTOS COMPLETADOS ===")

if __name__ == '__main__':
    main()
