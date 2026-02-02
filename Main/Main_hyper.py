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
SPLIT_MODE = 'random' # 'random', 'generation', 'type'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Grid Search (o configuración única si prefieres)
N_WAY_LIST = [5]
N_SHOT_LIST = [5]     
N_QUERY = 1   

# Hypernetwork Config
EMBEDDING_SIZE = 128 * 5 * 5 # Conv4 output flattened

# WandB Configuration
WANDB_PROJECT = "Hypernetwork_Time_Train"
WANDB_KEY = "93d025aa0577b011c6d4081b9d4dc7daeb60ee6b"

# Paths
BASE_PATH = os.getcwd() # Asumiendo ejecución desde root
CSV_PATH = os.path.join(BASE_PATH, "Data/pokemon_data_linked.csv")
IMGS_PATH = os.path.join(BASE_PATH, "Data/pokemon_sprites")
SAVE_DIR = os.path.join(BASE_PATH, "Results/Models_pth/Hypernet_pth/")

LR = 1e-4

def main():
    set_all_seeds(151)
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Login WandB
    wandb.login(key=WANDB_KEY)

    # Transformaciones
    train_trans = transforms.Compose([
        transforms.Resize((84, 84)),
        transforms.ToTensor()
    ])
    
    # Dataset
    dataset = PokemonMetaDataset(csv_file=CSV_PATH, root_dir=IMGS_PATH, transform=train_trans)

    # Splits
    if SPLIT_MODE == 'random':
        train_lbls, val_lbls, test_lbls = get_structured_splits(dataset, split_mode='random')
    elif SPLIT_MODE == 'generation':
         train_lbls, _, val_lbls = get_structured_splits(dataset, split_mode='generation', 
                                                         train_vals=['generation-i', 'generation-iii'],
                                                         val_vals=['generation-ii'], test_vals=['generation-iv'])
    else:
        # Añade aquí lógica para 'type' si es necesario
        train_lbls, val_lbls, test_lbls = get_structured_splits(dataset, split_mode='random')

    # --- GRID SEARCH LOOP ---
    for N_WAY in N_WAY_LIST:
        for N_SHOT in N_SHOT_LIST:
            
            run_name = f"HyperNet_{TASK}_{SPLIT_MODE}_{N_WAY}way_{N_SHOT}shot"
            print(f"\n=== STARTING RUN: {run_name} ===")
            
            wandb.init(
                project=WANDB_PROJECT,
                name=run_name,
                config={
                    "n_way": N_WAY, "n_shot": N_SHOT, "lr": LR,
                    "task": TASK, "split": SPLIT_MODE
                },
                reinit=True
            )

            # Dataloaders
            train_loader = get_meta_dataloaders_oak(dataset, train_lbls, N_WAY, N_SHOT, N_QUERY, EPISODES_PER_EPOCH)
            val_loader   = get_meta_dataloaders_oak(dataset, val_lbls,   N_WAY, N_SHOT, N_QUERY)

            # Model Init
            backbone = ConvBackbone().to(DEVICE)
            hyper_model = HyperNetworkModel(backbone, embedding_size=EMBEDDING_SIZE, 
                                            n_way=N_WAY, n_query=N_QUERY).to(DEVICE)
            
            optimizer = optim.Adam(hyper_model.parameters(), lr=LR)
            criterion = nn.CrossEntropyLoss()
            
            best_val_acc = 0.0
            filename = f"Hypernet_{TASK}_{SPLIT_MODE}_{N_WAY}way_{N_SHOT}shot.pth"
            save_path = os.path.join(SAVE_DIR, filename)

            # --- [TIMING START] TOTAL TRAINING TIME ---
            if torch.cuda.is_available(): torch.cuda.synchronize()
            total_train_start = time.time()

            for epoch in range(MAX_EPOCHS):
                start = time.time()
                
                # Train
                hyper_model.train()
                train_losses, train_accs = [], []
                for batch in train_loader:
                    loss, acc = train_episode(hyper_model, batch, optimizer, criterion, N_WAY, N_SHOT, N_QUERY, DEVICE)
                    train_losses.append(loss)
                    train_accs.append(acc)
                
                # Validate
                hyper_model.eval()
                val_losses, val_accs = [], []
                with torch.no_grad():
                    for batch in val_loader:
                        loss, acc = validate_episode(hyper_model, batch, criterion, N_WAY, N_SHOT, N_QUERY, DEVICE)
                        val_losses.append(loss)
                        val_accs.append(acc)

                avg_train_loss = np.mean(train_losses)
                avg_train_acc  = np.mean(train_accs)
                avg_val_loss   = np.mean(val_losses)
                avg_val_acc    = np.mean(val_accs)
                
                elapsed = time.time() - start

                print(f"Epoch {epoch+1}/{MAX_EPOCHS} | {elapsed:.1f}s | Train Acc: {avg_train_acc:.2f}% | Val Acc: {avg_val_acc:.2f}%")

                wandb.log({
                    "epoch": epoch+1,
                    "train_loss": avg_train_loss, "train_acc": avg_train_acc,
                    "val_loss": avg_val_loss, "val_acc": avg_val_acc,
                    "epoch_time": elapsed
                })

                # Save Best
                if avg_val_acc > best_val_acc:
                    best_val_acc = avg_val_acc
                    torch.save(hyper_model.state_dict(), save_path)
                    print(f"  --> New Best Model Saved: {best_val_acc:.2f}%")

            # --- [TIMING END] TOTAL TRAINING TIME ---
            if torch.cuda.is_available(): torch.cuda.synchronize()
            total_train_end = time.time()
            total_train_time = total_train_end - total_train_start
            
            print(f"\n--- TIEMPO TOTAL DE ENTRENAMIENTO: {total_train_time:.2f} segundos ---")
            
            # Guardamos el tiempo total en WandB para comparar con el baseline
            wandb.log({"total_train_time": total_train_time})
            wandb.finish()
            
            # Cleanup
            del hyper_model, backbone, optimizer
            torch.cuda.empty_cache()
            gc.collect()

if __name__ == '__main__':
    main()