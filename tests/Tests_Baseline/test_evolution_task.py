import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import numpy as np
import pandas as pd
import os
import time  # Import time
import wandb # Import wandb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- LOCAL MODULES ---
from Utils.utils import set_all_seeds 
from Dataloaders.dataloader import PokemonMetaDataset
from Dataloaders.sampler import Oak
from Models.Baseline import ConvBackbone

# ==========================================
# 1. CONFIGURATION & HYPERPARAMETERS
# ==========================================
SEED = 151
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
TASK_NAME = "OAK_Task_Horizontal" 
MODEL_PATH = "Results/Models_pth/Baseline_pth/baseline_evolution_random_seed151.pth"
RESULTS_DIR = f"Results/Visuals/Baseline_{TASK_NAME}"

# WandB Configuration
WANDB_KEY = "93d025aa0577b011c6d4081b9d4dc7daeb60ee6b"
WANDB_PROJECT = "Baseline_Evaluation_Time" # Proyecto específico para evaluación

# Meta-Testing Parameters
N_WAYS_LIST = [5]      
K_SHOTS_LIST = [5]  
N_QUERY = 1                     
N_EPISODES = 600                

# Fine-Tuning (Inner Loop) Hyperparameters
FT_LR = 0.01        
FT_STEPS = 12        

SUPPORT_AUGMENTATIONS = nn.Sequential(
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.8, 1.2)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
)

# ==========================================
# 2. VISUALIZATION UTILITIES
# ==========================================
def denormalize_robust(tensor):
    img = tensor.cpu().detach().numpy().squeeze()
    if img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose((1, 2, 0))
    return np.clip(img, 0, 1)

def save_oak_visualization(support_x, support_species_y, query_x, query_species_y, 
                           preds_local, dataset, save_path, n_way, k_shot):
    rows = n_way
    cols = k_shot + N_QUERY
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.5 * rows))
    if rows == 1: axes = np.expand_dims(axes, 0)
    if cols == 1: axes = np.expand_dims(axes, 1)
    class_names_map = {} 

    for r in range(rows): 
        for k in range(k_shot): 
            idx = r * k_shot + k
            img = support_x[idx]
            specie_name = dataset.idx_to_name.get(support_species_y[idx].item(), "Unknown")
            if k == 0: class_names_map[r] = specie_name
            ax = axes[r, k]
            try: ax.imshow(denormalize_robust(img))
            except: pass
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values(): spine.set_visible(False)
            if k == 0:
                ax.set_ylabel(f"Class {r}", fontsize=12, fontweight='bold')
                ax.set_title(f"Sup: {specie_name}", fontsize=9, color='blue')
            else:
                ax.set_title(f"Sup: {specie_name}", fontsize=9, color='gray')

    for r in range(rows):
        q_idx = r 
        img = query_x[q_idx]
        real_name = dataset.idx_to_name.get(query_species_y[q_idx].item(), "Unknown")
        pred_idx = preds_local[q_idx].item()
        is_correct = (pred_idx == r)
        color = 'green' if is_correct else 'red'
        ax = axes[r, k_shot]
        try: ax.imshow(denormalize_robust(img))
        except: pass
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
        pred_name = class_names_map.get(pred_idx, f"Class {pred_idx}")
        title_text = f"QUERY\nTrue: {real_name}\n{'[MATCH]' if is_correct else f'Pred: {pred_name}'}"
        ax.set_title(title_text, fontsize=10, fontweight='bold', color=color)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

# ==========================================
# 3. FINE-TUNING & PREDICTION (INNER LOOP)
# ==========================================
class AdapterHead(nn.Module):
    def __init__(self, input_dim, n_way):
        super().__init__()
        self.fc = nn.Linear(input_dim, n_way)
    def forward(self, x):
        return self.fc(x)

def fine_tune_and_predict(backbone, support_x, support_y_fam, query_x, query_y_fam, 
                          support_y_specie, query_y_specie, 
                          n_way, device, dataset=None, episode_idx=0, k_shot=1):
    """
    Returns: (accuracy, adaptation_time)
    """
    
    # --- [TIMING START] ADAPTATION TIME PER EPISODE ---
    if torch.cuda.is_available(): torch.cuda.synchronize()
    adapt_start = time.time()

    # 1. Feature Extraction (Frozen Backbone)
    backbone.eval()
    with torch.no_grad():
        dummy_z = backbone(support_x[0:1].to(device))
        input_dim = dummy_z.shape[1]
    
    # 2. Initialize Adapter
    head = AdapterHead(input_dim, n_way).to(device)
    head.train()
    optimizer = optim.Adam(head.parameters(), lr=FT_LR)
    criterion = nn.CrossEntropyLoss()
    
    unique_fam_labels = torch.unique(support_y_fam) 
    label_map = {old.item(): new for new, old in enumerate(unique_fam_labels)}
    
    local_support_y = torch.tensor([label_map[y.item()] for y in support_y_fam]).to(device)
    local_query_y   = torch.tensor([label_map[y.item()] for y in query_y_fam]).to(device)
    
    support_x = support_x.to(device)
    query_x = query_x.to(device)

    # 3. Fine-Tuning Loop
    for _ in range(FT_STEPS):
        optimizer.zero_grad()
        with torch.no_grad():
            augmented_support = SUPPORT_AUGMENTATIONS(support_x)
            z_support = backbone(augmented_support)
            
        logits = head(z_support)
        loss = criterion(logits, local_support_y)
        loss.backward()
        optimizer.step()
    
    # --- [TIMING END] ADAPTATION TIME ---
    if torch.cuda.is_available(): torch.cuda.synchronize()
    adapt_end = time.time()
    adaptation_time = adapt_end - adapt_start

    # 4. Prediction on Query Set
    head.eval()
    with torch.no_grad():
        z_query = backbone(query_x)
        logits_q = head(z_query)
        preds_local = torch.argmax(logits_q, dim=1) 
        
        acc = (preds_local == local_query_y).float().mean().item()
        
        if episode_idx == 0 and dataset is not None:
            save_path = f"{RESULTS_DIR}/{n_way}way_{k_shot}shot.png"
            save_oak_visualization(
                support_x=support_x.cpu(),
                support_species_y=support_y_specie.cpu(),
                query_x=query_x.cpu(),
                query_species_y=query_y_specie.cpu(),
                preds_local=preds_local.cpu(),
                dataset=dataset,
                save_path=save_path,
                n_way=n_way,
                k_shot=k_shot
            )

    return acc, adaptation_time

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
def main():
    print(f"--- Evaluating Baseline (Task: {TASK_NAME}) ---")
    set_all_seeds(SEED)
    
    # --- WANDB INIT ---
    wandb.login(key=WANDB_KEY)
    wandb.init(
        project=WANDB_PROJECT,
        name=f"Eval_{TASK_NAME}_Baseline",
        config={
            "model_path": MODEL_PATH,
            "ft_steps": FT_STEPS,
            "ft_lr": FT_LR,
            "n_episodes": N_EPISODES
        }
    )
    
    transform = transforms.Compose([transforms.Resize((84,84)), transforms.ToTensor()])
    dataset = PokemonMetaDataset(csv_file="Data/pokemon_data_linked.csv", root_dir="Data/pokemon_sprites", transform=transform)
    all_indices = list(dataset.indices_by_label.keys())
    
    backbone = ConvBackbone()
    try:
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        clean_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        backbone.load_state_dict(clean_dict, strict=False)
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model from {MODEL_PATH}: {e}")
        return
    backbone.to(DEVICE)
    
    print(f"\n{'N-Way':<6} | {'K-Shot':<6} | {'Result (Mean +/- Conf)':<22} | {'Avg Adapt Time':<15}")
    print("-" * 65)
    
    # Tabla para guardar resumen de resultados en wandb al final
    wandb_summary_table = wandb.Table(columns=["N-Way", "K-Shot", "Accuracy", "Confidence", "Avg_Adapt_Time"])

    for n_way in N_WAYS_LIST:
        for k_shot in K_SHOTS_LIST:
            try:
                sampler = Oak(dataset, all_indices, n_way=n_way, n_shot=k_shot, n_query=N_QUERY, n_episodes=N_EPISODES)
                loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, num_workers=2)
                
                accuracies = []
                adapt_times = [] # List to store time per episode
                
                for i, (images, labels_species) in enumerate(loader):
                    labels_family = torch.tensor([dataset.idx_to_family[l.item()] for l in labels_species])
                    
                    p = k_shot + N_QUERY
                    x = images.view(n_way, p, 3, 84, 84)
                    y_fam = labels_family.view(n_way, p)
                    y_spec = labels_species.view(n_way, p)
                    
                    support_x = x[:, :k_shot].contiguous().view(-1, 3, 84, 84)
                    query_x   = x[:, k_shot:].contiguous().view(-1, 3, 84, 84)
                    support_y_fam = y_fam[:, :k_shot].contiguous().view(-1)
                    query_y_fam   = y_fam[:, k_shot:].contiguous().view(-1)
                    support_y_spec = y_spec[:, :k_shot].contiguous().view(-1)
                    query_y_spec   = y_spec[:, k_shot:].contiguous().view(-1)
                    
                    # Run Episode (Returns Acc AND Time)
                    acc, t_adapt = fine_tune_and_predict(
                        backbone, support_x, support_y_fam, query_x, query_y_fam,
                        support_y_spec, query_y_spec,
                        n_way, DEVICE, dataset, episode_idx=i, k_shot=k_shot
                    )
                    accuracies.append(acc)
                    adapt_times.append(t_adapt)
                
                # Metrics Calculation
                mean_acc = np.mean(accuracies)
                conf = 1.96 * np.std(accuracies) / np.sqrt(N_EPISODES)
                mean_time = np.mean(adapt_times)
                
                print(f"{n_way:<6} | {k_shot:<6} | {mean_acc*100:.2f}% +- {conf*100:.2f}%   | {mean_time:.4f}s")
                
                # --- LOGGING TO WANDB ---
                # Opción A: Loguear como métrica contínua (útil para gráficas de líneas)
                wandb.log({
                    "n_way": n_way,
                    "k_shot": k_shot,
                    "accuracy": mean_acc,
                    "confidence_interval": conf,
                    "avg_adaptation_time": mean_time,
                    # Nombres específicos para facilitar la comparativa de gráficas
                    f"acc_{n_way}way_{k_shot}shot": mean_acc,
                    f"time_{n_way}way_{k_shot}shot": mean_time
                })

                # Añadir fila a la tabla resumen
                wandb_summary_table.add_data(n_way, k_shot, mean_acc, conf, mean_time)
                
            except ValueError:
                print(f"{n_way:<6} | {k_shot:<6} | SKIPPED")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"{n_way:<6} | {k_shot:<6} | ERROR: {e}")

    # Subir la tabla resumen al final del experimento
    wandb.log({"evaluation_summary": wandb_summary_table})
    wandb.finish()

if __name__ == '__main__':
    main()