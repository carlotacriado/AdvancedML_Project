import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg') # Non-interactive backend for server/headless execution
import matplotlib.pyplot as plt

# --- LOCAL MODULES ---
# Ensure these files exist in your project structure
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
# Update this path to where your pre-trained weights are stored
MODEL_PATH = "Results/Models_pth/Baseline_pth/baseline_evolution_random_seed151.pth"
RESULTS_DIR = f"Results/Visuals/Baseline_{TASK_NAME}"

# Meta-Testing Parameters
N_WAYS_LIST = [2, 3, 4, 5]      # Number of classes per episode
K_SHOTS_LIST = [1, 2, 3, 4, 5]  # Number of support examples per class
N_QUERY = 1                     # Number of query examples to test on
N_EPISODES = 600                # Total episodes for statistical significance

# Fine-Tuning (Inner Loop) Hyperparameters
FT_LR = 0.01        
FT_STEPS = 12        

# Data Augmentation for Support Set (improves generalization during fine-tuning)
SUPPORT_AUGMENTATIONS = nn.Sequential(
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.8, 1.2)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
)

# ==========================================
# 2. VISUALIZATION UTILITIES
# ==========================================

def denormalize_robust(tensor):
    """Converts a normalized tensor back to a numpy image (0-1 range)."""
    img = tensor.cpu().detach().numpy().squeeze()
    if img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose((1, 2, 0))
    return np.clip(img, 0, 1)

def save_oak_visualization(support_x, support_species_y, query_x, query_species_y, 
                           preds_local, dataset, save_path, n_way, k_shot):
    """
    Generates a grid visualization of the Few-Shot task.
    Rows = Classes (Families), Columns = Support Shots + Query.
    """
    

    rows = n_way
    cols = k_shot + N_QUERY
    
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.5 * rows))
    
    # Handle dimensions for 1-way or 1-shot edge cases
    if rows == 1: axes = np.expand_dims(axes, 0)
    if cols == 1: axes = np.expand_dims(axes, 1)
    
    class_names_map = {} 

    # --- Plot Support Set (Left Columns) ---
    for r in range(rows): 
        for k in range(k_shot): 
            idx = r * k_shot + k
            img = support_x[idx]
            specie_name = dataset.idx_to_name.get(support_species_y[idx].item(), "Unknown")
            
            # Save the first support name as the representative for this Class ID
            if k == 0: class_names_map[r] = specie_name
            
            ax = axes[r, k]
            try: ax.imshow(denormalize_robust(img))
            except: pass
            
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values(): spine.set_visible(False)
            
            # Labeling
            if k == 0:
                ax.set_ylabel(f"Class {r}", fontsize=12, fontweight='bold')
                ax.set_title(f"Sup: {specie_name}", fontsize=9, color='blue')
            else:
                ax.set_title(f"Sup: {specie_name}", fontsize=9, color='gray')

    # --- Plot Query Set (Right Column) ---
    for r in range(rows):
        q_idx = r # Assumes N_QUERY = 1, ordered by class
        img = query_x[q_idx]
        real_name = dataset.idx_to_name.get(query_species_y[q_idx].item(), "Unknown")
        pred_idx = preds_local[q_idx].item()
        
        is_correct = (pred_idx == r)
        color = 'green' if is_correct else 'red'
        
        ax = axes[r, k_shot]
        try: ax.imshow(denormalize_robust(img))
        except: pass
        
        ax.set_xticks([]); ax.set_yticks([])
        
        # Highlight prediction result with border
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
    """Simple Linear Layer to adapt the backbone features to the specific N-Way task."""
    def __init__(self, input_dim, n_way):
        super().__init__()
        self.fc = nn.Linear(input_dim, n_way)
        
    def forward(self, x):
        return self.fc(x)

def fine_tune_and_predict(backbone, support_x, support_y_fam, query_x, query_y_fam, 
                          support_y_specie, query_y_specie, 
                          n_way, device, dataset=None, episode_idx=0, k_shot=1):
    """
    Performs the 'Inner Loop':
    1. Extracts features using the frozen Backbone.
    2. Trains a temporary AdapterHead on the Support Set.
    3. Evaluates on the Query Set.
    """
    
    # 1. Feature Extraction (Frozen Backbone)
    backbone.eval()
    with torch.no_grad():
        # Get feature dimension dynamically
        dummy_z = backbone(support_x[0:1].to(device))
        input_dim = dummy_z.shape[1]
    
    # 2. Initialize Adapter
    head = AdapterHead(input_dim, n_way).to(device)
    head.train()
    optimizer = optim.Adam(head.parameters(), lr=FT_LR)
    criterion = nn.CrossEntropyLoss()
    
    # Remap Global Family IDs to Local Labels (0 to N-Way-1)
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
            # Augment support images to make the head more robust
            augmented_support = SUPPORT_AUGMENTATIONS(support_x)
            z_support = backbone(augmented_support)
            
        logits = head(z_support)
        loss = criterion(logits, local_support_y)
        loss.backward()
        optimizer.step()
        
    # 4. Prediction on Query Set
    head.eval()
    with torch.no_grad():
        z_query = backbone(query_x)
        logits_q = head(z_query)
        preds_local = torch.argmax(logits_q, dim=1) 
        
        acc = (preds_local == local_query_y).float().mean().item()
        
        # Visualize first episode
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

    return acc

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
def main():
    print(f"--- Evaluating Baseline (Task: {TASK_NAME}) ---")
    

    set_all_seeds(SEED)
    
    # Load Dataset
    transform = transforms.Compose([transforms.Resize((84,84)), transforms.ToTensor()])
    dataset = PokemonMetaDataset(csv_file="Data/pokemon_data_linked.csv", root_dir="Data/pokemon_sprites", transform=transform)
    all_indices = list(dataset.indices_by_label.keys())
    
    # Load Backbone
    backbone = ConvBackbone()
    try:
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        # Handle `DataParallel` prefix if present
        clean_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        backbone.load_state_dict(clean_dict, strict=False)
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model from {MODEL_PATH}: {e}")
        return
    backbone.to(DEVICE)
    
    # --- Meta-Testing Loop ---
    print(f"\n{'N-Way':<10} | {'K-Shot':<10} | {'Result (Mean +/- Conf)':<25}")
    print("-" * 50)
    
    for n_way in N_WAYS_LIST:
        for k_shot in K_SHOTS_LIST:
            try:
                # Initialize "Oak" Sampler (Episodes based on Evolutionary Families)
                sampler = Oak(dataset, all_indices, n_way=n_way, n_shot=k_shot, n_query=N_QUERY, n_episodes=N_EPISODES)
                loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, num_workers=2)
                
                accuracies = []
                
                for i, (images, labels_species) in enumerate(loader):
                    # Convert Species ID (e.g., 25 Pikachu) -> Family ID (e.g., 10 Pichu-Line)
                    labels_family = torch.tensor([dataset.idx_to_family[l.item()] for l in labels_species])
                    
                    # Reshape for Few-Shot format: (N_Way, K+Q, C, H, W)
                    p = k_shot + N_QUERY
                    x = images.view(n_way, p, 3, 84, 84)
                    y_fam = labels_family.view(n_way, p)
                    y_spec = labels_species.view(n_way, p)
                    
                    # Split Support / Query
                    support_x = x[:, :k_shot].contiguous().view(-1, 3, 84, 84)
                    query_x   = x[:, k_shot:].contiguous().view(-1, 3, 84, 84)
                    
                    support_y_fam = y_fam[:, :k_shot].contiguous().view(-1)
                    query_y_fam   = y_fam[:, k_shot:].contiguous().view(-1)
                    
                    support_y_spec = y_spec[:, :k_shot].contiguous().view(-1)
                    query_y_spec   = y_spec[:, k_shot:].contiguous().view(-1)
                    
                    # Run Episode
                    acc = fine_tune_and_predict(
                        backbone, support_x, support_y_fam, query_x, query_y_fam,
                        support_y_spec, query_y_spec,
                        n_way, DEVICE, dataset, episode_idx=i, k_shot=k_shot
                    )
                    accuracies.append(acc)
                
                # Statistics
                mean = np.mean(accuracies)
                conf = 1.96 * np.std(accuracies) / np.sqrt(N_EPISODES)
                print(f"{n_way:<10} | {k_shot:<10} | {mean*100:.2f}% +- {conf*100:.2f}%")
                
            except ValueError:
                print(f"{n_way:<10} | {k_shot:<10} | SKIPPED (Not enough classes/images for this setting)")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"{n_way:<10} | {k_shot:<10} | ERROR: {e}")

if __name__ == '__main__':
    main()