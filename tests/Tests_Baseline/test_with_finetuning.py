import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import os

# --- LOCAL MODULES ---
from Utils.utils import set_all_seeds, visualize_batch 
from Dataloaders.dataloader import PokemonMetaDataset, get_structured_splits
from Dataloaders.sampler import Pokedex
from Models.Baseline import ConvBackbone
# from Utils.globals import * # Removed wildcard import for clarity

# ==========================================
# 1. CONFIGURATION
# ==========================================
SEED = 151
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Evaluation Mode
# Options: 'random', 'generation', 'type'
CURRENT_SPLIT_NAME = "generation" 

# Model Weights Path
MODEL_PATH = "Results/Models_pth/Baseline_pth/baseline_evolution_random_seed151.pth"

# Meta-Test Settings
N_WAYS_LIST = [2, 3, 4, 5]      # Classes per task
K_SHOTS_LIST = [1, 2, 3, 4, 5]  # Examples per class
N_QUERY = 1                     # Queries per class
N_EPISODES = 600                # Total tasks to evaluate

# Inner Loop (Fine-Tuning) Hyperparameters
FT_LR = 0.01        
FT_STEPS = 12        

# Data Augmentation for Support Set (Crucial for 1-shot/5-shot generalization)
SUPPORT_AUGMENTATIONS = nn.Sequential(
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.8, 1.2)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
)

# ==========================================
# 2. INNER LOOP MODULES
# ==========================================

class AdapterHead(nn.Module):
    """
    Temporary classification head initialized for each specific N-Way task.
    """
    def __init__(self, input_dim, n_way):
        super().__init__()
        self.fc = nn.Linear(input_dim, n_way)
        
    def forward(self, x):
        return self.fc(x)

def fine_tune_and_predict(backbone, support_x, support_y, query_x, query_y, n_way, device, 
                          dataset=None, episode_idx=0, k_shot=1):
    """
    Executes the Meta-Learning 'Inner Loop':
    1. Freeze Backbone.
    2. Train AdapterHead on Support Set (Fine-Tuning).
    3. Evaluate AdapterHead on Query Set.
    """
    

    # --- 1. Feature Extraction ---
    backbone.eval()
    with torch.no_grad():
        # Pass one dummy image to get the feature vector size automatically
        dummy_z = backbone(support_x[0:1].to(device))
        input_dim = dummy_z.shape[1]
    
    # Initialize task-specific head
    head = AdapterHead(input_dim, n_way).to(device)
    head.train()
    optimizer = optim.Adam(head.parameters(), lr=FT_LR)
    criterion = nn.CrossEntropyLoss()
    
    # --- 2. Label Mapping (Global -> Local) ---
    # support_y contains Global IDs (e.g., Pikachu=25, Mewtwo=150).
    # We need Local IDs (0, 1, 2...) for the CrossEntropyLoss of this specific task.
    unique_labels = torch.unique(support_y) 
    label_map = {old.item(): new for new, old in enumerate(unique_labels)}
    
    local_support_y = torch.tensor([label_map[y.item()] for y in support_y]).to(device)
    local_query_y   = torch.tensor([label_map[y.item()] for y in query_y]).to(device)
    
    support_x, query_x = support_x.to(device), query_x.to(device)

    # --- 3. Fine-Tuning Steps ---
    for _ in range(FT_STEPS):
        optimizer.zero_grad()
        with torch.no_grad():
            # Augment support images to avoid overfitting on small K-Shots
            augmented_support = SUPPORT_AUGMENTATIONS(support_x)
            z_support = backbone(augmented_support)
        
        logits = head(z_support)
        loss = criterion(logits, local_support_y)
        loss.backward()
        optimizer.step()
        
    # --- 4. Prediction ---
    head.eval()
    with torch.no_grad():
        z_query = backbone(query_x)
        logits_q = head(z_query)
        
        # Get Local Predictions (0..N-1)
        preds_local = torch.argmax(logits_q, dim=1) 
        
        # Calculate Accuracy
        acc = (preds_local == local_query_y).float().mean().item()
        
        # --- 5. Visualization (First Episode Only) ---
        if episode_idx == 0 and dataset is not None:
            # Map Local Predictions back to Global IDs for display
            preds_local_cpu = preds_local.cpu()
            preds_global = unique_labels[preds_local_cpu]   
            true_global = query_y.cpu()                     
            
            save_path = f"Results/Images/Images_Baseline/Baseline_{CURRENT_SPLIT_NAME}/{n_way}way_{k_shot}shot.png"
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            visualize_batch(
                images=query_x, 
                true_labels=true_global, 
                pred_labels=preds_global, 
                dataset=dataset, 
                save_path=save_path
            )

    return acc

# ==========================================
# 3. MAIN EVALUATION LOOP
# ==========================================
def main():
    print(f"--- Evaluating Baseline (Split: {CURRENT_SPLIT_NAME}) ---")
    set_all_seeds(SEED)
    
    # 1. Load Data
    transform = transforms.Compose([transforms.Resize((84,84)), transforms.ToTensor()])
    dataset = PokemonMetaDataset(csv_file="Data/pokemon_data_linked.csv", root_dir="Data/pokemon_sprites", transform=transform)
    
    # 2. Define Train/Test Splits
    if CURRENT_SPLIT_NAME == 'random':
        _, test_labels, _ = get_structured_splits(dataset, split_mode='random')
    elif CURRENT_SPLIT_NAME == 'generation':
        # Train on Gen 1 & 3, Test on Gen 4
        _, test_labels, _ = get_structured_splits(dataset, split_mode='generation', train_vals=['generation-i', 'generation-iii'], test_vals=['generation-iv'])
    elif CURRENT_SPLIT_NAME == 'type':
        # Train on Water, Test on rare types
        _, test_labels, _ = get_structured_splits(dataset, split_mode='type', train_vals=['water'], val_vals=['rock'], test_vals=['ghost', 'dragon', 'ice', 'steel', 'dark', 'flying', 'fairy'])
        
    print(f"Number of Test Classes: {len(test_labels)}")

    # 3. Load Backbone Model
    backbone = ConvBackbone()
    try:
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        # Handle 'Sequential' prefix if model was saved wrapped in one
        clean_dict = {k.replace('0.', ''): v for k, v in state_dict.items() if k.startswith('0.')}
        if not clean_dict: clean_dict = state_dict 
            
        backbone.load_state_dict(clean_dict, strict=False)
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    backbone.to(DEVICE)
    
    # 4. Meta-Testing
    print(f"\n{'N-Way':<10} | {'K-Shot':<10} | {'Result (Mean +/- Conf)':<25}")
    print("-" * 50)

    for n_way in N_WAYS_LIST:
        for k_shot in K_SHOTS_LIST:
            
            # Use 'Pokedex' Sampler for Standard Classification Tasks
            try:
                sampler = Pokedex(dataset, test_labels, n_way=n_way, n_shot=k_shot, n_query=N_QUERY, n_episodes=N_EPISODES)
                loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, num_workers=2)
                
                accuracies = []
                
                # Use tqdm for progress bar
                loader_tqdm = tqdm(loader, desc=f"{n_way}Way {k_shot}Shot", leave=False)
                
                for i, (images, labels) in enumerate(loader_tqdm):
                    images = images.to(DEVICE)
                    
                    # Reshape: [N_Way, K+Q, C, H, W]
                    p = k_shot + N_QUERY
                    x = images.view(n_way, p, 3, 84, 84)
                    y = labels.view(n_way, p)
                    
                    # Split Support vs Query
                    support_x = x[:, :k_shot].contiguous().view(-1, 3, 84, 84)
                    support_y = y[:, :k_shot].contiguous().view(-1)
                    
                    query_x   = x[:, k_shot:].contiguous().view(-1, 3, 84, 84)
                    query_y   = y[:, k_shot:].contiguous().view(-1)
                    
                    # Run Inner Loop
                    acc = fine_tune_and_predict(
                        backbone, support_x, support_y, query_x, query_y, n_way, DEVICE,
                        dataset=dataset,      
                        episode_idx=i,        
                        k_shot=k_shot          
                    )
                    accuracies.append(acc)
                
                # Statistics
                mean = np.mean(accuracies)
                conf = 1.96 * np.std(accuracies) / np.sqrt(N_EPISODES)
                
                # Clear tqdm line and print result
                loader_tqdm.close()
                print(f"{n_way:<10} | {k_shot:<10} | {mean*100:.2f}% +- {conf*100:.2f}%")

            except ValueError:
                 print(f"{n_way:<10} | {k_shot:<10} | SKIPPED (Insufficient data)")

if __name__ == '__main__':
    main()