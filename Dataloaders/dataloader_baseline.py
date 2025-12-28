import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from Utils.utils import seed_worker

# --- DATASET WRAPPER ---
class MappedSubset(Dataset):
    """
    Wraps a subset of indices and maps Global IDs to Local Model IDs.
    
    Crucial Logic:
    - If a label exists in the 'label_map' (Training Class), it returns 0..N.
    - If a label is NOT in 'label_map' (Unseen/Validation Class), it returns -100.
    """
    def __init__(self, dataset, indices, label_map):
        self.dataset = dataset
        self.indices = indices
        self.label_map = label_map

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        image, global_label = self.dataset[real_idx]
        
        # Try to map Global ID -> Local ID
        # Default to -100 if the class was not in the Training set.
        # PyTorch CrossEntropyLoss ignores target -100.
        target_label = self.label_map.get(global_label, -100)
        
        return image, target_label

# --- MAIN DATALOADER FUNCTION ---
def get_baseline_dataloaders(train_dataset, val_dataset, train_labels, 
                             task_mode='pokedex', batch_size=64, seed=None, val_labels=None):      
    
    print(f"\n--- Configuring Baseline Dataloaders (Task: {task_mode.upper()}) ---")
    

    # 1. Configure Generator for Reproducibility
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)

    # 2. Create Label Map (Global ID -> Local ID 0..N)
    # IMPORTANT: We ONLY use train_labels to define the Classifier Output.
    # The model has neurons only for classes seen during training.
    active_species = sorted(list(set(train_labels)))
    label_map = {}
    
    if task_mode == 'pokedex':
        # Map Species ID directly
        for i, species_id in enumerate(active_species):
            label_map[species_id] = i
        num_classes = len(active_species)
        print(f"Baseline Pokedex: {num_classes} active classes in Train (Output Neurons).")

    elif task_mode == 'oak':
        # Map Species ID -> Family ID -> Local ID
        family_to_int = {}
        current_fam_idx = 0
        
        for species_id in active_species:
            # Look up family string in the dataset helper dict
            fam_str = train_dataset.idx_to_family[species_id]
            
            if fam_str not in family_to_int:
                family_to_int[fam_str] = current_fam_idx
                current_fam_idx += 1
            
            label_map[species_id] = family_to_int[fam_str]
            
        num_classes = current_fam_idx
        print(f"Baseline Oak: {num_classes} active families in Train.")
    
    else:
        raise ValueError("task_mode must be 'pokedex' or 'oak'")

    # 3. Collect Training Indices
    # Uses 'train_dataset' (which typically has Augmentation enabled)
    train_indices = []
    for label in train_labels:
        if label in train_dataset.indices_by_label:
            train_indices.extend(train_dataset.indices_by_label[label])
    
    # 4. Collect Validation Indices
    # Uses 'val_dataset' (Clean/No Augmentation)
    # 'val_labels' might be disjoint from 'train_labels' (e.g., Gen 2 vs Gen 1)
    val_indices = []
    if val_labels is not None:
        for label in val_labels:
            if label in val_dataset.indices_by_label:
                val_indices.extend(val_dataset.indices_by_label[label])
    
    print(f"Training Images (Augmented): {len(train_indices)} | Validation Images (Clean): {len(val_indices)}")

    # 5. Instantiate Wrappers
    # Train: Will map correctly to 0..N
    train_ds = MappedSubset(train_dataset, train_indices, label_map)
    
    # Val: If val_labels are distinct from train_labels, this will return -100 for targets
    val_ds = MappedSubset(val_dataset, val_indices, label_map)

    # 6. Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        worker_init_fn=seed_worker,
        generator=g
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        worker_init_fn=seed_worker,
        generator=g
    )

    return train_loader, val_loader, num_classes
