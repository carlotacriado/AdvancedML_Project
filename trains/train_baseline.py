#!/usr/bin/env python
# coding: utf-8

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from Utils.utils import seed_worker
from Utils.globals import SEED

class MappedSubset(Dataset):
    """
    Wrapper that maps global dataset indices to a specific subset
    and remaps Global IDs (e.g., #25 Pikachu) to Local Model Class IDs (0..N).
    """
    def __init__(self, dataset, indices, label_map):
        self.dataset = dataset
        self.indices = indices
        self.label_map = label_map

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # 1. Get real index in the original dataset
        real_idx = self.indices[idx]
        
        # 2. Get image and Global ID
        image, global_label = self.dataset[real_idx]
        
        # 3. Map Global ID -> Local Class ID (0 to N-1) for CrossEntropy
        target_label = self.label_map[global_label]
        
        return image, target_label

def get_baseline_dataloaders(train_dataset, val_dataset, train_labels, val_labels=None, 
                             task_mode='pokedex', batch_size=64, seed=SEED):
    """
    Creates DataLoaders for standard supervised learning.
    
    Args:
        train_dataset: Dataset with training augmentations.
        val_dataset: Dataset with clean/eval transforms.
        train_labels: List of Global IDs to include in training.
        val_labels: List of Global IDs to include in validation.
        task_mode: 'pokedex' (Species Classification) or 'oak' (Evolutionary Family Classification).
    """
    
    print(f"\n--- Configuring Baseline Dataloaders (Task: {task_mode.upper()}) ---")
    

    # 1. Configure Generator for Reproducibility
    g = torch.Generator()
    g.manual_seed(seed)

    # 2. Create Label Map (Global ID -> Local ID 0..N)
    # We must ensure the map covers ALL classes involved (Train + Val).
    all_active_labels = sorted(list(set(train_labels)))
    if val_labels:
        all_active_labels = sorted(list(set(all_active_labels + list(val_labels))))
        
    label_map = {}
    
    if task_mode == 'pokedex':
        # Simple mapping: Species ID -> 0..N
        for i, species_id in enumerate(all_active_labels):
            label_map[species_id] = i
        num_classes = len(all_active_labels)
        print(f"Baseline Pokedex: {num_classes} active classes.")

    elif task_mode == 'oak':
        # Complex mapping: Species ID -> Family String -> Family ID (0..N)
        family_to_int = {}
        current_fam_idx = 0
        
        for species_id in all_active_labels:
            # We access the helper dict inside the dataset
            fam_str = train_dataset.idx_to_family[species_id]
            
            if fam_str not in family_to_int:
                family_to_int[fam_str] = current_fam_idx
                current_fam_idx += 1
            
            # Map the species ID to its Family ID
            label_map[species_id] = family_to_int[fam_str]
            
        num_classes = current_fam_idx
        print(f"Baseline Oak: {num_classes} evolutionary families.")
        
    else:
        raise ValueError("task_mode must be 'pokedex' or 'oak'")

    # 3. Collect Indices
    def get_indices_from_labels(dataset, labels):
        indices = []
        for label in labels:
            if label in dataset.indices_by_label:
                indices.extend(dataset.indices_by_label[label])
        return indices

    train_indices = get_indices_from_labels(train_dataset, train_labels)

    # 4. Handle Validation Split Strategy
    if val_labels is not None and len(val_labels) > 0:
        # STRATEGY A: Explicit Split (Generation/Type)
        # We use the separate indices and the separate datasets (augmented vs clean)
        print("Using Explicit Validation Split (defined by Generation or Type).")
        val_indices = get_indices_from_labels(val_dataset, val_labels)
        
        train_ds = MappedSubset(train_dataset, train_indices, label_map)
        val_ds   = MappedSubset(val_dataset, val_indices, label_map)
        
    else:
        # STRATEGY B: Random Internal Split (80/20)
        # Fallback if no specific validation labels were provided
        print("Using Random 80/20 Split on Training Data.")
        t_idx, v_idx = train_test_split(
            train_indices, 
            test_size=0.2, 
            random_state=seed, 
            shuffle=True
        )
        train_ds = MappedSubset(train_dataset, t_idx, label_map)
        # Note: We use val_dataset here to ensure validation images are not augmented
        val_ds   = MappedSubset(val_dataset, v_idx, label_map)

    # 5. Create DataLoaders
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