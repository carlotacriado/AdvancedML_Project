#!/usr/bin/env python
# coding: utf-8

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from Utils.utils import seed_worker, set_all_seeds
from Utils.globals import *

class MappedSubset(Dataset):
    def __init__(self, dataset, indices, label_map):
        self.dataset = dataset
        self.indices = indices
        self.label_map = label_map

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        image, global_label = self.dataset[real_idx]
        target_label = self.label_map[global_label]
        return image, target_label

def get_baseline_dataloaders(dataset, train_labels, val_labels, task_mode='pokedex', batch_size=64, seed=SEED):
    
    print(f"\n--- Configurando Baseline Dataloaders (Task: {task_mode.upper()}) ---")

    # 1. Configurar Generador
    g = torch.Generator()
    g.manual_seed(seed)

    # 2. Crear Mapa de Etiquetas
    active_species = sorted(list(set(train_labels)))
    label_map = {}
    
    if task_mode == 'pokedex':
        for i, species_id in enumerate(active_species):
            label_map[species_id] = i
        num_classes = len(active_species)
        print(f"Baseline Pokedex: {num_classes} clases.")

    elif task_mode == 'oak':
        family_to_int = {}
        current_fam_idx = 0
        for species_id in active_species:
            fam_str = dataset.idx_to_family[species_id]
            if fam_str not in family_to_int:
                family_to_int[fam_str] = current_fam_idx
                current_fam_idx += 1
            label_map[species_id] = family_to_int[fam_str]
        num_classes = current_fam_idx
        print(f"Baseline Oak: {num_classes} clases.")
    else:
        raise ValueError("task_mode debe ser 'pokedex' u 'oak'")

    # 3. Recolectar indices
    all_train_indices = []
    for label in train_labels:
        if label in dataset.indices_by_label:
            all_train_indices.extend(dataset.indices_by_label[label])
    
    train_idx, val_idx = train_test_split(
        all_train_indices, 
        test_size=0.2, 
        random_state=SEED, 
        shuffle=True
    )

    # 4. Crear Datasets
    train_ds = MappedSubset(dataset, train_idx, label_map)
    val_ds = MappedSubset(dataset, val_idx, label_map)

    # 5. Crear Loaders (Usando el seed_worker importado)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True, 
        num_workers=2,
        worker_init_fn=seed_worker, # function for worker reproducibility
        generator=g
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        worker_init_fn=seed_worker, # function for worker reproducibility
        generator=g
    )

    return train_loader, val_loader, num_classes