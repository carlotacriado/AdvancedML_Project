import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from Utils.utils import seed_worker

# --- Wrapper del Dataset ---
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
        
        # Intentamos mapear la etiqueta global a la local (0..N-1)
        # Si la etiqueta global (ej. Gen 2) no está en el mapa (entrenado en Gen 1+3),
        # devolvemos -100.
        # NOTA: CrossEntropyLoss por defecto ignora el índice -100.
        target_label = self.label_map.get(global_label, -100)
        
        return image, target_label

# --- Función Principal Modificada ---
def get_baseline_dataloaders(train_dataset, val_dataset, train_labels, task_mode='pokedex', batch_size=64, seed=None, val_labels=None):     
    
    print(f"\n--- Configurando Baseline Dataloaders (Task: {task_mode.upper()}) ---")

    # 1. Configurar Generador
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)

    # 2. Crear Mapa de Etiquetas (Global -> Local 0..N)
    # SOLO usamos train_labels para definir la arquitectura de salida de la red.
    # El modelo solo tendrá neuronas para las clases de Train.
    active_species = sorted(list(set(train_labels)))
    label_map = {}
    
    if task_mode == 'pokedex':
        for i, species_id in enumerate(active_species):
            label_map[species_id] = i
        num_classes = len(active_species)
        print(f"Baseline Pokedex: {num_classes} clases activas en Train (Output Neurons).")

    elif task_mode == 'oak':
        family_to_int = {}
        current_fam_idx = 0
        for species_id in active_species:
            fam_str = train_dataset.idx_to_family[species_id]
            if fam_str not in family_to_int:
                family_to_int[fam_str] = current_fam_idx
                current_fam_idx += 1
            label_map[species_id] = family_to_int[fam_str]
        num_classes = current_fam_idx
        print(f"Baseline Oak: {num_classes} familias activas en Train.")
    
    else:
        raise ValueError("task_mode debe ser 'pokedex' u 'oak'")

    # 3. Recolectar índices de TRAIN
    # Usamos train_dataset (el que tiene Augmentation)
    train_indices = []
    for label in train_labels:
        if label in train_dataset.indices_by_label:
            train_indices.extend(train_dataset.indices_by_label[label])
    
    # 4. Recolectar índices de VALIDATION
    # Usamos val_dataset (el limpio) y val_labels (Gen 2, disjuntas)
    val_indices = []
    if val_labels is not None:
        for label in val_labels:
            if label in val_dataset.indices_by_label:
                val_indices.extend(val_dataset.indices_by_label[label])
    
    print(f"Imagenes Train (Augmented): {len(train_indices)} | Imagenes Val (Disjoint): {len(val_indices)}")

    # 5. Crear los Wrappers
    # Train: mapeara correctamente a 0..N
    train_ds = MappedSubset(train_dataset, train_indices, label_map)
    
    # Val: Como las labels de Gen 2 no estan en 'label_map', devolvera -100
    val_ds = MappedSubset(val_dataset, val_indices, label_map)

    # 6. Loaders
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