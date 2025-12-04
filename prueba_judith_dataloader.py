import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# Importamos tu código maestro (asegúrate de que el archivo anterior se llame pokemon_meta_master.py)
from judith_dataloader import PokemonMetaDataset, get_train_val_test_splits, create_classification_task, SPLIT_CONFIG

# --- CONFIGURACIÓN DE RUTAS ---
CSV_FILE = "pokemon_data_gen1-5.csv"
IMGS_DIR = "pokemon_sprites"  # Tu carpeta de imágenes

# Configuración específica para forzar la prueba visual
# Vamos a pedir explícitamente ver a Pikachu en Test para comprobarlo
AUDIT_CONFIG = {
    'test_names': ['Pikachu', 'Charizard'], 
    'val_names': ['Snorlax'],
    'test_gens': ['generation-v'],
    'val_gens': [],
    'test_types': [],
    'val_types': [],
    'seed': 42
}

def denormalize(tensor):
    """Convierte el tensor normalizado de vuelta a imagen visible (0-1)"""
    # Deshacer: (image - 0.5) / 0.5  -> image * 0.5 + 0.5
    tensor = tensor * 0.5 + 0.5
    return tensor.permute(1, 2, 0).numpy() # De (C,H,W) a (H,W,C) para Matplotlib

def get_label_text(dataset, sample_idx):
    """Recupera texto legible para un índice de muestra"""
    meta = dataset.samples[sample_idx]
    
    # Invertir diccionarios para obtener texto
    idx_to_type = {v: k for k, v in dataset.type_to_idx.items()}
    idx_to_gen = {v: k for k, v in dataset.gen_to_idx.items()}
    
    name = dataset.idx_to_name[meta['label_idx']]
    type1 = idx_to_type[meta['type1_idx']]
    gen = idx_to_gen[meta['gen_idx']]
    
    return f"{name}\n({type1}, {gen})"

def plot_grid(dataset, indices, title, rows=2, cols=4):
    """Dibuja una cuadrícula de imágenes con sus metadatos"""
    fig, axes = plt.subplots(rows, cols, figsize=(12, 6))
    fig.suptitle(title, fontsize=16, color='darkblue')
    
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i < len(indices):
            idx = indices[i]
            img_tensor, _ = dataset[idx]
            
            # Mostrar imagen
            ax.imshow(denormalize(img_tensor))
            
            # Mostrar Metadatos (Nombre, Tipo, Gen)
            label_text = get_label_text(dataset, idx)
            ax.set_title(label_text, fontsize=9)
            
            # Mostrar Ruta (Para verificar lo de las carpetas)
            path_short = ".../" + os.path.basename(os.path.dirname(dataset.samples[idx]['path']))
            ax.set_xlabel(path_short, fontsize=7, color='gray')
            
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# ==========================================
# EJECUCIÓN DEL CHECK
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(CSV_FILE) or not os.path.isdir(IMGS_DIR):
        print("❌ ERROR: No encuentro el CSV o la carpeta de imágenes.")
    else:
        print("🔍 CARGANDO DATASET...")
        ds = PokemonMetaDataset(CSV_FILE, IMGS_DIR, transform=None) # Transform None para que podamos visualizar raw si quisiéramos, pero usamos el default
        
        # IMPORTANTE: Aplicar transform manualmente si no se pasó en init
        # (El código maestro ya tiene preprocess, así que usaremos ese)
        from judith_dataloader import preprocess
        ds.transform = preprocess

        print(f"✅ Dataset cargado: {len(ds)} imágenes totales.")

        # 1. VERIFICAR SPLITS
        print("\n🔍 GENERANDO SPLITS (Test: Pikachu, Charizard, Gen 5)...")
        train_fams, val_fams, test_fams = get_train_val_test_splits(ds, AUDIT_CONFIG)
        
        # Recolectar índices de muestra para visualizar
        def get_samples_from_fams(fams, count=8):
            indices = []
            if not fams: return []
            chosen_fams = np.random.choice(fams, min(len(fams), count), replace=False)
            for fam in chosen_fams:
                # Coger una foto aleatoria de esta familia
                img_idx = np.random.choice(ds.indices_by_family[fam])
                indices.append(img_idx)
            return indices

        print("\n📸 MOSTRANDO MUESTRAS DE CADA SPLIT...")
        
        # TRAIN
        train_idxs = get_samples_from_fams(train_fams)
        if train_idxs: plot_grid(ds, train_idxs, "Muestras de TRAINING (Gimnasio)")
        
        # VAL
        val_idxs = get_samples_from_fams(val_fams)
        if val_idxs: plot_grid(ds, val_idxs, "Muestras de VALIDATION (Snorlax debería estar aquí)")
        
        # TEST
        test_idxs = get_samples_from_fams(test_fams)
        if test_idxs: plot_grid(ds, test_idxs, "Muestras de TEST (Pikachu/Charizard/Gen5 deberían estar aquí)")

        # 2. VERIFICAR UNA TAREA GENERADA
        print("\n🔍 VERIFICANDO GENERACIÓN DE TAREA (Clasificación)...")
        # Creamos una tarea de TEST (debe tener a Pikachu o Gen 5)
        task = create_classification_task(ds, test_fams, n_way=3, n_shot=3, n_query=2)
        
        if task:
            print(f"   Tarea: {task.name}")
            print(f"   Clases: {task.class_names}")
            
            # Visualizar Support Set de la tarea
            # Extraemos los datos del loader para plotear
            sx, sy, smeta = next(iter(task.support_loader))
            
            # Plot de la tarea
            fig, axes = plt.subplots(1, 5, figsize=(15, 3))
            fig.suptitle(f"Support Set de Tarea: {task.class_names}", fontsize=14)
            for i in range(5): # Mostrar primeros 5 del batch
                if i < len(sx):
                    ax = axes[i]
                    ax.imshow(denormalize(sx[i]))
                    # El label 'sy' es 0, 1 o 2. Lo mapeamos al nombre real.
                    real_name = task.class_names[sy[i].item()]
                    # Metadata extraída del tensor
                    gen_idx = smeta['gen'][i].item()
                    # Invertir gen_idx a texto
                    gen_text = [k for k,v in ds.gen_to_idx.items() if v == gen_idx][0]
                    
                    ax.set_title(f"{real_name}\n({gen_text})")
                    ax.axis('off')
            plt.tight_layout()
            plt.show()
            
        print("\n✅ Auditoría finalizada. Revisa las imágenes generadas.")