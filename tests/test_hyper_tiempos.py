import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import gc
import time
import wandb
from tqdm import tqdm
from torchvision import transforms

# --- LOCAL MODULES ---
from Utils.utils import set_all_seeds, apply_support_aug
from Utils.globals import *
from Dataloaders.dataloader import PokemonMetaDataset, get_structured_splits, get_meta_dataloaders_oak
from Models.Baseline import ConvBackbone 
from Models.Hypernetwork import HyperNetworkModel 
from trains.train_hyper import split_batch # Necesitamos esto para preparar los datos antes de medir

# ==========================================
# 1. CONFIGURATION
# ==========================================
TASK = 'oak'
SPLIT_MODE = 'random' 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# WandB Config
WANDB_PROJECT = "Hypernetwork_Time_Test"
WANDB_KEY = "93d025aa0577b011c6d4081b9d4dc7daeb60ee6b"

# Configuración Test (Igual que Baseline para comparar)
N_WAY_LIST = [5]
N_SHOT_LIST = [5]
N_QUERY = 1     
TEST_EPISODES = 100 

# Paths
BASE_PATH = os.getcwd()
CSV_PATH = os.path.join(BASE_PATH, "Data/pokemon_data_linked.csv")
IMGS_PATH = os.path.join(BASE_PATH, "Data/pokemon_sprites")
MODEL_DIR = os.path.join(BASE_PATH, "Results/Models_pth/Hypernet_pth/")

def main():
    set_all_seeds(151)
    
    # Inicializar WandB globalmente o por run
    wandb.login(key=WANDB_KEY)
    wandb.init(project=WANDB_PROJECT, name="Evaluation_Comparison_Summary")
    
    # Tabla resumen
    wandb_summary_table = wandb.Table(columns=["N-Way", "K-Shot", "Accuracy", "Confidence", "Avg_Adapt_Time"])

    # Dataset Setup
    transform = transforms.Compose([transforms.Resize((84,84)), transforms.ToTensor()])
    dataset = PokemonMetaDataset(csv_file=CSV_PATH, root_dir=IMGS_PATH, transform=transform)
    
    # Splits (para obtener los datos de test correctos)
    if SPLIT_MODE == 'random':
        _, _, test_lbls = get_structured_splits(dataset, split_mode='random')
    else:
        # Simplificación: Ajustar según tu lógica de split
        _, _, test_lbls = get_structured_splits(dataset, split_mode='random')

    print(f"\n{'N-Way':<6} | {'K-Shot':<6} | {'Result (Mean +/- Conf)':<22} | {'Avg Adapt Time':<15}")
    print("-" * 65)

    for N_WAY in N_WAY_LIST:
        for N_SHOT in N_SHOT_LIST:
            
            # Cargar Modelo Específico (Asumimos que entrenaste uno para cada config o uno genérico)
            # Si usas un modelo único para todo, cambia esta línea para cargar siempre el mismo
            # Aquí asumo que quieres probar el modelo entrenado para 5-way 5-shot (el más robusto)
            # o cargar el específico si hiciste grid search en training.
            # Para comparar justamente, a veces se usa el modelo entrenado en 5w-5s para testear en 2w-1s.
            # Ajusta el nombre del archivo según lo que tengas en la carpeta.
            
            model_name = f"Hypernet_{TASK}_{SPLIT_MODE}_5way_5shot.pth" # Usamos el "mejor" modelo para evaluar todo
            model_path = os.path.join(MODEL_DIR, model_name)
            
            if not os.path.exists(model_path):
                print(f"Model {model_name} not found, skipping...")
                continue

            # Load Loader
            test_loader = get_meta_dataloaders_oak(dataset, test_lbls, N_WAY, N_SHOT, N_QUERY, TEST_EPISODES)
            
            # Init Model
            backbone = ConvBackbone().to(DEVICE)
            # Nota: embedding_size debe coincidir con el del training
            hyper_model = HyperNetworkModel(backbone, embedding_size=128*5*5, n_way=N_WAY, n_query=N_QUERY).to(DEVICE)
            
            try:
                state_dict = torch.load(model_path, map_location=DEVICE)
                hyper_model.load_state_dict(state_dict)
                hyper_model.eval()
            except Exception as e:
                print(f"Error loading {model_name}: {e}")
                continue

            accuracies = []
            adapt_times = []

            with torch.no_grad():
                for batch in test_loader:
                    images, _ = batch # Ignoramos etiquetas globales
                    
                    # Preparación de datos (fuera del tiempo de inferencia de la red)
                    support_x, query_x, query_y = split_batch(images, None, N_WAY, N_SHOT, N_QUERY, DEVICE)
                    
                    # --- [TIMING START] ADAPTATION TIME ---
                    # Aquí medimos SOLO el tiempo que la red tarda en adaptarse (generar pesos) y predecir
                    if torch.cuda.is_available(): torch.cuda.synchronize()
                    start_t = time.time()
                    
                    # El forward de la hypernet incluye: 
                    # 1. Extracción características soporte
                    # 2. Generación de pesos (Adaptación)
                    # 3. Predicción query
                    logits = hyper_model(support_x, query_x, N_WAY, N_SHOT, N_QUERY)
                    
                    if torch.cuda.is_available(): torch.cuda.synchronize()
                    end_t = time.time()
                    # --- [TIMING END] ---
                    
                    adapt_times.append(end_t - start_t)
                    
                    # Calc Acc
                    _, preds = logits.max(1)
                    acc = preds.eq(query_y).float().mean().item()
                    accuracies.append(acc)

            # Metrics
            mean_acc = np.mean(accuracies)
            conf = 1.96 * np.std(accuracies) / np.sqrt(TEST_EPISODES)
            mean_time = np.mean(adapt_times)

            print(f"{N_WAY:<6} | {N_SHOT:<6} | {mean_acc*100:.2f}% +- {conf*100:.2f}%   | {mean_time:.6f}s")
            
            # Log to WandB
            wandb.log({
                f"acc_{N_WAY}way_{N_SHOT}shot": mean_acc,
                f"time_{N_WAY}way_{N_SHOT}shot": mean_time,
                "n_way": N_WAY,
                "k_shot": N_SHOT,
                "current_accuracy": mean_acc,
                "current_time": mean_time
            })
            
            wandb_summary_table.add_data(N_WAY, N_SHOT, mean_acc, conf, mean_time)

            # Cleanup
            del hyper_model, backbone
            torch.cuda.empty_cache()

    wandb.log({"evaluation_summary": wandb_summary_table})
    wandb.finish()

if __name__ == '__main__':
    main()