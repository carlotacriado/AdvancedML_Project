import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import gc
from torchvision import transforms
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# --- LOCAL MODULES ---
from Utils.utils import set_all_seeds, apply_support_aug
from Utils.globals import *
from Dataloaders.dataloader import PokemonMetaDataset, get_structured_splits, get_meta_dataloaders_pokedex, get_meta_dataloaders_oak
from Models.Baseline import ConvBackbone 
from Models.Hypernetwork import HyperNetworkModel 

# ==========================================
# 1. CONFIGURATION
# ==========================================
TASK = 'oak'
SPLIT_MODE = 'random' 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Grid Search 
N_WAY_LIST = [2, 3, 4, 5]
N_SHOT_LIST = [1, 2, 3, 4, 5]

# Configuración Test
N_QUERY = 1     
TEST_EPISODES = 100  

# Paths
BASE_PATH = "/fhome/amlai08/AdvancedML_Project"
CSV_PATH = os.path.join(BASE_PATH, "Data/pokemon_data_linked.csv")
IMGS_PATH = os.path.join(BASE_PATH, "Data/pokemon_sprites")
MODELS_DIR = os.path.join(BASE_PATH, "Results/Models_pth/Hypernetwork_pth/oak")
RESULTS_DIR = os.path.join(BASE_PATH, "Results/Hyper/oak")
os.makedirs(RESULTS_DIR, exist_ok=True)

# =================================================

# --- 1. VISUALIZATIONS & PLOTS ---

def denormalize(tensor):
    img = tensor.cpu().numpy().transpose((1, 2, 0))
    img = np.clip(img, 0, 1)
    return img

def save_grid_view(query_imgs, query_labels, preds_rel, dataset, save_path):
    """Saves a grid view (Green/Red) of the first batch"""
    idx_to_name = dataset.idx_to_name
    unique_classes_episode = torch.unique(query_labels) 

    num_show = min(len(query_labels), 20)
    cols = 5
    rows = (num_show + cols - 1) // cols
    
    plt.figure(figsize=(15, 3.5 * rows))
    for i in range(num_show):
        ax = plt.subplot(rows, cols, i + 1)
        img = denormalize(query_imgs[i])
        ax.imshow(img)
        ax.axis('off')
        
        true_global = query_labels[i].item()
        true_name = idx_to_name.get(true_global, "???")
        
        pred_r = preds_rel[i].item()
        
        # Mapping safely
        if pred_r < len(unique_classes_episode):
            pred_global = unique_classes_episode[pred_r].item()
            pred_name = idx_to_name.get(pred_global, "Err")
        else:
            pred_name = "Inv"
            pred_global = -1

        is_correct = (pred_global == true_global)
        color = 'green' if is_correct else 'red'
        symbol = '✔' if is_correct else '✘'
        
        ax.set_title(f"L: {true_name}\nP: {pred_name} {symbol}", color=color, fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_filtered_confusion_matrix(all_true, all_preds, dataset, save_path):
    """Heatmap with the Top 20 errors most frequent"""
    labels = list(set(all_true) | set(all_preds))
    cm = confusion_matrix(all_true, all_preds, labels=labels)
    
    # Anulate diagonal to only see mistakes
    np.fill_diagonal(cm, 0)
    errors_per_class = cm.sum(axis=1)
    
    # Top 20 classes with more mistakes
    n_top = min(len(labels), 20)
    top_indices = np.argsort(errors_per_class)[-n_top:] 
    top_labels = [labels[i] for i in top_indices]
    
    # Map names
    top_names = [dataset.idx_to_name.get(l, f"ID_{l}")[:10] for l in top_labels] # Cortar nombres largos
    
    cm_small = confusion_matrix(all_true, all_preds, labels=top_labels)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_small, annot=True, fmt='d', xticklabels=top_names, yticklabels=top_names, cmap='Reds')
    plt.title(f"Top {n_top} Confusiones (Excluyendo aciertos)")
    plt.ylabel('Verdadero')
    plt.xlabel('Predicho')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_accuracy_histogram(accuracies, save_path, mean_acc):
    """Shows if the model is stable or unstable"""
    plt.figure(figsize=(8, 6))
    plt.hist(accuracies, bins=10, range=(0, 1), color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(mean_acc, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_acc:.2f}')
    plt.title('Distribución de Accuracy por Episodio')
    plt.xlabel('Accuracy')
    plt.ylabel('Frecuencia (Num Episodios)')
    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    plt.savefig(save_path)
    plt.close()

# --- 2. MAIN ---

def main():
    print(f"--- 🚀 EVALUACIÓN DEFINITIVA ({TASK}) ---")
    set_all_seeds(SEED)
    
    # Cargar Datos
    print("📂 Cargando Dataset...")
    transform = transforms.Compose([transforms.Resize((84,84)), transforms.ToTensor()])
    dataset = PokemonMetaDataset(csv_file=CSV_PATH, root_dir=IMGS_PATH, transform=transform)
    
    if len(dataset) == 0:
        print("❌ ERROR CRÍTICO: El dataset está vacío. Revisa la ruta 'root_dir'.")
        print(f"Ruta intentada: {IMGS_PATH}")
        return
    
    if SPLIT_MODE == 'random':
        train_labels, test_labels, val_labels = get_structured_splits(dataset, split_mode='random')
    else:
        train_labels, test_labels, val_labels = get_structured_splits(dataset, split_mode='type', 
                                                    train_vals=['fairy', 'dark', 'dragon', 'rock', 'bug', 'psychic', 'flying', 'water', 'fire', 'grass'],
                                                    val_vals=['steel', 'ground', 'ghost'],
                                                    test_vals=['ice', 'poison', 'fighting', 'electric', 'normal']
                                                    )

    
    summary_results = []

    # Bucle Grid Search
    for N_WAY in N_WAY_LIST:
        for N_SHOT in N_SHOT_LIST:
            
            filename = f"Hypernet_{TASK}_{SPLIT_MODE}_{N_WAY}way_{N_SHOT}shot.pth"
            #filename = f"Hypernet_pokedex_{SPLIT_MODE}_{N_WAY}way_{N_SHOT}shot.pth"
            path = os.path.join(MODELS_DIR, filename)
            
            if not os.path.exists(path):
                continue 
            
            print(f"\n⚡ Evaluando: {filename}...")
            
            # Folder for this model
            current_config_dir = os.path.join(RESULTS_DIR, f"{N_WAY}way_{N_SHOT}shot")
            os.makedirs(current_config_dir, exist_ok=True)

            # Dataloader & Model
            loader_func = get_meta_dataloaders_pokedex if TASK == 'pokedex' else get_meta_dataloaders_oak
            _, test_loader, _ = loader_func(dataset, dataset, train_labels, test_labels, val_labels, 
                                            n_way=N_WAY, n_shot=N_SHOT, n_query=N_QUERY, episodes=TEST_EPISODES)
            
            backbone = ConvBackbone()
            hyper_model = HyperNetworkModel(backbone, feature_dim=3200, num_classes=N_WAY).to(DEVICE)
            
            try:
                hyper_model.load_state_dict(torch.load(path, map_location=DEVICE))
            except Exception as e:
                print(f"❌ Error cargando {filename}: {e}")
                continue
                
            hyper_model.eval()
            
            # --- LOOP EPISODIOS ---
            accuracies = []
            all_true = []
            all_pred = []
            
            # Tracking the difficulty by class {GlobalID: [Correct, Total]}
            class_performance = {} 

            with torch.no_grad():
                for batch_idx, (images, global_labels) in enumerate(tqdm(test_loader, desc=f"Episodios ({N_WAY}w{N_SHOT}s)")):
                    images, global_labels = images.to(DEVICE), global_labels.to(DEVICE)
                    
                    p = N_SHOT + N_QUERY
                    x = images.view(N_WAY, p, 3, 84, 84)
                    
                    x_support = x[:, :N_SHOT].contiguous().view(-1, 3, 84, 84)
                    x_support = apply_support_aug(x_support)
                    x_query   = x[:, N_SHOT:].contiguous().view(-1, 3, 84, 84)
                    
                    # IDs
                    support_globals = global_labels.view(N_WAY, p)[:, 0] 
                    query_globals_true = global_labels.view(N_WAY, p)[:, N_SHOT:].contiguous().view(-1)
                    
                    # Inference
                    logits = hyper_model(x_support, x_query, N_WAY, N_SHOT, N_QUERY)
                    preds_rel = torch.argmax(logits, dim=1)
                    
                    # Accuracy Episodio
                    targets_rel = torch.arange(N_WAY).repeat_interleave(N_QUERY).to(DEVICE)
                    acc = (preds_rel == targets_rel).float().mean().item()
                    accuracies.append(acc)
                    
                    # Globals for Matriz
                    preds_global = support_globals[preds_rel].cpu().numpy()
                    true_global = query_globals_true.cpu().numpy()
                    
                    all_true.extend(true_global)
                    all_pred.extend(preds_global)
                    
                    # Tracking Per-Class Performance
                    for t, p_val in zip(true_global, preds_global):
                        t_item = int(t)
                        if t_item not in class_performance:
                            class_performance[t_item] = [0, 0] # [Correct, Total]
                        class_performance[t_item][1] += 1
                        if t == p_val:
                            class_performance[t_item][0] += 1

                    # Guardar Grid (Solo Batch 0)
                    if batch_idx == 0:
                        save_grid_view(x_query, query_globals_true, preds_rel, dataset, 
                                       os.path.join(current_config_dir, "viz_grid.png"))

            # --- METRICS CALCULATION ---
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            min_acc = np.min(accuracies)
            max_acc = np.max(accuracies)
            ci95 = 1.96 * (std_acc / np.sqrt(len(accuracies)))
            
            # --- GENERATE REPORTS ---
            
            # 1. Confusion Matrix
            save_filtered_confusion_matrix(all_true, all_pred, dataset, 
                                           os.path.join(current_config_dir, "confusion_matrix.png"))
            
            # 2. Histogram
            save_accuracy_histogram(accuracies, 
                                    os.path.join(current_config_dir, "accuracy_histogram.png"), mean_acc)
            
            # 3. Hardest Classes CSV
            hardest_list = []
            for pid, stats in class_performance.items():
                p_name = dataset.idx_to_name.get(pid, f"ID_{pid}")
                acc_cls = stats[0] / stats[1] if stats[1] > 0 else 0
                hardest_list.append({"Pokemon": p_name, "Global_ID": pid, "Acc": acc_cls, "Samples": stats[1]})
            
            df_hard = pd.DataFrame(hardest_list)
            df_hard.sort_values(by="Acc", ascending=True, inplace=True) # The worst first
            df_hard.head(50).to_csv(os.path.join(current_config_dir, "hardest_classes.csv"), index=False)

            print(f"   -> Acc: {mean_acc*100:.2f}% (±{ci95*100:.2f}) | Worst: {min_acc*100:.2f}%")

            # 4. Summary Row
            summary_results.append({
                "N_Way": N_WAY,
                "N_Shot": N_SHOT,
                "Mean_Acc": mean_acc,
                "CI_95": ci95,
                "Worst_Episode": min_acc,
                "Best_Episode": max_acc,
                "Std_Dev": std_acc
            })

            # Clear RAM/VRAM
            del hyper_model
            del backbone
            torch.cuda.empty_cache()
            gc.collect()

    # --- FINAL COMPARATIVE CSV ---
    df_final = pd.DataFrame(summary_results)
    df_final.sort_values(by="Mean_Acc", ascending=False, inplace=True)
    
    final_path = os.path.join(RESULTS_DIR, "Ultimate_Benchmark.csv")
    df_final.to_csv(final_path, index=False)
    
    print("\n" + "="*50)
    print(f"✅ EVALUACIÓN COMPLETADA.")
    print(f"📊 Reporte General: {final_path}")
    print(f"📂 Detalles por modelo en: {RESULTS_DIR}/<config>")
    print("="*50)

if __name__ == '__main__':
    main()