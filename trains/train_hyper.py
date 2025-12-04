import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- TUS IMPORTACIONES ---
# Asumo que tienes estos archivos creados con tus clases
# from dataset import PokemonDataset
# from samplers import EpisodicSampler
# from models import MyHyperNetwork, TargetNetworkArchitecture

# --- CONFIGURACIÓN (HIPERPARÁMETROS) ---
N_WAY = 5          # 5 clases por tarea (ej: Pikachu, Bulbasaur...)
K_SHOT = 5         # 5 fotos para aprender (Support)
Q_QUERY = 15       # 15 fotos para examinar (Query)
EPISODES = 1000    # Número total de "tareas" a entrenar
VAL_FREQ = 50      # Cada cuánto validamos
LR = 0.001         # Learning Rate

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # 1. PREPARAR DATOS (POKEMON / EVOLUCIÓN)
    # ---------------------------------------
    # Aquí es donde decides si entrenas en especies o evoluciones cambiando los labels
    
    # Cargar todo el dataset
    # train_dataset = PokemonDataset(split='train') 
    # val_dataset = PokemonDataset(split='val')

    # Crear el Sampler (El "Director de Orquesta")
    # train_sampler = EpisodicSampler(labels=train_dataset.labels, n_way=N_WAY, k_shot=K_SHOT + Q_QUERY, episodes=EPISODES)
    # val_sampler = EpisodicSampler(labels=val_dataset.labels, n_way=N_WAY, k_shot=K_SHOT + Q_QUERY, episodes=100)

    # Crear el DataLoader (El "Camarero")
    # train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=2)
    # val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=2)

    # 2. INICIALIZAR MODELOS
    # ----------------------
    # La Hypernetwork toma imágenes y escupe PESOS
    # hypermodel = MyHyperNetwork(input_shape=(3, 84, 84)).to(DEVICE)
    
    # Optimizador (Solo entrenamos la Hypernetwork, la Target Network es un cascarón vacío)
    optimizer = optim.Adam(hypermodel.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # 3. BUCLE DE ENTRENAMIENTO
    # -------------------------
    print("Iniciando entrenamiento de Hypernetwork...")
    
    for episode, batch in enumerate(tqdm(train_loader, desc="Entrenando")):
        hypermodel.train()
        
        # A. Desempaquetar el Batch del Sampler
        # El sampler te devuelve (N*K+Q) imágenes. Hay que separarlas.
        images, labels = batch
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        # Dividir en Support (Estudio) y Query (Examen)
        # Nota: Esto depende de cómo tu Sampler ordene los datos.
        # Asumimos: las primeras K son support, las siguientes Q son query por cada clase.
        support_imgs, support_labels, query_imgs, query_labels = split_support_query(
            images, labels, N_WAY, K_SHOT, Q_QUERY
        )

        optimizer.zero_grad()

        # B. PASO 1: LA HYPERNETWORK GENERA LOS PESOS
        # Le damos el Support Set para que "estudie"
        generated_weights = hypermodel(support_imgs, support_labels)

        # C. PASO 2: INFERENCIA FUNCIONAL
        # Usamos esos pesos generados para clasificar el Query Set.
        # Aquí no usamos "model(x)", usamos una función que aplica los pesos manualmente.
        predictions = functional_forward(query_imgs, generated_weights)

        # D. CÁLCULO DE PÉRDIDA Y BACKPROP
        loss = criterion(predictions, query_labels)
        loss.backward()
        optimizer.step()

        # 4. VALIDACIÓN
        # -------------
        if (episode + 1) % VAL_FREQ == 0:
            val_acc = evaluate(hypermodel, val_loader)
            print(f"Episode {episode+1} | Loss: {loss.item():.4f} | Val Acc: {val_acc:.2f}%")

# --- FUNCIONES AUXILIARES (La Magia Técnica) ---

def split_support_query(images, labels, n_way, k_shot, q_query):
    """
    Separa el chorizo de imágenes que viene del DataLoader en Support y Query sets ordenados.
    Asume que el sampler envía: [Clase1_imgs..., Clase2_imgs...]
    """
    total_imgs_per_class = k_shot + q_query
    
    # Reshape para separar por clases: [N_WAY, TOTAL_IMGS, CHANNELS, H, W]
    imgs_reshaped = images.view(n_way, total_imgs_per_class, *images.shape[1:])
    lbls_reshaped = labels.view(n_way, total_imgs_per_class)

    # Cortar
    support_imgs = imgs_reshaped[:, :k_shot].reshape(-1, *images.shape[1:])
    query_imgs   = imgs_reshaped[:, k_shot:].reshape(-1, *images.shape[1:])
    
    # Re-hacer las labels (0..N_WAY-1) para que coincidan con la predicción local
    # Importante: En Few-Shot, la label 25 (Pikachu) pasa a ser label 0 de la tarea actual.
    support_labels = torch.arange(n_way).repeat_interleave(k_shot).to(DEVICE)
    query_labels   = torch.arange(n_way).repeat_interleave(q_query).to(DEVICE)
    
    return support_imgs, support_labels, query_imgs, query_labels

def functional_forward(input_images, weights):
    """
    Aquí simulas ser una red neuronal (Target Network) usando los pesos generados.
    Ejemplo para una red simple de 1 capa Lineal (ajusta según tu arquitectura).
    """
    # weights suele ser un diccionario o tupla de tensores generados
    # Ejemplo: weights['weight'] y weights['bias']
    
    # Aplanar imagen si es necesario
    x = input_images.view(input_images.size(0), -1) 
    
    # Aplicar capa lineal manualmente: y = xA^T + b
    # weights[0] = matriz de pesos, weights[1] = bias
    output = torch.nn.functional.linear(x, weights['weight'], weights['bias'])
    
    return output

def evaluate(model, loader):
    model.eval()
    total_acc = 0
    total_tasks = 0
    
    with torch.no_grad():
        for batch in loader:
            images, labels = batch
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            s_x, s_y, q_x, q_y = split_support_query(images, labels, N_WAY, K_SHOT, Q_QUERY)
            
            # Generar pesos y predecir
            weights = model(s_x, s_y)
            preds = functional_forward(q_x, weights)
            
            # Calcular Accuracy
            acc = (preds.argmax(dim=1) == q_y).float().mean()
            total_acc += acc.item()
            total_tasks += 1
            
    return (total_acc / total_tasks) * 100

if __name__ == "__main__":
    main()