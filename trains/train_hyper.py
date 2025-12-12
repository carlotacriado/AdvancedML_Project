import torch
from Utils.utils import apply_support_aug

def split_batch(imgs, targets, n_way, k_shot, q_query, device):
    """
    Separa el batch que viene del Sampler (intercalado) en Support y Query limpios.
    Input structure: [C1_S, C1_Q, C2_S, C2_Q, ...]
    """
    imgs = imgs.to(device)
    # targets globales no nos importan para la loss, crearemos targets locales 0..N-1
    
    # Dimensiones
    img_per_class = k_shot + q_query
    c, h, w = imgs.size(1), imgs.size(2), imgs.size(3)
    
    # 1. Reshape para tener [N_Way, K+Q, C, H, W]
    imgs_reshaped = imgs.view(n_way, img_per_class, c, h, w)
    
    # 2. Separar Support y Query
    # Support: Tomamos los primeros K de cada clase
    support_imgs = imgs_reshaped[:, :k_shot].reshape(-1, c, h, w) # [N*K, C, H, W]
    support_imgs = apply_support_aug(support_imgs)
    
    # Query: Tomamos los últimos Q de cada clase
    query_imgs = imgs_reshaped[:, k_shot:].reshape(-1, c, h, w)   # [N*Q, C, H, W]
    
    # 3. Crear Labels Locales (0, 1, 2, 3, 4...)
    # Para support no hacen falta labels (se usan implícitamente por el orden), 
    # pero para query sí necesitamos saber la "Ground Truth".
    # Query Labels: [0, 0... (q veces), 1, 1... (q veces), ...]
    query_labels = torch.arange(n_way, device=device).repeat_interleave(q_query)
    
    return support_imgs, query_imgs, query_labels

def train_episode(model, batch, optimizer, criterion, n_way, k_shot, q_query, device):
    model.train()
    optimizer.zero_grad()
    
    images, _ = batch # Ignoramos labels globales
    
    # Separar datos
    support_x, query_x, query_y = split_batch(images, _, n_way, k_shot, q_query, device)
    
    # Forward Pass
    logits = model(support_x, query_x, n_way, k_shot, q_query)
    
    # Loss y Backprop
    loss = criterion(logits, query_y)
    loss.backward()
    optimizer.step()
    
    # Acc
    _, preds = logits.max(1)
    acc = preds.eq(query_y).float().mean().item() * 100
    
    return loss.item(), acc

def validate_episode(model, batch, criterion, n_way, k_shot, q_query, device):
    model.eval()
    with torch.no_grad():
        images, _ = batch
        support_x, query_x, query_y = split_batch(images, _, n_way, k_shot, q_query, device)
        
        logits = model(support_x, query_x, n_way, k_shot, q_query)
        loss = criterion(logits, query_y)
        
        _, preds = logits.max(1)
        acc = preds.eq(query_y).float().mean().item() * 100
        
    return loss.item(), acc