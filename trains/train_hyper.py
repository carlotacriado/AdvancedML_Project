import cv2 

def separar_support_query(batch_images, batch_labels, n_way, n_shot, n_query):
    """
    Toma el churro gigante del DataLoader y lo separa en Support y Query
    para la Hypernetwork.
    """
    # 1. Detectar el dispositivo (CPU o GPU)
    device = batch_images.device
    
    # 2. Re-organizar las dimensiones
    # El DataLoader te da: [n_way * (n_shot + n_query), C, H, W]
    # Lo convertimos a:    [n_way, n_shot + n_query, C, H, W]
    data = batch_images.view(n_way, n_shot + n_query, *batch_images.shape[1:])
    
    # 3. Cortar el pastel
    # Las primeras 'n_shot' son para estudiar (Support)
    support_images = data[:, :n_shot]  # Shape: [n_way, n_shot, C, H, W]
    
    # Las siguientes 'n_query' son para el examen (Query)
    query_images = data[:, n_shot:]    # Shape: [n_way, n_query, C, H, W]
    
    # 4. Preparar las etiquetas para el examen
    # En meta-learning, las etiquetas suelen ser relativas (0 a N-way-1)
    # Creamos un vector simple: [0, 0, 0... 1, 1, 1... 2, 2, 2...] para las queries
    query_labels = torch.arange(n_way).unsqueeze(1).repeat(1, n_query).view(-1)
    query_labels = query_labels.to(device)
    
    # Aplanamos las query images para pasarlas por el modelo una a una (o en batch)
    # Shape final query: [n_way * n_query, C, H, W]
    query_images_flat = query_images.contiguous().view(-1, *batch_images.shape[1:])
    
    return support_images, query_images_flat, query_labels