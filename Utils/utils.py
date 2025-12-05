import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import os
from Utils.globals import *


### class prediction accuracy:
def top1_acc(pred: torch.FloatTensor, 
             y: torch.LongTensor):
    """calculates accuracy over a batch as a float
    given predicted logits 'pred' and integer targets 'y'"""
    return (pred.argmax(axis=1) == y).float().mean().item()  

def top5_acc(pred: torch.FloatTensor, 
             y: torch.LongTensor):
    """calculates top5 accuracy (whether any of the top 5 logits
    correspond to the true label), given predicted logits 'pred' 
    and integer targets 'y'"""
    top5_idxs = torch.sort(pred, dim=1, descending=True).indices[:,:5]
    correct = (top5_idxs == y.unsqueeze(1)).any(dim=1)
    avg_acc = correct.float().mean().item()
    return avg_acc    

# VISUALISATION

def visualize_episode(images, n_way, n_shot, n_query):
    """
    Visualizes a meta-learning episode as a grid.
    
    Args:
        images: Tensor of shape [Batch_Size, C, H, W]
        n_way: Number of classes (rows)
        n_shot: Support shots (left columns)
        n_query: Query shots (right columns)
    """
    # 1. Prepare dimensions
    images_per_class = n_shot + n_query
    _, c, h, w = images.shape
    
    # 2. Reshape: [Batch_Size, ...] -> [N_Way, K+Q, C, H, W]
    # We use .contiguous() to ensure memory is safe for .view()
    try:
        batch_grid = images.contiguous().view(n_way, images_per_class, c, h, w)
    except RuntimeError:
        print("Error: Batch size does not match n_way * (n_shot + n_query)")
        return

    rows_list = []

    # 3. Build the Grid
    for cls_idx in range(n_way):
        row_imgs = []
        for img_idx in range(images_per_class):
            
            # Extract single image tensor
            img = batch_grid[cls_idx, img_idx]
            
            # Convert: Tensor(C,H,W) -> Numpy(H,W,C)
            img = img.permute(1, 2, 0).cpu().numpy()
            
            # De-normalize if necessary (Assuming 0-1 range for now)
            # If your images look black, multiply by 255 here.
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
            
            # Convert RGB (PyTorch) to BGR (OpenCV)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Add a thin black border around every image for clarity
            img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[0,0,0])
            
            row_imgs.append(img)

        # 4. Create the Support/Query Separator (A Red Vertical Line)
        separator = np.zeros((h + 2, 5, 3), dtype=np.uint8) 
        separator[:] = (0, 0, 255) # BGR Red

        # 5. Concatenate the row: [Support Images] + [Red Line] + [Query Images]
        support_part = cv2.hconcat(row_imgs[:n_shot])
        query_part   = cv2.hconcat(row_imgs[n_shot:])
        
        full_row = cv2.hconcat([support_part, separator, query_part])
        rows_list.append(full_row)

    # 6. Stack all rows vertically
    final_grid = cv2.vconcat(rows_list)

    # 7. Display
    cv2.imshow(f"Episode: {n_way}-Way (Rows), {n_shot}-Shot (Left) | Query (Right)", final_grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def visualize_sample(dataset, idx):
    """
    Fetches a single sample from the dataset, converts it to OpenCV format,
    and displays it with metadata in the title.
    """
    # 1. Get the Tensor and Label Index
    # dataset[idx] triggers __getitem__, so transparency fix and transforms happen here
    img_tensor, label_idx = dataset[idx] 
    
    # 2. Retrieve Metadata using our new lookups
    name = dataset.idx_to_name[label_idx]
    dex = dataset.idx_to_dex[label_idx]
    
    # 3. Convert PyTorch Tensor (C, H, W) -> Numpy (H, W, C)
    # .permute(1, 2, 0) moves Channel from first to last
    img_np = img_tensor.permute(1, 2, 0).numpy()
    
    # 4. Scale and Cast
    # PyTorch Tensors are usually 0.0-1.0 floats. OpenCV needs 0-255 uint8.
    img_np = (img_np * 255).astype(np.uint8)
    
    # 5. Convert Color Space (RGB -> BGR for OpenCV)
    img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # 6. Optional: Resize (Zoom in) 
    # 84x84 is tiny on screen, let's make it 4x bigger (336x336)
    img_cv2 = cv2.resize(img_cv2, (0,0), fx=4, fy=4, interpolation=cv2.INTER_NEAREST)

    # 7. Create Window Title
    window_title = f"ID: {label_idx} | Dex: #{dex} | Name: {name}"
    
    print(f"Displaying -> {window_title}")
    
    cv2.imshow(window_title, img_cv2)
    cv2.waitKey(0) # Wait for any key press
    cv2.destroyAllWindows() 

# ===================================================
# Functions to add a seed (to ensure reproducibility)
# ===================================================
def set_all_seeds(seed=SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # numpy
    np.random.seed(seed)
    
    # PyTorch 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # Para multi-GPU
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"[Info] Semilla global fijada en: {seed}")

def seed_worker(worker_id):
    """
    Función para inicializar la seed de los workers de los Dataloaders.
    Esto asegura que la carga de datos en paralelo sea determinista.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
# =============================
# Function to save Loss curves
# =============================
import matplotlib.pyplot as plt

def save_plots(train_acc, val_acc, train_loss, val_loss, filename="training_curves.png"):
    epochs = range(1, len(train_acc) + 1)

    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, 'b-', label='Training Acc')
    plt.plot(epochs, val_acc, 'r-', label='Validation Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close() # Cierra la figura para liberar memoria
    print(f"[Info] Gráfico guardado en: {filename}")