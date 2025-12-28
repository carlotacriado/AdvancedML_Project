import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2
import random
import os
from Utils.globals import *

  
# --- VISUALISATION FUNCTIONS ---

def visualize_episode(images, n_way, n_shot, n_query, file_name):
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
        separator[:] = (0, 0, 0) # BGR Red

        # 5. Concatenate the row: [Support Images] + [Red Line] + [Query Images]
        support_part = cv2.hconcat(row_imgs[:n_shot])
        query_part   = cv2.hconcat(row_imgs[n_shot:])
        
        full_row = cv2.hconcat([support_part, separator, query_part])
        rows_list.append(full_row)

    # 6. Stack all rows vertically
    final_grid = cv2.vconcat(rows_list)

    # 7. Save
    cv2.imwrite(file_name, final_grid)  

def visualize_sample(dataset, idx):
    """
    Fetches a single sample from the dataset, converts it to OpenCV format,
    and displays it with metadata in the title.
    """
    # 1. Get the Tensor and Label Index
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

def denormalize(tensor):
    """
    Reverses the ImageNet normalization for visualization.
    Assumes standard mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
    Adjust if you used different values in EVAL_TRANSFORMS.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(tensor.device)
    return tensor * std + mean

def save_support_visualization(x_support, y_support, file_name="debug_support_aug.png"):
    """
    Saves a grid of augmented support images with their labels.
    """
    # 1. REMOVE DENORMALIZE
    # Since you only used ToTensor(), your images are already [0, 1]
    # denormalize() was corrupting them by applying ImageNet math to raw pixels.
    images = x_support.clone()
    
    # 2. Clip values (Good safety measure for ColorJitter)
    images = torch.clamp(images, 0, 1)
    
    # 3. Create a Grid
    grid_img = torchvision.utils.make_grid(images, nrow=5, padding=2)
    
    # 4. Save to disk
    os.makedirs("Results/Debug", exist_ok=True)
    save_path = os.path.join("Results/Debug", file_name)
    torchvision.utils.save_image(grid_img, save_path)
    
    print(f"Saved visualization debug grid to: {save_path}")


def visualize_batch(images, true_labels, pred_labels, dataset, save_path):
    """
    Visualize a batch of images with predicted vs real labels.
    Green --> Right, Red --> Wrong
    """
    # Undo tensor transform
    images_np = images.cpu().detach().numpy().transpose((0, 2, 3, 1))
    images_np = np.clip(images_np, 0, 1)
    
    # Configure grid
    batch_size = len(images)
    cols = 5
    rows = (batch_size + cols - 1) // cols
    
    fig = plt.figure(figsize=(15, 3.5 * rows))
    
    idx_to_name = dataset.idx_to_name
    
    for i in range(batch_size):
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(images_np[i])
        ax.axis('off')
        
        # Get names and IDs
        t_id = int(true_labels[i])
        p_id = int(pred_labels[i])
        
        true_name = idx_to_name.get(t_id, f"ID_{t_id}")
        pred_name = idx_to_name.get(p_id, f"ID_{p_id}")
        
        # Text and Colour
        is_correct = (t_id == p_id)
        color = 'green' if is_correct else 'red'
        title = f"T: {true_name}\nP: {pred_name}"
        
        ax.set_title(title, color=color, fontsize=10, fontweight='bold')
        
    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"--> Visualizacion guardada en: {save_path}")

# --- SET SEED FUNCTIONS --- (to ensure reproducibility)

def set_all_seeds(seed=SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # numpy
    np.random.seed(seed)
    
    # PyTorch 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
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
    

# --- DATA AUGMENTATION FUNCTIONS ---
    
def apply_support_aug(x_support):
    """
    Applies augmentation to each image in the support set individually.
    x_support shape: [N_support_total, 3, 84, 84]
    """
    # Create a list to hold the augmented single images
    augmented_imgs = []
    
    # Iterate through the batch dimension (dim 0)
    for i in range(x_support.size(0)):
        # Extract single image: shape [3, 84, 84]
        single_img = x_support[i]
        
        # Apply transform (Generates NEW random params for this specific image)
        aug_img = SUPPORT_AUGMENTATIONS(single_img)
        
        augmented_imgs.append(aug_img)
    
    # Stack them back into a batch: [N_support_total, 3, 84, 84]
    return torch.stack(augmented_imgs)
