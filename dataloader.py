import os
import pandas as pd
from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import Sampler
from torch.utils.data import Dataset
import numpy as np


class EpisodicSampler(Sampler):
    def __init__(self, dataset, n_way, n_shot, n_query, n_episodes):
        """
        Args:
            dataset: Instance of PokemonMetaDataset
            n_way: Number of classes (Pokemon) per episode
            n_shot: Number of support images per class
            n_query: Number of query images per class
            n_episodes: Number of episodes (batches) to generate per epoch
        """
        self.indices_by_label = dataset.indices_by_label
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_episodes = n_episodes
        
        # Filter out classes that don't have enough images (shot + query)
        self.valid_labels = [
            label for label, indices in self.indices_by_label.items() 
            if len(indices) >= (n_shot + n_query)
        ]
        
        if len(self.valid_labels) < n_way:
            raise ValueError(f"Not enough classes with sufficient images. Need {n_way}, found {len(self.valid_labels)}")

    def __iter__(self):
        for _ in range(self.n_episodes):
            batch_indices = []
            
            # 1. Randomly select N classes (Pokemon)
            selected_classes = np.random.choice(self.valid_labels, self.n_way, replace=False)
            
            for cls in selected_classes:
                # 2. Within each class, select K + Q images
                # We use replace=False so support and query sets don't overlap
                cls_indices = self.indices_by_label[cls]
                selected_imgs = np.random.choice(
                    cls_indices, 
                    self.n_shot + self.n_query, 
                    replace=False
                )
                batch_indices.extend(selected_imgs)
            
            yield batch_indices

    def __len__(self):
        return self.n_episodes

class PokemonMetaDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.pokemon_info = pd.read_csv(csv_file)
        
        self.samples = [] # Flat list: [(path, label_idx), ...]
        self.indices_by_label = {} # NEW: { label_idx: [0, 1, 2, ...] }

        self.idx_to_name = self.pokemon_info['name'].to_dict() # This allows to map pokemon name to the output neuron
        self.idx_to_dex = self.pokemon_info['dex_number'].to_dict() # This allows to map pokemon dex number to the output neuron
        
        # --- Loading Logic ---
        for idx, row in self.pokemon_info.iterrows():
            dex_str = str(row['dex_number']).zfill(3)
            name_str = row['name'].lower()
            folder_name = f"{dex_str}-{name_str}" # Using dash based on your previous msg
            folder_path = os.path.join(root_dir, folder_name)
            
            # Initialize list for this label index
            if idx not in self.indices_by_label:
                self.indices_by_label[idx] = []

            if os.path.isdir(folder_path):
                images = list(Path(folder_path).glob('*.*'))
                for img_path in images:
                    if img_path.suffix.lower() in ['.jpg', '.png']:
                        # The current length of self.samples is the index of this new image
                        current_idx = len(self.samples)
                        
                        self.samples.append((str(img_path), idx))
                        
                        # Add this index to the class bucket
                        self.indices_by_label[idx].append(current_idx)

    def __len__(self):
        return len(self.samples)
    
    def _load_and_fix_transparency(self, path):
        """
        1. Fixes Alpha channel transparency.
        2. Detects solid black backgrounds and Flood Fills them to white.
        """
        image = Image.open(path)
        image = image.convert('RGBA') # Convert everything to RGBA first
        
        # --- Step 1: Handle Existing Transparency ---
        # Create a white canvas
        bg = Image.new('RGBA', image.size, (255, 255, 255, 255))
        # Paste the image using its own alpha channel as mask
        bg.paste(image, mask=image.split()[-1])
        
        # Convert to RGB (now we have a white background for transparent images)
        rgb_img = bg.convert('RGB')
        
        # --- Step 2: Handle "Fake" Black Backgrounds ---
        # Convert to NumPy for OpenCV processing
        np_img = np.array(rgb_img)
        
        # Check the top-left pixel (0,0). If it's effectively black, clean it.
        # We allow a small threshold (e.g., < 15) in case of JPEG artifacts
        top_left_pixel = np_img[0, 0]
        
        if np.all(top_left_pixel < 15): 
            # FLOOD FILL: simple and fast
            # cv2.floodFill(image, mask, seedPoint, newVal)
            # loDiff and upDiff allow for slight variations in the black color
            h, w = np_img.shape[:2]
            mask = np.zeros((h+2, w+2), np.uint8) # Mask must be 2 pixels larger
            
            # Fill starting from (0,0) with White (255,255,255)
            cv2.floodFill(
                np_img, 
                mask, 
                (0,0), 
                (255, 255, 255), 
                loDiff=(20, 20, 20), 
                upDiff=(20, 20, 20)
            )
            
            # Also check the bottom-right corner (sometimes sprites are offset)
            if np.all(np_img[-1, -1] < 15):
                cv2.floodFill(
                    np_img, 
                    mask, 
                    (w-1, h-1), 
                    (255, 255, 255), 
                    loDiff=(20, 20, 20), 
                    upDiff=(20, 20, 20)
                )

        # Convert back to PIL
        return Image.fromarray(np_img)

    def __getitem__(self, idx):
        img_path, label_idx = self.samples[idx]
        
        # USE THE NEW HELPER FUNCTION HERE
        image = self._load_and_fix_transparency(img_path)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label_idx
    
    def get_pokemon_details(self, label_idx):
        """Returns the Name and Dex Number for a given label index"""
        # Handle tensor inputs just in case
        if isinstance(label_idx, torch.Tensor):
            label_idx = label_idx.item()
            
        name = self.idx_to_name[label_idx]
        dex = self.idx_to_dex[label_idx]
        return name, dex