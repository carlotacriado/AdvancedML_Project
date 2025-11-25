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

class PokemonSpritesDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (str): Path to the csv file with metadata.
            root_dir (str): Parent folder (e.g., "pokemon_sprites").
            transform (callable, optional): Transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # 1. Load the CSV
        self.pokemon_info = pd.read_csv(csv_file)
        
        # 2. FLATTEN THE DATA
        self.samples = [] 
        
        for idx, row in self.pokemon_info.iterrows():
            dex_num_str = str(row['dex_number']).zfill(3)
            name = row['name'].lower()
            folder_name = dex_num_str + "-" + name
            folder_path = os.path.join(root_dir, folder_name)
            print(folder_path) # debug
            
            # Check if folder exists to avoid crashing
            if os.path.isdir(folder_path):
                images = list(Path(folder_path).glob('*.*')) 
                
                for img_path in images:
                    # We store the specific image path AND the index of the metadata
                    self.samples.append((str(img_path), idx))
    
    def __len__(self):
        # Returns the total number of IMAGES
        return len(self.samples)

    def __getitem__(self, idx):
        # 1. Retrieve the image path and the metadata index
        img_path, metadata_idx = self.samples[idx]
        
        # 2. Load Image
        image = Image.open(img_path).convert('RGB')
        
        # 3. Retrieve Metadata (Optional: choose what you need)
        meta_row = self.pokemon_info.iloc[metadata_idx]
        
        # Example: Let's extract the Type 1 and Type 2 as labels
        pokemon = meta_row['name']
        type1 = meta_row['type_1']
        dex_number = meta_row['dex_number']
        gen = meta_row['generation']
        
        # 4. Apply Transforms
        if self.transform:
            image = self.transform(image)
            
        # 5. Return whatever your training loop needs
        return image, pokemon, dex_number, type1, gen
    

class PokemonMetaDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.pokemon_info = pd.read_csv(csv_file)
        
        self.samples = [] # Flat list: [(path, label_idx), ...]
        self.indices_by_label = {} # NEW: { label_idx: [0, 1, 2, ...] }
        
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

    def __getitem__(self, idx):
        img_path, label_idx = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        # Return just image and label_idx for simplicity in training
        return image, label_idx   