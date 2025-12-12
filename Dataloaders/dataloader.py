import os
import pandas as pd
from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import Sampler, Dataset, DataLoader
import numpy as np
import cv2
import random
from Utils.globals import *
from Dataloaders.sampler import Pokedex, Oak

class PokemonMetaDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.pokemon_info = pd.read_csv(csv_file)
        
        self.samples = [] # Flat list: [(path, label_idx), ...]
        self.indices_by_label = {} # NEW: { label_idx: [0, 1, 2, ...] }

        self.idx_to_name = self.pokemon_info['name'].to_dict() # This allows to map pokemon name to the output neuron
        self.idx_to_dex = self.pokemon_info['dex_number'].to_dict() # This allows to map pokemon dex number to the output neuron
        self.idx_to_type1 = self.pokemon_info['type_1'].to_dict() # This allows to map pokemon Type 1 to the output neuron
        self.idx_to_type2 = self.pokemon_info['type_2'].to_dict() # This allows to map pokemon Type 2 to the output neuron
        self.idx_to_gen = self.pokemon_info['generation'].to_dict() # This allows to map pokemon generation to the output neuron
        self.idx_to_pre_evo = self.pokemon_info['pre_evolution'].to_dict() # This allows to map pokemon pre-evolution to the output neuron
        self.idx_to_evo = self.pokemon_info['evolution'].to_dict() # This allows to map pokemon pre-evolution to the output neuron
        self.idx_to_family = self.pokemon_info['family_id'].to_dict() # This allows to map pokemon family ID to the output neuron

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
        type1 = self.idx_to_type1[label_idx]
        type2 = self.idx_to_type2[label_idx]
        gen = self.idx_to_gen[label_idx]
        pre_evo = self.idx_to_pre_evo[label_idx]
        evo = self.idx_to_evo[label_idx]
        family_idx = self.idx_to_family[label_idx]

        return name, dex, type1, type2, gen, pre_evo, evo, family_idx
    
def get_structured_splits(dataset, split_mode='generation', train_vals=None, val_vals=None, test_vals=None):
    """
    Returns two lists of label indices (train_labels, test_labels) based on metadata.
    
    Args:
        dataset: Your PokemonMetaDataset
        split_mode: 'generation', 'type', or 'stage'
        train_vals: List of values to keep in train (e.g., [1, 2, 3] for gens)
        val_vals: List of values to keep in validation (e.g., [3] for gens)
        test_vals: List of values to keep in test (e.g., [4, 5])
    """
    all_labels = list(dataset.indices_by_label.keys())
    train_labels = []
    test_labels = []
    val_labels = []
    
    print(f"--- Splitting by {split_mode.upper()} ---")
    
    # NORMAL RANDOM SPLIT
    if split_mode == 'random':
      random.shuffle(all_labels)
      
      n_total = len(all_labels)
      
      n_train = int(n_total * 0.6)
      n_val = int(n_total * 0.2)
      
      train_labels = all_labels[:n_train]
      val_labels = all_labels[n_train: n_train + n_val]
      test_labels = all_labels[n_train + n_val:]
      
    else:    
        for label_idx in all_labels:
            # 1. Fetch the specific metadata for this label
            # (We use the lookups you created in __init__)
            if split_mode == 'generation':
                val = dataset.idx_to_gen[label_idx]
            elif split_mode == 'type':
                val = dataset.idx_to_type1[label_idx] # Primary type
            elif split_mode == 'stage':
                # 1: Basic, 2: Stage 1, 3: Stage 2
                # You might need to clean your CSV logic if it uses strings like "Basic"
                val = dataset.pokemon_info.iloc[label_idx]['evolution_stage'] 
            else:
                raise ValueError("Unknown split mode")
    
            # 2. Sort into buckets
            if train_vals and val in train_vals:
                train_labels.append(label_idx)
            elif (val_vals is not None) and (val in val_vals):
                val_labels.append(label_idx)
            elif test_vals and val in test_vals:
                test_labels.append(label_idx)
      
        if val_vals is None:
            random.shuffle(train_labels)

            num_total_train = len(train_labels)
            num_val = int(num_total_train * VAL_SPLIT)
            
            val_labels = train_labels[:num_val]
            train_labels = train_labels[num_val:]
            
    print(f"Train Classes: {len(train_labels)} | Test Classes: {len(test_labels)} | Validation Classes: {len(val_labels)}")
    return train_labels, test_labels, val_labels

def get_meta_dataloaders_pokedex(tds, train_labels, test_labels, val_labels, n_way, n_shot, n_query, episodes):
    
    # --- Train Loader ---
    # We manually inject the filtered labels into the sampler
    train_sampler = Pokedex(
        dataset=ds,
        target_labels=train_labels, 
        n_way=n_way, n_shot=n_shot, n_query=n_query, n_episodes=episodes
    )
    
    train_loader = DataLoader(ds, batch_sampler=train_sampler, num_workers=2)

    # --- Test Loader ---
    test_sampler = Pokedex(
        dataset=ds, 
        target_labels=test_labels,
        n_way=n_way, n_shot=n_shot, n_query=n_query, n_episodes=episodes
    )
    
    test_loader = DataLoader(ds, batch_sampler=test_sampler, num_workers=2)

    # --- Validation Loader ---
    val_sampler = Pokedex(
        dataset=ds, 
        target_labels=val_labels,
        n_way=n_way, n_shot=n_shot, n_query=n_query, n_episodes=episodes
    )
    
    val_loader = DataLoader(ds, batch_sampler=val_sampler, num_workers=2)
    
    return train_loader, test_loader, val_loader

def get_meta_dataloaders_oak(ds, train_labels, test_labels, val_labels, n_way, n_shot, n_query, episodes):
    
    # --- Train Loader ---
    # We manually inject the filtered labels into the sampler
    train_sampler = Oak(
        dataset=ds,
        target_labels=train_labels, 
        n_way=n_way, n_shot=n_shot, n_query=n_query, n_episodes=episodes
    )
    
    train_loader = DataLoader(ds, batch_sampler=train_sampler, num_workers=2)

    # --- Test Loader ---
    test_sampler = Oak(
        dataset=ds, 
        target_labels=test_labels,
        n_way=n_way, n_shot=n_shot, n_query=n_query, n_episodes=episodes
    )
    
    test_loader = DataLoader(ds, batch_sampler=test_sampler, num_workers=2)

    # --- Validation Loader ---
    val_sampler = Oak(
        dataset=ds, 
        target_labels=val_labels,
        n_way=n_way, n_shot=n_shot, n_query=n_query, n_episodes=episodes
    )
    
    val_loader = DataLoader(ds, batch_sampler=val_sampler, num_workers=2)
    
    return train_loader, test_loader, val_loader