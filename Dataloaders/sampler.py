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

class Pokedex(Sampler): # Task Who's That Pokémon
    def __init__(self, dataset, target_labels, n_way, n_shot, n_query, n_episodes):
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
    
        
        # Filter specifically from the source_labels
        self.valid_labels = []

        for label in target_labels:
            if label in dataset.indices_by_label:
                 # 2. Count Check: Do we have enough images?
                # We need at least (K_SHOT + Q_QUERY) images to make a valid task
                image_indices = dataset.indices_by_label[label]
                required_count = n_shot + n_query
                
                if len(image_indices) >= required_count:
                    self.valid_labels.append(label)
                

        
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
    
class Oak(Sampler): # Task: Same Evolution Line, Different Stage
    def __init__(self, dataset, target_labels, n_way, n_shot, n_query, n_episodes):
        """
        Args:
            dataset: Instance of PokemonMetaDataset
            target_labels: List of allowed label indices (Train/Val/Test split)
            n_way: Number of Families per episode
            n_shot: Number of support images (from Stage A)
            n_query: Number of query images (from Stage B)
        """
        self.indices_by_label = dataset.indices_by_label
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_episodes = n_episodes
        
        # Group Labels by Family ID
        self.families = {} # Format: { family_id: [label_idx_1, label_idx_2, ...] }
        
        for label in target_labels:
            if label in dataset.indices_by_label:
                fam_id = dataset.idx_to_family[label]
                
                if fam_id not in self.families: #This should never happen, but just in case ig
                    self.families[fam_id] = []
                self.families[fam_id].append(label)
        
        # Filter Valid Families
        self.valid_families = []
        
        for fam_id, members in self.families.items():
            # To perform this task, a family needs at least 2 distinct members (stages)
            # AND those members must have enough images.
            
            valid_members = []
            for m in members:
                required = max(n_shot, n_query)
                if len(dataset.indices_by_label[m]) >= required:
                    valid_members.append(m)
            
            # If the family has at least 2 valid stages, it can be used
            if len(valid_members) >= 2:
                self.valid_families.append(valid_members)

        if len(self.valid_families) < n_way:
            raise ValueError(f"Not enough valid families (needs {n_way}, found {len(self.valid_families)}). "
                             f"A valid family for this task needs at least 2 evolution stages present in the split.")

    def __iter__(self):
        for _ in range(self.n_episodes):
            batch_indices = []
            
            # 1. Randomly select N Families
            # We pick indices from our list of valid families
            selected_fam_indices = np.random.choice(len(self.valid_families), self.n_way, replace=False)
            
            for f_idx in selected_fam_indices: # type: ignore
                family_members = self.valid_families[f_idx]
                
                # 2. Select Two DIFFERENT Pokemon (Stages) from this family
                # stage_a -> Support
                # stage_b -> Query
                stage_a, stage_b = np.random.choice(family_members, 2, replace=False)
                
                # 3. Get Images
                # Support: n_shot images from Stage A
                indices_a = self.indices_by_label[stage_a]
                support_imgs = np.random.choice(indices_a, self.n_shot, replace=False)
                
                # Query: n_query images from Stage B
                indices_b = self.indices_by_label[stage_b]
                query_imgs = np.random.choice(indices_b, self.n_query, replace=False)
                
                # 4. Add to batch
                # Output structure: [Supp_Fam1..., Query_Fam1..., Supp_Fam2..., Query_Fam2...]
                batch_indices.extend(support_imgs)
                batch_indices.extend(query_imgs)
            
            yield batch_indices

    def __len__(self):
        return self.n_episodes
