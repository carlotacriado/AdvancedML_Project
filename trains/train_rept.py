from Dataloaders.dataloader import *
from Utils.utils import *
from Utils.globals import *

from torchvision import transforms
from torch.utils.data import DataLoader
import cv2
import time

from Models.Reptile import *
from Models.Baseline import *


# ==========================================
# 0. HELPER PARA MEDICIÓN PRECISA EN GPU
# ==========================================
class CudaTimer:
    """Mide tiempo real en GPU usando eventos de CUDA."""
    def __init__(self, device):
        self.device = device
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.elapsed = 0.0

    def start(self):
        if self.device.type == 'cuda':
            self.start_event.record()
        else:
            self.start_time = time.time()

    def stop(self):
        if self.device.type == 'cuda':
            self.end_event.record()
            torch.cuda.synchronize() # Espera a que todo termine
            self.elapsed = self.start_event.elapsed_time(self.end_event) # devuelve milisegundos
        else:
            self.elapsed = (time.time() - self.start_time) * 1000 # a ms

    def get_time_ms(self):
        return self.elapsed

    def get_time_sec(self):
        return self.elapsed / 1000.0

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_all_seeds(SEED)

    # 1. Initialize Dataset
    transform_pipeline = EVAL_TRANSFORMS
    augment_pipeline = TRAIN_TRANSFORMS

    # Augmented DS
    augmented_ds = PokemonMetaDataset('Data/pokemon_data_linked.csv', 'Data/pokemon_sprites', transform=augment_pipeline)
    # Not Augmented DS
    ds = PokemonMetaDataset('Data/pokemon_data_linked.csv', 'Data/pokemon_sprites', transform=transform_pipeline)

    # 2. Create Loaders
    
    # -- GENERATION PARTITION --
    # train_labels, test_labels, val_labels = get_structured_splits(
    #     ds, 
    #     split_mode='generation', 
    #     train_vals=['generation-i', 'generation-iii'],
    #     val_vals=['generation-ii'],
    #     test_vals=['generation-iv']
    # )
    
    # -- TYPE PARTITION --
    # train_labels, test_labels, val_labels = get_structured_splits(
    #     ds, 
    #     split_mode='type', 
    #     train_vals=['fairy', 'dark', 'dragon', 'rock', 'bug', 'psychic', 'flying', 'water', 'fire', 'grass'],
    #     val_vals=['steel', 'ground', 'ghost'],
    #     test_vals=['ice', 'poison', 'fighting', 'electric', 'normal']
    # )

    # -- RANDOM PARTITION --
    train_labels, test_labels, val_labels = get_structured_splits(
        ds, 
        split_mode='random'
    )

    for n_way in [2,3,4,5]:
        for n_shot in [1,2,3,4,5]:

            train_loader, test_loader, val_loader = get_meta_dataloaders_pokedex(ds, augmented_ds, train_labels, test_labels, val_labels, n_way, n_shot, N_QUERY, EPISODES_PER_EPOCH)
            
            oak_train_loader, oak_test_loader, oak_val_loader = get_meta_dataloaders_oak(ds, augmented_ds, train_labels, test_labels, val_labels, n_way, n_shot, N_QUERY, EPISODES_PER_EPOCH)

            # 3. INITIALIZE MODEL
            print("Initializing Model...")
            print(f"Training with N-WAY:{n_way}, N-SHOT: {n_shot}")
            meta_model = ConvBackbone().to(device)

            # 4. START TRAINING
            train_reptile(meta_model, oak_train_loader, oak_test_loader, oak_val_loader, device, n_way, n_shot, N_QUERY)