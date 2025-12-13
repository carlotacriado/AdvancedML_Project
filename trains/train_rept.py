from Dataloaders.dataloader import *
from Utils.utils import *
from Utils.globals import *

from torchvision import transforms
from torch.utils.data import DataLoader
import cv2

from Models.Reptile import *
from Models.Baseline import *

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_all_seeds(SEED)

    # 1. Initialize Dataset
    # -- NO DATA AUGMENTATION --

    transform_pipeline = EVAL_TRANSFORMS

    
    ds = PokemonMetaDataset('Data/pokemon_data_linked.csv', 'Data/pokemon_sprites', transform=transform_pipeline)

    # 2. Create Loaders
    
    # -- GENERATION PARTITION --
    # train_labels, test_labels, val_labels = get_structured_splits(
    #     ds, 
    #     split_mode='generation', 
    #     train_vals=['generation-i', 'generation-iii'],
    #     #val_vals=['generation-ii'],
    #     test_vals=['generation-iv']
    # )
    
    # -- TYPE PARTITION --
    # train_labels, test_labels, val_labels = get_structured_splits(
    #     ds, 
    #     split_mode='type', 
    #     train_vals=['fairy', 'dark', 'dragon', 'rock', 'bug', 'psychic', 'flying', 'water', 'fire', 'grass'],
    #     val_vals=['steel', 'ground', 'ghost']
    #     test_vals=['ice', 'poison', 'fighting', 'electric', 'normal']
    # )

    # -- RANDOM PARTITION --
    train_labels, test_labels, val_labels = get_structured_splits(
        ds, 
        split_mode='random'
    )

    for n_way in [2,3,4,5]: #
        for n_shot in [1,2,3,4,5]: #
            train_loader, test_loader, val_loader = get_meta_dataloaders_pokedex(ds, train_labels, test_labels, val_labels, n_way, n_shot, N_QUERY, EPISODES_PER_EPOCH)
            
            oak_train_loader, oak_test_loader, oak_val_loader = get_meta_dataloaders_oak(ds, train_labels, test_labels, val_labels, n_way, n_shot, N_QUERY, EPISODES_PER_EPOCH)

            # 3. INITIALIZE MODEL
            print("Initializing Model...")
            print(f"Training with N-WAY:{n_way}, N-SHOT: {n_shot}")
            meta_model = ConvBackbone().to(device)

            # 4. START TRAINING
            train_reptile(meta_model, train_loader, test_loader, val_loader, device, n_way, n_shot, N_QUERY)