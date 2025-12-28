import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image

# --- CONFIGURATION ---
# Replace these paths with your actual local paths
BASE_PATH = r"C:\Users\Administrator\OneDrive\Escritorio\IA\Curso 4\Advanced ML\Project Rep\AdvancedML_Project\AdvancedML_Project"
CSV_FILE_NAME = "pokemon_data_gen1-5.csv"
SPRITES_DIR_NAME = "pokemon_sprites"

CSV_PATH = os.path.join(BASE_PATH, CSV_FILE_NAME)
ROOT_IMG_DIR = os.path.join(BASE_PATH, SPRITES_DIR_NAME)

# --- 1. DATA LOADING & CLEANING ---
try:
    df = pd.read_csv(CSV_PATH)
    print(f"CSV loaded successfully with {len(df)} Pokemon.")
except FileNotFoundError:
    print(f"Error: CSV not found at {CSV_PATH}")
    exit()

# Handle missing evolution data
df["pre_evolution"] = df["pre_evolution"].fillna("None")
df["evolution"] = df["evolution"].fillna("None")

print("\n---- Evolutionary Chain Analysis ----")
[Image of Pokemon evolutionary line diagram]

# --- 2. EVOLUTIONARY ROLE CLASSIFICATION ---

def classify_evolution_role(row):
    """Classifies the Pokemon based on its position in the evolution chain."""
    has_pre = row["pre_evolution"] != "None"
    has_evo = row["evolution"] != "None"

    if not has_pre and has_evo:
        return "First Link (Just Evolves)"
    elif has_pre and has_evo:
        return 'Intermediate Link (Pre & Post)'
    elif has_pre and not has_evo:
        return 'Last Link (Just Pre-Evolves)'
    else:
        return 'No Evolution'

df["evolution_role"] = df.apply(classify_evolution_role, axis=1)

# Output: Role Distribution
print("\n## Summary per Evolutionary Role")
print(df["evolution_role"].value_counts().to_markdown())

# Count valid task Pokemon (those that can evolve)
valid_task_pokemons = df[df["evolution_role"].isin(["First Link (Just Evolves)", 'Intermediate Link (Pre & Post)'])]
print(f"\nTotal Pokemon part of an evolutionary task: {len(valid_task_pokemons)}")

# Output: Distribution per Generation
print("\n## Task Distribution per Generation")
gen_distribution = df.groupby("generation")["evolution_role"].value_counts().unstack(fill_value=0)
print(gen_distribution.to_markdown())

# Identify unique chains
first_stage_pokemons = df[df['evolution_role'] == "First Link (Just Evolves)"]['name'].tolist()
print(f"\nApproximate number of unique evolutionary chains: {len(first_stage_pokemons)}") 

# Analyze image count column provided in dataset
print("\n## Dataset Image Count Analysis")
images_unique_counts = df["num_of_images"].unique()
if len(images_unique_counts) == 1:
    print(f"All classes have the same number of images: {images_unique_counts[0]}")
else:
    print(f"ATTENTION: Image counts vary {images_unique_counts}. Review for Meta-Learning balance.")

# --- 3. FOLDER MAPPING (CSV <-> DIRECTORIES) ---

print("\n--- Linking CSV Data to Image Folders ---")

if not os.path.exists(ROOT_IMG_DIR):
    print(f"Error: Image directory does not exist: {ROOT_IMG_DIR}")
    exit()

all_folders = os.listdir(ROOT_IMG_DIR)
folder_map = {}

# Map ID to Folder Name (Assumes folder format "001-Name")
for folder_name in all_folders:
    if not os.path.isdir(os.path.join(ROOT_IMG_DIR, folder_name)):
        continue
        
    try:
        parts = folder_name.split('-')
        if len(parts) > 0 and parts[0].isdigit():
            pokemon_id = int(parts[0]) 
            folder_map[pokemon_id] = folder_name
    except ValueError:
        continue 

print(f"Mapping created. Identified {len(folder_map)} valid folders.")

def get_real_path(row):
    dex_num = row['dex_number']
    return folder_map.get(dex_num, "NOT_FOUND")

df['folder_name'] = df.apply(get_real_path, axis=1)

# Verify Mapping
found = df[df['folder_name'] != "NOT_FOUND"]
missing = df[df['folder_name'] == "NOT_FOUND"]

print(f"Success: {len(found)} folders linked.")
print(f"Missing: {len(missing)} folders linked.")

if len(found) > 0:
    linked_csv_name = "pokemon_data_linked.csv"
    df.to_csv(linked_csv_name, index=False)
    print(f"\nSaved updated dataset with folder paths to '{linked_csv_name}'")

# --- 4. TYPE DISTRIBUTION ANALYSIS ---

print("\n--- Type Distribution Analysis ---")
[Image of seaborn heatmap example]

# Map generation names to short labels for plotting
gen_map = {
    'generation-i': 'Gen 1', 
    'generation-ii': 'Gen 2', 
    'generation-iii': 'Gen 3', 
    'generation-iv': 'Gen 4', 
    'generation-v': 'Gen 5'
}
df['gen_short'] = df['generation'].map(gen_map)

# Process Types (Combine Type 1 and Type 2)
type1 = df[['gen_short', 'type_1']].rename(columns={'type_1': 'type'})
type2 = df[['gen_short', 'type_2']].rename(columns={'type_2': 'type'}).dropna()
all_types_df = pd.concat([type1, type2])

# Create Pivot Table
type_pivot = pd.crosstab(all_types_df['type'], all_types_df['gen_short'])
col_order = ['Gen 1', 'Gen 2', 'Gen 3', 'Gen 4', 'Gen 5']
type_pivot = type_pivot.reindex(columns=col_order)

# Generate and Save Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(type_pivot, annot=True, fmt='d', cmap='YlGnBu', linewidths=.5)
plt.title('Type Frequency per Generation', fontsize=16)
plt.xlabel('Generation', fontsize=12)
plt.ylabel('Elemental Type', fontsize=12)

plt.savefig("heatmap_types.png", dpi=300, bbox_inches='tight')
print("Heatmap saved as 'heatmap_types.png'")

# --- 5. IMAGE SIZE VERIFICATION ---

print("\n--- Verifying Actual Image Dimensions ---")
[Image of Pokemon pixel sprite]

unique_sizes = set()
count = 0

# Check dimensions of the first image in each folder
for idx, row in df.iterrows():
    if row['folder_name'] == "NOT_FOUND":
        continue

    folder_path = os.path.join(ROOT_IMG_DIR, row['folder_name'])
    
    if os.path.exists(folder_path):
        files = os.listdir(folder_path)
        images = [f for f in files if f.endswith(('.png', '.jpg'))]
        
        if images:
            img_path = os.path.join(folder_path, images[0])
            with Image.open(img_path) as img:
                unique_sizes.add(img.size)
                count += 1

print(f"Analyzed {count} images.")

if len(unique_sizes) == 1:
    print(f"Success: All images have the same dimensions: {list(unique_sizes)[0]}")
else:
    print("Warning: Image sizes vary.")
    print(f"Unique sizes found: {len(unique_sizes)}")
    print(f"Examples: {list(unique_sizes)[:5]}")