import pandas as pd
import numpy as np
import matplotlib as plt
import os

# Dataset
df = pd.read_csv(r"C:\Users\Administrator\OneDrive\Escritorio\IA\Curso 4\Advanced ML\Project Rep\AdvancedML_Project\AdvancedML_Project\pokemon_data_gen1-5.csv")

# Cleaning
df["pre_evolution"] = df["pre_evolution"].fillna("None")
df["evolution"] = df["evolution"].fillna("None")

print("---- Analysis of the distribution of the evolutionary chain ----")

# ----- Classify the tasks (evolutions) -----

# Classify by rol in the evolutionary chain
def classify_evolution_role(row):
    has_pre = row["pre_evolution"] != "None"
    has_evo = row["evolution"] != "None"

    if not has_pre and has_evo:
        return "First Link (Just Evolves)"
    elif has_pre and has_evo:
        return 'Intermediate Link (Pre & Post)'
    elif has_pre and not has_evo:
        return 'last Link (Just Pre-Evolves)'
    else:
        return 'No Evolution'

df["evolution_role"] = df.apply(classify_evolution_role, axis = 1)

# See results
print("\n## Summary per Evolutionary Rol")
role_distribution = df["evolution_role"].value_counts()
print(role_distribution.to_markdown())

valid_task_pokemons = df[df["evolution_role"].isin(["First Link (Just Evolves)", 'Intermediate Link (Pre & Post)'])]
print(f"\nTotal of Pokémon that are **Part of an Evolutionary Task (have 'evolution'):** {len(valid_task_pokemons)}")

# Analysis per generation
print("\n## Tasks Distribution per generation")
gen_distribution = df.groupby("generation")["evolution_role"].value_counts().unstack(fill_value=0)
print(gen_distribution.to_markdown())

# Chain identification (to understand the complexity)
first_stage_pokemons = df[df['evolution_role'] == "First Link (Just Evolves)"]['name'].tolist()
print(f"\nTotal number of unique evolutionary chain (Approximated): {len(first_stage_pokemons)}") 
print("\nExamples of First Link (Starting point):")
print(first_stage_pokemons[:10])

# Analysis of images (num_of_images)
print("\n## Analysis of the number of images per Pokemon")
images_unique_counts = df["num_of_images"].unique()
print(f"Unique values in 'num_of_images': {images_unique_counts}")

if len(images_unique_counts) == 1:
    print(f"All the Pokemon classes have the same number of images: {images_unique_counts[0]}")
else:
    print("ATENTION: The number of images varies --> Revise for Meta-Learning")

# ---- ANALISIS DE LAS PROPIAS IMAGENES ----
ROOT_IMG_DIR = r"C:\Users\Administrator\OneDrive\Escritorio\IA\Curso 4\Advanced ML\Project Rep\AdvancedML_Project\AdvancedML_Project\pokemon_sprites"

CSV_PATH = r"C:\Users\Administrator\OneDrive\Escritorio\IA\Curso 4\Advanced ML\Project Rep\AdvancedML_Project\AdvancedML_Project\pokemon_data_gen1-5.csv"

# 1. Cargar CSV
try:
    df = pd.read_csv(CSV_PATH)
    print(f"CSV cargado con {len(df)} Pokémon.")
except:
    print("❌ No se encuentra el CSV. Revisa la ruta CSV_PATH.")
    exit()

# 2. Escanear Carpetas Reales
if not os.path.exists(ROOT_IMG_DIR):
    print(f"❌ La ruta de imágenes no existe: {ROOT_IMG_DIR}")
    exit()

all_folders = os.listdir(ROOT_IMG_DIR)
print(f"Se encontraron {len(all_folders)} elementos en la carpeta de imágenes.")

# 3. Crear Mapa: ID -> Nombre de Carpeta Real
# La lógica: Asumimos que la carpeta empieza con el número (ej: "001-...")
folder_map = {}

print("--- Iniciando Emparejamiento ---")

for folder_name in all_folders:
    # Ignorar archivos sueltos, solo queremos directorios
    if not os.path.isdir(os.path.join(ROOT_IMG_DIR, folder_name)):
        continue
        
    # Intentar extraer el ID del principio (ej: "001" -> 1)
    try:
        # Separamos por guion o espacio para pillar el número
        # Asumimos formato "001-nombre"
        parts = folder_name.split('-')
        if len(parts) > 0 and parts[0].isdigit():
            pokemon_id = int(parts[0]) # Convertir "001" a 1
            folder_map[pokemon_id] = folder_name
    except:
        continue # Si la carpeta no empieza por número, la ignoramos

print(f"Mapa creado. Se identificaron {len(folder_map)} carpetas válidas con ID.")

# 4. Cruzar con el DataFrame
def get_real_path(row):
    dex_num = row['dex_number'] # Asumimos que esta columna tiene el ID (1, 2, 3...)
    
    if dex_num in folder_map:
        return folder_map[dex_num]
    else:
        return "NOT_FOUND"

df['folder_name'] = df.apply(get_real_path, axis=1)

# 5. Verificar Resultados
found = df[df['folder_name'] != "NOT_FOUND"]
missing = df[df['folder_name'] == "NOT_FOUND"]

print(f"\n✅ Pokémon encontrados correctamente: {len(found)}")
print(f"❌ Pokémon sin carpeta: {len(missing)}")

if len(missing) > 0:
    print("\nEjemplos de Pokémon que fallaron (Primeros 5):")
    print(missing[['dex_number', 'name']].head().to_markdown())
else:
    print("\n🎉 ¡ÉXITO! Todos los Pokémon del CSV tienen su carpeta asignada.")

# Guardar el CSV arreglado para no tener que hacer esto de nuevo
if len(found) > 0:
    df.to_csv("pokemon_data_linked.csv", index=False)
    print("\n💾 Se ha guardado un nuevo archivo 'pokemon_data_linked.csv' con la columna 'folder_name' correcta.")