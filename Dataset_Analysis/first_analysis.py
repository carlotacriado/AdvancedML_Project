import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import os
import seaborn as sns # type: ignore

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


# ANALISIS POR TIPOS
try:
    df = pd.read_csv("pokemon_data_linked.csv")
    
    # 1. Mapeo: Usamos etiquetas descriptivas para que el análisis de texto posterior funcione
    gen_map = {
        'generation-i': 'Gen 1', 
        'generation-ii': 'Gen 2', 
        'generation-iii': 'Gen 3', 
        'generation-iv': 'Gen 4', 
        'generation-v': 'Gen 5'
    }
    df['gen_short'] = df['generation'].map(gen_map)

except FileNotFoundError:
    print("❌ No encuentro 'pokemon_data_linked.csv'. Asegúrate de haber ejecutado el paso anterior.")
    exit()

print("--- � Análisis de Distribución de Tipos ---")

# 2. Procesar Tipos
type1 = df[['gen_short', 'type_1']].rename(columns={'type_1': 'type'})
type2 = df[['gen_short', 'type_2']].rename(columns={'type_2': 'type'}).dropna()
all_types_df = pd.concat([type1, type2])

# 3. Crear Tabla de Contingencia
type_pivot = pd.crosstab(all_types_df['type'], all_types_df['gen_short'])

# 4. Reordenar columnas (CRUCIAL: Deben coincidir exactamente con los valores de gen_map)
col_order = ['Gen 1', 'Gen 2', 'Gen 3', 'Gen 4', 'Gen 5']
type_pivot = type_pivot.reindex(columns=col_order)

# 5. Visualización y GUARDADO
plt.figure(figsize=(12, 10))
sns.heatmap(type_pivot, annot=True, fmt='d', cmap='YlGnBu', linewidths=.5)

plt.title('Frecuencia de Tipos por Generación', fontsize=16)
plt.xlabel('Conjunto de Datos (Generación)', fontsize=12)
plt.ylabel('Tipo Elemental', fontsize=12)

# --- AQUÍ ESTÁ EL GUARDADO ---
plt.savefig("heatmap_tipos.png", dpi=300, bbox_inches='tight')
print("✅ Gráfico guardado como 'heatmap_tipos.png'")

# plt.show() # Puedes comentar esto si no quieres que se abra la ventana

# 6. Análisis Numérico de Desbalance
print("\n## ⚠️ Detección de Tipos Raros en Test (Gen 5)")

# Nota: Esto requiere que el nombre de la columna contenga la palabra "Test" o "Train"
test_types = all_types_df[all_types_df['gen_short'].str.contains('Test', na=False)]['type'].value_counts()
train_types = all_types_df[all_types_df['gen_short'].str.contains('Train', na=False)]['type'].value_counts()

for t in test_types.index:
    count_train = train_types.get(t, 0)
    count_test = test_types.get(t, 0)
    
    if count_train < 10:
        print(f"⚠️ ALERTA: El tipo '{t}' aparece {count_test} veces en Test, pero solo {count_train} veces en todo el Train set.")
        print(f"   -> El modelo podría tener dificultades para reconocer características de '{t}'.")

print("\nAnálisis completado.")


# ANALISIS DE TAMAÑOS
import os
import pandas as pd
from PIL import Image # type: ignore

# --- CONFIGURA ESTAS RUTAS ---
ROOT_IMG_DIR = r"C:\Users\Administrator\OneDrive\Escritorio\IA\Curso 4\Advanced ML\Project Rep\AdvancedML_Project\AdvancedML_Project\pokemon_sprites"
CSV_PATH = "pokemon_data_linked.csv"

# 1. Cargar datos
try:
    df = pd.read_csv(CSV_PATH)
except:
    print("❌ No se encuentra el CSV.")
    exit()

print("Verificando tamaños...")

unique_sizes = set()
count = 0

# 2. Revisar una imagen por cada Pokémon
for idx, row in df.iterrows():
    folder_path = os.path.join(ROOT_IMG_DIR, row['folder_name'])
    
    if os.path.exists(folder_path):
        # Buscar la primera imagen que pille
        files = os.listdir(folder_path)
        images = [f for f in files if f.endswith(('.png', '.jpg'))]
        
        if images:
            img_path = os.path.join(folder_path, images[0])
            with Image.open(img_path) as img:
                unique_sizes.add(img.size) # Guarda (ancho, alto)
                count += 1

# 3. Resultado Final
print(f"\n--- RESULTADO ({count} imágenes analizadas) ---")

if len(unique_sizes) == 1:
    width, height = unique_sizes.pop()
    print(f"✅ SÍ. Todas son iguales.")
    print(f"� Tamaño: {width} x {height} píxeles.")
else:
    print(f"❌ NO. Tienen tamaños diferentes.")
    print(f"Variedad encontrada: {len(unique_sizes)} tamaños distintos.")
    print(f"Ejemplos (Ancho x Alto): {list(unique_sizes)[:5]}")