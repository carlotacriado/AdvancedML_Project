import numpy as np
import pandas as pd

# --- AÑADIR COLUMNA FAMILY_ID ---
print("\n--- �‍�‍�‍� Generando IDs de Familia Evolutiva ---")

# 1. Recargar el CSV por si acaso (o usar el df que ya tienes en memoria)
df = pd.read_csv("pokemon_data_linked.csv")
df['pre_evolution'] = df['pre_evolution'].fillna("None")

# 2. Mapa para rastrear el "Fundador" de cada Pokémon
# Clave: Nombre del Pokémon -> Valor: Nombre del Fundador (Family Leader)
family_leaders = {}

# Convertimos el DF a diccionarios para acceso rápido
poke_to_pre = dict(zip(df['name'], df['pre_evolution']))
all_names = set(df['name'])

def find_leader(pokemon_name):
    """Función recursiva para encontrar al Pokémon base de la cadena"""
    # Si no está en nuestro dataset (ej. casos raros), devolvemos el mismo nombre
    if pokemon_name not in poke_to_pre:
        return pokemon_name
    
    pre = poke_to_pre[pokemon_name]
    
    # Caso Base: No tiene pre-evolución, él es el líder
    if pre == "None":
        return pokemon_name
    
    # Caso Recursivo: Buscamos al líder de su pre-evolución
    return find_leader(pre)

# 3. Aplicar la búsqueda a todos
df['family_leader'] = df['name'].apply(find_leader)

# 4. Asignar un ID numérico único a cada líder
# Obtenemos los líderes únicos y les damos un número
unique_leaders = df['family_leader'].unique()
leader_to_id = {name: i for i, name in enumerate(unique_leaders)}

df['family_id'] = df['family_leader'].map(leader_to_id)

# 5. Guardar y Mostrar
print(f"Familias únicas encontradas: {len(unique_leaders)}")
df.to_csv("pokemon_data_linked.csv", index=False)
print("✅ Archivo actualizado con columnas 'family_leader' y 'family_id'.")

# Ejemplo de verificación
example_fam = df[df['family_leader'] == 'Charmander'][['name', 'family_id', 'generation']]
print("\nEjemplo (Familia Charmander):")
print(example_fam.to_markdown(index=False))