import numpy as np
import pandas as pd

CSV_FILE = "pokemon_data_linked.csv"

print("\n--- Generating Evolutionary Family IDs ---")


[Image of phylogenetic tree diagram]


# 1. Load Data
try:
    df = pd.read_csv(CSV_FILE)
    df['pre_evolution'] = df['pre_evolution'].fillna("None")
except FileNotFoundError:
    print(f"Error: Could not find '{CSV_FILE}'.")
    exit()

# 2. Setup Lookup Dictionary
# Mapping: Pokemon Name -> Pre-Evolution Name for fast access
poke_to_pre = dict(zip(df['name'], df['pre_evolution']))

def find_leader(pokemon_name):
    """
    Recursive function to trace back to the base (founder) of the evolutionary line.
    """
    # Safety check: if pokemon is not in dataset, return itself
    if pokemon_name not in poke_to_pre:
        return pokemon_name
    
    pre = poke_to_pre[pokemon_name]
    
    # Base Case: No pre-evolution means this is the leader (e.g., Charmander)
    if pre == "None":
        return pokemon_name
    
    # Recursive Case: Continue tracing back the pre-evolution (e.g., Charizard -> Charmeleon -> Charmander)
    return find_leader(pre)

# 3. Apply Logic
print("Tracing evolutionary lineages...")
df['family_leader'] = df['name'].apply(find_leader)

# 4. Assign Numeric IDs
# Identify unique leaders and assign a unique integer to each family
unique_leaders = df['family_leader'].unique()
leader_to_id = {name: i for i, name in enumerate(unique_leaders)}

df['family_id'] = df['family_leader'].map(leader_to_id)

# 5. Save and Verify
print(f"Total unique families identified: {len(unique_leaders)}")

df.to_csv(CSV_FILE, index=False)
print(f"✅ Success: File '{CSV_FILE}' updated with 'family_leader' and 'family_id'.")

# Verification Example
print("\nExample Verification (Charmander Family):")
example_fam = df[df['family_leader'] == 'Charmander'][['name', 'family_id', 'generation']]
print(example_fam.to_markdown(index=False))