"""
ESTRUCTURA DESEADA
    - Dataset (guarda: imagen, etiqueta, tipo1, tipo2, generación, id_evolución)
    - Classification_task
    - Evolution_task
    - La task como tal (en tensores y con el support_meta para poder hacer el entrenamiento condicionado)
"""
   

import os
import pandas as pd
from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
import numpy as np
import random

# ==================================================================================
# 1. ZONA DE CONTROL (CONFIGURACIÓN)
# ==================================================================================
# Aquí es donde tú mandas. Define qué va a dónde.
# Puedes dejar listas vacías [] si no quieres usar ese criterio.

SPLIT_CONFIG = {
    # --- NIVEL 1: ESPECÍFICOS POR NOMBRE (Prioridad Máxima) ---
    # Si pones un nombre aquí, SU FAMILIA ENTERA se mueve a ese set.
    'test_names': ['Lucario', 'Pikachu', 'Eevee'],  
    'val_names':  ['Snorlax', 'Bulbasaur'],
    
    # --- NIVEL 2: POR GENERACIÓN ---
    # "Todo lo de Gen 5 que no haya sido asignado antes, va a Test"
    'test_gens': ['generation-v'], 
    'val_gens':  ['generation-iv'], # Gen 4 para validar
    
    # --- NIVEL 3: POR TIPO ---
    # "Todo lo de tipo Dragón que sobre, va a Test"
    'test_types': ['dragon'], 
    'val_types':  ['ghost'],
    
    # --- SEMILLA (Para que el azar sea repetible) ---
    'seed': 42
}

# Configuración de imagen
IMAGE_SIZE = 84
preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ==================================================================================
# 2. EL DATASET INTELIGENTE
# ==================================================================================
class PokemonMetaDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # --- CARGA Y LIMPIEZA DE DATOS ---
        self.df = pd.read_csv(csv_file)
        # Rellenamos huecos vacíos para evitar errores luego
        self.df['pre_evolution'] = self.df['pre_evolution'].fillna('')
        self.df['evolution'] = self.df['evolution'].fillna('')
        self.df['type_2'] = self.df['type_2'].fillna('None')
        
        # --- PREPARACIÓN DE DICCIONARIOS (Texto a Números) ---
        # Esto sirve para cuando quieras usar Conditioning (Darle el tipo a la red)
        all_types = sorted(list(set(self.df['type_1'].unique()) | set(self.df['type_2'].unique())))
        self.type_to_idx = {t: i for i, t in enumerate(all_types)}
        
        all_gens = sorted(self.df['generation'].unique())
        self.gen_to_idx = {g: i for i, g in enumerate(all_gens)}
        
        # Mapa rápido para buscar datos por nombre
        self.name_to_row = {row['name'].lower(): row for _, row in self.df.iterrows()}
        self.idx_to_name = {}
        
        # --- CONSTRUCCIÓN DE FAMILIAS (El paso más importante) ---
        # Agrupamos Pokémon para evitar que Charmander esté en Train y Charizard en Test.
        self.family_map = self._build_families()
        
        # Listas principales
        self.samples = []             # Lista plana de todas las imágenes
        self.indices_by_label = {}    # Índice: Pokémon individual (ej: ID 25 -> Todas las fotos de Pikachu)
        self.indices_by_family = {}   # Índice: Familia (ej: ID 1 -> Fotos de Bulba, Ivy y Venusaur)
        
        # Metadatos de Familia: Nos servirá para aplicar los filtros de Split
        # Estructura: family_id -> { 'names':Set, 'gens':Set, 'types':Set }
        self.family_metadata = {} 

        print(">>> [DATASET] Indexando imágenes y reconstruyendo familias...")
        
        # Iteramos por cada Pokémon en el CSV
        for idx, row in self.df.iterrows():
            name = row['name']
            dex = row['dex_number']
            
            # Buscamos la carpeta (intentamos varios formatos de nombre)
            dex_str = str(dex).zfill(3)
            possible_folders = [f"{dex_str}-{name.lower()}", f"{dex}-{name.lower()}", name.lower()]
            
            folder_path = None
            for f in possible_folders:
                p = os.path.join(root_dir, f)
                if os.path.isdir(p):
                    folder_path = p
                    break
            
            # Si encontramos la carpeta con imágenes...
            if folder_path:
                images = list(Path(folder_path).glob('*.*'))
                
                # Obtenemos el ID de familia calculado
                fam_id = self.family_map.get(name.lower(), dex)
                
                # Calculamos el 'Stage' (0=Bebé, 1=Evo1, etc.) recorriendo hacia atrás
                stage = 0
                current_pre = row['pre_evolution']
                while current_pre:
                    stage += 1
                    if current_pre.lower() in self.name_to_row:
                        current_pre = self.name_to_row[current_pre.lower()]['pre_evolution']
                    else:
                        current_pre = '' # Fin de la cadena

                # Guardamos los metadatos de la familia (para el Splitter)
                if fam_id not in self.family_metadata:
                    self.family_metadata[fam_id] = {'names': set(), 'gens': set(), 'types': set()}
                
                self.family_metadata[fam_id]['names'].add(name.lower())
                self.family_metadata[fam_id]['gens'].add(row['generation'])
                self.family_metadata[fam_id]['types'].add(row['type_1'])
                if row['type_2'] != 'None': self.family_metadata[fam_id]['types'].add(row['type_2'])

                # Inicializamos listas
                if idx not in self.indices_by_label: self.indices_by_label[idx] = []
                if fam_id not in self.indices_by_family: self.indices_by_family[fam_id] = []
                self.idx_to_name[idx] = name

                # Guardamos cada imagen encontrada
                for img_path in images:
                    if img_path.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                        current_sample_idx = len(self.samples)
                        
                        # --- PAQUETE DE DATOS COMPLETO ---
                        meta_info = {
                            'path': str(img_path),
                            'label_idx': idx,      # ID específico (ej: Pikachu)
                            'family_id': fam_id,   # ID de grupo (ej: Familia Pikachu)
                            'stage': stage,        # Etapa evolutiva
                            'type1_idx': self.type_to_idx[row['type_1']],
                            'type2_idx': self.type_to_idx[row['type_2']],
                            'gen_idx': self.gen_to_idx[row['generation']]
                        }
                        
                        self.samples.append(meta_info)
                        self.indices_by_label[idx].append(current_sample_idx)
                        self.indices_by_family[fam_id].append(current_sample_idx)

    def _build_families(self):
        """Conecta hijos con padres para darles el mismo ID"""
        parent_map = {}
        for name, row in self.name_to_row.items():
            if row['pre_evolution']: parent_map[name] = row['pre_evolution'].lower()
        
        family_ids = {}
        for name in self.name_to_row.keys():
            root = name
            steps = 0
            while root in parent_map and steps < 10:
                root = parent_map[root]
                steps += 1
            if root in self.name_to_row:
                family_ids[name] = self.name_to_row[root]['dex_number']
            else:
                family_ids[name] = self.name_to_row[name]['dex_number']
        return family_ids
        
    def _load_and_fix(self, path):
        """Carga imagen y arregla fondo transparente a blanco"""
        try:
            image = Image.open(path).convert('RGBA')
            bg = Image.new('RGBA', image.size, (255, 255, 255, 255))
            bg.paste(image, mask=image.split()[-1])
            return bg.convert('RGB')
        except:
            return Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE))

    def __len__(self): return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = self._load_and_fix(sample['path'])
        if self.transform: image = self.transform(image)
        return image, sample

# ==================================================================================
# 3. EL GRAN SPLITTER (DIVISOR DE DATOS)
# ==================================================================================
def get_train_val_test_splits(dataset, config):
    """
    Esta función decide el destino de cada familia de Pokémon basándose en la configuración.
    Devuelve 3 listas: [IDs Familia Train], [IDs Familia Val], [IDs Familia Test]
    """
    all_fams = list(dataset.indices_by_family.keys())
    train_fams, val_fams, test_fams = [], [], []
    
    # Preparamos los sets de búsqueda para ir rápido (normalizamos a minúsculas)
    cfg_test_names = set([n.lower() for n in config.get('test_names', [])])
    cfg_val_names  = set([n.lower() for n in config.get('val_names', [])])
    
    cfg_test_gens  = set(config.get('test_gens', []))
    cfg_val_gens   = set(config.get('val_gens', []))
    
    cfg_test_types = set(config.get('test_types', []))
    cfg_val_types  = set(config.get('val_types', []))

    print("\n>>> [SPLITTER] Iniciando división de datos...")

    for fam_id in all_fams:
        # Recuperamos los datos de esta familia (nombres, gens, tipos de sus miembros)
        f_meta = dataset.family_metadata[fam_id]
        assigned = False
        
        # --- NIVEL 1: NOMBRES ESPECÍFICOS (Prioridad Máxima) ---
        # Si algún miembro de la familia está en la lista prohibida, movemos TODA la familia.
        if not f_meta['names'].isdisjoint(cfg_test_names):
            test_fams.append(fam_id); assigned = True
            print(f"   -> [Específico] Familia {list(f_meta['names'])[0]} movida a TEST")
        elif not f_meta['names'].isdisjoint(cfg_val_names):
            val_fams.append(fam_id); assigned = True
            print(f"   -> [Específico] Familia {list(f_meta['names'])[0]} movida a VAL")
            
        if assigned: continue # Ya tiene destino, pasamos al siguiente

        # --- NIVEL 2: GENERACIONES ---
        if not f_meta['gens'].isdisjoint(cfg_test_gens):
            test_fams.append(fam_id); assigned = True
        elif not f_meta['gens'].isdisjoint(cfg_val_gens):
            val_fams.append(fam_id); assigned = True
            
        if assigned: continue

        # --- NIVEL 3: TIPOS ---
        if not f_meta['types'].isdisjoint(cfg_test_types):
            test_fams.append(fam_id); assigned = True
        elif not f_meta['types'].isdisjoint(cfg_val_types):
            val_fams.append(fam_id); assigned = True
            
        if assigned: continue

        # --- NIVEL 4: LO QUE SOBRA (Entrenamiento) ---
        train_fams.append(fam_id)

    # Resumen
    print("-" * 40)
    print(f"TOTAL FAMILIAS: {len(all_fams)}")
    print(f"TRAIN: {len(train_fams)} familias (Gimnasio)")
    print(f"VAL:   {len(val_fams)} familias (Examen de prueba)")
    print(f"TEST:  {len(test_fams)} familias (Examen final)")
    print("-" * 40)
    
    return train_fams, val_fams, test_fams

# ==================================================================================
# 4. GENERADORES DE TAREAS (META-LEARNING)
# ==================================================================================

class MetaTask:
    """Clase que empaqueta una tarea lista para ser consumida por el modelo"""
    def __init__(self, support_data, query_data, task_name, class_names):
        self.name = task_name
        self.class_names = class_names
        # Datos (X = Imágenes, Y = Etiquetas, Meta = Datos Extra)
        self.support_x, self.support_y, self.support_meta = support_data
        self.query_x, self.query_y, self.query_meta = query_data
        
        #! que hago con el self.meta ¿? lo hago opcional? lo añado siempre?
        # Loaders de PyTorch
        self.support_loader = DataLoader(TensorDataset(self.support_x, self.support_y), batch_size=32, shuffle=True)
        self.query_loader = DataLoader(TensorDataset(self.query_x, self.query_y), batch_size=32, shuffle=True)
    
    def inspect(self):
        print(f"\nTask: {self.name}")
        print(f"Clases: {self.class_names}")
        print(f"Support Set: {self.support_x.shape} imágenes")

def get_batch_tensors(dataset, indices, relative_labels):
    """Auxiliar: Convierte índices de dataset a Tensores de PyTorch con metadatos"""
    imgs, types1, gens = [], [], []
    for idx in indices:
        img, meta = dataset[idx]
        imgs.append(img)
        types1.append(meta['type1_idx'])
        gens.append(meta['gen_idx'])
    
    # Creamos diccionario de tensores para condicionamiento
    meta_tensors = {
        'type_1': torch.tensor(types1).long(),
        'gen': torch.tensor(gens).long()
    }
    return (torch.stack(imgs), torch.tensor(relative_labels).long(), meta_tensors)

# --- TAREA A: CLASIFICACIÓN (¿Quién es este Pokémon?) ---
def create_classification_task(dataset, allowed_families, n_way=5, n_shot=5, n_query=5):
    """
    Genera una tarea eligiendo Pokémon SOLO de las 'allowed_families'.
    Esto asegura que si pasas 'train_families', nunca saldrá un Pokémon de test.
    """
    # 1. Obtener todos los Pokémon individuales disponibles en esas familias
    allowed_labels = []
    for fam_id in allowed_families:
        img_idxs = dataset.indices_by_family[fam_id]
        # Extraemos los label_idx únicos (ej: Pikachu, Raichu)
        labels_in_fam = set([dataset.samples[i]['label_idx'] for i in img_idxs])
        allowed_labels.extend(list(labels_in_fam))
    
    # 2. Filtrar los que tengan suficientes fotos (shot + query)
    valid_labels = [l for l in allowed_labels 
                    if l in dataset.indices_by_label 
                    and len(dataset.indices_by_label[l]) >= (n_shot + n_query)]
    
    if len(valid_labels) < n_way: return None # No hay suficientes clases
        
    # 3. Muestrear N clases al azar
    selected = np.random.choice(valid_labels, n_way, replace=False)
    class_names = [dataset.idx_to_name[i] for i in selected]
    
    s_idxs, q_idxs, s_lbls, q_lbls = [], [], [], []
    
    for i, cls_idx in enumerate(selected):
        all_idxs = dataset.indices_by_label[cls_idx]
        # Elegimos fotos sin repetir
        sel_imgs = np.random.choice(all_idxs, n_shot + n_query, replace=False)
        
        # Dividimos en Support y Query
        s_idxs.extend(sel_imgs[:n_shot])
        s_lbls.extend([i]*n_shot) # Etiqueta relativa 0, 1, 2...
        
        q_idxs.extend(sel_imgs[n_shot:])
        q_lbls.extend([i]*n_query)
        
    return MetaTask(get_batch_tensors(dataset, s_idxs, s_lbls),
                    get_batch_tensors(dataset, q_idxs, q_lbls),
                    "Clasificación", class_names)

# --- TAREA B: EVOLUCIÓN (¿Quién evoluciona a quién?) ---
def create_evolution_task(dataset, allowed_families, n_way=5, n_shot=1, n_query=1):
    """
    Genera una tarea donde:
    Support = Etapa Baja (ej. Charmander)
    Query = Etapa Alta (ej. Charizard)
    Objetivo: Relacionar conceptos evolutivos.
    """
    # 1. Buscar familias con al menos 2 etapas evolutivas dentro de las permitidas
    valid_subset = []
    for fam_id in allowed_families:
        if fam_id in dataset.indices_by_family:
            idxs = dataset.indices_by_family[fam_id]
            stages = set([dataset.samples[i]['stage'] for i in idxs])
            if len(stages) >= 2:
                valid_subset.append(fam_id)
            
    if len(valid_subset) < n_way: return None

    # 2. Seleccionar N familias
    selected = np.random.choice(valid_subset, n_way, replace=False)
    s_idxs, q_idxs, s_lbls, q_lbls, names = [], [], [], [], []
    
    for i, fam_id in enumerate(selected):
        fam_idxs = dataset.indices_by_family[fam_id]
        
        # Agrupar fotos por stage
        stages_dict = {}
        for idx in fam_idxs:
            st = dataset.samples[idx]['stage']
            if st not in stages_dict: stages_dict[st] = []
            stages_dict[st].append(idx)
        
        stages = sorted(stages_dict.keys())
        min_st = stages[0]  # El más pequeño disponible
        max_st = stages[-1] # El más grande disponible
        
        # Seleccionar fotos (con reemplazo por si hay pocas)
        try:
            s_sel = np.random.choice(stages_dict[min_st], n_shot, replace=True)
            q_sel = np.random.choice(stages_dict[max_st], n_query, replace=True)
        except: continue

        s_idxs.extend(s_sel); s_lbls.extend([i]*n_shot)
        q_idxs.extend(q_sel); q_lbls.extend([i]*n_query)
        
        base_name = dataset.samples[stages_dict[min_st][0]]['name']
        names.append(f"Fam. {base_name}")

    return MetaTask(get_batch_tensors(dataset, s_idxs, s_lbls),
                    get_batch_tensors(dataset, q_idxs, q_lbls),
                    "Evolución", names)

# ==================================================================================
# 5. EJECUCIÓN Y DEMOSTRACIÓN
# ==================================================================================
if __name__ == "__main__":
    CSV = "pokemon_data_gen1-5.csv"
    IMGS = "dataset_images" # Ajusta a tu ruta real
    
    if os.path.exists(CSV):
        print(">>> Iniciando Sistema de Meta-Learning Pokémon")
        
        # 1. Cargar Dataset
        ds = PokemonMetaDataset(CSV, IMGS, transform=preprocess)
        
        # 2. Generar los 3 Splits (Train, Val, Test)
        train_fams, val_fams, test_fams = get_train_val_test_splits(ds, SPLIT_CONFIG)
        
        # 3. Simular Entrenamiento
        print("\n=== [FASE 1] ENTRENAMIENTO (TRAIN) ===")
        # Aquí crearías tareas infinitamente para entrenar
        task = create_classification_task(ds, train_fams, n_way=5)
        if task: task.inspect()
        
        # 4. Simular Validación (Durante el entrenamiento)
        print("\n=== [FASE 2] VALIDACIÓN (VAL) ===")
        # Comprobamos qué tal va el modelo con Pokémon de validación
        # (Nota: Usamos 'create_evolution_task' para ver si generaliza la evolución)
        task_val = create_evolution_task(ds, val_fams, n_way=2) # 2 vías si hay pocos datos
        if task_val: task_val.inspect()
            
        # 5. Simular Test Final (Al terminar el proyecto)
        print("\n=== [FASE 3] TEST FINAL (TEST) ===")
        # Esto debería tener a los Pokémon específicos que pediste (Pikachu, Lucario, Gen 5...)
        task_test = create_classification_task(ds, test_fams, n_way=3)
        if task_test: task_test.inspect()
        
    else:
        print(f"Error: No se encontró {CSV}")