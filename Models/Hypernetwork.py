# se usará el mismo encoder (backbone) que para los otros modelos

import torch
import torch.nn as nn

class HyperNetworkModel(nn.Module):
    def __init__(self, backbone, classifier_input_dim, num_classes_per_episode):
        super().__init__()
        
        # 1. Conectamos el encoder (backbone)
        self.backbone = backbone #! comprobar que sirve la que tenemos 
                                 #! mirar tamaño vectores
        
        # Para que no se re-entrene el encoder si queremos congelarlo (Opcional)
        # for param in self.backbone.parameters():
        #     param.requires_grad = False
        
        # 2. La red que genera los pesos (Hypernetwork)
        # Transforma las features del support set en pesos para el clasificador
        self.weight_generator = nn.Sequential( #! asi nos sirve?
            nn.Linear(classifier_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, classifier_input_dim * num_classes_per_episode) 
            # Salida = tamaño de la matriz de pesos necesaria
        )

        self.num_classes = num_classes_per_episode
        self.feat_dim = classifier_input_dim
        
    def forward(self, support_images, query_images):
        """
        support_images: Las 5 fotos del Pokémon nuevo (K-shot) [cite: 105]
        query_images: La foto que queremos clasificar
        """
        
        # 1. Extraer features de las imagenes de soporte
        support_features = self.backbone(support_images) 
        # Promediamos las features de las 5 fotos para tener un "prototipo"
        prototype = support_features.mean(dim=0) 
        
        # 2. La Hypernetwork genera los pesos específicos para este episodio
        generated_weights = self.weight_generator(prototype)
        
        # 3. Formatear los pesos para usarlos (N-way, Input_dim) #! que es esto bien bien?
        weights_matrix = generated_weights.view(-1, support_features.size(1))
        
        # 4. Clasificar la imagen de Query usando esos pesos dinámicos
        query_features = self.backbone(query_images)
        
        # Simulación de capa lineal manual: input * weights_transposed
        logits = torch.matmul(query_features, weights_matrix.t())
        
        return logits