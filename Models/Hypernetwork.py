import torch
import torch.nn as nn
import torch.nn.functional as F

class HyperNetworkModel(nn.Module):
    def __init__(self, backbone, feature_dim=128, num_classes=5):
        """
        Args:
            backbone: Tu ConvBackbone (output size 128*5*5 = 3200 aplanado, o 128 si ya haces pool).
                      *OJO*: Tu ConvBackbone en Baseline.py hace flatten a 128 * 25 = 3200? 
                      No, en Baseline.py dice: x = x.view(x.size(0), -1).
                      Si el output es 5x5x128, feature_dim debe ser 3200.
            feature_dim: Dimensión del vector de características que sale del backbone.
            num_classes: N-Way (cuántas clases por episodio, usualmente 5).
        """
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        # HYPERNETWORK (Generador de Pesos)
        # Input: Un prototipo de clase (vector de características)
        # Output: Un vector de pesos para clasificar ESA clase.
        #         Queremos generar una fila de la matriz de pesos final.
        self.weight_generator = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim) # Generamos un peso del mismo tamaño que el input feature
        )
        
        # Bias generator (Opcional, pero recomendado para estabilidad)
        self.bias_generator = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Un escalar por clase
        )

    def forward(self, support_x, query_x, n_way, k_shot, q_query):
        """
        Args:
            support_x: Imágenes de soporte [N_Way * K_Shot, C, H, W]
            query_x: Imágenes de query [N_Way * Q_Query, C, H, W]
            n_way, k_shot, q_query: Enteros para hacer reshape
        """
        
        # 1. EXTRACT FEATURES
        # Pasamos TODO por el backbone
        # z_support shape: [N*K, Feature_Dim]
        z_support = self.backbone(support_x) 
        # z_query shape: [N*Q, Feature_Dim]
        z_query = self.backbone(query_x)     

        # 2. CALCULAR PROTOTIPOS (Input para la Hypernet)
        # Reshape para agrupar por clase: [N_Way, K_Shot, Feature_Dim]
        z_support_reshaped = z_support.view(n_way, k_shot, -1)
        
        # Promedio sobre los K-Shots -> [N_Way, Feature_Dim]
        prototypes = z_support_reshaped.mean(dim=1)

        # 3. GENERAR PESOS (Hypernetwork)
        # Pasamos los 5 prototipos a la vez. 
        # generated_weights shape: [N_Way, Feature_Dim] -> Esta es nuestra matriz W
        generated_weights = self.weight_generator(prototypes)
        
        # generated_bias shape: [N_Way, 1] -> Este es nuestro vector b
        generated_bias = self.bias_generator(prototypes).squeeze()

        # 4. CLASIFICACIÓN (Functional Forward)
        # Queremos hacer: Logits = Query @ Weights.T + Bias
        # Query: [N*Q, Feat_Dim]
        # Weights.T: [Feat_Dim, N_Way]
        
        logits = F.linear(z_query, generated_weights, generated_bias)
        # logits shape: [N*Q, N_Way] (Probabilidad de cada query de ser cada una de las 5 clases)

        return logits