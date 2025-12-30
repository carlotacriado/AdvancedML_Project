import torch
import torch.nn as nn
import torch.nn.functional as F

class HyperNetworkModel(nn.Module):
    """
        Hypernetwork for Few-Shot Classification.
    """
    def __init__(self, backbone, feature_dim=128, num_classes=5):
        super().__init__()
        
        # Components
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        
        self.weight_generator = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim) 
        
        # Bias generator, to stabilize 
        self.bias_generator = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # one scalar bias per class
        )

    def forward(self, support_x, query_x, n_way, k_shot, q_query):
        """
            support_x: [N_Way * K_Shot, C, H, W]
            query_x:   [N_Way * Q_Query, C, H, W]
            n_way: number of classes
            k_shot: number of support examples per class
        """
        
        # 1. EXTRACT FEATURES
        # We pass all through the backbone
        # z_support shape: [N*K, Feature_Dim]
        z_support = self.backbone(support_x) 
        # z_query shape: [N*Q, Feature_Dim]
        z_query = self.backbone(query_x)     

        # 2. CALCULATE PROTOTYPES (Input para la Hypernet)
        # Reshape to group by class: [N_Way, K_Shot, Feature_Dim]
        z_support_reshaped = z_support.view(n_way, k_shot, -1)
        
        # Ponderated average over the K-Shots -> [N_Way, Feature_Dim]
        prototypes = z_support_reshaped.mean(dim=1)

        # 3. GENERATE WEIGHTS (Hypernetwork)
        # We pass the 5 prototypes at once. 
        # generated_weights shape: [N_Way, Feature_Dim] -> This is our W matrix
        generated_weights = self.weight_generator(prototypes)
        
        # generated_bias shape: [N_Way, 1] -> This is our b vector
        generated_bias = self.bias_generator(prototypes).squeeze()

        # 4. CLASSIFICATION (Functional Forward)
        # We want to do: Logits = Query @ Weights.T + Bias
        # Query: [N*Q, Feat_Dim]
        # Weights.T: [Feat_Dim, N_Way]
        
        logits = F.linear(z_query, generated_weights, generated_bias)
        # logits shape: [N*Q, N_Way] ( Probability of each query being each of the 5 classes)

        return logits