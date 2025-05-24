import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class SimpleGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate=0.2): # Reduced dropout for small data
        super().__init__()
        # If no node features (in_channels=0 or None), use an embedding for nodes
        # For now, assuming in_channels > 0 (actual node features)
        if in_channels <= 0: # Handle case with no initial features by using node degree or an embedding
            print("Warning: in_channels <=0. Consider using node embeddings or degrees as features.")
            # As a placeholder, let's assume we'll create a dummy feature if none.
            # This part needs to be adapted to your actual data.
            # If you have node degrees, that's a common GCN starting point.
            # Or you can create learnable embeddings per node if num_nodes is fixed,
            # but that's tricky for graph-batched data unless it's within the forward.
            # For simplicity, let's assume you will provide some `data.x`.
            # If not, this model needs adjustment.
            self.uses_dummy_features = True # Flag this
            self.dummy_feature_dim = hidden_channels # Create dummy features of this size
            self.conv1 = GCNConv(self.dummy_feature_dim, hidden_channels)
        else:
            self.uses_dummy_features = False
            self.conv1 = GCNConv(in_channels, hidden_channels)

        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout_rate = dropout_rate

    def forward(self, data): # PyG convention often passes the whole Data object
        x, edge_index, batch = data.x, data.edge_index, data.batch

        if self.uses_dummy_features and x is None:
            # Create dummy features if no 'x' and configured to do so.
            # This is a basic way; a learnable embedding per node would be better
            # if node identities are consistent and meaningful across graphs.
            # For now, let's use a constant feature for all nodes if x is missing.
            # This isn't ideal but makes the GCN runnable.
            # A better approach for no features is to use torch.eye(data.num_nodes)
            # but that's for single graphs, not batches easily.
            # Or use node degrees.
            # For now, let's just illustrate the GCN structure.
            # YOU WILL LIKELY NEED TO ADJUST FEATURE HANDLING HERE.
            # Example: x = torch.ones((data.num_nodes, self.dummy_feature_dim), device=edge_index.device)
            pass # This part highly depends on how you want to handle no features

        if x is None:
            raise ValueError("Node features 'x' are None. The model needs node features or specific handling for featureless graphs.")


        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)

        # If graph classification, add pooling:
        from torch_geometric.nn import global_mean_pool # Or global_add_pool, etc.
        x_pooled = global_mean_pool(x, batch) # `batch` vector from DataLoader is crucial
        return x_pooled # Logits for graph classification


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.data import Data

class GINEGraphClassifier(nn.Module):
    def __init__(self, node_in_channels: int, edge_in_channels: int,
                 hidden_channels: int, out_channels: int,
                 num_gine_layers: int = 2,
                 dropout_gine: float = 0.5, # Dropout after GINE layers' activation
                 dropout_mlp: float = 0.5,  # Dropout in the final MLP
                 pooling_type: str = 'mean', # 'mean', 'add', or 'max'
                 eps: float = 0.,            # Epsilon for GIN (0 for GIN, learnable for GINE)
                 train_eps: bool = False     # Whether epsilon is learnable
                ):
        super(GINEGraphClassifier, self).__init__()

        if num_gine_layers < 1:
            raise ValueError("Number of GINE layers must be at least 1.")

        self.num_gine_layers = num_gine_layers
        self.dropout_gine = dropout_gine
        self.dropout_mlp = dropout_mlp
        self.pooling_type = pooling_type

        self.gine_layers = nn.ModuleList()

        # The MLP for GINEConv processes node features.
        # Edge features are summed with node features (after an optional MLP on edge features if you want,
        # but GINEConv's internal nn processes the combination).
        # GINEConv: x_j + edge_attr_ji passed through an MLP.
        # Output dim of nn inside GINEConv should match hidden_channels.
        
        current_dim = node_in_channels

        for i in range(num_gine_layers):
            # The MLP inside GINEConv maps from current_dim to hidden_channels
            # This MLP is applied to x_i (node features of the central node)
            # and to the aggregated messages (which are transformations of x_j + edge_attr_ji)
            # So, the `nn` should map `current_dim` to `hidden_channels`.
            # And `edge_dim` is the dimensionality of your `edge_attr`.
            
            # Define the MLP for the GINEConv layer
            # This MLP processes the node features (x_i) and the aggregated messages
            # Its input dimension should be `current_dim` and output `hidden_channels`
            layer_nn = nn.Sequential(
                nn.Linear(current_dim, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )
            
            self.gine_layers.append(
                GINEConv(nn=layer_nn, eps=eps, train_eps=train_eps, edge_dim=edge_in_channels)
            )


            current_dim = hidden_channels # Output of GINE is hidden_channels

        self.gine_output_dim = current_dim # Dimension before pooling

        # Pooling layer
        if pooling_type == 'mean':
            self.pool = global_mean_pool
        elif pooling_type == 'add':
            self.pool = global_add_pool
        elif pooling_type == 'max':
            self.pool = global_max_pool
        else:
            raise ValueError(f"Unsupported pooling type: {pooling_type}")

        # Classifier MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.gine_output_dim, hidden_channels // 2 if hidden_channels // 2 > 0 else 1),
            nn.ReLU(),
            nn.Dropout(p=dropout_mlp),
            nn.Linear(hidden_channels // 2 if hidden_channels // 2 > 0 else 1, out_channels)
        )

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        if x is None:
            raise ValueError("Node features 'x' are None. Model requires node features.")
        if x.dtype == torch.long:
            x = x.float()
        
        if edge_attr is None:
            raise ValueError("Edge attributes 'edge_attr' are None. GINEConv requires edge_attr.")
        if edge_attr.dtype == torch.long: # Edge attributes are typically float
            edge_attr = edge_attr.float()


        # Pass through GINE layers
        for gine_layer in self.gine_layers:
            x = gine_layer(x, edge_index, edge_attr=edge_attr)
            x = F.relu(x) # GINE paper often uses ReLU after the layer's MLP and aggregation
            x = F.dropout(x, p=self.dropout_gine, training=self.training)

        # Graph pooling
        if batch is None: # Handle single graph case
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        x_pooled = self.pool(x, batch)

        # Classification
        out_logits = self.mlp(x_pooled)

        return out_logits
    

import torch
import torch.nn.functional as F
from torch.nn import Embedding, Linear
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

class GATGraphClassifier(torch.nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, out_channels,
                 edge_feature_dim, heads=8, dropout=0.6, pooling_type='mean'):
        super(GATGraphClassifier, self).__init__()
        self.node_feature_dim = node_feature_dim
        self.dropout = dropout
        self.heads = heads
        self.pooling_type = pooling_type


        self.conv1 = GATConv(node_feature_dim, hidden_dim, heads=heads,edge_dim=edge_feature_dim, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, edge_dim=edge_feature_dim, dropout=dropout)


        # Classifier MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2 if hidden_dim // 2 > 0 else 1),   
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2 if hidden_dim // 2 > 0 else 1, out_channels)
        )

    def forward(self, data: Data) -> torch.Tensor:

        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = x.float()

        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)

        if self.pooling_type == "mean":
            x_graph = global_mean_pool(x, batch)  # x shape: [total_nodes, features_after_conv2], batch shape: [total_nodes]
        elif self.pooling_type == "add":
            x_graph = global_add_pool(x, batch)
        elif self.pooling_type == "max":
            x_graph = global_max_pool(x, batch)
        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling_type}")
        
        x = self.mlp(x_graph)
        return x
