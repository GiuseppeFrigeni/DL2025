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
                 train_eps: bool = False,    # Whether epsilon is learnable
                 use_batch_norm: bool = True, # Whether to use BatchNorm after GINE layers
                ):
        super(GINEGraphClassifier, self).__init__()

        if num_gine_layers < 1:
            raise ValueError("Number of GINE layers must be at least 1.")

        self.num_gine_layers = num_gine_layers
        self.dropout_gine = dropout_gine
        self.dropout_mlp = dropout_mlp
        self.pooling_type = pooling_type
        self.use_batch_norm = use_batch_norm

        self.gine_layers = nn.ModuleList()
        if use_batch_norm:
            self.bn_gine = nn.ModuleList() # Optional: BatchNorm for GINE layers, if needed

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
                nn.Linear(hidden_channels, hidden_channels),
            )
            
            self.gine_layers.append(
                GINEConv(nn=layer_nn, eps=eps, train_eps=train_eps, edge_dim=edge_in_channels)
            )
            if use_batch_norm:
                self.bn_gine.append(nn.BatchNorm1d(hidden_channels)) # Optional: BatchNorm after each GINE layer


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

        x_current = x

        x_after_first_gine = self.gine_layers[0](x_current, edge_index, edge_attr=edge_attr)
        if self.use_batch_norm: x_after_first_gine = self.bn_gine[0](x_after_first_gine) if x_after_first_gine.size(0) > 0 else x_after_first_gine
        x_current = F.relu(x_after_first_gine)
        x_current = F.dropout(x_current, p=self.dropout_gine, training=self.training)

        for i in range(1, self.num_gine_layers):
            x_identity = x_current
            # 1. Apply GINEConv
            x_after_gine = self.gine_layers[i](x_current, edge_index, edge_attr=edge_attr)

            # 2. Apply BatchNorm1d (if enabled and tensor is not empty)
            if self.use_batch_norm:
                if x_after_gine.size(0) > 0: # BN1d requires non-empty input
                    x_after_bn = self.bn_gine[i](x_after_gine)
                else:
                    x_after_bn = x_after_gine # Pass through if empty
            else:
                x_after_bn = x_after_gine # Skip BN if not enabled
            # 3. Apply ReLU
            x_after_relu = F.relu(x_after_bn)

            # 4. Apply Dropout
            x_processed = F.dropout(x_after_relu, p=self.dropout_gine, training=self.training)

            x_current = x_identity + x_processed

        # Final GNN features before pooling
        x_gnn_out = x_current

        # Graph pooling
        if batch is None: # Handle single graph case not using DataLoader's batching
            if x_gnn_out.size(0) > 0:
                batch = torch.zeros(x_gnn_out.size(0), dtype=torch.long, device=x_gnn_out.device)
            else: # Single graph has 0 nodes
                # Pooled output should be zeros of appropriate dimension
                num_graphs_for_pooling = 1
                x_pooled = torch.zeros((num_graphs_for_pooling, self.gine_output_dim), device=x_gnn_out.device if x_gnn_out.device else 'cpu')
                out_logits = self.mlp(x_pooled)
                return out_logits # Early exit for this specific case

        # Handle case where x_gnn_out might be empty (e.g., a batch of empty graphs)
        if x_gnn_out.size(0) == 0:
            # This assumes 'batch' correctly identifies the number of graphs.
            # If batch is also empty or problematic, this needs more robust handling.
            num_graphs_in_batch = batch.max().item() + 1 if batch is not None and batch.numel() > 0 else 0
            if num_graphs_in_batch > 0:
                 x_pooled = torch.zeros((num_graphs_in_batch, self.gine_output_dim), device=x_gnn_out.device if x_gnn_out.device else 'cpu')
            else: # No graphs in batch, or batch is ill-defined for empty x
                 # This case should ideally not happen with a standard DataLoader.
                 # Return empty or appropriately shaped zero tensor if necessary.
                 return torch.empty((0, self.mlp[-1].out_features), device=x_gnn_out.device if x_gnn_out.device else 'cpu')
        else:
            x_pooled = self.pool(x_gnn_out, batch)


        # Classification
        out_logits = self.mlp(x_pooled)

        return out_logits
    


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data
import math

class AttentionPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim // 2
        
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x, batch):
        # Compute attention weights
        attention_weights = self.attention(x)  # [num_nodes, 1]
        attention_weights = torch.softmax(attention_weights, dim=0)
        
        # Apply attention-weighted pooling
        weighted_x = x * attention_weights
        return global_add_pool(weighted_x, batch)

class EnhancedGINEGraphClassifier(nn.Module):
    def __init__(self, node_in_channels: int, edge_in_channels: int,
                 hidden_channels: int, out_channels: int,
                 num_gine_layers: int = 2,
                 dropout_gine: float = 0.3,  # Reduced dropout
                 dropout_mlp: float = 0.5,
                 pooling_type: str = 'attention',  # 'mean', 'add', 'max', 'attention', 'multi'
                 eps: float = 0.,
                 train_eps: bool = True,  # Enable learnable eps
                 use_batch_norm: bool = True,
                 use_layer_norm: bool = False,  # Alternative to batch norm
                 use_residual: bool = True,
                 edge_mlp_hidden: int = None,  # Hidden dim for edge MLP
                 virtual_node: bool = False,  # Whether to use a virtual node
                 ):
        super(EnhancedGINEGraphClassifier, self).__init__()

        if num_gine_layers < 1:
            raise ValueError("Number of GINE layers must be at least 1.")

        self.num_gine_layers = num_gine_layers
        self.dropout_gine = dropout_gine
        self.dropout_mlp = dropout_mlp
        self.pooling_type = pooling_type
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        self.virtual_node = virtual_node
        
        # Edge feature preprocessing
        if edge_mlp_hidden is None:
            edge_mlp_hidden = max(edge_in_channels * 2, hidden_channels)
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in_channels, edge_mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(edge_mlp_hidden, edge_in_channels)
        )

        self.gine_layers = nn.ModuleList()
        if use_batch_norm:
            self.bn_gine = nn.ModuleList()
        if use_layer_norm:
            self.ln_gine = nn.ModuleList()

        current_dim = node_in_channels

        for i in range(num_gine_layers):
            # More sophisticated MLP for GINEConv
            layer_nn = nn.Sequential(
                nn.Linear(current_dim, hidden_channels),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
            
            self.gine_layers.append(
                GINEConv(nn=layer_nn, eps=eps, train_eps=train_eps, edge_dim=edge_in_channels)
            )
            
            if use_batch_norm:
                self.bn_gine.append(nn.BatchNorm1d(hidden_channels))
            if use_layer_norm:
                self.ln_gine.append(nn.LayerNorm(hidden_channels))

            current_dim = hidden_channels

        self.gine_output_dim = current_dim

        # Multiple pooling strategies
        if pooling_type == 'mean':
            self.pool = global_mean_pool
        elif pooling_type == 'add':
            self.pool = global_add_pool
        elif pooling_type == 'max':
            self.pool = global_max_pool
        elif pooling_type == 'attention':
            self.pool = AttentionPooling(self.gine_output_dim)
        elif pooling_type == 'multi':
            self.pool_mean = global_mean_pool
            self.pool_max = global_max_pool
            self.pool_add = global_add_pool
            self.gine_output_dim = self.gine_output_dim * 3  # Concatenate all three
        else:
            raise ValueError(f"Unsupported pooling type: {pooling_type}")

        # Enhanced classifier with residual connections - optimized for noisy labels
        mlp_layers = []
        current_mlp_dim = self.gine_output_dim
        
        # First layer - wider for robustness
        mlp_layers.extend([
            nn.Linear(current_mlp_dim, hidden_channels * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channels * 2),
            nn.Dropout(dropout_mlp * 0.3)  # Lower dropout early
        ])
        current_mlp_dim = hidden_channels * 2
        
        # Second layer
        mlp_layers.extend([
            nn.Linear(current_mlp_dim, hidden_channels),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channels),
            nn.Dropout(dropout_mlp * 0.5)
        ])
        current_mlp_dim = hidden_channels
        
        # Third layer - bottleneck before output
        mlp_layers.extend([
            nn.Linear(current_mlp_dim, hidden_channels // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.Dropout(dropout_mlp)  # Higher dropout before output
        ])
        current_mlp_dim = hidden_channels // 2
        
        # Output layer with temperature scaling capability
        mlp_layers.append(nn.Linear(current_mlp_dim, out_channels))
        
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Temperature parameter for calibration
        self.temperature = nn.Parameter(torch.ones(1))
        
        # Jump connections: direct connections from intermediate layers to output
        self.jump_connections = nn.ModuleList()
        for i in range(num_gine_layers):
            self.jump_connections.append(
                nn.Linear(hidden_channels, hidden_channels // 4)
            )
        
        # Final combination layer for jump connections
        if num_gine_layers > 1:
            self.jump_combiner = nn.Linear(
                self.gine_output_dim + (hidden_channels // 4) * (num_gine_layers - 1),
                self.gine_output_dim
            )

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        if x is None:
            raise ValueError("Node features 'x' are None.")
        if x.dtype == torch.long:
            x = x.float()
        
        if edge_attr is None:
            raise ValueError("Edge attributes 'edge_attr' are None.")
        if edge_attr.dtype == torch.long:
            edge_attr = edge_attr.float()

        # Preprocess edge features
        edge_attr = self.edge_mlp(edge_attr)

        x_current = x
        jump_features = []

        # First layer (no residual)
        x_after_gine = self.gine_layers[0](x_current, edge_index, edge_attr=edge_attr)
        x_after_gine = self._apply_normalization(x_after_gine, 0)
        x_current = F.relu(x_after_gine)
        x_current = F.dropout(x_current, p=self.dropout_gine, training=self.training)

        # Subsequent layers with optional residual connections
        for i in range(1, self.num_gine_layers):
            x_identity = x_current
            
            # GINE layer
            x_after_gine = self.gine_layers[i](x_current, edge_index, edge_attr=edge_attr)
            x_after_gine = self._apply_normalization(x_after_gine, i)
            x_after_relu = F.relu(x_after_gine)
            x_processed = F.dropout(x_after_relu, p=self.dropout_gine, training=self.training)

            # Residual connection
            if self.use_residual:
                x_current = x_identity + x_processed
            else:
                x_current = x_processed
            
            # Store for jump connections (except the last layer)
            if i < self.num_gine_layers - 1:
                jump_features.append(x_current)

        # Apply jump connections
        if len(jump_features) > 0:
            jump_outputs = []
            for i, jump_feat in enumerate(jump_features):
                jump_out = self.jump_connections[i](jump_feat)
                jump_out = global_mean_pool(jump_out, batch) if hasattr(self, 'pool') else jump_out.mean(0, keepdim=True)
                jump_outputs.append(jump_out)
            
            # Pool final layer
            if self.pooling_type == 'multi':
                x_pooled_final = torch.cat([
                    self.pool_mean(x_current, batch),
                    self.pool_max(x_current, batch), 
                    self.pool_add(x_current, batch)
                ], dim=1)
            else:
                x_pooled_final = self.pool(x_current, batch)
            
            # Combine with jump connections
            x_combined = torch.cat([x_pooled_final] + jump_outputs, dim=1)
            x_pooled = self.jump_combiner(x_combined)
        else:
            # Standard pooling
            if self.pooling_type == 'multi':
                x_pooled = torch.cat([
                    self.pool_mean(x_current, batch),
                    self.pool_max(x_current, batch),
                    self.pool_add(x_current, batch)
                ], dim=1)
            else:
                x_pooled = self.pool(x_current, batch)

        # Handle empty graphs
        if x_pooled.size(0) == 0:
            return torch.empty((0, self.mlp[-1].out_features), device=x.device)

        # Classification with temperature scaling
        raw_logits = self.mlp(x_pooled)
        out_logits = raw_logits / self.temperature
        return out_logits
    
    def _apply_normalization(self, x, layer_idx):
        """Apply normalization based on configuration"""
        if x.size(0) == 0:
            return x
            
        if self.use_batch_norm:
            x = self.bn_gine[layer_idx](x)
        if self.use_layer_norm:
            x = self.ln_gine[layer_idx](x)
        return x
    

from torch_geometric.nn import NNConv, global_mean_pool # or other pooling

class NNConvNet(torch.nn.Module):
    def __init__(self, node_in_channels, edge_feature_dim, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        # MLP that maps edge features to the weights of the NNConv
        # Output dim: node_in_channels * hidden_channels for the first layer
        # Output dim: hidden_channels * hidden_channels for subsequent layers
        
        edge_nn1 = torch.nn.Sequential(
            torch.nn.Linear(edge_feature_dim, hidden_channels * node_in_channels), # Or some other size
        )
        self.convs.append(NNConv(node_in_channels, hidden_channels, nn=edge_nn1, aggr='mean')) # or 'add', 'max'
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
        current_dim = hidden_channels
        for _ in range(num_layers - 1):
            edge_nni = torch.nn.Sequential(
                torch.nn.Linear(edge_feature_dim, current_dim * hidden_channels),
            )
            self.convs.append(NNConv(current_dim, hidden_channels, nn=edge_nni, aggr='mean'))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            current_dim = hidden_channels
    
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels // 2, out_channels)
        )
        self.dropout_val = dropout

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        if x.dtype == torch.long: x = x.float()
        if edge_attr.dtype == torch.long: edge_attr = edge_attr.float()

        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index, edge_attr)
            if x.numel() > 0:
                x = self.bns[i](x)
            x = F.relu(x) # Or another activation
            x = F.dropout(x, p=self.dropout_val, training=self.training)
    
        if x.numel() == 0:
             num_graphs_in_batch = batch.max().item() + 1 if batch is not None and batch.numel() > 0 else 0
             if num_graphs_in_batch > 0:
                 return torch.zeros((num_graphs_in_batch, self.mlp[-1].out_features), device=x.device)
             else:
                 return torch.empty((0, self.mlp[-1].out_features), device=x.device)

        x_pooled = global_mean_pool(x, batch)
        return self.mlp(x_pooled)