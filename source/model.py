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
    
