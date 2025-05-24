import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    MessagePassing,
    GCNConv,
    GINConv,
    GINEConv,
    # GatedGCNConv, # PyG's GatedGCNConv is a bit different from paper's formula, often for node cls.
    # We might need a simpler GatedGCN or use NNConv/PNAConv for more complex edge handling
    # For simplicity, let's focus on GCN and GIN as base, GatedGCN can be tricky to generify here
    global_mean_pool,
    global_add_pool,
    global_max_pool
)
from torch_geometric.data import Data, Batch
from torch_geometric.utils import degree # For degree features if needed

# --- 1. Feed-Forward Network (FFN) Module ---
# As per paper's Eq (10): FFN(h) = BN(σ(hW_FFN1)W_FFN2 + h)
class FFN(nn.Module):
    def __init__(self, in_channels, hidden_channels_ffn_mult=2, dropout_ffn=0.1, activation_fn=nn.ReLU):
        super().__init__()
        # The paper suggests FFN enhances model's ability for complex feature transformations.
        # A common FFN structure is Linear -> Activation -> Dropout -> Linear
        # Their Eq 10: FFN(h) = BN(σ(hW_FFN1)W_FFN2 + h)
        # This means the output dimension of W_FFN2 must be in_channels for the residual.
        # Let's use a typical MLP structure for FFN, then add the residual and BN as per their formula.
        # The internal hidden dimension is often larger.
        ffn_hidden_dim = in_channels * hidden_channels_ffn_mult
        self.lin1 = nn.Linear(in_channels, ffn_hidden_dim)
        self.lin2 = nn.Linear(ffn_hidden_dim, in_channels) # Output dim matches input for residual
        self.activation = activation_fn()
        self.dropout = nn.Dropout(dropout_ffn)
        self.bn = nn.BatchNorm1d(in_channels) # BN on the final output

    def forward(self, h):
        h_res = h # For the final residual connection in FFN(h) = ... + h
        h = self.lin1(h)
        h = self.activation(h)
        h = self.dropout(h) # Dropout after activation
        h = self.lin2(h)
        # FFN(h) = BN(σ(hW_FFN1)W_FFN2 + h_res) ; where σ here is the activation in the FFN block
        # The paper's equation FFN(h) = BN(σ(hW_FFN1)W_FFN2 +h) implies hW_FFN1 is the input to sigma
        # and W_FFN2 operates on that, then residual.
        # Let's re-interpret their Eq 10 carefully: h_out = BN( activated_mlp_output + h_input_to_ffn )
        # The MLP part is σ(h_in W1)W2.
        # So:
        # h_ffn_out = self.lin2(self.dropout(self.activation(self.lin1(h_res)))) # MLP part
        # h_final = self.bn(h_ffn_out + h_res) # BN after residual sum
        # This is a common FFN structure (e.g., in Transformers).

        # Let's strictly follow Eq (10) from paper: FFN(h) = BN(σ(hW_FFN1)W_FFN2 + h)
        # Assume σ is applied after first linear, W_FFN2 is second linear
        x = self.lin1(h_res)
        x = self.activation(x) # σ(hW_FFN1)
        x = self.lin2(x)      # σ(hW_FFN1)W_FFN2
        x = x + h_res         # σ(hW_FFN1)W_FFN2 + h
        x = self.bn(x)
        return x

# --- 2. GNN+ Layer ---
class GNNPlusLayer(nn.Module):
    def __init__(self, in_channels, out_channels, base_gnn_layer: MessagePassing,
                 use_edge_features_in_base: bool = False, # Whether base_gnn takes edge_attr
                 dropout_rate=0.1, use_bn=True, use_residual=True, use_ffn=True,
                 hidden_channels_ffn_mult=2, dropout_ffn=0.1,
                 activation_fn=nn.ReLU):
        super().__init__()
        self.base_gnn_layer = base_gnn_layer
        self.use_edge_features_in_base = use_edge_features_in_base
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.use_ffn = use_ffn

        if self.use_bn:
            self.bn = nn.BatchNorm1d(out_channels)
        
        self.activation = activation_fn()
        self.dropout = nn.Dropout(dropout_rate)

        if self.use_residual:
            # If in_channels != out_channels, need a linear projection for residual
            if in_channels != out_channels:
                self.residual_proj = nn.Linear(in_channels, out_channels)
            else:
                self.residual_proj = nn.Identity()
        
        if self.use_ffn:
            self.ffn = FFN(out_channels, hidden_channels_ffn_mult, dropout_ffn, activation_fn)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        h_in = x # For residual connection

        # 1. Message Passing (with optional edge features)
        if self.use_edge_features_in_base and edge_attr is not None:
            # This assumes base_gnn_layer can take edge_attr (e.g., GINEConv, GatedGCNConv, NNConv)
            # For GINConv, edge_attr might be added to x before passing to GINConv's internal MLP
            # For GCNConv, standard PyG version doesn't take edge_attr for summation as in paper's Eq 6
            if isinstance(self.base_gnn_layer, GINConv):
                # GIN often adds edge features to node features (x_j + e_ij) before MLP
                # This is a simplification; proper GIN+edge might need specific handling
                # For now, assume GIN's nn handles x, and edge_attr is not directly passed here
                # Or if GIN is designed to take (x, edge_index, edge_features_for_mlp_on_neighbors)
                h = self.base_gnn_layer(x, edge_index) # GIN's nn handles x; how edge_attr is used depends on GIN variant
            else: # For GATConv, GINEConv, NNConv etc. that might take edge_attr
                h = self.base_gnn_layer(x, edge_index, edge_attr=edge_attr)
        else:
            h = self.base_gnn_layer(x, edge_index)

        # 2. Normalization (after GNN, before activation - paper Fig 2 & Eq 7)
        if self.use_bn:
            h = self.bn(h)
        
        # 3. Activation
        h = self.activation(h)

        # 4. Dropout (after activation - paper Fig 2 & Eq 8)
        h = self.dropout(h)

        # 5. Residual Connection (after dropout - paper Fig 2 & Eq 9)
        if self.use_residual:
            h = h + self.residual_proj(h_in)
            
        # 6. Feed-Forward Network (after residual - paper Fig 2 & Eq 11 uses FFN around previous GNN output)
        # Eq 11: h_l = FFN(Dropout(BN(GNN_AGG(...) + h_l-1)))
        # This means FFN is the outermost operation for the layer's node embeddings.
        if self.use_ffn:
            h = self.ffn(h)
            
        return h

# --- 3. GNN+ Model ---
class GNNPlusModel(nn.Module):
    def __init__(self, node_feature_dim: int, edge_feature_dim: int, pe_dim: int,
                 num_classes: int, num_layers: int, hidden_dim: int,
                 base_gnn_type: str = 'gin', # 'gcn', 'gin', 'gatedgcn' (gatedgcn needs more work)
                 use_edge_features: bool = True, # Global switch for edge features
                 dropout_gnn_layer: float = 0.1,
                 use_bn: bool = True,
                 use_residual: bool = True,
                 use_ffn_in_gnn: bool = True,
                 ffn_hidden_mult: int = 2,
                 dropout_ffn: float = 0.1,
                 use_pe: bool = True, # Whether to use positional encoding
                 activation_fn_name: str = 'relu',
                 pooling_type: str = 'mean',
                 dropout_readout: float = 0.5,
                 readout_hidden_dim_mult: int = 1):
        super().__init__()

        self.use_pe = use_pe
        self.use_edge_features = use_edge_features

        if activation_fn_name.lower() == 'relu':
            activation_creator = nn.ReLU
        elif activation_fn_name.lower() == 'elu':
            activation_creator = nn.ELU
        else:
            raise ValueError("Unsupported activation function")

        # 1. Input Embedding / Positional Encoding Projection (Paper Eq 12)
        # X_v_emb = [x_v || x_RWSE] W_PE
        current_input_dim = node_feature_dim
        if self.use_pe and pe_dim > 0:
            current_input_dim += pe_dim
        
        self.input_projection = nn.Linear(current_input_dim, hidden_dim)
        # Alternatively, separate projections then add/concat:
        # self.node_feat_proj = nn.Linear(node_feature_dim, hidden_dim)
        # if self.use_pe and pe_dim > 0:
        #     self.pe_proj = nn.Linear(pe_dim, hidden_dim)


        self.gnn_layers = nn.ModuleList()
        current_channels = hidden_dim # After input projection

        for i in range(num_layers):
            # --- Create Base GNN Layer ---
            if base_gnn_type.lower() == 'gcn':
                base_gnn = GCNConv(current_channels, hidden_dim)
                # Note: Standard GCNConv doesn't use edge_attr as per paper's Eq 6.
                # To truly follow paper's GCN+ with edge features, a custom GCNConv is needed.
                # Or use a layer like NNConv with GCN-like aggregation.
                # We'll set use_edge_features_in_base=False for standard GCNConv
                # unless it's a custom GCN that can handle it.
                can_base_use_edge = False
            elif base_gnn_type.lower() == 'gin':
                # GINConv requires an MLP for its transformation
                gin_mlp = nn.Sequential(
                    nn.Linear(current_channels, hidden_dim * 2), # Example MLP for GIN
                    activation_creator(),
                    nn.Linear(hidden_dim * 2, hidden_dim)
                )
                base_gnn = GINConv(nn=gin_mlp, train_eps=True) # train_eps from OGB GIN
                # GIN can be adapted for edge features by adding them to node features
                # before its internal MLP, or using GINEConv.
                # For now, let's assume it doesn't directly take edge_attr.
                can_base_use_edge = False # Unless it's GINEConv
            # elif base_gnn_type.lower() == 'gatedgcn':
            #     base_gnn = GatedGCNConv(current_channels, hidden_dim, edge_dim=edge_feature_dim if use_edge_features else 0)
            #     can_base_use_edge = True # GatedGCN can use edge_dim
            elif base_gnn_type.lower() == 'gine': # Using GINE as a good example for edge features
                 gine_mlp = nn.Sequential(
                    nn.Linear(current_channels, hidden_dim * 2),
                    activation_creator(),
                    nn.Linear(hidden_dim * 2, hidden_dim)
                )
                 base_gnn = GINEConv(nn=gine_mlp, train_eps=True, edge_dim=edge_feature_dim if use_edge_features else None)
                 can_base_use_edge = True
            else:
                raise ValueError(f"Unsupported base_gnn_type: {base_gnn_type}")

            gnn_plus_layer = GNNPlusLayer(
                in_channels=current_channels,
                out_channels=hidden_dim,
                base_gnn_layer=base_gnn,
                use_edge_features_in_base=can_base_use_edge and self.use_edge_features,
                dropout_rate=dropout_gnn_layer,
                use_bn=use_bn,
                use_residual=use_residual,
                use_ffn=use_ffn_in_gnn,
                hidden_channels_ffn_mult=ffn_hidden_mult,
                dropout_ffn=dropout_ffn,
                activation_fn=activation_creator
            )
            self.gnn_layers.append(gnn_plus_layer)
            current_channels = hidden_dim # Output of GNNPlusLayer is hidden_dim

        # 3. Graph Pooling
        if pooling_type.lower() == 'mean':
            self.pool = global_mean_pool
        elif pooling_type.lower() == 'add':
            self.pool = global_add_pool
        elif pooling_type.lower() == 'max':
            self.pool = global_max_pool
        else:
            raise ValueError(f"Unsupported pooling type: {pooling_type}")

        # 4. Readout MLP (Classifier)
        readout_input_dim = hidden_dim
        self.readout_mlp = nn.Sequential(
            nn.Linear(readout_input_dim, readout_input_dim * readout_hidden_dim_mult),
            activation_creator(),
            nn.Dropout(p=dropout_readout),
            nn.Linear(readout_input_dim * readout_hidden_dim_mult, num_classes)
        )

    def forward(self, data: Batch) -> torch.Tensor:
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        if x is None and not (self.use_pe and hasattr(data, 'pe') and data.pe is not None):
            raise ValueError("Node features 'x' are None and no PE is provided/used.")
        
        # Input projection / PE integration
        if self.use_pe and hasattr(data, 'pe') and data.pe is not None:
            if x is None: # Only PE is available
                x_processed = data.pe
            else: # Concatenate node features and PE
                x_processed = torch.cat([x, data.pe], dim=-1)
        else: # Only node features (or x is None but use_pe is False)
            x_processed = x
        
        if x_processed is None: # Should be caught by earlier check, but defensive
            raise ValueError("Effective input features are None after PE handling.")
            
        h = self.input_projection(x_processed.float()) # Ensure float
        h = F.relu(h) # Activation after initial projection

        # GNN layers
        for layer in self.gnn_layers:
            h = layer(h, edge_index, edge_attr if self.use_edge_features else None, batch)
            
        # Graph pooling
        h_graph = self.pool(h, batch)
        
        # Readout MLP
        out_logits = self.readout_mlp(h_graph)
        
        return out_logits