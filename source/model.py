
import torch
import torch.nn.functional as F    
from torch_geometric.nn import NNConv, global_mean_pool # or other pooling

class NNConvNet(torch.nn.Module):
    def __init__(self, node_in_channels, edge_feature_dim, 
                 out_channels_gnn, # Dimension of GNN output embeddings (Z_B)
                 hidden_dim_edge_nn, # Hidden dim for the NN in NNConv
                 mlp_hidden_dim_factor, # Factor to determine MLP hidden layer size (e.g., 0.5 for out_channels_gnn // 2)
                 out_channels_final, # Final number of classes
                 num_layers=2, dropout_rate=0.5,
                 return_embeddings=False): # New flag
        super().__init__()
        self.return_embeddings = return_embeddings
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        # Example: Assuming out_channels_gnn is the output dim of each NNConv layer
        # and also the input dim to the final MLP.
        
        # Edge NN for the first layer
        edge_nn1 = torch.nn.Sequential(
            torch.nn.Linear(edge_feature_dim, hidden_dim_edge_nn),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim_edge_nn, node_in_channels * out_channels_gnn) # Output to match NNConv
        )
        self.convs.append(NNConv(node_in_channels, out_channels_gnn, nn=edge_nn1, aggr='mean'))
        self.bns.append(torch.nn.BatchNorm1d(out_channels_gnn))
        
        current_dim_gnn = out_channels_gnn
        for _ in range(num_layers - 1):
            edge_nni = torch.nn.Sequential(
                torch.nn.Linear(edge_feature_dim, hidden_dim_edge_nn),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim_edge_nn, current_dim_gnn * out_channels_gnn)
            )
            self.convs.append(NNConv(current_dim_gnn, out_channels_gnn, nn=edge_nni, aggr='mean'))
            self.bns.append(torch.nn.BatchNorm1d(out_channels_gnn))
            current_dim_gnn = out_channels_gnn # current_dim_gnn will be out_channels_gnn after loop
    
        # MLP classifier
        mlp_hidden_dim = int(out_channels_gnn * mlp_hidden_dim_factor)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(out_channels_gnn, mlp_hidden_dim), # Input is GNN output embedding dim
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(mlp_hidden_dim, out_channels_final) # Output final classes
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        if x is not None and x.dtype == torch.long: x = x.float()
        if edge_attr is not None and edge_attr.dtype == torch.long: edge_attr = edge_attr.float()

        # GNN layers
        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index, edge_attr)
            if x.numel() > 0:
                x = self.bns[i](x)
            x = F.relu(x)

    
        if x.numel() == 0: # Handle empty graphs
            num_graphs_in_batch = batch.max().item() + 1 if batch is not None and batch.numel() > 0 else 0
            empty_logits = torch.empty((num_graphs_in_batch, self.mlp[-1].out_features), device=data.x.device if data.x is not None else torch.device('cpu'))
            empty_embeddings = torch.empty((num_graphs_in_batch, self.convs[-1].out_channels), device=data.x.device if data.x is not None else torch.device('cpu')) # Use out_channels of last conv
            if self.return_embeddings:
                return empty_logits, empty_embeddings
            else:
                return empty_logits

        # Pooling to get graph embeddings (Z_B)
        x_pooled = global_mean_pool(x, batch) # This is Z_B
        
        # Final MLP for classification
        logits = self.mlp(x_pooled)
        
        if self.return_embeddings:
            return logits, x_pooled
        else:
            return logits
        
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool, global_max_pool

class GINENetForGCOD(nn.Module):
    def __init__(self, node_in_channels,
                 edge_feature_dim,       # New: Dimension of edge features
                 gnn_hidden_channels,    # Dimension of GINE hidden layers AND output embeddings (Z_B)
                 num_gnn_layers,
                 mlp_hidden_channels,    # Hidden dimension for the MLP classifier
                 out_channels_final,     # Final number of classes
                 dropout_rate=0.5,
                 return_embeddings=False,
                 pooling_type='mean',    # 'mean', 'add', 'max'
                 eps=0.,                 # Epsilon for GINEConv (like in GIN)
                 train_eps=False):       # Whether epsilon is learnable
        super(GINENetForGCOD, self).__init__()

        self.node_in_channels = node_in_channels
        self.edge_feature_dim = edge_feature_dim
        self.gnn_hidden_channels = gnn_hidden_channels
        self.num_gnn_layers = num_gnn_layers
        self.mlp_hidden_channels = mlp_hidden_channels
        self.out_channels_final = out_channels_final
        self.dropout_rate = dropout_rate
        self.return_embeddings = return_embeddings
        self.pooling_type = pooling_type

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # Input layer for GINEConv
        # The MLP for GINEConv's `nn` argument.
        # Input to this MLP is node_in_channels (after aggregation with edge features implicitly handled by GINEConv)
        # GINEConv's internal aggregation: (1+eps)*x_i + sum(relu(x_j + e_ji))
        # The `nn` processes the result of this aggregation.
        # So, the input dimension to the nn.Linear inside GINE's nn should be node_in_channels
        # if edge_dim is not explicitly used to change the input size to nn.
        # GINEConv's nn input dimension is the same as its input node feature dimension.

        # First GINE layer
        self.convs.append(GINEConv(
            nn.Sequential(
                nn.Linear(node_in_channels, gnn_hidden_channels),
                nn.ReLU(),
                nn.Linear(gnn_hidden_channels, gnn_hidden_channels),
                nn.ReLU()
            ),
            eps=eps,
            train_eps=train_eps,
            edge_dim=self.edge_feature_dim # Specify edge feature dimension
        ))
        self.batch_norms.append(nn.BatchNorm1d(gnn_hidden_channels))

        # Hidden GINE layers
        for _ in range(num_gnn_layers - 1):
            self.convs.append(GINEConv(
                nn.Sequential(
                    nn.Linear(gnn_hidden_channels, gnn_hidden_channels), # Input is previous GINE output
                    nn.ReLU(),
                    nn.Linear(gnn_hidden_channels, gnn_hidden_channels),
                    nn.ReLU()
                ),
                eps=eps,
                train_eps=train_eps,
                edge_dim=self.edge_feature_dim
            ))
            self.batch_norms.append(nn.BatchNorm1d(gnn_hidden_channels))

        # MLP classifier
        self.mlp_classifier = nn.Sequential(
            nn.Linear(gnn_hidden_channels, mlp_hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_channels, out_channels_final)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        if x is None: # Should not happen given your feature engineering
            x = torch.ones((data.num_nodes, self.node_in_channels), device=edge_index.device)
        if edge_attr is None: # GINEConv requires edge_attr
             # Create dummy edge features if they are missing, or raise error
            if self.edge_feature_dim > 0:
                # This indicates an issue if edge_attr is expected but not provided
                # For now, let's create zeros. In a real scenario, this should be handled.
                print(f"Warning: GINENetForGCOD expects edge_attr but received None. Using zeros for {self.edge_feature_dim} edge features.")
                edge_attr = torch.zeros((edge_index.size(1), self.edge_feature_dim), device=edge_index.device)
            # If self.edge_feature_dim is 0, then GINE is essentially GIN, but GINEConv still expects edge_dim.
            # It's better to use GINConv if edge_feature_dim is truly 0.

        if x.dtype == torch.long: x = x.float()
        if edge_attr.dtype == torch.long: edge_attr = edge_attr.float()


        # GINE layers
        for i in range(self.num_gnn_layers):
            x = self.convs[i](x, edge_index, edge_attr=edge_attr) # Pass edge_attr
            if x.numel() > 0:
                 x = self.batch_norms[i](x)
            x = F.relu(x)
            # Optional dropout
            # x = F.dropout(x, p=self.dropout_rate, training=self.training)

        if x.numel() == 0:
            num_graphs_in_batch = batch.max().item() + 1 if batch is not None and batch.numel() > 0 else 0
            empty_logits = torch.empty((num_graphs_in_batch, self.out_channels_final), device=edge_index.device if edge_index is not None else torch.device('cpu'))
            empty_embeddings = torch.empty((num_graphs_in_batch, self.gnn_hidden_channels), device=edge_index.device if edge_index is not None else torch.device('cpu'))
            if self.return_embeddings:
                return empty_logits, empty_embeddings
            else:
                return empty_logits

        # Readout/Pooling
        if self.pooling_type == 'mean':
            x_pooled = global_mean_pool(x, batch)
        elif self.pooling_type == 'add':
            x_pooled = global_add_pool(x, batch)
        elif self.pooling_type == 'max':
            x_pooled = global_max_pool(x, batch)
        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling_type}")

        logits = self.mlp_classifier(x_pooled)

        if self.return_embeddings:
            return logits, x_pooled
        else:
            return logits