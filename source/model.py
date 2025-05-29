
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