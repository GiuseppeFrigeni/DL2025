
import torch
import torch.nn.functional as F    
from torch_geometric.nn import NNConv, global_mean_pool # or other pooling

class NNConvNet(torch.nn.Module):
    def __init__(self, node_in_channels, edge_feature_dim,hidden_dim, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        
        edge_nn1 = torch.nn.Sequential(
            torch.nn.Linear(edge_feature_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_channels * node_in_channels) # Or some other size
        )
        self.convs.append(NNConv(node_in_channels, hidden_channels, nn=edge_nn1, aggr='mean')) # or 'add', 'max'
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
        current_dim = hidden_channels
        for _ in range(num_layers - 1):
            edge_nni = torch.nn.Sequential(
            torch.nn.Linear(edge_feature_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_channels * current_dim)
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

        edge_attr = F.dropout(edge_attr, p=0.2, training=self.training)

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