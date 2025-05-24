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


from torch_geometric.utils import remove_self_loops
from torch_geometric.nn import MessagePassing
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool


class GIN_Conv(MessagePassing):
    def __init__(self, MLP, eps = 0.0):
        super().__init__(aggr='add')  # Aggregation function over the messages.
        self.mlp = MLP
        self.epsilon = torch.nn.Parameter(torch.tensor([eps]))

    def message(self, x_j):
      return x_j

    def update(self,aggr_out,x):
      x = (1+self.epsilon) * x + aggr_out
      return self.mlp(x)

    def forward(self, x, edge_index):
      #TODO
      # Step 1: remove self-loops to the adjacency matrix.
      edge_index, _ = remove_self_loops(edge_index)
      # Step 2: Start propagating messages.
      return self.propagate(edge_index, x=x)
    
class MLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MLP, self).__init__()
        self.first_fc = Linear(in_dim, hidden_dim)
        self.second_fc = Linear(hidden_dim, out_dim)
        self.activation = torch.nn.ReLU()

        # You could use torch.nn.Sequential

    def forward(self, x):
        x = self.activation(self.first_fc(x))
        x = self.activation(self.second_fc(x))

        return x

class Graph_Net(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes):
        super(Graph_Net, self).__init__()
        self.mlp_input =  MLP(in_dim, hidden_dim, hidden_dim)
        self.mlp_hidden =  MLP(hidden_dim, hidden_dim, hidden_dim)
        self.conv1 = GIN_Conv(self.mlp_input)
        self.conv2 = GIN_Conv(self.mlp_hidden)
        self.conv3 = GIN_Conv(self.mlp_hidden)
        self.class_layer = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings throug the 3 convolutional layers
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()

        # 2. Global average pooling layer
        x = global_mean_pool(x, batch)
        # 3. Classification layer
        x = self.class_layer(x)
        return x