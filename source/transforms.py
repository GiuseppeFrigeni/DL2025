import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree, to_networkx
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops

class StructuralFeatures(BaseTransform):
    def __call__(self, data):

        features_to_stack = []
        if data.num_nodes > 0:
            if hasattr(data, 'edge_index') and data.edge_index is not None and data.edge_index.numel() > 0:
                deg = degree(data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.float)
                temp_data_for_nx = Data(edge_index=data.edge_index, num_nodes=data.num_nodes)
                if temp_data_for_nx.edge_index is not None:
                    edge_index_no_loops, _ = remove_self_loops(temp_data_for_nx.edge_index)
                    temp_data_for_nx.edge_index = edge_index_no_loops
                nx_graph = to_networkx(temp_data_for_nx, to_undirected=False)
                if nx_graph.is_directed():
                    undirected_nx_graph = nx_graph.to_undirected()
                else:
                    undirected_nx_graph = nx_graph
                try:
                    clustering_coeffs_dict = nx.clustering(undirected_nx_graph)
                    cc_list = [clustering_coeffs_dict.get(i, 0.0) for i in range(data.num_nodes)]
                    cc_tensor = torch.tensor(cc_list, dtype=torch.float)
                except Exception as e:
                    print(f"Warning: Could not compute clustering coefficient for a graph: {e}. Using zeros.")
                    cc_tensor = torch.zeros(data.num_nodes, dtype=torch.float)

            else:
                deg = torch.zeros(data.num_nodes, dtype=torch.float)
                cc_tensor = torch.zeros(data.num_nodes, dtype=torch.float)

            deg_sq = deg**2
            features_to_stack.append(deg.unsqueeze(-1))
            features_to_stack.append(deg_sq.unsqueeze(-1))
            features_to_stack.append(cc_tensor.unsqueeze(-1))
        
        if features_to_stack:
            data.x = torch.cat(features_to_stack, dim=-1)
        else:
            data.x = torch.empty(0, 3, dtype=torch.float)

        return data

class NormalizeNodeFeatures(BaseTransform):
    def __init__(self, norm_params_list):
        """
        Args:
            norm_params_list: A list of tuples, where each tuple contains
                              (min_val, max_val) for a feature dimension.
                              Or (mean_val, std_val) if doing standardization.
        """
        super().__init__()
        self.norm_params_list = norm_params_list

    def __call__(self, data: Data):
        if data.x is not None and data.x.numel() > 0:
            x_normalized = data.x.clone() # Important to clone to avoid modifying original in-place if not desired
            for dim_idx, params in enumerate(self.norm_params_list):
                if dim_idx < x_normalized.shape[1]: # Check if feature dimension exists
                    min_val, max_val = params
                    # Min-Max Normalization
                    if (max_val - min_val) > 1e-6: # Avoid division by zero
                        x_normalized[:, dim_idx] = (x_normalized[:, dim_idx] - min_val) / (max_val - min_val)
                    else: # If max == min, set to 0 or 0.5, or handle as needed
                        x_normalized[:, dim_idx] = 0.0
                    

            data.x = x_normalized
        return data