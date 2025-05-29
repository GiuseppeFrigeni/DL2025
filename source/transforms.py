import torch
from torch_geometric.transforms import BaseTransform, AddLaplacianEigenvectorPE
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
    

class CombinedPreTransform(BaseTransform):
    def __init__(self, k_lap_pe, num_structural_features=3):
        super().__init__()
        self.structural_features = StructuralFeatures()
        self.k_lap_pe = k_lap_pe
        self.num_structural_features = num_structural_features
        self.expected_total_features = self.num_structural_features + self.k_lap_pe

        if self.k_lap_pe > 0:
            # cat=True: concatenates PE to data.x.
            # attr_name='lap_pe': PEs are temporarily stored in data.lap_pe then concatenated to data.x.
            # If data.x doesn't exist, it will create data.x from PE.
            # It handles padding if num_nodes < k_lap_pe.
            self.lap_pe_transform = AddLaplacianEigenvectorPE(
                k=k_lap_pe, attr_name='lap_pe', cat=True, is_undirected=True
            )
        else:
            self.lap_pe_transform = None

    def __call__(self, data):
        # Apply structural features first
        data = self.structural_features(data) # This should create data.x (N, num_structural_features) or (0, num_structural_features)

        if self.lap_pe_transform is not None:
            if data.num_nodes > 0:
                # If StructuralFeatures produced no data.x (e.g. graph has nodes but SF decided no features)
                # ensure data.x exists before LapPE if it's going to cat to it.
                if data.x is None or data.x.numel() == 0:
                     data.x = torch.zeros((data.num_nodes, self.num_structural_features), 
                                          dtype=torch.float, device=data.edge_index.device if data.edge_index is not None else torch.device('cpu'))

                # LapPE transform will append k_lap_pe features to data.x
                data = self.lap_pe_transform(data)
            else: # No nodes, ensure data.x is (0, total_features)
                data.x = torch.empty((0, self.expected_total_features), dtype=torch.float)
        
        # Ensure data.x has the correct final dimension, especially for 0-node graphs
        # or if LapPE was not applied (k_lap_pe=0)
        if data.x is not None and data.x.shape[0] == 0 and data.x.shape[1] != self.expected_total_features:
            data.x = torch.empty((0, self.expected_total_features), dtype=data.x.dtype, device=data.x.device)
        elif data.x is None and data.num_nodes == 0: # Should be handled by SF already
            data.x = torch.empty((0, self.expected_total_features), dtype=torch.float)

        return data