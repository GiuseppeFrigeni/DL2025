import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree

class AddDegreeSquaredFeatures(BaseTransform):
    def __call__(self, data):
        if data.num_nodes > 0:
            if hasattr(data, 'edge_index') and data.edge_index is not None and data.edge_index.numel() > 0:
                deg = degree(data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.float)
            else:
                deg = torch.zeros(data.num_nodes, dtype=torch.float)
            deg_sq = deg**2
            # Normalize if scales are very different, though for deg and deg^2 it might be okay
            # For instance, you could standardize each channel later across the whole dataset
            data.x = torch.stack([deg, deg_sq], dim=-1) # Shape [num_nodes, 2]
        else:
            data.x = torch.empty(0, 2, dtype=torch.float)
        return data
