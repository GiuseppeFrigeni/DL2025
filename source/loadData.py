import gzip
import json
import torch
from torch_geometric.data import Dataset, Data
import os
from tqdm import tqdm
# from torch_geometric.loader import DataLoader # Not needed for Dataset definition

# Your helper function, kept from the robust version
def dictToGraphObject(graph_dict):
    edge_index = torch.tensor(graph_dict["edge_index"], dtype=torch.long)
    
    edge_attr_raw = graph_dict.get("edge_attr")
    if edge_attr_raw is not None and len(edge_attr_raw) > 0:
        edge_attr = torch.tensor(edge_attr_raw, dtype=torch.float)
    else:
        edge_attr = None

    num_nodes = graph_dict.get("num_nodes")
    if num_nodes is None:
        if edge_index.numel() > 0:
            num_nodes = int(edge_index.max().item()) + 1
        else:
            num_nodes = 0 

    y_raw = graph_dict.get("y")
    if y_raw is not None: 
        if isinstance(y_raw, list):
            if len(y_raw) > 0:
                y = torch.tensor([y_raw[0]], dtype=torch.long) 
            else:
                y = None 
        else: 
             y = torch.tensor([y_raw], dtype=torch.long)
    else:
        y = None
        
    return Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes, y=y)


class GraphDataset(Dataset):
    def __init__(self, raw_path, transform=None, pre_transform=None, pre_filter=None, force_reload=False):
        self.raw_path_arg = os.path.abspath(raw_path)

        _root = os.path.dirname(self.raw_path_arg)
        self._raw_filename_basename = os.path.basename(self.raw_path_arg)
        
        name = self._raw_filename_basename
        if name.endswith(".json.gz"):
            base = name[:-len(".json.gz")]
        elif name.endswith(".tar.gz"): 
            base = name[:-len(".gz")] 
        elif name.endswith(".gz"):
            base = name[:-len(".gz")]
        elif name.endswith(".json"):
            base = name[:-len(".json")]
        else: # General fallback: remove last extension if one exists
            last_dot_idx = name.rfind('.')
            base = name[:last_dot_idx] if last_dot_idx != -1 and last_dot_idx != 0 else name
        
        self._processed_file_basename = base if base else "data" # Ensure not empty string

        # Pass force_reload to the superclass constructor
        super().__init__(_root, transform, pre_transform, pre_filter, force_reload=force_reload)
        
        # Load data from processed path
        try:
            self.graphs = torch.load(self.processed_paths[0], weights_only=False)
            print(f"Successfully loaded processed data from: {self.processed_paths[0]}")
        except FileNotFoundError:
            # This might happen if process() failed or if the file was deleted externally.
            # PyG's __init__ should have ensured process() ran if files were missing.
            print(f"ERROR: Processed file not found at {self.processed_paths[0]} after super().__init__.")
            print("This may indicate an issue during the process() step or file system problems.")
            raise
        except Exception as e:
            print(f"Error loading processed file {self.processed_paths[0]}: {e}")
            raise

    @property
    def raw_dir(self):
        # Override default PyG behavior (self.root / "raw").
        # Raw files are expected to be directly in self.root.
        return self.root

    @property
    def raw_file_names(self):
        # Filename relative to self.raw_dir (which is now self.root).
        return [self._raw_filename_basename]

    @property
    def processed_file_names(self):
        # Filename for the cached data in self.processed_dir.
        return [f'{self._processed_file_basename}_processed.pt']

    def download(self):
        # This method is called by PyG if self.raw_paths[0] is not found.
        # Since self.raw_paths[0] will be self.raw_path_arg (due to raw_dir override),
        # if this is called, it means the user-provided raw file does not exist.
        if not os.path.exists(self.raw_path_arg):
            raise FileNotFoundError(
                f"Raw file not found: {self.raw_path_arg}. "
                f"Please ensure this path provided to GraphDataset is correct."
            )
        # If actual download logic from a URL were needed, it would go here,
        # placing the file at self.raw_paths[0].

    def process(self):
        # self.raw_paths[0] now correctly points to the user's original raw file.
        print(f"Processing raw data from: {self.raw_paths[0]}...")
        print("This may take a few minutes, please wait...")

        with gzip.open(self.raw_paths[0], "rt", encoding="utf-8") as f:
            graphs_dicts = json.load(f)

        data_list = [] # Renamed from graphs to data_list to match PyG terminology
        for graph_dict in tqdm(graphs_dicts, desc="Converting to Data objects", unit="graph"):
            data = dictToGraphObject(graph_dict)
            
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            
            data_list.append(data)
        
        # PyG creates self.processed_dir, but ensuring it exists is safe.
        os.makedirs(self.processed_dir, exist_ok=True)
        
        processed_save_path = self.processed_paths[0]
        print(f"Saving {len(data_list)} processed graphs to: {processed_save_path}...")
        torch.save(data_list, processed_save_path) # Save the list of Data objects
        print("Processing and saving complete.")

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        # self.graphs already contains the Data objects.
        # PyG's base Dataset.get() method (which is implicitly called if not overridden,
        # or if super().get(idx) is called) handles applying self.transform.
        # If we directly return self.graphs[idx], self.transform is bypassed here.
        # However, PyG might wrap this get. For clarity:
        data = self.graphs[idx]
        # if self.transform: # This is usually handled by PyG's DataLoader or base get
        #     data = self.transform(data)
        return data