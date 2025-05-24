import gzip
import json
import torch
from torch_geometric.data import Dataset, Data, Batch
import os
from tqdm import tqdm

# --- dictToGraphObject ---
def dictToGraphObject(graph_dict): # Same as before
    edge_index = torch.tensor(graph_dict["edge_index"], dtype=torch.long)
    edge_attr_raw = graph_dict.get("edge_attr")
    edge_attr = torch.tensor(edge_attr_raw, dtype=torch.float) if edge_attr_raw is not None else None
    num_nodes = graph_dict["num_nodes"]
    y_raw = graph_dict.get("y")
    y = None
    if y_raw is not None:
        y_val = y_raw[0] if isinstance(y_raw, list) and y_raw else (y_raw if not isinstance(y_raw, list) else None)
        y = torch.tensor(y_val, dtype=torch.long) if y_val is not None else None
    x_raw = graph_dict.get("x")
    x = torch.tensor(x_raw, dtype=torch.float) if x_raw is not None else None
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes, y=y)

class ProcessedGraphDataset(Dataset):
    def __init__(self, root: str, raw_filename: str, # Direct path to your .json.gz
                 processed_file_suffix: str = "",
                 transform=None, pre_transform=None, pre_filter=None, log: bool = True):
        
        self.raw_file_to_process_from = raw_filename# Store the absolute path
        if not os.path.exists(self.raw_file_to_process_from):
            raise FileNotFoundError(
                f"Raw data file not found at the specified absolute path: {self.raw_file_to_process_from}"
            )
            
        self.processed_file_suffix = processed_file_suffix
        
        # `root` is only for storing processed files.
        super().__init__(root, transform, pre_transform, pre_filter, log)

        # Attempt to load data if not already handled by superclass's processing logic
        if not (hasattr(self, '_data') and self._data is not None and hasattr(self, 'slices') and self.slices is not None):
            try:
                # This path is constructed by super().__init__ using self.processed_dir and self.processed_file_names
                self._data, self.slices = torch.load(self.processed_paths[0])
                if self.log:
                    print(f"Successfully loaded processed data from {self.processed_paths[0]}")
            except FileNotFoundError:
                if self.log:
                    # This is expected if process() was called by super() and just finished, or if it's the very first run.
                    # If super() called process(), then self._data should be populated by process() itself.
                    pass
            except Exception as e:
                if self.log:
                    print(f"Error during explicit load of {self.processed_paths[0]}: {e}.")

    @property
    def raw_file_names(self):
        # Signal that raw files are handled externally / not managed by PyG in root/raw.
        # An empty list typically means no raw files to check/download.
        return []

    @property
    def processed_file_names(self):
        # Name of the file where processed data will be saved in self.processed_dir
        return [f'processed_data{self.processed_file_suffix}.pt']

    def download(self):
        # This method should not be called if raw_file_names is empty.
        # If it were called, it would mean there's a misunderstanding in PyG's logic
        # or our setup. We can leave it as a pass or raise an error.
        pass

    def process(self):
        # Read directly from the absolute path provided in __init__
        raw_path_to_use = self.raw_file_to_process_from
        
        if self.log:
            print(f"Processing raw data directly from {raw_path_to_use} for suffix '{self.processed_file_suffix}'...")
        
        # Redundant check, already did in __init__, but good for safety if process is called externally
        if not os.path.exists(raw_path_to_use):
            raise FileNotFoundError(f"Cannot process. Raw file not found at: {raw_path_to_use}")

        with gzip.open(raw_path_to_use, "rt", encoding="utf-8") as f:
            graphs_dicts = json.load(f)
        
        data_list = []
        for graph_dict in tqdm(graphs_dicts, desc=f"Processing graphs ({self.processed_file_suffix})", unit="graph", disable=not self.log):
            data_list.append(dictToGraphObject(graph_dict))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            transformed_data_list = []
            for data_obj in tqdm(data_list, desc=f"Pre-transforming ({self.processed_file_suffix})", unit="graph", disable=not self.log):
                transformed_data_obj = self.pre_transform(data_obj)
                if transformed_data_obj is not None:
                    transformed_data_list.append(transformed_data_obj)
            data_list = transformed_data_list
        
        if not data_list:
            if self.log:
                print(f"Warning: data_list is empty after pre_filter/pre_transform for {self.processed_paths[0]}. Saving empty data structure.")
            # Save an empty collated structure
            empty_collated_data, empty_slices = self.collate([])
            torch.save((empty_collated_data, empty_slices), self.processed_paths[0])
            self._data, self.slices = empty_collated_data, empty_slices # Populate for current instance
            return

        # Collate the list of Data objects using the inherited collate method
        collated_data, slices_dict = self.collate(data_list)
        torch.save((collated_data, slices_dict), self.processed_paths[0])
        
        if self.log:
            print(f"Saved {len(data_list)} processed graphs to {self.processed_paths[0]}")
        
        # Populate self._data and self.slices for the current instance
        self._data, self.slices = collated_data, slices_dict

    def len(self):
        if hasattr(self, 'slices') and self.slices is not None:
            # Assuming 'y' exists for graph classification and indicates number of graphs
            if 'y' in self.slices and self.slices['y'] is not None and torch.is_tensor(self.slices['y']):
                 return self.slices['y'].size(0)
            # Fallback if y is not there or not a tensor (e.g. unsupervised)
            # Use a common node/edge attribute's slice pointer length
            elif 'x' in self.slices and self.slices['x'] is not None and torch.is_tensor(self.slices['x']): # Pointer for nodes
                 return self.slices['x'].size(0) - 1 if self.slices['x'].size(0) > 0 else 0
        # If _data was loaded as a list (older saving method, not used here)
        elif hasattr(self, '_data') and isinstance(self._data, list):
            return len(self._data)
        return 0

    def get(self, idx: int) -> Data:
        # Base class's get should work if self._data and self.slices are loaded correctly
        if not (hasattr(self, '_data') and self._data is not None and hasattr(self, 'slices') and self.slices is not None):
            try:
                self._data, self.slices = torch.load(self.processed_paths[0])
                if self.log: print(f"Data reloaded in get() for index {idx}")
            except Exception as e:
                raise RuntimeError(f"Data not loaded for get({idx}). Processed file {self.processed_paths[0]} might be missing or corrupted: {e}")
        return super().get(idx)
