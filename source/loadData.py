import gzip
import json
import torch
from torch_geometric.data import Dataset, Data
import os
from tqdm import tqdm
# from torch_geometric.loader import DataLoader # Not strictly needed for the Dataset class itself

# Your helper function, slightly more robust
def dictToGraphObject(graph_dict):
    edge_index = torch.tensor(graph_dict["edge_index"], dtype=torch.long)
    
    # Handle edge_attr being potentially None or empty
    edge_attr_raw = graph_dict.get("edge_attr")
    if edge_attr_raw is not None and len(edge_attr_raw) > 0:
        edge_attr = torch.tensor(edge_attr_raw, dtype=torch.float)
    else:
        edge_attr = None

    # num_nodes: use provided, or infer from edge_index, or default to 0
    num_nodes = graph_dict.get("num_nodes")
    if num_nodes is None:
        if edge_index.numel() > 0:
            num_nodes = int(edge_index.max().item()) + 1
        else:
            num_nodes = 0 # For an empty graph with no edges

    # Handle y being potentially None or an empty list
    y_raw = graph_dict.get("y")
    if y_raw is not None: # Assuming y_raw is like [target_value] or target_value
        if isinstance(y_raw, list):
            if len(y_raw) > 0:
                y = torch.tensor([y_raw[0]], dtype=torch.long) # Take first element if list
            else:
                y = None # Or handle empty list case as an error or default
        else: # Assume it's a single value
             y = torch.tensor([y_raw], dtype=torch.long)
    else:
        y = None
        
    return Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes, y=y)


class GraphDataset(Dataset):
    def __init__(self, raw_path, transform=None, pre_transform=None, pre_filter=None, force_reload=False):
        """
        Args:
            raw_path (str): The full path to the raw .json.gz file.
            transform (callable, optional): A function/transform that takes in an
                `torch_geometric.data.Data` object and returns a transformed version.
                The data object will be transformed before every access.
            pre_transform (callable, optional): A function/transform that takes in
                an `torch_geometric.data.Data` object and returns a
                transformed version. The data object will be transformed before
                being saved to disk.
            pre_filter (callable, optional): A function that takes in an
                `torch_geometric.data.Data` object and returns a boolean
                value, indicating whether the data object should be included in the
                final dataset.
            force_reload (bool, optional): If True, the dataset will be processed
                again, even if processed files exist. Defaults to False.
        """
        self.raw_path_arg = os.path.abspath(raw_path) # Store the full raw path
        self.force_reload_flag = force_reload # Store force_reload state

        # Determine root directory and base filename for raw data
        _root = os.path.dirname(self.raw_path_arg)
        self._raw_filename_basename = os.path.basename(self.raw_path_arg)
        
        # Determine the base name for processed files (without .json.gz)
        self._processed_file_basename = self._raw_filename_basename.replace(".json.gz", "")
        if not self._processed_file_basename: # Handle cases like just ".json.gz"
            self._processed_file_basename = "data"

        super().__init__(_root, transform, pre_transform, pre_filter)
        
        # After super().__init__(), self.process() would have been called if necessary.
        # Now, load the processed data.
        try:
            self.graphs = torch.load(self.processed_paths[0])
            print(f"Successfully loaded processed data from: {self.processed_paths[0]}")
        except FileNotFoundError:
            print(f"ERROR: Processed file not found at {self.processed_paths[0]} after initialization.")
            print("This might indicate an issue with the process() method or file system permissions.")
            raise
        except Exception as e:
            print(f"Error loading processed file {self.processed_paths[0]}: {e}")
            raise


    @property
    def raw_file_names(self):
        # This should be the filename relative to self.raw_dir (which is self.root)
        return [self._raw_filename_basename]

    @property
    def processed_file_names(self):
        # This is the name of the file that will be saved in self.processed_dir
        # self.processed_dir is automatically root/processed/ by PyG
        return [f'{self._processed_file_basename}_processed.pt']

    def download(self):
        # This method is called if raw_files are not found.
        # We assume the raw_path_arg provided by the user is the correct location.
        # self.raw_paths[0] will be os.path.join(self.root, self.raw_file_names[0])
        # which should resolve to self.raw_path_arg.
        if not os.path.exists(self.raw_paths[0]):
            raise FileNotFoundError(
                f"Raw file not found: {self.raw_paths[0]}. "
                f"Please ensure '{self.raw_path_arg}' is a valid path."
            )
        # No actual download needed if file is local.

    def process(self):
        print(f"Processing raw data from: {self.raw_paths[0]}...")
        print("This may take a few minutes, please wait...")

        with gzip.open(self.raw_paths[0], "rt", encoding="utf-8") as f:
            graphs_dicts = json.load(f)

        graphs = []
        for graph_dict in tqdm(graphs_dicts, desc="Converting to Data objects", unit="graph"):
            # Perform conversion
            data = dictToGraphObject(graph_dict)
            
            # Apply pre-filter if provided
            if self.pre_filter is not None and not self.pre_filter(data):
                continue # Skip this data object
            
            # Apply pre-transform if provided
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            
            graphs.append(data)
        
        # PyG creates self.processed_dir if it doesn't exist before calling process,
        # but double-checking or creating it here can be good for standalone use of process().
        os.makedirs(self.processed_dir, exist_ok=True)
        
        processed_save_path = self.processed_paths[0]
        print(f"Saving {len(graphs)} processed graphs to: {processed_save_path}...")
        torch.save(graphs, processed_save_path)
        print("Processing and saving complete.")

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        # The self.transform (not pre_transform) is typically applied here.
        # PyG's base Dataset.get() handles this.
        # If we directly access self.graphs, we might bypass self.transform.
        # However, for simple list loading, this is fine.
        # If self.transform needs to be applied per-item on access:
        data = self.graphs[idx]
        # if self.transform:
        #     data = self.transform(data) # Usually handled by super().get() if we called it
        return data

    # Override _process to incorporate force_reload logic properly
    # PyG's Dataset.__init__ calls self._download() and then self._process().
    # self._process() checks if processed files exist and if force_reload (passed from init) is False.
    def _process(self):
        if self.force_reload_flag:
            # If forcing reload, remove existing processed files first
            # This ensures self.process() is called by the superclass logic.
            print("Force reload enabled: Checking for existing processed files to remove...")
            for path in self.processed_paths:
                if os.path.exists(path):
                    print(f"Removing existing processed file: {path}")
                    os.remove(path)
            # If the processed directory becomes empty, PyG might have issues.
            # However, it usually recreates it. For safety, ensure it exists before super()._process()
            # or rely on PyG to handle it.
            # os.makedirs(self.processed_dir, exist_ok=True) # PyG should handle this

        # Call the standard PyG _process which checks existence and then calls self.process() if needed
        # The force_reload behavior is intrinsically handled by PyG's Dataset _process method
        # if its internal `force_reload` argument is set.
        # Here, by deleting files, we ensure process() is called.
        super()._process()