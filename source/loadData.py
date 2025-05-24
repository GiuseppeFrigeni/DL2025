import gzip
import json
import torch
from torch_geometric.data import Dataset, Data
import os
from tqdm import tqdm
import shutil # For copying the raw file

# Your dictToGraphObject function remains the same
def dictToGraphObject(graph_dict):
    edge_index = torch.tensor(graph_dict["edge_index"], dtype=torch.long)
    # Robustly get edge_attr
    edge_attr_raw = graph_dict.get("edge_attr") # Use .get to avoid KeyError
    edge_attr = torch.tensor(edge_attr_raw, dtype=torch.float) if edge_attr_raw is not None else None
    
    num_nodes = graph_dict["num_nodes"]
    
    # Robustly get y
    y_raw = graph_dict.get("y")
    if y_raw is not None:
        # Assuming y_raw is a list and we take the first element, or it's a single scalar
        y_val = y_raw[0] if isinstance(y_raw, list) and y_raw else (y_raw if not isinstance(y_raw, list) else None)
        y = torch.tensor(y_val, dtype=torch.long) if y_val is not None else None
    else:
        y = None
        
    # Robustly get x (node features) if they exist in your dict
    x_raw = graph_dict.get("x")
    x = torch.tensor(x_raw, dtype=torch.float) if x_raw is not None else None

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes, y=y)


class ProcessedGraphDataset(Dataset):
    def __init__(self, root: str, raw_filename: str, transform=None, pre_transform=None, pre_filter=None, log: bool = True):
        """
        Args:
            root (str): Root directory where the dataset should be saved.
                        A 'processed' subdirectory will be created here.
            raw_filename (str): The path to your raw gzipped JSON file.
                                This file will be copied to root/raw/ if not already there.
            transform: PyG transform to apply to data object on access.
            pre_transform: PyG transform to apply to data object before saving to disk.
            pre_filter: PyG pre-filter to apply before saving.
            log (bool): Whether to print status messages.
        """
        self.raw_filename_arg = raw_filename # Store the original path to the raw file
        self.actual_raw_filename = os.path.basename(raw_filename) # Just the filename part
        super().__init__(root, transform, pre_transform, pre_filter, log)
        
        # After super().__init__(), if process() was called and successful,
        # self.data should be populated. If processed files were found, they are loaded.
        # PyG >=2.3 loads data into self.data, self.slices and provides _data_list for direct access.
        # For older versions, it might directly load into a variable that get() uses.
        # We can load it explicitly if using torch.load in process() for older PyG.
        # For newer PyG, this should automatically load into self._data if using save/load_data_list
        try:
            self._data, self.slices = torch.load(self.processed_paths[0])
            if self.log:
                 print(f"Loaded processed data from {self.processed_paths[0]}")
        except FileNotFoundError:
            # This should not happen if super().__init__ worked correctly and called process()
            # Or if process() failed without raising an error and didn't create the file.
            if self.log:
                print(f"Processed file not found at {self.processed_paths[0]}. This might indicate an issue if processing was expected.")
        except Exception as e:
            if self.log:
                print(f"Error loading processed file: {e}. Data might not be loaded.")


    @property
    def raw_file_names(self):
        # This is the name of the file as it will be stored in self.raw_dir
        return [self.actual_raw_filename]

    @property
    def processed_file_names(self):
        # This is the name of the file where processed data will be saved
        return ['processed_graph_data.pt']

    def download(self):
        # This method is called if files in raw_file_names are not found in self.raw_dir.
        # We copy the user-provided raw file to self.raw_dir.
        source_path = self.raw_filename_arg
        destination_path = os.path.join(self.raw_dir, self.actual_raw_filename)
        
        if not os.path.exists(destination_path):
            if not os.path.exists(source_path):
                raise FileNotFoundError(f"Raw data file not found at specified path: {source_path}")
            if self.log:
                print(f"Copying raw data from {source_path} to {destination_path}...")
            shutil.copy(source_path, destination_path)
        else:
            if self.log:
                print(f"Raw file {self.actual_raw_filename} already exists in {self.raw_dir}.")

    def process(self):
        # This method is called if files in processed_file_names are not found.
        # It should read from self.raw_paths, process, and save to self.processed_paths.
        raw_path = self.raw_paths[0] # Path to the file in root/raw/
        
        if self.log:
            print(f"Processing raw data from {raw_path}...")
            print("This may take a few minutes, please wait...")
        
        with gzip.open(raw_path, "rt", encoding="utf-8") as f:
            graphs_dicts = json.load(f)
        
        data_list = []
        for graph_dict in tqdm(graphs_dicts, desc="Processing graphs into Data objects", unit="graph", disable=not self.log):
            data_list.append(dictToGraphObject(graph_dict))

        if self.pre_filter is not None:
            if self.log: print("Applying pre_filter...")
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            if self.log: print("Applying pre_transform...")
            processed_data_list = []
            for data in tqdm(data_list, desc="Applying pre_transform", unit="graph", disable=not self.log):
                processed_data_list.append(self.pre_transform(data))
            data_list = processed_data_list


        # Save the processed data_list.
        # For PyG >= 2.3, `save_data_list` is preferred.
        # For older, `torch.save(self.collate(data_list), self.processed_paths[0])` was common.
        # Let's use `torch.save` for a list of Data objects directly.
        # The collate method is typically used by DataLoader.
        data, slices = self.collate(data_list) # Collate to create the Batch object structure needed by PyG
        torch.save((data, slices), self.processed_paths[0]) # Save the collated data and slices
        
        if self.log:
            print(f"Saved {len(data_list)} processed graphs to {self.processed_paths[0]}")
        
        # After saving, PyG expects the data to be loaded.
        # For newer PyG, this might happen automatically.
        # For robustness, especially with older PyG, explicitly load it.
        self._data, self.slices = torch.load(self.processed_paths[0])


    def len(self):
        # This should return the number of graphs in the dataset.
        # PyG >= 2.3 uses self.len(), which often refers to self._data_list or len(self.slices[self.data_key]) - 1
        # Accessing self.slices or self._data which should be populated by __init__ or process
        if hasattr(self, 'slices') and self.slices is not None:
            # Assuming 'y' or 'x' or some consistent key is present in slices.
            # A common way is to count graphs based on a key that appears once per graph.
            # If self.collate created a 'ptr' for a batch-like object:
            if 'x' in self.slices: # Check for a common key like 'x' or 'y'
                 return len(self.slices['x']) -1 if self.slices['x'] is not None else 0
            elif 'y' in self.slices and self.slices['y'] is not None: # Fallback to y if x might be None
                 return len(self.slices['y'])
            elif hasattr(self, '_data') and self._data is not None and hasattr(self._data, 'num_graphs'): # if _data is a Batch object
                 return self._data.num_graphs
            elif hasattr(self, '_data_list'): # If process saved a list and __init__ loaded it directly
                 return len(self._data_list)

        # Fallback if slices/data not structured as expected, this is a weak point if not using PyG's default data handling
        # The explicit loading into self._data in __init__ and process should make this more robust.
        if hasattr(self, '_data') and self._data is not None:
             if isinstance(self._data, Data) and hasattr(self._data, 'num_graphs'): # If it's a Batch object
                 return self._data.num_graphs
             elif isinstance(self._data, list): # If you saved a list of Data objects
                 return len(self._data)

        # If you saved a list of Data objects and loaded it into self._data in __init__
        if hasattr(self, '_data_list_from_load') and self._data_list_from_load is not None:
             return len(self._data_list_from_load)

        return 0 # Should not reach here if data is loaded


    def get(self, idx):
        # This should retrieve a single Data object by index.
        # PyG >= 2.3: self.get(idx) works if self._data, self.slices are set.
        # The `super().__init__` and `self.process()` should handle loading data
        # such that the default `get` can work, or we load it into an internal list.
        # If we explicitly loaded `(data, slices)` into `self._data, self.slices`:
        if not hasattr(self, '_data') or self._data is None:
            # This might happen if loading failed silently or process was skipped and no file existed.
            # Attempt to reload if _data is not populated but processed file exists.
            # This is a bit defensive.
            try:
                self._data, self.slices = torch.load(self.processed_paths[0])
            except:
                raise RuntimeError("Data not loaded. Processed file might be missing or corrupted.")

        # Reconstruct individual Data object from the Batch-like structure
        # This is how PyG's Dataset.get works internally when data is stored as a Batch object
        data = Data()
        if hasattr(self, 'slices'):
            for key in self._data.keys: # Iterate over attributes in the Batch object (e.g. 'x', 'edge_index', 'y')
                item, slices = self._data[key], self.slices[key]
                if torch.is_tensor(item):
                    s = torch.narrow(item, 0, slices[idx], slices[idx+1] - slices[idx])
                elif isinstance(item, list) and torch.is_tensor(item[0]): # For list of tensors like in HeteroData
                    s = item[slices[idx]:slices[idx+1]]
                else: # For attributes that are not sliced per graph (e.g. num_nodes if stored per graph)
                    # This part is tricky if not using standard PyG Batch attributes.
                    # For simple Data objects, 'num_nodes' might be what you want if it's a list.
                    # However, Data objects collated into a Batch usually have 'ptr'.
                    # If y is (num_graphs,), then simple indexing is fine.
                    if key == 'y' and item.size(0) == self.len(): # Assuming y is [num_graphs]
                        s = item[idx]
                    else: # Default to taking the whole attribute if not sure how to slice
                        s = item 
                data[key] = s
        else: # Fallback if not using slices (e.g. saved a list of Data objects)
            if isinstance(self._data, list):
                 return self._data[idx] # This would be if you saved a list and not a (data,slices) tuple
            else:
                 raise RuntimeError("Cannot get item: _data is not a list and slices are not defined.")
        return data

