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
    def __init__(self, root: str, raw_filename: str,
                 transform=None, pre_transform=None, pre_filter=None, log: bool = True):
        
        self.raw_file_to_process_from = raw_filename
        if not os.path.exists(self.raw_file_to_process_from):
            raise FileNotFoundError(f"Raw data file: {self.raw_file_to_process_from}")
        self.processed_file_suffix = raw_filename.split('.')[-2] 
        self._dummy_raw_filename = f"dummy_raw_ref{self.processed_file_suffix}.source"
        super().__init__(root, transform, pre_transform, pre_filter, log)

        if not (hasattr(self, '_data') and self._data is not None and hasattr(self, 'slices') and self.slices is not None):
            try:
                loaded_data = torch.load(self.processed_paths[0])
                if isinstance(loaded_data, tuple) and len(loaded_data) == 2:
                    self._data, self.slices = loaded_data
                elif isinstance(loaded_data, Data): # If only Batch object was saved
                    self._data = loaded_data
                    self.slices = self._data.__slices__ # Batch object stores slices internally
                else:
                    raise TypeError("Loaded processed file is not in expected (data, slices) or Batch format.")
                if self.log: print(f"Loaded processed data: {self.processed_paths[0]}")
            except FileNotFoundError: pass
            except Exception as e:
                if self.log: print(f"Error loading {self.processed_paths[0]}: {e}.")

    @property
    def raw_file_names(self):
        return [self._dummy_raw_filename]

    @property
    def processed_file_names(self):
        return [f'processed_data{self.processed_file_suffix}.pt']

    def download(self):
        if not os.path.exists(self.raw_file_to_process_from):
            raise FileNotFoundError(f"Original raw data: {self.raw_file_to_process_from}")
        dummy_raw_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        if not os.path.exists(dummy_raw_path):
            if self.log: print(f"Download() creating dummy: {dummy_raw_path}")
            if not os.path.exists(self.raw_dir): os.makedirs(self.raw_dir, exist_ok=True)
            with open(dummy_raw_path, 'w') as f: f.write(f"Actual: {self.raw_file_to_process_from}")
        elif self.log: print(f"Download() dummy exists: {dummy_raw_path}")

    def process(self):
        raw_path_to_use = self.raw_file_to_process_from
        if self.log: print(f"Processing from: {raw_path_to_use} (suffix '{self.processed_file_suffix}')")
        
        with gzip.open(raw_path_to_use, "rt", encoding="utf-8") as f: graphs_dicts = json.load(f)
        
        data_list = [dictToGraphObject(gd) for gd in tqdm(graphs_dicts, desc=f"DictToObj ({self.processed_file_suffix})", disable=not self.log)]

        if self.pre_filter: data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform:
            data_list = [self.pre_transform(d) for d in tqdm(data_list, desc=f"PreTransform ({self.processed_file_suffix})", disable=not self.log) if d is not None]
        
        if not data_list:
            if self.log: print(f"Warning: Empty data_list for {self.processed_paths[0]}. Saving empty.")
            # Create an empty Batch object and its slices
            empty_batch = Batch.from_data_list([])
            # For (data, slices) format, data is the Batch obj, slices is its __slices__ dict
            torch.save((empty_batch, empty_batch.__slices__), self.processed_paths[0])
            self._data, self.slices = empty_batch, empty_batch.__slices__
            return

        # Collate using Batch.from_data_list()
        collated_batch_object = Batch.from_data_list(data_list)
        
        # The Batch object itself contains the concatenated data and the __slices__ attribute.
        # We can save this tuple: (Batch_object_as_Data, its_slices_dictionary)
        # This is what InMemoryDataset's collate effectively produces and saves.
        data_to_save = collated_batch_object 
        slices_to_save = collated_batch_object.__slices__ # Access internal slices

        torch.save((data_to_save, slices_to_save), self.processed_paths[0])
        
        if self.log: print(f"Saved {len(data_list)} graphs to {self.processed_paths[0]}")
        self._data, self.slices = data_to_save, slices_to_save


    def len(self):
        if hasattr(self, 'slices') and self.slices and 'y' in self.slices and self.slices['y'] is not None:
            return self.slices['y'].size(0)
        elif hasattr(self, '_data') and hasattr(self._data, 'num_graphs'): # If self._data is a Batch obj
            return self._data.num_graphs
        return 0

    def get(self, idx: int) -> Data:
        if not (hasattr(self, '_data') and self._data and hasattr(self, 'slices') and self.slices):
            try:
                loaded_data = torch.load(self.processed_paths[0])
                if isinstance(loaded_data, tuple) and len(loaded_data) == 2: self._data, self.slices = loaded_data
                elif isinstance(loaded_data, Data): self._data, self.slices = loaded_data, loaded_data.__slices__
                else: raise TypeError("Unexpected loaded format")
            except Exception as e: raise RuntimeError(f"Data not loaded for get({idx}): {e}")
        
        # Manually reconstruct the Data object for index idx
        # This is what super().get(idx) would do if self._data and self.slices are correctly populated
        # in the (Batch_object, slices_dict) format.
        data = Data()
        if not self._data.keys: # Empty Batch object
            return data # Return an empty Data object

        for key in self._data.keys:
            item = self._data[key]
            slices_for_key = self.slices[key]
            
            if torch.is_tensor(item):
                if item.ndim == 0: # Scalar tensor (should not happen for attributes like x, edge_index, y in a batch)
                    data[key] = item
                elif slices_for_key.dim() == 1 and slices_for_key.size(0) == self.len() + 1: # Node/edge level
                    start, end = slices_for_key[idx].item(), slices_for_key[idx+1].item()
                    data[key] = item.narrow(self._data.__cat_dim__(key, item), start, end - start)
                elif item.size(0) == self.len(): # Graph level (e.g. y)
                    data[key] = item[idx]
                else: # Default / unknown structure
                    data[key] = item 
            else: # Non-tensor attribute (e.g., a list stored directly)
                data[key] = item
        return data