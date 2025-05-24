import gzip
import json
import torch
from torch_geometric.data import Dataset, Data
import os
from tqdm import tqdm
# from torch_geometric.loader import DataLoader # Not strictly needed for this specific change

# Helper function to convert dictionary to PyG Data object
def dictToGraphObject(graph_dict):
    edge_index = torch.tensor(graph_dict["edge_index"], dtype=torch.long)

    edge_attr_raw = graph_dict.get("edge_attr")
    edge_attr = None
    if edge_attr_raw and isinstance(edge_attr_raw, list) and len(edge_attr_raw) > 0:
        try:
            # Ensure edge_attr is 2D: [num_edges, num_edge_features]
            # If edge_attr_raw is like [[feat1, feat2], [feat3, feat4]], this is fine.
            # If edge_attr_raw is like [val1, val2] for num_edges=1, num_edge_features=2,
            # it should be [[val1, val2]]. torch.tensor([val1,val2]) -> 1D
            # torch.tensor([[val1,val2]]) -> 2D.
            # Assuming the list structure is already appropriate for direct conversion.
            temp_edge_attr = torch.tensor(edge_attr_raw, dtype=torch.float)
            if temp_edge_attr.ndim == 1 and temp_edge_attr.numel() > 0:
                # If it became 1D and is not empty, it might represent features for a single edge
                # or a single feature for multiple edges. PyG expects [num_edges, num_edge_features].
                # This heuristic might need adjustment based on actual data format.
                # For example, if edge_index indicates M edges, and edge_attr is 1D of size M,
                # it means M edges, 1 feature each. Reshape to [M, 1].
                if edge_index.shape[1] == temp_edge_attr.numel(): # num_edges == num_elements in edge_attr
                     edge_attr = temp_edge_attr.view(-1, 1)
                else: # Cannot safely infer shape, leave as is or warn
                     edge_attr = temp_edge_attr
            else:
                edge_attr = temp_edge_attr

        except Exception as e:
            # print(f"Warning: Could not convert edge_attr to tensor: {edge_attr_raw}. Error: {e}. Setting to None.")
            edge_attr = None # Fallback to None if conversion fails
    # else: edge_attr remains None (if raw is None, empty, or not a list)

    num_nodes = graph_dict["num_nodes"]

    y_raw = graph_dict.get("y")
    y = None
    if y_raw is not None:
        try:
            if isinstance(y_raw, list):
                if len(y_raw) > 0:
                    y = torch.tensor(y_raw[0], dtype=torch.long) # Taking the first element
                # else: y remains None for empty list
            elif isinstance(y_raw, (int, float)): # If y is already a scalar
                y = torch.tensor(y_raw, dtype=torch.long) # Assuming classification; adjust dtype for regression
            # else: y remains None for other types / unable to convert
        except Exception as e:
            # print(f"Warning: Could not convert y_raw to tensor: {y_raw}. Error: {e}. Setting y to None.")
            pass # y remains None

    return Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes, y=y)


class GraphDataset(Dataset):
    def __init__(self, filename, transform=None, pre_transform=None):
        self.input_filepath = filename # Store the original filename (could be .json.gz or .json)
        
        # Get path to unzipped JSON, performing unzipping if needed and caching it.
        self.unzipped_json_path = self._ensure_unzipped_json_available(self.input_filepath)
        
        # Load graph dictionaries from the unzipped JSON file
        loaded_graphs_as_data_objects = self._load_graphs_from_json(self.unzipped_json_path)
        
        # Apply pre_transform if provided
        if pre_transform is not None:
            print("Applying pre-transform to loaded graphs...")
            self.graphs = [pre_transform(graph) for graph in tqdm(loaded_graphs_as_data_objects, desc="Pre-transforming")]
        else:
            self.graphs = loaded_graphs_as_data_objects
            
        # Initialize the parent Dataset class.
        # `transform` will be applied on-the-fly by PyG's __getitem__.
        # `pre_transform` is stored by PyG; we've already manually applied it.
        super().__init__(root=None, transform=transform, pre_transform=pre_transform)

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        data = self.graphs[idx]
        # If self.transform is set (passed to super().__init__),
        # PyG's Dataset.__getitem__ (which is called if we don't override it,
        # or if we call super().__getitem__ if we do override it)
        # will apply it. So, no need to do it manually here.
        return data

    @staticmethod
    def _ensure_unzipped_json_available(filepath):
        """
        Ensures an unzipped JSON version of the filepath exists and is up-to-date.
        If filepath doesn't end with .gz, it's assumed to be the target JSON and returned as is (after existence check).
        Otherwise, it checks for path/to/file.json for path/to/file.json.gz.
        If not found or if the .gz file is newer, it unzips.
        Returns the path to the unzipped JSON file.
        """
        if not isinstance(filepath, str):
             raise ValueError(f"Input filepath must be a string, got {type(filepath)}")

        if not filepath.endswith(".gz"):
            if not os.path.exists(filepath):
                 raise FileNotFoundError(f"Specified input file does not exist: {filepath}")
            # print(f"Input file '{filepath}' does not end with .gz. Assuming it's an unzipped JSON.")
            return filepath

        # At this point, filepath is a .gz file
        gzipped_path = filepath
        if not os.path.exists(gzipped_path):
            raise FileNotFoundError(f"Specified gzipped file does not exist: {gzipped_path}")

        unzipped_target_path = gzipped_path[:-3]  # Remove ".gz" suffix

        needs_unzip = True
        if os.path.exists(unzipped_target_path):
            try:
                gzipped_mod_time = os.path.getmtime(gzipped_path)
                unzipped_mod_time = os.path.getmtime(unzipped_target_path)
                if gzipped_mod_time <= unzipped_mod_time:
                    print(f"Found cached unzipped file: {unzipped_target_path} (up-to-date).")
                    needs_unzip = False
                else:
                    print(f"Cached unzipped file {unzipped_target_path} is older than {gzipped_path}. Re-unzipping.")
            except OSError as e:
                print(f"Warning: Could not get modification times. Assuming re-unzip is needed. Error: {e}")
        
        if needs_unzip:
            print(f"Unzipping {gzipped_path} to {unzipped_target_path}...")
            try:
                with gzip.open(gzipped_path, "rb") as f_in: # Read as bytes
                    with open(unzipped_target_path, "wb") as f_out: # Write as bytes
                        # For very large files, shutil.copyfileobj(f_in, f_out) might be more memory efficient
                        f_out.write(f_in.read())
                print(f"Successfully unzipped to {unzipped_target_path}")
            except Exception as e:
                print(f"Error unzipping {gzipped_path} to {unzipped_target_path}: {e}")
                # Clean up partially created file if unzipping failed
                if os.path.exists(unzipped_target_path):
                    try:
                        os.remove(unzipped_target_path)
                    except OSError as oe_remove:
                        print(f"Error removing partially unzipped file {unzipped_target_path}: {oe_remove}")
                raise  # Re-raise the original exception to halt execution
        
        return unzipped_target_path

    @staticmethod
    def _load_graphs_from_json(json_path):
        """
        Loads graph data from an unzipped JSON file.
        Converts each graph dictionary into a PyG Data object.
        """
        print(f"Loading graph dictionaries from {json_path}...")
        with open(json_path, "rt", encoding="utf-8") as f:
            graphs_dicts = json.load(f)
        
        graphs_as_data_objects = []
        # The tqdm description now clarifies this step
        for graph_dict in tqdm(graphs_dicts, desc="Converting dicts to Data objects", unit="graph"):
            graphs_as_data_objects.append(dictToGraphObject(graph_dict)) # dictToGraphObject is defined globally
        return graphs_as_data_objects