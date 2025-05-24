import os
import datetime
import torch
import logging
import argparse
from source.loadData import GraphDataset
from torch_geometric.loader import DataLoader
from torch_geometric import seed_everything
from source.transforms import AddDegreeSquaredFeatures
from source.model import SimpleGCN, GINEGraphClassifier
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch_geometric.data import Data, Dataset # Or your specific dataset class
from typing import List, Union

from source.loss import SCELoss
from torch import optim
from torch_geometric.transforms import BaseTransform

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from torch.utils.data import Subset


torch.backends.cuda.matmul.allow_tf32 = True  # Default False in PyTorch 1.12+
torch.backends.cudnn.allow_tf32 = True  # Default True

def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)  
    return data

def get_node_feature_stats(dataset: Dataset, feature_dim: int):
    """
    Calculates min and max for a specific node feature dimension across the dataset.
    Alternatively, calculate mean and std for standardization.
    """
    all_features_dim = []
    loader = DataLoader(dataset, batch_size=64, shuffle=False) # Use DataLoader for efficiency
    print(f"Calculating stats for node feature dimension {feature_dim}...")
    for batch_data in loader:
        if batch_data.x is not None and batch_data.x.numel() > 0 and batch_data.x.shape[1] > feature_dim:
            all_features_dim.append(batch_data.x[:, feature_dim].cpu())

    if not all_features_dim:
        print(f"Warning: No features found for dimension {feature_dim} to calculate stats.")
        return None, None # Or torch.tensor(0.0), torch.tensor(1.0) for no-op normalization

    features_tensor_dim = torch.cat(all_features_dim)
    min_val = features_tensor_dim.min()
    max_val = features_tensor_dim.max()
    # For standardization:
    # mean_val = features_tensor_dim.mean()
    # std_val = features_tensor_dim.std()
    # return mean_val, std_val
    print(f"Stats for dim {feature_dim}: Min={min_val.item()}, Max={max_val.item()}")
    return min_val, max_val


# --- Step 2: Create a Transform to Apply Normalization ---
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
                    
                    # For Standardization:
                    # mean_val, std_val = params
                    # if std_val > 1e-6: # Avoid division by zero
                    #     x_normalized[:, dim_idx] = (x_normalized[:, dim_idx] - mean_val) / std_val
                    # else:
                    #     x_normalized[:, dim_idx] = 0.0 # Or just (x - mean_val)

            data.x = x_normalized
        return data


def plot_training_progress(train_losses, train_accuracies, output_dir):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 6))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Training Loss", color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Training Accuracy", color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy per Epoch')

    # Save plots in the current directory
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_progress.png"))
    plt.close()


def train(data_loader, model, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data in data_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)  # Assuming model returns a tuple
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(data_loader)


def evaluate(data_loader, model, device, calculate_accuracy=False):
    model.eval()
    correct = 0
    total = 0
    predictions = []
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            output = model(data)  # Assuming model returns a tuple
            pred = output.argmax(dim=1)
            predictions.extend(pred.cpu().numpy())
            if calculate_accuracy:
                correct += (pred == data.y).sum().item()
                total += data.y.size(0)
            
    if calculate_accuracy:
        accuracy = correct / total
        return accuracy, predictions
    return predictions

def get_feature_statistics(dataset: Union[Dataset, List[Data]], batch_size: int = 64, feature_names_x: List[str] = None, feature_names_edge: List[str] = None):
    """
    Computes and prints statistics for node and edge features in a PyG dataset.

    Args:
        dataset: A PyG Dataset object or a list of Data objects.
        batch_size: Batch size for DataLoader if processing a large dataset.
        feature_names_x: Optional list of names for node features.
        feature_names_edge: Optional list of names for edge features.
    """
    all_node_features = []
    all_edge_features = []
    num_graphs = 0
    total_nodes = 0
    total_edges = 0

    # Use DataLoader for efficient iteration, especially for large datasets
    # If dataset is small and already in memory as a list of Data, DataLoader is not strictly necessary
    # but good practice.
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print("Processing dataset to gather features...")
    for i, batch_data in enumerate(loader):
        # For a single Data object if not using DataLoader, or if dataset is a list
        # if isinstance(dataset, Dataset): data_list = [dataset[i] for i in range(len(dataset))]
        # else: data_list = dataset
        # for data in data_list:

        # If using DataLoader, batch_data is a Batch object.
        # We need to access features per graph or collect them all.
        # For overall statistics, it's easier to collect all features.

        if hasattr(batch_data, 'x') and batch_data.x is not None and batch_data.x.numel() > 0:
            all_node_features.append(batch_data.x.cpu()) # Move to CPU to avoid OOM on GPU
            total_nodes += batch_data.num_nodes
        else:
            if i == 0: print("Warning: No node features (data.x) found in the first batch/graph.")


        if hasattr(batch_data, 'edge_attr') and batch_data.edge_attr is not None and batch_data.edge_attr.numel() > 0:
            all_edge_features.append(batch_data.edge_attr.cpu()) # Move to CPU
            total_edges += batch_data.num_edges
        else:
            if i == 0: print("Warning: No edge features (data.edge_attr) found in the first batch/graph.")

        num_graphs += batch_data.num_graphs if hasattr(batch_data, 'num_graphs') else 1 # Handle single Data or Batch


    print(f"\n--- Dataset Overview ---")
    print(f"Total number of graphs: {num_graphs}") # This might be more accurately len(dataset) if not using loader on full dataset
    print(f"Total number of nodes: {total_nodes}")
    print(f"Total number of edges: {total_edges}")
    if total_nodes > 0:
        print(f"Average nodes per graph: {total_nodes / num_graphs:.2f}")
    if total_edges > 0 and num_graphs > 0 :
         print(f"Average edges per graph: {total_edges / num_graphs:.2f}")


    if not all_node_features:
        print("\nNo node features found in the dataset to analyze.")
    else:
        # Concatenate all node features
        node_features_tensor = torch.cat(all_node_features, dim=0)
        num_node_feature_dims = node_features_tensor.shape[1]
        print(f"\n--- Node Feature Statistics (data.x) ---")
        print(f"Shape of concatenated node features: {node_features_tensor.shape}")
        print(f"Number of node feature dimensions: {num_node_feature_dims}")

        if feature_names_x and len(feature_names_x) != num_node_feature_dims:
            print(f"Warning: Mismatch between provided node feature names ({len(feature_names_x)}) and actual dimensions ({num_node_feature_dims}). Using generic names.")
            feature_names_x = [f"NodeFeat_{j}" for j in range(num_node_feature_dims)]
        elif not feature_names_x:
            feature_names_x = [f"NodeFeat_{j}" for j in range(num_node_feature_dims)]

        for j in range(num_node_feature_dims):
            feature_col = node_features_tensor[:, j]
            print(f"\n  Feature: {feature_names_x[j]} (Dimension {j})")
            print(f"    Min: {feature_col.min().item():.4f}")
            print(f"    Max: {feature_col.max().item():.4f}")
            print(f"    Mean: {feature_col.mean().item():.4f}")
            print(f"    Std Dev: {feature_col.std().item():.4f}")
            print(f"    Median: {feature_col.median().item():.4f}")
            # Check for NaNs or Infs
            if torch.isnan(feature_col).any():
                print(f"    WARNING: Contains NaNs!")
            if torch.isinf(feature_col).any():
                print(f"    WARNING: Contains Infs!")
        # Overall stats
        print(f"\n  Overall Node Feature Stats:")
        print(f"    Min (all features): {node_features_tensor.min().item():.4f}")
        print(f"    Max (all features): {node_features_tensor.max().item():.4f}")
        print(f"    Mean (all features): {node_features_tensor.mean().item():.4f}")
        print(f"    Std Dev (all features): {node_features_tensor.std().item():.4f}")


    if not all_edge_features:
        print("\nNo edge features found in the dataset to analyze.")
    else:
        # Concatenate all edge features
        edge_features_tensor = torch.cat(all_edge_features, dim=0)
        num_edge_feature_dims = edge_features_tensor.shape[1]
        print(f"\n--- Edge Feature Statistics (data.edge_attr) ---")
        print(f"Shape of concatenated edge features: {edge_features_tensor.shape}")
        print(f"Number of edge feature dimensions: {num_edge_feature_dims}")

        if feature_names_edge and len(feature_names_edge) != num_edge_feature_dims:
            print(f"Warning: Mismatch between provided edge feature names ({len(feature_names_edge)}) and actual dimensions ({num_edge_feature_dims}). Using generic names.")
            feature_names_edge = [f"EdgeFeat_{j}" for j in range(num_edge_feature_dims)]
        elif not feature_names_edge:
            feature_names_edge = [f"EdgeFeat_{j}" for j in range(num_edge_feature_dims)]


        for j in range(num_edge_feature_dims):
            feature_col = edge_features_tensor[:, j]
            print(f"\n  Feature: {feature_names_edge[j]} (Dimension {j})")
            print(f"    Min: {feature_col.min().item():.4f}")
            print(f"    Max: {feature_col.max().item():.4f}")
            print(f"    Mean: {feature_col.mean().item():.4f}")
            print(f"    Std Dev: {feature_col.std().item():.4f}")
            print(f"    Median: {feature_col.median().item():.4f}")
            if torch.isnan(feature_col).any():
                print(f"    WARNING: Contains NaNs!")
            if torch.isinf(feature_col).any():
                print(f"    WARNING: Contains Infs!")
        # Overall stats
        print(f"\n  Overall Edge Feature Stats:")
        print(f"    Min (all features): {edge_features_tensor.min().item():.4f}")
        print(f"    Max (all features): {edge_features_tensor.max().item():.4f}")
        print(f"    Mean (all features): {edge_features_tensor.mean().item():.4f}")
        print(f"    Std Dev (all features): {edge_features_tensor.std().item():.4f}")

    # Add label statistics if data.y is present
    all_labels = []
    has_labels = False
    # Re-iterate or assume labels were collected if small enough
    # For simplicity here, let's re-iterate for labels, or better, check first graph
    if hasattr(dataset[0] if isinstance(dataset, Dataset) else dataset[0], 'y') and \
       (dataset[0] if isinstance(dataset, Dataset) else dataset[0]).y is not None:
        has_labels = True
        print("\nCollecting label information...")
        for batch_data in loader: # Iterate again or integrate into the first loop if memory allows
            if hasattr(batch_data, 'y') and batch_data.y is not None:
                all_labels.append(batch_data.y.cpu())

    if has_labels and all_labels:
        labels_tensor = torch.cat(all_labels)
        print(f"\n--- Label Statistics (data.y) ---")
        print(f"Shape of concatenated labels: {labels_tensor.shape}")
        if labels_tensor.numel() > 0:
            print(f"Min label: {labels_tensor.min().item()}")
            print(f"Max label: {labels_tensor.max().item()}")
            unique_labels, counts = torch.unique(labels_tensor, return_counts=True)
            print(f"Unique labels and their counts:")
            for label, count in zip(unique_labels, counts):
                print(f"  Label {label.item()}: {count.item()} occurrences")
        else:
            print("Labels tensor is empty.")
    elif has_labels:
        print("\nLabels attribute (data.y) found, but no labels collected (possibly all None).")
    else:
        print("\nNo labels (data.y) found in the dataset.")



def main(args):
    seed_everything(42)  # Set random seed for reproducibility

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir_name = os.path.basename(os.path.dirname(args.test_path))

    # Define log file path relative to the script's directory
    logs_folder = os.path.join(os.getcwd(), "logs", test_dir_name)
    log_file = os.path.join(logs_folder, "training.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler()) 

    # Define checkpoint path relative to the script's directory
    model_name = 'GINEGraphClassifier'    #"SimpleGCN"  # or "GINConv"
    checkpoints_folder = os.path.join(os.getcwd(), "saved_models", test_dir_name, model_name)
    os.makedirs(checkpoints_folder, exist_ok=True)



    submission_dir = os.path.join(os.getcwd(), 'submission')
    os.makedirs(submission_dir, exist_ok=True)


    #transfrm
    my_transform = AddDegreeSquaredFeatures()

    IN_CHANNELS = 2
    HIDDEN_CHANNELS = 64 # Example, tune this
    NUM_CLASSES = 6    # For your subset
    LEARNING_RATE = 5e-4
    EPOCHS = 50 # Increase for the small subset
    WEIGHT_DECAY = 1e-4 # Add some regularization

    test_dataset = GraphDataset(args.test_path, transform=my_transform)

    if args.train_path:

        

        # Remove previous checkpoints for the same test dataset
        for filePath in os.listdir(checkpoints_folder):
            if test_dir_name in filePath:
                os.remove(os.path.join(checkpoints_folder,filePath))
                print(f"Removed previous checkpoint: {filePath}")


        train_dataset = GraphDataset(args.train_path, transform=my_transform)
        
        min_deg, max_deg = get_node_feature_stats(train_dataset, feature_dim=0)
        min_deg_sq, max_deg_sq = get_node_feature_stats(train_dataset, feature_dim=1)
        norm_params = []
        norm_params.append((min_deg, max_deg))
        norm_params.append((min_deg_sq, max_deg_sq))

        if hasattr(train_dataset, 'transform') and train_dataset.transform is not None:
            from torch_geometric.transforms import Compose
            train_dataset.transform = Compose([train_dataset.transform, normalizer])
            if test_dataset is not None:
                 test_dataset.transform = Compose([test_dataset.transform, normalizer]) # Use TRAIN stats for val/test
        else:
            train_dataset.transform = normalizer
            if test_dataset is not None:
                test_dataset.transform = normalizer

        # Create the normalization transform instance
        normalizer = NormalizeNodeFeatures(norm_params_list=norm_params)

        #node_feature_names = ["zeros"]
        #edge_feature_names = [f"EdgeOriginalFeat_{j}" for j in range(7)] # Example edge feature names
        #get_feature_statistics(train_dataset, batch_size=32, feature_names_x=node_feature_names, feature_names_edge=edge_feature_names)

        labels = []
        for i in range(len(train_dataset)):
            labels.append(train_dataset[i].y.item()) # .item() if y is a 0-dim tensor

        import collections
        label_counts = collections.Counter(labels)

        all_labels_in_training_set = []
        for label, count in label_counts.items():
            all_labels_in_training_set.extend([label] * count)
        unique_classes_in_train = np.arange(NUM_CLASSES) # Assuming classes are 0 to NUM_CLASSES-1
        if len(unique_classes_in_train) == NUM_CLASSES:
            class_weights_values = compute_class_weight('balanced', classes=unique_classes_in_train, y=np.array(all_labels_in_training_set))
            class_weights_tensor = torch.tensor(class_weights_values, dtype=torch.float).to(device)
        else:
            print("Warning: Mismatch in expected vs. found classes. Using uniform weights for SCE.")
            class_weights_tensor = torch.ones(NUM_CLASSES, dtype=torch.float).to(device) # Fallback
        
        NODE_IN_CHANNELS = 2   # e.g., from your degree + degree_sq features
        EDGE_IN_CHANNELS = 7    # From your data.edge_attr shape
        HIDDEN_CHANNELS = 32
        NUM_CLASSES = 6
        NUM_GINE_LAYERS = 2
        DROPOUT_GINE = 0.5
        DROPOUT_MLP = 0.3
        POOLING_TYPE = 'mean'
    
        

        #model = SimpleGCN(in_channels=IN_CHANNELS, hidden_channels=HIDDEN_CHANNELS, out_channels=NUM_CLASSES).to(device)
        model = GINEGraphClassifier(node_in_channels=NODE_IN_CHANNELS,
                                   edge_in_channels=EDGE_IN_CHANNELS,
                                   hidden_channels=HIDDEN_CHANNELS,
                                   out_channels=NUM_CLASSES,
                                   num_gine_layers=NUM_GINE_LAYERS,
                                   dropout_gine=DROPOUT_GINE,
                                   dropout_mlp=DROPOUT_MLP,
                                   pooling_type=POOLING_TYPE).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        #criterion = torch.nn.CrossEntropyLoss() # Standard CE for now
        criterion = SCELoss(alpha=1.0, beta=0.5, num_classes=NUM_CLASSES, reduction='mean')


        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        best_accuracy = 0.0
        train_losses = []
        train_accuracies = []

        # Training loop
        for epoch in range(EPOCHS):
                
            train_loss = train(train_loader, model, optimizer, criterion, device)
            train_acc, _ = evaluate(train_loader, model, device, calculate_accuracy=True)

            # Save logs for training progress
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)

            print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            if (epoch + 1) % 5 == 0:
                logging.info(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

                # Save best model
            if train_acc > best_accuracy:
                best_accuracy = train_acc
                checkpoint_path = os.path.join(checkpoints_folder, f"model_{test_dir_name}_epoch_{epoch+1}.pth")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Best model updated and saved at {checkpoint_path}")




    epoch_best_model = max([int(checkpoint.split('_')[-1].split('.')[0]) for checkpoint in os.listdir(checkpoints_folder)])
    best_model_state_dict = torch.load(os.path.join(checkpoints_folder, f"model_{test_dir_name}_epoch_{epoch_best_model}.pth"))
    model = SimpleGCN(in_channels=IN_CHANNELS, hidden_channels=HIDDEN_CHANNELS, out_channels=NUM_CLASSES).to(device)
    model.load_state_dict(best_model_state_dict)

     # Prepare test dataset and loader
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Evaluate and save test predictions
    predictions = evaluate(test_loader, model, device, calculate_accuracy=False)
    test_graph_ids = list(range(len(predictions)))

    # Save predictions to CSV
    test_dir_name = os.path.dirname(args.test_path).split(os.sep)[-1]
    output_csv_path = os.path.join(f"submission/testset_{test_dir_name}.csv")
    output_df = pd.DataFrame({
        "id": test_graph_ids,
        "pred": predictions
    })
    output_df.to_csv(output_csv_path, index=False)
    print(f"Test predictions saved to {output_csv_path}")
    

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Train and evaluate a GCN model on graph datasets.")
    parser.add_argument("--train_path", type=str, default=None, help="Path to the training dataset (optional).")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test dataset.")
    args = parser.parse_args()
    main(args)