import os
import torch
import logging
import argparse
from source.loadData import GraphDataset
from torch_geometric.loader import DataLoader
from torch_geometric import seed_everything
from source.transforms import StructuralFeatures, NormalizeNodeFeatures, CombinedPreTransform
from source.model import  NNConvNet
from source.loss import SCELoss
import pandas as pd
from torch_geometric.data import Dataset # Or your specific dataset class

from torch import optim
from sklearn.model_selection import train_test_split

import numpy as np



def get_node_feature_stats(dataset: Dataset, feature_dim: int):
    """
    Calculates min and max for a specific node feature dimension across the dataset.
    Alternatively, calculate mean and std for standardization.
    """
    all_features_dim = []
    loader = DataLoader(dataset, batch_size=64, shuffle=False) # Use DataLoader for efficiency
    #print(f"Calculating stats for node feature dimension {feature_dim}...")
    for batch_data in loader:
        if batch_data.x is not None and batch_data.x.numel() > 0 and batch_data.x.shape[1] > feature_dim:
            all_features_dim.append(batch_data.x[:, feature_dim].cpu())

    if not all_features_dim:
        print(f"Warning: No features found for dimension {feature_dim} to calculate stats.")
        return None, None # Or torch.tensor(0.0), torch.tensor(1.0) for no-op normalization

    features_tensor_dim = torch.cat(all_features_dim)
    min_val = features_tensor_dim.min()
    max_val = features_tensor_dim.max()
    return min_val, max_val


def train(data_loader, model, optimizer, criterion, device, class_weights=None):
    model.train()
    total_loss = 0
    for data in data_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)  # Assuming model returns a tuple
        loss = criterion(output, data.y.squeeze())
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
            output = model(data)
            pred = output.argmax(dim=1)
            predictions.extend(pred.cpu().numpy())
            if calculate_accuracy:
                correct += (pred == data.y.squeeze()).sum().item()
                total += data.y.size(0)
            
    if calculate_accuracy:
        accuracy = correct / total
        return accuracy, predictions
    return predictions


def main(args):
    seed_everything(42)  # Set random seed for reproducibility

    model_name = 'NNConvNet'  # Name of the model to be used in logging and saving

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir_name = os.path.basename(os.path.dirname(args.test_path))

    # Define log file path relative to the script's directory
    logs_folder = os.path.join(os.getcwd(), "logs", test_dir_name, model_name)
    os.makedirs(logs_folder, exist_ok=True)
    log_file = os.path.join(logs_folder, "training.log")
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler()) 

    # Define checkpoint path relative to the script's directory
    checkpoints_folder = os.path.join(script_dir, "saved_models", test_dir_name)
    os.makedirs(checkpoints_folder, exist_ok=True)

    submission_dir = os.path.join(os.getcwd(), 'submission')
    os.makedirs(submission_dir, exist_ok=True)

    NUM_CLASSES = 6    
    LEARNING_RATE = 5e-4
    EPOCHS = 200 
    WEIGHT_DECAY = 1e-4 # Add some regularization
    ALPHA = 1.0  # Weight for Cross Entropy
    BETA = 0.5  # Weight for Reverse Cross Entropy
    K_LAP_PE = 8
    NUM_STRUCTURAL_FEATURES = 3    # Since we have 1st and 2nd degree
    NODE_FEATURE_DIM = K_LAP_PE + NUM_STRUCTURAL_FEATURES
    NUM_CLASSES = 6
    EDGE_FEATURE_DIM = 7
    DROPOUT_RATE = 0.5
    #use_batch_norm = True
    HIDDEN_DIM = 1024 # Hidden dimension for GAT layers
    HIDDEN_CHANNELS = 32 # Hidden dimension for GINE layers
    BATCH_SIZE = 32
    NUM_LAYERS = 2 # Number of GINE layers in the model

    
    #transform
    my_transform = CombinedPreTransform(k_lap_pe=K_LAP_PE, num_structural_features=NUM_STRUCTURAL_FEATURES)

    if test_dir_name == 'B':
        BATCH_SIZE = 16


    if args.train_path:

        # Remove previous checkpoints for the same test dataset
        for filePath in os.listdir(checkpoints_folder):
            if test_dir_name in filePath:
                os.remove(os.path.join(checkpoints_folder,filePath))
                print(f"Removed previous checkpoint: {filePath}")

        train_dataset_for_stats = GraphDataset(args.train_path, pre_transform=my_transform, force_reload=False)

        norm_params = []
        for i in range(NODE_FEATURE_DIM): # Iterate up to the new total NODE_FEATURE_DIM
            min_v, max_v = get_node_feature_stats(train_dataset_for_stats, feature_dim=i)
            if min_v is None or max_v is None: # Handle case where a feature dim might be all zeros
                print(f"Warning: No data for feature dim {i}. Using default norm params (0,1).")
                min_v, max_v = torch.tensor(0.0), torch.tensor(1.0) 
            norm_params.append((min_v, max_v))
    
        normalizer = NormalizeNodeFeatures(norm_params_list=norm_params)

        train_dataset_full = GraphDataset(args.train_path, pre_transform=my_transform, transform=normalizer, force_reload=False)
        
        
        indices = list(range(len(train_dataset_full)))
        train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)

        train_dataset = train_dataset_full.index_select(train_indices)
        validation_dataset = train_dataset_full.index_select(val_indices)

        model =  NNConvNet(
            node_in_channels=NODE_FEATURE_DIM,
            edge_feature_dim=EDGE_FEATURE_DIM, 
            hidden_channels=HIDDEN_CHANNELS,
            hidden_dim=HIDDEN_DIM,
            out_channels=NUM_CLASSES,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT_RATE,
            ).to(device)

        

        
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=15, min_lr=1e-6)
        #criterion = torch.nn.CrossEntropyLoss() # Standard CE for now
        criterion = SCELoss(alpha=ALPHA, beta=BETA, num_classes=NUM_CLASSES, reduction='mean')


        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        vali_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        best_val_acc = 0

        # Training loop
        for epoch in range(EPOCHS):
            train_loss = train(train_loader, model, optimizer, criterion, device)
            
            train_acc, _ = evaluate(train_loader, model, device, calculate_accuracy=True) # criterion uses reduction='mean'
            val_acc, _ = evaluate(vali_loader, model, device, calculate_accuracy=True) # criterion_val uses reduction='mean'
            
            print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            if (epoch + 1) % 10 == 0:
                logging.info(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

            # Save best model based on val_acc1 or val_acc2 (or average, or max)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                checkpoint_path = os.path.join(checkpoints_folder, f"model_{test_dir_name}_epoch_{best_epoch}.pth")
                torch.save({
                    'epoch': best_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), # Optional, if you might resume training
                    'best_val_acc': best_val_acc,
                    'norm_params': norm_params, 
                }, checkpoint_path)
                print(f"+++++ Saved new best model at epoch {best_epoch} with val acc {best_val_acc:.4f}")
            
            scheduler.step(val_acc)  # Step scheduler based on validation accuracy

    

    
    
    best_epoch = max([int(checkpoint.split('_')[-1].split('.')[0]) for checkpoint in os.listdir(checkpoints_folder)])
    checkpoint = torch.load(os.path.join(checkpoints_folder, f"model_{test_dir_name}_epoch_{best_epoch}.pth"))
    norm_params = checkpoint['norm_params']
    normalizer = NormalizeNodeFeatures(norm_params_list=norm_params)

    print(f"Loading best model from epoch {best_epoch} for model")


    model =  NNConvNet(
            node_in_channels=NODE_FEATURE_DIM,
    edge_feature_dim=EDGE_FEATURE_DIM, 
    hidden_channels=HIDDEN_CHANNELS,
    hidden_dim=HIDDEN_DIM,
    out_channels=NUM_CLASSES,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT_RATE,
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])

     # Prepare test dataset and loader
    test_dataset = GraphDataset(args.test_path, pre_transform=my_transform, transform=normalizer, force_reload=False)
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