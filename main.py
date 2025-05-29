import os
import torch
import logging
import argparse
from source.loadData import GraphDataset
from torch_geometric.loader import DataLoader
from torch_geometric import seed_everything
from source.transforms import StructuralFeatures, NormalizeNodeFeatures, CombinedPreTransform
from source.model import  NNConvNet, GINENetForGCOD
from source.loss import SCELoss
import pandas as pd
from torch_geometric.data import Dataset # Or your specific dataset class
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from sklearn.model_selection import train_test_split

import numpy as np

class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
    def __len__(self):
        return len(self.original_dataset)
    def __getitem__(self, idx):
        return self.original_dataset[idx], idx

def train_gcod(data_loader, model, criterion_gcod, optimizer_model, optimizer_u, device, current_epoch_train_acc, train_dataset_indices_map=None):
    model.train()
    total_loss_model_epoch = 0
    total_loss_u_epoch = 0
    
    for batch_idx, (data_batch, batch_original_indices) in enumerate(data_loader): 
        data_batch = data_batch.to(device)
        u_indices_for_batch = batch_original_indices # Assuming direct mapping

        optimizer_model.zero_grad()
        optimizer_u.zero_grad()

        logits, embeddings_batch_Z_B = model(data_batch) 
        u_for_batch = criterion_gcod.get_u_for_batch(u_indices_for_batch)

        loss_model_batch, loss_u_batch = criterion_gcod(logits, embeddings_batch_Z_B, data_batch.y.squeeze(), u_for_batch, current_epoch_train_acc)

        if torch.isnan(loss_model_batch) or torch.isinf(loss_model_batch) or \
           torch.isnan(loss_u_batch) or torch.isinf(loss_u_batch):
            print(f"Warning: NaN/Inf loss in batch {batch_idx}. ModelLoss: {loss_model_batch}, ULoss: {loss_u_batch}. Skipping batch update.")
            continue
        
        # First backward pass (for u)
        # Since loss_model_batch also depends on logits/embeddings, we need to retain the graph here.
        loss_u_batch.backward(retain_graph=True) 
        optimizer_u.step()
        with torch.no_grad(): 
            criterion_gcod.u_all_samples.clamp_(0.0, 1.0)

        # Second backward pass (for model parameters)
        # Now this can proceed because retain_graph=True was set on the previous backward call
        # for the shared parts of the graph.
        loss_model_batch.backward() 
        optimizer_model.step()
        
        total_loss_model_epoch += loss_model_batch.item() * data_batch.num_graphs
        total_loss_u_epoch += loss_u_batch.item() * data_batch.num_graphs

    avg_loss_model = total_loss_model_epoch / len(data_loader.dataset) if len(data_loader.dataset) > 0 else 0
    avg_loss_u = total_loss_u_epoch / len(data_loader.dataset) if len(data_loader.dataset) > 0 else 0
    return avg_loss_model, avg_loss_u

class GCODLoss(nn.Module):
    def __init__(self, num_train_samples, num_classes, embedding_dim, device, initial_u_std=1e-9):
        super(GCODLoss, self).__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.device = device
        self.u_all_samples = nn.Parameter(torch.normal(mean=1e-8, std=initial_u_std, size=(num_train_samples,), device=device),requires_grad=True)
        self.current_class_centroids = torch.zeros(num_classes, embedding_dim, device=device, requires_grad=False) # Centroids are not learned directly by model's optimizer
        self.cross_entropy_noreduction = nn.CrossEntropyLoss(reduction='none')

    def get_u_for_batch(self, indices):
        return self.u_all_samples[indices.to(self.device).long()]

    @torch.no_grad()
    def update_class_centroids(self, embeddings_epoch, labels_epoch, u_values_epoch, epoch, total_epochs):
        self.current_class_centroids.fill_(0.0)
        counts = torch.zeros(self.num_classes, device=self.device)
        for c in range(self.num_classes):
            class_mask = (labels_epoch == c)
            if class_mask.sum() == 0:
                self.current_class_centroids[c] = torch.randn(self.embedding_dim, device=self.device) * 0.01
                continue
            class_embeddings = embeddings_epoch[class_mask]
            class_u_values = u_values_epoch[class_mask]
            percentage_to_use = 1.0 - 0.5 * (epoch / total_epochs)
            num_samples_in_class = class_embeddings.size(0)
            num_to_select = max(1, int(num_samples_in_class * percentage_to_use))
            if num_samples_in_class > 0:
                sorted_indices = torch.argsort(class_u_values)
                selected_indices = sorted_indices[:num_to_select]
                if len(selected_indices) > 0:
                    centroid_c = class_embeddings[selected_indices].mean(dim=0)
                    self.current_class_centroids[c] = centroid_c
                else:
                    self.current_class_centroids[c] = torch.randn(self.embedding_dim, device=self.device) * 0.01
        norms = torch.norm(self.current_class_centroids, p=2, dim=1, keepdim=True)
        self.current_class_centroids = self.current_class_centroids / (norms + 1e-8)

    def calculate_soft_labels_y_bar(self, embeddings_batch, targets_batch_one_hot):
        h_i = F.normalize(embeddings_batch, p=2, dim=1)
        class_indices = torch.argmax(targets_batch_one_hot, dim=1)
        h_c_for_batch = self.current_class_centroids[class_indices]
        similarity = (h_i * h_c_for_batch).sum(dim=1)
        y_bar_magnitudes = torch.clamp(similarity, min=0.0)
        y_bar_batch = targets_batch_one_hot * y_bar_magnitudes.unsqueeze(1)
        return y_bar_batch

    def forward(self, logits, embeddings_batch, targets_batch, u_batch, training_accuracy_epoch):
        batch_size = logits.size(0)
        targets_one_hot = F.one_hot(targets_batch.long(), num_classes=self.num_classes).float().to(self.device)
        y_bar_batch = self.calculate_soft_labels_y_bar(embeddings_batch, targets_one_hot)

        diag_u_y_hard = torch.diag_embed(u_batch) @ targets_one_hot
        modified_logits_L1 = logits + training_accuracy_epoch * diag_u_y_hard
        loss1_per_sample = -torch.sum(y_bar_batch * F.log_softmax(modified_logits_L1, dim=1), dim=1)
        L1 = loss1_per_sample.mean()

        pred_probs = F.softmax(logits, dim=1)
        y_hat_batch_one_hot = F.one_hot(torch.argmax(pred_probs, dim=1), num_classes=self.num_classes).float().to(self.device)
        u_y_hard_term = u_batch.unsqueeze(1) * targets_one_hot
        L2_term_inside_norm = y_hat_batch_one_hot + u_y_hard_term - targets_one_hot
        L2 = (1.0 / self.num_classes) * L2_term_inside_norm.pow(2).sum(dim=1).mean()

        model_prob_true_class = F.softmax(logits, dim=1).gather(1, targets_batch.long().unsqueeze(1)).squeeze(1)
        model_prob_true_class_clamped = torch.clamp(model_prob_true_class, 1e-7, 1.0 - 1e-7)
        u_batch_clamped = torch.clamp(u_batch, 1e-7, 1.0 - 1e-7)
        target_kl_L3_scalar = torch.sigmoid(-torch.log(u_batch_clamped))
        
        target_kl_L3_scalar_clamped = torch.clamp(target_kl_L3_scalar, 1e-7, 1.0 - 1e-7)

        kl_term_1 = target_kl_L3_scalar_clamped * torch.log(target_kl_L3_scalar_clamped / model_prob_true_class_clamped + 1e-7) # Added epsilon for log
        kl_term_2 = (1 - target_kl_L3_scalar_clamped) * torch.log((1 - target_kl_L3_scalar_clamped) / (1 - model_prob_true_class_clamped + 1e-7) + 1e-7)

        L3_per_sample = (1.0 - training_accuracy_epoch) * (kl_term_1 + kl_term_2)
        L3 = L3_per_sample.mean()
        if torch.isnan(L3) or torch.isinf(L3):
            L3 = torch.tensor(0.0, device=self.device) # Ensure L3 is differentiable even if 0

        loss_model = L1 + L3
        loss_u = L2
        return loss_model, loss_u

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
    predictions_list = [] 
    with torch.no_grad():
        for data_batch in data_loader: # DataLoader now yields PyG Data/Batch objects directly
            # No need to check for tuple if using the _for_eval loaders
            data_batch = data_batch.to(device) # This should now work
            
            output, _ = model(data_batch) # Assuming model returns (logits, embeddings)
            pred = output.argmax(dim=1)
            predictions_list.extend(pred.cpu().numpy())
            if calculate_accuracy:
                correct += (pred == data_batch.y.squeeze()).sum().item()
                total += data_batch.y.size(0)
            
    if calculate_accuracy:
        accuracy = correct / total if total > 0 else 0
        return accuracy, predictions_list
    return predictions_list


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
    checkpoints_folder = os.path.join(script_dir, "checkpoint", test_dir_name)
    os.makedirs(checkpoints_folder, exist_ok=True)

    submission_dir = os.path.join(os.getcwd(), 'submission')
    os.makedirs(submission_dir, exist_ok=True)

    NUM_CLASSES = 6    
    LEARNING_RATE = 1e-3
    EPOCHS = 300 
    WEIGHT_DECAY = 1e-4 # Add some regularization
    ALPHA = 1.0  # Weight for Cross Entropy
    BETA = 0.5  # Weight for Reverse Cross Entropy
    K_LAP_PE = 8
    NUM_STRUCTURAL_FEATURES = 3    # Since we have 1st and 2nd degree
    NODE_FEATURE_DIM = K_LAP_PE + NUM_STRUCTURAL_FEATURES
    NUM_CLASSES = 6
    EDGE_FEATURE_DIM = 7
    DROPOUT_RATE = 0.3
    #use_batch_norm = True
    HIDDEN_DIM = 1024 # Hidden dimension for GAT layers
    HIDDEN_CHANNELS = 32 # Hidden dimension for GINE layers
    BATCH_SIZE = 32
    NUM_LAYERS = 2 # Number of GINE layers in the model

    LEARNING_RATE_U = 1 
    INITIAL_U_STD = 1e-9
    GCOD_EMBEDDING_DIM = HIDDEN_CHANNELS

    GNN_TYPE = 'GINE' # Switch to GINE
    EDGE_FEATURE_DIM = 7 # Your edge feature dimension
    GIN_HIDDEN_CHANNELS = 300 # Or your desired hidden size for GINE
    GIN_NUM_LAYERS = 5
    GIN_MLP_HIDDEN_CHANNELS = 128
    GCOD_EMBEDDING_DIM = GIN_HIDDEN_CHANNELS
    # GINE specific params (optional, defaults are often fine)
    GINE_EPS = 0.0
    GINE_TRAIN_EPS = False

    
    #transform
    my_transform = CombinedPreTransform(k_lap_pe=K_LAP_PE, num_structural_features=NUM_STRUCTURAL_FEATURES)



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

        num_samples_for_gcod_u = len(train_dataset)


        if GNN_TYPE == 'GINE':
            model = GINENetForGCOD(
        node_in_channels=NODE_FEATURE_DIM,
        edge_feature_dim=EDGE_FEATURE_DIM, # Pass edge feature dim
        gnn_hidden_channels=GIN_HIDDEN_CHANNELS,
        num_gnn_layers=GIN_NUM_LAYERS,
        mlp_hidden_channels=GIN_MLP_HIDDEN_CHANNELS,
        out_channels_final=NUM_CLASSES,
        dropout_rate=DROPOUT_RATE,
        return_embeddings=True,
        pooling_type='mean',
        eps=GINE_EPS,
        train_eps=GINE_TRAIN_EPS
    ).to(device)
        else:
            model = NNConvNet(
            node_in_channels=NODE_FEATURE_DIM, # Or node_feature_dim_from_model when loading
            edge_feature_dim=EDGE_FEATURE_DIM,
            out_channels_gnn=GCOD_EMBEDDING_DIM,      # GNN output dim
            hidden_dim_edge_nn=HIDDEN_DIM,           # For NN inside NNConv
            mlp_hidden_dim_factor=0.5,               # For MLP after GNN embeddings
            out_channels_final=NUM_CLASSES,          # Final output classes
            num_layers=NUM_LAYERS,
            return_embeddings=True                   # Crucial for GCOD
        ).to(device)

        

        
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        #criterion = torch.nn.CrossEntropyLoss() # Standard CE for now
        criterion = SCELoss(alpha=ALPHA, beta=BETA, num_classes=NUM_CLASSES, reduction='mean')

        criterion_gcod = GCODLoss(num_train_samples=num_samples_for_gcod_u, 
                                      num_classes=NUM_CLASSES, 
                                      embedding_dim=GCOD_EMBEDDING_DIM, 
                                      device=device,
                                      initial_u_std=INITIAL_U_STD)
        optimizer_u = optim.SGD([criterion_gcod.u_all_samples], lr=LEARNING_RATE_U)

        # For GCOD training (needs indices)
        indexed_train_dataset = IndexedDataset(train_dataset)
        train_loader_for_gcod = DataLoader(indexed_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True) 

        # For evaluation (does not need indices from IndexedDataset directly in evaluate)
        # Pass the original PyG datasets to these loaders
        train_loader_for_eval = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True) # Use train_dataset
        vali_loader_for_eval = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True) # Use validation_dataset

        best_val_acc = 0
        best_epoch_val = 0
        current_epoch_train_acc = 0.0

        # Training loop
        for epoch_num in range(EPOCHS):
            model.eval() 
            all_train_embeddings_list = []
            all_train_labels_list = []
            # u_values should correspond to the samples in train_dataset
            u_values_for_centroids = criterion_gcod.u_all_samples.detach().clone()

            with torch.no_grad():
                # Create a temporary loader for train_dataset (not indexed_train_dataset)
                temp_centroid_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE * 2, shuffle=False)
                for data_item_for_centroid in temp_centroid_loader:
                    data_item_for_centroid = data_item_for_centroid.to(device)
                    # Model returns (logits, embeddings)
                    _, emb = model(data_item_for_centroid) 
                    all_train_embeddings_list.append(emb.cpu())
                    all_train_labels_list.append(data_item_for_centroid.y.cpu())

            if not all_train_embeddings_list: # Handle empty train_dataset case
                logging.warning(f"Epoch {epoch_num+1}: train_dataset is empty. Skipping centroid update.")
            else:
                all_train_embeddings_tensor = torch.cat(all_train_embeddings_list, dim=0)
                all_train_labels_tensor = torch.cat(all_train_labels_list, dim=0).squeeze()
                    
                criterion_gcod.update_class_centroids(
                    all_train_embeddings_tensor.to(device), 
                    all_train_labels_tensor.to(device), 
                    u_values_for_centroids.to(device), 
                    epoch_num, EPOCHS)
                

            model.train()

            train_loss_model, train_loss_u = train_gcod(
                    train_loader_for_gcod, model, criterion_gcod, 
                    optimizer, optimizer_u, device, 
                    current_epoch_train_acc
                )
            train_loss_display = train_loss_model
            
            current_epoch_train_acc, _ = evaluate(train_loader_for_eval, model, device, calculate_accuracy=True) # Uses updated model
            val_acc, _ = evaluate(vali_loader_for_eval, model, device, calculate_accuracy=True)
            
            log_msg_epoch = f"Epoch {epoch_num+1}/{EPOCHS}, Train Loss: {train_loss_display:.4f}"
            log_msg_epoch += f", U Loss: {train_loss_u:.4f}"
            log_msg_epoch += f", Train Acc: {current_epoch_train_acc:.4f}, Val Acc: {val_acc:.4f}"
            print(log_msg_epoch)
            if (epoch_num + 1) % 10 == 0: 
                logging.info(log_msg_epoch)
                with torch.no_grad(): # Ensure no gradients are computed for this
                    u_values = criterion_gcod.u_all_samples.cpu().numpy() # Move to CPU and convert to numpy
                    
                    u_min = np.min(u_values)
                    u_max = np.max(u_values)
                    u_mean = np.mean(u_values)
                    u_median = np.median(u_values)
                    u_std = np.std(u_values)
                    
                    # Count samples with u > threshold (e.g., 0.5)
                    threshold = 0.5
                    num_u_above_threshold = np.sum(u_values > threshold)
                    percent_u_above_threshold = (num_u_above_threshold / len(u_values)) * 100
                    
                    u_stats_msg = (
                        f"  U Stats (Epoch {epoch_num+1}): "
                        f"Min={u_min:.4f}, Max={u_max:.4f}, Mean={u_mean:.4f}, Median={u_median:.4f}, Std={u_std:.4f}, "
                        f"Num > {threshold}={num_u_above_threshold} ({percent_u_above_threshold:.2f}%)"
                    )
                    print(u_stats_msg)


        
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch_num + 1
                checkpoint_path = os.path.join(checkpoints_folder, f"model_{test_dir_name}_epoch_{best_epoch}.pth")
                torch.save({
                    'epoch': best_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), # Optional, if you might resume training
                    'best_val_acc': best_val_acc,
                    'norm_params': norm_params,
                    'gcod_embedding_dim': GCOD_EMBEDDING_DIM, # Save this
                    'u_values': criterion_gcod.u_all_samples.detach().cpu()
                }, checkpoint_path)
                print(f"+++++ Saved new best model at epoch {best_epoch} with val acc {best_val_acc:.4f}")
            

    

    
    
    best_epoch = max([int(checkpoint.split('_')[-1].split('.')[0]) for checkpoint in os.listdir(checkpoints_folder)])
    checkpoint = torch.load(os.path.join(checkpoints_folder, f"model_{test_dir_name}_epoch_{best_epoch}.pth"))
    norm_params = checkpoint['norm_params']
    normalizer = NormalizeNodeFeatures(norm_params_list=norm_params)

    print(f"Loading best model from epoch {best_epoch} for model")


    model = GINENetForGCOD(
        node_in_channels=NODE_FEATURE_DIM,
        edge_feature_dim=EDGE_FEATURE_DIM, # Pass edge feature dim
        gnn_hidden_channels=GIN_HIDDEN_CHANNELS,
        num_gnn_layers=GIN_NUM_LAYERS,
        mlp_hidden_channels=GIN_MLP_HIDDEN_CHANNELS,
        out_channels_final=NUM_CLASSES,
        dropout_rate=DROPOUT_RATE,
        return_embeddings=True,
        pooling_type='mean',
        eps=GINE_EPS,
        train_eps=GINE_TRAIN_EPS
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