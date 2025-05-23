import os
import datetime
import torch
import logging
import argparse
from source.loadData import GraphDataset
from torch_geometric.loader import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import source.GNNPlus  # noqa, register custom modules
from source.GNNPlus.optimizer.extra_optimizers import ExtendedSchedulerConfig
from source.loss import SCELoss
from torch import optim

from torch.utils.data import Subset

#from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (cfg, set_cfg, load_cfg)
#from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.logger import set_printing
from torch_geometric.graphgym.optim import create_optimizer, \
    create_scheduler, OptimizerConfig
from torch_geometric.graphgym.model_builder import create_model
#from torch_geometric.graphgym.train import GraphGymDataModule
from torch_geometric.graphgym.utils.comp_budget import params_count
#from torch_geometric.graphgym.utils.device import auto_select_device
#from torch_geometric.graphgym.register import train_dict
from torch_geometric import seed_everything

#from source.GNNPlus.finetuning import load_pretrained_model_cfg, init_model_from_pretrained
#from source.GNNPlus.logger import create_logger

torch.backends.cuda.matmul.allow_tf32 = True  # Default False in PyTorch 1.12+
torch.backends.cudnn.allow_tf32 = True  # Default True

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

      
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class SimpleGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate=0.2): # Reduced dropout for small data
        super().__init__()
        # If no node features (in_channels=0 or None), use an embedding for nodes
        # For now, assuming in_channels > 0 (actual node features)
        if in_channels <= 0: # Handle case with no initial features by using node degree or an embedding
            print("Warning: in_channels <=0. Consider using node embeddings or degrees as features.")
            # As a placeholder, let's assume we'll create a dummy feature if none.
            # This part needs to be adapted to your actual data.
            # If you have node degrees, that's a common GCN starting point.
            # Or you can create learnable embeddings per node if num_nodes is fixed,
            # but that's tricky for graph-batched data unless it's within the forward.
            # For simplicity, let's assume you will provide some `data.x`.
            # If not, this model needs adjustment.
            self.uses_dummy_features = True # Flag this
            self.dummy_feature_dim = hidden_channels # Create dummy features of this size
            self.conv1 = GCNConv(self.dummy_feature_dim, hidden_channels)
        else:
            self.uses_dummy_features = False
            self.conv1 = GCNConv(in_channels, hidden_channels)

        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout_rate = dropout_rate

    def forward(self, data): # PyG convention often passes the whole Data object
        x, edge_index, batch = data.x, data.edge_index, data.batch

        if self.uses_dummy_features and x is None:
            # Create dummy features if no 'x' and configured to do so.
            # This is a basic way; a learnable embedding per node would be better
            # if node identities are consistent and meaningful across graphs.
            # For now, let's use a constant feature for all nodes if x is missing.
            # This isn't ideal but makes the GCN runnable.
            # A better approach for no features is to use torch.eye(data.num_nodes)
            # but that's for single graphs, not batches easily.
            # Or use node degrees.
            # For now, let's just illustrate the GCN structure.
            # YOU WILL LIKELY NEED TO ADJUST FEATURE HANDLING HERE.
            # Example: x = torch.ones((data.num_nodes, self.dummy_feature_dim), device=edge_index.device)
            pass # This part highly depends on how you want to handle no features

        if x is None:
            raise ValueError("Node features 'x' are None. The model needs node features or specific handling for featureless graphs.")


        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)

        # If graph classification, add pooling:
        from torch_geometric.nn import global_mean_pool # Or global_add_pool, etc.
        x_pooled = global_mean_pool(x, batch) # `batch` vector from DataLoader is crucial
        return x_pooled # Logits for graph classification

    

def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)  
    return data


def train(data_loader, model, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data in data_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)[0]  # Assuming model returns a tuple
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
            output = model(data)[0]  # Assuming model returns a tuple
            pred = output.argmax(dim=1)
            predictions.extend(pred.cpu().numpy())
            if calculate_accuracy:
                correct += (pred == data.y).sum().item()
                total += data.y.size(0)
            
    if calculate_accuracy:
        accuracy = correct / total
        return accuracy, predictions
    return predictions

def new_optimizer_config(cfg):
    return OptimizerConfig(optimizer=cfg.optim.optimizer,
                           base_lr=cfg.optim.base_lr,
                           weight_decay=cfg.optim.weight_decay,
                           momentum=cfg.optim.momentum)


def new_scheduler_config(cfg):
    return ExtendedSchedulerConfig(
        scheduler=cfg.optim.scheduler,
        steps=cfg.optim.steps, lr_decay=cfg.optim.lr_decay,
        max_epoch=cfg.optim.max_epoch, reduce_factor=cfg.optim.reduce_factor,
        schedule_patience=cfg.optim.schedule_patience, min_lr=cfg.optim.min_lr,
        num_warmup_epochs=cfg.optim.num_warmup_epochs,
        train_mode=cfg.train.mode, eval_period=cfg.train.eval_period)

def main(args):
    num_epochs = 30  # Number of epochs for training
    lr = 1e-3  # Learning rate
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir_name = os.path.basename(os.path.dirname(args.test_path))

    # Define log file path relative to the script's directory
    logs_folder = os.path.join(script_dir, "logs", test_dir_name)
    log_file = os.path.join(logs_folder, "training.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler()) 

    # Define checkpoint path relative to the script's directory
    checkpoints_folder = os.path.join(script_dir, "checkpoints", test_dir_name)
    os.makedirs(checkpoints_folder, exist_ok=True)

    seed_everything(42)  # Set random seed for reproducibility

    os.makedirs(os.getcwd().join('submission'), exist_ok=True)

    # Model import
    set_cfg(cfg)
    load_cfg(cfg, args)
    set_printing()
    cfg.accelerator = device
    model = create_model(dim_out=6)  # Assuming 6 classes for classification
    cfg.params = params_count(model)
    logging.info('Num parameters: %s', cfg.params)
    print("gnn dropout: ", cfg.gnn.dropout)

    


    device = torch.device(device)

    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    alpha_val = 1.0
    beta_val = 0.0
    criterion = SCELoss(alpha=alpha_val, beta=beta_val, num_classes=6, reduction='mean')

    


    if args.train_path:

        # Remove previous checkpoints for the same test dataset
        for filePath in os.listdir(checkpoints_folder):
            if test_dir_name in filePath:
                os.remove(filePath)
                print(f"Removed previous checkpoint: {filePath}")

        subset_size_desired = 100 
        train_dataset = GraphDataset(args.train_path, transform=add_zeros)
        indices_for_subset = list(range(min(subset_size_desired, train_dataset.len())))
        train_subset = Subset(train_dataset, indices_for_subset)
        labels = []
        for i in range(len(train_subset)):
            labels.append(train_subset[i].y.item()) # .item() if y is a 0-dim tensor

        
        if len(train_subset) > 0:
            sample_data = train_subset[0]
            if sample_data.x is not None:
                IN_CHANNELS = sample_data.x.size(1)
                print(f"Determined IN_CHANNELS from data: {IN_CHANNELS}")
            else:
                print("ERROR: sample_data.x is None! Cannot determine IN_CHANNELS.")
                print("You need to provide node features or use a transform to create them.")
                IN_CHANNELS = -1 # Placeholder, will cause error
        else:
            print("ERROR: subset_dataset is empty!")
            IN_CHANNELS = -1 # Placeholder

        HIDDEN_CHANNELS = 64 # Example, tune this
        NUM_CLASSES = 6    # For your subset
        LEARNING_RATE = 1e-3
        EPOCHS = 50 # Increase for the small subset
        WEIGHT_DECAY = 1e-4 # Add some regularization

        model = SimpleGCN(in_channels=IN_CHANNELS,
                      hidden_channels=HIDDEN_CHANNELS,
                      out_channels=NUM_CLASSES).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        criterion = torch.nn.CrossEntropyLoss() # Standard CE for now



        train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
        
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

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            if (epoch + 1) % 5 == 0:
                logging.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

                # Save best model
            if train_acc > best_accuracy:
                best_accuracy = train_acc
                checkpoint_path = os.path.join(script_dir, "checkpoints", f"model_{test_dir_name}_epoch_{epoch+1}.pth")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Best model updated and saved at {checkpoint_path}")




    epoch_best_model = max([int(checkpoint.split('_')[-1].split('.')[0]) for checkpoint in os.listdir(checkpoints_folder)])
    best_model_state_dict = torch.load(os.path.join(checkpoints_folder, f"model_{test_dir_name}_epoch_{epoch_best_model}.pth"))
    model = create_model(dim_out=6)  # Assuming 6 classes for classification
    model.load_state_dict(best_model_state_dict)

     # Prepare test dataset and loader
    test_dataset = GraphDataset(args.test_path, transform=add_zeros)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

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
    #simulating what parse_args from graphgym does
    parser.add_argument('--cfg', dest='cfg_file', type=str,required=False, default=os.getcwd()+'/DL2025/source/configs/gatedgcn/ppa.yaml')
    parser.add_argument('--repeat', type=int, default=1, help='The number of repeated jobs.')
    parser.add_argument('--mark_done', action='store_true',help='Mark yaml as done after a job has finished.')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER, help='See graphgym/config.py for remaining options.')
    args = parser.parse_args()
    main(args)