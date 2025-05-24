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

from source.loss import SCELoss
from torch import optim

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from torch.utils.data import Subset


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

    if args.train_path:

        

        # Remove previous checkpoints for the same test dataset
        for filePath in os.listdir(checkpoints_folder):
            if test_dir_name in filePath:
                os.remove(filePath)
                print(f"Removed previous checkpoint: {filePath}")


        train_dataset = GraphDataset(args.train_path, transform=my_transform)

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
        
        NODE_IN_CHANNELS = 2    # e.g., from your degree + degree_sq features
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
    test_dataset = GraphDataset(raw_filename=args.test_path, transform=my_transform)
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