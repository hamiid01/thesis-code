import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from imblearn.metrics import geometric_mean_score
import logging
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MLP(nn.Module):
    """
    A simple Multi-Layer Perceptron model with dropout layers to reduce overfitting.
    """
    def __init__(self, input_size, output_size, device):
        super(MLP, self).__init__()
        h = 2 * (input_size + output_size) // 3
        self.device = device
        self.hidden = nn.Linear(input_size, h).to(device)
        self.dropout = nn.Dropout(0.5)
        self.output = nn.Linear(h, 1).to(device)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.dropout(x)
        x = self.output(x)
        return x

def load_data(train_features, train_labels, train_ids, test_ids, batch_size, device):
    """
    Load and create tensor datasets for both training and validation data.
    """
    features_tensor = torch.from_numpy(train_features).float().to(device)
    labels_tensor = torch.from_numpy(train_labels).float().to(device)
    dataset = TensorDataset(features_tensor, labels_tensor)

    train_subsampler = SubsetRandomSampler(train_ids)
    test_subsampler = SubsetRandomSampler(test_ids)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_subsampler)

    return train_loader, val_loader

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, early_stopping_patience):
    """
    Train the model and evaluate it on the validation set.
    """
    best_loss = float('inf')
    patience_counter = 0
    metrics = {'accuracy': [], 'f1': [], 'recall': [], 'roc_auc': [], 'g_mean': []}
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * features.size(0)

        train_loss /= len(train_loader.sampler)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        all_labels = []
        all_outputs = []
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device).unsqueeze(1)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * features.size(0)
                all_labels.extend(labels.cpu().numpy().flatten())  # Flatten the labels before extending
                all_outputs.extend(outputs.cpu().numpy().flatten())  # Flatten the outputs before extending

        val_loss /= len(val_loader.sampler)
        val_losses.append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            logging.info(f"Early stopping at epoch {epoch}")
            break

        all_outputs = np.array(all_outputs)  # convert all_outputs to an array
        all_outputs = torch.sigmoid(torch.tensor(all_outputs)).numpy()
        preds = (all_outputs > 0.5).astype(int)
        all_labels = np.array(all_labels).flatten()  # convert all_labels to a flattened array
        metrics['accuracy'].append(accuracy_score(all_labels, preds))
        metrics['f1'].append(f1_score(all_labels, preds))
        metrics['recall'].append(recall_score(all_labels, preds))
        metrics['roc_auc'].append(roc_auc_score(all_labels, all_outputs))
        metrics['g_mean'].append(geometric_mean_score(all_labels, preds))

        #logging.info(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss}, Validation Loss: {val_loss}")

    return metrics, train_losses, val_losses

def main(train_data, config):
    """
    Main function to run the training and testing of the MLP model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.BCEWithLogitsLoss()

    train_features = train_data.drop('class_label', axis=1).values
    train_labels = train_data['class_label'].values

    skf = StratifiedKFold(n_splits=config['k_folds'], shuffle=True, random_state=config['random_seed'])

    overall_metrics = {'accuracy': [], 'f1': [], 'recall': [], 'roc_auc': [], 'g_mean': []}
    overall_train_losses = []
    overall_val_losses = []

    for repeat in range(10):
        exp_metrics = {'accuracy': [], 'f1': [], 'recall': [], 'roc_auc': [], 'g_mean': []}
        exp_train_losses = []
        exp_val_losses = []

        logging.info(f"Starting repeat {repeat + 1} of 10")
        for fold, (train_ids, test_ids) in enumerate(skf.split(train_features, train_labels)):
            #logging.info(f"Starting fold {fold + 1} of {config['k_folds']}")
            train_loader, val_loader = load_data(train_features, train_labels, train_ids, test_ids, config['batch_size'], device)
            model = MLP(train_features.shape[1], len(np.unique(train_labels)), device)
            optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
            fold_metrics, train_losses, val_losses = train_model(model, train_loader, val_loader, criterion,
                                                                 optimizer, device, config['num_epochs'], config['early_stopping_patience'])

            exp_train_losses.extend(train_losses)  # Flatten the list of train losses
            exp_val_losses.extend(val_losses)  # Flatten the list of val losses

            for key in overall_metrics:
                overall_metrics[key].append(np.mean(fold_metrics[key]))
                exp_metrics[key].append(np.mean(fold_metrics[key]))

        avg_metrics_fold = {key: np.mean(exp_metrics[key]) for key in exp_metrics}
        overall_train_losses.extend(exp_train_losses)  # Flatten the list of train losses
        overall_val_losses.extend(exp_val_losses)  # Flatten the list of val losses

        logging.info(f"Experiment Average Metrics: {avg_metrics_fold}")
        logging.info(f"Experiment Train Losses: {np.mean(exp_train_losses, axis=0)}")
        logging.info(f"Experiment Validation Losses: {np.mean(exp_val_losses, axis=0)}")

    avg_metrics = {key: np.mean(overall_metrics[key]) for key in overall_metrics}
    logging.info(f"Overall Average Metrics: {avg_metrics}")
    logging.info(f"Overall Train Losses: {np.mean(overall_train_losses, axis=0)}")
    logging.info(f"Overall Validation Losses: {np.mean(overall_val_losses, axis=0)}")

    return avg_metrics
