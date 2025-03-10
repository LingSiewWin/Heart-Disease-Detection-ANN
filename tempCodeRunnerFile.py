import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, roc_auc_score)
from sklearn.utils import class_weight
import joblib

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Loading and Preprocessing
def load_and_prepare_data(filepath):
    data = pd.read_csv(filepath)
    
    # Check for missing values
    if data.isnull().any().any():
        print("Handling missing values...")
        # Add imputation strategy if needed
        
    # Check class distribution
    print("Class distribution in target:")
    print(data['target'].value_counts(normalize=True))
    
    X = data.iloc[:, :-1].values
    y = data['target'].values
    
    return X, y

# PyTorch Model Architecture
class HeartDiseaseModel(nn.Module):
    def __init__(self, input_dim):
        super(HeartDiseaseModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(32, 1)  # No sigmoid here
        )
    
    def forward(self, x):
        return self.model(x)

# Training with Cross-Validation
def train_and_evaluate(X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Standardize
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        
        # Convert to PyTorch tensors
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16)
        
        # Class weights
        class_weights = class_weight.compute_class_weight(
            'balanced', classes=np.unique(y_train), y=y_train)
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        
        # Initialize model, loss, optimizer
        model = HeartDiseaseModel(X_train.shape[1]).to(device)
        
        # Use BCEWithLogitsLoss and pos_weight for class balancing
        pos_weight = class_weights[1] / class_weights[0]  # Weight for class 1
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
        
        # Early stopping parameters
        best_val_loss = float('inf')
        patience = 15
        trigger_times = 0
        
        for epoch in range(200):
            # Training
            model.train()
            train_loss = 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            all_preds = []
            all_probs = []
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    
                    probs = torch.sigmoid(outputs).cpu().numpy()
                    preds = (probs > 0.5).astype(int)
                    all_preds.extend(preds)
                    all_probs.extend(probs)
            
            # Early stopping check
            avg_val_loss = val_loss / len(val_loader)
            scheduler.step(avg_val_loss)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                trigger_times = 0
                torch.save(model.state_dict(), f'model_fold_{fold}.pth')
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    print(f'Early stopping at epoch {epoch}')
                    break
        
        # Load best model and evaluate
        model.load_state_dict(torch.load(f'model_fold_{fold}.pth'))
        model.eval()
        all_preds = []
        all_probs = []
        with torch.no_grad():
            for inputs, _ in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                all_preds.extend(preds)
                all_probs.extend(probs)
        
        y_val = np.concatenate([y for _, y in val_loader])
        accuracy = accuracy_score(y_val, all_preds)
        roc_auc = roc_auc_score(y_val, all_probs)
        
        results.append({
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'report': classification_report(y_val, all_preds, output_dict=True)
        })
        
        # Save scaler
        joblib.dump(scaler, f'scaler_fold_{fold}.pkl')
    
    return results

# Main Execution
if __name__ == "__main__":
    X, y = load_and_prepare_data('Dataset--Heart-Disease-Prediction-using-ANN.csv')
    cv_results = train_and_evaluate(X, y)
    
    # Print cross-validation results
    for i, res in enumerate(cv_results, 1):
        print(f"\nFold {i} Results:")
        print(f"Accuracy: {res['accuracy']:.2f}")
        print(f"ROC AUC: {res['roc_auc']:.2f}")
        print("Classification Report:")
        print(pd.DataFrame(res['report']).transpose())