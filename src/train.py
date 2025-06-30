import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from data_loader import load_and_preprocess_data
from model import TransformerModel
import pandas as pd
import numpy as np
import itertools

# --- HYPERPARAMETER GRID ---
param_grid = {
    'learning_rate': [0.001, 0.0005],
    'model_dim': [32, 64],
    'num_heads': [2, 4],
    'num_layers': [2, 4]
}

# --- STATIC PARAMS ---
sequence_length = 10
batch_size = 64
epochs = 100
patience = 5
input_dim = 3 # From our feature engineering

# --- DATA LOADING ---
file_path = '../data/synthetic_ticks_custom.csv'
(X_train, y_train), (X_val, y_val), (X_test, y_test) = load_and_preprocess_data(file_path, sequence_length)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# --- GRID SEARCH ---
best_global_val_loss = float('inf')
best_hyperparameters = None
keys, values = zip(*param_grid.items())

print("Starting Hyperparameter Search...")

for v in itertools.product(*values):
    params = dict(zip(keys, v))
    print("-" * 40)
    print(f"Testing hyperparameters: {params}")

    # --- MODEL INITIALIZATION ---
    model = TransformerModel(
        input_dim=input_dim,
        model_dim=params['model_dim'],
        num_heads=params['num_heads'],
        num_layers=params['num_layers']
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

    # --- TRAINING & EARLY STOPPING ---
    best_local_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.permute(1, 0, 2)
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.permute(1, 0, 2)
                outputs = model(batch_X)
                val_loss += criterion(outputs.squeeze(), batch_y).item()
        val_loss /= len(val_loader)

        if (epoch + 1) % 10 == 0:
             print(f'Epoch [{epoch+1}/{epochs}], Val Loss: {val_loss:.4f}')

        if val_loss < best_local_val_loss:
            best_local_val_loss = val_loss
            epochs_no_improve = 0
            # Temporarily save the best model for this run
            torch.save(model.state_dict(), 'temp_best_model.pth')
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f'Early stopping at epoch {epoch+1}.')
            break
    
    print(f"Best validation loss for this run: {best_local_val_loss:.4f}")

    if best_local_val_loss < best_global_val_loss:
        best_global_val_loss = best_local_val_loss
        best_hyperparameters = params
        print(f"!!! New best model found. Saving to 'best_model_from_search.pth' !!!")
        # This is the new best model overall, so we save it
        torch.save(torch.load('temp_best_model.pth'), 'best_model_from_search.pth')


print("\n" + "="*40)
print("Hyperparameter search finished.")
print(f"Best Validation Loss: {best_global_val_loss:.4f}")
print(f"Best Hyperparameters: {best_hyperparameters}")
print("="*40)

# --- FINAL EVALUATION ---
print("\nLoading best model from search for final evaluation...")
best_model_params = best_hyperparameters
final_model = TransformerModel(
    input_dim=input_dim,
    model_dim=best_model_params['model_dim'],
    num_heads=best_model_params['num_heads'],
    num_layers=best_model_params['num_layers']
)
final_model.load_state_dict(torch.load('best_model_from_search.pth'))
final_model.eval()

test_loss = 0
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X = batch_X.permute(1, 0, 2)
        outputs = final_model(batch_X)
        test_loss += criterion(outputs.squeeze(), batch_y).item()

test_loss /= len(test_loader)
print(f'\nFinal Test MSE with best model: {test_loss:.4f}')