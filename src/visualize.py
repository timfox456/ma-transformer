import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from model import TransformerModel
from data_loader import load_and_preprocess_data
import numpy as np

# --- CONFIGURATION ---
MODEL_PATH = '../models/best_model_from_search.pth'
# Use the best hyperparameters from the search
BEST_HYPERPARAMS = {
    'learning_rate': 0.001, 
    'model_dim': 64, 
    'num_heads': 2, 
    'num_layers': 4
}
INPUT_DIM = 3
SEQUENCE_LENGTH = 10
BATCH_SIZE = 64 # Should be the same as in training for consistency

# --- DATA LOADING ---
file_path = '../data/synthetic_ticks_custom.csv'
(_, _), (_, _), (X_test, y_test) = load_and_preprocess_data(file_path, SEQUENCE_LENGTH)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- MODEL LOADING ---
model = TransformerModel(
    input_dim=INPUT_DIM,
    model_dim=BEST_HYPERPARAMS['model_dim'],
    num_heads=BEST_HYPERPARAMS['num_heads'],
    num_layers=BEST_HYPERPARAMS['num_layers']
)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()
print("Model loaded successfully.")

# --- PREDICTION ---
all_predictions = []
all_actuals = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X = batch_X.permute(1, 0, 2)
        outputs = model(batch_X)
        # The output is of shape (sequence_length, batch_size, 1)
        # We are interested in the prediction for the next time step, which is the last element of the sequence output
        # However, our current model outputs a prediction for each input time step.
        # For this architecture, we'll take the output corresponding to the last input time step.
        predictions = outputs[-1, :, :].squeeze()
        all_predictions.extend(predictions.tolist())
        all_actuals.extend(batch_y.tolist())

print(f"Generated {len(all_predictions)} predictions.")

# --- VISUALIZATION ---
plt.figure(figsize=(15, 7))
plt.plot(all_actuals, label='Actual Prices', color='blue', alpha=0.7)
plt.plot(all_predictions, label='Predicted Prices', color='red', linestyle='--', alpha=0.7)
plt.title('Model Predictions vs. Actual Prices on Test Data')
plt.xlabel('Time Step')
plt.ylabel('Price')
plt.legend()
plt.grid(True)

# Save the plot
output_path = '../prediction_visualization.png'
plt.savefig(output_path)
print(f"Plot saved to {output_path}")

plt.show()
