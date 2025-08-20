import torch
import torch.nn as nn
from torch.optim import Adam
from src.data_loader import create_data_loaders
from src.model import TransformerModel
import argparse
import yaml
import os
from itertools import product
import numpy as np

def train_model(config):
    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # DataLoaders
    train_loader, val_loader, _ = create_data_loaders(
        config['file_path'], 
        config['sequence_length'], 
        config['batch_size']
    )
    
    # Hyperparameter search
    best_val_loss = float('inf')
    best_model_state = None
    best_hyperparams = {}

    param_grid = list(product(
        config['param_grid']['learning_rate'],
        config['param_grid']['model_dim'],
        config['param_grid']['num_heads'],
        config['param_grid']['num_layers']
    ))

    for lr, model_dim, num_heads, num_layers in param_grid:
        print(f"Training with lr={lr}, model_dim={model_dim}, num_heads={num_heads}, num_layers={num_layers}")

        model = TransformerModel(
            input_dim=config['input_dim'], 
            model_dim=model_dim, 
            num_heads=num_heads, 
            num_layers=num_layers
        ).to(device)
        
        criterion = nn.MSELoss()
        optimizer = Adam(model.parameters(), lr=lr)
        
        # Early stopping
        patience_counter = 0
        
        for epoch in range(config['epochs']):
            model.train()
            total_train_loss = 0
            for batch_features, batch_target in train_loader:
                batch_features, batch_target = batch_features.to(device), batch_target.to(device)
                
                optimizer.zero_grad()
                output = model(batch_features).squeeze(-1)
                loss = criterion(output, batch_target)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader)
            
            # Validation
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch_features, batch_target in val_loader:
                    batch_features, batch_target = batch_features.to(device), batch_target.to(device)
                    output = model(batch_features).squeeze(-1)
                    loss = criterion(output, batch_target)
                    total_val_loss += loss.item()
            
            avg_val_loss = total_val_loss / len(val_loader)
            
            print(f"Epoch {epoch+1}/{config['epochs']}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict()
                best_hyperparams = {
                    'learning_rate': lr,
                    'model_dim': model_dim,
                    'num_heads': num_heads,
                    'num_layers': num_layers
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config['patience']:
                    print("Early stopping triggered.")
                    break
    
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best hyperparameters: {best_hyperparams}")
    
    # Save the best model
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(best_model_state, 'models/best_model_from_search.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Transformer model.")
    parser.add_argument('--config', type=str, default='configs/default_config.yaml', help='Path to config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    train_model(config)