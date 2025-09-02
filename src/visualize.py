 # SPDX-License-Identifier: Apache-2.0
import torch
import matplotlib.pyplot as plt
import pandas as pd
from src.data_loader import TickDataset, create_data_loaders
from src.model import TransformerModel
import yaml
import argparse
import numpy as np

def visualize_predictions(config, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    _, _, dataset = create_data_loaders(
        config['file_path'], 
        config['sequence_length'], 
        config['batch_size']
    )
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Load model
    best_hyperparams = {
        'model_dim': 32,
        'num_heads': 2,
        'num_layers': 2
    }

    model = TransformerModel(
        input_dim=config['input_dim'],
        model_dim=best_hyperparams['model_dim'],
        num_heads=best_hyperparams['num_heads'],
        num_layers=best_hyperparams['num_layers']
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    actuals = []
    predictions = []
    
    with torch.no_grad():
        for features, target in loader:
            features, target = features.to(device), target.to(device)
            output = model(features)
            predictions.extend(output.cpu().numpy().flatten())
            actuals.extend(target.cpu().numpy().flatten())
            
    # Un-normalize the data to see the real price movements
    wap_mean = dataset.feature_means['wap']
    wap_std = dataset.feature_stds['wap']
    
    actuals_price = [a * wap_std + wap_mean for a in actuals]
    predictions_price = [p * wap_std + wap_mean for p in predictions]

    plt.figure(figsize=(15, 7))
    plt.plot(actuals_price, label='Actual WAP', color='blue', alpha=0.7)
    plt.plot(predictions_price, label='Predicted WAP', color='red', linestyle='--', alpha=0.7)
    plt.title('Model Predictions vs. Actual WAP')
    plt.xlabel('Time Step')
    plt.ylabel('WAP')
    plt.legend()
    plt.grid(True)
    plt.savefig('prediction_visualization.png')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize model predictions.")
    parser.add_argument('--config', type=str, default='configs/default_config.yaml', help='Path to config file')
    parser.add_argument('--model_path', type=str, default='models/best_model_from_search.pth', help='Path to the trained model')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    visualize_predictions(config, args.model_path)
