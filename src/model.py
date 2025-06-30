import torch
import torch.nn as nn
import math

class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.model_dim = model_dim
        self.embedding = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(model_dim, num_heads, model_dim * 4, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(model_dim, 1)

    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.model_dim)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

if __name__ == '__main__':
    # Example usage
    input_dim = 1
    model_dim = 32
    num_heads = 2
    num_layers = 2
    
    model = TransformerModel(input_dim, model_dim, num_heads, num_layers)
    
    # Create a dummy input tensor
    # (sequence_length, batch_size, input_dim)
    src = torch.rand((10, 64, input_dim))
    
    output = model(src)
    
    print("Model architecture:", model)
    print("Input shape:", src.shape)
    print("Output shape:", output.shape)
