"""Encoder architectures for VAE"""
import torch
import torch.nn as nn


class MLPEncoder(nn.Module):
    """MLP Encoder"""
    def __init__(self, window_size=96, latent_dim=32, hidden_dims=[128, 128], dropout=0.3):
        super().__init__()
        layers = []
        in_dim = window_size
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = h_dim
        
        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
    
    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar
    
    def encode_deterministic(self, x):
        h = self.encoder(x)
        return self.fc_mu(h)


class Conv1DEncoder(nn.Module):
    """Conv1D Encoder"""
    def __init__(self, window_size=96, latent_dim=32, hidden_dims=[64, 128, 256], dropout=0.3):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(1, hidden_dims[0], kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden_dims[0], hidden_dims[1], kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(2)
        )
        
        conv_out_size = window_size // 4
        fc_input_size = hidden_dims[1] * conv_out_size
        
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, hidden_dims[2]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.fc_mu = nn.Linear(hidden_dims[2], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[2], latent_dim)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        return z, mu, logvar
    
    def encode_deterministic(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.fc_mu(x)


class LSTMEncoder(nn.Module):
    """LSTM Encoder"""
    def __init__(self, window_size=96, latent_dim=32, hidden_dims=[128, 128], dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dims[0],
            num_layers=len(hidden_dims),
            dropout=dropout if len(hidden_dims) > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
    
    def forward(self, x):
        x = x.unsqueeze(-1)
        _, (h_n, _) = self.lstm(x)
        h = h_n[-1]
        h = self.fc(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar
    
    def encode_deterministic(self, x):
        x = x.unsqueeze(-1)
        _, (h_n, _) = self.lstm(x)
        h = h_n[-1]
        h = self.fc(h)
        return self.fc_mu(h)


class TransformerEncoder(nn.Module):
    """Transformer Encoder"""
    def __init__(self, window_size=96, latent_dim=32, hidden_dims=[128, 128], dropout=0.3):
        super().__init__()
        d_model = hidden_dims[0]
        
        self.input_proj = nn.Linear(1, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, window_size, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=len(hidden_dims))
        
        self.fc = nn.Sequential(
            nn.Linear(d_model, hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
    
    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.input_proj(x)
        x = x + self.pos_encoding
        x = self.transformer(x)
        h = x.mean(dim=1)
        h = self.fc(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar
    
    def encode_deterministic(self, x):
        x = x.unsqueeze(-1)
        x = self.input_proj(x)
        x = x + self.pos_encoding
        x = self.transformer(x)
        h = x.mean(dim=1)
        h = self.fc(h)
        return self.fc_mu(h)
