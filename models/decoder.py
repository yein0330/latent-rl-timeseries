"""Decoder for VAE"""
import torch
import torch.nn as nn


class Decoder(nn.Module):
    """Decoder"""
    def __init__(self, latent_dim=32, window_size=96, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, window_size)
        )
        self.output_scale = nn.Parameter(torch.ones(1))
        self.output_shift = nn.Parameter(torch.zeros(1))
    
    def forward(self, z):
        x = self.decoder(z)
        return x * self.output_scale + self.output_shift
