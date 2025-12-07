"""VAE Autoencoder"""
import torch.nn as nn


class LatentAutoEncoder(nn.Module):
    """VAE"""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar
