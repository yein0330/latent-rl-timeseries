"""VAE training"""
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


def train_vae(autoencoder, data, window_size=96, epochs=100, batch_size=64,
              kl_weight=0.01, device='cuda', verbose=True):
    """Train VAE"""
    autoencoder.train()
    optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)
    
    windows = []
    for i in range(len(data) - window_size):
        windows.append(data[i:i+window_size])
    windows = np.array(windows)
    
    num_batches = len(windows) // batch_size
    best_loss = float('inf')
    patience = 0
    
    for epoch in range(epochs):
        total_loss = 0
        indices = np.random.permutation(len(windows))
        
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = start + batch_size
            batch = windows[indices[start:end]]
            batch_tensor = torch.FloatTensor(batch).to(device)
            
            x_recon, mu, logvar = autoencoder(batch_tensor)
            recon_loss = F.mse_loss(x_recon, batch_tensor)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_loss = kl_loss / batch_tensor.size(0)
            loss = recon_loss + kl_weight * kl_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        
        if verbose and (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience = 0
        else:
            patience += 1
        
        if patience >= 10:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            break
    
    autoencoder.eval()
    return best_loss
