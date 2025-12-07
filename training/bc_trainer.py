"""Behavioral Cloning training"""
import torch
import torch.nn.functional as F
import torch.optim as optim


def train_behavioral_cloning(policy, latent_traj, epochs=500, device='cuda', verbose=True):
    """Train BC policy"""
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    num_steps = len(latent_traj) - 1
    best_loss = float('inf')
    patience = 0
    losses = []
    
    for epoch in range(epochs):
        total_loss = 0
        
        for t in range(num_steps):
            state = latent_traj[t].unsqueeze(0)
            target_action = latent_traj[t+1] - latent_traj[t]
            
            pred_action = policy(state).squeeze(0)
            loss = F.mse_loss(pred_action, target_action)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / num_steps
        losses.append(avg_loss)
        
        if verbose and (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience = 0
        else:
            patience += 1
        
        if patience >= 30 or avg_loss < 1e-5:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            break
    
    return losses, best_loss
