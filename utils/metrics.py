"""Evaluation metrics"""
import torch
import torch.nn.functional as F
import numpy as np


def extract_trajectory(autoencoder, data, start_idx, window_size=96,
                      num_steps=20, stride=5, device='cuda'):
    """Extract latent trajectory"""
    latent_traj = []
    window_traj = []
    
    with torch.no_grad():
        for i in range(num_steps + 1):
            window_start = start_idx + (i * stride)
            window_end = window_start + window_size
            
            window = data[window_start:window_end]
            window_tensor = torch.FloatTensor(window).to(device)
            
            z = autoencoder.encoder.encode_deterministic(
                window_tensor.unsqueeze(0)
            ).squeeze(0)
            
            latent_traj.append(z)
            window_traj.append(window_tensor)
    
    return torch.stack(latent_traj), torch.stack(window_traj)


def evaluate_policy(policy, autoencoder, latent_traj, window_traj, device='cuda'):
    """Evaluate policy"""
    num_steps = len(latent_traj) - 1
    z_current = latent_traj[0].clone()
    z_trajectory = [z_current]
    
    with torch.no_grad():
        for t in range(num_steps):
            state = z_current.unsqueeze(0)
            
            if hasattr(policy, 'select_action'):
                action = policy.select_action(state, evaluate=True).squeeze(0)
            else:
                action = policy(state).squeeze(0)
            
            z_current = z_current + action
            z_trajectory.append(z_current)
    
    distances = []
    for i, z in enumerate(z_trajectory):
        dist = torch.norm(z - latent_traj[i]).item()
        distances.append(dist)
    
    avg_distance = np.mean(distances)
    final_distance = distances[-1]
    
    final_recon = autoencoder.decoder(z_trajectory[-1].unsqueeze(0)).squeeze(0)
    final_target = window_traj[-1]
    recon_mse = F.mse_loss(final_recon, final_target).item()
    
    return {
        'avg_distance': avg_distance,
        'final_distance': final_distance,
        'recon_mse': recon_mse,
        'distances': distances
    }
