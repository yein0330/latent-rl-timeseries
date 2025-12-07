"""
Conv1D_BC Complete Forecasting Pipeline
Active Pattern Discovery + Multi-Horizon Forecasting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import json
import pickle

from models.encoders import Conv1DEncoder
from models.decoder import Decoder
from models.autoencoder import LatentAutoEncoder
from models.policy import Actor
from utils.data_loader import load_data, find_smooth_segment


def load_conv1d_bc_model(base_dir, device='cuda'):
    """Load Conv1D_BC model"""
    print(f"📂 Loading Conv1D_BC model...")
    
    exp_name = 'Encoder_Conv1D_BC'
    exp_dir = os.path.join(base_dir, 'experiments', exp_name)
    
    config_path = os.path.join(exp_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"✓ Config loaded")
    
    hidden_str = '_'.join(map(str, config['hidden_dims']))
    checkpoint_name = (f"vae_{config['encoder_type']}_"
                      f"ws{config['window_size']}_"
                      f"ld{config['latent_dim']}_"
                      f"hd{hidden_str}.pt")
    checkpoint_path = os.path.join(base_dir, 'checkpoints', checkpoint_name)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    encoder = Conv1DEncoder(
        config['window_size'],
        config['latent_dim'],
        config['hidden_dims']
    ).to(device)
    
    decoder = Decoder(config['latent_dim'], config['window_size']).to(device)
    
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    
    autoencoder = LatentAutoEncoder(encoder, decoder)
    autoencoder.eval()
    
    print(f"✓ Model loaded")
    
    return autoencoder, config


def train_policy_on_teacher(autoencoder, train_data, config, device='cuda'):
    """Train policy on teacher trajectory"""
    print(f"\n🎓 Training policy...")
    
    window_size = config['window_size']
    latent_dim = config['latent_dim']
    num_steps = 20
    stride = 5
    
    teacher_start = find_smooth_segment(train_data, window_size, num_steps, stride)
    
    latent_traj = []
    with torch.no_grad():
        for i in range(num_steps + 1):
            ws = teacher_start + (i * stride)
            we = ws + window_size
            
            window = train_data[ws:we]
            window_tensor = torch.FloatTensor(window).to(device)
            
            z = autoencoder.encoder.encode_deterministic(
                window_tensor.unsqueeze(0)
            ).squeeze(0)
            
            latent_traj.append(z)
    
    latent_traj = torch.stack(latent_traj)
    
    policy = Actor(latent_dim, latent_dim, action_scale=config['action_scale']).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    
    for epoch in range(500):
        total_loss = 0
        
        for t in range(len(latent_traj) - 1):
            state = latent_traj[t].unsqueeze(0)
            target = latent_traj[t+1] - latent_traj[t]
            
            pred = policy(state).squeeze(0)
            loss = F.mse_loss(pred, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / (len(latent_traj) - 1)
        
        if avg_loss < 1e-5:
            break
    
    policy.eval()
    
    print(f"✓ Policy trained (loss: {avg_loss:.6f}, teacher: {teacher_start})")
    
    return policy, teacher_start, latent_traj


def evaluate_segment(policy, autoencoder, data, start_idx,
                     window_size, num_steps, stride, device):
    """Evaluate segment"""
    latent_traj = []
    
    with torch.no_grad():
        for i in range(num_steps + 1):
            ws = start_idx + (i * stride)
            we = ws + window_size
            
            if we > len(data):
                return None
            
            window = data[ws:we]
            window_tensor = torch.FloatTensor(window).to(device)
            
            z = autoencoder.encoder.encode_deterministic(
                window_tensor.unsqueeze(0)
            ).squeeze(0)
            
            latent_traj.append(z)
    
    latent_traj = torch.stack(latent_traj)
    
    with torch.no_grad():
        z_current = latent_traj[0].clone()
        errors = []
        mags = []
        
        for t in range(num_steps):
            action = policy(z_current.unsqueeze(0)).squeeze(0)
            mags.append(torch.norm(action).item())
            
            z_next = z_current + action
            error = torch.norm(z_next - latent_traj[t+1]).item()
            errors.append(error)
            
            z_current = z_next
    
    avg_error = np.mean(errors)
    smoothness = -np.std(np.diff(mags))
    confidence = 1.0 / (1.0 + avg_error) + smoothness * 0.1
    
    return {
        'avg_tracking_error': avg_error,
        'confidence_score': confidence,
        'latent_trajectory': latent_traj.cpu().numpy(),
        'start_idx': start_idx
    }


def active_discovery(policy, autoencoder, data, teacher_start,
                     window_size, num_steps, stride,
                     top_k=5, min_gap=300, sample_stride=50, device='cuda'):
    """Active pattern discovery"""
    print(f"\n🔍 Active Discovery...")
    
    candidates = []
    required = (num_steps * stride) + window_size
    
    num_samples = 0
    for start_idx in range(0, len(data) - required, sample_stride):
        
        if abs(start_idx - teacher_start) < min_gap:
            continue
        
        result = evaluate_segment(
            policy, autoencoder, data, start_idx,
            window_size, num_steps, stride, device
        )
        
        if result is not None:
            candidates.append(result)
            num_samples += 1
            
            if num_samples % 50 == 0:
                print(f"  {num_samples} segments...")
    
    print(f"✓ Evaluated {num_samples}")
    
    candidates.sort(key=lambda x: x['confidence_score'], reverse=True)
    
    selected = []
    for cand in candidates:
        is_overlap = False
        for prev in selected:
            if abs(cand['start_idx'] - prev['start_idx']) < min_gap:
                is_overlap = True
                break
        
        if not is_overlap:
            selected.append(cand)
        
        if len(selected) >= top_k:
            break
    
    return selected


def multi_horizon_forecast(policy, autoencoder, train_data, teacher_start,
                           window_size, num_steps, stride, horizons, device):
    """Multi-horizon forecasting"""
    print(f"\n📈 Multi-horizon forecasting...")
    
    last_idx = teacher_start + (num_steps * stride)
    results = []
    
    for horizon in horizons:
        true_futures = []
        for h in range(1, horizon + 1):
            future_idx = last_idx + (h * stride)
            if future_idx + window_size > len(train_data):
                break
            true_futures.append(train_data[future_idx:future_idx + window_size])
        
        actual_h = len(true_futures)
        if actual_h == 0:
            continue
        
        with torch.no_grad():
            init_win = train_data[last_idx:last_idx + window_size]
            curr_win = torch.FloatTensor(init_win).to(device)
            
            pred_wins = []
            
            for h in range(actual_h):
                z_curr = autoencoder.encoder.encode_deterministic(
                    curr_win.unsqueeze(0)
                ).squeeze(0)
                
                action = policy(z_curr.unsqueeze(0)).squeeze(0)
                z_next = z_curr + action
                
                win_pred = autoencoder.decoder(z_next.unsqueeze(0)).squeeze(0)
                pred_wins.append(win_pred.cpu().numpy())
                
                if stride < window_size:
                    curr_win = torch.cat([
                        curr_win[stride:],
                        win_pred[-stride:].detach()
                    ])
                else:
                    curr_win = win_pred.detach()
        
        pred_wins = np.array(pred_wins)
        true_futures = np.array(true_futures)
        
        mse = np.mean((pred_wins - true_futures)**2)
        mae = np.mean(np.abs(pred_wins - true_futures))
        
        results.append({
            'horizon': actual_h,
            'mse': mse,
            'mae': mae,
            'predictions': pred_wins,
            'ground_truth': true_futures
        })
        
        print(f"  H={actual_h:2d}: MSE={mse:.6f}, MAE={mae:.6f}")
    
    return results


def complete_conv1d_bc_pipeline():
    """Complete pipeline"""
    print("="*80)
    print("🚀 CONV1D_BC FORECASTING")
    print("="*80)
    
    # Setup
    try:
        from google.colab import drive
        if not os.path.exists('/content/drive'):
            drive.mount('/content/drive')
        base_dir = '/content/drive/MyDrive/latent_rl_experiments'
    except:
        base_dir = './latent_rl_experiments'
    
    os.makedirs(os.path.join(base_dir, 'plots'), exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load
    autoencoder, config = load_conv1d_bc_model(base_dir, device)
    train_data, _, _, _ = load_data()
    
    # Train policy
    policy, teacher_start, teacher_traj = train_policy_on_teacher(
        autoencoder, train_data, config, device
    )
    
    # Discovery
    window_size = config['window_size']
    num_steps = 20
    stride = 5
    
    discovered = active_discovery(
        policy, autoencoder, train_data, teacher_start,
        window_size, num_steps, stride,
        top_k=5, min_gap=300, sample_stride=50,
        device=device
    )
    
    print(f"\n✓ Discovered {len(discovered)}:")
    for i, p in enumerate(discovered):
        print(f"  {i+1}. Conf={p['confidence_score']:.4f}")
    
    # Fine-tune
    print("\n⚙️  Fine-tuning...")
    top = discovered[:3]
    
    trajs = [torch.FloatTensor(p['latent_trajectory']).to(device) for p in top]
    confs = [p['confidence_score'] for p in top]
    
    all_trajs = [teacher_traj] + trajs
    all_confs = [1.0] + confs
    
    weights = np.array(all_confs) / np.sum(all_confs)
    
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)
    
    for epoch in range(300):
        total_loss = 0
        total_samples = 0
        
        for traj, weight in zip(all_trajs, weights):
            for t in range(len(traj) - 1):
                state = traj[t].unsqueeze(0)
                target = traj[t+1] - traj[t]
                
                pred = policy(state).squeeze(0)
                loss = F.mse_loss(pred, target)
                
                weighted = loss * weight * len(all_trajs)
                
                optimizer.zero_grad()
                weighted.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_samples += 1
        
        if (epoch + 1) % 100 == 0:
            avg_loss = total_loss / total_samples
            print(f"  Epoch {epoch+1}, Loss: {avg_loss:.6f}")
    
    print(f"✓ Complete")
    
    # Forecast
    horizons = [5, 10, 15, 20, 30]
    
    results = multi_horizon_forecast(
        policy, autoencoder, train_data, teacher_start,
        window_size, num_steps, stride, horizons, device
    )
    
    print(f"\n{'='*80}")
    print("🎉 COMPLETE!")
    print(f"{'='*80}")
    
    return results


if __name__ == "__main__":
    results = complete_conv1d_bc_pipeline()
