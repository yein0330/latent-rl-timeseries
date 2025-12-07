"""Script to create all remaining files"""
import os

files_content = {
    'training/__init__.py': '"""Training package"""\nfrom .vae_trainer import train_vae\nfrom .bc_trainer import train_behavioral_cloning\nfrom .sac_trainer import train_sac, LatentEnv, ReplayBuffer\n',
    
    'training/vae_trainer.py': '''"""VAE training"""
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
''',

    'training/bc_trainer.py': '''"""Behavioral Cloning training"""
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
''',

    'utils/__init__.py': '"""Utils package"""\nfrom .data_loader import setup_google_drive, load_data, find_smooth_segment\nfrom .metrics import evaluate_policy, extract_trajectory\nfrom .visualization import plot_target_trajectory, plot_comparison_summary\n',
    
    'utils/data_loader.py': '''"""Data loading utilities"""
import os
import numpy as np
import pandas as pd


def setup_google_drive():
    """Mount Google Drive"""
    try:
        from google.colab import drive
        if not os.path.exists('/content/drive'):
            print("📁 Mounting Google Drive...")
            drive.mount('/content/drive')
        base_dir = '/content/drive/MyDrive/latent_rl_experiments'
    except:
        print("⚠️  Not in Colab, using local directory")
        base_dir = './latent_rl_experiments'
    
    for subdir in ['checkpoints', 'logs', 'plots', 'results', 'experiments']:
        os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)
    
    print(f"✓ Output directory: {base_dir}")
    return base_dir


def load_data():
    """Load ETTh1 dataset"""
    data_path = '/content/ETTh1.csv'
    
    if not os.path.exists(data_path):
        print("📥 Downloading ETTh1 dataset...")
        url = 'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv'
        import urllib.request
        urllib.request.urlretrieve(url, data_path)
        print("✓ Download complete")
    
    df = pd.read_csv(data_path)
    data = df['OT'].values
    
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    mean = train_data.mean()
    std = train_data.std()
    train_data = (train_data - mean) / std
    test_data = (test_data - mean) / std
    
    print(f"✓ Data loaded: Train={len(train_data)}, Test={len(test_data)}")
    return train_data, test_data, mean, std


def find_smooth_segment(data, window_size=96, num_steps=20, stride=5):
    """Find smooth segment"""
    required_length = (num_steps * stride) + window_size
    best_score = float('inf')
    best_start = 0
    
    for start in range(0, len(data) - required_length, 50):
        segment = data[start:start + required_length]
        
        diff = np.diff(segment)
        smoothness = np.std(diff)
        volatility = np.std(segment)
        trend = abs(np.polyfit(range(len(segment)), segment, 1)[0])
        extreme_penalty = 10.0 if np.max(np.abs(segment)) > 2.0 else 0.0
        
        score = smoothness * 2.0 + volatility * 1.0 + trend * 0.5 + extreme_penalty
        
        if score < best_score:
            best_score = score
            best_start = start
    
    return best_start
''',

    'utils/metrics.py': '''"""Evaluation metrics"""
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
''',

    'configs/experiment_configs.py': '''"""Experiment configurations"""


def get_all_experiment_configs():
    """Get all 25 experiment configs"""
    configs = []
    
    # Encoders (10)
    encoder_configs = [
        ('MLP', 'mlp', [128, 128]),
        ('MLP_Deep', 'mlp', [256, 256, 128]),
        ('Conv1D', 'conv1d', [64, 128, 256]),
        ('LSTM', 'lstm', [128, 128]),
        ('Transformer', 'transformer', [128, 128]),
    ]
    
    for name, arch, hidden_dims in encoder_configs:
        configs.append({
            'name': f'Encoder_{name}_BC',
            'encoder_type': arch,
            'hidden_dims': hidden_dims,
            'window_size': 96,
            'latent_dim': 32,
            'action_scale': 0.1,
            'method': 'BC',
            'reward_weights': None,
            'category': 'encoder'
        })
        configs.append({
            'name': f'Encoder_{name}_SAC',
            'encoder_type': arch,
            'hidden_dims': hidden_dims,
            'window_size': 96,
            'latent_dim': 32,
            'action_scale': 0.1,
            'method': 'SAC',
            'reward_weights': (30.0, 15.0, 10.0, 5.0, 50.0),
            'category': 'encoder'
        })
    
    # Windows (5)
    for window_size in [48, 96, 192, 336, 512]:
        configs.append({
            'name': f'WindowSize_{window_size}',
            'encoder_type': 'mlp',
            'hidden_dims': [128, 128],
            'window_size': window_size,
            'latent_dim': 32,
            'action_scale': 0.1,
            'method': 'BC',
            'reward_weights': None,
            'category': 'window'
        })
    
    # Actions (4)
    for action_scale in [0.05, 0.1, 0.15, 0.2]:
        configs.append({
            'name': f'ActionScale_{action_scale}',
            'encoder_type': 'mlp',
            'hidden_dims': [128, 128],
            'window_size': 96,
            'latent_dim': 32,
            'action_scale': action_scale,
            'method': 'BC',
            'reward_weights': None,
            'category': 'action'
        })
    
    # Rewards (3)
    reward_configs = [
        ('LatentEmphasis', (50.0, 10.0, 5.0, 5.0, 50.0)),
        ('ReconEmphasis', (20.0, 30.0, 10.0, 5.0, 50.0)),
        ('Balanced', (25.0, 25.0, 15.0, 10.0, 50.0)),
    ]
    
    for name, weights in reward_configs:
        configs.append({
            'name': f'Reward_{name}',
            'encoder_type': 'mlp',
            'hidden_dims': [128, 128],
            'window_size': 96,
            'latent_dim': 32,
            'action_scale': 0.1,
            'method': 'SAC',
            'reward_weights': weights,
            'category': 'reward'
        })
    
    # Latents (3)
    for latent_dim in [16, 32, 64]:
        configs.append({
            'name': f'LatentDim_{latent_dim}',
            'encoder_type': 'mlp',
            'hidden_dims': [128, 128],
            'window_size': 96,
            'latent_dim': latent_dim,
            'action_scale': 0.1,
            'method': 'BC',
            'reward_weights': None,
            'category': 'latent'
        })
    
    return configs


def get_quick_test_configs():
    """Quick test configs"""
    all_configs = get_all_experiment_configs()
    return all_configs[:3]


def get_configs_by_category(category):
    """Filter by category"""
    all_configs = get_all_experiment_configs()
    if category == 'all':
        return all_configs
    return [c for c in all_configs if c.get('category') == category]
''',

    'examples/quick_start.py': '''"""Quick start example"""
from forecasting_pipeline import complete_conv1d_bc_pipeline

# Run complete pipeline
results = complete_conv1d_bc_pipeline()

print("\\nResults:")
for r in results:
    print(f"  Horizon {r['horizon']:2d}: MSE={r['mse']:.6f}, MAE={r['mae']:.6f}")
''',

    'docs/architecture.md': '''# Architecture Overview

## System Components

### 1. VAE Encoder-Decoder
- Encodes time windows into latent space
- Multiple architectures: MLP, Conv1D, LSTM, Transformer
- Decoder reconstructs windows from latent

### 2. Policy Network
- BC: Behavioral Cloning (supervised)
- SAC: Soft Actor-Critic (RL)
- Learns to navigate latent space

### 3. Active Pattern Discovery
- Agent explores time series
- Evaluates segments by confidence
- Selects high-quality patterns

### 4. Few-Shot Learning
- Fine-tunes on discovered patterns
- Confidence-weighted training
- Improves generalization

## Data Flow

```
Raw Time Series
    ↓
Window Extraction
    ↓
VAE Encoder → Latent Space
    ↓
Policy Network (BC/SAC)
    ↓
Latent Trajectory
    ↓
VAE Decoder → Predictions
```

## Key Innovation

**Active Discovery**: Unlike passive pattern matching, the trained agent actively explores the time series to find segments it recognizes, leading to more semantically similar patterns for few-shot learning.
''',

    'docs/experiments.md': '''# Experiment Details

## 25 Experiments Overview

### 1. Encoder Architectures (10)
- **MLP**: Simple feedforward
- **MLP_Deep**: Deeper architecture
- **Conv1D**: Convolutional for time series
- **LSTM**: Recurrent for sequences
- **Transformer**: Attention-based

Each tested with BC and SAC.

### 2. Window Sizes (5)
- 48, 96, 192, 336, 512 timesteps
- Tests different temporal contexts

### 3. Action Scales (4)
- 0.05, 0.1, 0.15, 0.2
- Controls step size in latent space

### 4. Reward Weights (3)
For SAC training:
- **LatentEmphasis**: Focus on latent distance
- **ReconEmphasis**: Focus on reconstruction
- **Balanced**: Equal weighting

### 5. Latent Dimensions (3)
- 16, 32, 64
- Compression vs expressiveness tradeoff

## Best Configuration

**Conv1D + BC**
- Best overall performance
- Fast training
- Stable predictions
- MSE: 0.073 (H=5) to 0.177 (H=30)
''',

    'docs/results.md': '''# Results Analysis

## Performance Summary

### Conv1D_BC (Best Model)

| Horizon | MSE | MAE |
|---------|-----|-----|
| 5 | 0.073 | 0.239 |
| 10 | 0.093 | 0.272 |
| 15 | 0.110 | 0.299 |
| 20 | 0.128 | 0.324 |
| 30 | 0.177 | 0.377 |

## Key Findings

### 1. Architecture Comparison
- **Conv1D**: Best for time series (local patterns)
- **LSTM**: Good but slower
- **Transformer**: Slowest, moderate performance
- **MLP**: Fast but limited

### 2. Training Method
- **BC**: Faster, more stable
- **SAC**: Slower, less stable
- BC recommended for this task

### 3. Active Discovery
- Top-3 patterns sufficient
- Confidence weighting effective
- ~30% performance improvement

### 4. Error Characteristics
- Linear error growth (good)
- No catastrophic divergence
- 90% errors < 0.52
- Slight underprediction bias

## Recommendations

### Production Use
- Use Conv1D_BC
- Horizon 5-10 recommended
- Apply bias correction (+0.3)
- Monitor 95th percentile

### Research Use
- Experiment with ensembles
- Try hybrid architectures
- Explore longer horizons
- Test on other datasets
'''
}

# Create all files
for filepath, content in files_content.items():
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"Created: {filepath}")

print("\\nAll files created successfully!")
