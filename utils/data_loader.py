"""Data loading utilities"""
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
