"""Experiment configurations"""


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
