"""
Pattern-Conditioned Latent RL for Time-Series Forecasting
Complete Experimental Pipeline - 25 Experiments

Usage:
    python main_experiments.py
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
import time
import random
import pickle
import json
from collections import deque
from datetime import datetime

from models.encoders import MLPEncoder, Conv1DEncoder, LSTMEncoder, TransformerEncoder
from models.decoder import Decoder
from models.autoencoder import LatentAutoEncoder
from models.policy import Actor, ActorSAC, Critic
from training.vae_trainer import train_vae
from training.bc_trainer import train_behavioral_cloning
from training.sac_trainer import train_sac, LatentEnv, ReplayBuffer
from utils.data_loader import setup_google_drive, load_data, find_smooth_segment
from utils.metrics import evaluate_policy, extract_trajectory
from utils.visualization import plot_target_trajectory, plot_comparison_summary
from configs.experiment_configs import (
    get_all_experiment_configs,
    get_quick_test_configs,
    get_configs_by_category
)


def set_seed(seed):
    """Set random seed"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_encoder(encoder_type, window_size, latent_dim, hidden_dims, dropout=0.3):
    """Encoder factory"""
    if encoder_type == 'mlp':
        return MLPEncoder(window_size, latent_dim, hidden_dims, dropout)
    elif encoder_type == 'conv1d':
        return Conv1DEncoder(window_size, latent_dim, hidden_dims, dropout)
    elif encoder_type == 'lstm':
        return LSTMEncoder(window_size, latent_dim, hidden_dims, dropout)
    elif encoder_type == 'transformer':
        return TransformerEncoder(window_size, latent_dim, hidden_dims, dropout)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


def run_single_experiment(config, train_data, test_data, base_dir, device, seed=42):
    """Run single experiment"""
    exp_name = config['name']
    print(f"\n{'='*80}")
    print(f"Experiment: {exp_name}")
    print(f"{'='*80}")

    set_seed(seed)
    start_time = time.time()

    exp_dir = os.path.join(base_dir, 'experiments', exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    try:
        # VAE checkpoint
        hidden_str = '_'.join(map(str, config['hidden_dims']))
        checkpoint_name = (f"vae_{config['encoder_type']}_"
                          f"ws{config['window_size']}_"
                          f"ld{config['latent_dim']}_"
                          f"hd{hidden_str}.pt")
        checkpoint_path = os.path.join(base_dir, 'checkpoints', checkpoint_name)

        if os.path.exists(checkpoint_path):
            print(f"✓ Loading VAE: {checkpoint_name}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

            encoder = create_encoder(
                config['encoder_type'],
                config['window_size'],
                config['latent_dim'],
                config['hidden_dims']
            ).to(device)

            decoder = Decoder(config['latent_dim'], config['window_size']).to(device)
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            decoder.load_state_dict(checkpoint['decoder_state_dict'])
            autoencoder = LatentAutoEncoder(encoder, decoder)
            vae_loss = checkpoint['vae_loss']

        else:
            print(f"Training VAE: {checkpoint_name}")
            encoder = create_encoder(
                config['encoder_type'],
                config['window_size'],
                config['latent_dim'],
                config['hidden_dims']
            ).to(device)
            decoder = Decoder(config['latent_dim'], config['window_size']).to(device)
            autoencoder = LatentAutoEncoder(encoder, decoder)

            vae_loss = train_vae(
                autoencoder, train_data,
                window_size=config['window_size'],
                device=device, verbose=False
            )

            torch.save({
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'vae_loss': vae_loss
            }, checkpoint_path)

        print(f"  VAE Loss: {vae_loss:.4f}")

        # Extract trajectories
        smooth_start = find_smooth_segment(train_data, window_size=config['window_size'])
        train_latent_traj, train_window_traj = extract_trajectory(
            autoencoder, train_data, smooth_start,
            window_size=config['window_size'], device=device
        )

        # Train policy
        if config['method'] == 'BC':
            policy = Actor(
                config['latent_dim'],
                config['latent_dim'],
                action_scale=config['action_scale']
            ).to(device)

            train_losses, final_loss = train_behavioral_cloning(
                policy, train_latent_traj, device=device, verbose=False
            )
            print(f"  BC Loss: {final_loss:.6f}")

        else:  # SAC
            if config['reward_weights'] is None:
                reward_weights = (30.0, 15.0, 10.0, 5.0, 50.0)
            else:
                reward_weights = config['reward_weights']

            env = LatentEnv(
                autoencoder,
                train_latent_traj,
                train_window_traj,
                reward_weights,
                device
            )

            policy, episode_rewards, episode_distances = train_sac(
                env,
                config['latent_dim'],
                config['latent_dim'],
                config['action_scale'],
                episodes=200,
                device=device,
                verbose=False
            )

            final_loss = -np.mean(episode_rewards[-10:])
            print(f"  SAC Reward: {-final_loss:.2f}")

        # Evaluate on test
        test_start = find_smooth_segment(test_data, window_size=config['window_size'])
        test_latent_traj, test_window_traj = extract_trajectory(
            autoencoder, test_data, test_start,
            window_size=config['window_size'], device=device
        )

        test_results = evaluate_policy(
            policy, autoencoder, test_latent_traj, test_window_traj, device
        )

        print(f"  Test Avg Dist: {test_results['avg_distance']:.4f}")
        print(f"  Test Recon MSE: {test_results['recon_mse']:.4f}")

        training_time = time.time() - start_time
        print(f"  Time: {training_time:.1f}s")

        result = {
            'config': config,
            'vae_loss': vae_loss,
            'final_loss': final_loss,
            'test_results': test_results,
            'training_time': training_time,
            'success': True
        }

        with open(os.path.join(exp_dir, 'results.pkl'), 'wb') as f:
            pickle.dump(result, f)

        return result

    except Exception as e:
        print(f"  ❌ Failed: {e}")
        import traceback
        traceback.print_exc()

        return {
            'config': config,
            'success': False,
            'error': str(e)
        }


def run_experiments(configs, train_data, test_data, base_dir, device):
    """Run multiple experiments"""
    print(f"\n{'='*80}")
    print(f"🚀 RUNNING {len(configs)} EXPERIMENTS")
    print(f"{'='*80}")

    all_results = []

    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] {config['name']}")
        result = run_single_experiment(config, train_data, test_data, base_dir, device)

        if result['success']:
            all_results.append(result)

    return all_results


def main():
    """Main execution"""
    print("="*80)
    print("🔬 LATENT RL FORECASTING - COMPLETE EXPERIMENTAL PIPELINE")
    print("="*80)

    base_dir = setup_google_drive()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    print("\n📊 Loading data...")
    train_data, test_data, mean, std = load_data()

    # Menu
    print("\n" + "="*80)
    print("EXPERIMENT MENU")
    print("="*80)
    print("1. Quick Test (3 experiments)")
    print("2. Encoder Architectures (10 experiments)")
    print("3. Window Sizes (5 experiments)")
    print("4. Action Scales (4 experiments)")
    print("5. Reward Weights (3 experiments)")
    print("6. Latent Dimensions (3 experiments)")
    print("7. ALL Experiments (25 experiments)")
    print("8. Single Custom Experiment + Visualization")
    print("="*80)

    choice = input("\nSelect option (1-8) [default=1]: ").strip() or '1'

    if choice == '1':
        configs = get_quick_test_configs()
        print(f"\n✓ Quick test: {len(configs)} experiments")
    elif choice == '2':
        configs = get_configs_by_category('encoder')
        print(f"\n✓ Encoder experiments: {len(configs)} experiments")
    elif choice == '3':
        configs = get_configs_by_category('window')
        print(f"\n✓ Window size experiments: {len(configs)} experiments")
    elif choice == '4':
        configs = get_configs_by_category('action')
        print(f"\n✓ Action scale experiments: {len(configs)} experiments")
    elif choice == '5':
        configs = get_configs_by_category('reward')
        print(f"\n✓ Reward weight experiments: {len(configs)} experiments")
    elif choice == '6':
        configs = get_configs_by_category('latent')
        print(f"\n✓ Latent dimension experiments: {len(configs)} experiments")
    elif choice == '7':
        configs = get_all_experiment_configs()
        print(f"\n✓ ALL experiments: {len(configs)} experiments")
    elif choice == '8':
        config = {
            'name': 'Custom_MLP_BC',
            'encoder_type': 'mlp',
            'hidden_dims': [128, 128],
            'window_size': 96,
            'latent_dim': 32,
            'action_scale': 0.1,
            'method': 'BC',
            'reward_weights': None,
            'category': 'custom'
        }
        configs = [config]
        print(f"\n✓ Single custom experiment")

        print("\n📊 Creating target trajectory visualization...")
        set_seed(42)

        encoder = create_encoder('mlp', 96, 32, [128, 128]).to(device)
        decoder = Decoder(32, 96).to(device)
        autoencoder = LatentAutoEncoder(encoder, decoder)

        vae_loss = train_vae(autoencoder, train_data, device=device, verbose=False)
        smooth_start = find_smooth_segment(train_data)

        viz_path = os.path.join(base_dir, 'plots', 'target_trajectory.png')
        plot_target_trajectory(
            train_data, smooth_start, 96, 20, 5,
            autoencoder, device, viz_path
        )
        print(f"✓ Visualization saved: {viz_path}")
    else:
        print("Invalid option, using quick test")
        configs = get_quick_test_configs()

    # Confirm
    print(f"\n{'='*80}")
    for cfg in configs:
        print(f"  - {cfg['name']}")
    print(f"{'='*80}")

    confirm = input(f"\nProceed with {len(configs)} experiments? (y/n) [default=y]: ").strip().lower() or 'y'

    if confirm != 'y':
        print("Aborted.")
        return

    # Run
    all_results = run_experiments(configs, train_data, test_data, base_dir, device)

    # Comparison plot
    if len(all_results) > 1:
        print("\n📊 Creating comparison plot...")
        comparison_path = os.path.join(base_dir, 'plots', 'comparison_summary.png')
        plot_comparison_summary(all_results, comparison_path)

    # Summary
    print(f"\n{'='*80}")
    print("🎉 ALL EXPERIMENTS COMPLETE!")
    print(f"{'='*80}")
    print(f"\n📊 Summary:")
    print(f"  Total experiments: {len(configs)}")
    print(f"  Successful: {len(all_results)}")
    print(f"  Failed: {len(configs) - len(all_results)}")

    if len(all_results) > 0:
        print(f"\n🏆 Best Results:")
        best_by_dist = min(all_results, key=lambda x: x['test_results']['avg_distance'])
        print(f"  Best Avg Distance: {best_by_dist['config']['name']} "
              f"({best_by_dist['test_results']['avg_distance']:.4f})")

        best_by_recon = min(all_results, key=lambda x: x['test_results']['recon_mse'])
        print(f"  Best Recon MSE: {best_by_recon['config']['name']} "
              f"({best_by_recon['test_results']['recon_mse']:.4f})")

    print(f"\n📁 Results saved to: {base_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
