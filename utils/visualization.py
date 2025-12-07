"""Visualization utilities"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def plot_target_trajectory(train_data, start_idx, window_size, num_steps,
                           stride, autoencoder, device, save_path):
    """Plot target trajectory"""
    from utils.metrics import extract_trajectory
    
    latent_traj, window_traj = extract_trajectory(
        autoencoder, train_data, start_idx,
        window_size, num_steps, stride, device
    )
    
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    fig.suptitle('Target Trajectory Visualization', fontsize=20, fontweight='bold')
    
    # Raw windows
    sample_steps = [0, num_steps//2, num_steps]
    for idx, step in enumerate(sample_steps):
        ax = fig.add_subplot(gs[0, idx])
        window = window_traj[step].cpu().numpy()
        time_axis = np.arange(len(window))
        ax.plot(time_axis, window, linewidth=2, color='steelblue')
        ax.fill_between(time_axis, window, alpha=0.3, color='steelblue')
        ax.set_title(f'Raw Window at Step {step}', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Latent dims
    latent_np = latent_traj.cpu().numpy()
    steps = np.arange(num_steps + 1)
    
    for dim in range(3):
        ax = fig.add_subplot(gs[1, dim])
        ax.plot(steps, latent_np[:, dim], 'o-', linewidth=2.5,
               markersize=8, color=f'C{dim}')
        ax.set_title(f'Latent Dimension {dim}', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_comparison_summary(all_results, save_path):
    """Plot experiment comparison"""
    if len(all_results) == 0:
        print("⚠️  No results to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Experiment Comparison Summary', fontsize=18, fontweight='bold')
    
    exp_names = [r['config']['name'] for r in all_results]
    
    metrics_data = {
        'Avg Distance': [r['test_results']['avg_distance'] for r in all_results],
        'Final Distance': [r['test_results']['final_distance'] for r in all_results],
        'Recon MSE': [r['test_results']['recon_mse'] for r in all_results],
        'Training Time': [r['training_time'] for r in all_results]
    }
    
    for (metric_name, values), ax in zip(metrics_data.items(), axes.flat):
        x_pos = np.arange(len(exp_names))
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(exp_names)))
        
        bars = ax.bar(x_pos, values, alpha=0.7, color=colors)
        
        best_idx = np.argmin(values)
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(3)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels([n[:12] for n in exp_names], rotation=45, ha='right', fontsize=8)
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✓ Comparison plot saved: {save_path}")
