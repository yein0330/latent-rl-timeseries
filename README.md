# Pattern-Conditioned Latent RL for Time-Series Forecasting

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Active Pattern Discovery + Few-Shot Learning for Time-Series Forecasting**

This repository implements a novel approach to time-series forecasting by combining:
- Latent space reinforcement learning (BC/SAC)
- Active pattern discovery using trained agents
- Confidence-weighted few-shot learning
- Multi-horizon forecasting with Conv1D architectures

## 🎯 Key Features

- **25 Comprehensive Experiments**: Encoder architectures, hyperparameters, training methods
- **Active Discovery**: Agent explores time series to find familiar patterns
- **Few-Shot Learning**: Fine-tune on discovered high-confidence patterns
- **Multi-Architecture Support**: MLP, Conv1D, LSTM, Transformer encoders
- **Complete Visualization**: Performance analysis, error distributions, pattern discovery

## 📊 Results Highlights

| Model | Horizon 5 | Horizon 10 | Horizon 20 | Horizon 30 |
|-------|-----------|------------|------------|------------|
| Conv1D_BC (MSE) | 0.073 | 0.093 | 0.128 | 0.177 |
| Conv1D_BC (MAE) | 0.239 | 0.272 | 0.324 | 0.377 |

**Best Performance**: Conv1D + Behavioral Cloning + Active Discovery

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/latent-rl-forecasting.git
cd latent-rl-forecasting
pip install -r requirements.txt
```

### Basic Usage

```python
from forecasting_pipeline import complete_conv1d_bc_pipeline

# Run complete forecasting pipeline
results = complete_conv1d_bc_pipeline()
```

### Run Experiments

```python
from main_experiments import main

# Interactive menu for 25 experiments
main()
```

## 📖 Documentation

- [Architecture Overview](docs/architecture.md)
- [Experiment Details](docs/experiments.md)
- [Results Analysis](docs/results.md)

## 🔬 Methodology

### 1. Latent Space Learning
- Train VAE to encode time windows into latent space
- Use various architectures (MLP, Conv1D, LSTM, Transformer)

### 2. Policy Training
- Behavioral Cloning (BC): Direct imitation of teacher trajectory
- Soft Actor-Critic (SAC): RL with custom reward shaping

### 3. Active Pattern Discovery
- Trained agent explores time series
- Evaluates segments by tracking error and confidence
- Selects top-K familiar patterns

### 4. Few-Shot Fine-Tuning
- Confidence-weighted training on discovered patterns
- Improves generalization across diverse patterns

### 5. Multi-Horizon Forecasting
- Iterative prediction in latent space
- Decode to observation space
- Evaluate across multiple horizons

## 📁 Project Structure

```
latent-rl-forecasting/
├── models/           # Neural network architectures
├── training/         # Training procedures (VAE, BC, SAC)
├── utils/            # Data loading, metrics, visualization
├── configs/          # Experiment configurations
├── examples/         # Example scripts
└── docs/             # Documentation
```

## 🎨 Visualizations

The pipeline generates comprehensive visualizations:

1. **Discovered Patterns**: Agent-identified high-confidence segments
2. **Multi-Horizon Performance**: MSE/MAE across forecast horizons
3. **Error Analysis**: Distribution, percentiles, accumulation
4. **Comparison Summary**: 25 experiments side-by-side

## 📊 Experiments

### Encoder Architectures (10 experiments)
- MLP, MLP_Deep, Conv1D, LSTM, Transformer
- Each with BC and SAC variants

### Hyperparameter Sweeps
- **Window Sizes**: 48, 96, 192, 336, 512
- **Action Scales**: 0.05, 0.1, 0.15, 0.2
- **Latent Dimensions**: 16, 32, 64
- **Reward Weights**: 3 configurations for SAC

## 🔧 Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- Pandas
- Matplotlib
- Google Colab (optional, for GPU)

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{latent-rl-forecasting,
  author = {Your Name},
  title = {Pattern-Conditioned Latent RL for Time-Series Forecasting},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/latent-rl-forecasting}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- ETT dataset from [ETDataset](https://github.com/zhouhaoyi/ETDataset)
- Inspired by recent advances in latent RL and time-series forecasting

## 📧 Contact

For questions or feedback, please open an issue or contact [djdj447937@naver.com]

---

**⭐ Star this repo if you find it useful!**
