# Experiment Details

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
