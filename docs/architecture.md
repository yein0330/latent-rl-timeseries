# Architecture Overview

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
