# Results Analysis

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
