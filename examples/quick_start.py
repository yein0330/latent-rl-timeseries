"""Quick start example"""
from forecasting_pipeline import complete_conv1d_bc_pipeline

# Run complete pipeline
results = complete_conv1d_bc_pipeline()

print("\nResults:")
for r in results:
    print(f"  Horizon {r['horizon']:2d}: MSE={r['mse']:.6f}, MAE={r['mae']:.6f}")
