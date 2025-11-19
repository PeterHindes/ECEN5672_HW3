# Autoencoder Comparison Scripts

This directory contains scripts to compare two autoencoder models:
- **Model A**: Clean autoencoder (trained on clean→clean images, NO noise)
- **Model C**: Denoising autoencoder (trained on noisy→clean images, WITH noise σ=0.3)

## Scripts Overview

### Training Scripts
- **`a.py`** - Trains the clean autoencoder (Model A)
  - No noise during training
  - Improved architecture: BatchNorm, LeakyReLU, 64 filters, He initialization
  - Output: `mnist_autoencoder_clean_model.keras`

- **`c.py`** - Trains the denoising autoencoder (Model C)
  - Dynamic noise generation (σ=0.3) - unique noise per epoch
  - Same improved architecture as Model A
  - Output: `mnist_autoencoder_denoiser_model.keras`

### Comparison Scripts
- **`compare_models.py`** - Compares models across different noise levels
  - Tests: σ = 0.0, 0.1, 0.2, 0.3, 0.4, 0.5
  - Shows which model performs better at denoising
  - Output: `noise_comparison.png`

- **`compare_quantization.py`** - Compares models with quantized bottlenecks
  - Tests: 8-bit, 4-bit, 3-bit, 2-bit, 1-bit quantization
  - Shows robustness to bottleneck compression
  - Output: `quantization_comparison.png`

### Visualization Scripts
- **`b.py`** - Visualizes effective filters and bottleneck activations
  - Shows all filters learned by the model
  - Analyzes bottleneck neuron specialization per digit
  - Output: `effective_filters.png`, `bottleneck_activations.png`

- **`showqc.py`** - Interactive quantization visualizer
  - Shows real-time quantization effects on bottleneck

## Usage

### 1. Train Both Models
```bash
# Train clean autoencoder (Model A)
python p2/a.py

# Train denoising autoencoder (Model C)
python p2/c.py
```

### 2. Run Comparisons
```bash
# Compare on noise levels
python p2/compare_models.py

# Compare on quantization levels
python p2/compare_quantization.py

# Visualize filters and activations
python p2/b.py
```

## Expected Results

### Noise Comparison
- **Model A** should perform better at σ=0.0 (clean images)
- **Model C** should perform better at σ≥0.3 (noisy images)
- This demonstrates that training with noise improves denoising performance

### Quantization Comparison
- Shows how robust each model is to bottleneck compression
- Lower bit depths = more information loss
- Compares 8-bit down to 1-bit representations

## File Outputs

- `mnist_autoencoder_clean_model.keras` - Clean autoencoder weights
- `mnist_autoencoder_denoiser_model.keras` - Denoising autoencoder weights
- `noise_comparison.png` - Noise level comparison charts
- `quantization_comparison.png` - Quantization level comparison charts
- `effective_filters.png` - Filter visualizations
- `bottleneck_activations.png` - Bottleneck analysis

## Key Findings

The comparison should demonstrate:
1. ✅ Training with diverse noise examples improves generalization to noisy inputs
2. ✅ Clean-trained models excel at clean reconstruction but struggle with noise
3. ✅ Denoising models sacrifice some clean performance for noise robustness
4. ✅ Both architectures can be quantized, but with different degradation patterns