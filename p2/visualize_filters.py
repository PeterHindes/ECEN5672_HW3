import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

print("Loading saved denoising model...")
model = keras.models.load_model("mnist_autoencoder_denoiser_model.keras")
print("✓ Model loaded successfully!\n")

# Get all convolutional layers
conv_layers = []
conv_layer_names = []

for layer in model.layers:
    if isinstance(layer, (keras.layers.Conv2D, keras.layers.Conv2DTranspose)):
        conv_layers.append(layer)
        conv_layer_names.append(layer.name)

print(f"Found {len(conv_layers)} convolutional layers:")
for i, name in enumerate(conv_layer_names):
    print(f"  {i + 1}. {name}")
print()


# Function to normalize filters for visualization
def normalize_filter(f):
    """Normalize filter to 0-1 range for visualization"""
    f_min, f_max = f.min(), f.max()
    if f_max - f_min > 0:
        return (f - f_min) / (f_max - f_min)
    return f


# Visualize filters from each convolutional layer
for layer_idx, (layer, layer_name) in enumerate(zip(conv_layers, conv_layer_names)):
    weights = layer.get_weights()[0]  # Get kernel weights

    print(f"Visualizing layer {layer_idx + 1}/{len(conv_layers)}: {layer_name}")
    print(f"  Shape: {weights.shape}")

    # For Conv2D: (height, width, input_channels, output_channels)
    # For Conv2DTranspose: (height, width, output_channels, input_channels)

    if isinstance(layer, keras.layers.Conv2D):
        kernel_h, kernel_w, in_channels, out_channels = weights.shape
    else:  # Conv2DTranspose
        kernel_h, kernel_w, out_channels, in_channels = weights.shape
        # Transpose to match Conv2D format for easier handling
        weights = np.transpose(weights, (0, 1, 3, 2))

    # Limit number of filters to visualize (max 64 for readability)
    num_filters_to_show = min(out_channels, 64)

    # Calculate grid size
    grid_size = int(np.ceil(np.sqrt(num_filters_to_show)))

    # Create figure for this layer
    fig = plt.figure(figsize=(15, 15))
    fig.suptitle(
        f"{layer_name}\nShape: {kernel_h}×{kernel_w}, {in_channels} input channels → {out_channels} filters",
        fontsize=16,
        fontweight="bold",
    )

    for filter_idx in range(num_filters_to_show):
        ax = plt.subplot(grid_size, grid_size, filter_idx + 1)

        # Get the filter for this output channel
        # Average across input channels if there are multiple
        if in_channels == 1:
            filter_img = weights[:, :, 0, filter_idx]
        else:
            # For multi-channel inputs, show the channel with max variance
            filter_data = weights[:, :, :, filter_idx]
            channel_variances = [
                np.var(filter_data[:, :, i]) for i in range(in_channels)
            ]
            max_var_channel = np.argmax(channel_variances)
            filter_img = filter_data[:, :, max_var_channel]

        # Normalize and display
        filter_img_normalized = normalize_filter(filter_img)

        ax.imshow(filter_img_normalized, cmap="viridis", interpolation="nearest")
        ax.set_title(f"Filter {filter_idx + 1}", fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    print(f"  Displaying {num_filters_to_show} filters...\n")
    plt.show()

print("\n" + "=" * 50)
print("Filter Visualization Complete!")
print("=" * 50)
print("\nInterpretation Guide:")
print("  - Encoder layers (Conv2D): Learn to detect features")
print("    * Early layers: Edge detectors, simple patterns")
print("    * Deeper layers: More complex, abstract features")
print("  - Decoder layers (Conv2DTranspose): Learn to reconstruct")
print("    * Learn to synthesize image features from bottleneck")
print("\nColor coding:")
print("  - Bright (yellow): Positive weights (activate on bright pixels)")
print("  - Dark (purple): Negative weights (activate on dark pixels)")
print("  - Mid-range: Near-zero weights (less important)")
