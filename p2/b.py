import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

print("Loading model...")
model = keras.models.load_model("mnist_autoencoder_denoiser_model.keras")
print("✓ Model loaded\n")

# Load test data
print("Loading test data...")
ds_test = tfds.load("mnist", split="test[:100]", as_supervised=False)


def normalize_img(data):
    return tf.cast(data["image"], tf.float32) / 255.0


ds_test = ds_test.map(normalize_img).batch(1)

# Get a test image
for test_image in ds_test.take(1):
    break
print(f"Test image shape: {test_image.shape}\n")

# BUILD THE MODEL by calling it once
print("Building model...")
_ = model(test_image)  # This builds the model and defines inputs/outputs
print("✓ Model built\n")

# Now show model structure
print("Model structure:")
bottleneck_idx = None
conv_before_bottleneck_idx = None

for i, layer in enumerate(model.layers):
    try:
        output_shape = layer.output_shape
    except:
        output_shape = "unknown"

    print(
        f"  {i}: {layer.name:30s} {layer.__class__.__name__:20s} {str(output_shape):20s}"
    )

    # Auto-detect bottleneck
    if "bottleneck" in layer.name.lower():
        bottleneck_idx = i
        print(f"       ↑ BOTTLENECK DETECTED!")
    elif (
        isinstance(layer, keras.layers.Dense)
        and bottleneck_idx is None
        and i > len(model.layers) // 3
    ):
        # First Dense layer in latter part is likely bottleneck
        bottleneck_idx = i
        print(f"       ↑ POTENTIAL BOTTLENECK (Dense layer)")

    # Find last Conv layer before bottleneck or Flatten
    if isinstance(layer, keras.layers.Conv2D):
        if bottleneck_idx is None or i < bottleneck_idx:
            conv_before_bottleneck_idx = i

print(f"\nUsing:")
print(f"  Conv layer index: {conv_before_bottleneck_idx}")
print(f"  Bottleneck index: {bottleneck_idx}\n")

if bottleneck_idx is None:
    print("ERROR: Could not auto-detect bottleneck!")
    exit(1)

# Build intermediate models - NOW model.input is defined!
print("Extracting features...")

if conv_before_bottleneck_idx is not None:
    conv_model = keras.Model(inputs=model.input, outputs=model.layers[conv_before_bottleneck_idx].output)
    conv_features = conv_model.predict(test_image, verbose=0)
    print(f"Conv features shape: {conv_features.shape}")
else:
    conv_features = None
    print("No convolutional layer found before bottleneck")

bottleneck_model = keras.Model(inputs=model.input, outputs=model.layers[bottleneck_idx].output)
latent_code = bottleneck_model.predict(test_image, verbose=0)
print(f"Bottleneck shape: {latent_code.shape}\n")

# Get full reconstruction
reconstruction = model.predict(test_image, verbose=0)

# Create comprehensive visualization
fig = plt.figure(figsize=(18, 10))

# If we have conv features, plot them
if conv_features is not None and len(conv_features.shape) == 4:
    n_filters = min(32, conv_features.shape[-1])
    n_cols = 8
    n_rows = (n_filters + n_cols - 1) // n_cols

    for i in range(n_filters):
        ax = plt.subplot(n_rows + 3, n_cols, i + 1)
        ax.imshow(conv_features[0, :, :, i], cmap="viridis")
        ax.axis("off")
        ax.set_title(f"Filter {i}", fontsize=7)

    # Input/Output images
    ax = plt.subplot(n_rows + 3, n_cols, n_rows * n_cols + 1)
    ax.imshow(test_image[0].numpy().squeeze(), cmap="gray")
    ax.set_title("Input", fontsize=10, fontweight="bold")
    ax.axis("off")

    ax = plt.subplot(n_rows + 3, n_cols, n_rows * n_cols + 2)
    ax.imshow(reconstruction[0].squeeze(), cmap="gray")
    ax.set_title("Reconstructed", fontsize=10, fontweight="bold")
    ax.axis("off")

    # Bottleneck plot
    ax = plt.subplot(n_rows + 3, 1, n_rows + 2)
else:
    # No conv features - simpler layout
    ax = plt.subplot(3, 2, 1)
    ax.imshow(test_image[0].numpy().squeeze(), cmap="gray")
    ax.set_title("Input", fontsize=12, fontweight="bold")
    ax.axis("off")

    ax = plt.subplot(3, 2, 2)
    ax.imshow(reconstruction[0].squeeze(), cmap="gray")
    ax.set_title("Reconstructed", fontsize=12, fontweight="bold")
    ax.axis("off")

    ax = plt.subplot(3, 1, 2)

# Plot bottleneck
ax.bar(
    range(len(latent_code[0])),
    latent_code[0],
    color="steelblue",
    alpha=0.8,
    edgecolor="navy",
)
ax.set_title(
    f'Bottleneck: "{model.layers[bottleneck_idx].name}" ({len(latent_code[0])} dimensions)',
    fontsize=12,
    fontweight="bold",
)
ax.set_xlabel("Dimension", fontsize=10)
ax.set_ylabel("Activation", fontsize=10)
ax.grid(True, alpha=0.3, axis="y")
ax.axhline(y=0, color="black", linewidth=0.8)

# Add statistics
sparsity = np.sum(np.abs(latent_code[0]) < 0.01) / len(latent_code[0]) * 100
active = np.sum(np.abs(latent_code[0]) >= 0.01)
mean_val = np.mean(latent_code[0])
max_val = np.max(latent_code[0])

stats_text = f"Sparsity: {sparsity:.1f}%\nActive: {active}/{len(latent_code[0])}\nMean: {mean_val:.4f}\nMax: {max_val:.4f}"
ax.text(
    0.98,
    0.98,
    stats_text,
    transform=ax.transAxes,
    verticalalignment="top",
    horizontalalignment="right",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.9),
    fontsize=10,
    fontfamily="monospace",
)

plt.suptitle("Autoencoder Bottleneck Visualization", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig("bottleneck_visualization.png", dpi=150, bbox_inches="tight")
print("✓ Saved to: bottleneck_visualization.png")
plt.show()
