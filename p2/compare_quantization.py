import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

print("Loading models...")
print("  - Model A (clean-trained, NO noise)...")
model_a = keras.models.load_model("mnist_autoencoder_clean_model.keras")
print("  - Model C (denoiser, trained WITH noise)...")
model_c = keras.models.load_model("mnist_autoencoder_denoiser_model.keras")
print("✓ Models loaded successfully!\n")


# Find bottleneck layers
def find_bottleneck(model, model_name):
    for i, layer in enumerate(model.layers):
        if "bottleneck" in layer.name:
            print(f"{model_name} bottleneck: {layer.name} (layer {i})")
            return layer.name
    print(f"Warning: No bottleneck found in {model_name}")
    return None


bottleneck_a = find_bottleneck(model_a, "Model A")
bottleneck_c = find_bottleneck(model_c, "Model C")
print()

# Load test data first
print("Loading test data...")
ds_test = tfds.load(
    "mnist", split="test[:200]", shuffle_files=True, as_supervised=False
)


def normalize_img(data):
    image = tf.cast(data["image"], tf.float32) / 255.0
    image = tf.reshape(image, (28, 28, 1))
    return image


test_images = []
for data in ds_test:
    test_images.append(normalize_img(data))
test_images = tf.stack(test_images)
print(f"✓ Loaded {len(test_images)} test images\n")

# Build models by calling them with dummy data
print("Building models...")
dummy_input = tf.expand_dims(test_images[0], 0)
_ = model_a(dummy_input)
_ = model_c(dummy_input)
print("✓ Models built\n")

# Create bottleneck extractors
bottleneck_model_a = keras.Model(
    inputs=model_a.inputs, outputs=model_a.get_layer(bottleneck_a).output
)
bottleneck_model_c = keras.Model(
    inputs=model_c.inputs, outputs=model_c.get_layer(bottleneck_c).output
)

# Create decoder models (from bottleneck to output)
# Find bottleneck layer indices
bottleneck_idx_a = None
bottleneck_idx_c = None
for i, layer in enumerate(model_a.layers):
    if layer.name == bottleneck_a:
        bottleneck_idx_a = i
        break
for i, layer in enumerate(model_c.layers):
    if layer.name == bottleneck_c:
        bottleneck_idx_c = i
        break

print(f"Creating decoder models...")
print(f"  Model A: bottleneck at layer {bottleneck_idx_a}")
print(f"  Model C: bottleneck at layer {bottleneck_idx_c}")

# Get bottleneck output shape
bottleneck_shape_a = bottleneck_model_a.predict(dummy_input, verbose=0).shape[1]
bottleneck_shape_c = bottleneck_model_c.predict(dummy_input, verbose=0).shape[1]

# Build decoder for Model A
decoder_input_a = keras.Input(shape=(bottleneck_shape_a,))
x = decoder_input_a
for layer in model_a.layers[bottleneck_idx_a + 1 :]:
    x = layer(x)
decoder_a = keras.Model(decoder_input_a, x)

# Build decoder for Model C
decoder_input_c = keras.Input(shape=(bottleneck_shape_c,))
x = decoder_input_c
for layer in model_c.layers[bottleneck_idx_c + 1 :]:
    x = layer(x)
decoder_c = keras.Model(decoder_input_c, x)

print("✓ Decoder models created\n")


# Quantization functions
def quantize_to_bits(values, num_bits):
    """Quantize float32 values to specified bit depth (symmetric)"""
    if num_bits == 32:
        return values, 1.0, 0  # No quantization

    max_val = np.max(np.abs(values))
    if max_val == 0 or num_bits < 1:
        return np.zeros_like(values), 1.0, 0

    # Symmetric quantization
    qmax = 2 ** (num_bits - 1) - 1
    if qmax == 0:  # Handle 1-bit case
        qmax = 1

    scale = max_val / qmax
    if scale == 0:
        scale = 1.0

    quantized = np.round(values / scale)
    quantized = np.clip(quantized, -qmax - 1, qmax)

    return quantized.astype(np.int32), scale, 0


def dequantize(quantized, scale, zero_point):
    """Dequantize back to float32"""
    return quantized.astype(np.float32) * scale + zero_point


def reconstruct_with_quantized_bottleneck(
    bottleneck_model, decoder_model, image, num_bits
):
    """Reconstruct image using quantized bottleneck representation"""
    # Get bottleneck
    bottleneck_float = bottleneck_model.predict(image, verbose=0)[0]

    # Quantize
    bottleneck_quant, scale, zp = quantize_to_bits(bottleneck_float, num_bits)

    # Dequantize
    bottleneck_dequant = dequantize(bottleneck_quant, scale, zp)

    # Reconstruct using decoder
    reconstructed = decoder_model.predict(bottleneck_dequant.reshape(1, -1), verbose=0)

    return reconstructed, bottleneck_float, bottleneck_dequant


# Test quantization levels
quantization_levels = [8, 4, 3, 2, 1]  # bits (focusing on 8→1 bit cases)
print(f"Testing {len(quantization_levels)} quantization levels: {quantization_levels}")
print(f"Using {len(test_images)} test images\n")

results_a = {bits: [] for bits in quantization_levels}
results_c = {bits: [] for bits in quantization_levels}

print("Running quantization comparison...")
for i, image in enumerate(test_images):
    if (i + 1) % 50 == 0:
        print(f"  Processed {i + 1}/{len(test_images)} images...")

    image_batch = tf.expand_dims(image, 0)

    for num_bits in quantization_levels:
        # Model A
        recon_a, _, _ = reconstruct_with_quantized_bottleneck(
            bottleneck_model_a, decoder_a, image_batch, num_bits
        )
        mse_a = np.mean((image.numpy() - recon_a[0]) ** 2)
        results_a[num_bits].append(mse_a)

        # Model C
        recon_c, _, _ = reconstruct_with_quantized_bottleneck(
            bottleneck_model_c, decoder_c, image_batch, num_bits
        )
        mse_c = np.mean((image.numpy() - recon_c[0]) ** 2)
        results_c[num_bits].append(mse_c)

print("✓ Quantization comparison complete\n")

# Calculate average MSE for each quantization level
avg_mse_a = [np.mean(results_a[bits]) for bits in quantization_levels]
avg_mse_c = [np.mean(results_c[bits]) for bits in quantization_levels]

# Calculate improvement (handle division by zero and nan)
improvements = []
for i in range(len(quantization_levels)):
    if np.isnan(avg_mse_a[i]) or np.isnan(avg_mse_c[i]) or avg_mse_a[i] == 0:
        improvements.append(0.0)
    else:
        improvements.append((avg_mse_a[i] - avg_mse_c[i]) / avg_mse_a[i] * 100)

# Print summary
print("=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
print(f"{'Bits':<8} {'Model A MSE':<15} {'Model C MSE':<15} {'Improvement':<15}")
print("-" * 70)
for i, bits in enumerate(quantization_levels):
    mse_a_str = f"{avg_mse_a[i]:.6f}" if not np.isnan(avg_mse_a[i]) else "nan"
    mse_c_str = f"{avg_mse_c[i]:.6f}" if not np.isnan(avg_mse_c[i]) else "nan"
    imp_str = f"{improvements[i]:+.1f}%" if not np.isnan(improvements[i]) else "N/A"
    print(f"{bits:<8} {mse_a_str:<15} {mse_c_str:<15} {imp_str:>15}")
print("=" * 70)

# Get example images for visualization
example_idx = np.random.randint(0, len(test_images))
example_image = tf.expand_dims(test_images[example_idx], 0)

# Get reconstructions at different quantization levels
example_bits = [8, 3, 2]  # Show 3 examples
example_recons_a = []
example_recons_c = []
example_mses_a = []
example_mses_c = []

for bits in example_bits:
    recon_a, _, _ = reconstruct_with_quantized_bottleneck(
        bottleneck_model_a, decoder_a, example_image, bits
    )
    recon_c, _, _ = reconstruct_with_quantized_bottleneck(
        bottleneck_model_c, decoder_c, example_image, bits
    )

    example_recons_a.append(recon_a[0])
    example_recons_c.append(recon_c[0])
    example_mses_a.append(np.mean((example_image.numpy() - recon_a[0]) ** 2))
    example_mses_c.append(np.mean((example_image.numpy() - recon_c[0]) ** 2))

# Create comprehensive visualization
fig = plt.figure(figsize=(16, 14))
gs = fig.add_gridspec(4, 1, height_ratios=[1.2, 1.2, 1.2, 1.2], hspace=0.8)

# Plot 1: Performance comparison line plot
ax1 = fig.add_subplot(gs[0])
ax1.plot(
    quantization_levels,
    avg_mse_a,
    "o-",
    linewidth=2.5,
    markersize=8,
    label="Model A (Clean-trained)",
    color="#e74c3c",
)
ax1.plot(
    quantization_levels,
    avg_mse_c,
    "s-",
    linewidth=2.5,
    markersize=8,
    label="Model C (Noise-trained)",
    color="#27ae60",
)
ax1.set_xlabel("Quantization Bit Depth", fontsize=12, fontweight="bold")
ax1.set_ylabel("Average MSE", fontsize=12, fontweight="bold")
ax1.set_title(
    "Reconstruction Performance vs Quantization Level", fontsize=14, fontweight="bold"
)
ax1.legend(fontsize=11, loc="upper left")
ax1.grid(True, alpha=0.3)
ax1.invert_xaxis()  # Higher bits on left
ax1.set_xticks(quantization_levels)

# Plot 2: Improvement bar chart
ax2 = fig.add_subplot(gs[1])
colors = ["#c0392b" if x < 0 else "#27ae60" for x in improvements]
bars = ax2.bar(
    range(len(quantization_levels)),
    improvements,
    color=colors,
    alpha=0.7,
    edgecolor="black",
)
ax2.axhline(y=0, color="black", linestyle="-", linewidth=1)
ax2.set_xlabel("Quantization Bit Depth", fontsize=12, fontweight="bold")
ax2.set_ylabel("Improvement (%)", fontsize=12, fontweight="bold")
ax2.set_title(
    "Performance Improvement of Noise-trained over Clean-trained Model",
    fontsize=14,
    fontweight="bold",
)
ax2.set_xticks(range(len(quantization_levels)))
ax2.set_xticklabels([f"{bits}-bit" for bits in quantization_levels])
ax2.grid(axis="y", alpha=0.3)
ax2.margins(y=0.2)  # Add margin to prevent label overlap

# Add percentage labels on bars
for i, (bar, imp) in enumerate(zip(bars, improvements)):
    height = bar.get_height()
    y_offset = 0.5 if height > 0 else -0.5
    ax2.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + y_offset,
        f"{imp:+.1f}%",
        ha="center",
        va="bottom" if height > 0 else "top",
        fontsize=9,
        fontweight="bold",
    )

# Plot 3: Example reconstructions for different quantization levels
ax3 = fig.add_subplot(gs[2])
ax3.axis("off")
ax3.set_title(
    f"Model A (Clean-trained) Reconstructions", fontsize=13, fontweight="bold", pad=30
)

# Create subplots for Model A examples
gs3 = gs[2].subgridspec(1, len(example_bits) + 1, wspace=0.3)
ax_orig = fig.add_subplot(gs3[0])
ax_orig.imshow(example_image[0].numpy().squeeze(), cmap="gray")
ax_orig.set_title("Original", fontsize=11, fontweight="bold")
ax_orig.axis("off")

for i, (bits, recon, mse) in enumerate(
    zip(example_bits, example_recons_a, example_mses_a)
):
    ax = fig.add_subplot(gs3[i + 1])
    ax.imshow(recon.squeeze(), cmap="gray")
    ax.set_title(f"{bits}-bit\nMSE: {mse:.4f}", fontsize=11, fontweight="bold")
    ax.axis("off")

# Plot 4: Example reconstructions for Model C
ax4 = fig.add_subplot(gs[3])
ax4.axis("off")
ax4.set_title(
    f"Model C (Noise-trained) Reconstructions", fontsize=13, fontweight="bold", pad=30
)

gs4 = gs[3].subgridspec(1, len(example_bits) + 1, wspace=0.3)
ax_orig2 = fig.add_subplot(gs4[0])
ax_orig2.imshow(example_image[0].numpy().squeeze(), cmap="gray")
ax_orig2.set_title("Original", fontsize=11, fontweight="bold")
ax_orig2.axis("off")

for i, (bits, recon, mse) in enumerate(
    zip(example_bits, example_recons_c, example_mses_c)
):
    ax = fig.add_subplot(gs4[i + 1])
    ax.imshow(recon.squeeze(), cmap="gray")
    ax.set_title(f"{bits}-bit\nMSE: {mse:.4f}", fontsize=11, fontweight="bold")
    ax.axis("off")

plt.suptitle(
    f"Autoencoder Comparison: Clean-trained vs Noise-trained\n"
    f"Tested on {len(test_images)} images across {len(quantization_levels)} quantization levels",
    fontsize=15,
    fontweight="bold",
    y=0.995,
)

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig("quantization_comparison.png", dpi=150, bbox_inches="tight")
print("\n✓ Saved visualization to: quantization_comparison.png")
plt.show()

print("\n✓ Comparison complete!")
