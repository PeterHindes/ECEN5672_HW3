import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

print("Loading saved denoising model...")
model = keras.models.load_model("mnist_autoencoder_denoiser_model.keras")
print("✓ Model loaded successfully!\n")

# Find the bottleneck layer
print("Model layers:")
bottleneck_layer = None
bottleneck_name = None
for i, layer in enumerate(model.layers):
    print(f"  {i}: {layer.name} ({layer.__class__.__name__})")
    if "bottleneck" in layer.name:
        bottleneck_layer = layer
        bottleneck_name = layer.name
        print(f"  ↑ BOTTLENECK FOUND!")

if bottleneck_layer is None:
    print(
        "\nWarning: No layer named 'bottleneck_*' found. Using layer index for bottleneck."
    )
    # Fallback: try to guess the bottleneck
    for i, layer in enumerate(model.layers):
        if isinstance(layer, keras.layers.Dense) and i < len(model.layers) // 2:
            bottleneck_layer = layer
            bottleneck_name = layer.name
            print(f"  Using {layer.name} as bottleneck")
            break

# Create bottleneck extractor
bottleneck_model = keras.Model(
    inputs=model.input, outputs=model.get_layer(bottleneck_name).output
)

print("\nLoading test data...")
ds_test = tfds.load(
    "mnist", split="test[:2000]", shuffle_files=True, as_supervised=False
)


def normalize_img(data):
    image = tf.cast(data["image"], tf.float32) / 255.0
    return image


ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE).batch(128)
print("✓ Test data loaded\n")


# Quantization functions
def quantize_int8(values, scale=None, zero_point=None):
    """Quantize float32 values to int8"""
    if scale is None:
        # Symmetric quantization
        max_val = np.max(np.abs(values))
        scale = max_val / 127.0 if max_val > 0 else 1.0
        zero_point = 0

    # Quantize
    quantized = np.round(values / scale).astype(np.int8)
    return quantized, scale, zero_point


def dequantize_int8(quantized, scale, zero_point):
    """Dequantize int8 back to float32"""
    return quantized.astype(np.float32) * scale + zero_point


print("Press Ctrl+C to exit\n")

try:
    while True:
        # Get a random image
        print("Selecting random image...")
        for images in ds_test.take(1):
            random_idx = np.random.randint(0, images.shape[0])
            clean_image = images[random_idx : random_idx + 1]
            break

        # Add noise
        noise = tf.random.normal(shape=tf.shape(clean_image), mean=0.0, stddev=0.3)
        noisy_image = clean_image + noise

        # Extract bottleneck representation (float32)
        bottleneck_float = bottleneck_model.predict(noisy_image, verbose=0)[0]

        # Quantize to int8
        bottleneck_int8, scale, zero_point = quantize_int8(bottleneck_float)

        # Dequantize (for comparison)
        bottleneck_dequant = dequantize_int8(bottleneck_int8, scale, zero_point)

        # Get full reconstruction (original float32)
        denoised_float = model.predict(noisy_image, verbose=0)

        # Calculate metrics
        noise_mse = np.mean((clean_image.numpy() - noisy_image.numpy()) ** 2)
        denoised_mse = np.mean((clean_image.numpy() - denoised_float) ** 2)
        quantization_error = np.mean((bottleneck_float - bottleneck_dequant) ** 2)

        # Compression statistics
        float32_bits = len(bottleneck_float) * 32
        int8_bits = len(bottleneck_int8) * 8
        compression_ratio = float32_bits / int8_bits

        print(f"\nBottleneck Statistics:")
        print(f"  Shape: {bottleneck_float.shape}")
        print(f"  Float32 size: {float32_bits} bits ({float32_bits / 8} bytes)")
        print(f"  Int8 size: {int8_bits} bits ({int8_bits / 8} bytes)")
        print(f"  Compression: {compression_ratio:.1f}x")
        print(f"  Quantization scale: {scale:.6f}")
        print(f"  Quantization error (MSE): {quantization_error:.6f}")
        print(
            f"  Sparsity: {np.sum(np.abs(bottleneck_float) < 0.01) / len(bottleneck_float) * 100:.1f}%"
        )

        print(f"\nReconstruction Metrics:")
        print(f"  Noise MSE: {noise_mse:.6f}")
        print(f"  Denoised MSE: {denoised_mse:.6f}")
        print(f"  Improvement: {((noise_mse - denoised_mse) / noise_mse * 100):.2f}%\n")

        # Create comprehensive visualization
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Row 1: Images
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(clean_image[0].numpy().squeeze(), cmap="gray")
        ax1.set_title("Clean Original", fontsize=12, fontweight="bold")
        ax1.axis("off")

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(noisy_image[0].numpy().squeeze(), cmap="gray")
        ax2.set_title(
            f"Noisy Input\n(MSE: {noise_mse:.4f})", fontsize=12, fontweight="bold"
        )
        ax2.axis("off")

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(denoised_float[0].squeeze(), cmap="gray")
        ax3.set_title(
            f"Denoised Output\n(MSE: {denoised_mse:.4f})",
            fontsize=12,
            fontweight="bold",
        )
        ax3.axis("off")

        # Row 2: Bottleneck representations
        ax4 = fig.add_subplot(gs[1, :])
        x = np.arange(len(bottleneck_float))
        width = 0.35
        ax4.bar(
            x - width / 2,
            bottleneck_float,
            width,
            label="Float32 (original)",
            alpha=0.8,
            color="blue",
        )
        ax4.bar(
            x + width / 2,
            bottleneck_dequant,
            width,
            label="Int8 (quantized→dequantized)",
            alpha=0.8,
            color="orange",
        )
        ax4.set_xlabel("Bottleneck Dimension", fontsize=11)
        ax4.set_ylabel("Activation Value", fontsize=11)
        ax4.set_title(
            f'Bottleneck Layer: "{bottleneck_name}" - Float32 vs Int8 Quantized\n'
            f"Quantization Error MSE: {quantization_error:.6f} | Compression: {compression_ratio:.1f}x",
            fontsize=12,
            fontweight="bold",
        )
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color="black", linewidth=0.5)

        # Row 3: Quantized values and stats
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.bar(range(len(bottleneck_int8)), bottleneck_int8, color="green", alpha=0.7)
        ax5.set_xlabel("Dimension", fontsize=10)
        ax5.set_ylabel("Int8 Value", fontsize=10)
        ax5.set_title(
            "Int8 Quantized Values\n(Range: -128 to 127)",
            fontsize=11,
            fontweight="bold",
        )
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim(-128, 128)

        ax6 = fig.add_subplot(gs[2, 1])
        error_per_dim = np.abs(bottleneck_float - bottleneck_dequant)
        ax6.bar(range(len(error_per_dim)), error_per_dim, color="red", alpha=0.7)
        ax6.set_xlabel("Dimension", fontsize=10)
        ax6.set_ylabel("Absolute Error", fontsize=10)
        ax6.set_title(
            "Per-Dimension Quantization Error", fontsize=11, fontweight="bold"
        )
        ax6.grid(True, alpha=0.3)

        ax7 = fig.add_subplot(gs[2, 2])
        stats_text = f"""QUANTIZATION STATISTICS

Float32 Storage: {float32_bits / 8:.0f} bytes
Int8 Storage: {int8_bits / 8:.0f} bytes
Compression Ratio: {compression_ratio:.1f}x

Scale Factor: {scale:.6f}
Zero Point: {zero_point}

Quantization Error:
  Mean Abs: {np.mean(error_per_dim):.6f}
  Max Abs: {np.max(error_per_dim):.6f}
  MSE: {quantization_error:.6f}

Sparsity: {np.sum(np.abs(bottleneck_float) < 0.01) / len(bottleneck_float) * 100:.1f}%
Active Dims: {np.sum(np.abs(bottleneck_float) >= 0.01)}
"""
        ax7.text(
            0.1,
            0.5,
            stats_text,
            fontsize=9,
            verticalalignment="center",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        ax7.axis("off")

        plt.suptitle(
            f"Denoising Autoencoder with Int8 Quantized Bottleneck\n"
            f'Bottleneck: {bottleneck_float.shape} | Layer: "{bottleneck_name}"',
            fontsize=14,
            fontweight="bold",
        )

        print("Opening visualization window...")
        plt.show()

except KeyboardInterrupt:
    print("\n\nExiting...")
